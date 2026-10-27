"""P2: clasificadores candidatos del proxy de presión de flujo de la siguiente hora."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin interfaz: seguro en scripts, servidores y CI

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from proyecto_mibici.config import CANDIDATE_MODELS, SUPERVISED_HORIZON_HOURS, ProjectConfig
from proyecto_mibici.modeling.base import CandidateResult
from proyecto_mibici.modeling.datasets import P2_FEATURES, DatosSupervisados
from proyecto_mibici.modeling.supervised_baselines import PRESSURE_CLASSES, classification_metrics


def build_estimator(model: str, params: dict, seed: int):
    """class_weight='balanced' compensa clases altas raras, que dominan el F1 macro."""
    params = dict(params)
    if params.get("class_weight") == "none":
        params["class_weight"] = None
    if model == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **params))
    if model == "hgb":
        # Sin early stopping: su partición interna aleatoria mezclaría horas de train.
        return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)
    if model == "rf":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
    raise ValueError(f"Modelo P2 desconocido: {model}.")


def save_confusion_matrix_plot(
    matrix: np.ndarray, labels: list[str], model: str, split: str, output_dir: Path
) -> Path:
    """Matriz de confusión (conteos y normalizada por fila); devuelve la ruta del PNG."""
    path = output_dir / f"matriz_confusion_{split}.png"

    # Normalización por fila (recall por clase); filas sin observaciones quedan en 0.
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros(matrix.shape, dtype=float), where=row_sums > 0)

    panels = (
        (matrix, "Conteos", lambda v: f"{int(v)}"),
        (normalized, "Normalizada por fila (recall)", lambda v: f"{v:.0%}"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    try:
        for ax, (values, title, fmt) in zip(axes, panels):
            image = ax.imshow(values, cmap="Blues")
            ax.set_title(title)
            ax.set_xlabel("Clase predicha")
            ax.set_ylabel("Clase real")
            ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)), labels=labels)
            threshold = values.max() / 2 if values.max() > 0 else 0
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    ax.text(
                        j, i, fmt(values[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > threshold else "black",
                    )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"P2 {model.upper()} - Matriz de confusión ({split})")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)  # siempre libera la figura, incluso si algo falla

    if not path.exists():
        raise RuntimeError(f"No se pudo guardar la gráfica en {path}.")
    return path


class CandidatosP2:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def run(
        self,
        data: DatosSupervisados,
        model: str,
        params: dict,
        output_dir: Path,
        *,
        max_train_rows: int,
        evaluate_test: bool,
    ) -> CandidateResult:
        """Usa los percentiles congelados del baseline y compara en las mismas filas."""
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        train = data.split("p2", "train", max_rows=max_train_rows, seed=self.config.seed)
        estimator = build_estimator(model, params, self.config.seed).fit(train.X, train.y)
        result = CandidateResult(
            "P2",
            model,
            parameters={
                "model": model,
                **params,
                "horizon_hours": SUPERVISED_HORIZON_HOURS,
                "target": "clase_presion_siguiente_hora",
                "target_kind": "retrospective_net_flow_proxy_not_inventory",
                "pressure_q75": data.metadata["pressure_q75"],
                "pressure_q90": data.metadata["pressure_q90"],
                "quantile_fit_split": "train",
                "labels": ",".join(PRESSURE_CLASSES),
                "macro_f1_labels": "all_seven_fixed_classes",
                "max_train_rows": max_train_rows,
                "train_rows_used": len(train),
                "n_features": len(P2_FEATURES),
                "features": ",".join(P2_FEATURES),
                "evaluation_rows": "eligible_hours_with_capacity_and_complete_causal_features",
                "baseline_on_same_rows": PRESSURE_CLASSES[data.metadata["p2_majority_class"]],
            },
            model_family=CANDIDATE_MODELS["p2"][model]["family"],
            estimator=estimator,
            feature_names=P2_FEATURES,
            input_example=train.X[P2_FEATURES].head(5).copy(),
        )

        plot_paths: list[Path] = []
        for split in ("train", "validation", "test")[: 3 if evaluate_test else 2]:
            subset = train if split == "train" else data.split("p2", split)
            y_pred = estimator.predict(subset.X)
            matrix = confusion_matrix(subset.y, y_pred, labels=range(7))
            baseline = classification_metrics(
                confusion_matrix(subset.y, subset.baseline, labels=range(7))
            )
            metrics = classification_metrics(matrix)
            result.metrics.update({f"{split}_{key}": value for key, value in metrics.items()})
            result.metrics.update(
                {
                    f"{split}_baseline_f1_macro": baseline["f1_macro"],
                    f"{split}_f1_macro_delta_vs_baseline": metrics["f1_macro"]
                    - baseline["f1_macro"],
                }
            )
            # Una matriz por split, con los datos de ESE split.
            plot_paths.append(
                save_confusion_matrix_plot(matrix, list(PRESSURE_CLASSES), model, split, output_dir)
            )

        # Registro en MLflow: solo funciona si hay un run activo en este momento.
        if mlflow.active_run() is not None:
            for path in plot_paths:
                mlflow.log_artifact(str(path), artifact_path="plots")

        # Si el runner registra artefactos desde CandidateResult, aquí se le pasan las rutas.
        if hasattr(result, "artifacts"):
            result.artifacts.extend(str(p) for p in plot_paths)

        return result
