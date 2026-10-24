"""P2: clasificadores candidatos del proxy de presión de flujo de la siguiente hora."""

from pathlib import Path

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
        )
        for split in ("train", "validation", "test")[: 3 if evaluate_test else 2]:
            subset = train if split == "train" else data.split("p2", split)
            matrix = confusion_matrix(subset.y, estimator.predict(subset.X), labels=range(7))
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
        return result
