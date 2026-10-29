"""P1: regresores candidatos de salidas de la siguiente hora por estación."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin interfaz: seguro en scripts, servidores y CI

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from proyecto_mibici.config import CANDIDATE_MODELS, SUPERVISED_HORIZON_HOURS, ProjectConfig
from proyecto_mibici.modeling.base import CandidateResult
from proyecto_mibici.modeling.datasets import HOURLY_FEATURES, P1_FEATURES, DatosSupervisados

# El enlace log de Poisson extrapola mal los rezagos semanales.
FEATURES_BY_MODEL = {"poisson": HOURLY_FEATURES, "hgb": P1_FEATURES, "rf": P1_FEATURES}


def build_estimator(model: str, params: dict, seed: int):
    """La pérdida Poisson respeta que la demanda son conteos no negativos."""
    if model == "poisson":
        # log1p evita predicciones explosivas del enlace log con conteos crudos.
        counts = [name for name in HOURLY_FEATURES if name.startswith(("salidas_", "llegadas_"))]
        return make_pipeline(
            ColumnTransformer(
                [("conteos", FunctionTransformer(np.log1p), counts)], remainder="passthrough"
            ),
            StandardScaler(),
            PoissonRegressor(**params),
        )
    if model == "hgb":
        # Sin early stopping: su partición interna aleatoria mezclaría horas de train.
        return HistGradientBoostingRegressor(
            loss="poisson", early_stopping=False, random_state=seed, **params
        )
    if model == "rf":
        return RandomForestRegressor(n_jobs=-1, random_state=seed, **params)
    raise ValueError(f"Modelo P1 desconocido: {model}.")


def regression_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    """Mismas métricas que el baseline; el error firmado es predicho menos observado."""
    if not np.isfinite(forecast).all():
        raise ValueError("El modelo produjo predicciones no finitas.")
    error = forecast - actual
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_error": float(error.mean()),
        "n_observations": len(actual),
    }


def save_scatter_plot(
    actual: np.ndarray, predicted: np.ndarray, model: str, split: str, output_dir: Path
) -> Path:
    """Dispersión real vs predicho; devuelve la ruta del PNG ya escrito en disco."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    path = output_dir / f"dispersion_real_vs_predicho_{split}.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.scatter(actual, predicted, alpha=0.3, color="#1f77b4", edgecolors="none")
        lim_min = float(min(actual.min(), predicted.min()))
        lim_max = float(max(actual.max(), predicted.max()))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", alpha=0.7, label="Predicción perfecta")
        ax.set_title(f"P1 {model.upper()} - Real vs Predicho ({split})")
        ax.set_xlabel("Salidas Reales")
        ax.set_ylabel("Salidas Predichas")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)  # siempre libera la figura, incluso si algo falla

    if not path.exists():
        raise RuntimeError(f"No se pudo guardar la gráfica en {path}.")
    return path


class CandidatosP1:
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
        """Ajusta solo con entrenamiento y compara contra el baseline en las mismas filas."""
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        train = data.split("p1", "train", max_rows=max_train_rows, seed=self.config.seed)
        features = FEATURES_BY_MODEL[model]
        estimator = build_estimator(model, params, self.config.seed).fit(train.X[features], train.y)

        result = CandidateResult(
            "P1",
            model,
            parameters={
                "model": model,
                **params,
                "horizon_hours": SUPERVISED_HORIZON_HOURS,
                "target": "salidas_siguiente_hora",
                "max_train_rows": max_train_rows,
                "train_rows_used": len(train),
                "n_features": len(features),
                "features": ",".join(features),
                "weekly_history_features": features is P1_FEATURES,
                "evaluation_rows": "eligible_hours_with_complete_causal_features",
                "baseline_on_same_rows": "salidas_lag_168",
            },
            model_family=CANDIDATE_MODELS["p1"][model]["family"],
            estimator=estimator,
            feature_names=features,
            input_example=train.X[features].head(5).copy(),
        )

        plot_paths: list[Path] = []
        for split in ("train", "validation", "test")[: 3 if evaluate_test else 2]:
            subset = train if split == "train" else data.split("p1", split)
            y_pred = estimator.predict(subset.X[features])
            metrics = regression_metrics(subset.y, y_pred)
            baseline = regression_metrics(subset.y, subset.baseline)
            result.metrics.update({f"{split}_{key}": value for key, value in metrics.items()})
            result.metrics.update(
                {
                    f"{split}_baseline_mae": baseline["mae"],
                    f"{split}_baseline_rmse": baseline["rmse"],
                    f"{split}_rmse_delta_vs_baseline": metrics["rmse"] - baseline["rmse"],
                }
            )
            # Una gráfica por split, generada con los datos de ESE split.
            plot_paths.append(save_scatter_plot(subset.y, y_pred, model, split, output_dir))


        # Si el runner registra artefactos desde CandidateResult, aquí se le pasan las rutas.
        # (Ajusta el nombre del atributo a lo que use tu base.py.)
        result.artifacts.extend(plot_paths)

        return result
