from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# ==========================================
# Paths
# ==========================================
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MANIFEST_PATH = DATA_DIR / "data_manifest.json"

# ==========================================
# Configuración global del proyecto
# ==========================================
DATA_URL_ONLINE = "https://www.mibici.net/es/datos-abiertos/"

DEFAULT_RANDOM_STATE = 42

# Ventana de tiempo predeterminada para el dataset
DEFAULT_ANIO_INICIO = 2023
DEFAULT_MES_INICIO = 10
DEFAULT_ANIO_FIN = 2026
DEFAULT_MES_FIN = 9

# Contrato común de ingesta para P1, P2 y P3.
TRIPS_PARQUET_PATH = INTERIM_DATA_DIR / "viajes_unificados.parquet"
STATIONS_PARQUET_PATH = INTERIM_DATA_DIR / "estaciones.parquet"
STATIONS_SNAPSHOT_MONTH = "2026-09"
INGESTION_CHUNK_SIZE = 100_000
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_ATTEMPTS = 3


# Experimento base común.
EXPERIMENT_NAME = "mibici-equipo25"
TRAIN_END = "2025-07-01"
VALIDATION_END = "2025-10-01"
HOURS_PARQUET_PATH = PROCESSED_DATA_DIR / "horas_estacion.parquet"
LOCAL_TRACKING_URI = "http://127.0.0.1:5000"
ARTIFACTS_DIR = PROJ_ROOT / "mlartifacts" / EXPERIMENT_NAME
BASELINE_FIGURES_DIR = FIGURES_DIR / "baselines"
SUPERVISED_HORIZON_HOURS = 1
FEATURE_LAGS = (1, 3, 24, 168)
ROLLING_WINDOWS = (3, 24, 168)
BALANCE_WINDOWS = (3, 24)
WEEKLY_LAGS = (336, 504, 672)

# Candidatos de la Entrega 2 y sus hiperparámetros por defecto.
SUPERVISED_TABLE_PATH = PROCESSED_DATA_DIR / "tabla_supervisada.parquet"
CANDIDATE_FIGURES_DIR = FIGURES_DIR / "candidates"
# Filas de entrenamiento muestreadas con la semilla; 0 = todas.
CANDIDATE_MAX_TRAIN_ROWS = 2_000_000
SEARCH_MAX_TRAIN_ROWS = 500_000
CANDIDATE_MODELS = {
    "p1": {
        "poisson": {"family": "glm_poisson", "defaults": {"alpha": 1e-4, "max_iter": 300}},
        # Sin l2, la pérdida Poisson puede divergir con muchas iteraciones.
        "hgb": {
            "family": "gradient_boosting",
            "defaults": {
                "learning_rate": 0.1,
                "max_iter": 200,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 50,
                "l2_regularization": 1.0,
            },
        },
        "rf": {
            "family": "random_forest",
            "defaults": {
                "n_estimators": 100,
                "max_depth": 16,
                "min_samples_leaf": 50,
                "max_features": 0.5,
                "max_samples": 0.25,
            },
        },
    },
    "p2": {
        "logistic": {
            "family": "glm_logistic",
            "defaults": {"C": 1.0, "class_weight": "balanced", "max_iter": 300},
        },
        "hgb": {
            "family": "gradient_boosting",
            "defaults": {
                "learning_rate": 0.1,
                "max_iter": 200,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 50,
                "l2_regularization": 0.0,
                "class_weight": "balanced",
            },
        },
        "rf": {
            "family": "random_forest",
            "defaults": {
                "n_estimators": 100,
                "max_depth": 16,
                "min_samples_leaf": 50,
                "max_features": 0.5,
                "max_samples": 0.25,
                "class_weight": "balanced",
            },
        },
    },
    "p3": {
        "kmeans": {"family": "kmeans", "defaults": {"n_clusters": 3, "n_init": 20}},
        "agglomerative": {
            "family": "hierarchical",
            "defaults": {"n_clusters": 3, "linkage": "ward"},
        },
        # PCA previo: en 336 dimensiones la mezcla no es estimable con ~400 estaciones.
        "gmm": {
            "family": "gaussian_mixture",
            "defaults": {"n_clusters": 2, "covariance_type": "diag", "pca_components": 5},
        },
    },
}
# Rejillas de búsqueda; las claves no listadas toman el default del modelo.
SEARCH_SPACES = {
    "p1": {
        "hgb": {
            "learning_rate": [0.05, 0.1],
            "max_iter": [200, 500],
            "min_samples_leaf": [50, 200],
        }
    },
    "p2": {
        "hgb": {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [31, 63],
            "min_samples_leaf": [50, 200],
        }
    },
    "p3": {
        "kmeans": {"n_clusters": [2, 3, 4, 5, 6, 7, 8]},
        "gmm": {"n_clusters": [2, 3, 4], "pca_components": [0, 5, 10, 20]},
    },
}
# Las búsquedas eligen solo con validación.
SEARCH_CRITERIA = {
    "p1": ("validation_rmse", "min"),
    "p2": ("validation_f1_macro", "max"),
    "p3": ("validation_silhouette", "max"),
}


@dataclass(frozen=True)
class ProjectConfig:
    """Configuración inmutable compartida por ingesta, features y los tres baselines."""

    root: Path = PROJ_ROOT
    start_month: str = f"{DEFAULT_ANIO_INICIO}-{DEFAULT_MES_INICIO:02d}"
    end_month: str = f"{DEFAULT_ANIO_FIN}-{DEFAULT_MES_FIN:02d}"
    train_end: str = TRAIN_END
    validation_end: str = VALIDATION_END
    seed: int = DEFAULT_RANDOM_STATE
    recent_hours: int = FEATURE_LAGS[-1]
    weekly_lag: int = FEATURE_LAGS[-1]
    pressure_quantiles: tuple[float, float] = (0.75, 0.90)

    @property
    def start(self):
        import pandas as pd

        return pd.Period(self.start_month, freq="M").start_time

    @property
    def end(self):
        import pandas as pd

        return (pd.Period(self.end_month, freq="M") + 1).start_time

    @property
    def splits(self):
        import pandas as pd

        return {
            "train": (self.start, pd.Timestamp(self.train_end)),
            "validation": (pd.Timestamp(self.train_end), pd.Timestamp(self.validation_end)),
            "test": (pd.Timestamp(self.validation_end), self.end),
        }

    def path(self, canonical: Path) -> Path:
        return Path(self.root).resolve() / canonical.relative_to(PROJ_ROOT)

    @property
    def trips_path(self) -> Path:
        return self.path(TRIPS_PARQUET_PATH)

    @property
    def stations_path(self) -> Path:
        return self.path(STATIONS_PARQUET_PATH)

    @property
    def manifest_path(self) -> Path:
        return self.path(MANIFEST_PATH)

    @property
    def hours_path(self) -> Path:
        return self.path(HOURS_PARQUET_PATH)

    @property
    def supervised_table_path(self) -> Path:
        return self.path(SUPERVISED_TABLE_PATH)

    @property
    def tracking_uri(self) -> str:
        """URI del servidor MLflow; admite sustitución por entorno para pruebas aisladas."""
        return os.getenv("MLFLOW_TRACKING_URI", LOCAL_TRACKING_URI)

    def validate(self) -> None:
        import pandas as pd

        if (
            not self.start
            < pd.Timestamp(self.train_end)
            < pd.Timestamp(self.validation_end)
            < self.end
        ):
            raise ValueError(
                "Se requiere inicio < fin de entrenamiento < fin de validación < fin."
            )
        for boundary in (pd.Timestamp(self.train_end), pd.Timestamp(self.validation_end)):
            if boundary != boundary.floor("h"):
                raise ValueError("Los cortes deben coincidir con horas completas.")
        if self.recent_hours < 1 or self.weekly_lag < 1:
            raise ValueError("Las ventanas históricas deben ser positivas.")
        if self.seed < 0:
            raise ValueError("La semilla debe ser no negativa.")
        if not 0 < self.pressure_quantiles[0] < self.pressure_quantiles[1] < 1:
            raise ValueError("Los percentiles de presión deben estar ordenados entre cero y uno.")

    def parameters(self) -> dict:
        return {
            "start_month": self.start_month,
            "end_month": self.end_month,
            "train_start": self.start.isoformat(),
            "train_end_exclusive": self.train_end,
            "validation_end_exclusive": self.validation_end,
            "test_end_exclusive": self.end.isoformat(),
            "seed": self.seed,
            "supervised_horizon_hours": SUPERVISED_HORIZON_HOURS,
            "recent_activity_hours": self.recent_hours,
            "weekly_lag_hours": self.weekly_lag,
            "capacity_snapshot": STATIONS_SNAPSHOT_MONTH,
            "coverage_policy": "complete_source_months_and_activity_in_previous_window",
            "zero_policy": "analytical_zero_only_after_observed_past_activity_not_inventory",
        }


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
