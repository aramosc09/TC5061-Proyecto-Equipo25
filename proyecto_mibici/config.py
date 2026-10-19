"""Única fuente de configuración: rutas, fechas, semillas y parámetros del proyecto."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_URL_ONLINE = "https://www.mibici.net/es/datos-abiertos/"

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MANIFEST_PATH = DATA_DIR / "data_manifest.json"

# Parámetros del proyecto. Viven solo aquí: no se repiten en otros archivos.
DEFAULTS = {
    "project": {"team": "25", "experiment": "mibici-equipo25", "seed": 42},
    "data": {
        "start_month": "2023-10",
        "end_month": "2026-09",
        "max_duration_hours": 24,
        "trips_parquet": "data/interim/viajes_unificados.parquet",
        "stations_csv": "data/raw/nomenclatura_2026_09.csv",
    },
    "features": {
        "lags": [1, 24, 168],
        "rolling_hours": 24,
        "p1_horizon_hours": 24,
        "p2_window_hours": 2,
        "pressure_quantile": 0.90,
        "pressure_minimum": 0.01,
        # Capacidad retrospectiva: denominador de P2, nunca una variable predictora.
        "capacity_snapshot": "2026-09",
        "supervised": "data/processed/features_estacion_hora.parquet",
        "profiles": "data/processed/perfiles_estacion.parquet",
    },
    "split": {"train_end": "2026-04-01", "validation_end": "2026-07-01"},
    "models": {"clusters": 3, "include_candidates": False},
    "tracking": {
        "database": "mlruns/tracking.db",
        "artifacts": "mlartifacts",
        "figures": "reports/figures/generated",
    },
}

RANDOM_STATE = DEFAULTS["project"]["seed"]
DEFAULT_ANIO_INICIO, DEFAULT_MES_INICIO = map(int, DEFAULTS["data"]["start_month"].split("-"))
DEFAULT_ANIO_FIN, DEFAULT_MES_FIN = map(int, DEFAULTS["data"]["end_month"].split("-"))


@dataclass(frozen=True)
class Settings:
    """Configuración validada; las rutas relativas se resuelven contra una sola raíz."""

    values: dict
    root: Path = PROJ_ROOT

    def path(self, section: str, key: str) -> Path:
        value = Path(self.values[section][key])
        return value if value.is_absolute() else self.root / value

    @property
    def seed(self) -> int:
        return int(self.values["project"]["seed"])


def load_settings(*, root=None, overrides=None) -> Settings:
    """Aplica overrides `seccion.clave` y valida que la partición temporal sea correcta."""
    import pandas as pd

    values = deepcopy(DEFAULTS)
    for key, value in (overrides or {}).items():
        section, field = key.split(".", 1)
        if section not in values or field not in values[section]:
            raise ValueError(f"Parámetro desconocido: {key}")
        values[section][field] = value

    start = pd.Period(values["data"]["start_month"], freq="M").start_time
    end = (pd.Period(values["data"]["end_month"], freq="M") + 1).start_time
    train_end = pd.Timestamp(values["split"]["train_end"])
    validation_end = pd.Timestamp(values["split"]["validation_end"])
    if not start < train_end < validation_end < end:
        raise ValueError("Se requiere inicio < train_end < validation_end < fin de la ventana")
    for name, boundary in (("train_end", train_end), ("validation_end", validation_end)):
        if boundary != boundary.floor("h"):
            raise ValueError("Los cortes de partición deben caer en horas completas")
        values["split"][name] = boundary.isoformat()
    lags = values["features"]["lags"]
    if not lags or not all(isinstance(x, int) and x >= 1 for x in lags):
        raise ValueError("Todos los lags deben ser horas enteras positivas")
    for key in ("p1_horizon_hours", "p2_window_hours", "rolling_hours"):
        if not isinstance(values["features"][key], int) or values["features"][key] < 1:
            raise ValueError(f"{key} debe ser un entero positivo")
    if not 0 < values["features"]["pressure_quantile"] < 1:
        raise ValueError("pressure_quantile debe estar en (0, 1)")
    if values["features"]["pressure_minimum"] <= 0:
        raise ValueError("pressure_minimum debe ser positivo")
    if values["models"]["clusters"] < 2:
        raise ValueError("Se requieren al menos dos grupos")
    return Settings(values, Path(root).resolve() if root else PROJ_ROOT)
