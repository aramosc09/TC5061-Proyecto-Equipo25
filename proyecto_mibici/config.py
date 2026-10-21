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
DATA_URL_ONLINE = 'https://www.mibici.net/es/datos-abiertos/'

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


# ==========================================
# Configuración P2
# ==========================================
DEFAULT_DESFASE_HORAS = 2
DEFAULT_PERCENTIL_UMBRAL = 0.90
DEFAULT_FECHA_CORTE_SPLIT = '2026-04-01'

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
