import codecs
from contextlib import closing
from datetime import datetime
import hashlib
from html.parser import HTMLParser
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
import unicodedata
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from proyecto_mibici import config

app = typer.Typer(
    help="Ingesta reproducible y versionada de los datos de MiBici.",
    pretty_exceptions_show_locals=False,
)

COLUMNAS_PERSONALES = ["usuario_id", "genero", "anio_nacimiento"]


# ---------------------------------------------------------------------------
# Utilidades de hash de archivos
# ---------------------------------------------------------------------------
def calcular_sha256(ruta: Path, tamano_bloque: int = 1 << 20) -> str:
    """Calcula el hash SHA-256 de un archivo leyéndolo por bloques de 1 MB.

    Leer por bloques evita cargar archivos grandes completos en memoria.
    El mismo contenido siempre produce el mismo hash; cualquier cambio de un
    solo byte produce uno distinto. Es confiable para archivos que solo se
    copian (CSV crudos). Para un Parquet regenerado es solo informativo,
    porque sus bytes pueden variar con la versión de pyarrow.
    """
    h = hashlib.sha256()
    with Path(ruta).open("rb") as f:
        while bloque := f.read(tamano_bloque):
            h.update(bloque)
    return h.hexdigest()


def _ruta_relativa(ruta: Path) -> str:
    """Devuelve la ruta relativa a la raíz del proyecto (portable entre equipos)."""
    try:
        return Path(ruta).resolve().relative_to(config.PROJ_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(ruta).resolve().as_posix()


# ---------------------------------------------------------------------------
# Cargador
# ---------------------------------------------------------------------------
class MibiciDataLoader:
    """Clase para cargar y procesar los archivos históricos mensuales de MiBici,

    resolviendo problemas de codificación y ambigüedad en las fechas.
    Además de los datos, registra metadatos de cada archivo (hash, filas,
    codificación, estrategia de fechas) en `self.registros`, que luego
    alimentan el manifiesto de datos.
    """

    COLUMNAS_ESTANDAR = [
        "viaje_id",
        "usuario_id",
        "genero",
        "anio_nacimiento",
        "inicio_viaje",
        "fin_viaje",
        "origen_id",
        "destino_id",
    ]

    def __init__(self, ruta_datos: str | Path = config.RAW_DATA_DIR):
        self.ruta_datos = Path(ruta_datos)
        if not self.ruta_datos.exists():
            raise FileNotFoundError(
                f"No se encontró la ruta de datos: {self.ruta_datos}."
            )
        self.registros: list[dict] = []
        self.rango: dict = {}

    def _leer_csv(self, archivo: Path, nrows: int = None) -> tuple[pd.DataFrame, str]:
        codificaciones = ["utf-8", "cp1252"]
        for encoding in codificaciones:
            try:
                df = pd.read_csv(archivo, encoding=encoding, nrows=nrows)
                return df, encoding
            except UnicodeDecodeError:
                continue
        raise ValueError(f"No fue posible leer el archivo {archivo.name}")

    def _normalizar_columnas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifica columnas por nombre, sin depender de su orden en el CSV."""
        equivalencias = {
            "ano_de_nacimiento": "anio_nacimiento",
            "anio_de_nacimiento": "anio_nacimiento",
            # Encabezado dañado observado en las fuentes de febrero/mayo de 2024.
            "a_ae_nacimiento": "anio_nacimiento",
            "inicio_del_viaje": "inicio_viaje",
            "fin_del_viaje": "fin_viaje",
        }
        nombres = []
        for columna in df.columns:
            nombre = unicodedata.normalize("NFKD", str(columna).strip().lstrip("\ufeff"))
            nombre = nombre.encode("ascii", errors="ignore").decode().lower()
            nombre = re.sub(r"\W+", "_", nombre).strip("_")
            nombres.append(equivalencias.get(nombre, nombre))
        if len(set(nombres)) != len(nombres):
            raise ValueError("Hay nombres de columnas duplicados tras normalizar.")
        faltantes = set(self.COLUMNAS_ESTANDAR) - set(nombres)
        if faltantes:
            raise ValueError(f"Faltan columnas originales: {sorted(faltantes)}")
        return df.set_axis(nombres, axis=1).loc[:, self.COLUMNAS_ESTANDAR].copy()

    def _obtener_periodo_archivo(self, archivo: Path) -> tuple[int, int]:
        match = re.search(r"(\d{4})_(\d{2})", archivo.name)
        if not match:
            raise ValueError(f"No se pudo identificar el periodo de {archivo.name}")
        return int(match.group(1)), int(match.group(2))

    def _convertir_fechas(
        self, serie: pd.Series, archivo: Path
    ) -> tuple[pd.Series, str]:
        fechas = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
        texto = serie.astype("string").str.strip()
        for formato in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
        ):
            pendientes = fechas.isna()
            fechas.loc[pendientes] = pd.to_datetime(
                texto.loc[pendientes], format=formato, errors="coerce"
            )
        return fechas, "iso_o_dia_mes_anio_explicito"

    def _castear_tipos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Castea las columnas al tipo de dato adecuado."""

        df = df.copy()

        # Castear anio_nacimiento a entero nullable
        df["anio_nacimiento"] = pd.to_numeric(
            df["anio_nacimiento"], errors="coerce"
        ).astype("Int64")

        # Castear identificadores a entero
        df["viaje_id"] = pd.to_numeric(df["viaje_id"], errors="coerce").astype("Int64")
        df["usuario_id"] = pd.to_numeric(df["usuario_id"], errors="coerce").astype("Int64")
        df["origen_id"] = pd.to_numeric(df["origen_id"], errors="coerce").astype("Int64")
        df["destino_id"] = pd.to_numeric(df["destino_id"], errors="coerce").astype("Int64")

        # Género y mes_anio como string/categoría
        df["genero"] = df["genero"].astype("str").str.upper().str.strip()
        df["fecha_archivo"] = df["fecha_archivo"].astype("str")

        # Asegurar tipo datetime para inicio y fin de viaje
        df["inicio_viaje"] = pd.to_datetime(df["inicio_viaje"], errors="coerce")
        df["fin_viaje"] = pd.to_datetime(df["fin_viaje"], errors="coerce")

        return df

    def _cargar_con_metadatos(self, archivo: Path) -> tuple[pd.DataFrame, dict]:
        """Carga un archivo mensual y devuelve (DataFrame, metadatos del archivo)."""
        df, encoding = self._leer_csv(archivo)
        n_filas_crudas = len(df)

        df = self._normalizar_columnas(df)
        df["inicio_viaje"], estrategia_inicio = self._convertir_fechas(
            df["inicio_viaje"], archivo
        )
        df["fin_viaje"], estrategia_fin = self._convertir_fechas(
            df["fin_viaje"], archivo
        )

        # Agregar columna con la fecha del archivo
        anio, mes = self._obtener_periodo_archivo(archivo)
        df["fecha_archivo"] = f"{anio}-{mes:02d}"

        # Castear a los tipos finales requeridos
        df = self._castear_tipos(df)

        # Indicadores de calidad (informativos; no se descartan filas aquí)
        inicio = df["inicio_viaje"]
        fuera_de_periodo = int(
            ((inicio.dt.year != anio) | (inicio.dt.month != mes))[inicio.notna()].sum()
        )
        info = {
            "archivo": archivo.name,
            "ruta": _ruta_relativa(archivo),
            "sha256_crudo": calcular_sha256(archivo),
            "codificacion": encoding,
            "n_filas": n_filas_crudas,
            "n_fechas_invalidas": int(inicio.isna().sum() + df["fin_viaje"].isna().sum()),
            "n_inicio_fuera_del_periodo": fuera_de_periodo,
            "estrategia_fecha_inicio": estrategia_inicio,
            "estrategia_fecha_fin": estrategia_fin,
        }
        return df, info

    def cargar_archivo_mensual(self, archivo: Path) -> pd.DataFrame:
        df, _ = self._cargar_con_metadatos(archivo)
        return df

    def cargar_ventana_tiempo(
        self,
        anio_inicio: int = config.DEFAULT_ANIO_INICIO,
        mes_inicio: int = config.DEFAULT_MES_INICIO,
        anio_fin: int = config.DEFAULT_ANIO_FIN,
        mes_fin: int = config.DEFAULT_MES_FIN,
    ) -> tuple[pd.DataFrame, str]:
        """Carga y concatena los archivos en la ventana de tiempo.

        Retorna una tupla con el DataFrame concatenado y un string del rango de
        periodos procesados. Los metadatos de cada archivo quedan en
        `self.registros` y el rango en `self.rango`.
        """
        periodo_inicio = pd.Period(f"{anio_inicio}-{mes_inicio:02d}", freq="M")
        periodo_fin = pd.Period(f"{anio_fin}-{mes_fin:02d}", freq="M")

        archivos_viajes = sorted(self.ruta_datos.glob("datos_abiertos_*.csv"))
        dfs = []
        periodos_encontrados = []
        self.registros = []

        for archivo in archivos_viajes:
            anio, mes = self._obtener_periodo_archivo(archivo)
            periodo_actual = pd.Period(f"{anio}-{mes:02d}", freq="M")

            if periodo_inicio <= periodo_actual <= periodo_fin:
                logger.info(f"Cargando archivo: {archivo.name}")
                df_mes, info = self._cargar_con_metadatos(archivo)
                dfs.append(df_mes)
                self.registros.append(info)
                periodos_encontrados.append(periodo_actual)

        if not dfs:
            raise ValueError(
                "No se encontraron archivos en la ventana de tiempo especificada."
            )

        # Determinar el rango de fechas reales procesadas
        periodo_min = min(periodos_encontrados)
        periodo_max = max(periodos_encontrados)
        rango_str = f"{periodo_min} al {periodo_max}"
        self.rango = {"inicio": str(periodo_min), "fin": str(periodo_max)}

        df_concatenado = pd.concat(dfs, ignore_index=True)
        return df_concatenado, rango_str

    def registrar_nomenclatura(self) -> list[dict]:
        """Registra hash y número de estaciones de los archivos de nomenclatura."""
        registros = []
        for archivo in sorted(self.ruta_datos.glob("nomenclatura_*.csv")):
            df, encoding = self._leer_csv(archivo)
            registros.append(
                {
                    "archivo": archivo.name,
                    "ruta": _ruta_relativa(archivo),
                    "sha256_crudo": calcular_sha256(archivo),
                    "codificacion": encoding,
                    "n_filas": len(df),
                }
            )
        if not registros:
            logger.warning("No se encontró ningún archivo nomenclatura_*.csv.")
        return registros


# ---------------------------------------------------------------------------
# Manifiesto de INGESTA (CSV crudos -> Parquet intermedio)
# ---------------------------------------------------------------------------
class DataManifest:
    """Construye, guarda, carga y verifica el manifiesto de datos de ingesta.

    El manifiesto es un JSON versionado en Git (sin datos personales) que
    describe exactamente qué datos se usaron: hash y número de filas de cada
    archivo, rango de fechas, columnas de salida y una `data_version` global.
    """

    VERSION_ESQUEMA = 2

    def __init__(self, ruta: str | Path = config.MANIFEST_PATH):
        self.ruta = Path(ruta)

    @staticmethod
    def calcular_version(
        rango: dict, archivos: list[dict], estaciones: list[dict], columnas: list[str]
    ) -> str:
        """Hash corto que resume toda la entrada de datos.

        Solo depende de contenido (hashes de archivos crudos, filas, rango y
        columnas de salida). No incluye fechas de generación ni rutas
        absolutas, para que sea idéntico en cualquier computadora.
        """
        carga = {
            "normalizacion": NORMALIZATION_VERSION,
            "rango": rango,
            "archivos": [
                [a["archivo"], a["sha256_crudo"], a["n_filas"]]
                for a in sorted(archivos, key=lambda a: a["archivo"])
            ],
            "estaciones": [
                [e["archivo"], e["sha256_crudo"], e["n_filas"]]
                for e in sorted(estaciones, key=lambda e: e["archivo"])
            ],
            "columnas": columnas,
        }
        texto = json.dumps(carga, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]

    def construir(
        self,
        rango: dict,
        registros_viajes: list[dict],
        registros_estaciones: list[dict],
        ruta_parquet: Path,
        columnas: list[str],
        excluye_datos_personales: bool,
    ) -> dict:
        ruta_parquet = Path(ruta_parquet)
        n_filas_parquet = pq.ParquetFile(ruta_parquet).metadata.num_rows

        manifiesto = {
            "version_esquema": self.VERSION_ESQUEMA,
            "data_version": self.calcular_version(
                rango, registros_viajes, registros_estaciones, columnas
            ),
            "generado_en": datetime.now().astimezone().isoformat(timespec="seconds"),
            "rango_fechas": rango,
            "excluye_datos_personales": excluye_datos_personales,
            "entorno": {
                "pandas": metadata.version("pandas"),
                "pyarrow": metadata.version("pyarrow"),
            },
            "resumen": {
                "n_meses": len(registros_viajes),
                "n_viajes": int(sum(r["n_filas"] for r in registros_viajes)),
                "n_estaciones_nomenclatura": int(
                    sum(r["n_filas"] for r in registros_estaciones)
                ),
            },
            "viajes_mensuales": registros_viajes,
            "nomenclatura": registros_estaciones,
            "salida": {
                "ruta": _ruta_relativa(ruta_parquet),
                "n_filas": int(n_filas_parquet),
                "columnas": columnas,
                "sha256": calcular_sha256(ruta_parquet),
                "nota": "El hash verifica esta salida concreta. Regenerar Parquet con "
                "otra versión de pyarrow puede cambiar sus bytes y requiere actualizar "
                "el manifiesto. data_version usa entradas, esquema y reglas de limpieza.",
            },
        }
        return manifiesto

    def guardar(self, manifiesto: dict) -> Path:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.ruta.write_text(
            json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.ruta

    def cargar(self) -> dict:
        if not self.ruta.exists():
            raise FileNotFoundError(
                f"No existe el manifiesto en {self.ruta}. Ejecuta primero la ingesta."
            )
        return json.loads(self.ruta.read_text(encoding="utf-8"))

    def verificar(self) -> list[str]:
        """Recalcula hashes de los CSV y conteos del Parquet; devuelve discrepancias."""
        manifiesto = self.cargar()
        errores: list[str] = []

        for entrada in manifiesto["viajes_mensuales"] + manifiesto["nomenclatura"]:
            ruta = config.PROJ_ROOT / entrada["ruta"]
            if not ruta.exists():
                errores.append(f"Falta el archivo: {entrada['ruta']}")
            elif calcular_sha256(ruta) != entrada["sha256_crudo"]:
                errores.append(f"El hash no coincide: {entrada['ruta']}")

        errores.extend(self.verificar_parquet())
        if "salida_estaciones" in manifiesto:
            salida = manifiesto["salida_estaciones"]
            ruta = config.PROJ_ROOT / salida["ruta"]
            if not ruta.is_file():
                errores.append("Falta el Parquet de estaciones.")
            else:
                pf = pq.ParquetFile(ruta)
                if (pf.metadata.num_rows != salida["n_filas"]
                        or pf.schema_arrow.names != salida["columnas"]
                        or calcular_sha256(ruta) != salida["sha256"]):
                    errores.append("El Parquet de estaciones no coincide con el manifiesto.")
        return errores

    def verificar_parquet(self, ruta_parquet: Path | None = None) -> list[str]:
        """Valida el Parquet intermedio contra el manifiesto, sin necesitar los CSV."""
        m = self.cargar()
        salida = m["salida"]
        ruta = Path(ruta_parquet) if ruta_parquet else config.PROJ_ROOT / salida["ruta"]
        if not ruta.exists():
            return [f"No existe el Parquet: {ruta}"]

        errores: list[str] = []

        # 1. Estructura
        pf = pq.ParquetFile(ruta)
        if pf.metadata.num_rows != salida["n_filas"]:
            errores.append(
                f"Filas: {pf.metadata.num_rows} vs {salida['n_filas']} esperadas."
            )
        if pf.schema_arrow.names != salida["columnas"]:
            errores.append("Las columnas no coinciden con el manifiesto.")
        if m["excluye_datos_personales"] and set(COLUMNAS_PERSONALES) & set(
            pf.schema_arrow.names
        ):
            errores.append("El Parquet contiene columnas personales.")

        if errores:
            return errores

        # 2. Contenido: conteos por mes (independiente de la versión de pyarrow)
        df = pd.read_parquet(ruta, columns=["fecha_archivo", "viaje_id"])
        obtenidos = df["fecha_archivo"].value_counts().to_dict()
        for registro in m["viajes_mensuales"]:
            coincidencia = re.search(r"(\d{4})_(\d{2})", registro["archivo"])
            mes = f"{coincidencia.group(1)}-{coincidencia.group(2)}"
            esperadas = registro.get("n_filas_salida", registro["n_filas"])
            if obtenidos.get(mes, 0) != esperadas:
                errores.append(
                    f"{mes}: {obtenidos.get(mes, 0)} viajes vs {esperadas} esperados."
                )

        # 3. Informativo
        if df["viaje_id"].duplicated().any():
            logger.warning("Hay viaje_id duplicados (revisar si es esperado).")
        if calcular_sha256(ruta) != salida["sha256"]:
            errores.append("El hash del Parquet no coincide con el manifiesto.")
        return errores


def obtener_version_datos(ruta: str | Path = config.MANIFEST_PATH) -> str:
    """Devuelve la `data_version` del manifiesto (para registrarla en MLflow)."""
    return DataManifest(ruta).cargar()["data_version"]


# La versión incluye las reglas de limpieza, no solo los bytes de entrada.
NORMALIZATION_VERSION = "mibici-ingesta-v3"
TRIP_SCHEMA = pa.schema([
    ("viaje_id", pa.int64()),
    ("inicio_viaje", pa.timestamp("ns")),
    ("fin_viaje", pa.timestamp("ns")),
    ("origen_id", pa.int64()),
    ("destino_id", pa.int64()),
    ("fecha_archivo", pa.string()),
])


class _EnlacesCSV(HTMLParser):
    def __init__(self):
        super().__init__()
        self.enlaces = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.enlaces.extend(value for key, value in attrs if key == "href" and value)


def _url_oficial(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc in {"www.mibici.net", "mibici.net"}


def _solicitud(url: str) -> Request:
    """Única puerta a la red: rechaza esquemas como file:// y dominios ajenos a MiBici."""
    if not _url_oficial(url):
        raise ValueError("Solo se permiten descargas HTTPS del dominio oficial de MiBici.")
    return Request(url, headers={
        "User-Agent": "MiBici-Equipo25/1.0",
        "Accept": "text/html,text/csv,application/octet-stream,*/*;q=0.8",
    })


def _abrir(url: str):
    """Abre solo URLs que _solicitud acepta (HTTPS de mibici.net); por eso se omite B310."""
    return urlopen(_solicitud(url), timeout=config.DOWNLOAD_TIMEOUT_SECONDS)  # nosec B310


def descubrir_fuentes() -> dict[str, str]:
    """Resuelve URLs publicadas por MiBici sin inventar los IDs de sus directorios."""
    with _abrir(config.DATA_URL_ONLINE) as response:
        parser = _EnlacesCSV()
        parser.feed(response.read().decode("utf-8"))
    fuentes = {}
    for enlace in parser.enlaces:
        url = urljoin(config.DATA_URL_ONLINE, enlace)
        nombre = Path(unquote(urlparse(url).path)).name
        if _url_oficial(url) and nombre.lower().endswith(".csv"):
            fuentes[nombre] = url
    return fuentes


def descargar_archivo(url: str, destino: Path) -> None:
    """Descarga en un temporal y publica el CSV solo si la transferencia concluye."""
    if not _url_oficial(url):
        raise ValueError("Solo se permiten descargas HTTPS del dominio oficial de MiBici.")
    destino.parent.mkdir(parents=True, exist_ok=True)
    for intento in range(config.DOWNLOAD_ATTEMPTS):
        temporal = None
        try:
            with tempfile.NamedTemporaryFile(dir=destino.parent, delete=False) as output:
                temporal = Path(output.name)
                with _abrir(url) as response:
                    if not _url_oficial(response.geturl()):
                        raise ValueError("La descarga redirigió fuera del dominio oficial.")
                    shutil.copyfileobj(response, output)
                    expected = response.headers.get("Content-Length")
                    if expected is not None and output.tell() != int(expected):
                        raise OSError("Descarga incompleta.")
                if output.tell() == 0:
                    raise OSError("El archivo descargado está vacío.")
            os.replace(temporal, destino)
            return
        except (OSError, ValueError):
            if intento + 1 == config.DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(2 ** intento)
        finally:
            if temporal is not None:
                temporal.unlink(missing_ok=True)


def _codificacion(archivo: Path) -> str:
    """Detecta UTF-8/BOM, Windows-1252 o Latin-1 con decodificación estricta."""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        try:
            with archivo.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    decoder.decode(block)
                decoder.decode(b"", final=True)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {archivo.name}")


def _entero(serie: pd.Series) -> pd.Series:
    numeros = pd.to_numeric(serie, errors="coerce")
    return numeros.where(numeros.mod(1).eq(0) & numeros.gt(0)).astype("Int64")


def _normalizar_estaciones(archivo: Path) -> tuple[pd.DataFrame, dict]:
    encoding = _codificacion(archivo)
    frame = pd.read_csv(archivo, encoding=encoding, dtype="string")
    frame.columns = frame.columns.str.strip().str.lower()
    columnas = ["id", "name", "latitude", "longitude", "dpcapacity"]
    if not set(columnas).issubset(frame.columns):
        raise ValueError("La nomenclatura no contiene las columnas requeridas.")
    frame = frame.loc[:, columnas].copy()
    frame["id"] = _entero(frame["id"])
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("La nomenclatura contiene IDs inválidos o duplicados.")
    frame["name"] = frame["name"].str.strip()
    for name, lower, upper in (("latitude", -90, 90), ("longitude", -180, 180)):
        values = pd.to_numeric(frame[name], errors="coerce")
        frame[name] = values.where(values.between(lower, upper)).astype("Float64")
    frame["dpcapacity"] = _entero(frame["dpcapacity"])
    return frame, {
        "archivo": archivo.name,
        "ruta": _ruta_relativa(archivo),
        "sha256_crudo": calcular_sha256(archivo),
        "codificacion": encoding,
        "n_filas": len(frame),
        "n_sin_capacidad": int(frame["dpcapacity"].isna().sum()),
        "n_sin_coordenadas": int(frame[["latitude", "longitude"]].isna().any(axis=1).sum()),
    }


def _limpiar_lote(frame: pd.DataFrame, archivo: Path, loader: MibiciDataLoader) -> pd.DataFrame:
    frame = loader._normalizar_columnas(frame).drop(columns=COLUMNAS_PERSONALES)
    for name in ("viaje_id", "origen_id", "destino_id"):
        frame[name] = _entero(frame[name])
    for name in ("inicio_viaje", "fin_viaje"):
        frame[name], _ = loader._convertir_fechas(frame[name], archivo)
    year, month = loader._obtener_periodo_archivo(archivo)
    frame["fecha_archivo"] = f"{year}-{month:02d}"
    valid = (
        frame[["viaje_id", "origen_id", "destino_id", "inicio_viaje", "fin_viaje"]]
        .notna().all(axis=1)
        & frame["fin_viaje"].ge(frame["inicio_viaje"])
        & frame["inicio_viaje"].dt.year.eq(year)
        & frame["inicio_viaje"].dt.month.eq(month)
    )
    return frame.loc[valid, TRIP_SCHEMA.names].reset_index(drop=True)


def _sin_duplicados(frame: pd.DataFrame, connection: sqlite3.Connection) -> pd.DataFrame:
    """Conserva la primera aparición del viaje en orden de archivo/fila, sin cargar todos los IDs."""
    connection.execute("DELETE FROM lote")
    connection.executemany(
        "INSERT INTO lote VALUES (?, ?)",
        enumerate(int(value) for value in frame["viaje_id"]),
    )
    indices = [row[0] for row in connection.execute(
        "SELECT MIN(posicion) FROM lote WHERE id NOT IN (SELECT id FROM vistos) "
        "GROUP BY id ORDER BY MIN(posicion)"
    )]
    connection.execute("INSERT OR IGNORE INTO vistos SELECT id FROM lote")
    connection.commit()
    return frame.iloc[indices].copy()


def ejecutar_ingesta(
    *,
    anio_inicio: int = config.DEFAULT_ANIO_INICIO,
    mes_inicio: int = config.DEFAULT_MES_INICIO,
    anio_fin: int = config.DEFAULT_ANIO_FIN,
    mes_fin: int = config.DEFAULT_MES_FIN,
    input_path: Path = config.RAW_DATA_DIR,
    parquet_path: Path = config.TRIPS_PARQUET_PATH,
    estaciones_path: Path = config.STATIONS_PARQUET_PATH,
    manifiesto_path: Path = config.MANIFEST_PATH,
    sin_descarga: bool = False,
    actualizar: bool = False,
    chunk_size: int = config.INGESTION_CHUNK_SIZE,
) -> dict:
    """Descarga, limpia por lotes y versiona una base común sin columnas personales.

    Descarta fechas/IDs inválidos, duraciones negativas, inicios fuera del mes fuente
    y viajes repetidos. No corta llegadas al fin del rango ni aplica filtros propios
    de una pregunta. El catálogo no determina la actividad histórica de estaciones.
    """
    if chunk_size < 1:
        raise ValueError("chunk-size debe ser positivo.")
    if sin_descarga and actualizar:
        raise ValueError("--actualizar no puede combinarse con --sin-descarga.")
    start = pd.Period(year=anio_inicio, month=mes_inicio, freq="M")
    end = pd.Period(year=anio_fin, month=mes_fin, freq="M")
    if start > end:
        raise ValueError("El inicio del rango es posterior al fin.")
    input_path = Path(input_path)
    parquet_path, estaciones_path, manifiesto_path = map(
        Path, (parquet_path, estaciones_path, manifiesto_path)
    )
    if len({p.resolve() for p in (parquet_path, estaciones_path, manifiesto_path)}) != 3:
        raise ValueError("Viajes, estaciones y manifiesto requieren rutas diferentes.")
    months = pd.period_range(start, end, freq="M")
    expected = [f"datos_abiertos_{month.strftime('%Y_%m')}.csv" for month in months]
    station_name = f"nomenclatura_{config.STATIONS_SNAPSHOT_MONTH.replace('-', '_')}.csv"
    fuentes = {} if sin_descarga else descubrir_fuentes()
    archivos = []
    for nombre in [*expected, station_name]:
        local = input_path / nombre
        candidatos = []
        if nombre.startswith("datos_abiertos_"):
            candidatos = sorted(input_path.glob(nombre.removesuffix(".csv") + "-*.csv"))
            if (local.is_file() and candidatos) or len(candidatos) > 1:
                raise ValueError(f"Hay varios CSV para el mismo mes: {nombre}")
        if not local.is_file() and candidatos:
            local = candidatos[0]
        if actualizar or not local.is_file():
            if sin_descarga:
                raise FileNotFoundError(f"Falta el archivo requerido: {nombre}")
            url = fuentes.get(nombre)
            if url is None:
                alternativas = [key for key in fuentes if re.fullmatch(
                    re.escape(nombre.removesuffix('.csv')) + r'-\d+\.csv', key
                )]
                if len(alternativas) != 1:
                    raise ValueError(f"No hay un enlace único publicado para {nombre}")
                url = fuentes[alternativas[0]]
            descargar_archivo(url, local)
        archivos.append(local)
    if {path.resolve() for path in archivos} & {
        path.resolve() for path in (parquet_path, estaciones_path, manifiesto_path)
    }:
        raise ValueError("Las salidas no pueden reemplazar archivos de origen.")
    loader = MibiciDataLoader(input_path)
    stations, station_info = _normalizar_estaciones(archivos[-1])
    station_info["url"] = fuentes.get(station_name)
    for output in (parquet_path, estaciones_path, manifiesto_path):
        output.parent.mkdir(parents=True, exist_ok=True)
    registros = []
    with tempfile.TemporaryDirectory(prefix=".ingesta-", dir=parquet_path.parent) as tmp:
        temporary = Path(tmp)
        trips_tmp = temporary / "viajes.parquet"
        # closing() libera el archivo; Windows no permite borrar el temporal con la conexión abierta.
        with closing(sqlite3.connect(temporary / "deduplicacion.sqlite")) as connection:
            connection.execute("CREATE TABLE vistos (id INTEGER PRIMARY KEY)")
            connection.execute("CREATE TEMP TABLE lote (posicion INTEGER, id INTEGER)")
            with pq.ParquetWriter(trips_tmp, TRIP_SCHEMA, compression="snappy") as writer:
                for month, archivo in zip(months, archivos[:-1], strict=True):
                    encoding = _codificacion(archivo)
                    info = {
                        "archivo": archivo.name, "ruta": _ruta_relativa(archivo),
                        "url": fuentes.get(archivo.name) or fuentes.get(
                            f"datos_abiertos_{month.strftime('%Y_%m')}.csv"
                        ),
                        "sha256_crudo": calcular_sha256(archivo), "codificacion": encoding,
                        "n_filas": 0, "n_descartadas_calidad": 0, "n_duplicados": 0,
                        "n_filas_salida": 0,
                    }
                    for frame in pd.read_csv(
                        archivo, encoding=encoding, dtype="string", chunksize=chunk_size
                    ):
                        info["n_filas"] += len(frame)
                        cleaned = _limpiar_lote(frame, archivo, loader)
                        info["n_descartadas_calidad"] += len(frame) - len(cleaned)
                        unique = _sin_duplicados(cleaned, connection)
                        info["n_duplicados"] += len(cleaned) - len(unique)
                        info["n_filas_salida"] += len(unique)
                        writer.write_table(pa.Table.from_pandas(
                            unique, schema=TRIP_SCHEMA, preserve_index=False
                        ))
                    if info["n_filas"] == 0:
                        raise ValueError(f"El archivo no contiene registros: {archivo.name}")
                    registros.append(info)
                    logger.info("{}: {} registros válidos", month, info["n_filas_salida"])
        if not any(item["n_filas_salida"] for item in registros):
            raise ValueError("La ventana no contiene viajes válidos.")
        stations_tmp = temporary / "estaciones.parquet"
        stations.to_parquet(stations_tmp, index=False, compression="snappy")
        manifest = DataManifest(manifiesto_path).construir(
            {"inicio": str(start), "fin": str(end)}, registros, [station_info],
            trips_tmp, TRIP_SCHEMA.names, True,
        )
        manifest["salida"]["ruta"] = _ruta_relativa(parquet_path)
        manifest["salida_estaciones"] = {
            "ruta": _ruta_relativa(estaciones_path), "n_filas": len(stations),
            "columnas": list(stations.columns), "sha256": calcular_sha256(stations_tmp),
            "mes_referencia": config.STATIONS_SNAPSHOT_MONTH,
        }
        manifest["normalizacion"] = NORMALIZATION_VERSION
        manifest["resumen"]["n_viajes_limpios"] = sum(r["n_filas_salida"] for r in registros)
        manifest_tmp = temporary / "manifiesto.json"
        DataManifest(manifest_tmp).guardar(manifest)
        # El manifiesto se publica al final; no se anuncian resultados de una ingesta parcial.
        for source, output in ((trips_tmp, parquet_path), (stations_tmp, estaciones_path),
                               (manifest_tmp, manifiesto_path)):
            os.replace(source, output)
    return manifest


# ---------------------------------------------------------------------------
# Comandos de terminal
# ---------------------------------------------------------------------------
@app.command()
def procesar_datos(
    anio_inicio: int = config.DEFAULT_ANIO_INICIO,
    mes_inicio: int = config.DEFAULT_MES_INICIO,
    anio_fin: int = config.DEFAULT_ANIO_FIN,
    mes_fin: int = config.DEFAULT_MES_FIN,
    input_path: Path = config.RAW_DATA_DIR,
    output_filename: str = config.TRIPS_PARQUET_PATH.name,
    output_path: Path = config.INTERIM_DATA_DIR,
    manifiesto_path: Path = config.MANIFEST_PATH,
    sin_descarga: bool = False,
    actualizar: bool = False,
    chunk_size: int = config.INGESTION_CHUNK_SIZE,
):
    """Descarga CSV faltantes, limpia viajes/estaciones y publica Parquet y manifiesto."""
    manifest = ejecutar_ingesta(
        anio_inicio=anio_inicio, mes_inicio=mes_inicio,
        anio_fin=anio_fin, mes_fin=mes_fin,
        input_path=input_path, parquet_path=output_path / output_filename,
        estaciones_path=output_path / config.STATIONS_PARQUET_PATH.name,
        manifiesto_path=manifiesto_path, sin_descarga=sin_descarga,
        actualizar=actualizar, chunk_size=chunk_size,
    )
    logger.success(
        "Ingesta completa | data_version={} | viajes válidos={:,}",
        manifest["data_version"], manifest["resumen"]["n_viajes_limpios"],
    )


@app.command()
def verificar_manifiesto(
    manifiesto_path: Path = config.MANIFEST_PATH,
):
    """Comprueba que los CSV y el Parquet locales coinciden con el manifiesto."""
    errores = DataManifest(manifiesto_path).verificar()
    if errores:
        for error in errores:
            logger.error(error)
        raise typer.Exit(code=1)
    logger.success("Los datos locales coinciden con el manifiesto.")


@app.command()
def verificar_parquet(
    parquet_path: Path = config.INTERIM_DATA_DIR / "viajes_unificados.parquet",
    manifiesto_path: Path = config.MANIFEST_PATH,
):
    """Valida el Parquet intermedio contra el manifiesto (no requiere los CSV)."""
    errores = DataManifest(manifiesto_path).verificar_parquet(parquet_path)
    if errores:
        for error in errores:
            logger.error(error)
        raise typer.Exit(code=1)
    logger.success("El Parquet coincide con el manifiesto.")


if __name__ == "__main__":
    app()