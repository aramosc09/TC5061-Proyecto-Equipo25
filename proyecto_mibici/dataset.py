import hashlib
import json
from datetime import datetime
from importlib import metadata
from pathlib import Path
import re

from loguru import logger
import pandas as pd
import pyarrow.parquet as pq
import typer

from proyecto_mibici import config

app = typer.Typer(help = "Ingesta reproducible y versionada de los datos de MiBici.")

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
        return Path(ruta).name


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
        if len(df.columns) != len(self.COLUMNAS_ESTANDAR):
            raise ValueError(
                f"Se esperaban {len(self.COLUMNAS_ESTANDAR)} columnas "
                f"y se encontraron {len(df.columns)}."
            )
        df = df.copy()
        df.columns = self.COLUMNAS_ESTANDAR
        return df

    def _obtener_periodo_archivo(self, archivo: Path) -> tuple[int, int]:
        match = re.search(r"(\d{4})_(\d{2})", archivo.name)
        if not match:
            raise ValueError(f"No se pudo identificar el periodo de {archivo.name}")
        return int(match.group(1)), int(match.group(2))

    def _convertir_fechas(
        self, serie: pd.Series, archivo: Path
    ) -> tuple[pd.Series, str]:
        anio_esperado, mes_esperado = self._obtener_periodo_archivo(archivo)
        opciones = {}

        for estrategia, dayfirst in [("default", False), ("dayfirst", True)]:
            fechas = pd.to_datetime(serie, errors="coerce", dayfirst=dayfirst)
            coincide_periodo = (fechas.dt.year == anio_esperado) & (
                fechas.dt.month == mes_esperado
            )
            opciones[estrategia] = {
                "fechas": fechas,
                "coincidencias": coincide_periodo.sum(),
            }

        mejor_estrategia = max(
            opciones, key=lambda x: opciones[x]["coincidencias"]
        )
        return opciones[mejor_estrategia]["fechas"], mejor_estrategia

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

    VERSION_ESQUEMA = 1

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
                "nota": "El hash del Parquet es informativo: puede variar entre "
                "versiones de pyarrow. La versión de datos usa los hashes crudos.",
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

        salida = manifiesto["salida"]
        ruta_parquet = config.PROJ_ROOT / salida["ruta"]
        if not ruta_parquet.exists():
            errores.append(f"Falta el Parquet: {salida['ruta']}")
        else:
            if pq.ParquetFile(ruta_parquet).metadata.num_rows != salida["n_filas"]:
                errores.append("El número de filas del Parquet no coincide.")
            if calcular_sha256(ruta_parquet) != salida["sha256"]:
                logger.warning(
                    "El hash del Parquet difiere (puede deberse a otra versión de "
                    "pyarrow). Se valida con los hashes de los archivos crudos."
                )
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

        # 2. Contenido: conteos por mes (independiente de la versión de pyarrow)
        df = pd.read_parquet(ruta, columns=["fecha_archivo", "viaje_id"])
        obtenidos = df["fecha_archivo"].value_counts().to_dict()
        for registro in m["viajes_mensuales"]:
            coincidencia = re.search(r"(\d{4})_(\d{2})", registro["archivo"])
            mes = f"{coincidencia.group(1)}-{coincidencia.group(2)}"
            if obtenidos.get(mes) != registro["n_filas"]:
                errores.append(
                    f"{mes}: {obtenidos.get(mes)} viajes vs {registro['n_filas']} esperados."
                )

        # 3. Informativo
        if df["viaje_id"].duplicated().any():
            logger.warning("Hay viaje_id duplicados (revisar si es esperado).")
        if calcular_sha256(ruta) != salida["sha256"]:
            logger.warning(
                "El hash del archivo difiere del manifiesto. Es esperable si el "
                "Parquet se regeneró con otra versión de pyarrow; la validación "
                "se basa en estructura y conteos."
            )
        return errores


def obtener_version_datos(ruta: str | Path = config.MANIFEST_PATH) -> str:
    """Devuelve la `data_version` del manifiesto (para registrarla en MLflow)."""
    return DataManifest(ruta).cargar()["data_version"]


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
    output_filename: str = "viajes_unificados.parquet",
    manifiesto_path: Path = config.MANIFEST_PATH,
    excluir_datos_personales: bool = typer.Option(
        False,
        "--excluir-datos-personales/--conservar-datos-personales",
        help="Por defecto conserva usuario_id, genero y anio_nacimiento en el "
        "Parquet local. Con --excluir-datos-personales los elimina (útil para "
        "generar una versión compartible).",
    ),
):
    """Procesa los CSV, guarda el Parquet y genera el manifiesto de datos."""

    logger.info(f"Iniciando proceso desde la ruta local: {input_path}")
    cargador = MibiciDataLoader(input_path)

    # Cargar datos y obtener el rango de fechas detectadas
    df_procesado, rango_fechas = cargador.cargar_ventana_tiempo(
        anio_inicio, mes_inicio, anio_fin, mes_fin
    )

    logger.success(
        f"¡Archivos concatenados con éxito! Rango de fechas procesado: [{rango_fechas}]"
    )

    # Minimización de datos personales (la decisión queda en el manifiesto)
    if excluir_datos_personales:
        df_procesado = df_procesado.drop(columns=COLUMNAS_PERSONALES)
        logger.info(f"Columnas personales eliminadas: {COLUMNAS_PERSONALES}")

    # Definir ruta de salida en data/interim/ y asegurar que exista
    output_path = config.INTERIM_DATA_DIR / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar en formato Parquet comprimido
    df_procesado.to_parquet(output_path, index=False, compression="snappy")
    logger.success(f"Archivo intermedio guardado correctamente en: {output_path}")

    # Generar y guardar el manifiesto
    gestor = DataManifest(manifiesto_path)
    manifiesto = gestor.construir(
        rango=cargador.rango,
        registros_viajes=cargador.registros,
        registros_estaciones=cargador.registrar_nomenclatura(),
        ruta_parquet=output_path,
        columnas=list(df_procesado.columns),
        excluye_datos_personales=excluir_datos_personales,
    )
    ruta_manifiesto = gestor.guardar(manifiesto)

    resumen = manifiesto["resumen"]
    logger.success(
        f"Manifiesto guardado en {ruta_manifiesto} | "
        f"data_version={manifiesto['data_version']} | "
        f"meses={resumen['n_meses']} | viajes={resumen['n_viajes']:,}"
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