from pathlib import Path
import re
from loguru import logger
import pandas as pd
import typer

from proyecto_mibici.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


class MibiciDataLoader:
    """Clase para cargar y procesar los archivos históricos mensuales de MiBici,

    resolviendo problemas de codificación y ambigüedad en las fechas.
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

    def __init__(self, ruta_datos: str | Path):
        self.ruta_datos = Path(ruta_datos)
        if not self.ruta_datos.exists():
            raise FileNotFoundError(
                f"No se encontró la ruta de datos: {self.ruta_datos}."
            )

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
    
    def cargar_archivo_mensual(self, archivo: Path) -> pd.DataFrame:
        df, _ = self._leer_csv(archivo)
        df = self._normalizar_columnas(df)
        df["inicio_viaje"], _ = self._convertir_fechas(df["inicio_viaje"], archivo)
        df["fin_viaje"], _ = self._convertir_fechas(df["fin_viaje"], archivo)

        # Agregar columna con la fecha del archivo
        anio, mes = self._obtener_periodo_archivo(archivo)
        df["fecha_archivo"] = f"{anio}-{mes:02d}"
        
        # Castear a los tipos finales requeridos
        df = self._castear_tipos(df)

        return df

    def cargar_ventana_tiempo(
        self, anio_inicio: int, mes_inicio: int, anio_fin: int, mes_fin: int
    ) -> tuple[pd.DataFrame, str]:
        """Carga y concatena los archivos en la ventana de tiempo.

        Retorna una tupla con el DataFrame concatenado y un string del rango de
        periodos procesados.
        """
        periodo_inicio = pd.Period(f"{anio_inicio}-{mes_inicio:02d}", freq="M")
        periodo_fin = pd.Period(f"{anio_fin}-{mes_fin:02d}", freq="M")

        archivos_viajes = sorted(self.ruta_datos.glob("datos_abiertos_*.csv"))
        dfs = []
        periodos_encontrados = []

        for archivo in archivos_viajes:
            anio, mes = self._obtener_periodo_archivo(archivo)
            periodo_actual = pd.Period(f"{anio}-{mes:02d}", freq="M")

            if periodo_inicio <= periodo_actual <= periodo_fin:
                logger.info(f"Cargando archivo: {archivo.name}")
                df_mes = self.cargar_archivo_mensual(archivo)
                dfs.append(df_mes)
                periodos_encontrados.append(periodo_actual)

        if not dfs:
            raise ValueError(
                "No se encontraron archivos en la ventana de tiempo especificada."
            )

        # Determinar el rango de fechas reales procesadas
        periodo_min = min(periodos_encontrados)
        periodo_max = max(periodos_encontrados)
        rango_str = f"{periodo_min} al {periodo_max}"

        df_concatenado = pd.concat(dfs, ignore_index=True)
        return df_concatenado, rango_str


@app.command()
def procesar_datos(
    anio_inicio: int = 2023,
    mes_inicio: int = 10,
    anio_fin: int = 2026,
    mes_fin: int = 9,
    input_path: Path = RAW_DATA_DIR,
    output_filename: str = "viajes_unificados.parquet",
):
    """Comando de Typer para unificar los CSVs crudos, reportar el rango de

    fechas y guardarlo en formato interim (Parquet).
    """
    logger.info(f"Iniciando proceso desde la ruta local: {input_path}")
    cargador = MibiciDataLoader(input_path)

    # Cargar datos y obtener el rango de fechas detectadas
    df_procesado, rango_fechas = cargador.cargar_ventana_tiempo(
        anio_inicio, mes_inicio, anio_fin, mes_fin
    )

    logger.success(
        f"¡Archivos concatenados con éxito! Rango de fechas procesado: [{rango_fechas}]"
    )

    # Definir ruta de salida en data/interim/ y asegurar que exista
    output_path = INTERIM_DATA_DIR / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar en formato Parquet comprimido
    df_procesado.to_parquet(output_path, index=False, compression="snappy")

    logger.success(f"Archivo intermedio guardado correctamente en: {output_path}")


if __name__ == "__main__":
    app()
