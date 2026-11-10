"""Periodos de referencia, actual y A/B de P1, definidos por fecha sobre la tabla supervisada."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from proyecto_mibici.config import DEFAULT_RANDOM_STATE, ProjectConfig
from proyecto_mibici.modeling.datasets import P1_FEATURES

COLUMNAS_P1 = ["estacion", "hora", *P1_FEATURES, "salidas"]
COMPARACIONES = {
    "principal": ("referencia", "actual"),
    "estacional": ("referencia_estacional", "actual_estacional"),
}
_UN_ANIO = pd.DateOffset(years=1)
_LOTE = 500_000


@dataclass(frozen=True)
class Periodo:
    """Intervalo [inicio, fin) de horas de la tabla supervisada."""

    nombre: str
    inicio: pd.Timestamp
    fin: pd.Timestamp
    descripcion: str

    def contiene(self, horas) -> np.ndarray:
        horas = np.asarray(horas, dtype="datetime64[ns]")
        return (horas >= self.inicio.to_datetime64()) & (horas < self.fin.to_datetime64())

    def como_dict(self) -> dict:
        return {
            "inicio": self.inicio.isoformat(),
            "fin_exclusivo": self.fin.isoformat(),
            "descripcion": self.descripcion,
        }


def definir_periodos(config: ProjectConfig | None = None) -> dict[str, Periodo]:
    """Deriva todos los periodos de los cortes con los que se entrenó la v1."""
    config = config or ProjectConfig()
    train_end = pd.Timestamp(config.train_end)
    validation_end = pd.Timestamp(config.validation_end)
    # La v2 repite el protocolo de la v1 desplazado un año: entrena hasta train_end + 1 año.
    corte_v2 = train_end + _UN_ANIO
    definiciones = [
        ("referencia", config.start, train_end, "Entrenamiento de la v1 (champion)"),
        (
            "actual",
            validation_end,
            config.end,
            "Prueba de la v1: no participó en su entrenamiento ni en su selección",
        ),
        (
            "referencia_estacional",
            validation_end - _UN_ANIO,
            train_end,
            "Meses de la referencia con su equivalente en el periodo actual",
        ),
        (
            "actual_estacional",
            validation_end,
            corte_v2,
            "Mismos meses de calendario que la referencia estacional, un año después",
        ),
        ("entrenamiento_v2", config.start, corte_v2, "Ventana de entrenamiento de la v2"),
        ("evaluacion_ab", corte_v2, config.end, "Evaluación A/B: ninguna versión la vio"),
    ]
    periodos = {
        nombre: Periodo(nombre, pd.Timestamp(inicio), pd.Timestamp(fin), descripcion)
        for nombre, inicio, fin, descripcion in definiciones
    }
    validar_periodos(periodos, config)
    return periodos


def _disjuntos(a: Periodo, b: Periodo) -> bool:
    return a.fin <= b.inicio or b.fin <= a.inicio


def validar_periodos(periodos: dict[str, Periodo], config: ProjectConfig) -> None:
    """Rechaza periodos vacíos, fuera de los datos o que mezclen lo visto con lo evaluado."""
    for periodo in periodos.values():
        if not config.start <= periodo.inicio < periodo.fin <= config.end:
            raise ValueError(f"El periodo {periodo.nombre} está vacío o fuera de los datos.")
        if periodo.inicio != periodo.inicio.floor("h") or periodo.fin != periodo.fin.floor("h"):
            raise ValueError(f"El periodo {periodo.nombre} no coincide con horas completas.")
    pares_sin_traslape = [
        ("referencia", "actual"),
        ("referencia_estacional", "actual_estacional"),
        ("entrenamiento_v2", "evaluacion_ab"),
        ("referencia", "evaluacion_ab"),
    ]
    for a, b in pares_sin_traslape:
        if not _disjuntos(periodos[a], periodos[b]):
            raise ValueError(f"Los periodos {a} y {b} se traslapan.")
    if _meses(periodos["referencia_estacional"]) != _meses(periodos["actual_estacional"]):
        raise ValueError("La comparación estacional debe cubrir los mismos meses de calendario.")


def _meses(periodo: Periodo) -> list[int]:
    meses = pd.period_range(periodo.inicio, periodo.fin - pd.Timedelta(hours=1), freq="M")
    return sorted(mes.month for mes in meses)


def metadatos_tabla(config: ProjectConfig | None = None) -> dict:
    """Lee el linaje de la tabla y verifica que tenga los cortes de la configuración."""
    config = config or ProjectConfig()
    ruta = config.supervised_table_path.with_suffix(".json")
    if not ruta.is_file() or not config.supervised_table_path.is_file():
        raise FileNotFoundError(
            "No existe la tabla supervisada; genérala con el entrenamiento de P1 (Entrega 2)."
        )
    metadatos = json.loads(ruta.read_text(encoding="utf-8"))
    for clave in ("start_month", "end_month", "train_end", "validation_end"):
        if metadatos.get(clave) != getattr(config, clave):
            raise ValueError(
                f"La tabla supervisada tiene {clave}={metadatos.get(clave)!r} y la configuración "
                f"{getattr(config, clave)!r}; reconstrúyela con los cortes de la v1."
            )
    return metadatos


def cargar_periodo(
    periodo: Periodo,
    config: ProjectConfig | None = None,
    *,
    columnas: list[str] | None = None,
    max_filas: int = 0,
    semilla: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Lee las filas del periodo; con max_filas toma una muestra fija por semilla."""
    config = config or ProjectConfig()
    columnas = columnas or COLUMNAS_P1
    if max_filas < 0:
        raise ValueError("max_filas debe ser 0 (todas) o positivo.")
    # Se eligen posiciones con la columna de hora y se lee por lotes: la tabla completa no cabe
    # cómodamente en memoria.
    archivo = pq.ParquetFile(config.supervised_table_path)
    horas = archivo.read(columns=["hora"]).column("hora").to_numpy()
    posiciones = np.flatnonzero(periodo.contiene(horas))
    if not posiciones.size:
        raise ValueError(f"El periodo {periodo.nombre} no tiene filas en la tabla supervisada.")
    if max_filas and posiciones.size > max_filas:
        elegidas = np.random.default_rng(semilla).choice(posiciones, max_filas, replace=False)
        posiciones = np.sort(elegidas)
    partes, inicio_lote = [], 0
    for lote in archivo.iter_batches(batch_size=_LOTE, columns=columnas):
        fin_lote = inicio_lote + lote.num_rows
        desde, hasta = np.searchsorted(posiciones, [inicio_lote, fin_lote])
        if hasta > desde:
            partes.append(lote.take(pa.array(posiciones[desde:hasta] - inicio_lote)).to_pandas())
        inicio_lote = fin_lote
    return pd.concat(partes, ignore_index=True)


def resumen_periodo(periodo: Periodo, datos: pd.DataFrame) -> dict:
    """Filas, estaciones y horas extremas realmente leídas de un periodo."""
    return {
        **periodo.como_dict(),
        "filas": int(len(datos)),
        "estaciones": int(datos["estacion"].nunique()) if "estacion" in datos else None,
        "primera_hora": pd.Timestamp(datos["hora"].min()).isoformat(),
        "ultima_hora": pd.Timestamp(datos["hora"].max()).isoformat(),
    }
