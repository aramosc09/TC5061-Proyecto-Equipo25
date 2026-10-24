"""Base horaria común y variables causales de las tres preguntas de MiBici."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from loguru import logger
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from proyecto_mibici.config import (
    BALANCE_WINDOWS,
    FEATURE_LAGS,
    INGESTION_CHUNK_SIZE,
    ROLLING_WINDOWS,
    WEEKLY_LAGS,
    ProjectConfig,
)
from proyecto_mibici.dataset import DataManifest, calcular_sha256

FEATURE_VERSION = "mibici-horas-v1"
HOUR_SCHEMA = pa.schema(
    [
        ("estacion", pa.int64()),
        ("hora", pa.timestamp("ns")),
        ("salidas", pa.int64()),
        ("llegadas", pa.int64()),
    ]
)


def agregar_eventos_horarios(trips: pd.DataFrame, start, end) -> pd.DataFrame:
    """Cuenta salidas por inicio y llegadas por fin dentro de [start, end).

    Cada evento pertenece a su propia hora; una llegada no se asigna al mes de
    inicio del viaje. Se leen exclusivamente columnas operativas de la ingesta.
    """
    events = []
    for direction, station, timestamp in (
        ("salidas", "origen_id", "inicio_viaje"),
        ("llegadas", "destino_id", "fin_viaje"),
    ):
        selected = trips.loc[trips[timestamp].ge(start) & trips[timestamp].lt(end)]
        counts = (
            selected.groupby(
                [
                    selected[station].rename("estacion"),
                    selected[timestamp].dt.floor("h").rename("hora"),
                ]
            )
            .size()
            .rename(direction)
        )
        events.append(counts)
    combined = pd.concat(events, axis=1).fillna(0).astype("int64").reset_index()
    return combined.loc[:, HOUR_SCHEMA.names]


DAYS = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")
PROFILE_COLUMNS = [
    f"{direction}_{day}_h{hour:02d}"
    for direction in ("salidas", "llegadas")
    for day in DAYS
    for hour in range(24)
]
WEEKEND_MASK = np.tile(np.repeat(np.arange(7) >= 5, 24), 2)


def construir_features_calendario(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Codifica el calendario conocido para P1 y P2 sin consultar viajes.

    Los pares seno/coseno preservan la continuidad entre el final e inicio de
    hora y semana; fin de semana y mes representan rutinas previsibles al
    momento de predecir la siguiente hora.
    """
    features = pd.DataFrame(index=index)
    for name, values, period in (
        ("hora", index.hour, 24),
        ("dia_semana", index.dayofweek, 7),
    ):
        features[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        features[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    features["fin_semana"] = index.dayofweek >= 5
    features["mes"] = index.month
    return features


def construir_features_flujo_historico(hours: pd.DataFrame) -> pd.DataFrame:
    """Resume flujos anteriores para P1 y P2 sin tratar historia desconocida como cero.

    Los rezagos captan persistencia reciente y semanal; las medias suavizan la
    variabilidad y los balances describen dirección neta de flujo para el
    proxy de P2. Todas las columnas se desplazan al menos una hora y exigen
    cobertura elegible previa, por lo que no incorporan la hora objetivo ni el
    futuro. El balance no representa inventario de bicicletas.
    
    """
    coverage = hours.get("elegible", pd.Series(True, index=hours.index)).astype(bool)
    features = pd.DataFrame(index=hours.index)
    for direction in ("salidas", "llegadas"):
        for lag in FEATURE_LAGS:
            features[f"{direction}_lag_{lag}"] = (
                hours[direction].shift(lag).where(coverage.shift(lag, fill_value=False))
            )
        previous = hours[direction].shift(1).where(coverage.shift(1, fill_value=False))
        for window in ROLLING_WINDOWS:
            features[f"{direction}_media_{window}"] = previous.rolling(window).mean()
    balance = (hours.salidas - hours.llegadas).shift(1).where(coverage.shift(1, fill_value=False))
    for window in BALANCE_WINDOWS:
        features[f"balance_acumulado_{window}"] = balance.rolling(window).sum()
    return features


def construir_variables_horarias(hours: pd.DataFrame) -> pd.DataFrame:
    """Une las variables de calendario e historial causal para P1 y P2.

    La fila indexada por una hora contiene solo información disponible antes de
    esa hora. Así puede alinearse con el objetivo de la siguiente hora sin fuga
    temporal; los nulos señalan historial insuficiente y deben excluirse, no
    imputarse como ceros.
    """
    return pd.concat(
        [construir_features_calendario(hours.index), construir_features_flujo_historico(hours)],
        axis=1,
    )


def construir_features_semanales(hours: pd.DataFrame) -> pd.DataFrame:
    """Agrega memoria de 2 a 4 semanas para los candidatos de P1 y P2.

    Un solo rezago de 168 h es ruidoso. La media de las cuatro semanas previas
    a la misma hora estima el nivel típico de la estación con menos varianza.
    Cada rezago exige cobertura elegible en su propia hora y la media no omite
    nulos, así que historia desconocida no se trata como cero ni entra futuro.
    """
    coverage = hours.get("elegible", pd.Series(True, index=hours.index)).astype(bool)
    features = pd.DataFrame(index=hours.index)
    for direction in ("salidas", "llegadas"):
        lags = {
            lag: hours[direction].shift(lag).where(coverage.shift(lag, fill_value=False))
            for lag in (FEATURE_LAGS[-1], *WEEKLY_LAGS)
        }
        for lag in WEEKLY_LAGS:
            features[f"{direction}_lag_{lag}"] = lags[lag]
        features[f"{direction}_media_4_semanas"] = pd.concat(lags.values(), axis=1).mean(
            axis=1, skipna=False
        )
    return features


def construir_features_presion(features: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """Expresa flujos netos pasados en la escala de la etiqueta de P2.

    La clase de P2 es flujo neto entre capacidad; los árboles no dividen
    columnas entre sí, por lo que se les entrega la razón ya calculada. Solo
    usa columnas causales previas; sin capacidad válida el resultado es nulo.
    """
    net = {
        "balance_3": features.balance_acumulado_3,
        "balance_24": features.balance_acumulado_24,
        "neto_lag_1": features.salidas_lag_1 - features.llegadas_lag_1,
        "neto_lag_24": features.salidas_lag_24 - features.llegadas_lag_24,
        "neto_lag_168": features.salidas_lag_168 - features.llegadas_lag_168,
        "neto_media_4_semanas": features.salidas_media_4_semanas
        - features.llegadas_media_4_semanas,
    }
    return pd.DataFrame({f"{name}_capacidad": values / capacity for name, values in net.items()})


def construir_perfil_semanal(hours: pd.DataFrame) -> tuple[pd.Series | None, dict]:
    """Construye el perfil P3 de 336 tasas semanales con exposición observada.

    Para cada franja día-hora divide salidas y llegadas entre horas elegibles y
    normaliza el vector conjunto a uno. La corrección por exposición evita que
    ausencias de cobertura parezcan poco uso; una franja sin exposición queda
    como desconocida y excluye la estación, en vez de imputarse como cero.
    P3 es descriptiva: usa solo la ventana temporal que se está perfilando.
    """
    eligible = hours.loc[hours.elegible]
    slots = eligible.index.dayofweek * 24 + eligible.index.hour
    exposure = np.bincount(slots, minlength=168)
    audit = {
        "horas_elegibles": len(eligible),
        "franjas_cubiertas": int((exposure > 0).sum()),
        "eventos": int(eligible[["salidas", "llegadas"]].to_numpy().sum()),
    }
    if not (exposure > 0).all():
        return None, {**audit, "estado": "cobertura_semanal_incompleta"}
    if audit["eventos"] == 0:
        return None, {**audit, "estado": "sin_actividad"}
    rates = np.concatenate(
        [
            np.bincount(slots, weights=eligible[direction], minlength=168) / exposure
            for direction in ("salidas", "llegadas")
        ]
    )
    return pd.Series(rates / rates.sum(), index=PROFILE_COLUMNS), {**audit, "estado": "incluida"}


def proporcion_fin_semana(profiles: pd.DataFrame) -> pd.Series:
    """Calcula la proporción P3 de actividad relativa en sábado y domingo.

    Resume perfiles ya normalizados para el baseline descriptivo. No usa
    coordenadas, capacidad ni volumen bruto, por lo que compara la forma
    temporal del uso entre estaciones.
    """
    return profiles.loc[:, np.asarray(PROFILE_COLUMNS)[WEEKEND_MASK]].sum(axis=1)


class RepositorioParquet:
    """Verifica la base versionada y ofrece series por estación sin cargarla completa."""

    def __init__(self, config: ProjectConfig):
        config.validate()
        self.config = config

    def verify_base(self) -> dict:
        """Exige que manifiesto, rango y ambos Parquet correspondan a la base común."""
        manager = DataManifest(self.config.manifest_path)
        manifest = manager.cargar()
        if not manifest.get("excluye_datos_personales"):
            raise ValueError("La base común debe excluir atributos personales.")
        if manifest["rango_fechas"] != {
            "inicio": self.config.start_month,
            "fin": self.config.end_month,
        }:
            raise ValueError("El rango del manifiesto no coincide con el experimento.")
        for key, expected in (
            ("salida", self.config.trips_path),
            ("salida_estaciones", self.config.stations_path),
        ):
            actual = Path(self.config.root) / manifest[key]["ruta"]
            if actual.resolve() != expected.resolve():
                raise ValueError(f"La ruta {key} no coincide con el contrato común.")
        errors = manager.verificar()
        if errors:
            raise ValueError("Base no verificable: " + "; ".join(errors))
        return manifest

    def prepare(self, *, rebuild: bool = False) -> dict:
        """Agrega eventos a disco con suma por lotes y cache vinculado al hash de entrada."""
        manifest = self.verify_base()
        metadata_path = self.config.hours_path.with_suffix(".json")
        signature = {
            "feature_version": FEATURE_VERSION,
            "data_version": manifest["data_version"],
            "source_sha256": manifest["salida"]["sha256"],
            "start_month": self.config.start_month,
            "end_month": self.config.end_month,
        }
        if not rebuild and metadata_path.is_file() and self.config.hours_path.is_file():
            cached = json.loads(metadata_path.read_text())
            if all(cached.get(key) == value for key, value in signature.items()):
                if cached.get("sha256") == calcular_sha256(self.config.hours_path):
                    return cached
        output = self.config.hours_path
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".horas-", dir=output.parent) as directory:
            temporary = Path(directory)
            # closing() libera el archivo; Windows no permite borrar el temporal con la conexión abierta.
            with closing(sqlite3.connect(temporary / "horas.sqlite")) as connection:
                connection.execute(
                    "CREATE TABLE horas (estacion INTEGER, hora INTEGER, salidas INTEGER, "
                    "llegadas INTEGER, PRIMARY KEY(estacion,hora)) WITHOUT ROWID"
                )
                scanner = ds.dataset(self.config.trips_path, format="parquet").scanner(
                    columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"],
                    batch_size=INGESTION_CHUNK_SIZE,
                )
                for batch in scanner.to_batches():
                    counts = agregar_eventos_horarios(
                        batch.to_pandas(), self.config.start, self.config.end
                    )
                    rows = zip(
                        counts.estacion.tolist(),
                        counts.hora.astype("int64").tolist(),
                        counts.salidas.tolist(),
                        counts.llegadas.tolist(),
                        strict=True,
                    )
                    connection.executemany(
                        "INSERT INTO horas VALUES (?,?,?,?) ON CONFLICT(estacion,hora) "
                        "DO UPDATE SET salidas=salidas+excluded.salidas, "
                        "llegadas=llegadas+excluded.llegadas",
                        rows,
                    )
                    connection.commit()
                temporary_parquet = temporary / "horas.parquet"
                with pq.ParquetWriter(
                    temporary_parquet, HOUR_SCHEMA, compression="snappy"
                ) as writer:
                    for frame in pd.read_sql_query(
                        "SELECT * FROM horas ORDER BY estacion,hora",
                        connection,
                        chunksize=INGESTION_CHUNK_SIZE,
                    ):
                        frame["hora"] = pd.to_datetime(frame.hora, unit="ns")
                        writer.write_table(
                            pa.Table.from_pandas(frame, schema=HOUR_SCHEMA, preserve_index=False)
                        )
            signature["sha256"] = calcular_sha256(temporary_parquet)
            signature["n_horas_observadas"] = pq.ParquetFile(temporary_parquet).metadata.num_rows
            os.replace(temporary_parquet, output)
            metadata_path.write_text(json.dumps(signature, indent=2) + "\n")
        logger.info("Base horaria común preparada: {}", output)
        return signature

    def stations(self) -> pd.DataFrame:
        """Lee metadatos públicos; la capacidad no representa inventario histórico."""
        return pd.read_parquet(self.config.stations_path)

    def _complete(self, sparse: pd.DataFrame) -> pd.DataFrame:
        timeline = pd.date_range(self.config.start, self.config.end, freq="h", inclusive="left")
        hours = sparse.set_index("hora")[["salidas", "llegadas"]].reindex(timeline, fill_value=0)
        activity = hours.salidas + hours.llegadas
        hours["elegible"] = (
            activity.shift(1)
            .rolling(self.config.recent_hours, min_periods=self.config.recent_hours)
            .sum()
            .gt(0)
        )
        return hours

    def station_hours(self):
        """Genera horas completas y una máscara basada solo en actividad pasada.

        Se asume cobertura de los archivos mensuales verificados. Una hora es
        elegible tras una ventana completa de historial y al menos un evento en
        las horas previas; no se usa la primera/última aparición futura ni el
        catálogo final para inferir actividad. Un cero es analítico, no prueba de
        disponibilidad física. Las ausencias prolongadas dejan de ser elegibles.
        """
        pending = []
        previous = None
        for batch in pq.ParquetFile(self.config.hours_path).iter_batches(
            batch_size=INGESTION_CHUNK_SIZE
        ):
            frame = batch.to_pandas()
            for station, group in frame.groupby("estacion", sort=False):
                if previous is not None and station != previous:
                    yield int(previous), self._complete(pd.concat(pending, ignore_index=True))
                    pending = []
                pending.append(group)
                previous = station
        if pending:
            yield int(previous), self._complete(pd.concat(pending, ignore_index=True))
