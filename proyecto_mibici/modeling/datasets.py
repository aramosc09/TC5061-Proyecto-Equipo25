"""Conjuntos causales X/y de P1 y P2 y perfiles semanales de P3 sobre la partición común."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from loguru import logger
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from proyecto_mibici.config import FEATURE_LAGS, ProjectConfig
from proyecto_mibici.dataset import calcular_sha256
from proyecto_mibici.features import (
    FEATURE_VERSION,
    PROFILE_COLUMNS,
    construir_features_presion,
    construir_features_semanales,
    construir_perfil_semanal,
    construir_variables_horarias,
)
from proyecto_mibici.modeling.supervised_baselines import pressure_labels

TABLE_VERSION = "mibici-supervisada-v2"
SPLITS = ("train", "validation", "test")


def historial_causal(hours: pd.DataFrame) -> pd.DataFrame:
    """Variables horarias de la Entrega 1 más la memoria de 2 a 4 semanas."""
    return pd.concat([construir_variables_horarias(hours), construir_features_semanales(hours)], axis=1)


_EMPTY_HOURS = pd.DataFrame({"salidas": [], "llegadas": []}, index=pd.DatetimeIndex([]))
HOURLY_FEATURES = list(construir_variables_horarias(_EMPTY_HOURS).columns)
WEEKLY_FEATURES = list(construir_features_semanales(_EMPTY_HOURS).columns)
PRESSURE_FEATURES = list(
    construir_features_presion(historial_causal(_EMPTY_HOURS), 1.0).columns
)
P1_FEATURES = [*HOURLY_FEATURES, *WEEKLY_FEATURES]
# Capacidad del catálogo 2026-09, la misma que define las clases de P2.
P2_FEATURES = [*HOURLY_FEATURES, "capacidad", *PRESSURE_FEATURES]
TABLE_SCHEMA = pa.schema(
    [
        ("estacion", pa.int64()),
        ("hora", pa.timestamp("ns")),
        ("conjunto", pa.int8()),
        *[(name, pa.float32()) for name in P1_FEATURES],
        ("capacidad", pa.float32()),
        *[(name, pa.float32()) for name in PRESSURE_FEATURES],
        ("salidas", pa.float32()),
        ("presion", pa.float32()),
    ]
)


@dataclass
class Conjunto:
    """Filas de un periodo con su objetivo y la predicción del baseline en esas mismas filas."""

    X: pd.DataFrame
    y: np.ndarray
    baseline: np.ndarray
    estacion: np.ndarray
    hora: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def split_codes(index: pd.DatetimeIndex, config: ProjectConfig) -> np.ndarray:
    """Asigna 0/1/2 a train/validación/prueba y -1 fuera de la ventana."""
    codes = np.full(len(index), -1, dtype="int8")
    for code, name in enumerate(SPLITS):
        start, end = config.splits[name]
        codes[(index >= start) & (index < end)] = code
    return codes


class TablaSupervisada:
    """Materializa una vez las variables causales estación-hora con caché ligado a los datos."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.path = config.supervised_table_path
        self.metadata_path = self.path.with_suffix(".json")

    def signature(self, data_version: str) -> dict:
        return {
            "table_version": TABLE_VERSION,
            "feature_version": FEATURE_VERSION,
            "data_version": data_version,
            "start_month": self.config.start_month,
            "end_month": self.config.end_month,
            "train_end": self.config.train_end,
            "validation_end": self.config.validation_end,
            "recent_hours": self.config.recent_hours,
            "pressure_quantiles": list(self.config.pressure_quantiles),
        }

    def prepare(self, repository, data_version: str, *, rebuild: bool = False) -> dict:
        """Construye la tabla si no existe una versión idéntica verificada por hash."""
        signature = self.signature(data_version)
        if not rebuild and self.metadata_path.is_file() and self.path.is_file():
            cached = json.loads(self.metadata_path.read_text())
            if all(cached.get(key) == value for key, value in signature.items()):
                if cached.get("sha256") == calcular_sha256(self.path):
                    return cached
        metadata = {**signature, **self._build(repository)}
        self.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        logger.info("Tabla supervisada preparada: {}", self.path)
        return metadata

    def _build(self, repository) -> dict:
        """Recorre estaciones una vez; las filas se excluyen, no se imputan, sin historial."""
        capacities = repository.stations().set_index("id")["dpcapacity"]
        train_pressure = []
        rows = dict.fromkeys(SPLITS, 0)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tabla-", dir=self.path.parent) as directory:
            temporary = Path(directory) / "tabla.parquet"
            with pq.ParquetWriter(temporary, TABLE_SCHEMA, compression="snappy") as writer:
                for station, hours in repository.station_hours():
                    capacity = capacities.get(station)
                    if capacity is None or pd.isna(capacity) or capacity <= 0:
                        capacity = np.nan
                    codes = split_codes(hours.index, self.config)
                    eligible = hours.elegible.to_numpy(dtype=bool)
                    pressure = ((hours.salidas - hours.llegadas) / capacity).to_numpy(dtype=float)
                    # Mismo universo que el baseline de P2.
                    if not np.isnan(capacity):
                        train_pressure.append(pressure[eligible & (codes == 0)])
                    features = historial_causal(hours)
                    keep = eligible & (codes >= 0) & features.notna().all(axis=1).to_numpy()
                    if not keep.any():
                        continue
                    pressure_features = construir_features_presion(features, capacity)
                    frame = features.loc[keep].astype("float32")
                    frame["capacidad"] = np.float32(capacity)
                    for name in PRESSURE_FEATURES:
                        frame[name] = pressure_features.loc[keep, name].astype("float32")
                    frame.insert(0, "conjunto", codes[keep])
                    frame.insert(0, "hora", hours.index[keep])
                    frame.insert(0, "estacion", station)
                    frame["salidas"] = hours.salidas.to_numpy(dtype="float32")[keep]
                    frame["presion"] = pressure[keep].astype("float32")
                    writer.write_table(
                        pa.Table.from_pandas(frame, schema=TABLE_SCHEMA, preserve_index=False)
                    )
                    for code, name in enumerate(SPLITS):
                        rows[name] += int((codes[keep] == code).sum())
            training = np.concatenate(train_pressure) if train_pressure else np.array([])
            if not training.size or not all(rows.values()):
                raise ValueError("La tabla supervisada requiere filas en los tres periodos.")
            q75, q90 = np.quantile(np.abs(training), self.config.pressure_quantiles)
            majority = int(np.bincount(pressure_labels(training, q75, q90), minlength=7).argmax())
            digest = calcular_sha256(temporary)
            os.replace(temporary, self.path)
        return {
            "sha256": digest,
            "rows": rows,
            "pressure_q75": float(q75),
            "pressure_q90": float(q90),
            "p2_majority_class": majority,
        }

    def load(
        self, question: str, split: str, metadata: dict, *, max_rows: int = 0, seed: int = 0
    ) -> Conjunto:
        """Lee un periodo; la submuestra con semilla solo se aplica cuando se solicita."""
        features = P1_FEATURES if question == "p1" else P2_FEATURES
        table = pq.read_table(
            self.path,
            columns=["estacion", "hora", *features, "salidas", "presion"],
            filters=[("conjunto", "=", SPLITS.index(split))],
        ).to_pandas()
        if question == "p2":
            table = table.loc[table.presion.notna()]
        if max_rows and len(table) > max_rows:
            selected = np.random.default_rng(seed).choice(len(table), max_rows, replace=False)
            table = table.iloc[np.sort(selected)]
        if question == "p1":
            target = table.salidas.to_numpy(dtype=float)
            baseline = table[f"salidas_lag_{FEATURE_LAGS[-1]}"].to_numpy(dtype=float)
        else:
            target = pressure_labels(
                table.presion.to_numpy(dtype=float),
                metadata["pressure_q75"],
                metadata["pressure_q90"],
            )
            baseline = np.full(len(table), metadata["p2_majority_class"], dtype="int8")
        return Conjunto(
            X=table[features].reset_index(drop=True),
            y=target,
            baseline=baseline,
            estacion=table.estacion.to_numpy(),
            hora=table.hora.to_numpy(),
        )


def construir_perfiles(repository, config: ProjectConfig) -> tuple[dict, list]:
    """Perfiles P3 de 336 tasas por periodo, con la misma cobertura que el baseline."""
    profiles = {split: {} for split in SPLITS}
    coverage = []
    for station, hours in repository.station_hours():
        for split in SPLITS:
            start, end = config.splits[split]
            profile, audit = construir_perfil_semanal(
                hours.loc[(hours.index >= start) & (hours.index < end)]
            )
            coverage.append({"estacion": station, "conjunto": split, **audit})
            if profile is not None:
                profiles[split][station] = profile
    frames = {
        split: pd.DataFrame.from_dict(values, orient="index", columns=PROFILE_COLUMNS).rename_axis(
            "estacion"
        )
        for split, values in profiles.items()
    }
    for split, frame in frames.items():
        if frame.empty:
            raise ValueError(f"P3 no tiene perfiles semanales completos en {split}.")
    return frames, coverage


class DatosSupervisados:
    """Acceso de P1/P2 a la tabla; memoriza periodos para reutilizarlos entre trials."""

    def __init__(self, table: TablaSupervisada, metadata: dict):
        self.table = table
        self.metadata = metadata
        self._cache = {}

    def split(self, question: str, name: str, *, max_rows: int = 0, seed: int = 0) -> Conjunto:
        key = (question, name, max_rows, seed)
        if key not in self._cache:
            self._cache[key] = self.table.load(
                question, name, self.metadata, max_rows=max_rows, seed=seed
            )
        return self._cache[key]


@dataclass
class PerfilesP3:
    frames: dict
    coverage: list
