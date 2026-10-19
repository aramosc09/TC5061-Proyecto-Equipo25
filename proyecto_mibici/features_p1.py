"""Features estacion-hora sin fuga temporal y perfiles descriptivos de estacion.

Lee el Parquet unificado que produce `dataset.py` y escribe dos tablas en
`data/processed/`: una supervisada (P1 y P2) y los perfiles semanales (P3).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from proyecto_mibici.config_p1 import load_settings

app = typer.Typer(help="Construye las features de P1/P2 y los perfiles de P3.")

TRIP_COLUMNS = ["viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
STATION_COLUMNS = {
    "id": "station_id",
    "name": "name",
    "latitude": "latitude",
    "longitude": "longitude",
    "dpcapacity": "capacity",
}


def load_trips(settings, trips_path=None):
    """Carga el Parquet unificado y descarta viajes invalidos o fuera de la ventana.

    Invalido es el viaje con identificadores o fechas nulas, duracion no positiva
    o mayor al maximo configurado, o un `viaje_id` repetido.
    """
    path = Path(trips_path) if trips_path else settings.path("data", "trips_parquet")
    frame = pd.read_parquet(path, columns=TRIP_COLUMNS)
    for column in ("inicio_viaje", "fin_viaje"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ("viaje_id", "origen_id", "destino_id"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = numeric.where(numeric % 1 == 0).astype("Int64")
    start = pd.Period(settings.values["data"]["start_month"], freq="M").start_time
    end = (pd.Period(settings.values["data"]["end_month"], freq="M") + 1).start_time
    duration = (frame.fin_viaje - frame.inicio_viaje).dt.total_seconds()
    valid = (
        frame.notna().all(axis=1)
        & (duration > 0)
        & (duration <= settings.values["data"]["max_duration_hours"] * 3600)
        & (frame.inicio_viaje >= start)
        & (frame.inicio_viaje < end)
    )
    frame = frame[valid & ~frame.viaje_id.duplicated()]
    if frame.empty:
        raise ValueError("No quedan viajes validos en la ventana configurada")
    return frame


def load_stations(settings, stations_path=None):
    """Carga el catalogo de estaciones y valida los campos que P2 y P3 necesitan."""
    path = Path(stations_path) if stations_path else settings.path("data", "stations_csv")
    stations = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            stations = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if stations is None:
        raise ValueError(f"Codificacion no soportada: {path.name}")
    stations = stations.rename(columns=STATION_COLUMNS)[list(STATION_COLUMNS.values())]
    for column in ("station_id", "latitude", "longitude", "capacity"):
        stations[column] = pd.to_numeric(stations[column], errors="coerce")
    if stations[["station_id", "latitude", "longitude", "capacity"]].isna().any().any():
        raise ValueError("El catalogo tiene campos requeridos vacios o no numericos")
    if stations.station_id.duplicated().any():
        raise ValueError("Los identificadores de estacion deben ser unicos")
    stations["station_id"] = stations.station_id.astype("int64")
    return stations.set_index("station_id")


def aggregate_events(trips):
    """Reduce los viajes a conteos por estacion y hora; las llegadas usan su hora FIN."""
    parts = []
    for station, time, name in (
        ("origen_id", "inicio_viaje", "departures"),
        ("destino_id", "fin_viaje", "arrivals"),
    ):
        part = (
            trips.groupby(
                [trips[station].rename("station_id"), trips[time].dt.floor("h").rename("hour")]
            )
            .size()
            .rename(name)
        )
        parts.append(part)
    return pd.concat(parts, axis=1).fillna(0).sort_index()


def fit_pressure_threshold(totals, stations, settings, start, train_end):
    """Ajusta el umbral de P2 SOLO con entrenamiento, para no filtrar informacion futura."""
    window = settings.values["features"]["p2_window_hours"]
    warmup = max(
        max(settings.values["features"]["lags"]),
        settings.values["features"]["rolling_hours"],
    )
    samples = []
    for station_id, group in totals.groupby(level="station_id"):
        if station_id not in stations.index or stations.loc[station_id, "capacity"] <= 0:
            continue
        history = group.droplevel("station_id")
        history = history[(history.index >= start) & (history.index < train_end)]
        if history.empty:
            continue
        history = history.reindex(
            pd.date_range(history.index.min(), train_end, freq="h", inclusive="left"),
            fill_value=0,
        )
        values = (
            forward_sum(history.departures - history.arrivals, window)
            / stations.loc[station_id, "capacity"]
        )
        samples.append(values.iloc[warmup:].dropna().abs().to_numpy())
    if not samples or not sum(len(s) for s in samples):
        raise ValueError("No hay observaciones de entrenamiento para ajustar el umbral")
    quantile = settings.values["features"]["pressure_quantile"]
    return max(
        float(np.quantile(np.concatenate(samples), quantile)),
        settings.values["features"]["pressure_minimum"],
    )


def forward_sum(series, window):
    """Aggregate [t,t+window) for LABELS ONLY; require a fully observed window."""
    return series.iloc[::-1].rolling(window, min_periods=window).sum().iloc[::-1]


def partition_limit(index, settings):
    """Exclusive observation boundary for the partition containing each origin."""
    train_end = pd.Timestamp(settings.values["split"]["train_end"])
    validation_end = pd.Timestamp(settings.values["split"]["validation_end"])
    data_end = (pd.Period(settings.values["data"]["end_month"], freq="M") + 1).start_time
    return pd.to_datetime(
        np.where(
            index < train_end,
            train_end.value,
            np.where(index < validation_end, validation_end.value, data_end.value),
        )
    )


def station_features(hourly, station_id, capacity, settings, pressure_threshold):
    """Forecast P1's hour at t+h and P2's window [t,t+w) using only events before t.

    The station becomes eligible after its first observed event and the longest
    lag warmup. No future last-active date is used to remove quiet stations.
    Current-hour counts are targets only, never estimator input columns.
    """
    frame = hourly.copy()
    frame.index.name = "hour"
    frame["station_id"] = int(station_id)
    frame["hour_of_day"] = frame.index.hour
    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    for column in ("departures", "arrivals"):
        for lag in settings.values["features"]["lags"]:
            frame[f"{column}_lag_{lag}"] = frame[column].shift(lag)
        window = settings.values["features"]["rolling_hours"]
        frame[f"{column}_rolling_mean"] = frame[column].shift(1).rolling(window).mean()
    horizon = settings.values["features"]["p1_horizon_hours"]
    risk_window = settings.values["features"]["p2_window_hours"]
    limit = partition_limit(frame.index, settings)
    frame["target_trips"] = frame.departures.shift(-horizon)
    frame["target_hour"] = frame.index + pd.Timedelta(hours=horizon)
    frame.loc[frame.index + pd.Timedelta(hours=horizon + 1) > limit, "target_trips"] = np.nan
    frame["target_day_of_week"] = frame.target_hour.dt.dayofweek
    frame["target_month"] = frame.target_hour.dt.month
    threshold = pressure_threshold
    if pd.notna(capacity) and capacity > 0:
        pressure = forward_sum(frame.departures - frame.arrivals, risk_window) / capacity
        frame["target_pressure"] = np.select(
            [pressure >= threshold, pressure <= -threshold],
            ["bike_pressure", "dock_pressure"],
            default="balanced",
        )
        invalid = pressure.isna() | (frame.index + pd.Timedelta(hours=risk_window) > limit)
        frame.loc[invalid, "target_pressure"] = None
    else:
        frame["target_pressure"] = None
    train_end = pd.Timestamp(settings.values["split"]["train_end"])
    validation_end = pd.Timestamp(settings.values["split"]["validation_end"])
    # Each target interval must be fully contained in its partition.
    frame["split"] = np.select(
        [frame.index < train_end, frame.index < validation_end],
        ["train", "validation"],
        default="test",
    )
    feature_columns = predictor_columns(frame)
    frame = frame.dropna(subset=feature_columns)
    return frame.reset_index()


def predictor_columns(frame):
    """Explicit allowlist prevents labels, snapshot capacity and future counts leaking."""
    return [
        "station_id",
        "hour_of_day",
        "day_of_week",
        "month",
        "target_day_of_week",
        "target_month",
    ] + [c for c in frame.columns if "_lag_" in c or c.endswith("_rolling_mean")]


def station_profile(hourly):
    """P3 descriptive profile: normalized departures/arrivals by hour of week.

    This summarizes only the training window; it is not a future prediction.
    Coordinates are visualization metadata, not clustering input.
    """
    hour_of_week = hourly.index.dayofweek * 24 + hourly.index.hour
    profile = {}
    for column in ("departures", "arrivals"):
        vector = hourly[column].groupby(hour_of_week).mean().reindex(range(168), fill_value=0)
        vector = vector / max(float(vector.sum()), 1.0)
        profile.update({f"{column}_{i:03d}": float(v) for i, v in vector.items()})
    return profile


def build_features(settings, trips_path=None, stations_path=None):
    """Escribe la tabla supervisada (P1/P2) y los perfiles semanales (P3)."""
    trips = load_trips(settings, trips_path)
    stations = load_stations(settings, stations_path)
    totals = aggregate_events(trips)

    start = pd.Period(settings.values["data"]["start_month"], freq="M").start_time
    end = (pd.Period(settings.values["data"]["end_month"], freq="M") + 1).start_time
    train_end = pd.Timestamp(settings.values["split"]["train_end"])
    threshold = fit_pressure_threshold(totals, stations, settings, start, train_end)
    print(f"Umbral de P2 ajustado solo con entrenamiento: {threshold:.6f}", flush=True)

    supervised, profiles = [], []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for station_id, group in totals.groupby(level="station_id"):
        hourly = group.droplevel("station_id")
        hourly = hourly.loc[(hourly.index >= start) & (hourly.index < end)]
        if hourly.empty:
            continue
        # Se parte de la primera actividad observada, no de una fecha de alta posterior.
        hours = pd.date_range(max(start, hourly.index.min()), end, freq="h", inclusive="left")
        hourly = hourly.reindex(hours, fill_value=0)
        metadata = stations.loc[station_id] if station_id in stations.index else None
        capacity = metadata.capacity if metadata is not None else None
        frame = station_features(hourly, station_id, capacity, settings, threshold)
        if not frame.empty:
            supervised.append(frame)
            for split, count in frame.split.value_counts().items():
                split_counts[split] += int(count)
        training_history = hourly[hourly.index < train_end]
        if training_history.empty:
            continue
        profiles.append(
            {
                "station_id": int(station_id),
                **station_profile(training_history),
                "latitude": float(metadata.latitude) if metadata is not None else np.nan,
                "longitude": float(metadata.longitude) if metadata is not None else np.nan,
            }
        )
    if not all(split_counts.values()):
        raise ValueError(f"Particion temporal vacia tras el warmup: {split_counts}")

    supervised_path = settings.path("features", "supervised")
    profiles_path = settings.path("features", "profiles")
    supervised_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(supervised, ignore_index=True).to_parquet(supervised_path, index=False)
    pd.DataFrame(profiles).to_parquet(profiles_path, index=False)
    print(
        f"Features: {sum(split_counts.values()):,} filas {split_counts} | "
        f"perfiles: {len(profiles)} estaciones",
        flush=True,
    )
    meta = {
        "pressure_threshold": threshold,
        "split_rows": split_counts,
        "stations": len(profiles),
        "data_window": [
            settings.values["data"]["start_month"],
            settings.values["data"]["end_month"],
        ],
        "split": settings.values["split"],
        "features": settings.values["features"],
    }
    meta_path = supervised_path.with_name("features_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"supervised": supervised_path, "profiles": profiles_path, **meta}


@app.command()
def construir(
    trips_path: Path = None,
    stations_path: Path = None,
):
    """Construye y guarda las features desde el Parquet unificado."""
    build_features(load_settings(), trips_path, stations_path)


if __name__ == "__main__":
    app()
