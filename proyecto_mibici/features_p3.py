"""Feature engineering for the P3 station-use clustering task."""

from __future__ import annotations

import pandas as pd

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_COLUMNS = ("Inicio_del_viaje", "Fin_del_viaje")
STATION_COLUMNS = ("Origen_Id", "Destino_Id")
EVENT_COLUMNS = ("estacion", "mes", "direccion", "dia_semana", "hora", "viajes")
DIRECTIONS = ("salidas", "llegadas")
WEEKDAY_NAMES = (
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
    "sabado",
    "domingo",
)
WEEK_BINS = pd.MultiIndex.from_product(
    [DIRECTIONS, range(7), range(24)],
    names=["direccion", "dia_semana", "hora"],
)
PROFILE_COLUMNS = tuple(
    f"{direction}_{WEEKDAY_NAMES[day]}_h{hour:02d}"
    for direction, day, hour in WEEK_BINS
)


def prepare_p3_trips(trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Parse and validate trip columns needed for P3, returning rows and audit counts.

    The returned trips have normalized internal column names and datetime columns.
    User identifiers, demographics, and trip identifiers are neither required nor
    copied into the result.
    """
    required_columns = (*DATE_COLUMNS, *STATION_COLUMNS)
    missing_columns = sorted(set(required_columns) - set(trips.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas para P3: {missing_columns}")

    cleaned = trips.loc[:, required_columns].copy()
    cleaned.columns = ("inicio", "fin", "origen", "destino")
    audit: dict[str, int] = {"viajes_entrada": len(cleaned)}

    for date_column in ("inicio", "fin"):
        cleaned[date_column] = pd.to_datetime(
            cleaned[date_column],
            format=DATE_FORMAT,
            errors="coerce",
        )
        audit[f"fechas_{date_column}_invalidas"] = int(cleaned[date_column].isna().sum())

    for station_column in ("origen", "destino"):
        cleaned[station_column] = cleaned[station_column].astype("string").str.strip()
    audit["ids_estacion_vacios"] = int(
        (
            cleaned["origen"].isna()
            | cleaned["origen"].eq("")
            | cleaned["destino"].isna()
            | cleaned["destino"].eq("")
        ).sum()
    )

    negative_duration = cleaned["fin"].lt(cleaned["inicio"])
    audit["fin_antes_de_inicio"] = int(negative_duration.sum())
    valid_trip = (
        cleaned["inicio"].notna()
        & cleaned["fin"].notna()
        & ~negative_duration
        & cleaned["origen"].notna()
        & cleaned["origen"].ne("")
        & cleaned["destino"].notna()
        & cleaned["destino"].ne("")
    )
    cleaned = cleaned.loc[valid_trip].reset_index(drop=True)
    audit["viajes_validos"] = len(cleaned)
    return cleaned, pd.Series(audit, name="registros", dtype="int64")


def aggregate_p3_events(
    trips: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aggregate station departures and arrivals by month and weekly time slot.

    ``start`` is inclusive and ``end`` exclusive. Each event is assigned using its
    own timestamp, so a trip's departure and arrival can fall in different periods.
    """
    start_timestamp, end_timestamp = _validate_month_bounds(start, end)
    cleaned, audit = prepare_p3_trips(trips)
    event_frames: list[pd.DataFrame] = []

    for direction, time_column, station_column in (
        ("salidas", "inicio", "origen"),
        ("llegadas", "fin", "destino"),
    ):
        in_period = cleaned[time_column].ge(start_timestamp) & cleaned[time_column].lt(
            end_timestamp
        )
        audit[f"{direction}_fuera_de_ventana"] = int((~in_period).sum())
        events = cleaned.loc[in_period, [station_column, time_column]].rename(
            columns={station_column: "estacion", time_column: "fecha_hora"}
        )
        if events.empty:
            continue
        events = events.assign(
            mes=events["fecha_hora"].dt.strftime("%Y-%m"),
            direccion=direction,
            dia_semana=events["fecha_hora"].dt.dayofweek,
            hora=events["fecha_hora"].dt.hour,
        )
        event_counts = (
            events.groupby(
                ["estacion", "mes", "direccion", "dia_semana", "hora"],
                observed=True,
                as_index=False,
            )
            .size()
            .rename(columns={"size": "viajes"})
        )
        event_frames.append(event_counts)

    if event_frames:
        counts = pd.concat(event_frames, ignore_index=True)
    else:
        counts = pd.DataFrame(columns=EVENT_COLUMNS)
    counts = counts.reindex(columns=EVENT_COLUMNS)
    return counts, audit


def build_p3_station_profiles(
    event_counts: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build interpretable weekly station profiles from aggregated P3 events.

    Counts are first converted to an average event rate per occurrence of each
    weekday-hour slot, correcting for unequal numbers of Mondays, Tuesdays, etc.
    in the selected calendar range. The 336 rates are then normalized per station
    to sum to one, representing each station's relative weekly use pattern rather
    than its total volume. A station with no events has no profile and is omitted.

    Returns a feature matrix indexed by station ID and a descriptor table with
    event totals and observed-month coverage. Descriptors are not clustering
    features and should be kept separate from the feature matrix.
    """
    start_timestamp, end_timestamp = _validate_month_bounds(start, end)
    missing_columns = sorted(set(EVENT_COLUMNS) - set(event_counts.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas de eventos para P3: {missing_columns}")

    selected = event_counts.loc[:, EVENT_COLUMNS].copy()
    selected["estacion"] = selected["estacion"].astype("string").str.strip()
    selected["mes"] = selected["mes"].astype("string")
    selected["direccion"] = selected["direccion"].astype("string")
    selected["viajes"] = pd.to_numeric(selected["viajes"], errors="raise")
    selected["dia_semana"] = pd.to_numeric(selected["dia_semana"], errors="raise")
    selected["hora"] = pd.to_numeric(selected["hora"], errors="raise")

    month_start = start_timestamp.strftime("%Y-%m")
    month_end = (end_timestamp - pd.Timedelta(days=1)).strftime("%Y-%m")
    selected = selected.loc[
        selected["mes"].between(month_start, month_end)
        & selected["estacion"].notna()
        & selected["estacion"].ne("")
    ]
    if selected.empty:
        raise ValueError("No hay eventos de estación dentro del periodo solicitado.")
    if not selected["direccion"].isin(DIRECTIONS).all():
        raise ValueError(f"Las direcciones deben pertenecer a {DIRECTIONS}.")
    if not selected["dia_semana"].between(0, 6).all():
        raise ValueError("dia_semana debe estar en el rango 0..6.")
    if not selected["hora"].between(0, 23).all():
        raise ValueError("hora debe estar en el rango 0..23.")
    if selected["viajes"].lt(0).any():
        raise ValueError("Los conteos de viajes no pueden ser negativos.")

    slot_counts = (
        selected.groupby(
            ["estacion", "direccion", "dia_semana", "hora"],
            observed=True,
        )["viajes"]
        .sum()
        .unstack(["direccion", "dia_semana", "hora"], fill_value=0)
        .reindex(columns=WEEK_BINS, fill_value=0)
    )
    slot_counts.columns = PROFILE_COLUMNS

    exposure = _count_weekly_slot_occurrences(start_timestamp, end_timestamp)
    rate_columns = [
        PROFILE_COLUMNS[index]
        for index, (_, day, hour) in enumerate(WEEK_BINS)
    ]
    exposure_per_feature = pd.Series(
        [exposure[(day, hour)] for _, day, hour in WEEK_BINS],
        index=rate_columns,
        dtype="float64",
    )
    rates = slot_counts.div(exposure_per_feature, axis="columns")
    rate_totals = rates.sum(axis=1)
    active = rate_totals.gt(0)
    profiles = rates.loc[active].div(rate_totals.loc[active], axis="index")
    profiles.index = profiles.index.astype("string")
    profiles.index.name = "estacion"
    profiles = profiles.astype("float64")

    selected["mes"] = pd.PeriodIndex(selected["mes"], freq="M").astype(str)
    coverage = selected.groupby("estacion", observed=True)["mes"].nunique()
    total_by_direction = selected.pivot_table(
        index="estacion",
        columns="direccion",
        values="viajes",
        aggfunc="sum",
        fill_value=0,
        observed=True,
    ).reindex(columns=DIRECTIONS, fill_value=0)
    descriptors = total_by_direction.rename(
        columns={"salidas": "salidas_totales", "llegadas": "llegadas_totales"}
    )
    descriptors["eventos_totales"] = descriptors.sum(axis=1)
    descriptors["meses_con_actividad"] = coverage
    descriptors = descriptors.loc[profiles.index].astype(
        {
            "salidas_totales": "int64",
            "llegadas_totales": "int64",
            "eventos_totales": "int64",
            "meses_con_actividad": "int64",
        }
    )
    descriptors.index = descriptors.index.astype("string")
    descriptors.index.name = "estacion"
    return profiles, descriptors


def attach_p3_station_metadata(
    profiles: pd.DataFrame,
    descriptors: pd.DataFrame,
    station_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Attach catalog descriptions and coordinates without adding them to features."""
    required_columns = {"id", "name", "latitude", "longitude", "dpcapacity"}
    missing_columns = sorted(required_columns - set(station_catalog.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas de nomenclatura para P3: {missing_columns}")
    if not profiles.index.is_unique or not descriptors.index.is_unique:
        raise ValueError("Los índices de perfiles y descriptores deben ser únicos.")

    catalog = station_catalog.loc[
        :, ["id", "name", "latitude", "longitude", "dpcapacity"]
    ].copy()
    catalog["id"] = catalog["id"].astype("string").str.strip()
    if catalog["id"].isna().any() or catalog["id"].eq("").any():
        raise ValueError("La nomenclatura contiene IDs de estación vacíos.")
    if catalog["id"].duplicated().any():
        raise ValueError("La nomenclatura contiene IDs de estación duplicados.")

    station_table = profiles.join(descriptors, how="left").reset_index()
    station_table["estacion"] = station_table["estacion"].astype("string")
    return station_table.merge(
        catalog.rename(columns={"id": "estacion"}),
        on="estacion",
        how="left",
        validate="one_to_one",
    )


def _validate_month_bounds(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp.tz is not None or end_timestamp.tz is not None:
        raise ValueError("Las fechas P3 deben ser marcas de tiempo sin zona horaria.")
    if (
        start_timestamp != start_timestamp.normalize()
        or end_timestamp != end_timestamp.normalize()
    ):
        raise ValueError("Los límites del periodo deben ser medianoches.")
    if start_timestamp.day != 1 or end_timestamp.day != 1:
        raise ValueError("El inicio y el fin exclusivo deben coincidir con el primer día del mes.")
    if start_timestamp >= end_timestamp:
        raise ValueError("El inicio del periodo debe ser anterior al fin exclusivo.")
    return start_timestamp, end_timestamp


def _count_weekly_slot_occurrences(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[tuple[int, int], int]:
    hours = pd.date_range(start=start, end=end, inclusive="left", freq="h")
    occurrences = (
        pd.DataFrame({"dia_semana": hours.dayofweek, "hora": hours.hour})
        .groupby(["dia_semana", "hora"])
        .size()
    )
    return {
        (day, hour): int(occurrences.get((day, hour), 0))
        for day in range(7)
        for hour in range(24)
    }
