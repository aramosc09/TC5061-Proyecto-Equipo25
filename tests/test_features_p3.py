import pandas as pd
import pytest

from proyecto_mibici.features_p3 import (
    PROFILE_COLUMNS,
    aggregate_p3_events,
    attach_p3_station_metadata,
    build_p3_station_profiles,
    prepare_p3_trips,
)


def sample_trips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Inicio_del_viaje": [
                "2023-10-02 09:00:00",
                "2023-10-02 09:30:00",
                "2023-10-03 10:00:00",
            ],
            "Fin_del_viaje": [
                "2023-10-02 09:10:00",
                "2023-10-02 09:40:00",
                "2023-10-03 10:10:00",
            ],
            "Origen_Id": ["10", "10", "20"],
            "Destino_Id": ["20", "20", "10"],
            "Usuario_Id": ["must-not-be-copied"] * 3,
        }
    )


def test_prepare_p3_trips_validates_required_fields_and_excludes_invalid_rows():
    trips = pd.concat(
        [
            sample_trips(),
            pd.DataFrame(
                {
                    "Inicio_del_viaje": ["not-a-date", "2023-10-04 10:00:00"],
                    "Fin_del_viaje": ["2023-10-04 10:10:00", "2023-10-04 09:00:00"],
                    "Origen_Id": ["30", "30"],
                    "Destino_Id": ["40", "40"],
                }
            ),
        ],
        ignore_index=True,
    )

    cleaned, audit = prepare_p3_trips(trips)

    assert len(cleaned) == 3
    assert list(cleaned.columns) == ["inicio", "fin", "origen", "destino"]
    assert audit["viajes_entrada"] == 5
    assert audit["fechas_inicio_invalidas"] == 1
    assert audit["fin_antes_de_inicio"] == 1
    assert audit["viajes_validos"] == 3

    with pytest.raises(ValueError, match="Faltan columnas"):
        prepare_p3_trips(pd.DataFrame({"Origen_Id": ["10"]}))


def test_aggregate_p3_events_assigns_each_direction_by_its_own_timestamp():
    trips = pd.DataFrame(
        {
            "Inicio_del_viaje": ["2023-10-31 23:55:00"],
            "Fin_del_viaje": ["2023-11-01 00:05:00"],
            "Origen_Id": ["10"],
            "Destino_Id": ["20"],
        }
    )

    events, audit = aggregate_p3_events(
        trips,
        start="2023-10-01",
        end="2023-11-01",
    )

    assert events[["estacion", "direccion", "viajes"]].to_dict("records") == [
        {"estacion": "10", "direccion": "salidas", "viajes": 1}
    ]
    assert audit["salidas_fuera_de_ventana"] == 0
    assert audit["llegadas_fuera_de_ventana"] == 1


def test_station_profiles_are_normalized_and_keep_descriptors_separate():
    events, _ = aggregate_p3_events(
        sample_trips(),
        start="2023-10-01",
        end="2023-11-01",
    )

    profiles, descriptors = build_p3_station_profiles(
        events,
        start="2023-10-01",
        end="2023-11-01",
    )

    assert profiles.shape == (2, 336)
    assert tuple(profiles.columns) == PROFILE_COLUMNS
    assert profiles.columns[0] == "salidas_lunes_h00"
    assert profiles.columns[-1] == "llegadas_domingo_h23"
    assert profiles.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert descriptors.loc["10", "salidas_totales"] == 2
    assert descriptors.loc["10", "llegadas_totales"] == 1
    assert descriptors.loc["10", "eventos_totales"] == 3
    assert descriptors.loc["10", "meses_con_actividad"] == 1


def test_profiles_correct_for_unequal_weekday_occurrences():
    days = pd.date_range("2023-10-01", "2023-10-31", freq="D")
    occurrences = pd.Series(days.dayofweek).value_counts()
    event_counts = pd.DataFrame(
        [
            {
                "estacion": "10",
                "mes": "2023-10",
                "direccion": direction,
                "dia_semana": weekday,
                "hora": hour,
                "viajes": int(occurrences[weekday]),
            }
            for direction in ("salidas", "llegadas")
            for weekday in range(7)
            for hour in range(24)
        ]
    )

    profiles, _ = build_p3_station_profiles(
        event_counts,
        start="2023-10-01",
        end="2023-11-01",
    )

    assert profiles.loc["10"].to_numpy() == pytest.approx([1 / 336] * 336)


def test_station_metadata_is_joined_for_interpretation_not_as_features():
    events, _ = aggregate_p3_events(
        sample_trips(),
        start="2023-10-01",
        end="2023-11-01",
    )
    profiles, descriptors = build_p3_station_profiles(
        events,
        start="2023-10-01",
        end="2023-11-01",
    )
    catalog = pd.DataFrame(
        {
            "id": ["10"],
            "name": ["Station 10"],
            "latitude": [20.1],
            "longitude": [-103.2],
            "dpcapacity": [15],
        }
    )

    station_table = attach_p3_station_metadata(profiles, descriptors, catalog)

    station_10 = station_table.set_index("estacion").loc["10"]
    station_20 = station_table.set_index("estacion").loc["20"]
    assert station_10["name"] == "Station 10"
    assert station_10["latitude"] == 20.1
    assert pd.isna(station_20["name"])
    assert not {"latitude", "longitude"}.intersection(profiles.columns)
