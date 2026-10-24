"""La tabla supervisada debe ser causal, reproducible y coherente con los baselines."""

import numpy as np
import pandas as pd

from proyecto_mibici.features import construir_features_presion
from proyecto_mibici.modeling.datasets import (
    P1_FEATURES,
    P2_FEATURES,
    DatosSupervisados,
    TablaSupervisada,
    construir_perfiles,
    historial_causal,
)
from proyecto_mibici.modeling.supervised_baselines import PipelineP2


def prepared(config, repository):
    table = TablaSupervisada(config)
    return table, table.prepare(repository, "synthetic-v1")


def test_rows_respect_time_splits_and_exclude_incomplete_history(
    candidate_config, synthetic_repository
):
    table, metadata = prepared(candidate_config, synthetic_repository(candidate_config))
    frame = pd.read_parquet(table.path)
    assert set(frame.conjunto) == {0, 1, 2}
    assert not frame[P2_FEATURES].isna().any().any()
    # El rezago de 4 semanas exige 672 h de historial.
    assert frame.hora.min() == candidate_config.start + pd.Timedelta(hours=672)
    for code, (start, end) in enumerate(candidate_config.splits.values()):
        hours = frame.loc[frame.conjunto == code, "hora"]
        assert hours.min() >= start and hours.max() < end
    assert sum(metadata["rows"].values()) == len(frame)


def test_future_changes_do_not_alter_training_rows_or_thresholds(
    candidate_config, synthetic_repository, tmp_path
):
    _, first = prepared(candidate_config, synthetic_repository(candidate_config))
    data = DatosSupervisados(TablaSupervisada(candidate_config), first)
    train = data.split("p2", "train")
    changed_config = candidate_config.__class__(
        **{**candidate_config.__dict__, "root": tmp_path / "changed"}
    )
    _, changed = prepared(
        changed_config, synthetic_repository(changed_config, change_future=True)
    )
    changed_train = DatosSupervisados(TablaSupervisada(changed_config), changed).split(
        "p2", "train"
    )
    pd.testing.assert_frame_equal(train.X, changed_train.X)
    np.testing.assert_array_equal(train.y, changed_train.y)
    for key in ("pressure_q75", "pressure_q90", "p2_majority_class"):
        assert first[key] == changed[key]


def test_p2_thresholds_and_majority_match_the_registered_baseline(
    candidate_config, synthetic_repository, tmp_path
):
    repository = synthetic_repository(candidate_config, missing_capacity_station=3)
    _, metadata = prepared(candidate_config, repository)
    baseline = PipelineP2(candidate_config).run(repository, tmp_path / "p2")
    assert metadata["pressure_q75"] == baseline.parameters["pressure_q75"]
    assert metadata["pressure_q90"] == baseline.parameters["pressure_q90"]
    data = DatosSupervisados(TablaSupervisada(candidate_config), metadata)
    p1 = data.split("p1", "validation")
    p2 = data.split("p2", "validation")
    # P1 conserva estaciones sin capacidad; P2 las excluye como su baseline.
    assert len(p2) < len(p1)
    assert 3 in set(p1.estacion) and 3 not in set(p2.estacion)


def test_p1_baseline_on_same_rows_is_weekly_persistence(candidate_config, synthetic_repository):
    _, metadata = prepared(candidate_config, synthetic_repository(candidate_config))
    subset = DatosSupervisados(TablaSupervisada(candidate_config), metadata).split(
        "p1", "validation"
    )
    assert list(subset.X.columns) == P1_FEATURES
    np.testing.assert_array_equal(subset.baseline, subset.X["salidas_lag_168"].to_numpy())


def test_seeded_sample_is_reproducible_and_bounded(candidate_config, synthetic_repository):
    table, metadata = prepared(candidate_config, synthetic_repository(candidate_config))
    first = table.load("p1", "train", metadata, max_rows=100, seed=42)
    second = table.load("p1", "train", metadata, max_rows=100, seed=42)
    other = table.load("p1", "train", metadata, max_rows=100, seed=43)
    assert len(first) == 100
    pd.testing.assert_frame_equal(first.X, second.X)
    assert not first.X.equals(other.X)


def test_p3_profiles_cover_every_station_and_period(candidate_config, synthetic_repository):
    frames, coverage = construir_perfiles(synthetic_repository(candidate_config), candidate_config)
    assert set(frames) == {"train", "validation", "test"}
    assert all(len(frame) == 6 for frame in frames.values())
    assert all(np.isclose(frame.sum(axis=1), 1).all() for frame in frames.values())
    assert len(coverage) == 18


def test_weekly_and_pressure_features_are_causal_and_scaled():
    timeline = pd.date_range("2024-01-01", periods=900, freq="h")
    hours = pd.DataFrame(
        {"salidas": np.arange(900) % 5, "llegadas": 1, "elegible": True}, index=timeline
    )
    first = historial_causal(hours)
    hours.loc[timeline[700] :, "salidas"] = 999
    changed = historial_causal(hours)
    pd.testing.assert_frame_equal(first.loc[: timeline[700]], changed.loc[: timeline[700]])
    row = first.iloc[800]
    expected = np.mean([hours.salidas.iloc[800 - lag] for lag in (168, 336, 504, 672)])
    assert row.salidas_media_4_semanas == expected
    assert row.salidas_lag_672 == hours.salidas.iloc[128]
    assert pd.isna(first.iloc[671].salidas_media_4_semanas)
    pressure = construir_features_presion(first, 10.0)
    assert pressure.iloc[800].neto_lag_1_capacidad == (row.salidas_lag_1 - row.llegadas_lag_1) / 10
    assert construir_features_presion(first, np.nan).isna().all().all()
