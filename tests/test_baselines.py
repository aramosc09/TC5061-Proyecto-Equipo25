"""Pruebas causales de features y de los baselines supervisados reconstruidos."""

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from proyecto_mibici.config import ProjectConfig
from proyecto_mibici.features import (
    RepositorioParquet,
    agregar_eventos_horarios,
    construir_variables_horarias,
)
from proyecto_mibici.modeling.supervised_baselines import (
    PipelineP1,
    PipelineP2,
    classification_metrics,
    pressure_labels,
)


def configuration(tmp_path):
    return ProjectConfig(
        root=tmp_path,
        start_month="2024-01",
        end_month="2024-01",
        train_end="2024-01-15",
        validation_end="2024-01-22",
        recent_hours=2,
        weekly_lag=2,
    )


class FakeRepository:
    def __init__(self, config, *, change_future=False, missing_capacity=False):
        timeline = pd.date_range(config.start, config.end, freq="h", inclusive="left")
        self.hours = pd.DataFrame(
            {
                "salidas": np.arange(len(timeline)) % 2,
                "llegadas": (np.arange(len(timeline)) + 1) % 3,
                "elegible": True,
            },
            index=timeline,
        )
        if change_future:
            self.hours.loc[self.hours.index >= config.train_end, "salidas"] += 100
        self.capacity = np.nan if missing_capacity else 10

    def station_hours(self):
        yield 1, self.hours.copy()
        yield 2, self.hours.copy()

    def stations(self):
        return pd.DataFrame({"id": [1, 2], "dpcapacity": [10, self.capacity]})


def test_hourly_events_use_their_own_timestamp():
    trips = pd.DataFrame(
        {
            "inicio_viaje": pd.to_datetime(["2024-01-31 23:50", "2023-12-31 23:50"]),
            "fin_viaje": pd.to_datetime(["2024-02-01 00:10", "2024-01-01 00:10"]),
            "origen_id": [1, 1],
            "destino_id": [2, 2],
        }
    )
    result = agregar_eventos_horarios(
        trips, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")
    )
    assert result.salidas.sum() == 1
    assert result.llegadas.sum() == 1
    assert set(result.hora) == {pd.Timestamp("2024-01-31 23:00"), pd.Timestamp("2024-01-01")}


def test_features_do_not_see_the_current_or_future_hour():
    timeline = pd.date_range("2024-01-01", periods=400, freq="h")
    frame = pd.DataFrame({"salidas": 1, "llegadas": 2}, index=timeline)
    first = construir_variables_horarias(frame)
    frame.loc[timeline[200] :, "salidas"] = 999
    changed = construir_variables_horarias(frame)
    pd.testing.assert_frame_equal(first.loc[: timeline[200]], changed.loc[: timeline[200]])
    assert first.iloc[168].salidas_lag_168 == 1
    assert first.iloc[168].balance_acumulado_24 == -24


def test_unknown_history_is_not_treated_as_observed_zero_in_features():
    timeline = pd.date_range("2024-01-01", periods=300, freq="h")
    hours = pd.DataFrame({"salidas": 1, "llegadas": 2, "elegible": False}, index=timeline)
    hours.loc[timeline[150] :, "elegible"] = True
    features = construir_variables_horarias(hours)
    assert pd.isna(features.iloc[200].salidas_lag_168)
    assert pd.isna(features.iloc[160].salidas_media_24)
    assert features.iloc[200].salidas_media_24 == 1
    assert features.iloc[200].balance_acumulado_24 == -24


def test_eligibility_uses_past_activity_not_future_presence(tmp_path):
    config = configuration(tmp_path)
    repository = RepositorioParquet(config)
    timestamp = pd.Timestamp("2024-01-10 10:00")
    sparse = pd.DataFrame({"hora": [timestamp], "salidas": [1], "llegadas": [0]})
    hours = repository._complete(sparse)
    assert not hours.loc[:timestamp, "elegible"].any()
    assert hours.loc[timestamp + pd.Timedelta(hours=1), "elegible"]
    assert not hours.loc[timestamp + pd.Timedelta(hours=3), "elegible"]


def test_weekly_persistence_on_periodic_signal_has_zero_error(tmp_path):
    config = configuration(tmp_path)
    result = PipelineP1(config).run(FakeRepository(config), tmp_path / "p1")
    for split in config.splits:
        assert result.metrics[f"{split}_mae"] == 0
        assert result.metrics[f"{split}_rmse"] == 0
        assert result.metrics[f"{split}_mean_error"] == 0
    assert all(path.is_file() for path in result.artifacts)


def test_pressure_class_boundaries_and_ties():
    values = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    assert pressure_labels(values, 1, 2).tolist() == [6, 4, 2, 0, 1, 3, 5]
    assert pressure_labels(np.array([-1, 0, 1]), 0, 0).tolist() == [6, 0, 5]


def test_macro_f1_matches_sklearn_and_undefined_precision_is_omitted():
    truth = np.array([0, 0, 1, 5])
    prediction = np.zeros(4, dtype=int)
    matrix = np.zeros((7, 7), dtype=int)
    np.add.at(matrix, (truth, prediction), 1)
    metrics = classification_metrics(matrix)
    assert metrics["f1_macro"] == f1_score(
        truth, prediction, labels=range(7), average="macro", zero_division=0
    )
    assert metrics["recall_extraccion_alta"] == 0
    assert "precision_extraccion_alta" not in metrics
    assert "recall_acumulacion_alta" not in metrics
    assert metrics["support_acumulacion_alta"] == 0


def test_future_data_do_not_change_p2_training_thresholds_or_majority(tmp_path):
    config = configuration(tmp_path)
    first = PipelineP2(config).run(FakeRepository(config), tmp_path / "first")
    changed = PipelineP2(config).run(
        FakeRepository(config, change_future=True), tmp_path / "changed"
    )
    for key in ("pressure_q75", "pressure_q90", "majority_class"):
        assert first.parameters[key] == changed.parameters[key]
    assert first.metrics["test_n_observations"] > 0
    assert all(path.is_file() for path in first.artifacts)


def test_stations_without_capacity_are_excluded_from_p2_only(tmp_path):
    config = configuration(tmp_path)
    full = PipelineP2(config).run(FakeRepository(config), tmp_path / "full")
    result = PipelineP2(config).run(
        FakeRepository(config, missing_capacity=True), tmp_path / "limited"
    )
    assert result.parameters["excluded_missing_capacity_stations"] == 1
    assert result.metrics["test_n_observations"] * 2 == full.metrics["test_n_observations"]


def test_invalid_time_splits_are_rejected(tmp_path):
    config = replace(configuration(tmp_path), train_end="2024-01-25", validation_end="2024-01-22")
    import pytest

    with pytest.raises(ValueError, match="inicio"):
        config.validate()
