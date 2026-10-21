"""P3 desde cero: exposición, separación temporal y métricas no definidas."""

import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.config import ProjectConfig
from proyecto_mibici.modeling.p3_baseline import (
    PipelineP3,
    perfil_estacion,
    silhouette_euclidea,
    weekend_shares,
)


def configuration(tmp_path):
    return ProjectConfig(
        root=tmp_path,
        start_month="2024-01",
        end_month="2024-01",
        train_end="2024-01-15",
        validation_end="2024-01-22",
    )


class RepositoryDouble:
    def __init__(
        self,
        config,
        *,
        change_future=False,
        degenerate=False,
        missing_validation=False,
        new_station=False,
    ):
        timeline = pd.date_range(config.start, config.end, freq="h", inclusive="left")
        self.frames = {}
        for station in range(1, 6 if new_station else 5):
            multiplier = 1 if degenerate else station
            counts = np.where(timeline.dayofweek >= 5, multiplier, 1)
            frame = pd.DataFrame(
                {"salidas": counts, "llegadas": counts, "elegible": True}, index=timeline
            )
            if change_future:
                mask = timeline >= config.train_end
                frame.loc[mask, "salidas"] = np.where(timeline[mask].dayofweek >= 5, 1000, 1)
            if missing_validation and station == 4:
                frame.loc[
                    (timeline >= config.train_end) & (timeline < config.validation_end), "elegible"
                ] = False
            if new_station and station == 5:
                frame.loc[timeline < config.train_end, "elegible"] = False
            self.frames[station] = frame

    def station_hours(self):
        for station, frame in self.frames.items():
            yield station, frame.copy()

    def stations(self):
        return pd.DataFrame(
            {
                "id": list(self.frames),
                "name": "synthetic-station",
                "latitude": 20.6,
                "longitude": -103.3,
                "dpcapacity": 10,
            }
        )


def test_profiles_adjust_for_exposure_instead_of_raw_counts():
    timeline = pd.date_range("2024-01-01", periods=14 * 24, freq="h")
    hours = pd.DataFrame({"salidas": 1, "llegadas": 1, "elegible": True}, index=timeline)
    first, audit = perfil_estacion(hours)
    assert audit["estado"] == "incluida"
    assert len(first) == 336
    assert first.sum() == pytest.approx(1)
    hours.loc[timeline[0], "elegible"] = False
    adjusted, audit = perfil_estacion(hours)
    np.testing.assert_allclose(first, adjusted)
    assert audit["franjas_cubiertas"] == 168
    assert weekend_shares(pd.DataFrame([adjusted])).iloc[0] == pytest.approx(2 / 7)


def test_silhouette_remains_finite_under_strict_arithmetic_checks():
    values = np.random.default_rng(42).random((473, 336))
    values /= values.sum(axis=1, keepdims=True)
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        result = silhouette_euclidea(pd.DataFrame(values), pd.Series(np.arange(473) % 2))
    assert np.isfinite(result)
    assert -1 <= result <= 1


def test_missing_weekly_slot_is_not_imputed_as_zero():
    hours = pd.DataFrame(
        {"salidas": [1], "llegadas": [1], "elegible": [True]}, index=pd.to_datetime(["2024-01-01"])
    )
    profile, audit = perfil_estacion(hours)
    assert profile is None
    assert audit["estado"] == "cobertura_semanal_incompleta"


def test_zero_activity_does_not_create_a_zero_profile():
    timeline = pd.date_range("2024-01-01", periods=7 * 24, freq="h")
    profile, audit = perfil_estacion(
        pd.DataFrame(
            {"salidas": 0, "llegadas": 0, "elegible": True},
            index=timeline,
        )
    )
    assert profile is None
    assert audit["estado"] == "sin_actividad"


def test_training_threshold_is_unchanged_by_future_data(tmp_path):
    config = configuration(tmp_path)
    first = PipelineP3(config).run(RepositoryDouble(config), tmp_path / "first")
    future = PipelineP3(config).run(
        RepositoryDouble(config, change_future=True), tmp_path / "future"
    )
    assert first.parameters["horizon_hours"] == "not_applicable_descriptive"
    assert first.parameters["training_median"] == future.parameters["training_median"]
    assert first.metrics["validation_ari_vs_train"] == 1
    assert first.metrics["test_ari_vs_train"] == 1
    assert "test_silhouette" in first.metrics
    assert all(path.is_file() for path in first.artifacts)
    png = next(path for path in first.artifacts if path.suffix == ".png")
    assert png.read_bytes().startswith(b"\x89PNG")


def test_degenerate_groups_do_not_invent_silhouette_or_perfect_stability(tmp_path):
    config = configuration(tmp_path)
    result = PipelineP3(config).run(
        RepositoryDouble(config, degenerate=True), tmp_path / "degenerate"
    )
    assert result.metrics["test_silhouette_defined"] == 0
    assert "test_silhouette" not in result.metrics
    assert result.metrics["test_ari_vs_train_defined"] == 0
    assert "test_ari_vs_train" not in result.metrics


def test_ari_uses_only_common_stations_and_reports_coverage(tmp_path):
    config = configuration(tmp_path)
    result = PipelineP3(config).run(
        RepositoryDouble(
            config,
            missing_validation=True,
            new_station=True,
        ),
        tmp_path / "coverage",
    )
    assert result.metrics["train_n_stations"] == 4
    assert result.metrics["validation_n_stations"] == 4
    assert result.metrics["validation_common_stations_vs_train"] == 3
    assert result.metrics["test_common_stations_vs_train"] == 4
    assert result.metrics["train_excluded_stations"] == 1
    assert result.metrics["validation_excluded_stations"] == 1
