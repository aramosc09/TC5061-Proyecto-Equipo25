"""Las variables de una prediccion en t solo pueden usar informacion anterior a t."""

import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.config import load_settings
from proyecto_mibici.features import predictor_columns, station_features, station_profile


def hourly_fixture(hours=900, seed=25):
    """Serie horaria determinista de salidas y llegadas para una estacion."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-10-01", periods=hours, freq="h")
    return pd.DataFrame(
        {
            "departures": rng.integers(0, 6, hours).astype(float),
            "arrivals": rng.integers(0, 6, hours).astype(float),
        },
        index=index,
    )


def test_future_events_cannot_change_current_predictors():
    """Alterar una hora futura no debe modificar ninguna variable predictora previa."""
    settings = load_settings()
    hourly = hourly_fixture()
    base = station_features(hourly, 1, 10, settings, 0.2)

    perturbed_input = hourly.copy()
    perturbed_input.iloc[-1, :] = 999
    perturbed = station_features(perturbed_input, 1, 10, settings, 0.2)

    columns = predictor_columns(base)
    left = base.loc[base.hour.isin(perturbed.hour), ["hour", *columns]].reset_index(drop=True)
    right = perturbed.loc[perturbed.hour.isin(base.hour), ["hour", *columns]].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(left, right)


def test_predictor_columns_exclude_labels_and_capacity():
    """La lista blanca impide que etiquetas o capacidad entren como predictores."""
    frame = station_features(hourly_fixture(), 1, 10, load_settings(), 0.2)
    columns = predictor_columns(frame)
    for leaked in ("target_trips", "target_pressure", "target_hour", "capacity", "departures"):
        assert leaked not in columns


def test_invalid_capacity_does_not_create_pressure_labels():
    """Sin capacidad valida no se puede aproximar el riesgo de P2."""
    frame = station_features(hourly_fixture(), 1, 0, load_settings(), 0.2)
    assert frame.target_pressure.isna().all()


def test_profile_is_normalized_weekly_shape():
    """El perfil de P3 resume la semana completa y cada serie suma uno."""
    profile = station_profile(hourly_fixture())
    for prefix in ("departures_", "arrivals_"):
        series = [v for k, v in profile.items() if k.startswith(prefix)]
        assert len(series) == 168
        assert sum(series) == pytest.approx(1.0, abs=1e-6)
