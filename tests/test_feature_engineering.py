"""Pruebas de las transformaciones reutilizables derivadas de la exploración."""

import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.features import (
    PROFILE_COLUMNS,
    construir_features_calendario,
    construir_features_flujo_historico,
    construir_perfil_semanal,
    proporcion_fin_semana,
)


def test_calendar_features_are_known_at_the_prediction_time():
    index = pd.to_datetime(["2024-01-01 00:00", "2024-01-07 23:00"])
    features = construir_features_calendario(index)

    assert features.loc[index[0], "fin_semana"] == np.False_
    assert features.loc[index[1], "fin_semana"] == np.True_
    assert features.loc[index[0], "hora_cos"] == 1
    assert features.loc[index[0], "dia_semana_cos"] == 1


def test_historical_flow_features_do_not_change_with_current_or_future_events():
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    hours = pd.DataFrame({"salidas": 1, "llegadas": 2, "elegible": True}, index=index)
    first = construir_features_flujo_historico(hours)

    hours.loc[index[100] :, "salidas"] = 999
    changed = construir_features_flujo_historico(hours)

    pd.testing.assert_frame_equal(first.loc[: index[100]], changed.loc[: index[100]])


def test_weekend_share_uses_only_normalized_profile_slots():
    index = pd.date_range("2024-01-01", periods=7 * 24, freq="h")
    hours = pd.DataFrame({"salidas": 1, "llegadas": 1, "elegible": True}, index=index)

    profile, audit = construir_perfil_semanal(hours)

    assert audit["estado"] == "incluida"
    assert profile is not None
    assert profile.index.tolist() == PROFILE_COLUMNS
    assert profile.sum() == pytest.approx(1)
    assert proporcion_fin_semana(pd.DataFrame([profile])).iloc[0] == pytest.approx(2 / 7)
