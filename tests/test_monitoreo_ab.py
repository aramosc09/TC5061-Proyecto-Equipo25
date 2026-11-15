"""Pruebas focalizadas del evaluador A/B."""

import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.monitoreo.ab import _metricas, _predecir_por_lotes


def test_metricas_comparan_predicciones_sobre_mismas_filas():
    reales = np.array([1.0, 2.0, 4.0])
    predicciones = np.array([1.0, 3.0, 3.0])

    resultado = _metricas(reales, predicciones)

    assert resultado["mae"] == pytest.approx(2 / 3)
    assert resultado["rmse"] == pytest.approx(np.sqrt(2 / 3))


def test_prediccion_por_lotes_conserva_todas_las_filas():
    class Modelo:
        def predecir(self, features):
            return features["valor"].to_numpy(dtype=float) + 1

    datos = pd.DataFrame({"valor": [1.0, 2.0, 3.0, 4.0, 5.0]})

    resultado = _predecir_por_lotes(Modelo(), datos, batch_size=2)

    np.testing.assert_array_equal(resultado, [2.0, 3.0, 4.0, 5.0, 6.0])


def test_prediccion_por_lotes_rechaza_tamano_invalido():
    with pytest.raises(ValueError, match="batch_size"):
        _predecir_por_lotes(object(), pd.DataFrame({"valor": [1]}), batch_size=0)
