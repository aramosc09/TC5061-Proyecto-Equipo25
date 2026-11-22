"""Pruebas de normalización del monitoreo de recursos."""

import pytest

from proyecto_mibici.monitoreo_recursos import _memoria_mb, _porcentaje


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [("23.28%", 23.28), ("1,25%", 1.25), ("n/a", 0.0)],
)
def test_porcentaje_normaliza_valores(entrada, esperado):
    assert _porcentaje(entrada) == pytest.approx(esperado)


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [("190MB", 190.0), ("190MiB", 190.0), ("1GiB", 1024.0), ("1.5GB", 1536.0)],
)
def test_memoria_normaliza_valores(entrada, esperado):
    assert _memoria_mb(entrada) == pytest.approx(esperado)
