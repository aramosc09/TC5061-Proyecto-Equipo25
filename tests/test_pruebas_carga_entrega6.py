"""Verifica entradas reales sintéticas y el perfil por escalones sin contenedores."""

import pandas as pd
import pytest

from proyecto_mibici.api.esquemas import CAMPOS_HISTORIAL, PeticionPrediccion
from pruebas_carga.escenarios import estado_en_segundo, leer_escalones
from pruebas_carga.generar_peticiones import construir_cuerpos


def datos_prueba():
    filas = []
    for estacion in [519, 520]:
        for hora in range(24):
            fila = {"estacion": estacion, "hora": pd.Timestamp(2026, 9, 10, hora),
                    **{campo: 0.0 for campo in CAMPOS_HISTORIAL}}
            filas.append(fila)
    return pd.DataFrame(filas)


def test_lotes_validos_y_reproducibles():
    cuerpos = construir_cuerpos(datos_prueba(), 519, "2026-09-10", 22)
    assert {k: len(v["registros"]) for k, v in cuerpos.items()} == {
        "una": 1, "estacion_dia": 24, "sistema_hora": 2
    }
    for cuerpo in cuerpos.values():
        assert PeticionPrediccion.model_validate(cuerpo)
    assert cuerpos == construir_cuerpos(datos_prueba(), 519, "2026-09-10", 22)


def test_rechaza_dia_incompleto():
    with pytest.raises(ValueError, match="24 horas"):
        construir_cuerpos(datos_prueba().iloc[1:], 519, "2026-09-10", 22)


def test_escalones_configurables(monkeypatch):
    escalones = leer_escalones("10:120,25:120,50:120,100:120")
    assert estado_en_segundo(0, escalones)[0] == 10
    assert estado_en_segundo(120, escalones)[0] == 25
    assert estado_en_segundo(240, escalones)[0] == 50
    assert estado_en_segundo(360, escalones)[0] == 100
    assert estado_en_segundo(480, escalones) is None
    with pytest.raises(ValueError):
        leer_escalones("10:-1")
