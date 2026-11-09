"""Pruebas del cliente sin requerir servidor MLflow ni contenedor."""

from datetime import datetime

import httpx
import pytest

from proyecto_mibici.api import cliente
from proyecto_mibici.api.esquemas import CAMPOS_HISTORIAL


def fila_valida():
    return {
        "estacion": 519,
        "hora": datetime(2026, 9, 10, 22),
        "conjunto": 2,
        "salidas": 1.0,
        **{nombre: 0.0 for nombre in CAMPOS_HISTORIAL},
    }


def test_preparar_peticion_no_envia_observado():
    cuerpo, observado = cliente.preparar_peticion(fila_valida())
    enviado = cuerpo.model_dump(mode="json")
    assert observado == 1.0
    assert len(enviado["registros"]) == 1
    assert len(CAMPOS_HISTORIAL) == 24
    assert "estacion" not in enviado["registros"][0]
    assert "salidas" not in enviado["registros"][0]
    assert enviado["registros"][0]["fecha_hora"] == "2026-09-10T22:00:00"


@pytest.mark.parametrize("observado", [float("nan"), float("inf"), -1.0])
def test_preparar_peticion_rechaza_observacion_invalida(observado):
    fila = fila_valida()
    fila["salidas"] = observado
    with pytest.raises(ValueError, match="finita y no negativa"):
        cliente.preparar_peticion(fila)


def test_solicitar_prediccion_valida_respuesta(monkeypatch):
    peticion, _ = cliente.preparar_peticion(fila_valida())
    recibido = {}

    def enviar(url, json, timeout):
        recibido["url"] = url
        recibido["datos"] = json
        recibido["timeout"] = timeout
        return httpx.Response(
            200,
            json={
                "predicciones": [{"salidas_esperadas": 1.25}],
                "modelo": {"alias": "champion", "version": "1"},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(cliente.httpx, "post", enviar)
    respuesta = cliente.solicitar_prediccion("http://localhost:8000/", peticion)
    assert recibido["url"] == "http://localhost:8000/predict"
    assert len(recibido["datos"]["registros"]) == 1
    assert respuesta.predicciones[0].salidas_esperadas == 1.25
    assert respuesta.modelo["alias"] == "champion"


def test_buscar_registro_rechaza_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="No existe"):
        cliente.buscar_registro(tmp_path / "inexistente.parquet", 519, datetime(2026, 9, 10, 22), 2)
