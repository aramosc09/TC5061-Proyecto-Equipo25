"""Pruebas focalizadas del generador reproducible de carga."""

import random

import pytest

from proyecto_mibici.api.esquemas import PeticionPrediccion
from proyecto_mibici.carga import batch_size_desde_entorno, crear_peticion


def test_crear_peticion_produce_lote_valido():
    peticion = crear_peticion(10, random.Random(42))

    modelo = PeticionPrediccion.model_validate(peticion)

    assert len(modelo.registros) == 10


@pytest.mark.parametrize("batch_size", [0, 1001])
def test_crear_peticion_rechaza_tamano_invalido(batch_size):
    with pytest.raises(ValueError, match="1 y 1000"):
        crear_peticion(batch_size, random.Random(42))


def test_generador_es_reproducible_con_misma_semilla():
    primera = crear_peticion(3, random.Random(123))
    segunda = crear_peticion(3, random.Random(123))

    assert primera == segunda


def test_batch_size_del_entorno_rechaza_valor_invalido(monkeypatch):
    monkeypatch.setenv("LOCUST_BATCH_SIZE", "0")

    with pytest.raises(ValueError, match="1 y 1000"):
        batch_size_desde_entorno()
