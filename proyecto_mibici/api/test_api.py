"""Pruebas de la API (PR 2). Usan un modelo falso: sin servidor ni datos reales."""

import copy
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from proyecto_mibici.api import main
from proyecto_mibici.api.esquemas import (
    CAMPOS_HISTORIAL,
    COLUMNAS_MODELO,
    MAX_REGISTROS,
    RegistroEntrada,
    calcular_calendario,
)

REGISTRO_VALIDO = {
    "fecha_hora": "2026-03-10T08:00:00",
    "salidas_lag_1": 4, "salidas_lag_3": 2, "salidas_lag_24": 5,
    "salidas_lag_168": 6, "salidas_lag_336": 5, "salidas_lag_504": 7,
    "salidas_lag_672": 4,
    "salidas_media_3": 3.0, "salidas_media_24": 1.2, "salidas_media_168": 1.4,
    "salidas_media_4_semanas": 5.5,
    "llegadas_lag_1": 3, "llegadas_lag_3": 1, "llegadas_lag_24": 4,
    "llegadas_lag_168": 5, "llegadas_lag_336": 4, "llegadas_lag_504": 6,
    "llegadas_lag_672": 3,
    "llegadas_media_3": 2.3, "llegadas_media_24": 1.1, "llegadas_media_168": 1.3,
    "llegadas_media_4_semanas": 4.5,
    "balance_acumulado_3": 2.0, "balance_acumulado_24": 2.4,
}  # fmt: skip


def con(**cambios):
    """Petición de un registro válido con algunos campos cambiados."""
    registro = copy.deepcopy(REGISTRO_VALIDO)
    registro.update(cambios)
    return {"registros": [registro]}


def sin(campo):
    """Petición de un registro válido sin un campo."""
    registro = copy.deepcopy(REGISTRO_VALIDO)
    del registro[campo]
    return {"registros": [registro]}


class ModeloFalso:
    """Imita la interfaz de cargar_modelo(): columnas, predecir, metadatos."""

    columnas = list(COLUMNAS_MODELO)
    metadatos = {
        "registered_model": "mibici-equipo25-p1-demanda",
        "version": "1",
        "alias": "champion",
        "source": "test",
    }

    def __init__(self, salida=2.5):
        self.salida = salida
        self.recibido = None

    def predecir(self, df):
        self.recibido = df
        return np.full(len(df), self.salida, dtype=float)


@pytest.fixture
def modelo_falso():
    return ModeloFalso()


@pytest.fixture
def cliente(monkeypatch, modelo_falso):
    monkeypatch.setattr(main, "cargar_modelo", lambda: modelo_falso)
    with TestClient(main.app) as c:  # el "with" ejecuta el lifespan
        yield c


@pytest.fixture
def cliente_sin_modelo(monkeypatch):
    def falla():
        raise FileNotFoundError("no hay modelo exportado")

    monkeypatch.setattr(main, "cargar_modelo", falla)
    with TestClient(main.app) as c:
        yield c


# --- Camino feliz -----------------------------------------------------------


def test_health_ok(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"status": "ok"}


def test_model_info_devuelve_version_y_alias(cliente):
    respuesta = cliente.get("/model-info")
    assert respuesta.status_code == 200
    assert respuesta.json()["version"] == "1"
    assert respuesta.json()["alias"] == "champion"


def test_predict_valido(cliente):
    respuesta = cliente.post("/predict", json=con())
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert len(cuerpo["predicciones"]) == 1
    assert cuerpo["predicciones"][0]["salidas_esperadas"] >= 0
    assert cuerpo["modelo"]["alias"] == "champion"


def test_predict_una_prediccion_por_registro(cliente):
    peticion = {"registros": [REGISTRO_VALIDO] * 3}
    respuesta = cliente.post("/predict", json=peticion)
    assert respuesta.status_code == 200
    predicciones = respuesta.json()["predicciones"]
    assert len(predicciones) == 3
    assert all(p["salidas_esperadas"] >= 0 for p in predicciones)


def test_predict_envia_las_30_variables_al_modelo(cliente, modelo_falso):
    cliente.post("/predict", json=con())
    recibido = modelo_falso.recibido
    assert set(recibido.columns) == set(COLUMNAS_MODELO)
    assert len(COLUMNAS_MODELO) == 30
    assert recibido["hora_sin"].iloc[0] == pytest.approx(math.sin(2 * math.pi / 3))
    assert recibido["mes"].iloc[0] == 3


def test_balance_puede_ser_negativo(cliente):
    peticion = con(balance_acumulado_3=-229.0, balance_acumulado_24=-10.5)
    assert cliente.post("/predict", json=peticion).status_code == 200


# --- Validación de entrada (422) -------------------------------------------

CASOS_INVALIDOS = {
    "campo_faltante": sin("salidas_lag_1"),
    "campo_de_mas": con(campo_inventado=1),
    "campo_mal_escrito": con(salida_lag_1=4),
    "conteo_negativo": con(salidas_lag_1=-1),
    "conteo_excesivo": con(salidas_lag_1=10**6),
    "balance_excesivo": con(balance_acumulado_24=10**6),
    "texto_en_campo_numerico": con(salidas_lag_1="mucho"),
    "hora_con_minutos": con(fecha_hora="2026-03-10T08:30:00"),
    "hora_con_segundos": con(fecha_hora="2026-03-10T08:00:15"),
    "mes_imposible": con(fecha_hora="2026-13-10T08:00:00"),
    "dia_imposible": con(fecha_hora="2026-02-30T08:00:00"),
    "fecha_basura": con(fecha_hora="ayer por la tarde"),
    "media_salidas_incoherente": con(salidas_media_4_semanas=9.0),
    "media_llegadas_incoherente": con(llegadas_media_4_semanas=0.0),
    "lista_vacia": {"registros": []},
    "demasiados_registros": {"registros": [REGISTRO_VALIDO] * (MAX_REGISTROS + 1)},
    "sin_registros": {},
    "registros_no_es_lista": {"registros": REGISTRO_VALIDO},
    "campo_de_mas_en_la_raiz": {**con(), "extra": 1},
}


@pytest.mark.parametrize(
    "peticion", CASOS_INVALIDOS.values(), ids=CASOS_INVALIDOS.keys()
)
def test_predict_rechaza_entradas_invalidas(cliente, peticion):
    assert cliente.post("/predict", json=peticion).status_code == 422


def test_predict_acepta_el_maximo_de_registros(cliente):
    peticion = {"registros": [REGISTRO_VALIDO] * MAX_REGISTROS}
    assert cliente.post("/predict", json=peticion).status_code == 200


# --- Sin modelo (503) -------------------------------------------------------


def test_health_503_sin_modelo(cliente_sin_modelo):
    respuesta = cliente_sin_modelo.get("/health")
    assert respuesta.status_code == 503
    assert "exportar" in respuesta.json()["detail"]


def test_predict_503_sin_modelo(cliente_sin_modelo):
    assert cliente_sin_modelo.post("/predict", json=con()).status_code == 503


def test_model_info_503_sin_modelo(cliente_sin_modelo):
    assert cliente_sin_modelo.get("/model-info").status_code == 503


# --- Salida del modelo fuera de rango (error controlado) --------------------


@pytest.mark.parametrize("salida", [-1.0, float("nan"), float("inf")])
def test_predict_salida_invalida_da_error_controlado(monkeypatch, salida):
    monkeypatch.setattr(main, "cargar_modelo", lambda: ModeloFalso(salida))
    with TestClient(main.app) as c:
        respuesta = c.post("/predict", json=con())
    assert respuesta.status_code == 500
    assert "fuera de rango" in respuesta.json()["detail"]


# --- Variables de calendario ------------------------------------------------


def test_calendario_domingo_a_medianoche():
    cal = calcular_calendario(datetime(2026, 3, 8, 0, 0))  # domingo
    assert cal["hora_sin"] == pytest.approx(0.0, abs=1e-12)
    assert cal["hora_cos"] == pytest.approx(1.0)
    assert cal["dia_semana_sin"] == pytest.approx(math.sin(2 * math.pi * 6 / 7))
    assert cal["dia_semana_cos"] == pytest.approx(math.cos(2 * math.pi * 6 / 7))
    assert cal["fin_semana"] == 1.0
    assert cal["mes"] == 3.0


def test_calendario_lunes_laboral():
    cal = calcular_calendario(datetime(2026, 3, 9, 8, 0))  # lunes
    assert cal["hora_sin"] == pytest.approx(math.sin(2 * math.pi * 8 / 24))
    assert cal["dia_semana_sin"] == pytest.approx(0.0, abs=1e-12)
    assert cal["dia_semana_cos"] == pytest.approx(1.0)
    assert cal["fin_semana"] == 0.0


def test_calendario_sabado_y_diciembre():
    cal = calcular_calendario(datetime(2026, 12, 5, 23, 0))  # sábado
    assert cal["fin_semana"] == 1.0
    assert cal["mes"] == 12.0
    assert cal["hora_sin"] == pytest.approx(math.sin(2 * math.pi * 23 / 24))


def test_a_fila_tiene_exactamente_las_30_variables():
    fila = RegistroEntrada(**REGISTRO_VALIDO).a_fila()
    assert set(fila) == set(COLUMNAS_MODELO)
    assert len(CAMPOS_HISTORIAL) == 24


# --- Coincidencia con el modelo real (solo si ya se exportó) ----------------

RUTA_MODELO = Path(os.environ.get("MODEL_DIR", "models/api/p1"))

@pytest.mark.skipif(not RUTA_MODELO.exists(), reason="No hay modelo exportado")
def test_columnas_coinciden_con_el_modelo_real(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "local")
    servido = main.cargar_modelo()
    assert set(servido.columnas) == set(COLUMNAS_MODELO)