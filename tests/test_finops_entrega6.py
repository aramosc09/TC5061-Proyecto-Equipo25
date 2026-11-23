"""Pruebas deterministas del análisis FinOps."""

import pytest

from proyecto_mibici.finops import capacidad_observada, escenarios, estimar_costos


RESUMEN = {"lote_registros": 10, "request_count": 100, "failure_count": 0,
           "requests_per_second": 50}


PRECIOS = [
    {"modelo_facturacion": "siempre_encendido", "usd_hora": "0.02",
     "proveedor": "AWS", "instancia": "VM", "region": "us-east-1",
     "fuente_url": "https://example.org/vm"},
    {"modelo_facturacion": "escala_a_cero", "usd_vcpu_segundo": "0.000024",
     "usd_gib_segundo": "0.0000025", "usd_millon_peticiones": "0.40", "vcpus": "1",
     "gib_memoria": "1", "proveedor": "GCP", "instancia": "Cloud Run",
     "region": "us-central1", "fuente_url": "https://example.org/cloud"},
]


def test_capacidad_es_predicciones_no_peticiones():
    assert capacidad_observada(RESUMEN) == pytest.approx(500)


def test_escenarios_suman_un_mes():
    assert escenarios(450) == {"base": 324000, "10x": 3240000, "100x": 32400000}


def test_opciones_y_costos_por_inferencia():
    tabla = estimar_costos(RESUMEN, PRECIOS)
    assert len(tabla) == 6
    vm = tabla[0]
    assert vm["costo_mensual_usd_estimado"] == pytest.approx(14.6)
    assert vm["costo_por_inferencia_usd"] == pytest.approx(14.6 / 324000)
    assert vm["costo_por_peticion_http_usd"] == pytest.approx(14.6 / 32400)
    assert vm["costo_teorico_capacidad_100pct_usd"] == round(0.02 / (500 * 3600), 11)
    assert vm["costo_teorico_capacidad_60pct_usd"] == round(0.02 / (500 * 3600 * 0.6), 11)
    assert tabla[1]["costo_por_peticion_http_usd"] == pytest.approx(
        tabla[1]["costo_mensual_usd_estimado"] / 32400
    )
    assert tabla[1]["costo_mensual_usd_estimado"] > 0
    assert tabla[1]["segundos_facturados_estimados"] >= 3240


def test_rechaza_utilizacion_invalida():
    with pytest.raises(ValueError):
        estimar_costos(RESUMEN, PRECIOS, utilizacion=0)
