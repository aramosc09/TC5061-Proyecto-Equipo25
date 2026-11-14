"""Periodos de monitoreo: derivados de los cortes de la v1, disjuntos y leídos por fecha."""

from dataclasses import replace
import json

import pandas as pd
import pytest

from proyecto_mibici.config import ProjectConfig
from proyecto_mibici.monitoreo import periodos as modulo
from proyecto_mibici.monitoreo.periodos import (
    COLUMNAS_P1,
    cargar_periodo,
    definir_periodos,
    metadatos_tabla,
    resumen_periodo,
    validar_periodos,
)


def test_periodos_coinciden_con_los_cortes_de_la_v1():
    config = ProjectConfig()
    periodos = definir_periodos(config)
    entrenamiento, _ = config.splits["train"]
    prueba_inicio, prueba_fin = config.splits["test"]
    assert (periodos["referencia"].inicio, periodos["referencia"].fin) == (
        entrenamiento,
        pd.Timestamp(config.train_end),
    )
    assert (periodos["actual"].inicio, periodos["actual"].fin) == (prueba_inicio, prueba_fin)
    assert periodos["evaluacion_ab"].inicio == periodos["entrenamiento_v2"].fin
    assert periodos["evaluacion_ab"].fin == config.end
    assert periodos["entrenamiento_v2"].fin == pd.Timestamp(config.train_end) + pd.DateOffset(
        years=1
    )


def test_comparacion_estacional_cubre_los_mismos_meses_un_anio_despues():
    periodos = definir_periodos()
    referencia, actual = periodos["referencia_estacional"], periodos["actual_estacional"]
    assert actual.inicio == referencia.inicio + pd.DateOffset(years=1)
    assert actual.fin == referencia.fin + pd.DateOffset(years=1)


@pytest.mark.parametrize(
    ("nombre", "inicio", "fin", "mensaje"),
    [
        ("evaluacion_ab", "2026-06-01", "2026-10-01", "se traslapan"),
        ("actual", "2025-06-01", "2026-10-01", "se traslapan"),
        ("actual", "2025-10-01", "2026-11-01", "fuera de los datos"),
        ("actual_estacional", "2025-10-01", "2026-08-01", "mismos meses"),
    ],
)
def test_validacion_rechaza_periodos_que_mezclan_lo_visto(nombre, inicio, fin, mensaje):
    config = ProjectConfig()
    periodos = definir_periodos(config)
    periodos[nombre] = replace(
        periodos[nombre], inicio=pd.Timestamp(inicio), fin=pd.Timestamp(fin)
    )
    with pytest.raises(ValueError, match=mensaje):
        validar_periodos(periodos, config)


def test_cargar_periodo_filtra_por_fecha_y_columnas(tabla_monitoreo, monkeypatch):
    config, tabla = tabla_monitoreo
    monkeypatch.setattr(modulo, "_LOTE", 97)
    periodo = definir_periodos(config)["actual"]
    datos = cargar_periodo(periodo, config)
    esperado = tabla.loc[periodo.contiene(tabla.hora)]
    assert list(datos.columns) == COLUMNAS_P1
    assert len(datos) == len(esperado)
    assert datos.hora.min() >= periodo.inicio and datos.hora.max() < periodo.fin
    pd.testing.assert_frame_equal(datos, esperado[COLUMNAS_P1].reset_index(drop=True))


def test_muestra_fija_por_semilla_y_en_el_orden_de_la_tabla(tabla_monitoreo, monkeypatch):
    config, _ = tabla_monitoreo
    monkeypatch.setattr(modulo, "_LOTE", 97)
    periodo = definir_periodos(config)["referencia"]
    primera = cargar_periodo(periodo, config, max_filas=50, semilla=3)
    repetida = cargar_periodo(periodo, config, max_filas=50, semilla=3)
    otra = cargar_periodo(periodo, config, max_filas=50, semilla=4)
    assert len(primera) == 50
    pd.testing.assert_frame_equal(primera, repetida)
    assert not primera.equals(otra)
    assert primera.sort_values(["estacion", "hora"]).index.tolist() == list(range(50))


def test_muestra_mayor_que_el_periodo_devuelve_todo(tabla_monitoreo):
    config, tabla = tabla_monitoreo
    periodo = definir_periodos(config)["evaluacion_ab"]
    datos = cargar_periodo(periodo, config, max_filas=10**6, columnas=["estacion", "hora"])
    assert len(datos) == int(periodo.contiene(tabla.hora).sum())
    resumen = resumen_periodo(periodo, datos)
    assert resumen["filas"] == len(datos) and resumen["estaciones"] == 2


def test_metadatos_rechazan_tabla_con_otros_cortes(tabla_monitoreo):
    config, _ = tabla_monitoreo
    assert metadatos_tabla(config)["data_version"] == "sintetica-v1"
    ruta = config.supervised_table_path.with_suffix(".json")
    metadatos = json.loads(ruta.read_text())
    ruta.write_text(json.dumps({**metadatos, "train_end": "2025-08-01"}))
    with pytest.raises(ValueError, match="train_end"):
        metadatos_tabla(config)


def test_sin_tabla_supervisada_el_error_es_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="tabla supervisada"):
        metadatos_tabla(ProjectConfig(root=tmp_path))
