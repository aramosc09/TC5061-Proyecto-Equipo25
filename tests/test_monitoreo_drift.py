"""Reporte de drift: lectura del resultado de Evidently, archivos generados y run de MLflow."""

import json

import mlflow
import pytest

from proyecto_mibici.monitoreo.drift import (
    COLUMNAS_DRIFT,
    ejecutar,
    registrar_en_mlflow,
    resumir_drift,
)


def _metrica(tipo, config, valor):
    return {"config": {"type": f"evidently:metric_v2:{tipo}", **config}, "value": valor}


def _resultado(drift_contado=2):
    return {
        "metrics": [
            _metrica(
                "ValueDrift", {"column": "a", "method": "K-S p_value", "threshold": 0.05}, 0.01
            ),
            _metrica(
                "ValueDrift", {"column": "b", "method": "K-S p_value", "threshold": 0.05}, 0.40
            ),
            _metrica(
                "ValueDrift",
                {"column": "c", "method": "Wasserstein distance (normed)", "threshold": 0.1},
                0.25,
            ),
            _metrica(
                "ValueDrift",
                {"column": "d", "method": "Jensen-Shannon distance", "threshold": 0.1},
                0.02,
            ),
            _metrica(
                "DriftedColumnsCount",
                {"drift_share": 0.5},
                {"count": drift_contado, "share": drift_contado / 4},
            ),
            _metrica("RowCount", {}, 120.0),
            _metrica("DuplicatedRowCount", {}, {"count": 3.0, "share": 0.025}),
        ]
    }


def test_resumen_distingue_p_valores_de_distancias():
    tabla, resumen = resumir_drift(_resultado())
    assert dict(zip(tabla.variable, tabla.drift)) == {"a": True, "b": False, "c": True, "d": False}
    assert resumen["columnas_con_drift"] == 2
    assert set(resumen["variables_con_drift"]) == {"a", "c"}
    assert resumen["drift_del_conjunto"] is True
    assert resumen["actual_filas"] == 120.0 and resumen["actual_filas_duplicadas"] == 3.0


def test_resumen_rechaza_conteo_inconsistente_con_evidently():
    with pytest.raises(ValueError, match="no coincide"):
        resumir_drift(_resultado(drift_contado=1))


def test_resumen_rechaza_resultado_sin_drift():
    with pytest.raises(ValueError, match="no contiene"):
        resumir_drift({"metrics": []})


def test_reporte_detecta_el_drift_sembrado_y_guarda_archivos(tabla_monitoreo):
    pytest.importorskip("evidently.presets")
    config, _ = tabla_monitoreo
    resultado = ejecutar("principal", config, max_filas=0)
    informe, tabla = resultado["informe"], resultado["tabla"]
    assert set(tabla.variable) == set(COLUMNAS_DRIFT)
    assert "salidas_media_168" in informe["variables_con_drift"]
    assert informe["drift_del_conjunto"] is False
    assert informe["data_version"] == "sintetica-v1"
    assert informe["periodos"]["referencia"]["nombre"] == "referencia"
    for ruta in resultado["archivos"].values():
        assert ruta.is_file() and ruta.stat().st_size > 0
    guardado = json.loads(resultado["archivos"]["resumen"].read_text(encoding="utf-8"))
    assert guardado["columnas_con_drift"] == informe["columnas_con_drift"]


def test_comparacion_desconocida_falla(tabla_monitoreo):
    config, _ = tabla_monitoreo
    with pytest.raises(ValueError, match="comparacion"):
        ejecutar("semanal", config)


def test_run_de_mlflow_es_de_monitoreo_y_no_candidato(tabla_monitoreo, monkeypatch):
    pytest.importorskip("evidently.presets")
    config, _ = tabla_monitoreo
    tracking_uri = f"sqlite:///{config.root / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    try:
        resultado = ejecutar("estacional", config, max_filas=0)
        run = mlflow.tracking.MlflowClient().get_run(registrar_en_mlflow(resultado, config))
    finally:
        mlflow.set_tracking_uri(None)
    assert run.info.status == "FINISHED"
    assert run.data.tags["stage_of_work"] == "monitoring"
    assert run.data.tags["run_kind"] == "drift_report"
    assert run.data.tags["git_commit"]
    assert run.data.params["referencia_periodo"] == "referencia_estacional"
    assert run.data.metrics["drift_columnas"] == resultado["informe"]["columnas_con_drift"]
    assert f"drift_{COLUMNAS_DRIFT[0]}" in run.data.metrics
