"""Pruebas unitarias para el módulo de reentrenamiento de la v2 (PR 3)."""

from unittest.mock import MagicMock
import pytest
from typer.testing import CliRunner

from proyecto_mibici.monitoreo.reentrenar import (
    DatosPorPeriodo,
    app,
    obtener_hiperparametros_champion,
)

runner = CliRunner()


def test_datos_por_periodo_p1(tabla_monitoreo):
    """Verifica que DatosPorPeriodo genere correctamente los conjuntos de entrenamiento y validación para P1."""
    config, _ = tabla_monitoreo
    datos = DatosPorPeriodo(config)

    train_set = datos.split("p1", "train", max_rows=100, seed=42)
    val_set = datos.split("p1", "validation")

    # Validar que los conjuntos tengan filas en X e y, y que baseline esté presente
    assert len(train_set.X) > 0
    assert len(train_set.y) > 0
    assert len(val_set.X) > 0
    assert len(val_set.y) > 0
    assert train_set.baseline is not None


def test_datos_por_periodo_pregunta_invalida(tabla_monitoreo):
    """Verifica que DatosPorPeriodo rechace preguntas distintas a P1 con ValueError."""
    config, _ = tabla_monitoreo
    datos = DatosPorPeriodo(config)

    with pytest.raises(ValueError, match="solo soporta la pregunta p1"):
        datos.split("p2", "train")


def test_obtener_hiperparametros_champion_mock():
    """Verifica la extracción e inferencia de hiperparámetros desde la versión champion."""
    mock_client = MagicMock()
    mock_version = MagicMock()
    mock_version.version = "1"
    mock_version.run_id = "run_champion_123"
    mock_client.get_model_version_by_alias.return_value = mock_version

    mock_run = MagicMock()
    mock_run.data.params = {
        "resolved_learning_rate": "0.05",
        "resolved_max_iter": "500",
        "resolved_max_leaf_nodes": "31",
        "resolved_min_samples_leaf": "200",
        "resolved_l2_regularization": "1.0",
    }
    mock_client.get_run.return_value = mock_run

    version_str, run_id, params = obtener_hiperparametros_champion(
        mock_client, ("mibici-equipo25-p1-demanda", "validation_rmse", "min")
    )

    assert version_str == "1"
    assert run_id == "run_champion_123"
    assert params["learning_rate"] == 0.05
    assert params["max_iter"] == 500
    assert params["max_leaf_nodes"] == 31
    assert params["min_samples_leaf"] == 200
    assert params["l2_regularization"] == 1.0


def test_cli_reentrenar_pregunta_no_soportada():
    """Verifica la respuesta explicativa y el código de salida 1 al solicitar P2 o P3 por CLI."""
    result = runner.invoke(app, ["--question", "p2"])

    assert result.exit_code == 1
    assert "REENTRENAMIENTO NO DISPONIBLE PARA 'P2'" in result.output
    assert "API REST en Contenedor" in result.output