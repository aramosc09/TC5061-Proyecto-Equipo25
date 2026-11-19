"""Registro de la copia exportada en un MLflow vacío (servicio registrar-modelo de Compose)."""

import json

import mlflow
from mlflow.tracking import MlflowClient
import pytest
from test_api_modelo import NAME, write_export

from proyecto_mibici.api import modelo as api_modelo
from proyecto_mibici.api import registrar

LINAJE = {
    "registered_model": NAME,
    "version": "1",
    "alias": "champion",
    "source_run_id": "77ccfab",
    "git_commit": "3f3e233",
    "data_version": "09a6a584dd5d",
    "model_family": "gradient_boosting",
}


@pytest.fixture
def servidor(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    # Con el servidor de Compose el experimento usa su --artifacts-destination; aquí, tmp_path.
    client.create_experiment(
        registrar.EXPERIMENTO, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    yield client
    mlflow.set_tracking_uri(None)


@pytest.fixture
def exportado(tmp_path):
    origen = write_export(tmp_path / "export")
    (origen / api_modelo.METADATA_FILE).write_text(json.dumps(LINAJE), encoding="utf-8")
    return origen


def test_registra_version_con_alias_y_linaje(servidor, exportado):
    info = registrar.registrar_exportado(servidor, exportado)

    assert info["estado"] == "registrado"
    version = servidor.get_model_version_by_alias(NAME, "champion")
    assert str(version.version) == info["version"] == "1"
    assert version.tags["git_commit"] == "3f3e233"
    assert version.tags["source_run_id"] == "77ccfab"
    assert version.tags["version_origen"] == "1"


def test_api_carga_desde_el_registry_tras_registrar(servidor, exportado):
    registrar.registrar_exportado(servidor, exportado)

    servido = api_modelo.cargar_modelo("registry")

    assert servido.metadatos["alias"] == "champion"
    assert servido.metadatos["data_version"] == "09a6a584dd5d"
    assert servido.metadatos["source"] == "registry"


def test_es_idempotente_si_el_alias_ya_existe(servidor, exportado):
    registrar.registrar_exportado(servidor, exportado)
    segunda = registrar.registrar_exportado(servidor, exportado)

    assert segunda == {"estado": "ya_registrado", "version": "1", "alias": "champion"}
    assert len(servidor.search_model_versions(f"name='{NAME}'")) == 1


def test_sin_copia_exportada_falla_con_instruccion(servidor, tmp_path):
    with pytest.raises(FileNotFoundError, match="modelo exportar"):
        registrar.registrar_exportado(servidor, tmp_path / "vacio")


def test_rechaza_modelo_de_otra_pregunta(servidor, tmp_path):
    origen = write_export(tmp_path / "export", question="P2")
    (origen / api_modelo.METADATA_FILE).write_text(json.dumps(LINAJE), encoding="utf-8")

    with pytest.raises(ValueError, match="se esperaba P1"):
        registrar.registrar_exportado(servidor, origen)
    assert servidor.search_model_versions(f"name='{NAME}'") == []
