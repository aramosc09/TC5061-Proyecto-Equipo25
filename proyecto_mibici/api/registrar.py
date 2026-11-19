"""Registra la copia exportada del modelo de P1 en un servidor MLflow vacío (el de Compose).

El servidor de Compose tiene su propio almacenamiento, así que su Registry empieza vacío. Este
comando sube la carpeta que ``modelo exportar`` dejó en ``MODEL_DIR`` (los mismos archivos, sin
reentrenar ni volver a serializar), crea una versión del modelo con el linaje original como tags y le
asigna el alias. Si el alias ya existe no hace nada, para que ``compose up`` pueda repetirse.
"""

import json
import os
from pathlib import Path
from typing import Any

from mlflow.exceptions import MlflowException
import typer

from proyecto_mibici.api.modelo import (
    LINEAGE_TAGS,
    METADATA_FILE,
    MODEL_SUBDIR,
    REGISTERED_MODEL,
    cargar_modelo,
)
from proyecto_mibici.config import API_MODEL_DIR
from proyecto_mibici.registry import _alias_version, _tracking_client, _validate_alias

app = typer.Typer(help="Registro del modelo exportado en el servidor MLflow de Compose.")

EXPERIMENTO = "mibici-equipo25-compose"


def _experimento(client, nombre: str) -> str:
    experimento = client.get_experiment_by_name(nombre)
    if experimento is not None:
        return experimento.experiment_id
    return client.create_experiment(nombre)


def registrar_exportado(
    client, origen: Path, *, alias: str = "champion", experimento: str = EXPERIMENTO
) -> dict[str, Any]:
    """Crea la versión con alias a partir de ``origen``; idempotente si el alias ya existe."""
    alias = _validate_alias(alias)
    metadata_path = origen / METADATA_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"No hay modelo exportado en {origen}. "
            "Ejecuta antes: python -m proyecto_mibici.api.modelo exportar"
        )
    exportado = json.loads(metadata_path.read_text(encoding="utf-8"))
    if exportado.get("registered_model") != REGISTERED_MODEL:
        raise ValueError(f"La copia exportada no es de {REGISTERED_MODEL}.")

    existente = _alias_version(client, REGISTERED_MODEL, alias)
    if existente is not None:
        return {"estado": "ya_registrado", "version": existente, "alias": alias}

    # Valida que sea P1 con firma antes de subir nada.
    cargar_modelo("local", model_dir=origen)

    linaje = {tag: str(exportado[tag]) for tag in LINEAGE_TAGS if exportado.get(tag) is not None}
    origen_tags = {
        "registrado_desde": "copia exportada (modelo exportar)",
        "version_origen": str(exportado.get("version", "")),
    }
    run = client.create_run(
        _experimento(client, experimento),
        run_name=f"registro-compose-{alias}",
        tags={**linaje, **origen_tags},
    )
    run_id = run.info.run_id
    try:
        client.log_artifacts(run_id, str(origen / MODEL_SUBDIR), artifact_path=MODEL_SUBDIR)
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    client.set_terminated(run_id)

    try:
        client.create_registered_model(REGISTERED_MODEL)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    version = client.create_model_version(
        REGISTERED_MODEL,
        source=f"{run.info.artifact_uri}/{MODEL_SUBDIR}",
        run_id=run_id,
        tags={**linaje, **origen_tags},
    )
    client.set_registered_model_alias(REGISTERED_MODEL, alias, version.version)
    return {
        "estado": "registrado",
        "version": str(version.version),
        "alias": alias,
        "run_id": run_id,
        **linaje,
    }


@app.command()
def main(
    origen: Path = typer.Option(
        Path(os.getenv("MODEL_DIR") or API_MODEL_DIR), help="Carpeta de modelo exportar"
    ),
    alias: str = typer.Option("champion", help="Alias que recibirá la versión"),
):
    """Sube la copia exportada al servidor de MLFLOW_TRACKING_URI y le asigna el alias."""
    info = registrar_exportado(_tracking_client(), origen, alias=alias)
    typer.echo(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
