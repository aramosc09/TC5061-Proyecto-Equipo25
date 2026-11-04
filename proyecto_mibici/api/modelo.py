"""Modelo de P1 que sirve la API, recuperado del Model Registry de MLflow.

Dos fuentes, elegidas con ``MODEL_SOURCE``:

- ``registry``: carga ``MODEL_URI`` (nombre + alias o versión) del servidor en ``MLFLOW_TRACKING_URI``.
- ``local`` (por defecto): carga la copia que ``exportar`` descargó del Registry en ``MODEL_DIR``,
  pensada para empaquetarse en la imagen del contenedor sin depender del servidor.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import mlflow
from mlflow.artifacts import download_artifacts
import numpy as np
import pandas as pd
import typer

from proyecto_mibici.config import API_MODEL_DIR
from proyecto_mibici.registry import MODELS, _tracking_client

app = typer.Typer(help="Exportación y verificación del modelo de P1 servido por la API.")

REGISTERED_MODEL = MODELS["p1"][0]
DEFAULT_MODEL_URI = f"models:/{REGISTERED_MODEL}@champion"
METADATA_FILE = "metadata_registro.json"
MODEL_SUBDIR = "model"
LINEAGE_TAGS = (
    "source_run_id",
    "git_commit",
    "data_version",
    "model_family",
    "selection_metric",
    "selection_value",
)
_MODEL_URI = re.compile(r"^models:/(?P<name>[\w.-]+)(?:@(?P<alias>[\w-]+)|/(?P<version>\d+))$")


def parse_model_uri(uri: str) -> tuple[str, str | None, str | None]:
    """Acepta solo el modelo de P1 por alias o versión; nunca la última versión implícita."""
    match = _MODEL_URI.match(uri.strip())
    if match is None:
        raise ValueError(
            "MODEL_URI debe tener la forma models:/<nombre>@<alias> o models:/<nombre>/<versión>."
        )
    if match["name"] != REGISTERED_MODEL:
        raise ValueError(f"La API sirve solo {REGISTERED_MODEL}, no {match['name']}.")
    return match["name"], match["alias"], match["version"]


def resolve_version(client, uri: str) -> dict[str, Any]:
    """Traduce el alias a una versión concreta y reúne el linaje registrado en ella."""
    name, alias, version = parse_model_uri(uri)
    if alias:
        model_version = client.get_model_version_by_alias(name, alias)
    else:
        model_version = client.get_model_version(name, version)
    tags = dict(model_version.tags or {})
    return {
        "registered_model": name,
        "version": str(model_version.version),
        "alias": alias,
        "load_uri": f"models:/{name}/{model_version.version}",
        "run_id": model_version.run_id,
        **{tag: tags.get(tag) for tag in LINEAGE_TAGS},
    }


@dataclass(frozen=True)
class ModeloServido:
    """Modelo cargado con su linaje; predice en el orden y tipos de su firma."""

    modelo: Any
    metadatos: dict[str, Any]

    @property
    def columnas(self) -> list[str]:
        return self.modelo.metadata.get_input_schema().input_names()

    def predecir(self, features: pd.DataFrame) -> np.ndarray:
        faltantes = [name for name in self.columnas if name not in features.columns]
        if faltantes:
            raise ValueError(f"Faltan variables del modelo: {', '.join(faltantes)}.")
        tipos = self.modelo.metadata.get_input_schema().numpy_types()
        entrada = features[self.columnas].astype(dict(zip(self.columnas, tipos)))
        prediccion = np.asarray(self.modelo.predict(entrada), dtype=float).ravel()
        if not np.isfinite(prediccion).all():
            raise ValueError("El modelo produjo predicciones no finitas.")
        return prediccion


def _verificar_pregunta(modelo) -> None:
    """Evita servir por error el modelo de otra pregunta con el esquema de P1."""
    pregunta = (modelo.metadata.metadata or {}).get("question")
    if pregunta != "P1":
        raise ValueError(f"El modelo cargado es de la pregunta {pregunta!r}, se esperaba P1.")
    if modelo.metadata.get_input_schema() is None:
        raise ValueError("El modelo cargado no tiene firma de entrada.")


def cargar_modelo(
    fuente: str | None = None,
    *,
    model_uri: str | None = None,
    model_dir: Path | str | None = None,
) -> ModeloServido:
    """Carga el modelo según MODEL_SOURCE (local | registry); los argumentos tienen prioridad."""
    fuente = (fuente or os.getenv("MODEL_SOURCE", "local")).strip().lower()
    if fuente == "registry":
        info = resolve_version(_tracking_client(), model_uri or os.getenv("MODEL_URI", DEFAULT_MODEL_URI))
        modelo = mlflow.pyfunc.load_model(info["load_uri"])
    elif fuente == "local":
        path = Path(model_dir or os.getenv("MODEL_DIR") or API_MODEL_DIR)
        metadata_path = path / METADATA_FILE
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"No hay modelo exportado en {path}. "
                "Ejecuta: python -m proyecto_mibici.api.modelo exportar"
            )
        info = json.loads(metadata_path.read_text(encoding="utf-8"))
        modelo = mlflow.pyfunc.load_model(str(path / MODEL_SUBDIR))
    else:
        raise ValueError("MODEL_SOURCE debe ser 'local' o 'registry'.")
    _verificar_pregunta(modelo)
    return ModeloServido(modelo=modelo, metadatos={**info, "source": fuente})


def exportar_modelo(client, model_uri: str, destino: Path) -> dict[str, Any]:
    """Descarga una versión concreta del Registry y guarda su linaje junto al modelo."""
    info = resolve_version(client, model_uri)
    destino.mkdir(parents=True, exist_ok=True)
    model_path = destino / MODEL_SUBDIR
    if model_path.exists():
        shutil.rmtree(model_path)
    download_artifacts(artifact_uri=info["load_uri"], dst_path=str(model_path))
    (destino / METADATA_FILE).write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return info


@app.command("exportar")
def exportar_cmd(
    model_uri: str = typer.Option(DEFAULT_MODEL_URI, help="models:/<nombre>@<alias> o /<versión>"),
    destino: Path = typer.Option(API_MODEL_DIR, help="Carpeta que copiará la imagen"),
):
    """Descarga el modelo del Registry para empaquetarlo en la imagen."""
    info = exportar_modelo(_tracking_client(), model_uri, destino)
    typer.echo(json.dumps(info, indent=2, ensure_ascii=False))


@app.command("verificar")
def verificar_cmd(
    fuente: str = typer.Option("local", help="local o registry"),
    model_uri: str = typer.Option(DEFAULT_MODEL_URI, help="Solo con --fuente registry"),
):
    """Carga el modelo y predice su input_example (agregado, sin datos personales)."""
    servido = cargar_modelo(fuente, model_uri=model_uri)
    ejemplo = servido.modelo.input_example
    if ejemplo is None:
        raise ValueError("El modelo no guardó input_example.")
    prediccion = servido.predecir(pd.DataFrame(ejemplo))
    typer.echo(
        json.dumps(
            {**servido.metadatos, "columnas": len(servido.columnas), "prediccion": prediccion.tolist()},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    app()
