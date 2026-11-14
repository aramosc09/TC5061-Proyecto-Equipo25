"""Selección auditada y promoción explícita a Model Registry (Entrega 2).

No modifica el Registry al inspeccionar candidatos ni sin --approve.
Todos los nombres, etiquetas y métricas se obtienen de los runs existentes.
"""

import json
import os
import re
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import typer

from proyecto_mibici.config import EXPERIMENT_NAME

app = typer.Typer(help="Inspección y promoción controlada de candidatos MiBici.")

MODELS = {
    "p1": ("mibici-equipo25-p1-demanda", "validation_rmse", "min"),
    "p2": ("mibici-equipo25-p2-presion", "validation_f1_macro", "max"),
    "p3": ("mibici-equipo25-p3-grupos", "validation_silhouette", "max"),
}
REQUIRED_TAGS = ("team", "question", "git_commit", "data_version", "model_family")
DEFAULT_ALIAS = "champion"
_ALIAS_PATTERN = re.compile(r"[a-z0-9_-]{1,255}")

def _validate_alias(alias: str) -> str:
    """Normaliza y valida el alias antes de registrar nada en el Registry."""
    a = alias.strip().lower()
    if not _ALIAS_PATTERN.fullmatch(a):
        raise ValueError(
            "El alias solo admite letras, números, guion y guion bajo (máximo 255)."
        )
    if a == "latest" or re.fullmatch(r"v\d+", a):
        raise ValueError("El alias no puede ser 'latest' ni parecerse a una versión (v1, v2...).")
    return a


def _alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    """Versión a la que apunta un alias, o None si el alias no existe."""
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None

def _tracking_client() -> MlflowClient:
    """Exige apuntar al servidor; no se admite registrar en un mlruns accidental."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not uri.startswith(("http://", "https://", "sqlite:///")):
        raise ValueError(
            "Configura MLFLOW_TRACKING_URI al servidor MLflow con SQLite "
            "(por ejemplo http://127.0.0.1:5000)."
        )
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _question(question: str) -> tuple[str, str, str, str]:
    q = question.lower().strip()
    if q not in MODELS:
        raise ValueError("La pregunta debe ser p1, p2 o p3.")
    name, metric, direction = MODELS[q]
    return q, name, metric, direction


def inspect_candidates(client: MlflowClient, question: str) -> list[dict[str, Any]]:
    """Lista candidatos ordenados SOLO por validación; nunca selecciona por test."""
    q, _, metric, direction = _question(question)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"No existe el experimento {EXPERIMENT_NAME!r}.")
    runs = client.search_runs([experiment.experiment_id], max_results=10000)
    candidates = []
    for run in runs:
        tags = run.data.tags
        if tags.get("question", "").lower() != q or tags.get("stage_of_work") != "candidate":
            continue
        if run.info.status != "FINISHED" or metric not in run.data.metrics:
            continue
        candidates.append(
            {
                "run_id": run.info.run_id,
                "model_family": tags.get("model_family"),
                "model_uri": tags.get("candidate_model_uri"),
                "data_version": tags.get("data_version"),
                "git_commit": tags.get("git_commit"),
                metric: run.data.metrics[metric],
                "test_metric_solo_reporte": run.data.metrics.get(metric.replace("validation_", "test_")),
            }
        )
    candidates.sort(key=lambda row: row[metric], reverse=direction == "max")
    return candidates


def validate_candidate(client: MlflowClient, question: str, run_id: str):
    """Acepta solo una corrida candidata completa del experimento y pregunta correctos."""
    q, name, metric, direction = _question(question)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"No existe el experimento {EXPERIMENT_NAME!r}.")
    run = client.get_run(run_id)
    if run.info.experiment_id != experiment.experiment_id:
        raise ValueError("El run no pertenece al experimento del Equipo 25.")
    if run.info.status != "FINISHED":
        raise ValueError("Solo se pueden registrar runs FINISHED.")
    tags = run.data.tags
    if tags.get("question", "").lower() != q:
        raise ValueError("La pregunta del run no coincide con la solicitada.")
    if tags.get("stage_of_work") != "candidate":
        raise ValueError("Se requiere un candidato final, no un baseline ni un trial de búsqueda.")
    if tags.get("git_worktree_dirty") == "true":
        raise ValueError(
            "El run se entrenó con cambios sin commit. Registra los cambios y repite "
            "la corrida candidata antes de promoverla."
        )
    for tag in REQUIRED_TAGS:
        if not tags.get(tag):
            raise ValueError(f"Falta la etiqueta requerida: {tag}.")
    if tags["team"] != "25":
        raise ValueError("La etiqueta team no corresponde al Equipo 25.")
    if metric not in run.data.metrics:
        raise ValueError(f"Falta la métrica de selección {metric}.")
    uri = tags.get("candidate_model_uri", "")
    if not uri.startswith(("models:/", "runs:/")):
        raise ValueError(
            "El run no contiene un modelo guardado en MLflow. "
            "Reentrena el candidato con la integración de log_model."
        )
    return q, name, metric, direction, run, uri


def _load_registered_model(uri: str):
    """Carga una versión del Registry; permite aislar el acceso externo en pruebas."""
    return mlflow.pyfunc.load_model(uri)

def promote_candidate(
    client: MlflowClient,
    question: str,
    run_id: str,
    reason: str,
    *,
    approve: bool = False,
    alias: str = DEFAULT_ALIAS,
) -> dict[str, Any]:
    """Previsualiza por defecto. Con approve crea versión y asigna el alias.

    Con alias="champion" (por defecto) la versión pasa a ser champion. Con otro
    alias (por ejemplo "challenger") solo se mueve ese alias: champion no se toca.
    La decisión de negocio es explícita: no se selecciona automáticamente por test.
    """
    if len(reason.strip()) < 12:
        raise ValueError("Documenta la selección en --reason (al menos 12 caracteres).")
    alias = _validate_alias(alias)
    q, name, metric, direction, run, uri = validate_candidate(client, question, run_id)
    result = {
        "question": q,
        "registered_name": name,
        "run_id": run_id,
        "model_uri": uri,
        "selection_metric": metric,
        "selection_direction": direction,
        "selection_value": run.data.metrics[metric],
        "data_version": run.data.tags["data_version"],
        "git_commit": run.data.tags["git_commit"],
        "justification": reason.strip(),
        "alias": alias,
        "operation": "preview" if not approve else "promote",
    }
    if not approve:
        return result

    # Con un alias alternativo, recordar a qué versión apunta champion para
    # comprobar al final que no se movió.
    champion_antes = (
        _alias_version(client, name, DEFAULT_ALIAS) if alias != DEFAULT_ALIAS else None
    )

    version = mlflow.register_model(model_uri=uri, name=name, await_registration_for=300)
    v = str(version.version)
    description = (
        f"MiBici Equipo 25 · {q.upper()} · candidato elegido con {metric}="
        f"{run.data.metrics[metric]:.6g}. Justificación: {reason.strip()} "
        f"Origen: {run_id}; commit: {run.data.tags['git_commit']}; "
        f"data_version: {run.data.tags['data_version']}."
    )
    client.update_registered_model(name, description=f"Modelo MiBici Equipo 25, pregunta {q.upper()}.")
    client.update_model_version(name, v, description)
    for tag, value in (
        ("team", "25"),
        ("question", q.upper()),
        ("source_run_id", run_id),
        ("git_commit", run.data.tags["git_commit"]),
        ("data_version", run.data.tags["data_version"]),
        ("source_tree_sha256", run.data.tags.get("source_tree_sha256", "not_provided")),
        ("environment_lock_sha256", run.data.tags.get("environment_lock_sha256", "not_provided")),
        ("model_family", run.data.tags["model_family"]),
        ("selection_metric", metric),
        ("selection_value", str(run.data.metrics[metric])),
    ):
        client.set_model_version_tag(name, v, tag, value)

    # Antes de asignar el alias, verificar que esta versión es cargable y tiene firma.
    loaded = _load_registered_model(f"models:/{name}/{v}")
    if loaded.metadata.signature is None:
        raise ValueError(f"El modelo registrado no tiene firma: no se modificó el alias {alias}.")
    client.set_registered_model_alias(name, alias, v)
    asignado = client.get_model_version_by_alias(name, alias)
    if str(asignado.version) != v:
        raise RuntimeError(f"El alias {alias} no apunta a la versión recién registrada.")
    if alias != DEFAULT_ALIAS:
        if _alias_version(client, name, DEFAULT_ALIAS) != champion_antes:
            raise RuntimeError("El alias champion cambió al registrar una versión alternativa.")
        result["champion_version"] = champion_antes
    result.update(version=v, alias=alias, load_uri=f"models:/{name}@{alias}")
    return result


@app.command("candidates")
def candidates_cmd(question: str = typer.Option(..., help="p1, p2 o p3")):
    """Lista las corridas candidatas sin alterar el registro."""
    result = inspect_candidates(_tracking_client(), question)
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command("promote")
def promote_cmd(
    question: str = typer.Option(..., help="p1, p2 o p3"),
    run_id: str = typer.Option(..., help="ID exacto de la corrida candidata"),
    reason: str = typer.Option(..., help="Justificación humana de la elección"),
    approve: bool = typer.Option(False, "--approve", help="Registrar y asignar el alias"),
    alias: str = typer.Option(
        DEFAULT_ALIAS,
        "--alias",
        help="Alias a asignar. Por defecto champion; con otro (p. ej. challenger) no se mueve champion.",
    ),
):
    """Vista previa sin --approve; con él registra y verifica la versión con el alias dado."""
    result = promote_candidate(
        _tracking_client(), question, run_id, reason, approve=approve, alias=alias
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command("status")
def status_cmd():
    """Verifica las versiones con alias champion y si su modelo puede cargarse."""
    client = _tracking_client()
    rows = []
    for q, (name, _, _) in MODELS.items():
        try:
            version = client.get_model_version_by_alias(name, "champion")
        except Exception as exc:
            rows.append({"question": q, "name": name, "alias": "ausente", "detail": str(exc)})
            continue
        uri = f"models:/{name}@champion"
        loaded = _load_registered_model(uri)
        rows.append(
            {
                "question": q,
                "name": name,
                "version": str(version.version),
                "source_run_id": version.tags.get("source_run_id"),
                "git_commit": version.tags.get("git_commit"),
                "data_version": version.tags.get("data_version"),
                "load_uri": uri,
                "signature_present": loaded.metadata.signature is not None,
            }
        )
    typer.echo(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
