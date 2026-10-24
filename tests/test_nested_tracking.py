"""Búsquedas como nested runs y etiquetas obligatorias en candidatos y búsquedas."""

import json

import mlflow
import pytest

from proyecto_mibici.config import ProjectConfig
from proyecto_mibici.experiments import ProyectoMiBici
from proyecto_mibici.modeling.base import CandidateResult
from proyecto_mibici.tracking import RegistroExperimentos

REQUIRED_TAGS = ("team", "question", "git_commit", "model_family", "stage_of_work", "data_version")


@pytest.fixture
def registry(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    mlflow.set_tracking_uri(tracking_uri)
    config = ProjectConfig(root=tmp_path)
    manifest = {
        "data_version": "synthetic-v1",
        "resumen": {"n_viajes": 10},
        "salida": {"n_filas": 10, "sha256": "synthetic"},
    }
    config.manifest_path.parent.mkdir(parents=True)
    config.manifest_path.write_text(json.dumps(manifest))
    yield RegistroExperimentos(config), manifest
    mlflow.set_tracking_uri(None)


def fake_train(params, output_dir):
    # Prueba favorece al peor trial de validación.
    return CandidateResult(
        "P1",
        "hgb",
        parameters={"model": "hgb", **params},
        metrics={
            "validation_rmse": 1 / params["learning_rate"],
            "test_rmse": params["learning_rate"],
        },
        model_family="gradient_boosting",
    )


def test_search_has_one_parent_and_one_child_per_combination(registry):
    registry, manifest = registry
    trials = [{"learning_rate": rate} for rate in (0.05, 0.1, 0.2)]
    report = registry.search(
        manifest,
        "p1",
        "hgb",
        "gradient_boosting",
        trials,
        fake_train,
        ("validation_rmse", "min"),
        {"requested_model": "hgb"},
    )
    client = mlflow.tracking.MlflowClient()
    parent = client.get_run(report["parent_run_id"])
    children = client.search_runs(
        [parent.info.experiment_id], f"tags.mlflow.parentRunId = '{parent.info.run_id}'"
    )
    assert len(children) == 3
    assert parent.data.tags["search_role"] == "parent"
    assert parent.data.tags["best_trial_run_id"] == report["best_trial_run_id"]
    assert parent.data.params["best_learning_rate"] == "0.2"
    assert parent.data.metrics["best_validation_rmse"] == 5
    for run in [parent, *children]:
        assert run.info.status == "FINISHED"
        assert all(run.data.tags.get(tag) for tag in REQUIRED_TAGS)
        assert run.data.tags["stage_of_work"] == "search"
    assert {run.data.tags["search_role"] for run in children} == {"trial"}
    assert {run.data.params["learning_rate"] for run in children} == {"0.05", "0.1", "0.2"}


def test_search_refuses_to_select_with_test_metrics(registry):
    registry, manifest = registry
    with pytest.raises(ValueError, match="validación"):
        registry.search(
            manifest, "p1", "hgb", "gradient_boosting", [], fake_train, ("test_rmse", "min"), {}
        )


def test_candidate_run_has_filter_tags_and_reproducibility_artifacts(registry, tmp_path):
    registry, manifest = registry
    summary = registry.candidate(
        manifest,
        "p1",
        "hgb",
        "gradient_boosting",
        lambda output_dir: fake_train({"learning_rate": 0.1}, output_dir),
        {"requested_model": "hgb"},
    )
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(summary["run_id"])
    assert all(run.data.tags.get(tag) for tag in REQUIRED_TAGS)
    assert run.data.tags["stage_of_work"] == "candidate"
    assert run.data.tags["model_family"] == "gradient_boosting"
    assert run.data.tags["mlflow.runName"].startswith("p1-hgb-candidate")
    assert run.data.metrics["test_rmse"] == 0.1
    assert {item.path for item in client.list_artifacts(run.info.run_id, "reproducibilidad")} == {
        "reproducibilidad/codigo_ejecutado.zip",
        "reproducibilidad/data_manifest.json",
    }


def test_baseline_rejects_hyperparameters_and_unknown_questions(tmp_path):
    project = ProyectoMiBici.__new__(ProyectoMiBici)
    project.config = ProjectConfig(root=tmp_path)
    with pytest.raises(ValueError, match="baseline no acepta"):
        project.train("p1", "baseline", {"n_estimators": "10"})
    with pytest.raises(ValueError, match="p1, p2 o p3"):
        project.train("p4", "rf", {})
