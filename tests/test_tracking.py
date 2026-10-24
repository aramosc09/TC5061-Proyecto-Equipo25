"""La ejecución de Projects debe reutilizar un padre y registrar tres hijos completos."""

import json
import zipfile

import mlflow
from typer.testing import CliRunner

from proyecto_mibici.config import ProjectConfig
from proyecto_mibici.experiments import app
from proyecto_mibici.modeling.base import BaselineResult
from proyecto_mibici.tracking import RegistroExperimentos


class RepositoryDouble:
    def __init__(self, manifest):
        self.manifest = manifest

    def verify_base(self):
        return self.manifest


def pipeline_double(question):
    def run(self, repository, output_dir):
        output_dir.mkdir(parents=True)
        graph = output_dir / "agregado.png"
        graph.write_bytes(b"synthetic-aggregate-image")
        return BaselineResult(
            question,
            "synthetic",
            {"model": "synthetic"},
            {"validation_score": 0.5, "test_score": 0.5},
            [graph],
        )

    return type(f"Pipeline{question}", (), {"run": run})()


def test_projects_context_has_one_parent_three_children_and_source_snapshot(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    config = ProjectConfig(root=tmp_path)
    manifest = {
        "data_version": "synthetic-v1",
        "resumen": {"n_viajes": 10},
        "salida": {"n_filas": 10, "sha256": "synthetic"},
    }
    config.manifest_path.parent.mkdir(parents=True)
    config.manifest_path.write_text(json.dumps(manifest))
    registry = RegistroExperimentos(config)
    experiment_id = registry.configure()
    client = mlflow.tracking.MlflowClient()
    parent = client.create_run(experiment_id)
    monkeypatch.setenv("MLFLOW_RUN_ID", parent.info.run_id)
    report = registry.run(
        RepositoryDouble(manifest), [pipeline_double(q) for q in ("P1", "P2", "P3")]
    )
    assert report["parent_run_id"] == parent.info.run_id
    runs = client.search_runs([experiment_id])
    assert len(runs) == 4
    assert all(run.info.status == "FINISHED" for run in runs)
    for summary in report["runs"]:
        run = client.get_run(summary["run_id"])
        assert run.data.tags["mlflow.parentRunId"] == parent.info.run_id
        assert run.data.tags["question"] == summary["question"]
        assert run.data.tags["git_commit"]
        assert run.data.tags["git_worktree_dirty"] == "true"
        assert run.data.tags["stage_of_work"] == "baseline"
        assert run.data.tags["model_family"] == "synthetic"
        assert run.data.tags["data_version"] == "synthetic-v1"
        assert run.data.params["data_version"] == "synthetic-v1"
        assert run.data.metrics["test_score"] == 0.5
        archive = client.download_artifacts(
            run.info.run_id, "reproducibilidad/codigo_ejecutado.zip"
        )
        with zipfile.ZipFile(archive) as snapshot:
            assert "MLproject" in snapshot.namelist()
            assert "python_env.yaml" in snapshot.namelist()
            assert "requirements.txt" in snapshot.namelist()
            assert "uv.lock" in snapshot.namelist()
            assert not any(
                "data/raw" in name or name.endswith(".env") for name in snapshot.namelist()
            )
    result = CliRunner().invoke(app, ["verificar-experimento", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "synthetic-v1" in result.output
    mlflow.set_tracking_uri(None)
