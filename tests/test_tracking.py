"""La ejecución de Projects debe reutilizar un padre y registrar un hijo por pipeline."""

import json
import shutil
import subprocess
import zipfile

import mlflow
import pytest

from proyecto_mibici import tracking
from proyecto_mibici.config import ProjectConfig
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
    for summary in report["runs"]:
        run = client.get_run(summary["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.tags["mlflow.parentRunId"] == parent.info.run_id
        assert run.data.tags["question"] == summary["question"]
        assert run.data.tags["git_commit"]
        assert run.data.tags["git_worktree_dirty"] in {"true", "false"}
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
    mlflow.set_tracking_uri(None)


def test_git_queries_fail_clearly_when_git_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="git no está disponible"):
        tracking._git(tmp_path, "rev-parse", "HEAD")


@pytest.mark.skipif(shutil.which("git") is None, reason="requiere git")
def test_regenerated_manifest_does_not_mark_worktree_dirty(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path, check=True, capture_output=True,
        )

    (tmp_path / "data").mkdir()
    manifest = tmp_path / "data" / "data_manifest.json"
    manifest.write_text("{}")
    (tmp_path / "MLproject").write_text("name: x\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")

    manifest.write_text('{"generado_en": "otra máquina"}')
    assert tracking.worktree_dirty(tmp_path) is False
    (tmp_path / "MLproject").write_text("name: y\n")
    assert tracking.worktree_dirty(tmp_path) is True
