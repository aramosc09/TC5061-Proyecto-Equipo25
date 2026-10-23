"""MLflow Tracking aislado de los cálculos; trazabilidad de código sin commit automático."""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile

import mlflow
import numpy as np

from proyecto_mibici.config import (
    ARTIFACTS_DIR,
    BASELINE_FIGURES_DIR,
    EXPERIMENT_NAME,
    PROJ_ROOT,
    ProjectConfig,
)
from proyecto_mibici.dataset import calcular_sha256


class SourceSnapshot:
    """Identifica HEAD y conserva exactamente el código ejecutado aunque no esté committed."""

    def __init__(self, root: Path = PROJ_ROOT):
        self.root = root

    def create(self, directory: Path) -> tuple[dict, Path]:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=self.root, text=True
            ).strip()
        )
        files = sorted((self.root / "proyecto_mibici").rglob("*.py"))
        files += [
            self.root / name
            for name in (
                "MLproject",
                "python_env.yaml",
                "requirements.txt",
                "pyproject.toml",
                "uv.lock",
                ".python-version",
            )
        ]
        fingerprint = hashlib.sha256()
        archive = directory / "codigo_ejecutado.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path in files:
                name = path.relative_to(self.root).as_posix()
                digest = calcular_sha256(path)
                fingerprint.update(f"{name}\0{digest}\n".encode())
                output.write(path, name)
        return {
            "git_commit": commit,
            "git_worktree_dirty": str(dirty).lower(),
            "source_tree_sha256": fingerprint.hexdigest(),
            "environment_lock_sha256": calcular_sha256(self.root / "uv.lock"),
            "source_version_kind": "working_tree_snapshot" if dirty else "git_commit",
        }, archive


class RegistroExperimentos:
    """Crea un run padre y un hijo por pregunta, con parámetros y artefactos completos."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def configure(self) -> str:
        """Configura el único experimento antes de que MLflow Projects cree su run padre."""
        artifacts = self.config.path(ARTIFACTS_DIR)
        artifacts.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.config.tracking_uri)
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        return (
            experiment.experiment_id
            if experiment
            else mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifacts.as_uri())
        )

    def run(
        self,
        repository,
        pipelines,
        *,
        parent_kind: str = "baseline_suite",
        extra_parameters: dict | None = None,
    ) -> dict:
        """Registra uno o varios pipelines bajo el run padre creado por MLflow Projects."""
        manifest = repository.verify_base()
        experiment_id = self.configure()
        project_run = os.environ.pop("MLFLOW_RUN_ID", None)
        if (
            project_run
            and mlflow.tracking.MlflowClient().get_run(project_run).info.experiment_id
            != experiment_id
        ):
            raise ValueError("El run de MLflow Projects debe pertenecer a mibici-equipo25.")
        with mlflow.start_run(run_id=project_run, experiment_id=experiment_id) as parent:
            parent_id = parent.info.run_id
            output_root = self.config.path(BASELINE_FIGURES_DIR) / parent_id
            output_root.mkdir(parents=True, exist_ok=True)
            source, archive = SourceSnapshot().create(output_root)
            tags = {"team": "25", **source}
            parameters = {
                **self.config.parameters(),
                "data_version": manifest["data_version"],
                "python_version": platform.python_version(),
                **{
                    f"version_{package}": version(package)
                    for package in ("mlflow", "pandas", "pyarrow", "numpy", "scikit-learn")
                },
                "base_rows_raw": manifest["resumen"]["n_viajes"],
                "base_rows_clean": manifest["salida"]["n_filas"],
                "base_sha256": manifest["salida"]["sha256"],
                "source_tree_sha256": source["source_tree_sha256"],
                "environment_lock_sha256": source["environment_lock_sha256"],
                **(extra_parameters or {}),
            }
            mlflow.set_tags(
                {
                    **tags,
                    "question": "ALL",
                    "run_kind": parent_kind,
                    "mlflow.runName": f"baselines-{self.config.start_month}-{self.config.end_month}",
                }
            )
            mlflow.log_params(parameters)
            mlflow.log_artifact(str(archive), artifact_path="reproducibilidad")
            mlflow.log_artifact(str(self.config.manifest_path), artifact_path="reproducibilidad")
            completed = []
            for pipeline in pipelines:
                question = type(pipeline).__name__.removeprefix("Pipeline")
                with mlflow.start_run(
                    run_name=f"{question}-baseline-{self.config.start_month}-{self.config.end_month}",
                    experiment_id=experiment_id,
                    nested=True,
                ) as child:
                    mlflow.set_tags({**tags, "question": question, "run_kind": "baseline"})
                    result = pipeline.run(repository, output_root / question)
                    if result.question != question:
                        raise ValueError("La pregunta del resultado no coincide con el pipeline.")
                    mlflow.log_params({**parameters, **result.parameters})
                    metrics = {name: float(value) for name, value in result.metrics.items()}
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise ValueError(
                            "Las métricas indefinidas deben omitirse, no registrarse como NaN."
                        )
                    if not result.artifacts:
                        raise ValueError("Cada pregunta debe generar al menos un artefacto.")
                    mlflow.log_metrics(metrics)
                    for path in result.artifacts:
                        mlflow.log_artifact(str(path), artifact_path="resultados")
                    mlflow.log_artifact(str(archive), artifact_path="reproducibilidad")
                    mlflow.log_artifact(
                        str(self.config.manifest_path), artifact_path="reproducibilidad"
                    )
                    summary = {
                        "question": question,
                        "run_id": child.info.run_id,
                        "model": result.model_name,
                        "parameters": {**parameters, **result.parameters},
                        "metrics": metrics,
                    }
                    summary_path = output_root / question / "resumen_run.json"
                    summary_path.write_text(
                        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
                    )
                    mlflow.log_artifact(str(summary_path), artifact_path="resultados")
                    completed.append(summary)
            report = {
                "experiment": EXPERIMENT_NAME,
                "parent_run_id": parent_id,
                "data_version": manifest["data_version"],
                "source": source,
                "runs": completed,
            }
            report_path = output_root / "resumen_experimento.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            mlflow.log_artifact(str(report_path), artifact_path="resultados")
        return report
