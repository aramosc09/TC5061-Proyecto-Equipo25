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
    CANDIDATE_FIGURES_DIR,
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
    """Crea runs padre e hijos con etiquetas de filtrado, parámetros y artefactos completos."""

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

    def _start_parent(self, experiment_id: str):
        """Reutiliza el run creado por MLflow Projects o abre uno nuevo en el experimento."""
        project_run = os.environ.pop("MLFLOW_RUN_ID", None)
        if (
            project_run
            and mlflow.tracking.MlflowClient().get_run(project_run).info.experiment_id
            != experiment_id
        ):
            raise ValueError("El run de MLflow Projects debe pertenecer a mibici-equipo25.")
        return mlflow.start_run(run_id=project_run, experiment_id=experiment_id)

    def _context(self, manifest: dict, output_root: Path) -> tuple[dict, dict, dict, Path]:
        """Etiquetas, parámetros e identificación del código comunes a cualquier run."""
        output_root.mkdir(parents=True, exist_ok=True)
        source, archive = SourceSnapshot().create(output_root)
        tags = {"team": "25", "data_version": manifest["data_version"], **source}
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
        }
        return tags, parameters, source, archive

    def _log_reproducibility(self, archive: Path) -> None:
        mlflow.log_artifact(str(archive), artifact_path="reproducibilidad")
        mlflow.log_artifact(str(self.config.manifest_path), artifact_path="reproducibilidad")

    def _log_result(self, run_id: str, result, parameters: dict, output_dir: Path) -> dict:
        """Registra parámetros, métricas finitas, artefactos y un resumen JSON del run."""
        mlflow.log_params({**parameters, **result.parameters})
        metrics = {name: float(value) for name, value in result.metrics.items()}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Las métricas indefinidas deben omitirse, no registrarse como NaN.")
        mlflow.log_metrics(metrics)
        for path in result.artifacts:
            mlflow.log_artifact(str(path), artifact_path="resultados")
        summary = {
            "question": result.question,
            "run_id": run_id,
            "model": result.model_name,
            "parameters": {**parameters, **result.parameters},
            "metrics": metrics,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "resumen_run.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        mlflow.log_artifact(str(summary_path), artifact_path="resultados")
        return summary

    def run(
        self,
        repository,
        pipelines,
        *,
        parent_kind: str = "baseline_suite",
        extra_parameters: dict | None = None,
    ) -> dict:
        """Registra uno o varios baselines bajo el run padre creado por MLflow Projects."""
        manifest = repository.verify_base()
        experiment_id = self.configure()
        with self._start_parent(experiment_id) as parent:
            parent_id = parent.info.run_id
            output_root = self.config.path(BASELINE_FIGURES_DIR) / parent_id
            tags, parameters, source, archive = self._context(manifest, output_root)
            parameters.update(extra_parameters or {})
            mlflow.set_tags(
                {
                    **tags,
                    "question": "ALL",
                    "run_kind": parent_kind,
                    "model_family": "baseline",
                    "stage_of_work": "baseline",
                    "mlflow.runName": f"baselines-{self.config.start_month}-{self.config.end_month}",
                }
            )
            mlflow.log_params(parameters)
            self._log_reproducibility(archive)
            completed = []
            for pipeline in pipelines:
                question = type(pipeline).__name__.removeprefix("Pipeline")
                with mlflow.start_run(
                    run_name=f"{question}-baseline-{self.config.start_month}-{self.config.end_month}",
                    experiment_id=experiment_id,
                    nested=True,
                ) as child:
                    mlflow.set_tags(
                        {
                            **tags,
                            "question": question,
                            "run_kind": "baseline",
                            "stage_of_work": "baseline",
                        }
                    )
                    result = pipeline.run(repository, output_root / question)
                    if result.question != question:
                        raise ValueError("La pregunta del resultado no coincide con el pipeline.")
                    if not result.artifacts:
                        raise ValueError("Cada pregunta debe generar al menos un artefacto.")
                    mlflow.set_tag("model_family", result.model_name)
                    completed.append(
                        self._log_result(
                            child.info.run_id, result, parameters, output_root / question
                        )
                    )
                    self._log_reproducibility(archive)
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

    def _log_candidate_model(self, result, manifest: dict, question: str) -> str | None:
        """Guarda candidato ajustado + firma y muestra de features agregadas.

        Los tests de contratos que generan un CandidateResult sintético sin estimador
        pueden omitir esta etapa. Los candidatos reales siempre deben proveer ambos.
        """
        if result.estimator is None:
            return None
        if result.input_example is None or result.input_example.empty:
            raise ValueError("El candidato debe aportar input_example para inferir su firma.")
        import mlflow.sklearn
        from mlflow.models import infer_signature

        example = result.input_example
        signature = infer_signature(example, result.estimator.predict(example))
        info = mlflow.sklearn.log_model(
            sk_model=result.estimator,
            name="model",
            skops_trusted_types=["sklearn.compose._column_transformer._RemainderColsList"],
            signature=signature,
            input_example=example,
            metadata={
                "question": question.upper(),
                "data_version": manifest["data_version"],
                "task": "pressure_proxy_not_stockout" if question == "p2" else question,
            },
        )
        mlflow.set_tag("candidate_model_uri", info.model_uri)
        mlflow.set_tag("candidate_model_signed", "true")
        return info.model_uri

    def candidate(
        self,
        manifest: dict,
        question: str,
        model: str,
        family: str,
        train,
        extra_parameters: dict,
    ) -> dict:
        """Registra un candidato en un solo run; train(output_dir) devuelve CandidateResult."""
        experiment_id = self.configure()
        with self._start_parent(experiment_id) as run:
            output_root = self.config.path(CANDIDATE_FIGURES_DIR) / run.info.run_id
            tags, parameters, _, archive = self._context(manifest, output_root)
            mlflow.set_tags(
                {
                    **tags,
                    "question": question.upper(),
                    "run_kind": "candidate",
                    "model_family": family,
                    "stage_of_work": "candidate",
                    "mlflow.runName": f"{question}-{model}-candidate-"
                    f"{self.config.start_month}-{self.config.end_month}",
                }
            )
            output_dir = output_root / question.upper()
            result = train(output_dir)
            # MLflow Projects ya registró el valor pedido (p. ej. 'default'), que no puede cambiarse.
            requested = mlflow.get_run(run.info.run_id).data.params
            result.parameters = {
                f"resolved_{key}" if key in requested else key: value
                for key, value in result.parameters.items()
            }
            summary = self._log_result(
                run.info.run_id, result, {**parameters, **extra_parameters}, output_dir
            )
            # El manifiesto y la configuración también quedan en carpetas explícitas.
            mlflow.log_artifact(str(self.config.manifest_path), artifact_path="data")
            mlflow.log_dict(
                {"config": self.config.parameters(), "question": question, "model": model,
                 "resolved_parameters": result.parameters},
                "config/parametros_ejecucion.json",
            )
            self._log_candidate_model(result, manifest, question)
            self._log_reproducibility(archive)
        return {**summary, "result": result}

    def search(
        self,
        manifest: dict,
        question: str,
        model: str,
        family: str,
        trials: list[dict],
        train,
        criterion: tuple[str, str],
        extra_parameters: dict,
    ) -> dict:
        """Run padre por búsqueda y un hijo por combinación; elige solo con validación."""
        metric, direction = criterion
        if not metric.startswith("validation_"):
            raise ValueError("La búsqueda solo puede seleccionar con métricas de validación.")
        experiment_id = self.configure()
        stem = f"{question}-{model}-search-{self.config.start_month}-{self.config.end_month}"
        with self._start_parent(experiment_id) as parent:
            parent_id = parent.info.run_id
            output_root = self.config.path(CANDIDATE_FIGURES_DIR) / parent_id
            tags, parameters, _, archive = self._context(manifest, output_root)
            parameters.update(extra_parameters)
            common_tags = {
                **tags,
                "question": question.upper(),
                "model_family": family,
                "stage_of_work": "search",
            }
            mlflow.set_tags(
                {**common_tags, "run_kind": "search", "search_role": "parent", "mlflow.runName": stem}
            )
            mlflow.log_params(
                {
                    **parameters,
                    "model": model,
                    "n_trials": len(trials),
                    "search_space": json.dumps(trials, sort_keys=True),
                    "selection_metric": metric,
                    "selection_direction": direction,
                }
            )
            self._log_reproducibility(archive)
            completed = []
            best = None
            for index, params in enumerate(trials, start=1):
                with mlflow.start_run(
                    run_name=f"{stem}-trial-{index:02d}", experiment_id=experiment_id, nested=True
                ) as child:
                    mlflow.set_tags(
                        {
                            **common_tags,
                            "run_kind": "search_trial",
                            "search_role": "trial",
                            "trial_index": str(index),
                        }
                    )
                    trial_dir = output_root / f"trial-{index:02d}"
                    summary = self._log_result(
                        child.info.run_id, train(params, trial_dir), parameters, trial_dir
                    )
                    summary["trial_parameters"] = params
                    completed.append(summary)
                value = summary["metrics"].get(metric)
                if value is not None and (
                    best is None or (value < best[0] if direction == "min" else value > best[0])
                ):
                    best = (value, summary)
            if best is None:
                raise ValueError(f"Ningún trial produjo {metric}.")
            value, winner = best
            mlflow.set_tag("best_trial_run_id", winner["run_id"])
            mlflow.log_params(
                {f"best_{key}": item for key, item in winner["trial_parameters"].items()}
            )
            mlflow.log_metric(f"best_{metric}", value)
            report = {
                "experiment": EXPERIMENT_NAME,
                "parent_run_id": parent_id,
                "question": question.upper(),
                "model": model,
                "selection_metric": metric,
                "best_value": value,
                "best_trial_run_id": winner["run_id"],
                "best_parameters": winner["trial_parameters"],
                "trials": completed,
            }
            report_path = output_root / "resumen_busqueda.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            mlflow.log_artifact(str(report_path), artifact_path="resultados")
        return report
