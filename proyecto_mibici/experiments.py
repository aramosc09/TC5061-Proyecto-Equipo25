"""Entry points reproducibles de ingesta, features y baselines P1/P2/P3."""

from dataclasses import replace
import json
import os
from pathlib import Path

from loguru import logger
import mlflow
import typer

from proyecto_mibici.config import (
    CANDIDATE_MAX_TRAIN_ROWS,
    CANDIDATE_MODELS,
    EXPERIMENT_NAME,
    PROJ_ROOT,
    RAW_DATA_DIR,
    SEARCH_MAX_TRAIN_ROWS,
    ProjectConfig,
)
from proyecto_mibici.dataset import ejecutar_ingesta
from proyecto_mibici.features import RepositorioParquet
from proyecto_mibici.modeling.base import resolve_hyperparameters
from proyecto_mibici.modeling.datasets import (
    DatosSupervisados,
    PerfilesP3,
    TablaSupervisada,
    construir_perfiles,
)
from proyecto_mibici.modeling.p1_candidates import CandidatosP1
from proyecto_mibici.modeling.p2_candidates import CandidatosP2
from proyecto_mibici.modeling.p3_candidates import CandidatosP3
from proyecto_mibici.modeling.search import criterion, expand_grid
from proyecto_mibici.modeling.supervised_baselines import PipelineP1, PipelineP2
from proyecto_mibici.tracking import RegistroExperimentos

app = typer.Typer(pretty_exceptions_show_locals=False)
CANDIDATE_RUNNERS = {"p1": CandidatosP1, "p2": CandidatosP2, "p3": CandidatosP3}
# Parámetro del CLI -> argumento del estimador.
HYPERPARAMETER_NAMES = {
    "alpha": "alpha",
    "c": "C",
    "class_weight": "class_weight",
    "covariance_type": "covariance_type",
    "l2_regularization": "l2_regularization",
    "learning_rate": "learning_rate",
    "linkage": "linkage",
    "max_depth": "max_depth",
    "max_features": "max_features",
    "max_iter": "max_iter",
    "max_leaf_nodes": "max_leaf_nodes",
    "max_samples": "max_samples",
    "min_samples_leaf": "min_samples_leaf",
    "n_clusters": "n_clusters",
    "n_estimators": "n_estimators",
    "n_init": "n_init",
    "pca_components": "pca_components",
}


def optional_rows(value: str) -> int | None:
    """Convierte el sentinela 'default' del MLproject en None (default de config.py)."""
    if value == "default":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise typer.BadParameter("max_train_rows debe ser un entero o 'default'.") from error


def configuration(
    root: Path, start_month: str, end_month: str, train_end: str, validation_end: str, seed: int
) -> ProjectConfig:
    """Resuelve sentinelas del MLproject desde una única fuente de defaults."""
    changes = {"root": root.resolve()}
    for key, value in (
        ("start_month", start_month),
        ("end_month", end_month),
        ("train_end", train_end),
        ("validation_end", validation_end),
    ):
        if value != "default":
            changes[key] = value
    if seed != -1:
        changes["seed"] = seed
    config = replace(ProjectConfig(), **changes)
    config.validate()
    return config


class IngestorMiBici:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def run(self, *, offline: bool = False) -> dict:
        start_year, start_month = map(int, self.config.start_month.split("-"))
        end_year, end_month = map(int, self.config.end_month.split("-"))
        return ejecutar_ingesta(
            anio_inicio=start_year,
            mes_inicio=start_month,
            anio_fin=end_year,
            mes_fin=end_month,
            input_path=self.config.path(RAW_DATA_DIR),
            parquet_path=self.config.trips_path,
            estaciones_path=self.config.stations_path,
            manifiesto_path=self.config.manifest_path,
            sin_descarga=offline,
        )


class ProyectoMiBici:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.repository = RepositorioParquet(config)
        self.registry = RegistroExperimentos(config)

    def experiment(self) -> dict:
        from proyecto_mibici.modeling.p3_baseline import PipelineP3

        self.repository.prepare()
        return self.registry.run(
            self.repository,
            [
                PipelineP1(self.config),
                PipelineP2(self.config),
                PipelineP3(self.config),
            ],
        )

    def train(
        self, question: str, model: str, hyperparameters: dict, max_train_rows: int | None = None
    ) -> dict:
        """Ejecuta el baseline o un candidato; los hiperparámetros se validan contra el modelo."""
        question = question.lower()
        if question not in CANDIDATE_RUNNERS:
            raise ValueError("question debe ser p1, p2 o p3.")
        if model != "baseline":
            return self.candidate(question, model, hyperparameters, max_train_rows)
        if hyperparameters or max_train_rows is not None:
            raise ValueError("El baseline no acepta hiperparámetros.")
        from proyecto_mibici.modeling.p3_baseline import PipelineP3

        pipeline = {"p1": PipelineP1, "p2": PipelineP2, "p3": PipelineP3}[question](self.config)
        self.repository.prepare()
        return self.registry.run(
            self.repository,
            [pipeline],
            parent_kind="baseline_training",
            extra_parameters={"requested_question": question, "requested_model": model},
        )

    def _candidate_inputs(self, question: str, max_train_rows: int | None, default: int):
        """Prepara datos de una pregunta y los parámetros comunes de linaje del candidato."""
        manifest = self.repository.verify_base()
        self.repository.prepare()
        if question == "p3":
            frames, coverage = construir_perfiles(self.repository, self.config)
            return manifest, PerfilesP3(frames, coverage), 0, {}
        table = TablaSupervisada(self.config)
        metadata = table.prepare(self.repository, manifest["data_version"])
        rows = default if max_train_rows is None else max_train_rows
        if rows < 0:
            raise ValueError("max_train_rows debe ser 0 (todas) o positivo.")
        lineage = {
            "supervised_table_sha256": metadata["sha256"],
            "supervised_table_rows": json.dumps(metadata["rows"], sort_keys=True),
        }
        return manifest, DatosSupervisados(table, metadata), rows, lineage

    def candidate(
        self, question: str, model: str, hyperparameters: dict, max_train_rows: int | None
    ) -> dict:
        """Entrena un candidato con train y lo evalúa en validación y prueba (informativa)."""
        params = resolve_hyperparameters(question, model, hyperparameters)
        manifest, data, rows, lineage = self._candidate_inputs(
            question, max_train_rows, CANDIDATE_MAX_TRAIN_ROWS
        )
        runner = CANDIDATE_RUNNERS[question](self.config)
        return self.registry.candidate(
            manifest,
            question,
            model,
            CANDIDATE_MODELS[question][model]["family"],
            lambda output_dir: runner.run(
                data, model, params, output_dir, max_train_rows=rows, evaluate_test=True
            ),
            {"requested_question": question, "requested_model": model, **lineage},
        )

    def search(self, question: str, model: str, max_train_rows: int | None = None) -> dict:
        """Búsqueda nested: los trials no evalúan prueba y se eligen con validación."""
        question = question.lower()
        if question not in CANDIDATE_RUNNERS:
            raise ValueError("question debe ser p1, p2 o p3.")
        trials = expand_grid(question, model)
        manifest, data, rows, lineage = self._candidate_inputs(
            question, max_train_rows, SEARCH_MAX_TRAIN_ROWS
        )
        runner = CANDIDATE_RUNNERS[question](self.config)
        return self.registry.search(
            manifest,
            question,
            model,
            CANDIDATE_MODELS[question][model]["family"],
            trials,
            lambda params, output_dir: runner.run(
                data, model, params, output_dir, max_train_rows=rows, evaluate_test=False
            ),
            criterion(question),
            {"requested_question": question, "requested_model": model, **lineage},
        )

    def execute_project(self):
        """Ejecuta MLproject reutilizando el entorno bloqueado, sin Conda ni Docker."""
        experiment_id = self.registry.configure()
        # En Windows (cmd /c) una ruta absoluta con espacios se parte en varios argumentos.
        try:
            root = os.path.relpath(self.config.root, PROJ_ROOT)
        except ValueError:
            root = str(self.config.root)
        return mlflow.projects.run(
            uri=str(PROJ_ROOT),
            entry_point="main",
            env_manager="uv",
            experiment_id=experiment_id,
            synchronous=True,
            parameters={
                "root": root,
                "start_month": self.config.start_month,
                "end_month": self.config.end_month,
                "train_end": self.config.train_end,
                "validation_end": self.config.validation_end,
                "seed": self.config.seed,
            },
        )


@app.command()
def ingesta(
    root: Path = PROJ_ROOT,
    start_month: str = "default",
    end_month: str = "default",
    train_end: str = "default",
    validation_end: str = "default",
    seed: int = -1,
    offline: bool = False,
):
    config = configuration(root, start_month, end_month, train_end, validation_end, seed)
    manifest = IngestorMiBici(config).run(offline=offline)
    logger.success("Base lista: {}", manifest["data_version"])


@app.command()
def features(
    root: Path = PROJ_ROOT,
    start_month: str = "default",
    end_month: str = "default",
    train_end: str = "default",
    validation_end: str = "default",
    seed: int = -1,
):
    config = configuration(root, start_month, end_month, train_end, validation_end, seed)
    report = RepositorioParquet(config).prepare()
    logger.success("Features verificadas: {}", report["data_version"])


@app.command()
def experimento(
    root: Path = PROJ_ROOT,
    start_month: str = "default",
    end_month: str = "default",
    train_end: str = "default",
    validation_end: str = "default",
    seed: int = -1,
):
    config = configuration(root, start_month, end_month, train_end, validation_end, seed)
    report = ProyectoMiBici(config).experiment()
    logger.success("Baselines registrados: {}", report["parent_run_id"])
    for run in report["runs"]:
        logger.info("{}: {}", run["question"], run["run_id"])


@app.command()
def train(
    question: str = "p1",
    model: str = "baseline",
    max_train_rows: str = "default",
    alpha: str = "default",
    c: str = typer.Option("default", "--c"),
    class_weight: str = "default",
    covariance_type: str = "default",
    l2_regularization: str = "default",
    learning_rate: str = "default",
    linkage: str = "default",
    max_depth: str = "default",
    max_features: str = "default",
    max_iter: str = "default",
    max_leaf_nodes: str = "default",
    max_samples: str = "default",
    min_samples_leaf: str = "default",
    n_clusters: str = "default",
    n_estimators: str = "default",
    n_init: str = "default",
    pca_components: str = "default",
):
    """Entrena una pregunta; cada hiperparámetro en 'default' toma el valor de config.py."""
    values = locals()
    overrides = {
        estimator_name: values[name]
        for name, estimator_name in HYPERPARAMETER_NAMES.items()
        if values[name] != "default"
    }
    report = ProyectoMiBici(ProjectConfig()).train(
        question, model, overrides, optional_rows(max_train_rows)
    )
    run = report["runs"][0] if "runs" in report else report
    logger.success("{} registrado: {}", run["question"], run["run_id"])


@app.command()
def search(question: str = "p1", model: str = "hgb", max_train_rows: str = "default"):
    """Búsqueda de hiperparámetros como nested runs con la rejilla de config.py."""
    report = ProyectoMiBici(ProjectConfig()).search(question, model, optional_rows(max_train_rows))
    logger.success(
        "Búsqueda {}: mejor trial {} ({}={:.4f}) con {}",
        report["parent_run_id"],
        report["best_trial_run_id"],
        report["selection_metric"],
        report["best_value"],
        report["best_parameters"],
    )


@app.command()
def ejecutar_proyecto(
    root: Path = PROJ_ROOT,
    start_month: str = "default",
    end_month: str = "default",
    train_end: str = "default",
    validation_end: str = "default",
    seed: int = -1,
):
    config = configuration(root, start_month, end_month, train_end, validation_end, seed)
    run = ProyectoMiBici(config).execute_project()
    logger.success("MLflow Projects completado: {}", run.run_id)


@app.command()
def verificar_experimento(root: Path = PROJ_ROOT):
    """Comprueba runs FINISHED, parámetros, métricas, etiquetas y artefactos de cada pregunta."""
    config = replace(ProjectConfig(), root=root.resolve())
    registry = RegistroExperimentos(config)
    experiment_id = registry.configure()
    client = mlflow.tracking.MlflowClient()
    parents = client.search_runs(
        [experiment_id],
        "tags.run_kind = 'baseline_suite' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not parents:
        raise ValueError("No existe una ejecución base completa.")
    parent_id = parents[0].info.run_id
    runs = client.search_runs([experiment_id], f"tags.mlflow.parentRunId = '{parent_id}'")
    if {run.data.tags.get("question") for run in runs} != {"P1", "P2", "P3"} or len(runs) != 3:
        raise ValueError("El padre debe contener exactamente un baseline por pregunta.")
    report = []
    required_params = {
        "data_version",
        "start_month",
        "end_month",
        "seed",
        "model",
        "source_tree_sha256",
    }
    for run in runs:
        if run.info.status != "FINISHED" or not required_params.issubset(run.data.params):
            raise ValueError("Run incompleto o sin parámetros de reproducibilidad.")
        if not all(run.data.tags.get(tag) for tag in ("team", "question", "git_commit")):
            raise ValueError("Faltan etiquetas obligatorias.")
        if not any(name.startswith("test_") for name in run.data.metrics):
            raise ValueError("Faltan métricas de prueba.")
        artifacts = client.list_artifacts(run.info.run_id, "resultados")
        if not any(item.path.endswith(".png") for item in artifacts):
            raise ValueError("Falta un artefacto gráfico del baseline.")
        report.append(
            {
                "question": run.data.tags["question"],
                "run_id": run.info.run_id,
                "data_version": run.data.params["data_version"],
                "status": run.info.status,
            }
        )
    typer.echo(
        json.dumps(
            {"experiment": EXPERIMENT_NAME, "parent_run_id": parent_id, "runs": report}, indent=2
        )
    )


if __name__ == "__main__":
    app()
