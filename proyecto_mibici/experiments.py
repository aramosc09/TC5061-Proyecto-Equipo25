"""Entry points reproducibles de ingesta, features y baselines P1/P2/P3."""

from dataclasses import replace
import json
from pathlib import Path

from loguru import logger
import mlflow
import typer

from proyecto_mibici.config import (
    EXPERIMENT_NAME,
    PROJ_ROOT,
    RAW_DATA_DIR,
    ProjectConfig,
)
from proyecto_mibici.dataset import ejecutar_ingesta
from proyecto_mibici.features import RepositorioParquet
from proyecto_mibici.modeling.supervised_baselines import PipelineP1, PipelineP2
from proyecto_mibici.tracking import RegistroExperimentos

app = typer.Typer(pretty_exceptions_show_locals=False)


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

    def execute_project(self):
        """Ejecuta MLproject reutilizando el entorno bloqueado, sin Conda ni Docker."""
        experiment_id = self.registry.configure()
        return mlflow.projects.run(
            uri=str(PROJ_ROOT),
            entry_point="main",
            env_manager="local",
            experiment_id=experiment_id,
            synchronous=True,
            parameters={
                "root": str(self.config.root),
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
