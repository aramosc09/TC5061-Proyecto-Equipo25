"""Validación de calidad de datos con Deepchecks y prechequeos reproducibles."""

from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd
import typer

from proyecto_mibici.config import PROJ_ROOT, REPORTS_DIR, ProjectConfig
from proyecto_mibici.features import RepositorioParquet
from proyecto_mibici.modeling.datasets import (
    DatosSupervisados,
    PerfilesP3,
    TablaSupervisada,
    construir_perfiles,
)

app = typer.Typer()

QUALITY_OUTPUT_DIR = REPORTS_DIR / "data_quality"
COMPARED_SPLITS = ("train", "test")
# P2 se valida sobre sus siete clases, no sobre la presión continua de la que se derivan.
DATASET_SPECS = {
    "p1": {"label": "salidas", "label_type": "regression"},
    "p2": {"label": "clase_presion", "label_type": "multiclass"},
    "p3": {},
}


def supervised_frame(data: DatosSupervisados, question: str, split: str, **sample) -> pd.DataFrame:
    """Arma features, hora y objetivo de un periodo tal como los recibe el candidato."""
    subset = data.split(question, split, **sample)
    frame = subset.X.copy()
    frame["hora"] = subset.hora
    frame[DATASET_SPECS[question]["label"]] = subset.y
    return frame


def load_splits(
    question: str, data: DatosSupervisados | PerfilesP3, *, max_rows: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve train y test sin identificadores de estación."""
    if question not in DATASET_SPECS:
        raise ValueError("question debe ser p1, p2 o p3.")
    if max_rows < 1:
        raise ValueError("max_rows debe ser positivo.")
    if question == "p3":
        frames = [data.frames[split].reset_index(drop=True) for split in COMPARED_SPLITS]
    else:
        frames = [
            supervised_frame(data, question, split, max_rows=max_rows, seed=seed)
            for split in COMPARED_SPLITS
        ]
    if any(frame.empty for frame in frames):
        raise ValueError("Train y test deben contener filas.")
    return frames[0], frames[1]


def preflight_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Resume integridad básica antes de ejecutar la suite de Deepchecks."""
    if list(train.columns) != list(test.columns):
        raise ValueError("Train y test deben tener las mismas columnas y en el mismo orden.")
    return {
        "columns": list(train.columns),
        "dtypes": {column: str(dtype) for column, dtype in train.dtypes.items()},
        "missing_values": {
            "train": int(train.isna().sum().sum()),
            "test": int(test.isna().sum().sum()),
        },
        "duplicate_rows": {
            "train": int(train.duplicated().sum()),
            "test": int(test.duplicated().sum()),
        },
        "rows": {"train": len(train), "test": len(test)},
    }


def dataset_arguments(question: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Argumentos de `deepchecks.tabular.Dataset`: tipo de tarea explícito y hora como fecha."""
    arguments = {"cat_features": [], **DATASET_SPECS[question]}
    if "hora" in frame.columns:
        arguments["datetime_name"] = "hora"
    return arguments


def run_deepchecks(
    question: str, train: pd.DataFrame, test: pd.DataFrame, output_dir: Path
) -> dict[str, Any]:
    """Ejecuta la suite tabular y guarda el resultado en JSON y HTML."""
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import full_suite
    except ImportError as error:
        raise RuntimeError(
            "Deepchecks no está instalado. Usa requirements-quality.txt con Python 3.12."
        ) from error

    datasets = [Dataset(frame, **dataset_arguments(question, frame)) for frame in (train, test)]
    suite_result = full_suite().run(*datasets)
    report = {
        "tool": "deepchecks",
        "version": version("deepchecks"),
        "question": question,
        "preflight": preflight_summary(train, test),
        "suite": json.loads(suite_result.to_json()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"deepchecks_{question}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    suite_result.save_as_html(str(output_dir / f"deepchecks_{question}.html"), as_widget=False)
    return report


@app.command()
def main(question: str = "p1", max_rows: int = 10_000, root: Path = PROJ_ROOT):
    question = question.lower()
    config = ProjectConfig(root=root)
    repository = RepositorioParquet(config)
    manifest = repository.verify_base()
    repository.prepare()
    if question == "p3":
        data = PerfilesP3(*construir_perfiles(repository, config))
    else:
        table = TablaSupervisada(config)
        data = DatosSupervisados(table, table.prepare(repository, manifest["data_version"]))
    train, test = load_splits(question, data, max_rows=max_rows, seed=config.seed)
    output_dir = config.path(QUALITY_OUTPUT_DIR)
    run_deepchecks(question, train, test, output_dir)
    logger.success("Reporte Deepchecks de {} en {}", question, output_dir)


if __name__ == "__main__":
    app()
