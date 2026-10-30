"""Validación de calidad de datos con Deepchecks y prechequeos reproducibles."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from proyecto_mibici.modeling.datasets import P1_FEATURES, P2_FEATURES, SPLITS

QUALITY_COLUMNS = {
    "p1": (P1_FEATURES, "salidas"),
    "p2": (P2_FEATURES, "presion"),
}


def load_supervised_splits(
    path: Path, question: str, *, max_rows: int = 10_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga train y test de la tabla causal sin incluir identificadores temporales."""
    try:
        features, target = QUALITY_COLUMNS[question]
    except KeyError as error:
        raise ValueError("question debe ser p1 o p2.") from error
    columns = ["conjunto", *features, target]
    frame = pd.read_parquet(path, columns=columns)
    train = frame.loc[frame["conjunto"] == SPLITS.index("train")].drop(columns="conjunto")
    test = frame.loc[frame["conjunto"] == SPLITS.index("test")].drop(columns="conjunto")
    if train.empty or test.empty:
        raise ValueError("La tabla supervisada debe contener filas de train y test.")
    if max_rows < 1:
        raise ValueError("max_rows debe ser positivo.")
    train = train.sample(min(max_rows, len(train)), random_state=42)
    test = test.sample(min(max_rows, len(test)), random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)


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


def run_deepchecks(
    train: pd.DataFrame, test: pd.DataFrame, output_path: Path
) -> dict[str, Any]:
    """Ejecuta la suite tabular y guarda resultados legibles en JSON."""
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import full_suite
    except ImportError as error:
        raise RuntimeError(
            "Deepchecks no está instalado. Usa requirements-quality.txt con Python 3.12."
        ) from error

    train_dataset = Dataset(train, label=train.columns[-1], cat_features=[])
    test_dataset = Dataset(test, label=test.columns[-1], cat_features=[])
    suite_result = full_suite().run(train_dataset, test_dataset)
    deepchecks = json.loads(suite_result.to_json())
    report = {
        "tool": "deepchecks",
        "version": _package_version("deepchecks"),
        "preflight": preflight_summary(train, test),
        "suite": deepchecks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _package_version(package: str) -> str:
    from importlib.metadata import version

    return version(package)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", choices=sorted(QUALITY_COLUMNS), default="p1")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/tabla_supervisada.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data_quality/deepchecks.json")
    )
    parser.add_argument("--max-rows", type=int, default=10_000)
    args = parser.parse_args()
    train, test = load_supervised_splits(args.input, args.question, max_rows=args.max_rows)
    run_deepchecks(train, test, args.output)
    print(f"Reporte Deepchecks escrito en {args.output}")


if __name__ == "__main__":
    main()
