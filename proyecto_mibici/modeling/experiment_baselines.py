"""Run the P1, P2, and P3 baselines in one MLflow experiment."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from loguru import logger
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer

from proyecto_mibici.config import PROCESSED_DATA_DIR, PROJ_ROOT
from proyecto_mibici.config_p1 import load_settings
from proyecto_mibici.features_p1 import build_features
from proyecto_mibici.modeling.train_p1 import train

EXPERIMENT_NAME = "mibici-equipo25"
SOURCE_COLUMNS = (
    "Viaje_Id",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
    "mes_archivo",
)
OUTPUT_COLUMNS = (
    "viaje_id",
    "inicio_viaje",
    "fin_viaje",
    "origen_id",
    "destino_id",
    "mes_archivo",
)

app = typer.Typer(help="Ejecuta los baselines P1, P2 y P3 en un solo experimento MLflow.")


def prepare_supervised_input(
    source_path: Path,
    output_path: Path,
    *,
    start_month: str = "2023-10",
    end_month: str = "2026-09",
    batch_size: int = 500_000,
) -> int:
    """Project the audited raw Parquet into the private-data-free P1/P2 input."""
    if batch_size <= 0:
        raise ValueError("batch-size debe ser un entero positivo.")
    if not source_path.is_file():
        raise FileNotFoundError(f"No existe el Parquet de entrada: {source_path}")
    if start_month > end_month:
        raise ValueError("start-month debe ser anterior o igual a end-month.")

    dataset = ds.dataset(source_path, format="parquet")
    available = set(dataset.schema.names)
    missing = sorted(set(SOURCE_COLUMNS) - available)
    if missing:
        raise ValueError(f"Faltan columnas en el Parquet de entrada: {missing}")

    month = ds.field("mes_archivo")
    scanner = dataset.scanner(
        columns=list(SOURCE_COLUMNS),
        filter=(month >= start_month) & (month <= end_month),
        batch_size=batch_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    writer: pq.ParquetWriter | None = None

    with tempfile.TemporaryDirectory(prefix="mibici-p1p2-", dir=output_path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / output_path.name
        try:
            for batch in scanner.to_batches():
                table = pa.Table.from_batches([batch]).rename_columns(list(OUTPUT_COLUMNS))
                if writer is None:
                    writer = pq.ParquetWriter(temporary_path, table.schema, compression="snappy")
                writer.write_table(table)
                row_count += table.num_rows
        finally:
            if writer is not None:
                writer.close()

        if row_count == 0:
            raise ValueError("No hay viajes dentro del rango solicitado.")
        os.replace(temporary_path, output_path)

    logger.info("Proyectados {} viajes sin datos personales a {}", f"{row_count:,}", output_path)
    return row_count


@app.command()
def main(
    raw_parquet: Path = PROCESSED_DATA_DIR / "raw_data.parquet",
    station_catalog: Path = PROJ_ROOT / "data/raw/nomenclatura_2026_09.csv",
    tracking_database: Path = PROJ_ROOT / "mlruns/mibici-equipo25/tracking.db",
    tracking_artifacts: Path = PROJ_ROOT / "mlartifacts/mibici-equipo25",
    batch_size: int = 500_000,
    candidates_only: bool = typer.Option(
        False,
        "--candidates-only",
        help="Registra solo candidatos P1/P2 para compararlos con los baselines existentes.",
    ),
) -> None:
    """Build supervised features, then register baselines or P1/P2 candidates."""
    raw_parquet = raw_parquet.expanduser().resolve()
    station_catalog = station_catalog.expanduser().resolve()
    tracking_database = tracking_database.expanduser().resolve()
    tracking_artifacts = tracking_artifacts.expanduser().resolve()
    if not station_catalog.is_file():
        raise FileNotFoundError(f"No existe el catálogo de estaciones: {station_catalog}")

    supervised_parquet = PROJ_ROOT / "data/interim/viajes_p1_p2.parquet"
    prepare_supervised_input(raw_parquet, supervised_parquet, batch_size=batch_size)

    tracking_database.parent.mkdir(parents=True, exist_ok=True)
    tracking_artifacts.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{tracking_database.as_posix()}"
    settings = load_settings(
        overrides={
            "project.team": "equipo25",
            "project.experiment": EXPERIMENT_NAME,
            "data.trips_parquet": str(supervised_parquet),
            "data.stations_csv": str(station_catalog),
            "tracking.database": str(tracking_database),
            "tracking.artifacts": str(tracking_artifacts),
        }
    )

    build_features(settings, trips_path=supervised_parquet, stations_path=station_catalog)
    p1_p2_results = train(
        settings,
        questions=("P1", "P2"),
        candidates_only=candidates_only,
    )
    run_kind = "candidatos" if candidates_only else "baseline"
    logger.success("Runs {} P1/P2 registrados: {}", run_kind, len(p1_p2_results))

    if candidates_only:
        logger.success("Los candidatos P1/P2 quedaron en {}", EXPERIMENT_NAME)
        return

    subprocess.run(
        [
            sys.executable,
            "-m",
            "proyecto_mibici.modeling.experiment_p3",
            "--parquet-path",
            str(raw_parquet),
            "--tracking-uri",
            tracking_uri,
            "--batch-size",
            str(batch_size),
        ],
        cwd=PROJ_ROOT,
        check=True,
        env={**os.environ, "PYTHONPATH": str(PROJ_ROOT)},
    )
    logger.success("Los baselines P1, P2 y P3 quedaron en {}", EXPERIMENT_NAME)


if __name__ == "__main__":
    app()
