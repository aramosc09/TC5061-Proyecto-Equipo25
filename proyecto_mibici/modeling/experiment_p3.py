"""Run and track the P3 weekend-share clustering baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import silhouette_score
import typer

from proyecto_mibici.config import PROCESSED_DATA_DIR, PROJ_ROOT
from proyecto_mibici.features_p3 import (
    EVENT_COLUMNS,
    PROFILE_COLUMNS,
    aggregate_p3_events,
    build_p3_station_profiles,
)

EXPERIMENT_NAME = "mibici-equipo25"
FULL_START = "2023-10-01"
FULL_END = "2026-10-01"
TRAIN_START = "2023-10-01"
TRAIN_END = "2025-07-01"
VALIDATION_START = "2025-07-01"
VALIDATION_END = "2025-10-01"
TEST_START = "2025-10-01"
TEST_END = "2026-10-01"
PARQUET_COLUMNS = (
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
    "mes_archivo",
)
WEEKEND_FEATURES = tuple(
    column for column in PROFILE_COLUMNS if "_sabado_" in column or "_domingo_" in column
)

app = typer.Typer()


def weekend_activity_share(profiles: pd.DataFrame) -> pd.Series:
    """Return each station's share of normalized activity occurring on weekends."""
    missing_columns = sorted(set(WEEKEND_FEATURES) - set(profiles.columns))
    if missing_columns:
        raise ValueError(f"Faltan features de fin de semana: {missing_columns}")
    if profiles.empty:
        raise ValueError("No hay perfiles de estaciones para calcular la proporción.")
    return profiles.loc[:, WEEKEND_FEATURES].sum(axis=1).rename("proporcion_fin_semana")


def assign_weekend_median_baseline(
    profiles: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    """Assign stations above the training median to group 1 and the rest to group 0."""
    shares = weekend_activity_share(profiles)
    return shares.gt(threshold).astype("int8").rename("grupo")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJ_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _worktree_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJ_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _read_p3_event_counts(parquet_path: Path, batch_size: int) -> tuple[pd.DataFrame, pd.Series]:
    """Stream only P3 columns and return monthly station/time-slot event counts."""
    if not parquet_path.is_file():
        raise FileNotFoundError(f"No existe el Parquet requerido: {parquet_path}")

    dataset = ds.dataset(parquet_path, format="parquet")
    month_column = ds.field("mes_archivo")
    scanner = dataset.scanner(
        columns=list(PARQUET_COLUMNS),
        filter=(month_column >= "2023-10") & (month_column <= "2026-09"),
        batch_size=batch_size,
    )
    event_batches: list[pd.DataFrame] = []
    audit_totals: pd.Series | None = None

    for record_batch in scanner.to_batches():
        trips = record_batch.to_pandas()
        event_counts, audit = aggregate_p3_events(trips, FULL_START, FULL_END)
        if not event_counts.empty:
            event_batches.append(event_counts)
        audit_totals = audit if audit_totals is None else audit_totals.add(audit, fill_value=0)

    if not event_batches:
        raise ValueError("El Parquet no contiene eventos P3 en la ventana acordada.")

    combined = pd.concat(event_batches, ignore_index=True)
    combined = (
        combined.groupby(list(EVENT_COLUMNS), observed=True, as_index=False)["viajes"]
        .sum()
        .astype({"viajes": "int64"})
    )
    if audit_totals is None:
        raise ValueError("No se pudo generar la auditoría de lectura del Parquet.")
    return combined, audit_totals.astype("int64")


def _profiles_for_period(
    event_counts: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_month = pd.Timestamp(start).strftime("%Y-%m")
    end_month = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m")
    selected = event_counts.loc[event_counts["mes"].between(start_month, end_month)]
    return build_p3_station_profiles(selected, start, end)


def _silhouette_or_none(profiles: pd.DataFrame, assignments: pd.Series) -> float | None:
    labels = assignments.reindex(profiles.index)
    if labels.isna().any() or labels.nunique() < 2 or labels.nunique() >= len(labels):
        return None
    return float(silhouette_score(profiles.to_numpy(), labels.to_numpy()))


def _log_cluster_counts(period: str, assignments: pd.Series) -> None:
    counts = assignments.value_counts()
    mlflow.log_metric(f"{period}_station_count", len(assignments))
    for group in (0, 1):
        mlflow.log_metric(f"{period}_cluster_{group}_stations", int(counts.get(group, 0)))


def _plot_baseline(
    training_shares: pd.Series,
    validation_shares: pd.Series,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(training_shares, bins=20, color="#5470a8", edgecolor="white")
    axes[0].axvline(threshold, color="#ba3c3c", linestyle="--", label="Mediana de entrenamiento")
    axes[0].set(
        title="Proporción de actividad en fin de semana",
        xlabel="Sábado y domingo / actividad total",
        ylabel="Estaciones de entrenamiento",
    )
    axes[0].legend()

    validation_order = validation_shares.sort_values()
    validation_groups = validation_order.gt(threshold).astype(int)
    colors = validation_groups.map({0: "#5470a8", 1: "#d28b26"})
    axes[1].scatter(
        np.arange(len(validation_order)),
        validation_order.to_numpy(),
        c=colors.to_numpy(),
        s=24,
    )
    axes[1].axhline(threshold, color="#ba3c3c", linestyle="--")
    axes[1].set(
        title="Asignación de estaciones de validación",
        xlabel="Estaciones ordenadas por proporción",
        ylabel="Proporción de actividad en fin de semana",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


@app.command()
def main(
    parquet_path: Path = PROCESSED_DATA_DIR / "raw_data.parquet",
    batch_size: int = 500_000,
    tracking_uri: str = "",
) -> None:
    """Run the P3 baseline and log its validation/test diagnostics to MLflow."""
    if batch_size <= 0:
        raise ValueError("batch-size debe ser un entero positivo.")

    data_path = parquet_path.expanduser().resolve()
    logger.info("Agregando los datos P3 desde {}", data_path)
    event_counts, audit = _read_p3_event_counts(data_path, batch_size)
    train_profiles, _ = _profiles_for_period(event_counts, TRAIN_START, TRAIN_END)
    validation_profiles, _ = _profiles_for_period(
        event_counts, VALIDATION_START, VALIDATION_END
    )
    test_profiles, _ = _profiles_for_period(event_counts, TEST_START, TEST_END)

    training_shares = weekend_activity_share(train_profiles)
    threshold = float(training_shares.median())
    training_assignments = assign_weekend_median_baseline(train_profiles, threshold)
    if training_assignments.nunique() != 2:
        raise ValueError(
            "La mediana de entrenamiento no produce dos grupos no vacíos; "
            "revise la utilidad de esta heurística antes de registrarla."
        )

    validation_shares = weekend_activity_share(validation_profiles)
    validation_assignments = assign_weekend_median_baseline(
        validation_profiles, threshold
    )
    test_assignments = assign_weekend_median_baseline(test_profiles, threshold)
    validation_silhouette = _silhouette_or_none(
        validation_profiles, validation_assignments
    )
    test_silhouette = _silhouette_or_none(test_profiles, test_assignments)

    default_tracking_uri = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri or default_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    git_commit = _source_commit()
    dirty_worktree = _worktree_dirty()
    source_digest = hashlib.sha256(
        b"".join(
            path.read_bytes()
            for path in (
                PROJ_ROOT / "proyecto_mibici/modeling/experiment_p3.py",
                PROJ_ROOT / "proyecto_mibici/features_p3.py",
            )
        )
    ).hexdigest()
    data_digest = _sha256(data_path)
    run_name = "p3-baseline-weekend-median"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "team": "equipo25",
                "question": "P3",
                "git_commit": git_commit,
                "git_worktree_dirty": str(dirty_worktree).lower(),
                "baseline": "weekend_share_median",
            }
        )
        mlflow.log_params(
            {
                "date_start_inclusive": FULL_START,
                "date_end_exclusive": FULL_END,
                "train_start": TRAIN_START,
                "train_end_exclusive": TRAIN_END,
                "validation_start": VALIDATION_START,
                "validation_end_exclusive": VALIDATION_END,
                "test_start": TEST_START,
                "test_end_exclusive": TEST_END,
                "data_file": str(data_path.relative_to(PROJ_ROOT))
                if data_path.is_relative_to(PROJ_ROOT)
                else data_path.name,
                "data_sha256": data_digest,
                "source_sha256": source_digest,
                "batch_size": batch_size,
                "weekend_days": "saturday,sunday",
                "baseline_threshold_train_median": threshold,
                "threshold_rule": "weekend_share > training_median => group_1",
                "silhouette_metric": "euclidean",
            }
        )
        mlflow.log_metric("training_valid_trip_count", int(audit["viajes_validos"]))
        mlflow.log_metric(
            "training_start_events_outside_window",
            int(audit["salidas_fuera_de_ventana"]),
        )
        mlflow.log_metric(
            "training_arrival_events_outside_window",
            int(audit["llegadas_fuera_de_ventana"]),
        )
        _log_cluster_counts("train", training_assignments)
        _log_cluster_counts("validation", validation_assignments)
        _log_cluster_counts("test", test_assignments)

        mlflow.log_metric(
            "validation_silhouette_defined", int(validation_silhouette is not None)
        )
        if validation_silhouette is not None:
            mlflow.log_metric("validation_silhouette", validation_silhouette)
        mlflow.log_metric("test_silhouette_defined", int(test_silhouette is not None))
        if test_silhouette is not None:
            mlflow.log_metric("test_silhouette", test_silhouette)

        with tempfile.TemporaryDirectory(prefix="mibici-p3-") as temporary_directory:
            artifact_path = Path(temporary_directory) / "p3-baseline-weekend-share.png"
            _plot_baseline(
                training_shares,
                validation_shares,
                threshold,
                artifact_path,
            )
            mlflow.log_artifact(str(artifact_path), artifact_path="figures")

        logger.success(
            "Run P3 registrado: experiment={}, run_id={}, validation_silhouette={}, "
            "test_silhouette={}",
            EXPERIMENT_NAME,
            run.info.run_id,
            validation_silhouette,
            test_silhouette,
        )


if __name__ == "__main__":
    app()
