"""Entrena los baselines y candidatos de P1/P2/P3 y los registra en MLflow."""

import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from proyecto_mibici.config_p1 import load_settings
from proyecto_mibici.dataset import obtener_version_datos
from proyecto_mibici.features_p1 import predictor_columns

app = typer.Typer(help="Entrena y registra los modelos de P1/P2/P3 en MLflow.")


def git_commit(root):
    """Commit que produjo la corrida; es una etiqueta obligatoria de la entrega."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
        )
    except FileNotFoundError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def tracking_uri(settings):
    database = settings.path("tracking", "database").resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + database.as_posix()


def configure_tracking(settings):
    mlflow.set_tracking_uri(tracking_uri(settings))
    name = settings.values["project"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        artifacts = settings.path("tracking", "artifacts").resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(name)


def model_pipeline(question, settings, candidate=False):
    """Preprocessing is fitted inside Pipeline, exclusively on training rows."""
    seed = settings.seed
    if question == "P1":
        estimator = (
            HistGradientBoostingRegressor(random_state=seed, max_iter=80)
            if candidate
            else DummyRegressor(strategy="mean")
        )
    elif question == "P2":
        estimator = (
            HistGradientBoostingClassifier(random_state=seed, max_iter=80, class_weight="balanced")
            if candidate
            else DummyClassifier(strategy="most_frequent", random_state=seed)
        )
    else:
        k = settings.values["models"]["clusters"]
        estimator = (
            GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, n_init=3)
            if candidate
            else KMeans(n_clusters=k, random_state=seed, n_init=10)
        )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def supervised_metrics(question, actual, predicted, prefix):
    """Name every metric by its evaluation partition and expose P2 class recall."""
    if question == "P1":
        metrics = {
            f"{prefix}_mae": float(mean_absolute_error(actual, predicted)),
            f"{prefix}_rmse": float(root_mean_squared_error(actual, predicted)),
        }
        denominator = float(np.abs(actual).sum())
        if denominator:
            metrics[f"{prefix}_wape"] = float(np.abs(actual - predicted).sum() / denominator)
        return metrics
    labels = ["balanced", "bike_pressure", "dock_pressure"]
    report = classification_report(
        actual, predicted, labels=labels, output_dict=True, zero_division=0
    )
    return {
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        f"{prefix}_macro_f1": float(
            f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)
        ),
        **{f"{prefix}_{label}_recall": float(report[label]["recall"]) for label in labels},
    }


def evaluate_supervised(question, pipeline, frame, columns, output, figure):
    target = "target_trips" if question == "P1" else "target_pressure"
    frame = frame.dropna(subset=[target])
    train = frame[frame.split == "train"]
    if train.empty or (question == "P2" and train[target].nunique() < 2):
        raise ValueError(f"Insufficient training examples/classes for {question}")
    pipeline.fit(train[columns], train[target])
    metrics, counts, predictions = {}, {}, {}
    for split in ("validation", "test"):
        subset = frame[frame.split == split]
        if subset.empty:
            raise ValueError(f"No {question} examples in {split}")
        predicted = pipeline.predict(subset[columns])
        metrics.update(supervised_metrics(question, subset[target].to_numpy(), predicted, split))
        counts[split] = len(subset)
        predictions[split] = (subset, predicted)
    counts["train"] = len(train)
    test, predicted = predictions["test"]
    fig, ax = plt.subplots(figsize=(8, 4))
    if question == "P1":
        comparison = pd.DataFrame(
            {"hour": test.target_hour, "actual": test[target], "predicted": predicted}
        )
        summary = comparison.groupby("hour")[["actual", "predicted"]].sum().resample("D").sum()
        summary.to_csv(output / "daily_test_predictions.csv")
        summary.plot(ax=ax)
        ax.set_ylabel("Trips across eligible stations")
        ax.set_title("P1 held-out daily demand")
    else:
        labels = ["balanced", "bike_pressure", "dock_pressure"]
        matrix = confusion_matrix(test[target], predicted, labels=labels)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
            output / "test_confusion_matrix.csv"
        )
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(3), labels, rotation=15)
        ax.set_yticks(range(3), labels)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set_xlabel("Predicted pressure proxy")
        ax.set_ylabel("Observed flow proxy")
        ax.set_title("P2 proxy classification (not observed inventory)")
    fig.tight_layout()
    fig.savefig(figure, dpi=140)
    plt.close(fig)
    return metrics, counts


def evaluate_clusters(pipeline, profiles, columns, settings, output, figure):
    k = settings.values["models"]["clusters"]
    if len(profiles) <= k:
        raise ValueError("P3 needs more stations than clusters")
    pipeline.fit(profiles[columns])
    labels = pipeline.predict(profiles[columns])
    scaled = pipeline[:-1].transform(profiles[columns])
    groups = np.unique(labels)
    if not 1 < len(groups) < len(profiles):
        raise ValueError("P3 produced degenerate clusters; inspect station profiles")
    score = silhouette_score(scaled, labels)
    summary = profiles[["station_id", "latitude", "longitude"]].assign(cluster=labels)
    summary.to_csv(output / "station_clusters.csv", index=False)
    profiles[columns].assign(cluster=labels).groupby("cluster").mean().to_csv(
        output / "cluster_profiles.csv"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    located = summary.dropna(subset=["latitude", "longitude"])
    for label, group in located.groupby("cluster"):
        ax.scatter(group.longitude, group.latitude, s=24, label=f"Cluster {int(label)}")
    ax.legend(title="Training profiles")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="P3 station groups (catalog coordinates)")
    fig.tight_layout()
    fig.savefig(figure, dpi=140)
    plt.close(fig)
    return {
        "descriptive_silhouette": float(score),
        "descriptive_clusters": int(len(groups)),
        **{
            f"descriptive_cluster_{int(label)}_size": int((labels == label).sum())
            for label in groups
        },
    }, {"descriptive": len(profiles), "mapped": len(located)}


def train(settings, questions=("P1",)):
    """Registra una corrida por pregunta y tipo de modelo con parametros, metricas y artefactos."""
    supervised_path = settings.path("features", "supervised")
    profiles_path = settings.path("features", "profiles")
    if not supervised_path.exists() or not profiles_path.exists():
        raise ValueError("Faltan las features; ejecuta primero `features`")
    meta_path = supervised_path.with_name("features_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    configure_tracking(settings)
    commit = git_commit(settings.root)
    data_version = obtener_version_datos(settings.path("data", "manifest"))
    window = [settings.values["data"]["start_month"], settings.values["data"]["end_month"]]

    frame = profiles = None
    results = []
    for question in questions:
        if question not in {"P1", "P2", "P3"}:
            raise ValueError(f"Pregunta desconocida: {question}")
        if question == "P3":
            if profiles is None:
                profiles = pd.read_parquet(profiles_path)
            columns = [c for c in profiles if c.startswith(("departures_", "arrivals_"))]
        else:
            if frame is None:
                frame = pd.read_parquet(supervised_path)
            # Los lags ya codifican la demanda; el id numerico no tiene orden natural.
            columns = [c for c in predictor_columns(frame) if c != "station_id"]
        for candidate in (
            [False, True] if settings.values["models"]["include_candidates"] else [False]
        ):
            pipeline = model_pipeline(question, settings, candidate)
            kind = "candidate" if candidate else "baseline"
            model_params = pipeline.named_steps["model"].get_params()
            tags = {
                "team": settings.values["project"]["team"],
                "question": question,
                "git_commit": commit,
                "target_semantics": "flow_pressure_proxy"
                if question == "P2"
                else "descriptive"
                if question == "P3"
                else "future_trip_count",
            }
            run_name = f"{question}-{kind}-{window[0]}-{window[1]}"
            with mlflow.start_run(run_name=run_name, tags=tags) as run:
                figure = settings.path("tracking", "figures") / f"{run.info.run_id}.png"
                figure.parent.mkdir(parents=True, exist_ok=True)
                mlflow.log_params(
                    {
                        "start_month": window[0],
                        "end_month": window[1],
                        "data_version": data_version,
                        "seed": settings.seed,
                        "train_end": settings.values["split"]["train_end"],
                        "validation_end": settings.values["split"]["validation_end"],
                        "pressure_threshold": meta.get("pressure_threshold"),
                        "p1_horizon_hours": settings.values["features"]["p1_horizon_hours"],
                        "p2_window_hours": settings.values["features"]["p2_window_hours"],
                        "capacity_snapshot": settings.values["features"]["capacity_snapshot"],
                        "model": type(pipeline.named_steps["model"]).__name__,
                        **{f"model_{k}": v for k, v in model_params.items()},
                    }
                )
                with tempfile.TemporaryDirectory() as temporary:
                    output = Path(temporary)
                    if question == "P3":
                        metrics, counts = evaluate_clusters(
                            pipeline, profiles, columns, settings, output, figure
                        )
                        example = profiles[columns].head(5)
                    else:
                        metrics, counts = evaluate_supervised(
                            question, pipeline, frame, columns, output, figure
                        )
                        example = frame[columns].head(5)
                    mlflow.log_metrics(metrics)
                    mlflow.log_artifacts(str(output), "tables")
                mlflow.sklearn.log_model(pipeline, "model", input_example=example)
                mlflow.log_artifact(str(figure), "figures")
                mlflow.log_dict(settings.values, "configuration.json")
                results.append(
                    {
                        "run_id": run.info.run_id,
                        "question": question,
                        "kind": kind,
                        "model": type(pipeline.named_steps["model"]).__name__,
                        "data_version": data_version,
                        "rows": counts,
                        "metrics": metrics,
                    }
                )
                print(f"{run_name}: {run.info.run_id}", flush=True)
    return results


@app.command()
def ejecutar(
    candidates: bool = typer.Option(False, help="Tambien entrena los modelos candidatos."),
    questions: list[str] = typer.Option(["P1"], help="Preguntas a entrenar."),
):
    """Entrena los modelos y registra cada corrida en MLflow."""
    settings = load_settings(overrides={"models.include_candidates": candidates})
    train(settings, tuple(questions))


if __name__ == "__main__":
    app()
