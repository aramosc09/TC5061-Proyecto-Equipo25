"""Prueba real de MLflow Projects con datos sintéticos y almacenamiento aislado."""

import mlflow
import pandas as pd

from proyecto_mibici.config import PROJ_ROOT, ProjectConfig
from proyecto_mibici.dataset import ejecutar_ingesta
from proyecto_mibici.experiments import ProyectoMiBici


def test_features_entrypoint_uses_locked_environment_and_common_input(tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "Viaje_Id": [1, 2],
            "Usuario_Id": [101, 102],
            "Genero": ["X", "X"],
            "Año_de_nacimiento": [1990, 1990],
            "Inicio_del_viaje": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
            "Fin_del_viaje": ["2024-01-01 10:10:00", "2024-01-01 11:10:00"],
            "Origen_Id": [1, 2],
            "Destino_Id": [2, 1],
        }
    ).to_csv(raw / "datos_abiertos_2024_01.csv", index=False)
    pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["synthetic-a", "synthetic-b"],
            "latitude": [20.6, 20.7],
            "longitude": [-103.3, -103.4],
            "dpcapacity": [10, 10],
        }
    ).to_csv(raw / "nomenclatura_2026_09.csv", index=False)
    ejecutar_ingesta(
        anio_inicio=2024,
        mes_inicio=1,
        anio_fin=2024,
        mes_fin=1,
        input_path=raw,
        parquet_path=tmp_path / "data/interim/viajes_unificados.parquet",
        estaciones_path=tmp_path / "data/interim/estaciones.parquet",
        manifiesto_path=tmp_path / "data/data_manifest.json",
        sin_descarga=True,
    )
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'tracking.db'}")
        experiment_id = mlflow.create_experiment(
            "synthetic-project-entrypoint", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        submitted = mlflow.projects.run(
            str(PROJ_ROOT),
            entry_point="features",
            env_manager="local",
            synchronous=True,
            experiment_id=experiment_id,
            parameters={
                "root": str(tmp_path),
                "start_month": "2024-01",
                "end_month": "2024-01",
                "train_end": "2024-01-15",
                "validation_end": "2024-01-22",
            },
        )
        run = mlflow.tracking.MlflowClient().get_run(submitted.run_id)
        assert run.info.status == "FINISHED"
        assert (tmp_path / "data/processed/horas_estacion.parquet").is_file()
        hours = pd.read_parquet(tmp_path / "data/processed/horas_estacion.parquet")
        assert hours.salidas.sum() == 2
        assert hours.llegadas.sum() == 2
    finally:
        mlflow.set_tracking_uri(previous)


def test_full_project_records_three_new_baselines_on_synthetic_months(tmp_path, monkeypatch):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    records = []
    for timestamp in pd.date_range("2024-01-01", "2024-03-01", freq="h", inclusive="left"):
        for station in (1, 2, 3):
            for _ in range(station if timestamp.dayofweek >= 5 else 1):
                records.append(
                    {
                        "Viaje_Id": len(records) + 1,
                        "Usuario_Id": 101,
                        "Genero": "X",
                        "Año_de_nacimiento": 1990,
                        "Inicio_del_viaje": timestamp,
                        "Fin_del_viaje": timestamp + pd.Timedelta(minutes=70),
                        "Origen_Id": station,
                        "Destino_Id": station,
                    }
                )
    trips = pd.DataFrame(records)
    for month, group in trips.groupby(trips.Inicio_del_viaje.dt.strftime("%Y_%m")):
        group.to_csv(raw / f"datos_abiertos_{month}.csv", index=False)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["synthetic-a", "synthetic-b", "synthetic-c"],
            "latitude": [20.6, 20.7, 20.8],
            "longitude": [-103.3, -103.4, -103.5],
            "dpcapacity": [10, 10, 10],
        }
    ).to_csv(raw / "nomenclatura_2026_09.csv", index=False)
    config = ProjectConfig(
        root=tmp_path,
        start_month="2024-01",
        end_month="2024-02",
        train_end="2024-02-01",
        validation_end="2024-02-15",
    )
    ejecutar_ingesta(
        anio_inicio=2024,
        mes_inicio=1,
        anio_fin=2024,
        mes_fin=2,
        input_path=raw,
        parquet_path=config.trips_path,
        estaciones_path=config.stations_path,
        manifiesto_path=config.manifest_path,
        sin_descarga=True,
    )
    previous = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        mlflow.set_tracking_uri(tracking_uri)
        submitted = ProyectoMiBici(config).execute_project()
        client = mlflow.tracking.MlflowClient()
        assert client.get_run(submitted.run_id).info.status == "FINISHED"
        children = client.search_runs(
            [client.get_run(submitted.run_id).info.experiment_id],
            f"tags.mlflow.parentRunId = '{submitted.run_id}'",
        )
        assert {run.data.tags["question"] for run in children} == {"P1", "P2", "P3"}
        assert len(children) == 3
        for run in children:
            assert run.info.status == "FINISHED"
            assert run.data.params["start_month"] == config.start_month
            assert run.data.tags["git_commit"]
            assert any(
                item.path.endswith(".png")
                for item in client.list_artifacts(run.info.run_id, "resultados")
            )
    finally:
        mlflow.set_tracking_uri(previous)
