"""Verify the P1 workflow uses its own outputs and leaves shared inputs untouched."""

import hashlib
import json

import pandas as pd

from proyecto_mibici.config_p1 import load_settings
from proyecto_mibici.features_p1 import build_features
from proyecto_mibici.modeling.train_p1 import train


def test_p1_workflow_isolated_from_shared_inputs(tmp_path):
    settings = load_settings(root=tmp_path, overrides={
        "data.start_month": "2024-01",
        "data.end_month": "2024-03",
        "split.train_end": "2024-02-01",
        "split.validation_end": "2024-03-01",
        "project.experiment": "synthetic-p1-isolation-test",
        "models.include_candidates": True,
    })
    hours = pd.date_range("2024-01-01", "2024-04-01", freq="h", inclusive="left")
    trips = []
    for station in (1, 2, 3):
        for hour in hours:
            for _ in range((hour.hour + station) % 3):
                trips.append({
                    "viaje_id": len(trips) + 1,
                    "inicio_viaje": hour,
                    "fin_viaje": hour + pd.Timedelta(minutes=10),
                    "origen_id": station,
                    "destino_id": station % 3 + 1,
                })
    source = settings.path("data", "trips_parquet")
    source.parent.mkdir(parents=True)
    pd.DataFrame(trips).to_parquet(source, index=False)
    stations = settings.path("data", "stations_csv")
    stations.parent.mkdir(parents=True)
    pd.DataFrame({
        "id": [1, 2, 3], "name": ["synthetic-a", "synthetic-b", "synthetic-c"],
        "latitude": [20.65, 20.66, 20.67], "longitude": [-103.3, -103.31, -103.32],
        "dpcapacity": [10, 15, 20],
    }).to_csv(stations, index=False)
    manifest = settings.path("data", "manifest")
    manifest.write_text(json.dumps({"data_version": "synthetic-p1"}), encoding="utf8")
    input_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (source, stations, manifest)}
    shared_output = tmp_path / "data/processed/features_estacion_hora.parquet"
    shared_output.parent.mkdir(parents=True)
    shared_output.write_bytes(b"teammate output sentinel")

    build_features(settings)
    results = train(settings)

    assert len(results) == 2
    assert {run["question"] for run in results} == {"P1"}
    assert {run["data_version"] for run in results} == {"synthetic-p1"}
    assert all("test_mae" in run["metrics"] for run in results)
    for path, expected in input_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert shared_output.read_bytes() == b"teammate output sentinel"
    assert settings.path("tracking", "database").is_file()
    assert settings.path("features", "supervised").parent == tmp_path / "data/processed/p1"
    assert settings.path("tracking", "database").parent == tmp_path / "mlruns/p1"
