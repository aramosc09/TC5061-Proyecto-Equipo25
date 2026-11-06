"""Carga del modelo de la API; usa un modelo sintético, sin servidor MLflow ni datos reales."""

import json
from types import SimpleNamespace

import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from proyecto_mibici.api import modelo as api_modelo

NAME = api_modelo.REGISTERED_MODEL
COLUMNS = ["hora_sin", "salidas_lag_1", "salidas_media_24"]


def save_synthetic_model(path, question="P1"):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((60, len(COLUMNS))), columns=COLUMNS).astype("float32")
    y = rng.poisson(2, 60)
    estimator = HistGradientBoostingRegressor(loss="poisson", max_iter=5).fit(X, y)
    mlflow.sklearn.save_model(
        estimator,
        str(path),
        signature=infer_signature(X, estimator.predict(X)),
        input_example=X.head(2),
        metadata={"question": question},
        skops_trusted_types=[
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
        ],
    )


def write_export(tmp_path, question="P1"):
    save_synthetic_model(tmp_path / api_modelo.MODEL_SUBDIR, question)
    lineage = {"registered_model": NAME, "version": "4", "alias": "champion"}
    (tmp_path / api_modelo.METADATA_FILE).write_text(json.dumps(lineage), encoding="utf-8")
    return tmp_path


class RegistryFake:
    def __init__(self):
        self.version = SimpleNamespace(
            version=4,
            run_id="run-25",
            tags={"source_run_id": "run-25", "git_commit": "abc1234", "data_version": "09a6"},
        )

    def get_model_version_by_alias(self, name, alias):
        assert (name, alias) == (NAME, "champion")
        return self.version

    def get_model_version(self, name, version):
        assert (name, version) == (NAME, "4")
        return self.version


@pytest.mark.parametrize(
    "uri, expected",
    [
        (f"models:/{NAME}@champion", (NAME, "champion", None)),
        (f"models:/{NAME}/4", (NAME, None, "4")),
    ],
)
def test_parse_model_uri_accepts_alias_or_version(uri, expected):
    assert api_modelo.parse_model_uri(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        f"models:/{NAME}",
        f"models:/{NAME}/latest",
        "models:/mibici-equipo25-p2-presion@champion",
        "runs:/abc/model",
    ],
)
def test_parse_model_uri_rejects_implicit_or_foreign_models(uri):
    with pytest.raises(ValueError):
        api_modelo.parse_model_uri(uri)


def test_resolve_version_pins_alias_to_concrete_version_with_lineage():
    info = api_modelo.resolve_version(RegistryFake(), f"models:/{NAME}@champion")
    assert info["load_uri"] == f"models:/{NAME}/4"
    assert info["version"] == "4" and info["alias"] == "champion"
    assert info["git_commit"] == "abc1234" and info["data_version"] == "09a6"


def test_local_model_predicts_in_signature_order_and_types(tmp_path):
    servido = api_modelo.cargar_modelo("local", model_dir=write_export(tmp_path))
    features = pd.DataFrame(
        {"salidas_media_24": [1.5, 0.0], "extra": [9, 9], "hora_sin": [0.5, -1.0],
         "salidas_lag_1": [3.0, 0.0]}
    )
    prediction = servido.predecir(features)
    assert servido.columnas == COLUMNS
    assert prediction.shape == (2,) and np.isfinite(prediction).all()
    assert servido.metadatos["version"] == "4" and servido.metadatos["source"] == "local"


def test_predict_reports_missing_features(tmp_path):
    servido = api_modelo.cargar_modelo("local", model_dir=write_export(tmp_path))
    with pytest.raises(ValueError, match="salidas_lag_1"):
        servido.predecir(pd.DataFrame({"hora_sin": [0.1], "salidas_media_24": [1.0]}))


def test_local_source_without_export_explains_how_to_create_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="exportar"):
        api_modelo.cargar_modelo("local", model_dir=tmp_path)


def test_rejects_model_of_another_question(tmp_path):
    with pytest.raises(ValueError, match="P1"):
        api_modelo.cargar_modelo("local", model_dir=write_export(tmp_path, question="P2"))


def test_rejects_unknown_source():
    with pytest.raises(ValueError, match="MODEL_SOURCE"):
        api_modelo.cargar_modelo("s3")


def test_export_downloads_pinned_version_and_writes_lineage(tmp_path, monkeypatch):
    calls = []

    def fake_download(artifact_uri, dst_path):
        calls.append(artifact_uri)
        save_synthetic_model(dst_path)

    monkeypatch.setattr(api_modelo, "download_artifacts", fake_download)
    info = api_modelo.exportar_modelo(RegistryFake(), f"models:/{NAME}@champion", tmp_path)
    assert calls == [f"models:/{NAME}/4"]
    saved = json.loads((tmp_path / api_modelo.METADATA_FILE).read_text(encoding="utf-8"))
    assert saved == info and saved["source_run_id"] == "run-25"
    assert api_modelo.cargar_modelo("local", model_dir=tmp_path).metadatos["version"] == "4"
