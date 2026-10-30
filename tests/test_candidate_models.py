"""Contratos de hiperparámetros y métricas comparables de los candidatos P1/P2/P3."""

import numpy as np
import pytest

from proyecto_mibici.config import CANDIDATE_MODELS
from proyecto_mibici.modeling.base import resolve_hyperparameters
from proyecto_mibici.modeling.datasets import (
    DatosSupervisados,
    PerfilesP3,
    TablaSupervisada,
    construir_perfiles,
)
from proyecto_mibici.modeling.p1_candidates import (
    CandidatosP1,
    build_estimator,
    regression_metrics,
)
from proyecto_mibici.modeling.p2_candidates import CandidatosP2
from proyecto_mibici.modeling.p3_candidates import CandidatosP3

SMALL = {
    ("p1", "poisson"): {},
    ("p1", "hgb"): {"max_iter": 10},
    ("p1", "rf"): {"n_estimators": 5, "max_depth": 4},
    ("p2", "logistic"): {"max_iter": 50},
    ("p2", "hgb"): {"max_iter": 10},
    ("p2", "rf"): {"n_estimators": 5, "max_depth": 4},
}


def test_hyperparameters_are_typed_and_foreign_keys_rejected():
    params = resolve_hyperparameters("p1", "rf", {"n_estimators": "300", "max_features": "0.3"})
    assert params["n_estimators"] == 300 and params["max_features"] == 0.3
    assert params["max_depth"] == CANDIDATE_MODELS["p1"]["rf"]["defaults"]["max_depth"]
    with pytest.raises(ValueError, match="no acepta"):
        resolve_hyperparameters("p1", "poisson", {"n_estimators": "10"})
    with pytest.raises(ValueError, match="tipo int"):
        resolve_hyperparameters("p1", "rf", {"n_estimators": "0.5"})
    with pytest.raises(ValueError, match="no disponible"):
        resolve_hyperparameters("p2", "poisson", {})


def test_divergent_regression_predictions_are_rejected():
    with pytest.raises(ValueError, match="no finitas"):
        regression_metrics(np.array([1.0, 2.0]), np.array([1.0, np.inf]))


def test_random_forest_training_is_reproducible_with_same_seed():
    features = np.arange(60, dtype=float).reshape(20, 3)
    target = np.array([0, 1, 2, 1, 3] * 4, dtype=float)
    params = {"n_estimators": 7, "max_depth": 4, "min_samples_leaf": 1}

    first = build_estimator("rf", params, seed=42).fit(features, target)
    second = build_estimator("rf", params, seed=42).fit(features, target)

    first_predictions = first.predict(features)
    second_predictions = second.predict(features)
    np.testing.assert_allclose(first_predictions, second_predictions, rtol=0, atol=1e-12)
    assert regression_metrics(target, first_predictions) == pytest.approx(
        regression_metrics(target, second_predictions)
    )


@pytest.mark.parametrize(("question", "model"), list(SMALL))
def test_supervised_candidates_report_baseline_comparable_metrics(
    question, model, candidate_config, synthetic_repository, tmp_path
):
    table = TablaSupervisada(candidate_config)
    metadata = table.prepare(synthetic_repository(candidate_config), "synthetic-v1")
    data = DatosSupervisados(table, metadata)
    runner = {"p1": CandidatosP1, "p2": CandidatosP2}[question](candidate_config)
    params = resolve_hyperparameters(question, model, SMALL[(question, model)])
    result = runner.run(data, model, params, tmp_path, max_train_rows=0, evaluate_test=True)
    metric = "rmse" if question == "p1" else "f1_macro"
    for split in ("train", "validation", "test"):
        assert f"{split}_{metric}" in result.metrics
        assert f"{split}_baseline_{metric}" in result.metrics
    assert result.model_family == CANDIDATE_MODELS[question][model]["family"]
    if question == "p1":
        assert ("salidas_lag_672" in result.feature_names) == (model != "poisson")
    assert result.parameters["train_rows_used"] == metadata["rows"]["train"]
    trial = runner.run(data, model, params, tmp_path, max_train_rows=200, evaluate_test=False)
    assert trial.parameters["train_rows_used"] == 200
    assert not any(name.startswith("test_") for name in trial.metrics)


@pytest.mark.parametrize("model", list(CANDIDATE_MODELS["p3"]))
def test_p3_candidates_freeze_groups_in_train(
    model, candidate_config, synthetic_repository, tmp_path
):
    frames, coverage = construir_perfiles(synthetic_repository(candidate_config), candidate_config)
    params = resolve_hyperparameters("p3", model, {"n_clusters": 2})
    result = CandidatosP3(candidate_config).run(
        PerfilesP3(frames, coverage), model, params, tmp_path, max_train_rows=0, evaluate_test=True
    )
    assert result.metrics["train_n_stations"] == 6
    assert result.metrics["validation_ari_vs_train_defined"] in (0, 1)
    assert result.metrics["train_group_0_size"] + result.metrics["train_group_1_size"] == 6
    assert result.parameters["coordinates_used_for_clustering"] is False
    assert hasattr(result.estimator, "predict")
