"""Pruebas del proceso de promoción; NO requieren viajes reales ni alteran champion real."""

from types import SimpleNamespace

import pytest

from proyecto_mibici import registry


def fake_run(question="P1", *, stage="candidate", model_uri="models:/m-abc"):
    return SimpleNamespace(
        info=SimpleNamespace(experiment_id="exp-25", run_id="run-01", status="FINISHED"),
        data=SimpleNamespace(
            tags={
                "team": "25",
                "question": question,
                "stage_of_work": stage,
                "git_commit": "abcdef123",
                "data_version": "dataset-v1",
                "model_family": "gradient_boosting",
                "candidate_model_uri": model_uri,
            },
            metrics={"validation_rmse": 2.3, "test_rmse": 2.7},
        ),
    )


class ClientFake:
    def __init__(self, run=None):
        self.run = run or fake_run()
        self.operations = []
        self.alias = None

    def get_experiment_by_name(self, name):
        assert name == "mibici-equipo25"
        return SimpleNamespace(experiment_id="exp-25")

    def get_run(self, run_id):
        assert run_id == "run-01"
        return self.run

    def search_runs(self, experiment_ids, max_results):
        assert experiment_ids == ["exp-25"]
        return [self.run]

    def update_registered_model(self, *args, **kwargs):
        self.operations.append(("model_description", args, kwargs))

    def update_model_version(self, *args, **kwargs):
        self.operations.append(("version_description", args, kwargs))

    def set_model_version_tag(self, *args):
        self.operations.append(("version_tag", args))

    def set_registered_model_alias(self, name, alias, version):
        self.alias = (name, alias, version)

    def get_model_version_by_alias(self, name, alias):
        assert self.alias == (name, alias, "3")
        return SimpleNamespace(version="3")


def test_selection_preview_does_not_write_to_registry(monkeypatch):
    client = ClientFake()
    def must_not_register(**kwargs):
        raise AssertionError("preview must not register model")
    monkeypatch.setattr(registry.mlflow, "register_model", must_not_register)
    result = registry.promote_candidate(
        client, "p1", "run-01", "Mejor RMSE en validación y estable", approve=False
    )
    assert result["operation"] == "preview"
    assert result["selection_metric"] == "validation_rmse"
    assert result["data_version"] == "dataset-v1"
    assert client.operations == [] and client.alias is None


def test_rejects_baseline_and_missing_logged_model():
    for run in [fake_run(stage="baseline"), fake_run(model_uri="")]:
        with pytest.raises(ValueError):
            registry.validate_candidate(ClientFake(run), "p1", "run-01")


def test_rejects_mismatched_question_and_short_justification():
    client = ClientFake()
    with pytest.raises(ValueError, match="pregunta"):
        registry.validate_candidate(client, "p2", "run-01")
    with pytest.raises(ValueError, match="reason"):
        registry.promote_candidate(client, "p1", "run-01", "breve", approve=False)


def test_champion_requires_loadable_model_with_signature(monkeypatch):
    client = ClientFake()
    calls = []
    def register_model(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(version=3)
    monkeypatch.setattr(registry.mlflow, "register_model", register_model)
    monkeypatch.setattr(
        registry, "_load_registered_model",
        lambda uri: SimpleNamespace(metadata=SimpleNamespace(signature="signed")),
    )
    result = registry.promote_candidate(
        client, "p1", "run-01", "Menor RMSE y estabilidad temporal", approve=True
    )
    assert calls == [{"model_uri": "models:/m-abc", "name": "mibici-equipo25-p1-demanda", "await_registration_for": 300}]
    assert result["load_uri"] == "models:/mibici-equipo25-p1-demanda@champion"
    assert client.alias == ("mibici-equipo25-p1-demanda", "champion", "3")
    tag_actions = [r for r in client.operations if r[0] == "version_tag"]
    assert any(x[1][2] == "data_version" and x[1][3] == "dataset-v1" for x in tag_actions)


def test_unsigned_model_must_not_become_champion(monkeypatch):
    client = ClientFake()
    monkeypatch.setattr(
        registry.mlflow, "register_model", lambda **kwargs: SimpleNamespace(version=3)
    )
    monkeypatch.setattr(
        registry, "_load_registered_model",
        lambda uri: SimpleNamespace(metadata=SimpleNamespace(signature=None)),
    )
    with pytest.raises(ValueError, match="firma"):
        registry.promote_candidate(
            client, "p1", "run-01", "Candidato elegido justificadamente", approve=True
        )
    assert client.alias is None


def test_lists_candidates_without_choosing_on_test_metric():
    client = ClientFake()
    rows = registry.inspect_candidates(client, "p1")
    assert rows[0]["validation_rmse"] == 2.3
    assert rows[0]["test_metric_solo_reporte"] == 2.7


def test_refuses_dirty_worktree_candidate():
    run = fake_run()
    run.data.tags["git_worktree_dirty"] = "true"
    with pytest.raises(ValueError, match="sin commit"):
        registry.validate_candidate(ClientFake(run), "p1", "run-01")
