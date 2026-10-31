import pandas as pd
import pytest

from proyecto_mibici.data_quality import dataset_arguments, load_splits, preflight_summary
from proyecto_mibici.modeling.datasets import (
    DatosSupervisados,
    PerfilesP3,
    TablaSupervisada,
    construir_perfiles,
)


@pytest.fixture
def supervised_data(candidate_config, synthetic_repository):
    table = TablaSupervisada(candidate_config)
    metadata = table.prepare(synthetic_repository(candidate_config), "synthetic-v1")
    return DatosSupervisados(table, metadata)


def test_preflight_reports_missing_values_and_duplicates():
    train = pd.DataFrame({"feature": [1.0, 1.0, None], "target": [0.0, 0.0, 1.0]})
    test = pd.DataFrame({"feature": [2.0], "target": [1.0]})

    summary = preflight_summary(train, test)

    assert summary["missing_values"] == {"train": 1, "test": 0}
    assert summary["duplicate_rows"] == {"train": 1, "test": 0}
    assert summary["rows"] == {"train": 3, "test": 1}


def test_preflight_rejects_different_train_test_schema():
    train = pd.DataFrame({"feature": [1.0], "target": [0.0]})
    test = pd.DataFrame({"target": [0.0], "feature": [1.0]})

    with pytest.raises(ValueError, match="mismas columnas"):
        preflight_summary(train, test)


def test_p2_is_validated_on_the_seven_pressure_classes(supervised_data):
    train, test = load_splits("p2", supervised_data, max_rows=50, seed=0)

    assert "presion" not in train.columns
    assert set(train["clase_presion"]) | set(test["clase_presion"]) <= set(range(7))
    assert dataset_arguments("p2", train)["label_type"] == "multiclass"


def test_supervised_splits_keep_time_but_not_station(supervised_data):
    train, test = load_splits("p1", supervised_data, max_rows=20, seed=0)

    assert len(train) == 20 and "estacion" not in train.columns
    assert train["hora"].max() < test["hora"].min()
    arguments = dataset_arguments("p1", train)
    assert arguments["label_type"] == "regression" and arguments["datetime_name"] == "hora"


def test_sampling_is_reproducible_with_same_seed(supervised_data):
    first, _ = load_splits("p1", supervised_data, max_rows=20, seed=7)
    second, _ = load_splits("p1", supervised_data, max_rows=20, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_p3_profiles_are_validated_without_label(candidate_config, synthetic_repository):
    data = PerfilesP3(
        *construir_perfiles(synthetic_repository(candidate_config), candidate_config)
    )

    train, test = load_splits("p3", data, max_rows=10, seed=0)

    assert list(train.columns) == list(test.columns) and len(train.columns) == 336
    assert "label" not in dataset_arguments("p3", train)


@pytest.mark.parametrize(
    ("question", "max_rows", "message"), [("p4", 10, "question"), ("p1", 0, "max_rows")]
)
def test_invalid_arguments_are_rejected(supervised_data, question, max_rows, message):
    with pytest.raises(ValueError, match=message):
        load_splits(question, supervised_data, max_rows=max_rows, seed=0)
