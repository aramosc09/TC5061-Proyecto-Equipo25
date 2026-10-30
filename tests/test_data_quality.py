import pandas as pd
import pytest

from proyecto_mibici.data_quality import preflight_summary


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
