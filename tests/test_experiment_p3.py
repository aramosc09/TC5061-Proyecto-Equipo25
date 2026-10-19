import pandas as pd
import pytest

from proyecto_mibici.features_p3 import PROFILE_COLUMNS
from proyecto_mibici.modeling.experiment_p3 import (
    assign_weekend_median_baseline,
    weekend_activity_share,
)


def profiles_with_weekend_shares(shares: list[float]) -> pd.DataFrame:
    profiles = pd.DataFrame(
        0.0,
        index=[str(index) for index in range(len(shares))],
        columns=PROFILE_COLUMNS,
    )
    saturday_columns = [column for column in PROFILE_COLUMNS if "_sabado_" in column]
    weekday_columns = [
        column
        for column in PROFILE_COLUMNS
        if "_sabado_" not in column and "_domingo_" not in column
    ]
    for station_id, share in zip(profiles.index, shares, strict=True):
        profiles.loc[station_id, saturday_columns[0]] = share
        profiles.loc[station_id, weekday_columns[0]] = 1 - share
    return profiles


def test_weekend_activity_share_sums_saturday_and_sunday_features():
    profiles = profiles_with_weekend_shares([0.2, 0.8])

    shares = weekend_activity_share(profiles)

    assert shares.to_dict() == pytest.approx({"0": 0.2, "1": 0.8})


def test_median_baseline_uses_strictly_greater_than_training_threshold():
    profiles = profiles_with_weekend_shares([0.1, 0.3, 0.5])

    assignments = assign_weekend_median_baseline(profiles, threshold=0.3)

    assert assignments.to_dict() == {"0": 0, "1": 0, "2": 1}
