"""Features at time t must not change when anything after t changes."""

import numpy as np
import pandas as pd
import pytest

from aurora.features import FEATURE_COLUMNS, SW_COLUMNS, build_features, newell_coupling

START = pd.Timestamp("2024-01-01", tz="UTC")


def synthetic(seed: int = 0, days: int = 40):
    rng = np.random.default_rng(seed)
    hours = pd.date_range(START, periods=days * 24, freq="1h")
    sw = pd.DataFrame(
        rng.normal(size=(len(hours), len(SW_COLUMNS))), index=hours, columns=SW_COLUMNS
    )
    sw["speed"] = rng.uniform(300, 800, len(hours))
    sw["density"] = rng.uniform(1, 20, len(hours))
    sw.iloc[rng.choice(len(hours), 50, replace=False), 1] = np.nan  # gaps in Bz
    kp_index = pd.date_range(START - pd.Timedelta("30D"), hours[-1], freq="3h")
    kp = pd.Series(np.round(rng.uniform(0, 9, len(kp_index)) * 3) / 3, index=kp_index)
    return sw, kp


@pytest.mark.parametrize("cut_hour", [24 * 28 + 5, 24 * 30 + 7, 24 * 35 + 8])
def test_future_perturbation_does_not_change_past_features(cut_hour):
    sw, kp = synthetic()
    t = sw.index[cut_hour]
    before = build_features(sw, kp).loc[:t]

    rng = np.random.default_rng(1)
    sw2, kp2 = sw.copy(), kp.copy()
    future = sw2.index > t
    sw2.loc[future] = rng.normal(50, 10, size=(future.sum(), len(SW_COLUMNS)))
    # Kp intervals that end after issue time t + 1h are not yet known at t
    unknown = kp2.index + pd.Timedelta("3h") > t + pd.Timedelta("1h")
    kp2[unknown] = 9.0

    after = build_features(sw2, kp2).loc[:t]
    pd.testing.assert_frame_equal(before, after)


def test_output_columns_and_index():
    sw, kp = synthetic()
    f = build_features(sw, kp)
    assert list(f.columns) == FEATURE_COLUMNS
    assert f.index.equals(sw.index)


@pytest.mark.parametrize(
    ("row", "bin_start", "hours_into_bin"),
    [
        ("2024-01-05 08:00", "2024-01-05 06:00", 0),  # issue 09:00, 06-09 just finished
        ("2024-01-05 07:00", "2024-01-05 03:00", 2),  # issue 08:00, 06-09 still running
        ("2024-01-05 06:00", "2024-01-05 03:00", 1),
        ("2024-01-05 01:00", "2024-01-04 21:00", 2),  # crosses midnight
    ],
)
def test_kp_last_is_last_completed_interval(row, bin_start, hours_into_bin):
    sw, kp = synthetic()
    f = build_features(sw, kp)
    start = pd.Timestamp(bin_start, tz="UTC")
    assert f.loc[row, "kp_last"] == kp[start]
    assert f.loc[row, "kp_prev"] == kp[start - pd.Timedelta("3h")]
    assert f.loc[row, "kp_27d"] == kp[start - pd.Timedelta("27D")]
    assert f.loc[row, "hours_into_bin"] == hours_into_bin


def test_newell_zero_for_northward_and_max_for_southward():
    v, bt = 500.0, 10.0
    assert newell_coupling(v, bt, 0.0) == pytest.approx(0.0)
    south = newell_coupling(v, bt, np.pi)
    dawn = newell_coupling(v, bt, np.pi / 2)
    assert south == pytest.approx(v ** (4 / 3) * bt ** (2 / 3))
    assert 0 < dawn < south


def test_bz_south_hours_counts_consecutive_southward_hours():
    hours = pd.date_range(START, periods=8, freq="1h")
    sw = pd.DataFrame(1.0, index=hours, columns=SW_COLUMNS)
    sw["bz_gsm"] = [1, -1, -2, -3, 2, -1, np.nan, -1]
    kp = pd.Series(1.0, index=pd.date_range(START - pd.Timedelta("30D"), hours[-1], freq="3h"))
    run = build_features(sw, kp)["bz_south_hours"].tolist()
    assert run[:6] == [0, 1, 2, 3, 0, 1]
    assert np.isnan(run[6])
    assert run[7] == 1
