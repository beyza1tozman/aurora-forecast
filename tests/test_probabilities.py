"""Kp classes, targets and exceedance probabilities."""

import numpy as np
import pandas as pd
import pytest

from aurora.baselines import (
    climatology,
    conditional_persistence,
    fit_conditional_persistence,
    persistence,
    recurrence,
)
from aurora.dataset import (
    N_CLASSES,
    THRESHOLDS,
    exceedance_from_class_probs,
    kp_to_class,
    make_targets,
)

START = pd.Timestamp("2024-05-01", tz="UTC")


@pytest.mark.parametrize(
    ("kp", "cls"),
    [(0.0, 0), (3.333, 0), (3.667, 1), (4.333, 1), (4.667, 2), (6.0, 3), (6.667, 4), (9.0, 4)],
)
def test_kp_to_class_groups_thirds(kp, cls):
    assert kp_to_class(kp) == cls


def test_kp_to_class_keeps_nan():
    assert np.isnan(kp_to_class(np.nan))


@pytest.fixture
def kp_3h():
    rng = np.random.default_rng(0)
    index = pd.date_range(START - pd.Timedelta("30D"), periods=8 * 40, freq="3h")
    kp = np.round(rng.uniform(0, 9, len(index)) * 3) / 3
    return pd.DataFrame({"kp": kp, "kp_definitive": True}, index=index)


def test_target_is_interval_containing_issue_plus_horizon(kp_3h):
    index = pd.DatetimeIndex([pd.Timestamp("2024-05-05 07:00", tz="UTC")])
    targets = make_targets(index, kp_3h)
    # issue 08:00: +1h = 09:00 -> 09-12 bin; +3h = 11:00 -> 09-12; +6h = 14:00 -> 12-15
    assert targets["kp_h1"].iloc[0] == kp_3h.loc["2024-05-05 09:00", "kp"]
    assert targets["kp_h3"].iloc[0] == kp_3h.loc["2024-05-05 09:00", "kp"]
    assert targets["kp_h6"].iloc[0] == kp_3h.loc["2024-05-05 12:00", "kp"]


def test_non_definitive_targets_are_dropped(kp_3h):
    kp_3h.loc["2024-05-05 09:00", "kp_definitive"] = False
    index = pd.DatetimeIndex([pd.Timestamp("2024-05-05 07:00", tz="UTC")])
    assert np.isnan(make_targets(index, kp_3h)["kp_h1"].iloc[0])


def all_baselines(kp_3h):
    index = pd.date_range(START, periods=24 * 5, freq="1h")
    kp_last = kp_3h["kp"].reindex(index.floor("3h")).to_numpy()
    classes = kp_to_class(kp_last)
    table = fit_conditional_persistence(classes, np.roll(classes, -3))
    return [
        persistence(kp_last),
        climatology(classes, len(index)),
        recurrence(kp_3h["kp"], index, 3),
        conditional_persistence(kp_last, table),
    ]


def test_baseline_probabilities_are_valid_and_monotonic(kp_3h):
    for probs in all_baselines(kp_3h):
        assert probs.shape[1] == N_CLASSES
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        exceed = [exceedance_from_class_probs(probs, t) for t in THRESHOLDS]
        for p in exceed:
            assert ((p >= 0) & (p <= 1 + 1e-12)).all()
        for lower, higher in zip(exceed, exceed[1:], strict=False):
            assert (higher <= lower + 1e-12).all()


def test_conditional_persistence_rows_are_smoothed_distributions():
    table = fit_conditional_persistence([0, 0, 1], [0, 1, 1])
    np.testing.assert_allclose(table.sum(axis=1), 1.0)
    assert (table > 0).all()
    assert table[0, 0] > table[0, 4]
