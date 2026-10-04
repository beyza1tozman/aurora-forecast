"""Verification metrics on hand-checkable inputs."""

import numpy as np
import pandas as pd
import pytest

from aurora.metrics import (
    block_bootstrap_bss,
    brier,
    brier_skill,
    contingency,
    hss,
    pr_auc,
    reliability_curve,
    tss,
    week_blocks,
)


def test_brier_perfect_and_worst():
    y = np.array([0, 1, 1, 0])
    assert brier(y, y) == 0
    assert brier(1 - y, y) == 1


def test_bss_of_reference_against_itself_is_zero():
    y = np.array([0, 0, 0, 1])
    clim = np.full(4, 0.25)
    assert brier_skill(clim, y, clim) == 0
    assert brier_skill(y, y, clim) == 1


def test_contingency_tss_hss():
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1, 0.6])
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    table = contingency(p, y, 0.5)
    assert table == {"hits": 2, "false_alarms": 2, "misses": 1, "correct_negatives": 3}
    assert tss(table) == pytest.approx(2 / 3 - 2 / 5)
    # HSS = 2(ad - bc) / ((a+c)(c+d) + (a+b)(b+d)) = 2(6-2) / (3*4 + 4*5)
    assert hss(table) == pytest.approx(8 / 32)


def test_pr_auc_perfect_ranking_and_no_events():
    y = np.array([0, 0, 1, 1])
    assert pr_auc([0.1, 0.2, 0.8, 0.9], y) == pytest.approx(1.0)
    assert np.isnan(pr_auc([0.1, 0.2], [0, 0]))


def test_reliability_curve_bins():
    curve = reliability_curve([0.05, 0.05, 0.95, 0.95], [0, 0, 1, 0], n_bins=10)
    assert curve["count"].tolist() == [2, 2]
    assert curve["observed"].tolist() == [0.0, 0.5]


def test_block_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    times = pd.date_range("2020-01-01", periods=24 * 365, freq="1h", tz="UTC")
    y = (rng.random(len(times)) < 0.1).astype(float)
    p = np.clip(0.1 + 0.5 * (y - 0.1) + rng.normal(0, 0.1, len(y)), 0, 1)
    clim = np.full(len(y), 0.1)

    lo, hi = block_bootstrap_bss(p, y, clim, week_blocks(times), n_boot=300)
    assert lo < brier_skill(p, y, clim) < hi
