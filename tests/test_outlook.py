"""NOAA calibration, the 3-night windows and the 27-day hint."""

import json

import numpy as np
import pandas as pd
import pytest

from aurora.dataset import N_CLASSES
from aurora.location_score import sky_conditions
from aurora.outlook import (
    NoaaCalibration,
    lead_days,
    night_windows,
    parse_27day_outlook,
    parse_noaa_kp_forecast,
    probability_range,
    weeks_hint,
)
from tests.conftest import FIXTURE_NOW, FIXTURES


@pytest.fixture(scope="module")
def calibration():
    return NoaaCalibration.load()


def test_calibration_probs_are_distributions(calibration):
    kp = np.array([1.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    for lead in (1, 2, 3):
        probs, n = calibration.class_probs(kp, np.full(len(kp), lead))
        assert probs.shape == (len(kp), N_CLASSES)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        assert (n > 0).all()


def test_higher_noaa_forecast_means_higher_storm_probability(calibration):
    probs, _ = calibration.class_probs(np.array([2.0, 5.0, 7.0]), np.array([1, 1, 1]))
    p_ge5 = probs[:, 2:].sum(axis=1)
    assert p_ge5[0] < p_ge5[1] < p_ge5[2]


def test_empty_cell_falls_back_to_prior(calibration):
    # NOAA never forecast Kp 7+ three days ahead in the archive.
    assert calibration.counts[3][4].sum() == 0
    probs, n = calibration.class_probs(np.array([7.0]), np.array([3]))
    np.testing.assert_allclose(probs[0], calibration.prior[4])
    assert n[0] == pytest.approx(5.0)


def test_probability_range_brackets_p():
    lo, hi = probability_range(np.array([0.1, 0.5]), np.array([10.0, 1000.0]))
    assert (lo < [0.1, 0.5]).all() and (hi > [0.1, 0.5]).all()
    assert (hi - lo)[1] < (hi - lo)[0]  # more data, narrower range


def test_parse_kp_forecast_keeps_predicted_only():
    records = json.loads((FIXTURES / "noaa_kp_forecast.json").read_text(encoding="utf-8"))
    kp = parse_noaa_kp_forecast(records)
    assert len(kp) == sum(r["observed"] == "predicted" for r in records)
    assert kp.index.min() > FIXTURE_NOW - pd.Timedelta("3h")
    assert set(lead_days(kp.index, FIXTURE_NOW)) <= {1, 2, 3, 4}


def test_night_windows_are_dark_runs():
    times = pd.date_range(FIXTURE_NOW.floor("h"), periods=84, freq="1h")
    sky = sky_conditions(times, 48.14, 11.58, pd.DataFrame(dtype=float))
    nights = night_windows(sky)
    assert len(nights) == 3
    for night in nights:
        assert (sky.loc[night, "darkness"] > 0).all()
        assert 8 <= len(night) <= 16  # October in Munich


def test_27day_outlook_and_hint():
    text = (FIXTURES / "noaa_27day_outlook.txt").read_text(encoding="utf-8")
    outlook = parse_27day_outlook(text)
    assert len(outlook) == 27
    assert outlook["kp_max"].between(0, 9).all()

    history = pd.Series(4.0, index=pd.date_range("2026-09-20", periods=80, freq="3h", tz="UTC"))
    hint = weeks_hint(outlook, history, kp_needed=4.5, now=FIXTURE_NOW)
    assert hint["confidence"] == "low"
    dates = [d["date"] for d in hint["possible_activity"]]
    assert "2026-10-22" in dates  # outlook Kp 5
    assert all(d >= "2026-10-06" for d in dates)  # after the 3-night outlook
    oct22 = next(d for d in hint["possible_activity"] if d["date"] == "2026-10-22")
    assert oct22["reaches_kp_needed"] is True
