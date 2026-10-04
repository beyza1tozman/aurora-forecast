"""Sky factors and the combined location score."""

import numpy as np
import pandas as pd
import pytest

from aurora.data.open_meteo import parse_clouds
from aurora.location_score import (
    clear_sky_factor,
    combine,
    darkness_factor,
    location_info,
    moon_factor,
    sky_conditions,
)


def test_clear_sky_factor():
    assert clear_sky_factor(0.0, 0.0, 0.0) == 1.0
    assert clear_sky_factor(1.0, 0.0, 0.0) == 0.0
    assert clear_sky_factor(0.0, 1.0, 0.0) == 0.0  # overcast mid cloud blocks fully
    assert clear_sky_factor(0.0, 0.0, 1.0) == pytest.approx(0.5)  # cirrus only partly
    assert clear_sky_factor(0.5, 0.5, 0.0) == pytest.approx(0.25)  # random overlap
    assert np.isnan(clear_sky_factor(np.nan, 0.0, 0.0))


def test_parse_clouds_without_mid_high_is_conservative():
    payload = {
        "hourly": {"time": ["2026-10-03T00:00"], "cloud_cover": [100], "cloud_cover_low": [0]}
    }
    row = parse_clouds(payload).iloc[0]
    low, mid, high = row["cloud_cover_low"], row["cloud_cover_mid"], row["cloud_cover_high"]
    assert (mid, high) == (1.0, 0.0)
    assert clear_sky_factor(low, mid, high) == 0


def test_darkness_factor_edges():
    np.testing.assert_allclose(darkness_factor([10, -6, -9, -12, -30]), [0, 0, 0.5, 1, 1])


def test_moon_factor_is_mild():
    assert moon_factor(-10, 1.0) == 1.0  # below horizon
    assert moon_factor(60, 0.0) == 1.0  # new moon
    assert moon_factor(60, 1.0) == pytest.approx(0.6)  # worst case, still not zero


def test_location_info():
    info = location_info(53.55, 10.0, pd.Timestamp("2026-10-03", tz="UTC"))
    assert info["mlat"] == pytest.approx(53.7, abs=0.2)
    assert 5 < info["kp_needed"] < 6


def test_missing_clouds_only_drop_chance():
    times = pd.date_range("2026-10-03 20:00", periods=3, freq="1h", tz="UTC")
    clouds = pd.DataFrame(
        {
            "cloud_cover": [0.2, np.nan, 0.2],
            "cloud_cover_low": [0.0, np.nan, 0.0],
            "cloud_cover_mid": [0.1, np.nan, 0.1],
            "cloud_cover_high": [0.1, np.nan, 0.1],
        },
        index=times,
    )
    out = combine(0.5, sky_conditions(times, 53.55, 10.0, clouds))
    assert out["chance_if_clear"].notna().all()
    assert out["chance"].isna().tolist() == [False, True, False]
    assert (out["chance"].dropna() <= out["chance_if_clear"][out["chance"].notna()]).all()
    assert out["chance_if_clear"].max() <= 0.5
