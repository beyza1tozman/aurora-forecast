"""Sun/moon positions against known events."""

import numpy as np
import pandas as pd
import pytest

from aurora.physics.sky import moon_altitude, moon_illumination, sun_altitude


def at(s: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(s, tz="UTC")])


def test_sun_at_equinox_noon_greenwich():
    # Solar noon on 2024-03-20 is ~12:07 UTC; altitude = 90 - latitude + declination(~0).
    assert sun_altitude(at("2024-03-20 12:07"), 51.48, 0.0)[0] == pytest.approx(38.7, abs=0.3)


def test_sun_below_horizon_at_midnight():
    assert sun_altitude(at("2024-12-21 00:00"), 52.5, 13.4)[0] < -50


def test_full_and_new_moon():
    assert moon_illumination(at("2024-05-23 13:53"))[0] > 0.99  # full moon
    assert moon_illumination(at("2024-04-08 18:21"))[0] < 0.01  # new moon (solar eclipse)


def test_moon_covers_sun_during_eclipse():
    # Total solar eclipse, Dallas, 2024-04-08 18:42 UTC: moon and sun at the same altitude
    # (up to ~1 degree of topocentric parallax, which is ignored).
    t, lat, lon = at("2024-04-08 18:42"), 32.78, -96.80
    assert sun_altitude(t, lat, lon)[0] == pytest.approx(64.6, abs=1.0)
    assert moon_altitude(t, lat, lon)[0] == pytest.approx(sun_altitude(t, lat, lon)[0], abs=1.5)


def test_vectorised_and_requires_utc():
    times = pd.date_range("2026-10-03", periods=24, freq="1h", tz="UTC")
    assert sun_altitude(times, 48.1, 11.6).shape == (24,)
    assert np.all((moon_illumination(times) >= 0) & (moon_illumination(times) <= 1))
    with pytest.raises(ValueError):
        sun_altitude(pd.DatetimeIndex(["2026-10-03"]), 48.1, 11.6)
