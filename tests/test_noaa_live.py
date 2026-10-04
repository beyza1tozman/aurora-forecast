"""Live solar wind and Kp ingestion, and the live Kp forecast."""

import json

import numpy as np
import pandas as pd
import pytest

from aurora.data.noaa_live import (
    BOW_SHOCK_X_KM,
    combine_kp,
    hourly_average,
    parse_gfz_kp,
    parse_noaa_kp,
    rtsw_l1,
    shift_to_bow_shock,
    solar_wind_hourly,
)
from aurora.features import SW_COLUMNS
from aurora.live import MAX_STALENESS, NoUsableDataError, forecast_kp
from aurora.model import load_models
from tests.conftest import FIXTURE_NOW, FIXTURES


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sw():
    mag, wind = load("noaa_rtsw_mag_1m.json"), load("noaa_rtsw_wind_1m.json")
    return solar_wind_hourly(mag, wind, load("noaa_rtsw_ephemerides_1h.json"))


@pytest.fixture(scope="module")
def kp():
    return combine_kp(
        parse_gfz_kp(load("gfz_kp_nowcast.json")), parse_noaa_kp(load("noaa_kp.json")), FIXTURE_NOW
    )


def test_only_active_spacecraft(sw):
    _, meta = sw
    assert meta["spacecraft"] == ["SOLAR1"]  # the fixture also has inactive ACE and IMAP rows
    assert 1.2e6 < meta["l1_distance_km"] < 1.7e6


def test_shift_moves_samples_forward_by_travel_time():
    times = pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC")
    l1 = pd.DataFrame({c: 1.0 for c in SW_COLUMNS}, index=times).assign(speed=500.0)
    shifted = shift_to_bow_shock(l1, x_gse_km=1.5e6)
    delay = (shifted.index - times).total_seconds()
    assert np.allclose(delay, (1.5e6 - BOW_SHOCK_X_KM) / 500.0)


def test_hourly_average_drops_incomplete_hour():
    times = pd.date_range("2026-01-01 00:00", "2026-01-01 01:30", freq="1min", tz="UTC")
    df = pd.DataFrame({c: 1.0 for c in SW_COLUMNS}, index=times)
    hourly = hourly_average(df)
    assert hourly.index.tolist() == [pd.Timestamp("2026-01-01 00:00", tz="UTC")]


def test_hourly_rows_match_training_columns(sw):
    hourly, _ = sw
    assert list(hourly.columns) == SW_COLUMNS
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()


def test_rtsw_mag_bt_is_total_field():
    t = {"time_tag": "2026-01-01T00:00:00", "active": True}
    mag = [{**t, "bt": 5.0, "by_gsm": 3.0, "bz_gsm": -4.0}]
    wind = [{**t, "proton_speed": 400, "proton_density": 5, "proton_temperature": 1e5}]
    row = rtsw_l1(mag, wind).iloc[0]
    assert row["b_mag"] == 5.0 and row["bz_gsm"] == -4.0 and row["speed"] == 400


def test_combine_kp_drops_interval_in_progress_and_prefers_gfz(kp):
    # GFZ lists 15-18 UT with a provisional 0; at 16:10 that interval has not ended.
    assert kp.index.max() == pd.Timestamp("2026-10-03 12:00", tz="UTC")
    gfz = parse_gfz_kp(load("gfz_kp_nowcast.json"))
    assert kp["2026-10-03 12:00"] == gfz["2026-10-03 12:00"]
    assert kp.index.freq == pd.Timedelta("3h")


def test_live_forecast_on_fixtures(sw, kp):
    hourly, _ = sw
    fc = forecast_kp(hourly, kp, load_models(), FIXTURE_NOW)
    assert FIXTURE_NOW - MAX_STALENESS <= fc.issued <= FIXTURE_NOW + pd.Timedelta("2h")
    assert [h.horizon_h for h in fc.horizons] == [1, 3, 6]
    for h in fc.horizons:
        p = h.exceedance()
        assert 0 <= p[7] <= p[6] <= p[5] <= 1
        assert h.class_probs.sum() == pytest.approx(1.0)
    assert fc.inputs_missing == []


def test_live_forecast_refuses_stale_data(sw, kp):
    hourly, _ = sw
    with pytest.raises(NoUsableDataError):
        forecast_kp(hourly, kp, load_models(), FIXTURE_NOW + pd.Timedelta("12h"))
