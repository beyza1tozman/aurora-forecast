"""Contract tests for external feeds.

The offline tests check that our saved fixtures contain the fields the app relies on.
The `network` tests check that the *live* feeds still do. They are skipped in CI
(`pytest -m "not network"`); run them locally to detect provider format changes:
    pytest -m network
"""

import json
from pathlib import Path

import httpx
import pytest

from aurora import config

FIXTURES = Path(__file__).parent / "fixtures"

# feed -> (fixture file, live URL, required keys per record)
JSON_CONTRACTS = {
    "rtsw_mag": (
        "noaa_rtsw_mag_1m.json",
        config.NOAA_RTSW_MAG_URL,
        {"time_tag", "active", "source", "bt", "by_gsm", "bz_gsm"},
    ),
    "rtsw_wind": (
        "noaa_rtsw_wind_1m.json",
        config.NOAA_RTSW_WIND_URL,
        {"time_tag", "active", "source", "proton_speed", "proton_density"},
    ),
    "rtsw_ephemerides": (
        "noaa_rtsw_ephemerides_1h.json",
        config.NOAA_RTSW_EPHEMERIS_URL,
        {"time_tag", "active", "source", "x_gse"},
    ),
    "noaa_kp": ("noaa_kp.json", config.NOAA_KP_URL, {"time_tag", "Kp"}),
    "noaa_kp_forecast": (
        "noaa_kp_forecast.json",
        config.NOAA_KP_FORECAST_URL,
        {"time_tag", "kp", "observed"},
    ),
}


def _check_records(records, required):
    assert isinstance(records, list) and records, "expected a non-empty list of records"
    missing = required - set(records[0])
    assert not missing, f"missing keys: {missing}"


def _check_rtsw_has_active(records):
    assert any(r["active"] for r in records), "no active spacecraft rows"


@pytest.mark.parametrize("feed", JSON_CONTRACTS)
def test_fixture_contract(feed):
    fixture, _, required = JSON_CONTRACTS[feed]
    records = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    _check_records(records, required)
    if feed.startswith("rtsw"):
        _check_rtsw_has_active(records)


def test_fixture_gfz_and_open_meteo():
    gfz = json.loads((FIXTURES / "gfz_kp_nowcast.json").read_text(encoding="utf-8"))
    assert {"Kp", "datetime", "status"} <= set(gfz)
    assert len(gfz["Kp"]) == len(gfz["datetime"])

    om = json.loads((FIXTURES / "open_meteo_munich.json").read_text(encoding="utf-8"))
    assert {"time", "cloud_cover", "cloud_cover_low"} <= set(om["hourly"])


def test_fixture_noaa_text_products():
    three_day = (FIXTURES / "noaa_3day_forecast.txt").read_text(encoding="utf-8")
    assert "NOAA Kp index breakdown" in three_day
    outlook = (FIXTURES / "noaa_27day_outlook.txt").read_text(encoding="utf-8")
    assert "27-day Space Weather Outlook" in outlook


@pytest.mark.network
@pytest.mark.parametrize("feed", JSON_CONTRACTS)
def test_live_contract(feed):
    _, url, required = JSON_CONTRACTS[feed]
    records = httpx.get(url, timeout=30).json()
    _check_records(records, required)
    if feed.startswith("rtsw"):
        _check_rtsw_has_active(records)
