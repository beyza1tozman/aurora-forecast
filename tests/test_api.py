"""API tests: every external feed mocked from tests/fixtures, clock frozen."""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import ForecastService
from aurora import config
from aurora.briefing import Briefer, BriefingCache
from aurora.model import load_models
from aurora.outlook import NoaaCalibration
from tests.conftest import FIXTURE_NOW, fixture_transport

MUNICH = {"lat": 48.14, "lon": 11.58}


@pytest.fixture(scope="module")
def models():
    return load_models()


def make_client(models, fail=frozenset(), tmp_path=None) -> TestClient:
    service = ForecastService(
        client=httpx.Client(transport=fixture_transport(fail)),
        models=models,
        noaa_calibration=NoaaCalibration.load(),
        clock=lambda: FIXTURE_NOW,
    )
    # Template-only briefer: tests never call the real LLM, even if .env has a key.
    briefer = Briefer(
        cache=BriefingCache(tmp_path / "b.sqlite") if tmp_path else None,
        client_factory=lambda: None,
    )
    return TestClient(create_app(service, briefer))


def test_health(models):
    r = make_client(models).get("/health")
    assert r.status_code == 200
    assert r.json()["models"] == [1, 3, 6]


def test_forecast_shape(models):
    r = make_client(models).get("/api/forecast", params=MUNICH)
    assert r.status_code == 200
    body = r.json()
    assert body["errors"] == {}
    assert body["location"]["kp_needed"] > 7  # Munich needs a big storm

    now = body["now"]
    assert 1 <= len(now["hours"]) <= 6
    for hour in now["hours"]:
        assert 0 <= hour["p_kp"] <= 1
        assert hour["chance"] <= hour["chance_if_clear"] <= hour["p_kp"]
    for h in now["kp_forecast"]["horizons"]:
        p = h["p_kp_at_least"]
        assert p["7"] <= p["6"] <= p["5"]

    assert {h["confidence"] for h in now["hours"]} <= {"high", "medium"}

    nights = body["nights"]["nights"]
    assert 1 <= len(nights) <= 3
    # Tonight starts within the model's reach, later nights only have NOAA.
    assert nights[0]["sources"] == ["model", "noaa"]
    assert all(n["sources"] == ["noaa"] for n in nights[1:])
    for night in nights:
        if night["cloud_forecast_complete"]:
            lo, hi = night["chance_range"]
            assert lo <= night["chance"] <= hi
        else:
            assert night["chance"] is None

    assert body["weeks"]["confidence"] == "low"


def test_solar_wind_outage_degrades_only_now_panel(models):
    client = make_client(models, fail={config.NOAA_RTSW_MAG_URL})
    body = client.get("/api/forecast", params=MUNICH).json()
    assert body["now"] is None
    assert "solar_wind" in body["errors"]
    assert body["nights"] is not None and body["weeks"] is not None


def test_cloud_outage_keeps_chance_if_clear(models):
    client = make_client(models, fail={config.OPEN_METEO_URL})
    body = client.get("/api/forecast", params=MUNICH).json()
    assert "clouds" in body["errors"]
    hours = body["now"]["hours"]
    assert all(h["chance"] is None and h["chance_if_clear"] is not None for h in hours)


@pytest.mark.parametrize("params", [{"lat": 95, "lon": 0}, {"lat": 50}, {"lat": "x", "lon": 0}])
def test_bad_coordinates(models, params):
    assert make_client(models).get("/api/forecast", params=params).status_code == 422


def test_briefing_endpoint(models, tmp_path):
    r = make_client(models, tmp_path=tmp_path).get(
        "/api/briefing", params={**MUNICH, "tz": "Europe/Berlin", "place": "Munich"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "template"
    assert body["tz"] == "Europe/Berlin"
    assert "Tonight" in body["text"]


def test_briefing_unknown_tz_uses_utc(models, tmp_path):
    r = make_client(models, tmp_path=tmp_path).get(
        "/api/briefing", params={**MUNICH, "tz": "Mars/Olympus"}
    )
    assert r.json()["tz"] == "UTC"


def test_dashboard_is_served(models):
    client = make_client(models)
    r = client.get("/")
    assert r.status_code == 200
    assert "Aurora Forecast" in r.text
    for asset in ("/js/app.js", "/css/app.css", "/img/favicon.svg"):
        assert client.get(asset).status_code == 200


def test_view_lines(models):
    body = make_client(models).get("/api/view-lines").json()
    lines = {line["id"]: line for line in body["lines"]}
    assert set(lines) == {"now", "forecast"}
    # A 10% forecast line is at least as active as now, so it lies further south.
    assert lines["forecast"]["kp"] >= lines["now"]["kp"] - 1
    at_10e = {lon: lat for lat, lon in lines["now"]["points"]}
    assert 55 < at_10e[10.0] < 75  # quiet conditions: Scandinavia
