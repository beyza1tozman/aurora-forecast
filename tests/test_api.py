"""API tests: every external feed mocked from tests/fixtures, clock frozen."""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import ForecastService
from aurora import config
from aurora.model import load_models
from aurora.outlook import NoaaCalibration
from tests.conftest import FIXTURE_NOW, fixture_transport

MUNICH = {"lat": 48.14, "lon": 11.58}


@pytest.fixture(scope="module")
def models():
    return load_models()


def make_client(models, fail=frozenset()) -> TestClient:
    service = ForecastService(
        client=httpx.Client(transport=fixture_transport(fail)),
        models=models,
        noaa_calibration=NoaaCalibration.load(),
        clock=lambda: FIXTURE_NOW,
    )
    return TestClient(create_app(service))


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
