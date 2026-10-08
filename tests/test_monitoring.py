"""Forecast log, observations, live verification scores and /api/monitoring (SQLite in tmp)."""

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.scheduler import Jobs
from app.services import ForecastService
from aurora import monitoring
from aurora.briefing import Briefer, BriefingCache
from aurora.live import HorizonForecast, KpForecast
from aurora.model import load_models
from aurora.outlook import NoaaCalibration
from tests.conftest import FIXTURE_NOW, fixture_transport

T0 = pd.Timestamp("2026-10-01 00:00", tz="UTC")
QUIET = np.array([0.9, 0.08, 0.015, 0.004, 0.001])
ACTIVE = np.array([0.2, 0.5, 0.2, 0.08, 0.02])


@pytest.fixture
def engine(tmp_path):
    return db.make_engine(f"sqlite:///{(tmp_path / 'm.sqlite').as_posix()}")


def kp_forecast(issued: pd.Timestamp, probs: np.ndarray, kp_last: float = 2.0) -> KpForecast:
    horizons = [
        HorizonForecast(h, (issued + pd.Timedelta(hours=h)).floor("3h"), probs) for h in (1, 3, 6)
    ]
    return KpForecast(issued=issued, kp_last=kp_last, horizons=horizons, inputs_missing=[])


def test_normalize_url():
    assert db.normalize_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert db.normalize_url("postgresql://u:p@h/db?sslmode=require").startswith(
        "postgresql+psycopg://"
    )
    assert db.normalize_url("sqlite:///x.sqlite") == "sqlite:///x.sqlite"


def test_log_forecast_is_idempotent(engine):
    fc = kp_forecast(T0, QUIET)
    assert db.log_forecast(engine, fc, T0) == 3
    assert db.log_forecast(engine, fc, T0 + pd.Timedelta("15min")) == 0
    assert db.counts(engine)["forecasts_logged"] == 3


def test_observations_are_overwritten(engine):
    idx = pd.date_range(T0, periods=3, freq="3h")
    db.upsert_observations(engine, pd.Series([1.0, 2.0, np.nan], index=idx), T0)
    db.upsert_observations(engine, pd.Series([1.0, 2.33], index=idx[:2]), T0)
    obs = db.observed_series(engine, T0)
    assert obs.tolist() == [1.0, 2.33]
    assert obs.index[0] == T0  # timezone-aware UTC back from SQLite


def test_verified_pairs_and_scores(engine):
    # Quiet forecasts on quiet intervals, one active forecast before a Kp 5 interval.
    for i in range(8):
        db.log_forecast(engine, kp_forecast(T0 + pd.Timedelta(hours=3 * i), QUIET, 1.0), T0)
    storm_issue = T0 + pd.Timedelta(hours=24)
    db.log_forecast(engine, kp_forecast(storm_issue, ACTIVE, 3.0), T0)
    kp = pd.Series(1.0, index=pd.date_range(T0, periods=12, freq="3h"))
    kp[(storm_issue + pd.Timedelta(hours=1)).floor("3h")] = 4.67  # "5-" counts as Kp 5
    db.upsert_observations(engine, kp, T0)

    pairs = db.verified_pairs(engine)
    assert len(pairs) == 27
    assert set(pairs["horizon_h"]) == {1, 3, 6}

    rows = {(r["horizon_h"], r["threshold"]): r for r in monitoring.scores(pairs)}
    r = rows[(1, 4)]
    assert r["n"] == 9 and r["events"] == 1
    assert 0 <= r["brier"] < r["brier_climatology"]
    # Persistence never forecast the event (kp_last 3) -> beaten.
    assert r["bss_persistence"] > 0

    daily = monitoring.daily_brier(pairs, k=4)
    assert {d["horizon_h"] for d in daily} == {1, 3, 6}
    series = monitoring.forecast_series(pairs, horizon=3, k=4)
    assert all(0 <= s["p"] <= 1 for s in series)


def test_scores_empty():
    empty = pd.DataFrame()
    assert monitoring.scores(empty) == []
    assert monitoring.daily_brier(empty) == []
    assert monitoring.forecast_series(empty, 3, 4) == []


@pytest.fixture(scope="module")
def service():
    return ForecastService(
        client=httpx.Client(transport=fixture_transport()),
        models=load_models(),
        noaa_calibration=NoaaCalibration.load(),
        clock=lambda: FIXTURE_NOW,
    )


def test_jobs_log_and_verify_live_fixture(engine, service):
    jobs = Jobs(service, engine)
    assert jobs.log_forecast() == 3
    assert jobs.log_forecast() == 0  # same issue time
    assert jobs.verify() > 0
    assert jobs.last_run["log_forecast"]["ok"] and jobs.last_run["verify"]["ok"]


def test_monitoring_api(engine, service, tmp_path):
    briefer = Briefer(cache=BriefingCache(tmp_path / "b.sqlite"), client_factory=lambda: None)
    client = TestClient(create_app(service, briefer, engine))
    body = client.get("/api/monitoring").json()
    assert body["database"] == "sqlite"
    assert body["counts"]["verified"] == 0 and body["scores"] == []

    Jobs(service, engine).verify()
    for i in range(4):
        issued = FIXTURE_NOW.floor("h") - pd.Timedelta(hours=24 - 3 * i)
        db.log_forecast(engine, kp_forecast(issued, QUIET), FIXTURE_NOW)
    body = client.get("/api/monitoring").json()
    assert body["counts"]["verified"] > 0
    assert body["series"]["observed"]
    assert client.get("/monitoring").status_code == 200
