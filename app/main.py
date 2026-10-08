"""FastAPI app: /api/forecast, /api/view-lines, /api/briefing, /api/monitoring, /health,
the dashboard and /monitoring page (app/static), and the background scheduler."""

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.engine import Engine

from app import db
from app.scheduler import Jobs
from app.scheduler import start as start_scheduler
from app.services import ForecastService
from aurora import monitoring
from aurora.briefing import Briefer, resolve_tz

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if os.environ.get("SENTRY_DSN"):
    import sentry_sdk

    # Errors only (no tracing): stays well inside Sentry's free plan.
    sentry_sdk.init(dsn=os.environ["SENTRY_DSN"], traces_sample_rate=0.0, send_default_pii=False)

STATIC_DIR = Path(__file__).parent / "static"
MONITORING_SERIES_DAYS = 7


def create_app(
    service: ForecastService | None = None,
    briefer: Briefer | None = None,
    engine: Engine | None = None,
    scheduler: bool = False,
) -> FastAPI:
    # Built lazily so importing the module (e.g. in tests) does not load models or open clients.
    state: dict = {}
    if service is not None:
        state["service"] = service
    if briefer is not None:
        state["briefer"] = briefer
    if engine is not None:
        state["engine"] = engine

    def get_service() -> ForecastService:
        if "service" not in state:
            state["service"] = ForecastService()
        return state["service"]

    def get_briefer() -> Briefer:
        if "briefer" not in state:
            state["briefer"] = Briefer()
        return state["briefer"]

    def get_engine() -> Engine:
        if "engine" not in state:
            state["engine"] = db.make_engine()
        return state["engine"]

    def get_jobs() -> Jobs:
        if "jobs" not in state:
            state["jobs"] = Jobs(get_service(), get_engine())
        return state["jobs"]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        sched = start_scheduler(get_jobs()) if scheduler else None
        yield
        if sched is not None:
            sched.shutdown(wait=False)

    app = FastAPI(title="Aurora Forecast", version="0.1.0", lifespan=lifespan)

    # HEAD too: UptimeRobot's free plan checks with HEAD requests.
    @app.api_route("/health", methods=["GET", "HEAD"])
    def health() -> dict:
        try:
            svc = get_service()
        except Exception as err:  # models missing or unreadable
            raise HTTPException(status_code=503, detail=f"service not ready: {err!r}") from err
        return {
            "status": "ok",
            "models": sorted(svc.models),
            "time": svc.clock().isoformat(),
        }

    @app.get("/api/forecast")
    def forecast(
        lat: float = Query(..., ge=-90, le=90),
        lon: float = Query(..., ge=-180, le=180),
    ) -> dict:
        return get_service().forecast(lat, lon)

    @app.get("/api/view-lines")
    def view_lines() -> dict:
        return get_service().view_lines()

    @app.get("/api/briefing")
    def briefing(
        lat: float = Query(..., ge=-90, le=90),
        lon: float = Query(..., ge=-180, le=180),
        tz: str = Query("UTC", max_length=64, description="IANA time zone of the viewer"),
        place: str | None = Query(None, max_length=80),
    ) -> dict:
        zone = resolve_tz(tz)
        fc = get_service().forecast(lat, lon)
        result = get_briefer().brief(fc, zone, place).to_dict()
        return {**result, "generated": fc["generated"], "tz": str(zone)}

    @app.get("/api/monitoring")
    def monitoring_data() -> dict:
        engine = get_engine()
        now = get_service().clock()
        since = now - pd.Timedelta(days=MONITORING_SERIES_DAYS)
        try:
            pairs = db.verified_pairs(engine)
            observed = db.observed_series(engine, since)
            counts = db.counts(engine)
        except Exception as err:  # noqa: BLE001 - external DB down: say so, don't 500
            raise HTTPException(status_code=503, detail=f"database unavailable: {err!r}") from err
        recent = pairs[pairs["interval_start"] >= since] if not pairs.empty else pairs
        return {
            "generated": now.isoformat(),
            "database": "postgres" if db.is_postgres(engine) else "sqlite",
            "counts": {
                **{
                    k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
                    for k, v in counts.items()
                },
                "verified": int(len(pairs)),
            },
            "scores": monitoring.scores(pairs),
            "daily_brier": monitoring.daily_brier(pairs, k=4),
            "series": {
                "days": MONITORING_SERIES_DAYS,
                "observed": [
                    {"interval_start": t.isoformat(), "kp": float(v)} for t, v in observed.items()
                ],
                "forecast_h3_kp4": monitoring.forecast_series(recent, horizon=3, k=4),
            },
            "jobs": get_jobs().last_run if scheduler or "jobs" in state else {},
        }

    @app.get("/api/sentry-test", include_in_schema=False)
    def sentry_test(key: str = Query("", max_length=128)) -> dict:
        # Deliberate error to check that Sentry receives events. Hidden (404) unless
        # SENTRY_TEST_KEY is set and matches, so it can't be used to spam the error quota.
        expected = os.environ.get("SENTRY_TEST_KEY", "")
        if not expected or not secrets.compare_digest(key, expected):
            raise HTTPException(status_code=404)
        raise RuntimeError("Sentry test error (triggered via /api/sentry-test)")

    @app.get("/monitoring", include_in_schema=False)
    def monitoring_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "monitoring.html")

    # Mounted last so the API routes above take precedence.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app(scheduler=os.environ.get("SCHEDULER", "1") == "1")
