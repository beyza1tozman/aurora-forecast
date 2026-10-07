"""FastAPI app: /api/forecast, /api/view-lines, /api/briefing, /health and the
dashboard (app/static). Monitoring follows on day 6."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from app.services import ForecastService
from aurora.briefing import Briefer, resolve_tz

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"


def create_app(service: ForecastService | None = None, briefer: Briefer | None = None) -> FastAPI:
    app = FastAPI(title="Aurora Forecast", version="0.1.0")
    # Built lazily so importing the module (e.g. in tests) does not load models or open clients.
    state: dict = {}
    if service is not None:
        state["service"] = service
    if briefer is not None:
        state["briefer"] = briefer

    def get_service() -> ForecastService:
        if "service" not in state:
            state["service"] = ForecastService()
        return state["service"]

    def get_briefer() -> Briefer:
        if "briefer" not in state:
            state["briefer"] = Briefer()
        return state["briefer"]

    @app.get("/health")
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

    # Mounted last so the API routes above take precedence.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
