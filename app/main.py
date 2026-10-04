"""FastAPI app: /api/forecast and /health. Frontend and monitoring follow on days 5-6."""

from fastapi import FastAPI, HTTPException, Query

from app.services import ForecastService


def create_app(service: ForecastService | None = None) -> FastAPI:
    app = FastAPI(title="Aurora Forecast", version="0.1.0")
    # Built lazily so importing the module (e.g. in tests) does not load models or open clients.
    state: dict[str, ForecastService] = {}
    if service is not None:
        state["service"] = service

    def get_service() -> ForecastService:
        if "service" not in state:
            state["service"] = ForecastService()
        return state["service"]

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

    return app


app = create_app()
