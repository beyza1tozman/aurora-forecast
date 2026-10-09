"""Shared fixtures: every external feed served from tests/fixtures, clock frozen at fixture time."""

from pathlib import Path

import httpx
import pandas as pd
import pytest

from aurora import config

FIXTURES = Path(__file__).parent / "fixtures"
# Fixtures were saved on 2026-10-03 around 16:05 UTC.
FIXTURE_NOW = pd.Timestamp("2026-10-03 16:10", tz="UTC")

ROUTES = {
    config.NOAA_RTSW_MAG_URL: "noaa_rtsw_mag_1m.json",
    config.NOAA_RTSW_WIND_URL: "noaa_rtsw_wind_1m.json",
    config.NOAA_RTSW_EPHEMERIS_URL: "noaa_rtsw_ephemerides_1h.json",
    config.NOAA_KP_URL: "noaa_kp.json",
    config.NOAA_KP_FORECAST_URL: "noaa_kp_forecast.json",
    config.NOAA_3DAY_FORECAST_URL: "noaa_3day_forecast.txt",
    config.NOAA_27DAY_OUTLOOK_URL: "noaa_27day_outlook.txt",
    config.GFZ_KP_API_URL: "gfz_kp_nowcast.json",
    config.OPEN_METEO_URL: "open_meteo_munich.json",
    # Saved 2026-10-09 with timestamps shifted to start at FIXTURE_NOW.
    config.MET_NORWAY_URL: "met_norway_munich.json",
}


def fixture_transport(
    fail: set[str] = frozenset(), status: int = 500, calls: list[str] | None = None
) -> httpx.MockTransport:
    """Serve ROUTES from disk (ignoring query strings); URLs in ``fail`` return ``status``.
    Requested URLs are appended to ``calls`` if given."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url.copy_with(query=None))
        if calls is not None:
            calls.append(url)
        if url in fail:
            return httpx.Response(status)
        name = ROUTES.get(url)
        if name is None:
            return httpx.Response(404)
        return httpx.Response(200, content=(FIXTURES / name).read_bytes())

    return httpx.MockTransport(handler)


@pytest.fixture
def mock_client():
    with httpx.Client(transport=fixture_transport()) as client:
        yield client
