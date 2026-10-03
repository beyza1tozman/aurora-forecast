"""Save trimmed samples of every live feed to tests/fixtures/ for offline tests.

Re-run when a provider changes its format:
    python scripts/save_live_fixtures.py
"""

import json
from datetime import UTC, datetime, timedelta

import httpx

from aurora import config

FIXTURES = config.ROOT / "tests" / "fixtures"

# Keep the last N hours of the 1-min feeds so fixtures stay small but still
# contain rows from several spacecraft (active and inactive).
RTSW_KEEP_HOURS = 3


def _recent(records: list[dict], hours: int) -> list[dict]:
    latest = max(datetime.fromisoformat(r["time_tag"]) for r in records)
    cutoff = latest - timedelta(hours=hours)
    return [r for r in records if datetime.fromisoformat(r["time_tag"]) >= cutoff]


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

    json_feeds = {
        "noaa_rtsw_mag_1m.json": (config.NOAA_RTSW_MAG_URL, None, RTSW_KEEP_HOURS),
        "noaa_rtsw_wind_1m.json": (config.NOAA_RTSW_WIND_URL, None, RTSW_KEEP_HOURS),
        "noaa_rtsw_ephemerides_1h.json": (config.NOAA_RTSW_EPHEMERIS_URL, None, 24),
        "noaa_kp.json": (config.NOAA_KP_URL, None, None),
        "noaa_kp_forecast.json": (config.NOAA_KP_FORECAST_URL, None, None),
        "gfz_kp_nowcast.json": (
            config.GFZ_KP_API_URL,
            {
                "start": f"{now - timedelta(days=2):%Y-%m-%dT%H:00:00Z}",
                "end": f"{now:%Y-%m-%dT%H:00:00Z}",
                "index": "Kp",
                "status": "all",
            },
            None,
        ),
        "open_meteo_munich.json": (
            config.OPEN_METEO_URL,
            {
                "latitude": 48.14,
                "longitude": 11.58,
                "hourly": "cloud_cover,cloud_cover_low",
                "forecast_days": 3,
                "timezone": "UTC",
            },
            None,
        ),
    }
    text_feeds = {
        "noaa_3day_forecast.txt": config.NOAA_3DAY_FORECAST_URL,
        "noaa_27day_outlook.txt": config.NOAA_27DAY_OUTLOOK_URL,
    }

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for name, (url, params, keep_hours) in json_feeds.items():
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            if keep_hours:
                data = _recent(data, keep_hours)
            (FIXTURES / name).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
            print(f"saved {name}")

        for name, url in text_feeds.items():
            r = client.get(url)
            r.raise_for_status()
            (FIXTURES / name).write_text(r.text, encoding="utf-8")
            print(f"saved {name}")


if __name__ == "__main__":
    main()
