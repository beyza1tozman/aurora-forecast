"""Open-Meteo hourly cloud cover by layer (no API key)."""

import httpx
import numpy as np
import pandas as pd

from aurora import config

FORECAST_DAYS = 4  # covers the morning after the third night
LAYERS = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]


def parse_clouds(payload: dict) -> pd.DataFrame:
    """Hourly cloud cover per layer as fractions 0..1, UTC index.

    If the mid/high layers are missing, everything above the low layer is
    counted as mid cloud (the conservative choice: it blocks the view).
    """
    hourly = payload["hourly"]
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"]).tz_localize("UTC"), name="time")
    df = (
        pd.DataFrame({c: hourly[c] for c in LAYERS if c in hourly}, index=index, dtype=float)
        / 100.0
    )
    if "cloud_cover_mid" not in df:
        df["cloud_cover_mid"] = np.clip(df["cloud_cover"] - df["cloud_cover_low"], 0.0, 1.0)
    if "cloud_cover_high" not in df:
        df["cloud_cover_high"] = 0.0
    return df[LAYERS]


def fetch_clouds(client: httpx.Client, lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(LAYERS),
        "forecast_days": FORECAST_DAYS,
        "timezone": "UTC",
    }
    r = client.get(config.OPEN_METEO_URL, params=params)
    r.raise_for_status()
    return parse_clouds(r.json())
