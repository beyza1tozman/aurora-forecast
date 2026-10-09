"""MET Norway Locationforecast cloud cover by layer (no API key), the fallback for Open-Meteo.

MET's terms ask for an identifying User-Agent and at most 4 decimals in coordinates.
Steps are hourly for the first ~2.5 days, then 6-hourly; the 6-hourly part is
interpolated to hours so the third night still gets a (coarser) cloud estimate.
"""

import httpx
import pandas as pd

from aurora import config
from aurora.data.open_meteo import LAYERS

# MET name -> our column name (same as Open-Meteo's)
MET_LAYERS = {
    "cloud_area_fraction": "cloud_cover",
    "cloud_area_fraction_low": "cloud_cover_low",
    "cloud_area_fraction_medium": "cloud_cover_mid",
    "cloud_area_fraction_high": "cloud_cover_high",
}
MAX_GAP_H = 6  # interpolate across the 6-hourly steps, never further


def parse_clouds_met(payload: dict) -> pd.DataFrame:
    """Hourly cloud cover per layer as fractions 0..1, UTC index (same shape as Open-Meteo)."""
    rows = {
        pd.Timestamp(step["time"]): step["data"]["instant"]["details"]
        for step in payload["properties"]["timeseries"]
    }
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(MET_LAYERS))
    df = df.rename(columns=MET_LAYERS).astype(float) / 100.0
    df.index = pd.DatetimeIndex(df.index, name="time").tz_convert("UTC")
    hourly = pd.date_range(df.index[0], df.index[-1], freq="1h", name="time")
    df = df.reindex(hourly).interpolate(method="time", limit=MAX_GAP_H - 1, limit_area="inside")
    return df[LAYERS].clip(0.0, 1.0)


def fetch_clouds_met(client: httpx.Client, lat: float, lon: float) -> pd.DataFrame:
    r = client.get(
        config.MET_NORWAY_URL,
        params={"lat": round(lat, 4), "lon": round(lon, 4)},
        headers={"User-Agent": config.USER_AGENT},
    )
    r.raise_for_status()
    return parse_clouds_met(r.json())
