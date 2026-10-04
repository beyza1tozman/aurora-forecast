"""Live inputs: NOAA real-time solar wind (RTSW) and recent Kp, shaped like the training data.

Solar wind
----------
RTSW gives 1-min magnetic field and plasma measured *at L1* by several
spacecraft. Only rows with ``active == true`` are used (NOAA's operational
choice, with automatic failover). OMNI, which the model was trained on, is
time-shifted to the bow-shock nose, so each L1 sample is moved forward by
``(x_gse - BOW_SHOCK_X) / V`` using the spacecraft's measured distance.
The shifted samples are then averaged into hourly rows [t, t+1h), like OMNI.

Kp
--
The model needs ~28 days of Kp (``kp_27d``). GFZ's nowcast API provides that
history and is closest to the definitive Kp used in training; NOAA's estimated
Kp fills intervals GFZ has not published yet. GFZ already lists the interval in
progress (with a provisional value), so intervals that have not ended are dropped.
"""

import httpx
import numpy as np
import pandas as pd

from aurora import config
from aurora.features import KP_INTERVAL, SW_COLUMNS

BOW_SHOCK_X_KM = 90_000.0  # ~14 Earth radii, typical bow-shock nose distance
DEFAULT_L1_X_KM = 1.5e6  # used if no ephemeris row is available
SPEED_SMOOTHING = "10min"  # rolling median for the propagation speed
MIN_SAMPLES_PER_HOUR = 15  # minutes of data needed for an hourly average
KP_HISTORY = pd.Timedelta("30D")

MAG_COLUMNS = {"bt": "b_mag", "by_gsm": "by_gsm", "bz_gsm": "bz_gsm"}
WIND_COLUMNS = {
    "proton_speed": "speed",
    "proton_density": "density",
    "proton_temperature": "temperature",
}


def _active_frame(records: list[dict], columns: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    df = df[df["active"].astype(bool)]
    times = pd.to_datetime(df["time_tag"], utc=True).dt.floor("min")
    out = df[list(columns)].rename(columns=columns).apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(times, name="time")
    return out.groupby(level=0).mean()  # duplicates after flooring to the minute


def rtsw_l1(mag: list[dict], wind: list[dict]) -> pd.DataFrame:
    """1-min solar wind at L1 from the active spacecraft, with SW_COLUMNS."""
    df = _active_frame(mag, MAG_COLUMNS).join(_active_frame(wind, WIND_COLUMNS), how="outer")
    return df[SW_COLUMNS].sort_index()


def l1_distance_km(ephemerides: list[dict]) -> float:
    """Latest x_GSE of the active spacecraft (km), or a default."""
    rows = [r for r in ephemerides if r.get("active") and r.get("x_gse") is not None]
    if not rows:
        return DEFAULT_L1_X_KM
    return float(max(rows, key=lambda r: r["time_tag"])["x_gse"])


def shift_to_bow_shock(l1: pd.DataFrame, x_gse_km: float) -> pd.DataFrame:
    """Move each L1 sample forward by its travel time to the bow-shock nose."""
    speed = l1["speed"].rolling(SPEED_SMOOTHING, min_periods=1).median().ffill().bfill()
    delay = pd.to_timedelta((x_gse_km - BOW_SHOCK_X_KM) / speed, unit="s")
    shifted = l1.copy()
    shifted.index = l1.index + delay.to_numpy()
    return shifted.sort_index()


def hourly_average(shifted: pd.DataFrame) -> pd.DataFrame:
    """Hourly means [t, t+1h) like OMNI. The newest, still-filling hour is dropped."""
    if shifted.empty:
        return pd.DataFrame(columns=SW_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
    hourly = shifted.resample("1h").mean()
    counts = shifted.resample("1h").count()
    hourly = hourly.where(counts >= MIN_SAMPLES_PER_HOUR)
    complete = hourly.index + pd.Timedelta("1h") <= shifted.index.max()
    return hourly[complete]


def solar_wind_hourly(mag, wind, ephemerides) -> tuple[pd.DataFrame, dict]:
    """Hourly bow-shock solar wind ready for aurora.features, plus metadata."""
    l1 = rtsw_l1(mag, wind)
    x = l1_distance_km(ephemerides)
    hourly = hourly_average(shift_to_bow_shock(l1, x))
    speed = l1["speed"].dropna()
    meta = {
        "l1_distance_km": x,
        "latest_l1_sample": l1.index.max() if len(l1) else None,
        "propagation_minutes": (
            float((x - BOW_SHOCK_X_KM) / speed.iloc[-1] / 60) if len(speed) else None
        ),
        "spacecraft": sorted({r["source"] for r in mag if r.get("active")}),
    }
    return hourly, meta


def _thirds(kp) -> np.ndarray:
    return np.round(np.asarray(kp, dtype=float) * 3) / 3


def parse_noaa_kp(records: list[dict]) -> pd.Series:
    """NOAA estimated planetary Kp, indexed by 3-hour interval start (UTC)."""
    df = pd.DataFrame.from_records(records)
    index = pd.DatetimeIndex(pd.to_datetime(df["time_tag"], utc=True), name="time")
    return pd.Series(_thirds(df["Kp"]), index=index, name="kp").sort_index()


def parse_gfz_kp(payload: dict) -> pd.Series:
    """GFZ nowcast API response ({"datetime": [...], "Kp": [...]}) as a Kp series."""
    index = pd.DatetimeIndex(pd.to_datetime(payload["datetime"], utc=True), name="time")
    return pd.Series(_thirds(payload["Kp"]), index=index, name="kp").sort_index()


def combine_kp(gfz: pd.Series, noaa: pd.Series, now: pd.Timestamp) -> pd.Series:
    """Completed intervals only: GFZ where available, else NOAA. Regular 3-h index."""
    kp = gfz.combine_first(noaa)
    kp = kp[(kp.index == kp.index.floor(KP_INTERVAL)) & (kp.index + KP_INTERVAL <= now)]
    return kp.asfreq(KP_INTERVAL)


def fetch_json(client: httpx.Client, url: str, params: dict | None = None):
    r = client.get(url, params=params)
    r.raise_for_status()
    return r.json()


def fetch_solar_wind(client: httpx.Client) -> tuple[pd.DataFrame, dict]:
    return solar_wind_hourly(
        fetch_json(client, config.NOAA_RTSW_MAG_URL),
        fetch_json(client, config.NOAA_RTSW_WIND_URL),
        fetch_json(client, config.NOAA_RTSW_EPHEMERIS_URL),
    )


def fetch_kp(client: httpx.Client, now: pd.Timestamp) -> pd.Series:
    start = (now - KP_HISTORY).floor("D")
    params = {
        "start": f"{start:%Y-%m-%dT%H:%M:%SZ}",
        "end": f"{now.floor('h'):%Y-%m-%dT%H:%M:%SZ}",
        "index": "Kp",
        "status": "all",
    }
    gfz = parse_gfz_kp(fetch_json(client, config.GFZ_KP_API_URL, params))
    noaa = parse_noaa_kp(fetch_json(client, config.NOAA_KP_URL))
    return combine_kp(gfz, noaa, now)
