"""Assemble the forecast for one location from live feeds, models and sky conditions.

Global inputs (solar wind, Kp, NOAA forecasts) are the same for every location
and are cached for GLOBAL_TTL. Clouds are cached per ~0.1 degree cell for
CLOUD_TTL. If one source fails, only the panel that needs it is marked
unavailable; the rest of the response still works.
"""

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import pandas as pd

from aurora import config
from aurora.data.noaa_live import fetch_json, fetch_kp, fetch_solar_wind
from aurora.data.open_meteo import fetch_clouds
from aurora.live import KpForecast, forecast_kp
from aurora.location_score import combine, location_info, sky_conditions
from aurora.model import HorizonModel, load_models
from aurora.outlook import (
    NoaaCalibration,
    hourly_from_intervals,
    lead_days,
    night_windows,
    parse_27day_outlook,
    parse_noaa_kp_forecast,
    probability_range,
    weeks_hint,
)
from aurora.physics.oval import prob_kp_at_least

log = logging.getLogger(__name__)

GLOBAL_TTL = 300.0  # seconds
CLOUD_TTL = 1800.0
CLOUD_GRID = 0.1  # degrees
HOURS_AHEAD = 6
NIGHTS_LOOKAHEAD = pd.Timedelta("84h")
HORIZON_CONFIDENCE = {1: "high", 3: "medium", 6: "medium"}


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class TTLCache:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self._data: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key, compute: Callable[[], Any]):
        with self._lock:
            hit = self._data.get(key)
            if hit and time.monotonic() - hit[0] < self.ttl:
                return hit[1]
        value = compute()
        with self._lock:
            self._data[key] = (time.monotonic(), value)
        return value


@dataclass
class GlobalInputs:
    fetched: pd.Timestamp
    kp_forecast: KpForecast | None = None
    solar_wind_meta: dict | None = None
    kp_history: pd.Series | None = None
    noaa_kp: pd.Series | None = None
    outlook_27day: pd.DataFrame | None = None
    errors: dict[str, str] = field(default_factory=dict)


def _jsonable(value):
    """NaN -> None, numpy scalars -> Python, timestamps -> ISO strings (recursively)."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ForecastService:
    def __init__(
        self,
        client: httpx.Client | None = None,
        models: dict[int, HorizonModel] | None = None,
        noaa_calibration: NoaaCalibration | None = None,
        clock: Callable[[], pd.Timestamp] = utc_now,
    ):
        self.client = client or httpx.Client(timeout=20, follow_redirects=True)
        self.models = models if models is not None else load_models()
        self.noaa_calibration = noaa_calibration or NoaaCalibration.load()
        self.clock = clock
        self._global = TTLCache(GLOBAL_TTL)
        self._clouds = TTLCache(CLOUD_TTL)

    # --- inputs -------------------------------------------------------------

    def global_inputs(self) -> GlobalInputs:
        return self._global.get("global", self._fetch_global)

    def _fetch_global(self) -> GlobalInputs:
        now = self.clock()
        g = GlobalInputs(fetched=now)
        try:
            g.kp_history = fetch_kp(self.client, now)
        except Exception as err:  # noqa: BLE001 - any feed failure degrades one panel
            g.errors["kp"] = repr(err)
        try:
            sw, g.solar_wind_meta = fetch_solar_wind(self.client)
            if g.kp_history is not None:
                g.kp_forecast = forecast_kp(sw, g.kp_history, self.models, now)
        except Exception as err:  # noqa: BLE001
            g.errors["solar_wind"] = repr(err)
        try:
            g.noaa_kp = parse_noaa_kp_forecast(fetch_json(self.client, config.NOAA_KP_FORECAST_URL))
        except Exception as err:  # noqa: BLE001
            g.errors["noaa_3day"] = repr(err)
        try:
            r = self.client.get(config.NOAA_27DAY_OUTLOOK_URL)
            r.raise_for_status()
            g.outlook_27day = parse_27day_outlook(r.text)
        except Exception as err:  # noqa: BLE001
            g.errors["noaa_27day"] = repr(err)
        for source, err in g.errors.items():
            log.warning("feed %s failed: %s", source, err)
        return g

    def clouds(self, lat: float, lon: float) -> pd.DataFrame:
        key = (round(lat / CLOUD_GRID), round(lon / CLOUD_GRID))
        return self._clouds.get(key, lambda: fetch_clouds(self.client, lat, lon))

    # --- panels -------------------------------------------------------------

    def forecast(self, lat: float, lon: float) -> dict:
        now = self.clock()
        g = self.global_inputs()
        loc = location_info(lat, lon, now)
        errors = dict(g.errors)

        hours = pd.date_range(now.floor("h"), now.floor("h") + NIGHTS_LOOKAHEAD, freq="1h")
        try:
            clouds = self.clouds(lat, lon)
        except Exception as err:  # noqa: BLE001
            errors["clouds"] = repr(err)
            log.warning("clouds failed: %s", err)
            clouds = pd.DataFrame(dtype=float)
        sky = sky_conditions(hours, lat, lon, clouds)

        return _jsonable(
            {
                "generated": now,
                "location": loc,
                "now": self._now_panel(g, sky, loc["kp_needed"]),
                "nights": self._nights_panel(g, sky, loc["kp_needed"], now),
                "weeks": self._weeks_panel(g, loc["kp_needed"], now),
                "errors": errors,
            }
        )

    def _now_panel(self, g: GlobalInputs, sky: pd.DataFrame, kp_needed: float) -> dict | None:
        fc = g.kp_forecast
        if fc is None:
            return None
        hours = sky.index[:HOURS_AHEAD]
        p, horizon = model_hourly(fc, hours, kp_needed)
        scored = combine(p, sky.loc[hours])
        rows = [
            {
                "time": t,
                "horizon_h": int(horizon[i]),
                "confidence": HORIZON_CONFIDENCE[int(horizon[i])],
                "p_kp": p[i],
                **{k: scored.iloc[i][k] for k in _SKY_FIELDS},
            }
            for i, t in enumerate(hours)
            if not np.isnan(p[i])
        ]
        return {
            "source": "LightGBM on NOAA real-time solar wind (L1, shifted to the bow shock)",
            "kp_forecast": fc.to_dict(),
            "solar_wind": g.solar_wind_meta,
            "hours": rows,
        }

    def _nights_panel(
        self, g: GlobalInputs, sky: pd.DataFrame, kp_needed: float, now
    ) -> dict | None:
        """Per hour: the model where it reaches, calibrated NOAA beyond."""
        p_model = np.full(len(sky), np.nan)
        if g.kp_forecast is not None:
            p_model, _ = model_hourly(g.kp_forecast, sky.index, kp_needed)
        p_noaa = lo_noaa = hi_noaa = noaa_kp = np.full(len(sky), np.nan)
        if g.noaa_kp is not None and not g.noaa_kp.empty:
            intervals = g.noaa_kp.index
            probs, n_eff = self.noaa_calibration.class_probs(
                g.noaa_kp.to_numpy(), lead_days(intervals, now)
            )
            p_interval = prob_kp_at_least(probs, kp_needed)
            lo, hi = probability_range(p_interval, n_eff)
            per_interval = np.column_stack([p_interval, lo, hi, g.noaa_kp.to_numpy()])
            p_noaa, lo_noaa, hi_noaa, noaa_kp = hourly_from_intervals(
                intervals, per_interval, sky.index
            ).T
        if np.isnan(p_model).all() and np.isnan(p_noaa).all():
            return None

        use_model = ~np.isnan(p_model)
        scored = combine(np.where(use_model, p_model, p_noaa), sky)
        scored["source"] = np.where(use_model, "model", np.where(np.isnan(p_noaa), "", "noaa"))
        scored["noaa_kp"] = noaa_kp
        # The model is calibrated and has no table uncertainty; NOAA hours get the Beta range.
        weight = scored["darkness"] * scored["moon"] * scored["clear_sky"]
        scored["chance_lo"] = np.where(use_model, p_model, lo_noaa) * weight
        scored["chance_hi"] = np.where(use_model, p_model, hi_noaa) * weight

        nights = []
        for window in night_windows(scored):
            night = scored.loc[window]
            if night["p_kp"].isna().all():
                continue  # beyond both forecasts
            has_clouds = bool(night["clear_sky"].notna().all())
            key = "chance" if has_clouds else "chance_if_clear"
            best = night[key].idxmax()
            nights.append(
                {
                    "start": window[0],
                    "end": window[-1] + pd.Timedelta("1h"),
                    "best_hour": best,
                    "chance": night.loc[best, "chance"] if has_clouds else None,
                    "chance_range": [night.loc[best, "chance_lo"], night.loc[best, "chance_hi"]]
                    if has_clouds
                    else None,
                    "chance_if_clear": night["chance_if_clear"].max(),
                    "p_kp_max": night["p_kp"].max(),
                    "noaa_kp_max": night["noaa_kp"].max(),
                    "sources": sorted(set(night["source"]) - {""}),
                    "cloud_cover_mean": night["cloud_cover"].mean() if has_clouds else None,
                    "moon_illumination": night["moon_illumination"].mean(),
                    "cloud_forecast_complete": has_clouds,
                }
            )
        return {
            "confidence": "medium-low",
            "source": "Model for the next hours, then NOAA SWPC 3-day Kp forecast "
            "calibrated on archived forecasts 2022-2026",
            "nights": nights,
        }

    def _weeks_panel(self, g: GlobalInputs, kp_needed: float, now) -> dict | None:
        if g.outlook_27day is None:
            return None
        history = g.kp_history if g.kp_history is not None else pd.Series(dtype=float)
        return weeks_hint(g.outlook_27day, history, kp_needed, now)


_SKY_FIELDS = (
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "clear_sky",
    "darkness",
    "moon",
    "moon_illumination",
    "sun_alt",
    "chance_if_clear",
    "chance",
)


def _horizon_for_hour(fc: KpForecast, t: pd.Timestamp):
    """Shortest horizon whose interval contains t; before the first interval use the
    shortest horizon; after the last interval nothing (beyond the model's reach)."""
    containing = [h for h in fc.horizons if h.interval_start <= t < h.interval_end]
    if containing:
        return min(containing, key=lambda h: h.horizon_h)
    if t < min(h.interval_start for h in fc.horizons):
        return min(fc.horizons, key=lambda h: h.horizon_h)
    return None


def model_hourly(fc: KpForecast, hours: pd.DatetimeIndex, kp_needed: float):
    """(P(Kp >= kp_needed), horizon used) per hour; NaN / 0 where the model does not reach."""
    p = np.full(len(hours), np.nan)
    horizon = np.zeros(len(hours), dtype=int)
    for i, t in enumerate(hours):
        chosen = _horizon_for_hour(fc, t)
        if chosen is not None:
            p[i] = prob_kp_at_least(chosen.class_probs[None, :], kp_needed)[0]
            horizon[i] = chosen.horizon_h
    return p, horizon
