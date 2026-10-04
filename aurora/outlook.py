"""Nights and weeks: NOAA's 3-day Kp forecast (calibrated) and the 27-day outlook.

Next 3 nights
-------------
NOAA forecasts one deterministic Kp per 3-hour interval. Using the archived
forecasts (scripts/evaluate_noaa.py -> models/noaa_calibration.json) it is
turned into class probabilities: given lead day d and forecast class c, how
often was each Kp class observed? Each row is smoothed towards the same
forecast class pooled over all lead days (PRIOR_WEIGHT pseudo-counts), which
also fills cells NOAA never used (e.g. Kp 7+ three days ahead). The range shown
in the UI is a Beta interval from the row's sample size: the uncertainty of
this calibration, not of the whole forecast.

Coming weeks
------------
Text-only hints from NOAA's 27-day outlook (largest daily Kp) and the 27-day
recurrence of Kp. Both have little skill near solar maximum, so they are shown
as "possible activity", never as a number.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from aurora.dataset import kp_to_class
from aurora.features import KP_INTERVAL, RECURRENCE
from aurora.model import MODELS_DIR

PRIOR_WEIGHT = 5.0
RANGE_QUANTILES = (0.1, 0.9)
LEAD_DAYS = (1, 2, 3)
WEEKS_FROM_DAY = 3  # the weeks hint starts after the 3-night outlook
WEEKS_MIN_KP = 4  # outlook days below this are not mentioned


class NoaaCalibration:
    def __init__(self, counts: dict[str, list[list[int]]]):
        self.counts = {int(d): np.asarray(t, dtype=float) for d, t in counts.items()}
        pooled = sum(self.counts.values())
        self.prior = pooled / np.maximum(pooled.sum(axis=1, keepdims=True), 1.0)

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "noaa_calibration.json") -> "NoaaCalibration":
        return cls(json.loads(path.read_text())["counts"])

    def class_probs(self, noaa_kp, lead_day) -> tuple[np.ndarray, np.ndarray]:
        """(probs (n, N_CLASSES), effective sample size (n,)) for NOAA Kp values."""
        fc = kp_to_class(noaa_kp).astype(int)
        lead = np.clip(np.asarray(lead_day, dtype=int), min(LEAD_DAYS), max(LEAD_DAYS))
        rows = np.stack([self.counts[d][c] for d, c in zip(lead, fc, strict=True)])
        n = rows.sum(axis=1) + PRIOR_WEIGHT
        probs = (rows + PRIOR_WEIGHT * self.prior[fc]) / n[:, None]
        return probs, n


def probability_range(p, n_eff) -> tuple[np.ndarray, np.ndarray]:
    """Beta interval for a probability estimated from n_eff (pseudo-)observations."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    a, b = p * n_eff, (1 - p) * n_eff
    return beta.ppf(RANGE_QUANTILES[0], a, b), beta.ppf(RANGE_QUANTILES[1], a, b)


def parse_noaa_kp_forecast(records: list[dict]) -> pd.Series:
    """Predicted (not observed/estimated) Kp per 3-hour interval start (UTC)."""
    rows = [r for r in records if r.get("observed") == "predicted"]
    index = pd.DatetimeIndex(pd.to_datetime([r["time_tag"] for r in rows], utc=True), name="time")
    kp = np.round(np.asarray([r["kp"] for r in rows], dtype=float) * 3) / 3
    return pd.Series(kp, index=index, name="kp").sort_index()


def lead_days(interval_starts: pd.DatetimeIndex, now: pd.Timestamp) -> np.ndarray:
    """1 = today (UT), 2 = tomorrow, 3 = the day after; matches the calibration's 00:30 issues."""
    return np.asarray((interval_starts.floor("D") - now.floor("D")).days + 1)


def hourly_from_intervals(intervals: pd.DatetimeIndex, values: np.ndarray, hours: pd.DatetimeIndex):
    """Map per-3h-interval rows onto hours (row of the interval containing each hour, or NaN)."""
    pos = pd.Index(intervals).get_indexer(hours.floor(KP_INTERVAL))
    out = np.full((len(hours),) + values.shape[1:], np.nan)
    out[pos >= 0] = values[pos[pos >= 0]]
    return out


def night_windows(sky: pd.DataFrame, n_nights: int = 3) -> list[pd.DatetimeIndex]:
    """Consecutive runs of hours with darkness > 0, oldest first."""
    dark = sky["darkness"] > 0
    run_id = (dark != dark.shift()).cumsum()
    nights = [grp.index for _, grp in sky[dark].groupby(run_id[dark])]
    return nights[:n_nights]


def parse_27day_outlook(text: str) -> pd.DataFrame:
    """Daily rows (UTC date index) with radio_flux, ap and kp_max."""
    rows = re.findall(r"^(\d{4} \w{3} \d{2})\s+(\d+)\s+(\d+)\s+(\d+)\s*$", text, flags=re.M)
    index = pd.DatetimeIndex(pd.to_datetime([r[0] for r in rows], format="%Y %b %d", utc=True))
    data = np.asarray([r[1:] for r in rows], dtype=float)
    return pd.DataFrame(data, columns=["radio_flux", "ap", "kp_max"], index=index.rename("date"))


def weeks_hint(
    outlook: pd.DataFrame, kp_history: pd.Series, kp_needed: float, now: pd.Timestamp
) -> dict:
    """Days beyond the 3-night outlook with possible activity, as text-ready rows."""
    start = now.floor("D") + pd.Timedelta(days=WEEKS_FROM_DAY)
    future = outlook[outlook.index >= start]
    daily_max = kp_history.resample("1D").max()
    days = []
    for date, row in future.iterrows():
        recurrence = daily_max.get(date - RECURRENCE)
        if row["kp_max"] < WEEKS_MIN_KP:
            continue
        days.append(
            {
                "date": date.date().isoformat(),
                "outlook_kp_max": int(row["kp_max"]),
                "kp_27_days_ago": None
                if recurrence is None or pd.isna(recurrence)
                else float(recurrence),
                "reaches_kp_needed": bool(row["kp_max"] >= kp_needed),
            }
        )
    return {
        "confidence": "low",
        "source": "NOAA 27-day outlook + 27-day recurrence",
        "outlook_end": outlook.index.max().date().isoformat() if len(outlook) else None,
        "possible_activity": days,
    }
