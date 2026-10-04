"""Score NOAA's archived 3-day Kp forecasts against GFZ definitive Kp.

This is not a comparison with our model: NOAA forecasts 1-3 days ahead, our
model 1-6 hours ahead. It answers a different question: how much skill does
NOAA's day-scale forecast (the source of our "next 3 nights" panel) have over
day-scale baselines?

Downloads the 00:30 UT issues from the NCEI archive (March 2022 onwards) into
data/raw/noaa_3day/, then writes reports/noaa_3day.json and prints a markdown
table. Lead day 1 = the rest of the issue day (the 00-03 UT interval had already
started and is dropped), day 2 = the next UT day, day 3 = the day after.

Baselines (deterministic, like NOAA's forecast):
- persistence: the last completed Kp interval at issue time
- recurrence: Kp of the same interval 27 days earlier
Observed and baseline Kp are GFZ definitive values. In real time NOAA only had
its own estimated Kp, so persistence is slightly flattered here.

Also writes models/noaa_calibration.json: for each lead day and NOAA forecast
Kp class, the counts of observed Kp classes. aurora.outlook turns NOAA's
deterministic forecast into class probabilities with it. It uses the whole
archive, so it is a lookup table for the app, not an out-of-sample result.

Usage:
    python scripts/evaluate_noaa.py
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

from aurora.config import NOAA_3DAY_ARCHIVE_URL, PROCESSED_DIR, RAW_DIR, ROOT
from aurora.data.noaa_3day import parse_3day_forecast
from aurora.dataset import CLASS_LABELS, N_CLASSES, kp_to_class, threshold_class
from aurora.metrics import contingency, hss, tss

ARCHIVE_DIR = RAW_DIR / "noaa_3day"
ARCHIVE_START = pd.Timestamp("2022-03-25")  # first file in the NCEI archive
ISSUE = "0030"
THRESHOLDS = (5, 6, 7)
RECURRENCE = pd.Timedelta("27D")
WORKERS = 2  # NCEI resets connections when hit with more
RETRIES = 5


def fetch_issue(client: httpx.Client, day: pd.Timestamp) -> str | None:
    """Text of the 00:30 issue for ``day`` (cached on disk), or None if NCEI lacks it."""
    name = f"{day:%Y%m%d}{ISSUE}three_day_forecast.txt"
    path = ARCHIVE_DIR / name
    if not path.exists():
        url = f"{NOAA_3DAY_ARCHIVE_URL}/{day:%Y/%m}/{name}"
        for attempt in range(RETRIES):
            try:
                r = client.get(url)
                break
            except httpx.TransportError:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(2**attempt)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        path.write_bytes(r.content)  # as served (write_text doubles CR on Windows)
    return path.read_text(encoding="utf-8")


def download_archive(end: pd.Timestamp) -> list[str]:
    """Fetch every 00:30 issue up to ``end`` that is not on disk yet; return the texts."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(ARCHIVE_START, end, freq="D")
    # ~1 s per request, so a couple of parallel connections help; cached files are skipped.
    with (
        httpx.Client(timeout=60, follow_redirects=True) as client,
        ThreadPoolExecutor(WORKERS) as pool,
    ):
        texts = list(pool.map(lambda day: fetch_issue(client, day), days))
    found = [t for t in texts if t is not None]
    print(f"{len(found)} forecasts on disk, {len(days) - len(found)} days missing from the archive")
    return found


def forecast_table(texts: list[str], kp: pd.Series) -> pd.DataFrame:
    """One row per (issue, target interval) with NOAA, baselines and observed Kp."""
    frames = []
    for text in texts:
        try:
            frames.append(parse_3day_forecast(text))
        except ValueError as err:
            print(f"skip unparseable forecast: {err}")
    df = pd.concat(frames).rename(columns={"kp": "noaa"})
    df = df[df["lead_h"] >= 0]
    df["lead_day"] = (df.index.floor("D") - df["issued"].dt.floor("D")).dt.days + 1

    last_completed = df["issued"].dt.floor("3h") - pd.Timedelta("3h")
    df["persistence"] = kp.reindex(last_completed).to_numpy()
    df["recurrence"] = kp.reindex(df.index - RECURRENCE).to_numpy()
    df["observed"] = kp.reindex(df.index).to_numpy()
    return df.dropna(subset=["noaa", "persistence", "recurrence", "observed"])


def score(df: pd.DataFrame) -> list[dict]:
    results = []
    obs_cls = kp_to_class(df["observed"])
    for lead in sorted(df["lead_day"].unique()):
        sel = (df["lead_day"] == lead).to_numpy()
        for name in ("noaa", "persistence", "recurrence"):
            err = df.loc[sel, name] - df.loc[sel, "observed"]
            row = {
                "lead_day": int(lead),
                "forecast": name,
                "n": int(sel.sum()),
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt((err**2).mean())),
            }
            fc_cls = kp_to_class(df.loc[sel, name])
            for t in THRESHOLDS:
                c = threshold_class(t)
                table = contingency(fc_cls >= c, obs_cls[sel] >= c, 0.5)
                row[f"kp{t}"] = {
                    "events": table["hits"] + table["misses"],
                    "forecast_yes": table["hits"] + table["false_alarms"],
                    **table,
                    "tss": tss(table),
                    "hss": hss(table),
                }
            results.append(row)
    return results


def calibration_counts(df: pd.DataFrame) -> dict:
    """counts[lead_day][forecast_class][observed_class] over the whole archive."""
    fc, obs = kp_to_class(df["noaa"]).astype(int), kp_to_class(df["observed"]).astype(int)
    counts = {}
    for lead in sorted(df["lead_day"].unique()):
        sel = (df["lead_day"] == lead).to_numpy()
        table = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        np.add.at(table, (fc[sel], obs[sel]), 1)
        counts[str(int(lead))] = table.tolist()
    return counts


def print_table(results: list[dict]) -> None:
    print("\n| lead | forecast | MAE | RMSE | TSS Kp>=5 | HSS Kp>=5 | TSS Kp>=7 | HSS Kp>=7 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        k5, k7 = r["kp5"], r["kp7"]
        print(
            f"| day {r['lead_day']} | {r['forecast']} | {r['mae']:.2f} | {r['rmse']:.2f} "
            f"| {k5['tss']:.2f} | {k5['hss']:.2f} | {k7['tss']:.2f} | {k7['hss']:.2f} |"
        )


def main() -> None:
    kp_3h = pd.read_parquet(PROCESSED_DIR / "kp_3h.parquet")
    kp = kp_3h["kp"].where(kp_3h["kp_definitive"])
    texts = download_archive(end=pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D"))
    df = forecast_table(texts, kp)
    period = [f"{df.index.min():%Y-%m-%d}", f"{df.index.max():%Y-%m-%d}"]
    print(f"scored {df['issued'].nunique()} issues, {period[0]} to {period[1]}")
    results = score(df)
    print_table(results)
    out = {
        "period": period,
        "issue_time_ut": ISSUE,
        "results": results,
    }
    (ROOT / "reports" / "noaa_3day.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/noaa_3day.json")
    calibration = {
        "period": period,
        "issue_time_ut": ISSUE,
        "classes": CLASS_LABELS,
        "counts": calibration_counts(df),
    }
    (ROOT / "models" / "noaa_calibration.json").write_text(json.dumps(calibration, indent=1))
    print("wrote models/noaa_calibration.json")


if __name__ == "__main__":
    main()
