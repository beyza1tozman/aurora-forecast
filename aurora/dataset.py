"""Targets, Kp classes and the time-based train/val/test split.

Target for horizon h at row t: the GFZ Kp of the 3-hour interval containing
the issue time + h, i.e. t + 1h + h (see aurora.features for the convention).

Classes group the -/o/+ thirds into integer Kp (4.667 = "5-" counts as 5, like
NOAA's G1 scale): {<=3, 4, 5, 6, 7+} -> 0..4. Exceedance probabilities are tail
sums over these classes, so P(Kp>=7) <= P(Kp>=6) <= P(Kp>=5) by construction.
"""

import numpy as np
import pandas as pd

from aurora.config import PROCESSED_DIR
from aurora.features import ISSUE_LAG, KP_INTERVAL, SW_COLUMNS, build_features

HORIZONS = (1, 3, 6)  # hours after issue time
N_CLASSES = 5
CLASS_LABELS = ["<=3", "4", "5", "6", "7+"]
THRESHOLDS = (5, 6, 7)

# Half-open [start, end) periods; None = until the end of the data.
SPLITS = {
    "train": ("1998-01-01", "2015-01-01"),
    "val": ("2015-01-01", "2020-01-01"),
    "test": ("2020-01-01", None),
}
# Kp is autocorrelated and recurs every 27 days, so the first EMBARGO of val and
# test is dropped. It also covers the longest target lead (1h + 6h).
EMBARGO = pd.Timedelta("30D")


def kp_to_class(kp) -> np.ndarray:
    """Map Kp (in thirds) to class 0..4; NaN stays NaN."""
    kp = np.asarray(kp, dtype=float)
    integer_kp = np.floor(kp + 0.5)
    return np.clip(integer_kp - 3, 0, N_CLASSES - 1)


def threshold_class(threshold: int) -> int:
    """Index of the first class with Kp >= threshold."""
    return threshold - 3


def exceedance_from_class_probs(probs: np.ndarray, threshold: int) -> np.ndarray:
    """P(Kp >= threshold) as the tail sum of class probabilities (n, N_CLASSES)."""
    return probs[:, threshold_class(threshold) :].sum(axis=1)


def target_start(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Start of the Kp interval containing issue time + horizon."""
    return (index + ISSUE_LAG + pd.Timedelta(hours=horizon)).floor(KP_INTERVAL)


def make_targets(index: pd.DatetimeIndex, kp_3h: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Columns kp_h{h} and cls_h{h}. Non-definitive (nowcast) Kp targets become NaN."""
    kp = kp_3h["kp"].where(kp_3h["kp_definitive"])
    out = pd.DataFrame(index=index)
    for h in horizons:
        out[f"kp_h{h}"] = kp.reindex(target_start(index, h)).to_numpy()
        out[f"cls_h{h}"] = kp_to_class(out[f"kp_h{h}"])
    return out


def assign_split(index: pd.DatetimeIndex) -> pd.Series:
    """Label each row train / val / test / embargo (rows before the first split: NaN)."""
    labels = pd.Series(pd.NA, index=index, dtype="object")
    for name, (start, end) in SPLITS.items():
        start = pd.Timestamp(start, tz="UTC")
        end = pd.Timestamp(end, tz="UTC") if end else index.max() + pd.Timedelta("1h")
        labels[(index >= start) & (index < end)] = name
        if name != "train":
            labels[(index >= start) & (index < start + EMBARGO)] = "embargo"
    return labels


def load_modelling_frame() -> pd.DataFrame:
    """Features + targets + split label for every hour in the processed dataset."""
    hourly = pd.read_parquet(PROCESSED_DIR / "hourly.parquet")
    kp_3h = pd.read_parquet(PROCESSED_DIR / "kp_3h.parquet")
    features = build_features(hourly[SW_COLUMNS], kp_3h["kp"])
    targets = make_targets(features.index, kp_3h)
    frame = features.join(targets)
    frame["split"] = assign_split(frame.index)
    return frame
