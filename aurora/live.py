"""Live Kp forecast: bow-shock solar wind + recent Kp -> shared features -> calibrated models.

Which row is used
-----------------
Features in row t use solar wind up to t + 1h (bow-shock time) and are issued at
t + 1h. Because L1 is ~45-90 min upstream, the newest complete bow-shock hour can
end *after* the current wall-clock time, so the issue time may lie slightly in
the future. That is the real warning time L1 gives.

The row must also have its ``kp_last`` interval (the last one completed at issue
time) already observed. If the newest row's interval is still in progress, an
earlier row is used. Rows older than MAX_STALENESS are not used at all.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aurora.dataset import CLASS_LABELS, THRESHOLDS, exceedance_from_class_probs, target_start
from aurora.features import FEATURE_COLUMNS, ISSUE_LAG, KP_INTERVAL, SW_COLUMNS, build_features
from aurora.model import HorizonModel

MAX_STALENESS = pd.Timedelta("3h")  # issue time may lag wall clock by at most this


@dataclass
class HorizonForecast:
    horizon_h: int
    interval_start: pd.Timestamp
    class_probs: np.ndarray  # (N_CLASSES,)

    @property
    def interval_end(self) -> pd.Timestamp:
        return self.interval_start + KP_INTERVAL

    def exceedance(self) -> dict[int, float]:
        return {
            t: float(exceedance_from_class_probs(self.class_probs[None, :], t)[0])
            for t in THRESHOLDS
        }

    def to_dict(self) -> dict:
        return {
            "horizon_h": self.horizon_h,
            "interval_start": self.interval_start.isoformat(),
            "interval_end": self.interval_end.isoformat(),
            "class_probs": dict(zip(CLASS_LABELS, map(float, self.class_probs), strict=True)),
            "p_kp_at_least": {str(t): p for t, p in self.exceedance().items()},
        }


@dataclass
class KpForecast:
    issued: pd.Timestamp
    kp_last: float
    horizons: list[HorizonForecast]
    inputs_missing: list[str]  # solar-wind columns that were NaN in the used row

    def to_dict(self) -> dict:
        return {
            "issued": self.issued.isoformat(),
            "kp_last": self.kp_last,
            "inputs_missing": self.inputs_missing,
            "horizons": [h.to_dict() for h in self.horizons],
        }


class NoUsableDataError(RuntimeError):
    """No recent row with solar wind and a completed Kp interval."""


def select_issue_row(features: pd.DataFrame, now: pd.Timestamp) -> pd.Timestamp:
    issue = features.index + ISSUE_LAG
    usable = (
        features["kp_last"].notna()
        & features[SW_COLUMNS].notna().any(axis=1)
        & (issue >= now - MAX_STALENESS)
    )
    if not usable.any():
        raise NoUsableDataError("no recent solar wind row with a completed Kp interval")
    return features.index[usable][-1]


def forecast_kp(
    sw_hourly: pd.DataFrame,
    kp: pd.Series,
    models: dict[int, HorizonModel],
    now: pd.Timestamp,
) -> KpForecast:
    features = build_features(sw_hourly, kp)
    row_time = select_issue_row(features, now)
    row = features.loc[[row_time], FEATURE_COLUMNS]
    horizons = [
        HorizonForecast(
            horizon_h=h,
            interval_start=target_start(row.index, h)[0],
            class_probs=models[h].predict_proba(row)[0],
        )
        for h in sorted(models)
    ]
    return KpForecast(
        issued=row_time + ISSUE_LAG,
        kp_last=float(row["kp_last"].iloc[0]),
        horizons=horizons,
        inputs_missing=[c for c in SW_COLUMNS if pd.isna(row[c].iloc[0])],
    )
