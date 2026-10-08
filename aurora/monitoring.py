"""Live verification: logged forecasts vs the Kp that was later observed.

Same scores as the offline evaluation (aurora.metrics), computed on the
forecast log. Persistence is scored on the same rows from the logged kp_last,
climatology from the training-period base rate, so "skill" means the same thing
here as in MODEL_CARD.md.

Observed Kp is the GFZ nowcast (NOAA estimate where GFZ has not published),
not the definitive Kp the model was trained on; scores are provisional.
"""

import numpy as np
import pandas as pd

from aurora.dataset import kp_to_class, threshold_class
from aurora.metrics import brier, brier_skill
from aurora.physics.oval import CLIM_EXCEEDANCE

THRESHOLDS = (4, 5)  # Kp>=4 has enough events to score early; Kp>=5 is the aurora-relevant one
PROB_COLUMNS = [f"p_{i}" for i in range(5)]


def p_at_least(pairs: pd.DataFrame, k: int) -> np.ndarray:
    return pairs[PROB_COLUMNS[threshold_class(k) :]].sum(axis=1).to_numpy()


def observed_at_least(kp, k: int) -> np.ndarray:
    return (kp_to_class(kp) >= threshold_class(k)).astype(float)


def _none_if_nan(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def scores(pairs: pd.DataFrame) -> list[dict]:
    """Per horizon and threshold: counts, Brier scores and skill vs the baselines."""
    rows = []
    if pairs.empty:
        return rows
    for h, grp in pairs.groupby("horizon_h"):
        for k in THRESHOLDS:
            y = observed_at_least(grp["kp_observed"], k)
            p = p_at_least(grp, k)
            p_clim = np.full(len(grp), CLIM_EXCEEDANCE[k])
            p_pers = observed_at_least(grp["kp_last"], k)
            rows.append(
                {
                    "horizon_h": int(h),
                    "threshold": k,
                    "n": len(grp),
                    "events": int(y.sum()),
                    "brier": brier(p, y),
                    "brier_climatology": brier(p_clim, y),
                    "brier_persistence": brier(p_pers, y),
                    "bss_climatology": _none_if_nan(brier_skill(p, y, p_clim)),
                    "bss_persistence": _none_if_nan(brier_skill(p, y, p_pers)),
                }
            )
    return rows


def daily_brier(pairs: pd.DataFrame, k: int = 4) -> list[dict]:
    """Brier score of P(Kp >= k) per UTC day of the target interval and horizon."""
    if pairs.empty:
        return []
    df = pairs.assign(
        day=pairs["interval_start"].dt.floor("D"),
        p=p_at_least(pairs, k),
        y=observed_at_least(pairs["kp_observed"], k),
    )
    out = []
    for (day, h), grp in df.groupby(["day", "horizon_h"]):
        out.append(
            {
                "day": day.date().isoformat(),
                "horizon_h": int(h),
                "n": len(grp),
                "brier": brier(grp["p"], grp["y"]),
            }
        )
    return out


def forecast_series(pairs: pd.DataFrame, horizon: int, k: int) -> list[dict]:
    """Mean P(Kp >= k) per target interval for one horizon (several issues can
    target the same interval), with the observed Kp."""
    if pairs.empty:
        return []
    sub = pairs[pairs["horizon_h"] == horizon]
    if sub.empty:
        return []
    sub = sub.assign(p=p_at_least(sub, k))
    g = sub.groupby("interval_start").agg(p=("p", "mean"), kp=("kp_observed", "first"))
    return [
        {"interval_start": t.isoformat(), "p": float(r.p), "kp_observed": float(r.kp)}
        for t, r in g.iterrows()
    ]
