"""Parser for GFZ Potsdam Kp (Kp_ap_Ap_SN_F107_since_1932.txt).

One line per UT day with eight 3-hourly Kp values (Kp1 = 00-03 UT, ...).
Kp is given in thirds as decimals: 5- = 4.667, 5o = 5.000, 5+ = 5.333.
The last column D says whether Kp is definitive (D >= 1) or a nowcast (D = 0)
that will be revised later.
"""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = (
    ["year", "month", "day", "days", "days_m", "bsr", "db"]
    + [f"kp{i}" for i in range(1, 9)]
    + [f"ap{i}" for i in range(1, 9)]
    + ["Ap", "sn", "f107_obs", "f107_adj", "d"]
)


def parse_gfz_kp(path: Path) -> pd.DataFrame:
    """Return one row per 3-hour Kp interval, indexed by the interval's start (UTC).

    Columns: kp (float, rounded to thirds), ap, kp_definitive (bool).
    Missing values (-1) become NaN.
    """
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=COLUMNS)

    day = pd.to_datetime(raw[["year", "month", "day"]]).to_numpy()  # naive, UTC by definition
    kp = raw[[f"kp{i}" for i in range(1, 9)]].to_numpy(dtype=float)
    ap = raw[[f"ap{i}" for i in range(1, 9)]].to_numpy(dtype=float)

    starts = (day[:, None] + np.arange(8) * np.timedelta64(3, "h")).ravel()
    kp = kp.ravel()
    ap = ap.ravel()

    kp = np.where(kp < 0, np.nan, np.round(kp * 3) / 3)
    ap = np.where(ap < 0, np.nan, ap)

    return pd.DataFrame(
        {
            "kp": kp,
            "ap": ap,
            "kp_definitive": np.repeat(raw["d"].to_numpy() >= 1, 8),
        },
        index=pd.DatetimeIndex(starts, name="time", tz="UTC"),
    )


def join_kp_to_hours(hourly: pd.DataFrame, kp_3h: pd.DataFrame) -> pd.DataFrame:
    """Attach the Kp interval *containing* each hour as `kp_bin` (+ `kp_definitive`).

    `kp_bin` at hour t is only known once its 3-hour interval ends, so it is a
    target-side column, not a feature at time t.
    """
    kp = kp_3h.reindex(hourly.index.floor("3h"))
    out = hourly.copy()
    out["kp_bin"] = kp["kp"].to_numpy()
    out["kp_definitive"] = kp["kp_definitive"].to_numpy()
    return out
