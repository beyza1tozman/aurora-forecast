"""Feature pipeline: the single source of truth for training and serving.

Time convention
---------------
An hourly solar wind row labelled ``t`` is the average over ``[t, t+1h)``, so it
is only known at ``t + 1h``. Features in row ``t`` use solar wind rows <= ``t``
and the forecast is *issued* at ``t + 1h``.

Kp enters only through *completed* 3-hour intervals: at issue time ``t + 1h``
the newest finished interval starts at ``floor(t - 2h, 3h)``.

Only quantities that the live NOAA RTSW feed also provides are used (IMF By, Bz,
|B|, speed, density, temperature). Derived quantities (E_y, dynamic pressure,
Newell coupling) are computed here instead of taken from OMNI, so training and
serving use identical formulas.
"""

import numpy as np
import pandas as pd

# Solar wind inputs, as named in OMNI (aurora.data.omni). Live data must be renamed to match.
SW_COLUMNS = ["by_gsm", "bz_gsm", "b_mag", "speed", "density", "temperature"]

ROLLING_WINDOWS = (3, 6)  # hours
ROLLING_MEAN = ["bz_gsm", "newell", "speed", "density", "pdyn"]

ISSUE_LAG = pd.Timedelta("1h")  # row t is known at t + 1h
KP_INTERVAL = pd.Timedelta("3h")
RECURRENCE = pd.Timedelta("27D")  # Bartels solar rotation

FEATURE_COLUMNS = (
    SW_COLUMNS
    + ["bt", "clock_angle", "newell", "ey", "pdyn"]
    + [f"{c}_mean_{w}h" for w in ROLLING_WINDOWS for c in ROLLING_MEAN]
    + [f"bz_gsm_min_{w}h" for w in ROLLING_WINDOWS]
    + [f"bz_gsm_std_{w}h" for w in ROLLING_WINDOWS]
    + ["bz_south_hours"]
    + ["kp_last", "kp_prev", "kp_max_24h", "kp_27d", "hours_into_bin"]
    + ["doy_sin", "doy_cos", "ut_sin", "ut_cos"]
)


def newell_coupling(speed, bt, clock_angle):
    """Newell et al. (2007) coupling dPhi/dt = V^(4/3) Bt^(2/3) sin^(8/3)(theta/2).

    Arbitrary units (V in km/s, Bt in nT); only the relative size matters to the model.
    """
    return speed ** (4 / 3) * bt ** (2 / 3) * np.abs(np.sin(clock_angle / 2)) ** (8 / 3)


def last_completed_kp_start(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Start of the newest 3-hour Kp interval that has ended by issue time (row time + 1h)."""
    return (index + ISSUE_LAG - KP_INTERVAL).floor(KP_INTERVAL)


def _southward_run_hours(bz: pd.Series) -> pd.Series:
    """Consecutive hours with Bz < 0 ending at each row (NaN where Bz is missing)."""
    south = (bz < 0).astype(int)
    run_id = (south == 0).cumsum()
    run = south.groupby(run_id).cumsum().astype(float)
    return run.where(bz.notna())


def build_features(sw: pd.DataFrame, kp: pd.Series) -> pd.DataFrame:
    """Build the feature matrix, one row per hour of ``sw``.

    sw: hourly solar wind with SW_COLUMNS, UTC index (regularised to 1 h here).
    kp: Kp per 3-hour interval, indexed by interval start (UTC).
    Missing values stay NaN; LightGBM handles them.
    """
    sw = sw[SW_COLUMNS].asfreq("1h")
    f = sw.copy()

    by, bz, v, n = sw["by_gsm"], sw["bz_gsm"], sw["speed"], sw["density"]
    f["bt"] = np.hypot(by, bz)
    f["clock_angle"] = np.arctan2(by, bz)
    f["newell"] = newell_coupling(v, f["bt"], f["clock_angle"])
    f["ey"] = -v * bz * 1e-3  # mV/m
    f["pdyn"] = 2e-6 * n * v**2  # nPa, same formula as OMNI

    for w in ROLLING_WINDOWS:
        for c in ROLLING_MEAN:
            f[f"{c}_mean_{w}h"] = f[c].rolling(w, min_periods=1).mean()
    for w in ROLLING_WINDOWS:
        f[f"bz_gsm_min_{w}h"] = bz.rolling(w, min_periods=1).min()
    for w in ROLLING_WINDOWS:
        f[f"bz_gsm_std_{w}h"] = bz.rolling(w, min_periods=2).std()
    f["bz_south_hours"] = _southward_run_hours(bz)

    kp = kp.asfreq(KP_INTERVAL)
    kp_max_24h = kp.rolling(8, min_periods=1).max()
    last = last_completed_kp_start(f.index)
    f["kp_last"] = kp.reindex(last).to_numpy()
    f["kp_prev"] = kp.reindex(last - KP_INTERVAL).to_numpy()
    f["kp_max_24h"] = kp_max_24h.reindex(last).to_numpy()
    f["kp_27d"] = kp.reindex(last - RECURRENCE).to_numpy()
    issue = f.index + ISSUE_LAG
    f["hours_into_bin"] = ((issue - (last + KP_INTERVAL)) / pd.Timedelta("1h")).to_numpy()

    doy = 2 * np.pi * (f.index.dayofyear - 1) / 365.25
    ut = 2 * np.pi * issue.hour / 24
    f["doy_sin"], f["doy_cos"] = np.sin(doy), np.cos(doy)
    f["ut_sin"], f["ut_cos"] = np.sin(ut), np.cos(ut)

    return f[FEATURE_COLUMNS]
