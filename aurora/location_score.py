"""Location score: how likely you are to *see* the aurora from a place in a given hour.

    chance = P(Kp >= Kp_needed(mlat)) x clear_sky x darkness x moon

Only the first factor is a calibrated probability. The others are heuristic
weights in [0, 1], so the result is a score labelled "chance" in the UI, not a
probability. ``chance_if_clear`` drops the cloud factor (and is the only value
available where the cloud forecast ends).
"""

import numpy as np
import pandas as pd

from aurora.data.open_meteo import LAYERS as CLOUD_COLUMNS
from aurora.physics.geomag import geomagnetic_latitude
from aurora.physics.oval import kp_needed
from aurora.physics.sky import moon_altitude, moon_illumination, sun_altitude

DARK_SUN_ALT = -12.0  # nautical twilight ends: fully dark enough
BRIGHT_SUN_ALT = -6.0  # civil twilight: too bright
HIGH_CLOUD_WEIGHT = 0.5  # thin high cloud (cirrus) lets a bright aurora through
MOON_PENALTY = 0.4  # at most 40% off for a full moon high in the sky
MOON_FULL_EFFECT_ALT = 30.0  # degrees


def decimal_year(t: pd.Timestamp) -> float:
    return t.year + (t.dayofyear - 1) / 365.25


def location_info(lat: float, lon: float, when: pd.Timestamp) -> dict:
    mlat = float(geomagnetic_latitude(lat, lon, decimal_year(when)))
    return {"lat": lat, "lon": lon, "mlat": mlat, "kp_needed": float(kp_needed(mlat))}


def clear_sky_factor(low, mid, high):
    """Chance the northern sky is clear, layers overlapping at random.

    Low and mid cloud block the view; high cloud only partly.
    """
    low, mid, high = (np.asarray(x, dtype=float) for x in (low, mid, high))
    return (1.0 - low) * (1.0 - mid) * (1.0 - HIGH_CLOUD_WEIGHT * high)


def darkness_factor(sun_alt):
    """1 below -12 deg (nautical twilight), 0 above -6 deg, linear in between."""
    sun_alt = np.asarray(sun_alt, dtype=float)
    return np.clip((BRIGHT_SUN_ALT - sun_alt) / (BRIGHT_SUN_ALT - DARK_SUN_ALT), 0.0, 1.0)


def moon_factor(moon_alt, illumination):
    """Mild penalty for a bright moon above the horizon (strong aurora still shows)."""
    height = np.clip(np.sin(np.radians(moon_alt)) / np.sin(np.radians(MOON_FULL_EFFECT_ALT)), 0, 1)
    return 1.0 - MOON_PENALTY * np.asarray(illumination, dtype=float) * height


def sky_conditions(
    times: pd.DatetimeIndex, lat: float, lon: float, clouds: pd.DataFrame
) -> pd.DataFrame:
    """Per-hour sky factors at the middle of each hour. Clouds missing -> clear_sky NaN."""
    mid = times + pd.Timedelta("30min")
    sun = sun_altitude(mid, lat, lon)
    moon = moon_altitude(mid, lat, lon)
    illum = moon_illumination(mid)
    c = clouds.reindex(times, columns=CLOUD_COLUMNS)
    return pd.DataFrame(
        {
            "sun_alt": sun,
            "moon_alt": moon,
            "moon_illumination": illum,
            **{col: c[col].to_numpy() for col in CLOUD_COLUMNS},
            "clear_sky": clear_sky_factor(
                c["cloud_cover_low"], c["cloud_cover_mid"], c["cloud_cover_high"]
            ),
            "darkness": darkness_factor(sun),
            "moon": moon_factor(moon, illum),
        },
        index=times,
    )


def combine(p_kp, sky: pd.DataFrame) -> pd.DataFrame:
    """Add chance_if_clear and chance (NaN where there is no cloud forecast)."""
    out = sky.copy()
    out["p_kp"] = p_kp
    out["chance_if_clear"] = out["p_kp"] * out["darkness"] * out["moon"]
    out["chance"] = out["chance_if_clear"] * out["clear_sky"]
    return out
