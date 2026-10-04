"""Sun and moon altitude and moon illumination, in pure NumPy.

Low-precision formulas from the Astronomical Almanac: the sun to ~0.01 degrees,
the moon to ~0.3 degrees in ecliptic longitude (plus up to ~1 degree of
topocentric parallax, ignored). That is far more than a darkness / moonlight
heuristic needs, and avoids shipping a 17 MB JPL ephemeris.
"""

import numpy as np
import pandas as pd

J2000 = pd.Timestamp("2000-01-01 12:00", tz="UTC")


def _days_since_j2000(times) -> np.ndarray:
    times = pd.DatetimeIndex(times)
    if times.tz is None:
        raise ValueError("times must be timezone-aware (UTC)")
    return np.asarray((times - J2000) / pd.Timedelta("1D"), dtype=float)


def _sin(deg):
    return np.sin(np.radians(deg))


def _obliquity(d):
    return 23.439 - 4e-7 * d


def sun_ecliptic_longitude(d) -> np.ndarray:
    g = 357.528 + 0.9856003 * d  # mean anomaly
    mean_lon = 280.460 + 0.9856474 * d
    return (mean_lon + 1.915 * _sin(g) + 0.020 * _sin(2 * g)) % 360


# Periodic terms (amplitude deg, phase deg, rate deg per Julian century) of the
# moon's ecliptic longitude and latitude.
MOON_LON_TERMS = [
    (6.29, 134.9, 477198.85),
    (-1.27, 259.2, -413335.38),
    (0.66, 235.7, 890534.23),
    (0.21, 269.9, 954397.70),
    (-0.19, 357.5, 35999.05),
    (-0.11, 186.6, 966404.05),
]
MOON_LAT_TERMS = [
    (5.13, 93.3, 483202.03),
    (0.28, 228.2, 960400.87),
    (-0.28, 318.3, 6003.18),
    (-0.17, 217.6, -407332.20),
]


def moon_ecliptic(d) -> tuple[np.ndarray, np.ndarray]:
    """Geocentric ecliptic (longitude, latitude) of the moon in degrees."""
    t = d / 36525.0
    lon = 218.32 + 481267.881 * t + sum(a * _sin(p + r * t) for a, p, r in MOON_LON_TERMS)
    lat = sum(a * _sin(p + r * t) for a, p, r in MOON_LAT_TERMS)
    return lon % 360, lat


def _altitude(d, ecl_lon, ecl_lat, lat, lon) -> np.ndarray:
    """Altitude in degrees of an object at ecliptic (lon, lat) seen from geographic (lat, lon)."""
    eps = np.radians(_obliquity(d))
    lam, beta = np.radians(ecl_lon), np.radians(ecl_lat)
    ra = np.arctan2(np.sin(lam) * np.cos(eps) - np.tan(beta) * np.sin(eps), np.cos(lam))
    dec = np.arcsin(np.sin(beta) * np.cos(eps) + np.cos(beta) * np.sin(eps) * np.sin(lam))
    gmst = np.radians((280.46061837 + 360.98564736629 * d) % 360)
    hour_angle = gmst + np.radians(lon) - ra
    phi = np.radians(lat)
    sin_alt = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(sin_alt, -1, 1)))


def sun_altitude(times, lat: float, lon: float) -> np.ndarray:
    d = _days_since_j2000(times)
    return _altitude(d, sun_ecliptic_longitude(d), 0.0, lat, lon)


def moon_altitude(times, lat: float, lon: float) -> np.ndarray:
    d = _days_since_j2000(times)
    return _altitude(d, *moon_ecliptic(d), lat, lon)


def moon_illumination(times) -> np.ndarray:
    """Illuminated fraction of the moon's disc (0 = new, 1 = full)."""
    d = _days_since_j2000(times)
    moon_lon, moon_lat = moon_ecliptic(d)
    cos_elong = np.cos(np.radians(moon_lat)) * np.cos(
        np.radians(moon_lon - sun_ecliptic_longitude(d))
    )
    return (1 - cos_elong) / 2
