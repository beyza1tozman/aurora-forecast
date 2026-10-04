"""Geomagnetic latitude from the IGRF-14 centred dipole.

A centred dipole needs only the three degree-1 Gauss coefficients, so it is pure
NumPy (aacgmv2/apexpy need C/Fortran builds on Windows). In Germany the dipole
latitude is within ~1 degree of geographic latitude. The non-dipole field is
ignored, which matters more in e.g. the UK and North America; see aurora.physics.oval
for how the Kp relation is anchored to Central European experience.

Coefficients: IGRF-14, epoch 2025.0, with the 2025-2030 secular variation
(https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt).
"""

import numpy as np

EPOCH = 2025.0
G10, G11, H11 = -29350.0, -1410.3, 4545.5  # nT
SV_G10, SV_G11, SV_H11 = 12.6, 10.0, -21.5  # nT / year


def dipole_pole(year: float = EPOCH) -> tuple[float, float]:
    """Geographic (lat, lon) in degrees of the northern geomagnetic pole."""
    dt = year - EPOCH
    g10, g11, h11 = G10 + SV_G10 * dt, G11 + SV_G11 * dt, H11 + SV_H11 * dt
    b0 = np.sqrt(g10**2 + g11**2 + h11**2)
    lat = 90.0 - np.degrees(np.arccos(-g10 / b0))
    lon = np.degrees(np.arctan2(-h11, -g11))
    return float(lat), float(lon)


def geomagnetic_latitude(lat, lon, year: float = EPOCH):
    """Dipole geomagnetic latitude in degrees for geographic lat/lon in degrees."""
    pole_lat, pole_lon = np.radians(dipole_pole(year))
    lat, lon = np.radians(lat), np.radians(lon)
    s = np.sin(lat) * np.sin(pole_lat) + np.cos(lat) * np.cos(pole_lat) * np.cos(lon - pole_lon)
    return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))
