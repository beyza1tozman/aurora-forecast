"""Geomagnetic latitude, Kp needed, and the exceedance curve."""

import numpy as np
import pytest

from aurora.dataset import exceedance_from_class_probs
from aurora.physics.geomag import dipole_pole, geomagnetic_latitude
from aurora.physics.oval import kp_needed, prob_kp_at_least

CITIES = {  # name: (lat, lon)
    "Hamburg": (53.55, 10.0),
    "Berlin": (52.52, 13.40),
    "Munich": (48.14, 11.58),
    "Tromso": (69.65, 18.96),
}


def test_dipole_pole_2025():
    lat, lon = dipole_pole(2025.0)
    assert lat == pytest.approx(80.8, abs=0.1)  # published IGRF-14 dipole pole
    assert lon == pytest.approx(-72.8, abs=0.2)


@pytest.mark.parametrize("city", ["Hamburg", "Berlin", "Munich"])
def test_germany_close_to_geographic(city):
    lat, lon = CITIES[city]
    assert abs(geomagnetic_latitude(lat, lon) - lat) < 1.0


def test_north_america_higher_than_europe():
    # Same geographic latitude, much higher geomagnetic latitude near the pole's longitude.
    assert geomagnetic_latitude(45.0, -93.0) > geomagnetic_latitude(45.0, 10.0) + 7


def test_kp_needed_anchors():
    mlat = {c: geomagnetic_latitude(*CITIES[c]) for c in CITIES}
    assert 5.0 <= kp_needed(mlat["Hamburg"]) <= 6.0
    assert 7.5 <= kp_needed(mlat["Munich"]) <= 8.5
    assert kp_needed(mlat["Tromso"]) == 0.0
    assert kp_needed(mlat["Hamburg"]) < kp_needed(mlat["Berlin"]) < kp_needed(mlat["Munich"])


CLASS_PROBS = np.array([[0.70, 0.20, 0.06, 0.03, 0.01]])


def test_exceedance_curve_matches_model_at_anchors():
    for k in (4, 5, 6, 7):
        expected = exceedance_from_class_probs(CLASS_PROBS, k)[0]
        assert prob_kp_at_least(CLASS_PROBS, k)[0] == pytest.approx(expected)


def test_exceedance_curve_monotonic_and_bounded():
    ks = np.linspace(-1, 10, 111)
    p = np.array([prob_kp_at_least(CLASS_PROBS, k)[0] for k in ks])
    assert np.all(np.diff(p) <= 1e-12)
    assert p[0] == 1.0 and p[-1] == 0.0
    assert np.all((p >= 0) & (p <= 1))


def test_exceedance_curve_vectorised_over_rows():
    probs = np.vstack([CLASS_PROBS, [[0.2, 0.2, 0.2, 0.2, 0.2]]])
    p = prob_kp_at_least(probs, [5.5, 5.5])
    assert p.shape == (2,) and p[1] > p[0]
