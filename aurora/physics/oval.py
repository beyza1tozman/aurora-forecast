"""Kp needed to see the aurora from a given geomagnetic latitude, and P(Kp >= k).

View line
---------
The classic Kp map puts the equatorward edge of the auroral oval near
``66.5 - 2.04 * Kp`` degrees geomagnetic latitude (Kp 5 -> 56.3, Kp 7 -> 52.2).
Aurora at 100-300 km altitude is also visible low on the poleward horizon from
further equatorward, and the dipole latitude differs from corrected geomagnetic
latitude, so the line is shifted equatorward by VIEW_OFFSET. The offset is
anchored to Central European experience: northern Germany (Hamburg, mlat 53.7)
needs about Kp 5-6, Munich (48.2) about Kp 8. This is a heuristic, not a fit.
The southern oval mirrors the northern one, so |mlat| is used in both hemispheres.

Exceedance curve
----------------
The model gives class probabilities for {<=3, 4, 5, 6, 7+}, i.e. P(Kp >= k) for
k = 4..7. Below 4, the training-period climatology is used (floored at P(Kp>=4)
so the curve stays monotonic). Above 7, P(Kp >= 7) is scaled by the
climatological ratios P(Kp>=8)/P(Kp>=7) and P(Kp>=9)/P(Kp>=7). Between integer
anchors the curve is interpolated log-linearly.
"""

import numpy as np

from aurora.dataset import N_CLASSES, exceedance_from_class_probs

OVAL_EDGE_KP0 = 66.5  # degrees mlat at Kp 0
OVAL_DEG_PER_KP = 2.04
VIEW_OFFSET = 1.5  # degrees equatorward of the oval edge
KP_MAX = 9.0

# Integer-Kp exceedance rates in the 1998-2014 training period (GFZ definitive Kp).
CLIM_EXCEEDANCE = dict(
    enumerate([0.8265, 0.5262, 0.2861, 0.1235, 0.0438, 0.0153, 0.00529, 0.00189, 0.00046], start=1)
)


def kp_needed(mlat):
    """Kp at which the aurora may be seen low on the poleward horizon (0..9, or > 9 = never).

    Northern horizon in the northern hemisphere, southern in the southern (aurora australis).
    """
    view_line_kp0 = OVAL_EDGE_KP0 - VIEW_OFFSET
    return np.maximum(
        (view_line_kp0 - np.abs(np.asarray(mlat, dtype=float))) / OVAL_DEG_PER_KP, 0.0
    )


def exceedance_anchors(class_probs: np.ndarray) -> np.ndarray:
    """P(Kp >= k) for k = 0..9, shape (n, 10), from class probabilities (n, N_CLASSES)."""
    class_probs = np.atleast_2d(class_probs)
    assert class_probs.shape[1] == N_CLASSES
    p = np.empty((len(class_probs), 10))
    p[:, 0] = 1.0
    for k in (4, 5, 6, 7):
        p[:, k] = exceedance_from_class_probs(class_probs, k)
    for k in (1, 2, 3):
        p[:, k] = np.maximum(CLIM_EXCEEDANCE[k], p[:, 4])
    for k in (8, 9):
        p[:, k] = p[:, 7] * CLIM_EXCEEDANCE[k] / CLIM_EXCEEDANCE[7]
    return p


def prob_kp_at_least(class_probs: np.ndarray, k) -> np.ndarray:
    """P(Kp >= k) for real-valued k (broadcast over rows), log-linear between integer anchors."""
    anchors = exceedance_anchors(class_probs)
    k = np.broadcast_to(np.asarray(k, dtype=float), (len(anchors),))
    out = np.zeros(len(anchors))  # k > 9: never
    for i, (row, ki) in enumerate(zip(anchors, k, strict=True)):
        if ki <= 0:
            out[i] = 1.0
        elif ki <= KP_MAX:
            lo = min(int(np.floor(ki)), 8)
            frac = ki - lo
            log_p = np.log(np.clip(row[lo : lo + 2], 1e-12, 1.0))
            out[i] = float(np.exp(log_p[0] + frac * (log_p[1] - log_p[0])))
    return out


def kp_at_probability(class_probs: np.ndarray, p: float, step: float = 0.1) -> float:
    """Highest Kp (on a ``step`` grid) that is reached with probability >= p."""
    grid = np.arange(0.0, KP_MAX + step / 2, step)
    probs = np.array([prob_kp_at_least(class_probs, k)[0] for k in grid])
    reached = grid[probs >= p]
    return float(reached.max()) if len(reached) else 0.0


def view_line(kp: float, lons=None, year: float | None = None) -> list[tuple[float, float]]:
    """Northern-hemisphere (lat, lon) points where exactly ``kp`` is needed.

    North of the line the aurora may be seen at that Kp. Solved per longitude on a
    0.05 degree latitude grid; dipole latitude rises monotonically towards the pole
    along a meridian at these latitudes.
    """
    from aurora.physics.geomag import EPOCH, geomagnetic_latitude

    target = OVAL_EDGE_KP0 - VIEW_OFFSET - OVAL_DEG_PER_KP * float(kp)
    lons = np.arange(-180.0, 180.1, 2.0) if lons is None else np.asarray(lons, dtype=float)
    lats = np.arange(0.0, 89.95, 0.05)
    mlat = geomagnetic_latitude(lats[None, :], lons[:, None], EPOCH if year is None else year)
    points = []
    for lon, row in zip(lons, mlat, strict=True):
        above = np.nonzero(row >= target)[0]
        if len(above) == 0 or above[0] == 0:
            continue
        i = above[0]
        frac = (target - row[i - 1]) / (row[i] - row[i - 1])
        points.append((round(float(lats[i - 1] + frac * 0.05), 3), float(lon)))
    return points
