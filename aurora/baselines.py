"""Reference forecasts the model must beat.

Every baseline returns class probabilities of shape (n, N_CLASSES), so the
same tail sums and metrics apply to baselines and the model. Rows whose input
is missing get NaN probabilities.
"""

import numpy as np
import pandas as pd

from aurora.dataset import N_CLASSES, kp_to_class, target_start
from aurora.features import RECURRENCE


def one_hot(classes) -> np.ndarray:
    """(n, N_CLASSES) one-hot rows; NaN classes give NaN rows."""
    classes = np.asarray(classes, dtype=float)
    out = np.full((len(classes), N_CLASSES), np.nan)
    ok = ~np.isnan(classes)
    out[ok] = np.eye(N_CLASSES)[classes[ok].astype(int)]
    return out


def persistence(kp_last) -> np.ndarray:
    """Kp stays at its last completed value (deterministic, 0/1 probabilities)."""
    return one_hot(kp_to_class(kp_last))


def climatology(train_classes, n: int) -> np.ndarray:
    """Training-period class frequencies, the same for every row."""
    classes = np.asarray(train_classes, dtype=float)
    classes = classes[~np.isnan(classes)].astype(int)
    freq = np.bincount(classes, minlength=N_CLASSES) / len(classes)
    return np.tile(freq, (n, 1))


def recurrence(kp: pd.Series, index: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """Kp one solar rotation (27 days) before the target interval."""
    past = kp.reindex(target_start(index, horizon) - RECURRENCE).to_numpy()
    return one_hot(kp_to_class(past))


def fit_conditional_persistence(current_classes, target_classes, prior: float = 0.5) -> np.ndarray:
    """Transition table P(target class | current class) from training data.

    A small additive prior keeps unseen transitions (e.g. 7+ -> <=3) above zero.
    """
    cur = np.asarray(current_classes, dtype=float)
    tgt = np.asarray(target_classes, dtype=float)
    ok = ~(np.isnan(cur) | np.isnan(tgt))
    counts = np.full((N_CLASSES, N_CLASSES), prior)
    np.add.at(counts, (cur[ok].astype(int), tgt[ok].astype(int)), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def conditional_persistence(kp_last, table: np.ndarray) -> np.ndarray:
    """Probabilistic persistence: the transition-table row of the current Kp class."""
    cur = kp_to_class(kp_last)
    out = np.full((len(cur), N_CLASSES), np.nan)
    ok = ~np.isnan(cur)
    out[ok] = table[cur[ok].astype(int)]
    return out
