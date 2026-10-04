"""Verification metrics for binary exceedance forecasts (P(Kp >= k) vs outcome).

``p`` are forecast probabilities, ``y`` binary outcomes (0/1), both 1-D arrays.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def brier(p, y) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_skill(p, y, p_ref) -> float:
    """BSS = 1 - BS / BS_ref. 0 = no better than the reference, 1 = perfect."""
    ref = brier(p_ref, y)
    return float(1 - brier(p, y) / ref) if ref > 0 else float("nan")


def pr_auc(p, y) -> float:
    """Average precision; more informative than ROC-AUC for rare events."""
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def reliability_curve(p, y, n_bins: int = 10) -> pd.DataFrame:
    """Mean forecast vs observed frequency per probability bin (empty bins dropped)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y})
    out = df.groupby("bin").agg(p_mean=("p", "mean"), observed=("y", "mean"), count=("y", "size"))
    return out.reset_index(drop=True)


def contingency(p, y, threshold: float) -> dict[str, int]:
    """2x2 table for the yes/no forecast p >= threshold."""
    yes = np.asarray(p) >= threshold
    obs = np.asarray(y).astype(bool)
    return {
        "hits": int(np.sum(yes & obs)),
        "false_alarms": int(np.sum(yes & ~obs)),
        "misses": int(np.sum(~yes & obs)),
        "correct_negatives": int(np.sum(~yes & ~obs)),
    }


def tss(table: dict[str, int]) -> float:
    """True Skill Statistic = hit rate - false alarm rate (independent of base rate)."""
    a, b = table["hits"], table["false_alarms"]
    c, d = table["misses"], table["correct_negatives"]
    if a + c == 0 or b + d == 0:
        return float("nan")
    return a / (a + c) - b / (b + d)


def hss(table: dict[str, int]) -> float:
    """Heidke Skill Score: accuracy relative to random chance."""
    a, b = table["hits"], table["false_alarms"]
    c, d = table["misses"], table["correct_negatives"]
    denom = (a + c) * (c + d) + (a + b) * (b + d)
    return 2 * (a * d - b * c) / denom if denom else float("nan")


def best_tss_threshold(p, y, grid=None) -> float:
    """Probability threshold maximising TSS (choose on val, apply on test)."""
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    scores = [tss(contingency(p, y, t)) for t in grid]
    if np.all(np.isnan(scores)):
        return 0.5
    return float(grid[int(np.nanargmax(scores))])


def week_blocks(times: pd.DatetimeIndex) -> np.ndarray:
    """Block id per row for a 7-day block bootstrap (storms span days, rows are autocorrelated)."""
    return np.asarray(times.floor("7D").asi8)


def block_bootstrap_bss(
    p, y, p_ref, blocks, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """(lo, hi) percentile CI of the BSS, resampling whole blocks with replacement."""
    p, y, p_ref = (np.asarray(a, dtype=float) for a in (p, y, p_ref))
    _, block_idx = np.unique(blocks, return_inverse=True)
    n_blocks = block_idx.max() + 1
    se = np.bincount(block_idx, weights=(p - y) ** 2, minlength=n_blocks)
    se_ref = np.bincount(block_idx, weights=(p_ref - y) ** 2, minlength=n_blocks)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    ref_sum = se_ref[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bss = 1 - se[draws].sum(axis=1) / ref_sum
    bss = bss[ref_sum > 0]
    if len(bss) == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(bss, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
