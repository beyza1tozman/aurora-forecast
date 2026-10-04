"""Evaluation figures in the dashboard's dark theme (used by scripts/evaluate.py)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Theme tokens, matching the app's design direction. Series colours validated
# (dataviz validate_palette.js, dark mode) against the panel colour: CVD dE 17.3.
BG = "#0b1020"
PANEL = "#0f1629"
GRID = "#1e2740"
TEXT = "#e6e9f2"
MUTED = "#8b93a7"
MODEL = "#199e70"  # aurora aqua
BASELINE = "#9085e9"  # violet

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "legend.fontsize": 8,
        "font.family": ["Inter", "IBM Plex Sans", "Segoe UI", "DejaVu Sans"],
        "savefig.facecolor": BG,
        "savefig.dpi": 150,
    }
)


def reliability_grid(curves: dict, path) -> None:
    """Grid of reliability curves; curves[(threshold, horizon)] = {name: reliability_curve df}."""
    thresholds = sorted({k[0] for k in curves})
    horizons = sorted({k[1] for k in curves})
    fig, axes = plt.subplots(
        len(thresholds), len(horizons), figsize=(3.0 * len(horizons), 3.0 * len(thresholds))
    )
    styles = {"model": (MODEL, "o"), "cond. persistence": (BASELINE, "s")}
    for i, thr in enumerate(thresholds):
        for j, h in enumerate(horizons):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
            for name, df in curves[(thr, h)].items():
                color, marker = styles[name]
                ax.plot(
                    df["p_mean"], df["observed"], color=color, lw=2, marker=marker, ms=5, label=name
                )
            ax.set(xlim=(0, 1), ylim=(0, 1), title=f"Kp ≥ {thr}, +{h} h")
            if i == len(thresholds) - 1:
                ax.set_xlabel("forecast probability")
            if j == 0:
                ax.set_ylabel("observed frequency")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Reliability on the test period (2020–2026)", color=TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def feature_importance(importance: pd.Series, path, title: str) -> None:
    """Horizontal bars of normalised gain, largest on top."""
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(imp) + 0.8))
    ax.barh(imp.index, imp.to_numpy(), color=MODEL, height=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of total gain")
    ax.set_title(title, loc="left")
    for y, v in enumerate(imp.to_numpy()):
        ax.text(v, y, f" {v:.1%}", va="center", color=MUTED, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
