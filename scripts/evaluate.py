"""Score the model and baselines for P(Kp >= 5/6/7) on the validation and test periods.

Writes reports/metrics.json, reports/reliability.png, reports/feature_importance.png
and prints markdown tables.

All forecasts are scored on the same rows: definitive target, known current Kp
and known Kp 27 days earlier. The TSS/HSS decision threshold is chosen on val
and applied unchanged on test. "storms" = test rows whose target interval lies
within +-2 days of a Kp >= 7 interval.
"""

import os

# OpenBLAS (bundled with numpy and scipy) commits ~30 MB per CPU thread at import, once per
# library: ~1 GB on a 16-thread laptop. We do not need BLAS threads, so cap them first.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json

import numpy as np
import pandas as pd

from aurora.baselines import (
    climatology,
    conditional_persistence,
    fit_conditional_persistence,
    persistence,
    recurrence,
)
from aurora.config import PROCESSED_DIR, ROOT
from aurora.dataset import (
    HORIZONS,
    N_CLASSES,
    THRESHOLDS,
    exceedance_from_class_probs,
    kp_to_class,
    load_modelling_frame,
    target_start,
    threshold_class,
)
from aurora.features import FEATURE_COLUMNS, ISSUE_LAG
from aurora.metrics import (
    best_tss_threshold,
    block_bootstrap_bss,
    brier,
    brier_skill,
    contingency,
    hss,
    pr_auc,
    reliability_curve,
    tss,
    week_blocks,
)
from aurora.model import load_models

REPORTS_DIR = ROOT / "reports"
EVAL_SPLITS = ("val", "test")
STORM_KP = 7
STORM_WINDOW = pd.Timedelta("2D")
ONSET_QUIET = pd.Timedelta("48h")  # Kp < 7 for this long before an onset


def all_forecasts(frame, kp, models, h) -> dict[str, np.ndarray]:
    """Class probabilities of the model and every baseline for every row of ``frame``."""
    train = frame[frame["split"] == "train"]
    table = fit_conditional_persistence(kp_to_class(train["kp_last"]), train[f"cls_h{h}"])
    # The model is only scored on val/test; predicting just those rows keeps memory low.
    model = np.full((len(frame), N_CLASSES), np.nan)
    scored = frame["split"].isin(EVAL_SPLITS).to_numpy()
    model[scored] = models[h].predict_proba(frame.loc[scored, FEATURE_COLUMNS])
    return {
        "model": model,
        "cond_persistence": conditional_persistence(frame["kp_last"], table),
        "persistence": persistence(frame["kp_last"]),
        "climatology": climatology(train[f"cls_h{h}"], len(frame)),
        "recurrence": recurrence(kp, frame.index, h),
    }


def storm_mask(index: pd.DatetimeIndex, kp: pd.Series, h: int) -> np.ndarray:
    """Rows whose target interval is within STORM_WINDOW of a Kp >= STORM_KP interval."""
    storm = (kp_to_class(kp) >= threshold_class(STORM_KP)).astype(float)
    n = int(STORM_WINDOW / pd.Timedelta("3h"))
    near = pd.Series(storm, index=kp.index).rolling(2 * n + 1, center=True, min_periods=1).max()
    return near.reindex(target_start(index, h)).fillna(0).to_numpy() > 0


def score(name, p, y, ref_clim, ref_cond, ref_pers, blocks, decision) -> dict:
    table = contingency(p, y, decision)
    with_ci = blocks is not None
    return {
        "forecast": name,
        "n": len(y),
        "events": int(y.sum()),
        "base_rate": float(y.mean()),
        "brier": brier(p, y),
        "bss_clim": brier_skill(p, y, ref_clim),
        "bss_clim_ci95": block_bootstrap_bss(p, y, ref_clim, blocks) if with_ci else None,
        "bss_cond": brier_skill(p, y, ref_cond),
        "bss_cond_ci95": block_bootstrap_bss(p, y, ref_cond, blocks) if with_ci else None,
        "bss_pers": brier_skill(p, y, ref_pers),
        "pr_auc": pr_auc(p, y),
        "decision_threshold": decision,
        "tss": tss(table),
        "hss": hss(table),
    }


def evaluate_horizon(frame, kp, models, h):
    probs = all_forecasts(frame, kp, models, h)
    usable = frame[f"cls_h{h}"].notna().to_numpy().copy()
    for p in probs.values():
        usable &= ~np.isnan(p).any(axis=1)
    storms = storm_mask(frame.index, kp, h)

    rows, curves = [], {}
    for thr in THRESHOLDS:
        y_all = (frame[f"cls_h{h}"].to_numpy() >= threshold_class(thr)).astype(float)
        exceed = {name: exceedance_from_class_probs(p, thr) for name, p in probs.items()}
        decision = {}
        for split, subset in [("val", "all"), ("test", "all"), ("test", "storms")]:
            mask = usable & (frame["split"] == split).to_numpy()
            if subset == "storms":
                mask &= storms
            y = y_all[mask]
            refs = [exceed[r][mask] for r in ("climatology", "cond_persistence", "persistence")]
            blocks = week_blocks(frame.index[mask]) if split == "test" else None
            for name, p_all in exceed.items():
                p = p_all[mask]
                if split == "val":
                    decision[name] = best_tss_threshold(p, y)
                row = score(name, p, y, *refs, blocks, decision[name])
                rows.append(
                    {"split": split, "subset": subset, "horizon_h": h, "threshold": thr} | row
                )
            if (split, subset) == ("test", "all"):
                curves[(thr, h)] = {
                    "model": reliability_curve(exceed["model"][mask], y),
                    "cond. persistence": reliability_curve(exceed["cond_persistence"][mask], y),
                }
    return rows, curves


def storm_onsets(kp: pd.Series, frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Test-period Kp >= 7 intervals preceded by ONSET_QUIET without one."""
    storm = pd.Series(kp_to_class(kp) >= threshold_class(STORM_KP), index=kp.index)
    quiet_before = storm.shift(1, fill_value=False).astype(float).rolling(ONSET_QUIET).max() == 0
    onsets = storm.index[storm & quiet_before]
    test_start = frame.index[frame["split"] == "test"].min()
    return onsets[(onsets >= test_start) & (onsets <= frame.index.max())]


def onset_table(frame, kp, models) -> pd.DataFrame:
    """P(Kp >= 6) that each forecast gave for the onset interval, issued h hours before it."""
    train = frame[frame["split"] == "train"]
    tables = {
        h: fit_conditional_persistence(kp_to_class(train["kp_last"]), train[f"cls_h{h}"])
        for h in (3, 6)
    }
    rows = []
    for onset in storm_onsets(kp, frame):
        row = {"onset": onset, "kp": float(kp[onset])}
        for h in (3, 6):
            t = onset - pd.Timedelta(hours=h) - ISSUE_LAG  # issue time = onset - h
            if t not in frame.index:
                continue
            one = frame.loc[[t]]
            row[f"model_h{h}"] = float(
                exceedance_from_class_probs(models[h].predict_proba(one), 6)[0]
            )
            cond = conditional_persistence(one["kp_last"], tables[h])
            row[f"cond_h{h}"] = float(exceedance_from_class_probs(cond, 6)[0])
        rows.append(row)
    return pd.DataFrame(rows)


def gain_importance(models) -> pd.DataFrame:
    gains = {
        f"h{h}": pd.Series(m.booster.feature_importance("gain"), index=m.booster.feature_name())
        for h, m in models.items()
    }
    df = pd.DataFrame(gains)
    return df / df.sum()


def fmt(x) -> str:
    return "   nan" if x is None or np.isnan(x) else f"{x:+.3f}"


def fmt_ci(ci) -> str:
    return f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else ""


def print_tables(results: pd.DataFrame, split: str, subset: str) -> None:
    sub = results[(results["split"] == split) & (results["subset"] == subset)]
    for thr in THRESHOLDS:
        t = sub[sub["threshold"] == thr]
        first = t.iloc[0]
        print(
            f"\n### {split}/{subset}: P(Kp >= {thr})  "
            f"(events at h=1: {first.events}/{first.n}, base rate {first.base_rate:.2%})\n"
        )
        print("| forecast | h | Brier | BSS vs clim | BSS vs cond. pers. | PR-AUC | TSS | HSS |")
        print("|---|---|---|---|---|---|---|---|")
        for r in t.itertuples():
            print(
                f"| {r.forecast} | {r.horizon_h} | {r.brier:.5f} "
                f"| {fmt(r.bss_clim)}{fmt_ci(r.bss_clim_ci95)} "
                f"| {fmt(r.bss_cond)}{fmt_ci(r.bss_cond_ci95)} "
                f"| {r.pr_auc:.3f} | {fmt(r.tss)} | {fmt(r.hss)} |"
            )


def main() -> None:
    from aurora import plots

    frame = load_modelling_frame()
    kp = pd.read_parquet(PROCESSED_DIR / "kp_3h.parquet")["kp"]
    models = load_models()

    rows, curves = [], {}
    for h in HORIZONS:
        r, c = evaluate_horizon(frame, kp, models, h)
        rows += r
        curves |= c
    results = pd.DataFrame(rows)
    onsets = onset_table(frame, kp, models)
    importance = gain_importance(models)

    REPORTS_DIR.mkdir(exist_ok=True)
    report = {
        "results": results.to_dict(orient="records"),
        "storm_onsets": json.loads(onsets.to_json(orient="records", date_format="iso")),
        "feature_importance_gain": importance.round(5).to_dict(),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    plots.reliability_grid(curves, REPORTS_DIR / "reliability.png")
    top = importance["h3"].sort_values(ascending=False).head(15)
    plots.feature_importance(
        top, REPORTS_DIR / "feature_importance.png", "Top 15 features, +3 h model"
    )

    print_tables(results, "test", "all")
    print_tables(results, "test", "storms")
    print("\n### Storm onsets (Kp >= 7 after 48 h without): P(Kp >= 6) issued h hours before\n")
    print(onsets.set_index("onset").round(3).to_string())
    print("\nTop features by gain (+3 h):")
    print(top.round(3).to_string())
    print("\nwrote reports/metrics.json, reliability.png, feature_importance.png")


if __name__ == "__main__":
    main()
