"""Score forecasts of P(Kp >= 5/6/7) on the validation and test periods.

Day 2: baselines only (the bar the model has to beat).
Writes reports/baselines.json and prints markdown tables for the test period.

All baselines are scored on the same rows: definitive target, known current Kp
and known Kp 27 days earlier. The TSS/HSS decision threshold is chosen on val
and then applied unchanged on test.
"""

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
    THRESHOLDS,
    exceedance_from_class_probs,
    kp_to_class,
    load_modelling_frame,
    threshold_class,
)
from aurora.metrics import (
    best_tss_threshold,
    block_bootstrap_bss,
    brier,
    brier_skill,
    contingency,
    hss,
    pr_auc,
    tss,
    week_blocks,
)

REPORTS_DIR = ROOT / "reports"
EVAL_SPLITS = ("val", "test")


def baseline_probs(frame: pd.DataFrame, kp: pd.Series, h: int) -> dict[str, np.ndarray]:
    """Class probabilities of every baseline for every row of ``frame``."""
    train = frame[frame["split"] == "train"]
    table = fit_conditional_persistence(kp_to_class(train["kp_last"]), train[f"cls_h{h}"])
    return {
        "climatology": climatology(train[f"cls_h{h}"], len(frame)),
        "persistence": persistence(frame["kp_last"]),
        "recurrence": recurrence(kp, frame.index, h),
        "cond_persistence": conditional_persistence(frame["kp_last"], table),
    }


def evaluate_horizon(frame: pd.DataFrame, kp: pd.Series, h: int) -> list[dict]:
    probs = baseline_probs(frame, kp, h)
    usable = frame[f"cls_h{h}"].notna().to_numpy().copy()
    for p in probs.values():
        usable &= ~np.isnan(p).any(axis=1)

    rows = []
    for thr in THRESHOLDS:
        y_all = (frame[f"cls_h{h}"].to_numpy() >= threshold_class(thr)).astype(float)
        exceed = {name: exceedance_from_class_probs(p, thr) for name, p in probs.items()}
        decision = {}
        for split in EVAL_SPLITS:
            mask = usable & (frame["split"] == split).to_numpy()
            y = y_all[mask]
            ref_clim, ref_pers = exceed["climatology"][mask], exceed["persistence"][mask]
            blocks = week_blocks(frame.index[mask])
            for name, p_all in exceed.items():
                p = p_all[mask]
                if split == "val":
                    decision[name] = best_tss_threshold(p, y)
                table = contingency(p, y, decision[name])
                ci = block_bootstrap_bss(p, y, ref_clim, blocks) if split == "test" else None
                rows.append(
                    {
                        "split": split,
                        "horizon_h": h,
                        "threshold": thr,
                        "baseline": name,
                        "n": int(mask.sum()),
                        "events": int(y.sum()),
                        "base_rate": float(y.mean()),
                        "brier": brier(p, y),
                        "bss_clim": brier_skill(p, y, ref_clim),
                        "bss_clim_ci95": ci,
                        "bss_pers": brier_skill(p, y, ref_pers),
                        "pr_auc": pr_auc(p, y),
                        "decision_threshold": decision[name],
                        "tss": tss(table),
                        "hss": hss(table),
                    }
                )
    return rows


def fmt(x: float) -> str:
    return "  nan" if x is None or np.isnan(x) else f"{x:+.3f}"


def print_tables(results: pd.DataFrame, split: str = "test") -> None:
    sub = results[results["split"] == split]
    for thr in THRESHOLDS:
        t = sub[sub["threshold"] == thr]
        first = t.iloc[0]
        print(f"\n### {split}: P(Kp >= {thr})  (events at h=1: {first.events}/{first.n})\n")
        print("| baseline | h | Brier | BSS vs clim [95% CI] | BSS vs pers | PR-AUC | TSS | HSS |")
        print("|---|---|---|---|---|---|---|---|")
        for r in t.itertuples():
            ci = r.bss_clim_ci95
            ci_txt = f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
            print(
                f"| {r.baseline} | {r.horizon_h} | {r.brier:.5f} | {fmt(r.bss_clim)}{ci_txt} "
                f"| {fmt(r.bss_pers)} | {r.pr_auc:.3f} | {fmt(r.tss)} | {fmt(r.hss)} |"
            )


def main() -> None:
    frame = load_modelling_frame()
    kp = pd.read_parquet(PROCESSED_DIR / "kp_3h.parquet")["kp"]

    print("Base rates per split (h=3 target, definitive Kp only):")
    for split in ("train", "val", "test"):
        cls = frame.loc[frame["split"] == split, "cls_h3"].dropna()
        rates = "  ".join(f"Kp>={t}: {(cls >= threshold_class(t)).mean():.2%}" for t in THRESHOLDS)
        print(f"  {split:5s} n={len(cls):6d}  {rates}")

    results = pd.DataFrame([r for h in HORIZONS for r in evaluate_horizon(frame, kp, h)])

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "baselines.json"
    out.write_text(json.dumps(results.to_dict(orient="records"), indent=2))
    print_tables(results)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
