# Model card: Kp nowcast/forecast from L1 solar wind

## What it does
Three LightGBM models (one per lead time: **+1 h, +3 h, +6 h** after issue) predict the
planetary Kp index class `{≤3, 4, 5, 6, 7+}` from upstream solar-wind measurements. The app
uses the exceedance probabilities **P(Kp≥5), P(Kp≥6), P(Kp≥7)** for the "now and next hours"
panel. These are calibrated probabilities. The location "chance" that combines them with
clouds, darkness and the moon is a heuristic score, not a probability.

**Not for:** anything beyond ~6 h. Storms 1–3 days out come from CMEs that L1 cannot see yet,
so the "next 3 nights" panel uses NOAA's forecast instead (see the [NOAA comparison](#noaa-3-day-forecast-scored-separately)).

## Data
| | Source | Notes |
|---|---|---|
| Inputs | NASA OMNI2 hourly (`omni2_all_years.dat`) | Solar wind time-shifted to the bow-shock nose. 1998 onward (ACE/Wind/DSCOVR era). |
| Target | GFZ Potsdam definitive Kp | 3-hourly. Non-definitive (nowcast) values are dropped as targets. |
| Live inputs | NOAA SWPC `json/rtsw/*` 1-min feeds | Measured **at L1, not shifted**. Must be shifted by ≈ x_GSE / V_x before feature building. |

Only inputs that exist in the live feed are used: By, Bz (GSM), |B|, V, n, T. Derived
quantities (Newell coupling, E_y, dynamic pressure, clock angle) are recomputed from them.
OMNI's own Dst/AE/pressure/E-field columns are not used.

## Features (`aurora/features.py`, shared by training and serving)
35 features, all computed from data at or before the row time:
- Instantaneous: By, Bz, |B|, Bt, V, n, T, clock angle, **Newell coupling**
  `V^(4/3) Bt^(2/3) sin^(8/3)(θ/2)`, `E_y = −V·Bz`, dynamic pressure.
- History: 3 h and 6 h means of Bz, Newell, V, n, P_dyn; min and std of Bz; hours since Bz turned south.
- Kp history (only intervals completed at issue time): last Kp, the one before, 24 h max,
  Kp 27 days earlier, hours into the current interval.
- Season and time of day: sin/cos of day-of-year and UT (Russell–McPherron effect).

Data gaps stay NaN. LightGBM handles them natively, and no imputation is done.

**Timing convention:** an OMNI row labelled t averages [t, t+1 h), so the forecast is issued at
t+1 h. The target for horizon h is the Kp interval containing t + 1 h + h.

## Split
| Split | Period | Rows (h=3) | Kp≥5 | Kp≥6 | Kp≥7 |
|---|---|---|---|---|---|
| Train | 1998 – 2014 | 149,016 | 4.4% | 1.5% | 0.53% |
| Val | 2015 – 2019 | 43,104 | 3.5% | 0.9% | 0.15% |
| Test | 2020 – Aug 2026 (end of OMNI) | 57,716 | 3.9% | 1.3% | 0.52% |

Time-based split. The first **30 days** of val and test are an embargo gap (Kp is autocorrelated
and recurs every 27 days). Never random: neighbouring hours are near-duplicates. The test period
covers the rise and maximum of Solar Cycle 25, including the May 2024 (Gannon) and October 2024 storms.

## Training and calibration
- Multiclass LightGBM, learning rate 0.05, 31 leaves, ≥200 rows per leaf, 0.8 feature/bagging fraction,
  L2 = 1. Early stopping on val log loss (190 / 165 / 114 rounds for h = 1 / 3 / 6).
- **No class weights.** They would inflate rare-class probabilities and break calibration.
- **Temperature scaling** on val: one parameter T per horizon (0.98 / 0.97 / 0.95), so the raw model was
  already close to calibrated.
- Exceedance probabilities are **tail sums** of class probabilities, so
  P(Kp≥7) ≤ P(Kp≥6) ≤ P(Kp≥5) holds by construction (tested in `tests/test_probabilities.py`).

## Results (test period, `reports/metrics.json`)
Brier skill score vs climatology, 95% CI from a 7-day block bootstrap. *Cond. persistence* =
P(target class | current class) learned on train. It is the strongest baseline. Plain 0/1 persistence
scores −0.0 to −0.4 and 27-day recurrence about −0.9 (solar maximum: CME storms do not recur).

| Horizon | Threshold | Model | Cond. persistence |
|---|---|---|---|
| +1 h | Kp≥5 | **0.45** [0.39, 0.49] | 0.28 [0.23, 0.33] |
| +1 h | Kp≥6 | **0.42** [0.33, 0.50] | 0.30 [0.21, 0.39] |
| +1 h | Kp≥7 | **0.44** [0.29, 0.54] | 0.28 [0.14, 0.37] |
| +3 h | Kp≥5 | **0.32** [0.27, 0.36] | 0.19 [0.14, 0.24] |
| +3 h | Kp≥6 | **0.27** [0.18, 0.35] | 0.20 [0.11, 0.28] |
| +3 h | Kp≥7 | **0.27** [0.14, 0.36] | 0.18 [0.07, 0.26] |
| +6 h | Kp≥5 | **0.21** [0.16, 0.25] | 0.12 [0.08, 0.16] |
| +6 h | Kp≥6 | **0.16** [0.08, 0.22] | 0.12 [0.06, 0.18] |
| +6 h | Kp≥7 | 0.11 [0.05, 0.15] | 0.10 [0.04, 0.16] |

- The model beats every baseline at every horizon. Skill falls with lead time, as expected
  from the physics: L1 gives ~30–60 min of real warning, and beyond that skill comes from
  storm persistence.
- At +6 h for Kp≥7 it only ties conditional persistence.
- BSS vs 0/1 persistence: +0.35 to +0.48 at all horizons.
- **Storm windows** (±2 days around Kp≥7): BSS vs climatology is higher than overall (e.g. 0.66 at +1 h for Kp≥5,
  0.36 at +6 h), because that is where the solar-wind inputs carry most information.
- **Decision-threshold scores** (TSS, threshold chosen on val): 0.81–0.84 at +1 h, 0.72–0.76 at +3 h, 0.63–0.66 at +6 h.
- **Reliability** (`reports/reliability.png`): close to the diagonal for Kp≥5 and Kp≥6 at all horizons. For Kp≥7 the
  high-probability bins are noisy (few events) and slightly over-confident at +3 h and +6 h.

**Storm onsets.** There were 32 test-period onsets of Kp≥7 after ≥48 h without one. Three hours ahead, the model
gave P(Kp≥6) ≥ 0.10 (≈ 8× the base rate) for 12 of them, versus 11 for conditional persistence. Six hours ahead
it did so for 6. The biggest sudden-commencement storms were **missed**: May 10 2024, Oct 10 2024,
Nov 12 2025 and Jan 19 2026 all got P < 0.01. A CME shock arrives at L1 only ~30–60 min before
Earth, so a model issued 3 h ahead cannot see it. This is a physical limit, not a tuning problem.

**Feature importance** (gain, `reports/feature_importance.png`): last Kp dominates (35% at +1 h, 23% at +6 h).
Among solar-wind inputs, **Newell coupling** is top at +1 h (14%). At longer horizons, |B| and dynamic pressure
(markers of CME sheaths and stream interaction regions) take over.

## NOAA 3-day forecast (scored separately)
This is **not** a comparison with our model: NOAA forecasts 1–3 days ahead and we forecast 1–6 hours ahead. It checks
how much skill the source of our "next 3 nights" panel has. `scripts/evaluate_noaa.py` scores the 00:30 UT
issues of SWPC's 3-day forecast from the NCEI archive: 1,248 issues, Mar 2022 – Aug 2026, with ~22% of days missing
from the archive. Observed Kp is GFZ definitive. Baselines: persistence (last completed Kp at issue time) and
27-day recurrence. All three are deterministic Kp values, so the scores are MAE and TSS/HSS, not Brier.

| Lead | Forecast | MAE | TSS Kp≥5 | HSS Kp≥5 | TSS Kp≥7 | Kp≥7 hits / events |
|---|---|---|---|---|---|---|
| Day 1 (2.5–20.5 h) | NOAA | 1.14 | 0.24 | 0.21 | 0.23 | 17 / 74 |
| | persistence | **1.10** | 0.23 | 0.22 | **0.29** | 22 / 74 |
| Day 2 (23.5–44.5 h) | NOAA | **1.16** | **0.15** | **0.16** | **0.11** | 9 / 83 |
| | persistence | 1.36 | 0.06 | 0.06 | 0.04 | |
| Day 3 (47.5–68.5 h) | NOAA | **1.14** | **0.10** | **0.12** | 0.00 | 0 / 80 (never forecast) |
| | persistence | 1.45 | 0.02 | 0.02 | −0.01 | |

27-day recurrence is close to zero skill at every lead (TSS ≤ 0.06), as in the hourly evaluation: this is solar maximum.

What this means for the app:
- **Day 1:** NOAA is no better than persistence. For tonight, our hourly model plus the current Kp is the stronger signal.
- **Days 2–3:** NOAA clearly beats both baselines, because it uses CME observations that L1 cannot see. Its skill is
  still modest, and it never forecast Kp≥7 three days out. That is why the 3-night panel shows ranges and lower
  confidence.
- Caveat: persistence uses definitive Kp here. In real time only NOAA's estimate exists, so persistence
  is slightly flattered.

## Known limitations and skew
- **Bow-shock shift.** Training inputs are OMNI (shifted to the bow shock). Live inputs are at L1 and must be
  shifted by the measured spacecraft distance / V_x. An error in this shift moves features by up to ~1 h.
- **Definitive vs estimated Kp.** Training targets and Kp features use GFZ definitive Kp. Live, only NOAA's
  estimated Kp exists. They usually agree within a third, but the difference is real skew for the
  `kp_last` feature, the most important one.
- **Spacecraft changes.** The live feed switches between SWFO-L1, ACE and IMAP (`active == true`). Plasma
  instruments differ, especially for density and temperature.
- **Quiet calibration set.** Val (2015–2019) is the declining phase and minimum of Solar Cycle 24, with only
  0.15% Kp≥7 rows. The temperature is fitted mostly on quiet conditions. Test reliability is still good,
  but Kp≥7 calibration rests on few events.
- **Small storm sample.** Kp≥7 CIs are wide. Do not over-read differences of a few hundredths.
- **Kp is planetary.** It summarises activity at ~±50–60° geomagnetic latitude. Local auroral visibility also
  depends on substorm timing within the 3-hour interval and on the local time sector.

## Reproduce
```
python scripts/download_data.py
python scripts/build_dataset.py
python scripts/train.py        # writes models/kp_h{1,3,6}.txt + calibration.json
python scripts/evaluate.py     # writes reports/metrics.json + figures
python scripts/evaluate_noaa.py
```
