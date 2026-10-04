# Aurora Forecast: 7-Day Plan

## Context
Portfolio project for ML/space-industry Werkstudent applications. A user picks a location in Germany or Europe and sees the aurora chance (1) now and for the next few hours, (2) for the next 3 nights, and (3) as a low-confidence hint for the next weeks. Confidence must visibly fall as the horizon gets longer. The ML core forecasts Kp from L1 solar wind data. The production side is FastAPI, a map, Docker, HF Spaces, CI, Sentry, /health and /monitoring. You have 7 days and a Windows laptop with CPU only.

**Design direction:** clear and professional, like a modern space-data dashboard. Dark theme, aurora green/teal accents, clean typography and a dark map style. (This replaces the earlier "cozy / Ghibli" idea.)

**Repo state:** the local branch is `main`. `design-references/` exists locally but is gitignored, so it is not committed.

---

## 1. The most important decision: which horizon each part comes from

This is the decision interviewers are most likely to probe, and it shapes the whole app.

| Horizon | Source | Why | Confidence shown |
|---|---|---|---|
| Now and the next 1–6 h | **Your LightGBM model** on L1 solar wind | L1 is about 1.5 million km upstream. Solar wind at 400–800 km/s reaches Earth in about 30–60 min, so you *measure* what will hit. Past about 1 h, skill comes from the persistence of storm conditions, so skill falls quickly with lead time. | High → medium |
| Next 3 nights | **NOAA SWPC 3-day Kp forecast**, calibrated on its own archive (your model for the hours it reaches tonight) | Days-ahead storms come from CMEs that leave the Sun 1–3 days earlier. L1 data cannot see them yet. NOAA uses coronagraphs and WSA-Enlil modelling. Being honest about this is a strength in an interview. | Medium → low |
| Next ~4 weeks | **27-day recurrence**: Kp from one and two solar rotations ago, plus NOAA's 27-day outlook | Coronal holes last several rotations, so their fast-wind streams return about every 27 days. This works best in the declining phase of the solar cycle and badly for CMEs. | Low, shown as a "possible activity" hint only |

On the cloud side, Open-Meteo forecasts also get less reliable past about 48 h. That is a second reason the 3-night outlook is fuzzier.

---

## 2. ML decisions, explained simply

**Data**
- **Inputs:** NASA OMNI2 **hourly** (`omni2_all_years.dat`, SPDF). It is one file, loads fast on a CPU, and hourly resolution matches a 3-hour target. Use OMNI 5-min only if you finish early.
- **Target:** GFZ Potsdam definitive Kp (`Kp_ap_Ap_SN_F107_since_1932.txt`). The target for horizon *h* is the Kp of the 3-hour UT bin that contains *t + h*. Train one model per horizon: h = 1, 3 and 6 h.
- **Period:** 1998–present (ACE/DSCOVR era, good coverage).

**Train/serve skew (strong interview point)**
OMNI is **time-shifted to the bow shock**. NOAA's live feed (`services.swpc.noaa.gov/products/solar-wind/*.json`) is **at L1 and not shifted**. In production, shift live data by `delay ≈ 1.5e6 km / V_x` before you build features. Also, historical Kp is *definitive*, but live Kp is NOAA's *estimate*. Note this difference in the model card. Write the feature code once, in `aurora/features.py`, and use it for both training and serving.

**Features (physics-motivated, all computed from data up to t only)**
- IMF Bz (GSM), By, Bt; speed V; density n; dynamic pressure.
- **Newell coupling** `dΦ/dt = V^(4/3) · Bt^(2/3) · sin^(8/3)(θ/2)`, where θ is the IMF clock angle. This is the best single predictor of how much energy enters the magnetosphere, because southward Bz opens the magnetosphere.
- `E_y = −V·Bz` (reconnection electric field).
- Rolling mean/min/std over 1, 3 and 6 h, plus hours since Bz turned southward. The magnetosphere integrates its input, so history matters.
- Past Kp lags. These make persistence available to the model.
- sin/cos of day-of-year and UT, for the Russell–McPherron effect (more activity near the equinoxes).
- LightGBM handles NaNs natively, so data gaps from DSCOVR outages do not need imputation.

**Split: time-based, with an embargo gap**
- Train 1998–2014 · Val 2015–2019 · **Test 2020 → Sep 2026** (end of OMNI). Test covers the rise and maximum of Solar Cycle 25, including the May 2024 "Gannon" storm and the Oct 2024 storm.
- Leave a ~30-day gap between splits. Kp is autocorrelated and recurs every 27 days, so neighbouring rows would leak information.
- Why not a random split? Neighbouring hours are nearly identical. A random split tests memory, not forecasting.

**Model: one multiclass LightGBM per horizon** over Kp bins `{≤3, 4, 5, 6, 7+}`
- Get `P(Kp≥5)`, `P(Kp≥6)` and `P(Kp≥7)` as **tail sums**. They are then automatically monotonic (P≥7 ≤ P≥6 ≤ P≥5). Three separate binary models can contradict each other.
- **No class weights.** They make rare classes look more likely than they are and break calibration. Accept the imbalance and calibrate instead.
- **Calibration:** temperature scaling on the validation set. It has one parameter and keeps the probabilities monotonic.
- Rarity: Kp≥5 is only a few percent of 3-hour bins, and Kp≥7 is well under 1%. Say this explicitly.

**Conventions fixed on Day 2 (code: `aurora/features.py`, `aurora/dataset.py`)**
- **Issue time:** an OMNI row labelled t averages [t, t+1h), so it is known at t+1h. Features in row t use rows ≤ t; the forecast is issued at t+1h.
- **Kp as a feature:** only the last *completed* 3-h interval at issue time (start = floor(t−2h, 3h)), plus the one before, the 24 h max, the value 27 days earlier, and `hours_into_bin` (how stale it is).
- **Target for horizon h:** the Kp interval containing t+1h+h. Non-definitive (nowcast) Kp targets are dropped.
- **Classes:** −/o/+ grouped into integer Kp (5− counts as 5, like NOAA G1): {≤3, 4, 5, 6, 7+}.
- **Live-available inputs only:** By, Bz, |B|, V, n, T. E_y, dynamic pressure and Newell coupling are recomputed from them; OMNI's Dst/AE/E-field/pressure columns are not used.
- **Split:** train 1998–2014, val 2015–2019, test 2020 → end of OMNI (currently Aug 2026). The first 30 days of val and test are embargo.

**Baselines and metrics**
- Baselines: **persistence** (Kp now), **climatology** (base rate), and **27-day recurrence**. Plus **conditional persistence** (P(target class | current class), learned on train): 0/1 persistence has an inflated Brier score for rare events, so beating it proves little. This is the real bar.
- TSS/HSS decision thresholds are chosen on val, then applied on test. BSS confidence intervals use a 7-day block bootstrap, because hours are autocorrelated.
- **Baseline results (test, BSS vs climatology, 95% CI)** from `scripts/evaluate.py` → `reports/baselines.json`:

  | | Kp≥5 h=1 | h=3 | h=6 | Kp≥7 h=1 | h=3 | h=6 |
  |---|---|---|---|---|---|---|
  | persistence (0/1) | −0.07 | −0.24 | −0.41 | −0.00 | −0.21 | −0.37 |
  | recurrence | −0.88 | −0.88 | −0.88 | −1.01 | −1.01 | −1.01 |
  | conditional persistence | **+0.28** [0.23, 0.33] | **+0.19** | **+0.12** | **+0.28** [0.14, 0.37] | **+0.18** | **+0.10** |

  Recurrence is poor because the test period is solar maximum (CME-driven storms do not recur).
- **Model results (Day 3, test, BSS vs climatology)** from `scripts/evaluate.py` → `reports/metrics.json`.
  Details, CIs and limitations are in `MODEL_CARD.md`.

  | | Kp≥5 h=1 | h=3 | h=6 | Kp≥7 h=1 | h=3 | h=6 |
  |---|---|---|---|---|---|---|
  | LightGBM (calibrated) | **+0.45** [0.39, 0.49] | **+0.32** | **+0.21** | **+0.44** [0.29, 0.54] | **+0.27** | +0.11 |
  | conditional persistence | +0.28 | +0.19 | +0.12 | +0.28 | +0.18 | +0.10 |

  The model beats every baseline everywhere except Kp≥7 at 6 h, where it ties conditional persistence.
  Temperatures are 0.95–0.98, so the raw model was nearly calibrated already. Feature importance: last
  Kp first, then Newell coupling at 1 h and |B| / dynamic pressure at 3–6 h. The largest sudden-commencement storms
  (May 10 2024, Oct 10 2024, Nov 12 2025, Jan 19 2026) got P(Kp≥6) < 0.01 three hours ahead. This
  is the L1 warning-time limit and is the reason the 3-night panel uses NOAA.
- Metrics per threshold: **Brier score and Brier Skill Score vs persistence and climatology**, reliability diagrams, PR-AUC (better than ROC-AUC for rare events), and TSS/HSS at a decision threshold.
- **Storm-period evaluation:** test windows of ±2 days around each event with Kp≥7, scored separately. Also do an event-based check: did P(Kp≥6) rise before onset? Use bootstrap confidence intervals, because there are only a handful of Kp≥7 events and they will be wide. Say so.
- **NOAA comparison (done on Day 3):** NOAA's forecasts are 1–3 days ahead and yours are 1–6 h ahead, so a direct comparison is apples to oranges. Instead, NOAA is scored against day-scale baselines.
  NCEI archives the SWPC 3-day forecast text product from March 2022 (`aurora/data/noaa_3day.py` parses it;
  early files use whole-number Kp). `scripts/evaluate_noaa.py` → `reports/noaa_3day.json`, 1,248 issues to Aug 2026:
  - Day 1: NOAA ≈ persistence (MAE 1.14 vs 1.10, TSS Kp≥5 0.24 vs 0.23).
  - Days 2–3: NOAA beats persistence and recurrence (TSS Kp≥5 0.15 / 0.10 vs ≤ 0.06), but never forecast Kp≥7 on day 3.

  This supports the horizon split in section 1: our model for hours, NOAA with soft ranges for nights 2–3.

---

## 3. Location score, explained simply

`chance = P(Kp ≥ Kp_needed(mag_lat)) × clear_sky × darkness × moon_factor` (`aurora/location_score.py`)

- **Geomagnetic latitude:** use a dipole from IGRF coefficients (g10, g11, h11), not `aacgmv2`, which has a painful C build on Windows. In Germany, geomagnetic latitude is close to geographic latitude (within about 1°). In North America it is much higher, which is why Americans see aurora further south.
- **Kp_needed** (`aurora/physics/oval.py`, decided Day 4): the classic Kp-map oval edge, 66.5° − 2.04°·Kp, shifted 1.5° equatorward for aurora seen low on the northern horizon (a 5° shift was too optimistic). Anchored to Central European experience: Hamburg (mlat 53.7) needs Kp ≈ 5.5, Berlin ≈ 6.3, Munich ≈ 8.2. A dipole is ~2–4° off corrected geomagnetic latitude in the UK and North America, so the anchors are only trusted for Central Europe. P(Kp ≥ k) for non-integer k: log-linear between integer anchors; the model gives k = 4–7, climatology fills k < 4 and scales k = 8, 9 from P(Kp≥7).
- **Clouds** (changed Day 4): Open-Meteo low/mid/high layers, combined assuming random overlap: `(1−low)(1−mid)(1−0.5·high)`. Low and mid cloud block the view; thin high cloud only partly. (Using only total and low cloud gave a 40% chance under overcast altostratus.)
- **Darkness** (changed Day 4): 1 below −12° sun elevation, 0 above −6°, linear between. Sun and moon positions come from low-precision Astronomical Almanac formulas in pure NumPy (`aurora/physics/sky.py`; sun ~0.01°, moon ~0.3°), checked against the 2024 eclipse and known full/new moons. This replaces `skyfield` + the 17 MB `de421.bsp`, which is far more precision than a darkness weight needs.
- **Moon:** a mild penalty scaled by illumination and moon altitude, at most −40% for a full moon above 30°. A strong aurora can still be seen under a full moon.
- **Be honest:** this is a heuristic *score*, not a calibrated probability. Only the Kp part is calibrated. Label it "chance" in the UI and explain it in the README. Where the cloud forecast ends, only `chance_if_clear` is shown.

---

## 4. AI layer (daily briefing)

- Use the Claude API with **Haiku 4.5** (`claude-haiku-4-5`), the cheap and fast option.
- The LLM only *phrases* facts. You pass a structured JSON (probabilities, clouds, darkness window, moon, confidence tier), and the prompt forbids new numbers.
- **Cache:** key = (lat/lon rounded to a 0.5° grid, date, forecast-run id), stored in SQLite with about a 3 h TTL. Also use prompt caching on the system prompt.
- **Fallback:** if the API fails or the budget is exhausted, use a Jinja text template. The app never breaks because of the LLM.

---

## 5. Production notes and traps

- **HF Spaces free CPU is ephemeral.** SQLite is wiped on restart, and spaces sleep when inactive. So the **monitoring log must live outside the container**. Recommended: a free Postgres (Neon or Supabase) through `DATABASE_URL`, with SQLite locally. The alternative is appending to a HF Dataset repo.
- UptimeRobot pings `/health` every 5 min. This gives uptime monitoring and should also keep the Space awake (check this on Day 6).
- **Scheduler:** APScheduler in-process. Every 15 min: fetch NOAA, run the forecast, and log it. Every hour: fetch the observed Kp (GFZ nowcast JSON API) and join it to past forecasts.
- **Frontend:** plain HTML/CSS/JS plus **Leaflet**, served by FastAPI as static files. There is no Node build step, so Docker stays simple. Use the dark map tiles **CARTO Dark Matter**, which need no API key. Credit OpenStreetMap and CARTO on the map, and check CARTO's usage terms before relying on them.
- **Live NOAA endpoints (checked on Day 1, 2026-10-03):**
  - The old `products/solar-wind/plasma-*.json` and `mag-*.json` feeds are **gone (404)**. Real-time solar wind is now at `json/rtsw/rtsw_mag_1m.json` and `rtsw_wind_1m.json`: the last ~24 h at 1-min cadence.
  - Each rtsw file mixes several spacecraft (SOLAR1 = SWFO-L1, ACE, IMAP). Use only rows with `active == true`, which is NOAA's operational choice and gives automatic failover.
  - `json/rtsw/rtsw_ephemerides_1h.json` gives the spacecraft's `x_gse` distance (~1.41 million km), so the L1 → bow-shock delay can use the real distance, not a constant.
  - The Kp feeds (`noaa-planetary-k-index*.json`) now return a list of objects rather than a list of lists.
  - All URLs are in `aurora/config.py`. `tests/test_live_contracts.py` (`pytest -m network`) detects future format changes.

---

## 6. Folder structure

```
aurora-forecast/
├── README.md                  # pitch, screenshots, architecture, results table, limitations
├── CLAUDE.md                  # project guide for Claude Code
├── MODEL_CARD.md              # data, split, metrics, calibration, known skew
├── docs/PLAN.md               # this plan
├── pyproject.toml             # deps + ruff + pytest config
├── Dockerfile
├── .github/workflows/ci.yml   # lint + tests (+ deploy to HF on main)
├── .env.example               # ANTHROPIC_API_KEY, SENTRY_DSN, DATABASE_URL
├── design-references/         # local only, gitignored
├── data/                      # gitignored: raw/ + processed/ parquet
├── notebooks/
│   ├── 01_eda.ipynb           # class balance, autocorrelation, 27-day recurrence
│   └── 02_evaluation.ipynb    # reliability diagrams, storm case studies
├── scripts/
│   ├── download_data.py       # OMNI2 + GFZ Kp
│   ├── build_dataset.py
│   ├── train.py               # per-horizon models → models/
│   └── evaluate.py            # writes reports/metrics.json + figures
├── models/                    # small .txt LightGBM boosters + calibration.json (committed)
├── reports/                   # figures used in README
├── aurora/                    # importable package (shared train/serve)
│   ├── config.py
│   ├── data/                  # omni.py, gfz.py, noaa_live.py, open_meteo.py
│   ├── features.py            # SINGLE source of truth for features
│   ├── dataset.py             # targets, Kp classes, time split + embargo
│   ├── model.py               # load, predict, tail sums, temperature scaling
│   ├── baselines.py
│   ├── metrics.py             # Brier/BSS, reliability, TSS, bootstrap CIs
│   ├── physics/               # geomag.py (IGRF dipole), oval.py, sky.py (sun/moon)
│   ├── location_score.py
│   ├── outlook.py             # NOAA 3-day + 27-day recurrence
│   └── briefing.py            # LLM + cache + template fallback
├── app/
│   ├── main.py                # FastAPI, Sentry init, scheduler startup
│   ├── routes/                # forecast.py, health.py, monitoring.py, briefing.py
│   ├── db.py                  # SQLAlchemy: forecasts, observations, briefings
│   ├── scheduler.py
│   └── static/                # index.html, monitoring.html, css/, js/, img/
└── tests/
    ├── test_features_no_leakage.py  # perturb future rows → features at t unchanged
    ├── test_split.py                # no overlap, embargo respected
    ├── test_probabilities.py        # monotonic, in [0,1]
    ├── test_geomag.py               # known city values
    ├── test_location_score.py
    ├── test_api.py                  # TestClient, NOAA/Open-Meteo/LLM mocked
    └── fixtures/                    # saved NOAA JSON samples
```

---

## 7. Day-by-day plan

**Day 1: Data and skeleton**
- Repo skeleton, `pyproject`, ruff, pytest, a minimal CI that passes.
- Download OMNI2 hourly and GFZ Kp. Parse them, handle fill values (9999.9 etc.), align to UTC hourly, and save as parquet.
- EDA notebook: Kp distribution, autocorrelation, 27-day recurrence plot.
- Check every live NOAA endpoint and save sample JSON files as test fixtures.

**Day 2: Features, split, baselines**
- `features.py` with tests that check no future data is used. The time split with an embargo gap.
- Persistence, climatology and recurrence baselines, plus `metrics.py`. Your first results table is baselines only. This is the bar the model must beat.

**Day 3: Model and evaluation** (the core ML day, so protect it) — **done**
- Multiclass LightGBM per horizon, early stopping on val, temperature scaling.
- `evaluate.py`: BSS vs baselines, reliability diagrams, PR-AUC, storm windows, bootstrap CIs, feature importance (does Newell coupling come out on top?).
- 2 h timebox for archived NOAA forecasts. Write `MODEL_CARD.md` while the results are fresh.

**Day 4: Backend** — **done**
- Live ingestion with the L1→bow-shock shift, then the shared features, then the forecast.
- Physics modules (geomag, oval, sky), Open-Meteo client, location score, NOAA 3-day and 27-day outlook.
- `/api/forecast?lat&lon`, `/health`. API tests with mocked HTTP. DB models moved to Day 6, where the scheduler first writes to them.

*Day 4 decisions* (code: `aurora/data/noaa_live.py`, `aurora/live.py`, `aurora/outlook.py`, `app/services.py`):
- **Bow-shock shift:** each 1-min L1 sample moves forward by (x_GSE − 90,000 km) / V, using the active spacecraft's measured distance and a 10-min median speed, then hourly means [t, t+1h) like OMNI (≥15 samples per hour; the still-filling hour is dropped).
- **Issue time can be in the future:** the newest complete bow-shock hour may end after the wall clock (L1 lead ~45–90 min). The row must also have its `kp_last` interval already observed; otherwise an earlier row is used. Rows more than 3 h old are refused.
- **Live Kp:** GFZ nowcast API for 30 days of history (closest to the definitive training target), NOAA estimates for intervals GFZ has not published. GFZ already lists the interval in progress with a provisional value, so unfinished intervals are dropped (found on the fixture: a placeholder 0).
- **Nights:** NOAA's deterministic Kp is turned into class probabilities by `models/noaa_calibration.json` (lead day × forecast class → observed class counts from the archive, smoothed towards the same class pooled over lead days). The range is a Beta interval from the cell's sample size. For hours the model reaches (tonight), the model is used instead. Found live on 2026-10-04 during a Kp 5 storm: NOAA said 6% for tonight in Hamburg, the model 50%.
- **Failure isolation:** each feed failing only blanks its own panel; `errors` in the response names the source.

**Day 5: Frontend and briefing**
- Leaflet map on CARTO Dark Matter tiles, click to choose a location, three panels (Now and hours / 3 nights / Weeks).
- **Dashboard design system** in CSS variables:
  - near-black navy background (`#0b1020`-ish) with slightly lighter panels and thin borders;
  - aurora green/teal accents (`#2ee6a6` / `#14b8a6`-ish) for probabilities and key metrics; amber/red only for storm levels;
  - a clean sans-serif (Inter or IBM Plex Sans) and a monospace for numbers and timestamps, with tabular figures so numbers line up;
  - a dense, aligned grid of cards, small uppercase labels, "last updated" timestamps and source attribution on every panel.
- Show confidence visually:
  - hours: solid accent colour, exact percentages, sparkline charts;
  - nights: lower opacity and ranges instead of exact numbers;
  - weeks: dashed outlines and text only ("possible activity around Oct 28"), with a "low confidence" tag.
- Charts: lightweight library (e.g. Chart.js via CDN) styled with the same theme.
- Use `design-references/` for inspiration only. They are not committed.
- LLM briefing with caching and the template fallback.

**Day 6: Deployment and ops**
- Dockerfile, run locally, deploy to the HF Space, and set up the external Postgres.
- DB models (SQLAlchemy: forecasts, observations, briefings), scheduler, forecast logging and observed-Kp verification job. `/monitoring` page in the same dashboard style: Brier score over time, a forecast-vs-observed plot, and the number of verified forecasts.
- Sentry, UptimeRobot, and CI deploying to HF on push to `main`.

**Day 7: Buffer and polish**
- Fix whatever broke. README with a GIF, an architecture diagram, the results table and a **Limitations** section.
- Write down your interview talking points (sections 1–3 of this plan).

---

## 8. Too much for one week: cut or reduce

- **Neural model:** cut. Mention a GRU/TCN on 5-min OMNI as "next step". It is worth more as a reasoned roadmap item than as a rushed result.
- **Comparison with NOAA's archived forecasts:** timeboxed and likely to be cut (see above). It is apples to oranges anyway.
- **"Live accuracy over time":** after a few days of logging, live accuracy means almost nothing statistically, because there may be no storm at all. Build the pipeline and show "N verified forecasts". Don't claim skill from it. You can add a clearly labelled "historical replay" from the test set.
- **Visual extras:** no animated aurora backgrounds, 3D globes or custom illustrations. Spend design time on a consistent dark theme, clear typography and readable charts.
- **5-min OMNI and Hp30/Hp60 targets:** stretch goals only.
- **Risky but kept:** Day 6 (deployment, external DB, Sentry, UptimeRobot) is the most likely day to overrun. Day 7 is deliberately a buffer for it.

## 9. Verification (once built)
- `pytest` green locally and in CI (leakage, monotonicity, geomag, API with mocks).
- `scripts/evaluate.py` produces BSS > 0 vs persistence at h = 3 and 6 h on the test set. At h = 1 h, persistence is very hard to beat; report it honestly either way.
- `docker run` locally, then `/health` returns 200 and `/api/forecast?lat=53.5&lon=10` returns a sensible output.
- The deployed Space: UptimeRobot is green, a Sentry test error arrives, and `/monitoring` shows new rows after 1 h.
