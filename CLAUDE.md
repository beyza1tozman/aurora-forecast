# Aurora Forecast

Web app that shows the chance of seeing the aurora at a chosen location in Germany/Europe:
now and the next few hours, an outlook for the next 3 nights, and a low-confidence hint for
the coming weeks (27-day solar rotation). Confidence must visibly decrease with horizon.

Portfolio project for ML / space-industry Werkstudent applications, built in one week on a
Windows laptop, CPU only.

**The full plan is in [docs/PLAN.md](docs/PLAN.md).** It contains the day-by-day schedule, the folder
structure and the reasoning behind the ML and physics decisions. Read it before starting work and keep it up to date
when decisions change.

## Core pieces
- **ML:** LightGBM forecasting Kp 1/3/6 h ahead from L1 solar wind. Trained on NASA OMNI2 hourly
  data with GFZ Potsdam definitive Kp; NOAA SWPC live JSON feeds in production.
  Time-based split with embargo gaps, persistence/climatology baselines, calibrated
  P(Kp ≥ 5/6/7), separate evaluation on storm periods.
- **Location score:** forecast Kp × geomagnetic latitude × cloud cover (Open-Meteo) ×
  darkness × moon.
- **Briefing:** LLM-generated daily text (Claude Haiku 4.5), cached, with a template fallback.
- **Production:** FastAPI, Leaflet frontend, Docker, Hugging Face Spaces, pytest, GitHub
  Actions CI, Sentry, `/health` for UptimeRobot, `/monitoring` comparing logged forecasts to observed Kp.

## Rules that matter
- `aurora/features.py` is the single source of truth for features, shared by training and
  serving. Live L1 data must be time-shifted to match OMNI (bow-shock shifted).
- Never use a random train/test split; features at time t may only use data ≤ t.
- Exceedance probabilities must be monotonic: P(Kp≥7) ≤ P(Kp≥6) ≤ P(Kp≥5).
- The monitoring log lives in an external DB (`DATABASE_URL`), because HF Spaces storage is ephemeral.
- Secrets go in `.env` (gitignored); keep `.env.example` in sync.

## Design direction
Clear and professional, like a modern space-data dashboard. **Not** cute or cozy.
- Dark theme (near-black navy background, slightly lighter panels, thin borders).
- Aurora green/teal accents for probabilities and key metrics; amber/red only for storm levels.
- Clean sans-serif typography (Inter / IBM Plex Sans), monospace with tabular figures for numbers.
- Dark map tiles (Esri Dark Gray Canvas, tinted navy; CARTO now needs a key) with Leaflet.
- Confidence shown visually: solid and exact for hours, softer with ranges for nights, dashed
  and text-only for weeks.
- `design-references/` is local inspiration only and is gitignored.

## Git workflow
- Do not commit after every small step. Commit only when a meaningful milestone is done and
  working (e.g. the data download pipeline, the feature pipeline, a trained baseline), with
  tests and lint passing.
- Always ask the user before committing.

## Environment
- Windows 11, Python, CPU only. Prefer pure-Python dependencies that install cleanly on Windows.
- Default branch: `main`.
