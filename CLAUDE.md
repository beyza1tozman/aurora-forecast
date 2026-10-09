# Aurora Forecast

Web app that shows the chance of seeing the aurora at a chosen location in Germany/Europe:
now and the next few hours, an outlook for the next 3 nights, and a low-confidence hint for
the coming weeks (27-day solar rotation). Confidence must visibly decrease with horizon.

An ML and space-weather project, built in one week on a Windows laptop, CPU only.

**The full plan is in `docs/PLAN.md` (local only, gitignored).** It contains the day-by-day schedule, the folder
structure and the reasoning behind the ML and physics decisions. Read it before starting work and keep it up to date
when decisions change.

## Core pieces
- **ML:** LightGBM forecasting Kp 1/3/6 h ahead from L1 solar wind. Trained on NASA OMNI2 hourly
  data with GFZ Potsdam definitive Kp; NOAA SWPC live JSON feeds in production.
  Time-based split with embargo gaps, persistence/climatology baselines, calibrated
  P(Kp ≥ 5/6/7), separate evaluation on storm periods.
- **Location score:** forecast Kp × geomagnetic latitude × cloud cover (Open-Meteo, MET Norway fallback) ×
  darkness × moon.
- **Briefing:** LLM-generated daily text (Claude Haiku 4.5), cached, with a template fallback.
- **Production:** FastAPI, Leaflet frontend, Docker, Render (free web service), pytest, GitHub
  Actions CI, Sentry, `/health` for UptimeRobot, `/monitoring` comparing logged forecasts to observed Kp.

## Rules that matter
- `aurora/features.py` is the single source of truth for features, shared by training and
  serving. Live L1 data must be time-shifted to match OMNI (bow-shock shifted).
- Never use a random train/test split; features at time t may only use data ≤ t.
- Exceedance probabilities must be monotonic: P(Kp≥7) ≤ P(Kp≥6) ≤ P(Kp≥5).
- The monitoring log lives in an external DB (`DATABASE_URL`), because the Render free disk is ephemeral.
- Secrets go in `.env` (gitignored); keep `.env.example` in sync.

## Design direction
Clear and professional, like a modern space-data dashboard. **Not** cute or cozy.
- Palette picked from a photo of the aurora over snow: periwinkle `#546A9C`, slate blue `#6878A6`,
  lavender `#B0A3C4`, pale mint `#D5E0DA`, snow `#E8E8E8`. The page background is the sky
  (dark periwinkle fading to lavender); panels are dark translucent glass with thin borders.
- Pale mint accent (aurora curtain) for probabilities and key metrics, lavender as the secondary
  accent; amber/red only for storm levels. All colours are tokens in `app/static/css/app.css`.
- Clean sans-serif typography (Inter / IBM Plex Sans), monospace with tabular figures for numbers.
- Dark map tiles (Esri Dark Gray Canvas, tinted slate blue; CARTO now needs a key) with Leaflet.
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
