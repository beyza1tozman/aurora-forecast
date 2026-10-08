---
title: Aurora Forecast
emoji: 🌌
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Chance of seeing the aurora in Europe, now to weeks ahead
---

# Aurora Forecast

Chance of seeing the aurora at a chosen place in Germany and Europe: the next hours, the
next 3 nights, and a low-confidence hint for the coming weeks. Confidence visibly falls
with the horizon.

- **Next hours:** LightGBM forecasting Kp 1/3/6 h ahead from real-time L1 solar wind,
  trained on NASA OMNI2 with GFZ Potsdam Kp, calibrated, and scored against persistence
  and climatology. See [MODEL_CARD.md](MODEL_CARD.md).
- **Nights:** NOAA SWPC 3-day Kp forecast, calibrated on its own archive.
- **Weeks:** NOAA 27-day outlook and the 27-day solar rotation.
- **Location:** geomagnetic latitude, clouds (Open-Meteo), darkness and the moon.
- **/monitoring:** every live forecast is logged and scored against the Kp observed later.

The full plan and the reasoning behind each decision are in [docs/PLAN.md](docs/PLAN.md).

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000. Settings are in `.env` (see `.env.example`); all are optional.
