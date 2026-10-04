"""Paths and data-source URLs shared by scripts, training and serving."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# NASA SPDF OMNI2: hourly solar wind at the bow-shock nose, 1963-present.
OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"

# GFZ Potsdam definitive Kp/ap, 1932-present.
GFZ_KP_URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"

# First year used for modelling (ACE era, continuous L1 coverage).
START_YEAR = 1998

# --- Live feeds (verified 2026-10-03) ---
# NOAA SWPC real-time solar wind: last ~24 h at 1-min cadence, measured at L1 (not shifted).
# Each file mixes several spacecraft (e.g. SOLAR1 = SWFO-L1, ACE, IMAP); rows with
# active == true belong to the spacecraft NOAA currently treats as operational.
SWPC = "https://services.swpc.noaa.gov"
NOAA_RTSW_MAG_URL = f"{SWPC}/json/rtsw/rtsw_mag_1m.json"
NOAA_RTSW_WIND_URL = f"{SWPC}/json/rtsw/rtsw_wind_1m.json"
# Hourly spacecraft positions (x_gse in km) -> exact L1-to-Earth propagation delay.
NOAA_RTSW_EPHEMERIS_URL = f"{SWPC}/json/rtsw/rtsw_ephemerides_1h.json"
# Estimated planetary Kp (past ~7 days) and observed + forecast Kp (3-day outlook).
NOAA_KP_URL = f"{SWPC}/products/noaa-planetary-k-index.json"
NOAA_KP_FORECAST_URL = f"{SWPC}/products/noaa-planetary-k-index-forecast.json"
NOAA_3DAY_FORECAST_URL = f"{SWPC}/text/3-day-forecast.txt"
NOAA_27DAY_OUTLOOK_URL = f"{SWPC}/text/27-day-outlook.txt"

# NCEI archive of the SWPC 3-day forecast text product (March 2022 onwards),
# laid out as {YYYY}/{MM}/{YYYYMMDD}{0030|1230}three_day_forecast.txt.
NOAA_3DAY_ARCHIVE_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/3day_forecast"
)

# GFZ nowcast Kp API, used to verify logged forecasts on /monitoring.
GFZ_KP_API_URL = "https://kp.gfz.de/app/json/"

# Open-Meteo hourly weather forecast (cloud cover).
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
