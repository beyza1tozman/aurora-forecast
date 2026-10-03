"""Paths and data-source URLs shared by scripts, training and serving."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# NASA SPDF OMNI2: hourly solar wind at the bow-shock nose, 1963-present.
OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"

# GFZ Potsdam definitive Kp/ap, 1932-present.
GFZ_KP_URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"

# First year used for modelling (ACE era, continuous L1 coverage).
START_YEAR = 1998
