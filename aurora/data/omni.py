"""Parser for NASA OMNI2 hourly data (omni2_all_years.dat).

Format: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
55 whitespace-separated fields per hour; missing values are all-9 fill values.
Since 1995, OMNI hourly values are averages of high-resolution data that has
already been time-shifted from the spacecraft (L1) to the bow-shock nose.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# (column index in the file, our name, fill value). Indices are 0-based word numbers - 1.
FIELDS: list[tuple[int, str, float]] = [
    (4, "imf_sc_id", 99),
    (5, "plasma_sc_id", 99),
    (8, "b_mag", 999.9),  # field magnitude average |B|, nT
    (12, "bx_gse", 999.9),  # Bx is identical in GSE and GSM
    (15, "by_gsm", 999.9),
    (16, "bz_gsm", 999.9),
    (22, "temperature", 9999999.0),  # proton temperature, K
    (23, "density", 999.9),  # proton density, n/cm^3
    (24, "speed", 9999.0),  # bulk flow speed, km/s
    (28, "pressure", 99.99),  # flow pressure, nPa
    (35, "e_field", 999.99),  # -V*Bz, mV/m
    (36, "beta", 999.99),
    (37, "alfven_mach", 999.9),
    (38, "kp_omni", 99),  # Kp*10 as copied into OMNI; we use GFZ Kp instead
    (40, "dst", 99999),  # nT, Kyoto
    (41, "ae", 9999),  # nT, Kyoto
]

N_COLUMNS = 55

# Measured solar wind quantities; a row with none of these is a pure gap.
MEASURED = ["b_mag", "bx_gse", "by_gsm", "bz_gsm", "density", "speed", "temperature"]


def parse_omni2(path: Path) -> pd.DataFrame:
    """Read an OMNI2 hourly file into a DataFrame indexed by UTC hour.

    Fill values become NaN. Placeholder rows after the last real measurement
    (OMNI pads the current year to Dec 31) are dropped; gaps inside the
    record are kept as NaN rows so the index stays regular.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    if raw.shape[1] != N_COLUMNS:
        raise ValueError(f"expected {N_COLUMNS} columns, got {raw.shape[1]}")

    time = (
        pd.to_datetime(raw[0].astype(int).astype(str), format="%Y", utc=True)
        + pd.to_timedelta(raw[1] - 1, unit="D")
        + pd.to_timedelta(raw[2], unit="h")
    )

    df = pd.DataFrame(index=pd.DatetimeIndex(time, name="time"))
    for col, name, fill in FIELDS:
        values = raw[col].to_numpy()
        df[name] = np.where(values == fill, np.nan, values)

    df["kp_omni"] = df["kp_omni"] / 10.0

    has_data = df[MEASURED].notna().any(axis=1)
    if not has_data.any():
        return df.iloc[0:0]
    return df.loc[: has_data[has_data].index[-1]]
