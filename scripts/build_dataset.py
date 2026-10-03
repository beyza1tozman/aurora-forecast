"""Parse raw OMNI2 + GFZ Kp and write processed parquet files.

Outputs (data/processed/):
    kp_3h.parquet   one row per 3-hour Kp interval (GFZ, all years)
    hourly.parquet  OMNI2 hourly solar wind from START_YEAR, joined with GFZ Kp

LEAKAGE NOTE: in hourly.parquet, `kp_bin` at hour t is the Kp of the 3-hour
interval *containing* t. It is only known when that interval ends, so it must
never be used directly as a feature at time t. Day 2 features use the Kp of
the last *completed* interval instead.
"""

from aurora.config import PROCESSED_DIR, RAW_DIR, START_YEAR
from aurora.data.gfz import join_kp_to_hours, parse_gfz_kp
from aurora.data.omni import parse_omni2


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    kp_3h = parse_gfz_kp(RAW_DIR / "Kp_ap_Ap_SN_F107_since_1932.txt")
    kp_3h.to_parquet(PROCESSED_DIR / "kp_3h.parquet")

    omni = parse_omni2(RAW_DIR / "omni2_all_years.dat")
    omni = omni.loc[f"{START_YEAR}-01-01" :]
    hourly = join_kp_to_hours(omni, kp_3h)
    hourly.to_parquet(PROCESSED_DIR / "hourly.parquet")

    print(f"kp_3h:  {len(kp_3h):>7} rows  {kp_3h.index[0]} .. {kp_3h.index[-1]}")
    print(f"hourly: {len(hourly):>7} rows  {hourly.index[0]} .. {hourly.index[-1]}")
    print("missing fraction (hourly):")
    print(hourly.isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
