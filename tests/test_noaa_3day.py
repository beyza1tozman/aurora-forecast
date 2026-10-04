"""Parser tests for the NOAA 3-day forecast (live fixture + an archived year-end issue)."""

from pathlib import Path

import pandas as pd
import pytest

from aurora.data.noaa_3day import parse_3day_forecast

FIXTURES = Path(__file__).parent / "fixtures"

# Archived issue from the NCEI 3-day forecast archive: older header format,
# spans a year boundary.
ARCHIVE_YEAR_END = """\
Product: 3-Day Forecast
Issued: 2022 Dec 31 0030 UTC
Prepared by the U.S. Dept. of Commerce, NOAA, Space Weather Prediction
Center

A. NOAA Geomagnetic Activity Observation and Forecast

NOAA Kp index breakdown Dec 31-Jan 02 2023

             Dec 31       Jan 01       Jan 02
00-03UT       4.67 (G1)    2.67         2.33
03-06UT       3.67         2.67         2.67
06-09UT       3.67         2.00         2.33
09-12UT       2.67         2.33         2.33
12-15UT       2.33         2.67         2.00
15-18UT       2.00         2.67         2.00
18-21UT       2.67         2.67         2.00
21-00UT       3.00         2.33         2.33

Rationale: G1 (Minor) storm levels are likely early on 31 Dec.
"""


def test_live_fixture():
    df = parse_3day_forecast((FIXTURES / "noaa_3day_forecast.txt").read_text(encoding="utf-8"))
    assert len(df) == 24
    assert df.index.is_monotonic_increasing
    assert (df["issued"] == pd.Timestamp("2026-10-03 12:30", tz="UTC")).all()
    assert df.index[0] == pd.Timestamp("2026-10-03 00:00", tz="UTC")
    assert df["lead_h"].iloc[0] == -12.5
    assert df.loc["2026-10-04 03:00", "kp"].item() == pytest.approx(14 / 3)  # "4.67 (G1)"


def test_year_boundary_and_old_header():
    df = parse_3day_forecast(ARCHIVE_YEAR_END)
    assert df["issued"].iloc[0] == pd.Timestamp("2022-12-31 00:30", tz="UTC")
    assert df.index[0] == pd.Timestamp("2022-12-31 00:00", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2023-01-02 21:00", tz="UTC")
    assert df.loc["2022-12-31 00:00", "kp"].item() == pytest.approx(14 / 3)
    assert df.loc["2023-01-01 21:00", "kp"].item() == pytest.approx(7 / 3)
    assert df["lead_h"].iloc[-1] == 68.5


def test_rejects_text_without_table():
    with pytest.raises(ValueError):
        parse_3day_forecast("Issued: 2022 Dec 31 0030 UTC\nno table here\n")


def test_whole_number_kp_in_early_archive():
    text = ARCHIVE_YEAR_END.replace("4.67 (G1)", "5 (G1)   ").replace("2.33", "2   ")
    df = parse_3day_forecast(text)
    assert len(df) == 24
    assert df.loc["2022-12-31 00:00", "kp"].item() == 5
    assert df.loc["2022-12-31 12:00", "kp"].item() == 2


def test_doubled_carriage_returns():
    # A CRLF file re-saved as text on Windows: every table row is followed by an empty line.
    df = parse_3day_forecast(ARCHIVE_YEAR_END.replace("\n", "\r\r\n"))
    assert len(df) == 24
