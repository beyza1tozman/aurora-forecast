"""Parser for the NOAA SWPC "3-Day Forecast" text product.

The product is issued at 00:30 and 12:30 UT. Its "NOAA Kp index breakdown" table
gives a deterministic Kp (in thirds, e.g. 4.67 = 5-) for every 3-hour UT interval
of the issue day and the two following days. Storm levels are annotated, e.g.
"4.67 (G1)". The same parser reads the live product and the NCEI archive
(March 2022 onwards). Archive files before mid-2023 give whole-number Kp ("3"),
and their header lines lack the leading ':'.
"""

import re

import pandas as pd

ISSUED_RE = re.compile(r"Issued:\s*(\d{4} \w{3} \d{2} \d{4}) UTC")
ROW_RE = re.compile(r"^(\d{2})-\d{2}UT\s+(.*)$")
VALUE_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")  # not the 1 in "(G1)"


def parse_3day_forecast(text: str) -> pd.DataFrame:
    """Return one row per forecast Kp interval.

    Index: interval start (UTC). Columns: issued (UTC), kp, lead_h (interval
    start minus issue time, in hours; negative for intervals that had already
    started when the forecast was issued).
    """
    match = ISSUED_RE.search(text)
    if match is None:
        raise ValueError("no 'Issued:' line in 3-day forecast")
    issued = pd.to_datetime(match.group(1), format="%Y %b %d %H%M", utc=True)
    day0 = issued.floor("D")

    in_table = False
    starts, values = [], []
    for line in text.splitlines():
        if line.startswith("NOAA Kp index breakdown"):
            in_table = True
            continue
        if not in_table:
            continue
        line = line.strip()
        if not line:
            continue
        row = ROW_RE.match(line)
        if row is None:
            if starts:
                break  # first line after the table
            continue  # the column header (dates)
        kps = [float(v) for v in VALUE_RE.findall(row.group(2))]
        if len(kps) != 3:
            raise ValueError(f"expected 3 Kp values, got {line!r}")
        hour = pd.Timedelta(hours=int(row.group(1)))
        for day, kp in enumerate(kps):
            starts.append(day0 + pd.Timedelta(days=day) + hour)
            values.append(kp)

    if len(starts) != 24:
        raise ValueError(f"expected 24 Kp intervals, got {len(starts)}")
    kp = [round(v * 3) / 3 for v in values]  # 4.67 -> 4.666..., same as GFZ
    out = pd.DataFrame({"kp": kp}, index=pd.DatetimeIndex(starts, name="time")).sort_index()
    out.insert(0, "issued", issued)
    out["lead_h"] = (out.index - issued) / pd.Timedelta("1h")
    return out
