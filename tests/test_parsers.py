"""Parser tests using real rows from the May 2024 ("Gannon") storm."""

import numpy as np
import pandas as pd
import pytest

from aurora.data.gfz import join_kp_to_hours, parse_gfz_kp
from aurora.data.omni import parse_omni2

OMNI_ROWS = """\
2024 131 18 2601 51 52  59  34  31.1  24.3 -29.9 246.7  -8.3 -19.4 -12.1 -14.1 -18.0  14.2  23.2   8.7  16.7  13.7  871652.  32.9  697.   4.9   2.8 0.042 31.18  408684.  12.7   10.   3.2   3.5 0.020  12.55   1.42   6.4 87 172   -33 1102 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0 300 227.9  13.5  -791   311  4.3
2024 131 19 2601 51 52  63  38  31.1  23.3  18.9 278.0   3.0 -21.8   7.5 -23.0  -1.2   7.4  20.9  10.1  14.1  11.6  963992.  35.2  664.   1.4   2.2 0.026 28.61  202993.   9.9   30.   3.9   3.7 0.006   0.80   1.65   6.3 87 172  -131 1807 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0 300 227.9  10.6 -1794    14  4.1
2024 131 20 9999 99 99 999 999 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 9999999. 999.9 9999. 999.9 999.9 9.999 99.99 9999999. 999.9 9999. 999.9 999.9 9.999 999.99 999.99 999.9 99 999 99999 9999 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0 999 999.9 999.9 99999 99999 99.9
2024 131 21 2601 51 52  58  33  32.7  29.9 -59.9 212.3 -12.7  -8.0 -25.9   3.1 -26.9   3.7  13.1   4.3  10.9   5.9  303344.  48.1  702.   3.8   0.7 0.103 55.89   92480.   6.8   10.   0.9   2.2 0.028  18.88   0.81   7.4 87 172  -157  867 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0 300 227.9  10.9  -453   414  5.7
2024 131 22 9999 99 99 999 999 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 9999999. 999.9 9999. 999.9 999.9 9.999 99.99 9999999. 999.9 9999. 999.9 999.9 9.999 999.99 999.99 999.9 99 999 99999 9999 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0 999 999.9 999.9 99999 99999 99.9
"""  # noqa: E501

GFZ_ROWS = """\
# header line
#YYY MM DD  days  days_m  Bsr dB    Kp1    Kp2    Kp3    Kp4    Kp5    Kp6    Kp7    Kp8  ap1  ap2  ap3  ap4  ap5  ap6  ap7  ap8    Ap  SN F10.7obs F10.7adj D
2024 05 10 33733 33733.5 2601 20  2.667  2.667  2.333  2.000  3.667  7.667  8.667  8.667   12   12    9    7   22  179  300  300   105 172    223.4    227.9 2
2024 05 11 33734 33734.5 2601 21  9.000  8.333  8.333  9.000  8.667  8.333  7.667 -1.000  400  236  236  400  300  236  179   -1   271 173    213.7    218.0 0
"""  # noqa: E501


@pytest.fixture
def omni(tmp_path):
    path = tmp_path / "omni.dat"
    path.write_text(OMNI_ROWS)
    return parse_omni2(path)


@pytest.fixture
def kp(tmp_path):
    path = tmp_path / "kp.txt"
    path.write_text(GFZ_ROWS)
    return parse_gfz_kp(path)


def test_omni_timestamps_from_year_doy_hour(omni):
    # day-of-year 131 in 2024 (leap year) is May 10
    assert omni.index[0] == pd.Timestamp("2024-05-10 18:00", tz="UTC")
    assert (omni.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()


def test_omni_values_and_units(omni):
    row = omni.loc["2024-05-10 18:00"]
    assert row["bz_gsm"] == -18.0
    assert row["by_gsm"] == -14.1
    assert row["speed"] == 697.0
    assert row["density"] == 32.9
    assert row["kp_omni"] == 8.7  # Kp*10 = 87 -> 8.7 (8+)
    assert row["dst"] == -33


def test_omni_fill_values_become_nan_but_inner_gaps_are_kept(omni):
    gap = omni.loc["2024-05-10 20:00"]
    assert gap[["bz_gsm", "speed", "density", "dst"]].isna().all()


def test_omni_drops_trailing_placeholder_rows(omni):
    assert omni.index[-1] == pd.Timestamp("2024-05-10 21:00", tz="UTC")


def test_gfz_expands_to_three_hour_intervals(kp):
    assert len(kp) == 16
    assert kp.index[0] == pd.Timestamp("2024-05-10 00:00", tz="UTC")
    assert kp.index[1] - kp.index[0] == pd.Timedelta("3h")
    # Gannon storm: Kp 9o at 2024-05-11 00-03 UT
    assert kp.loc["2024-05-11 00:00", "kp"] == 9.0


def test_gfz_kp_is_rounded_to_thirds(kp):
    assert kp.loc["2024-05-10 15:00", "kp"] == pytest.approx(23 / 3)  # 8- = 7.667
    assert np.allclose((kp["kp"].dropna() * 3) % 1, 0)


def test_gfz_missing_and_definitive_flag(kp):
    assert np.isnan(kp.loc["2024-05-11 21:00", "kp"])
    assert kp.loc["2024-05-10 00:00", "kp_definitive"]
    assert not kp.loc["2024-05-11 00:00", "kp_definitive"]


def test_join_assigns_containing_three_hour_interval(omni, kp):
    joined = join_kp_to_hours(omni, kp)
    # hours 18-20 UT fall in the 18-21 interval (Kp 8.667), 21 UT in the 21-24 one
    assert joined.loc["2024-05-10 18:00", "kp_bin"] == pytest.approx(26 / 3)
    assert joined.loc["2024-05-10 21:00", "kp_bin"] == pytest.approx(26 / 3)
    assert joined.loc["2024-05-10 19:00", "kp_definitive"]
