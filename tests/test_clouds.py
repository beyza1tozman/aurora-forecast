"""Cloud providers: MET Norway parsing, and the stale-cache fallback."""

import json

import pandas as pd
import pytest

from app.services import TTLCache
from aurora.data.met_norway import parse_clouds_met
from aurora.data.open_meteo import LAYERS
from tests.conftest import FIXTURES


@pytest.fixture(scope="module")
def met():
    return json.loads((FIXTURES / "met_norway_munich.json").read_text(encoding="utf-8"))


def test_met_norway_parses_to_hourly_open_meteo_shape(met):
    df = parse_clouds_met(met)
    assert list(df.columns) == LAYERS
    assert str(df.index.tz) == "UTC"
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert df.notna().all().all()
    assert ((df >= 0) & (df <= 1)).all().all()


def test_met_norway_six_hourly_steps_are_interpolated(met):
    steps = met["properties"]["timeseries"]
    df = parse_clouds_met(met)
    # Find two neighbouring 6-hourly steps and check the hour between them.
    times = [pd.Timestamp(s["time"]) for s in steps]
    i = next(i for i in range(1, len(times)) if times[i] - times[i - 1] == pd.Timedelta("6h"))
    a, b = (
        s["data"]["instant"]["details"]["cloud_area_fraction"] / 100 for s in steps[i - 1 : i + 1]
    )
    mid = df.loc[times[i - 1] + pd.Timedelta("3h"), "cloud_cover"]
    assert mid == pytest.approx((a + b) / 2)


def test_ttl_cache_serves_stale_value_when_refresh_fails(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("app.services.time.monotonic", lambda: now[0])
    cache = TTLCache(ttl=10, max_stale=100)
    assert cache.get("k", lambda: "fresh") == "fresh"

    def fail():
        raise RuntimeError("provider down")

    now[0] = 50  # expired, but within max_stale
    assert cache.get("k", fail) == "fresh"
    now[0] = 200  # too old to reuse
    with pytest.raises(RuntimeError):
        cache.get("k", fail)
