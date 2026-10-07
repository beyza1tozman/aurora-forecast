"""Briefing: facts, number guard, template fallback, cache and budget. The LLM is faked."""

from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest

from app.services import ForecastService
from aurora.briefing import (
    Briefer,
    BriefingCache,
    briefing_facts,
    numbers_grounded,
    pct,
    pct_range,
    resolve_tz,
    template_briefing,
)
from aurora.model import load_models
from aurora.outlook import NoaaCalibration
from tests.conftest import FIXTURE_NOW, fixture_transport

BERLIN = ZoneInfo("Europe/Berlin")
HAMBURG = (53.55, 9.99)


@pytest.fixture(scope="module")
def forecast():
    service = ForecastService(
        client=httpx.Client(transport=fixture_transport()),
        models=load_models(),
        noaa_calibration=NoaaCalibration.load(),
        clock=lambda: FIXTURE_NOW,
    )
    return service.forecast(*HAMBURG)


class FakeLLM:
    """Stands in for anthropic.Anthropic: records calls, returns a fixed reply."""

    def __init__(self, reply="", fail=False, stop_reason="end_turn"):
        self.reply, self.fail, self.stop_reason = reply, fail, stop_reason
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("API down")
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=self.reply)],
        )


def make_briefer(tmp_path, llm, daily_limit=100):
    return Briefer(client=llm, cache=BriefingCache(tmp_path / "b.sqlite"), daily_limit=daily_limit)


def test_pct_formatting():
    assert pct(0.004) == "<1%"
    assert pct(0.126) == "13%"
    assert pct_range(0.02, 0.03) == "2–3%"
    assert pct_range(0.001, 0.03) == "up to 3%"
    assert pct_range(0.02, 0.024) == "2%"


def test_resolve_tz_falls_back_to_utc():
    assert str(resolve_tz("Europe/Berlin")) == "Europe/Berlin"
    assert str(resolve_tz("Not/AZone")) == "UTC"
    assert str(resolve_tz("../../etc")) == "UTC"


def test_facts_are_local_and_preformatted(forecast):
    facts = briefing_facts(forecast, BERLIN, "Hamburg")
    assert facts["location"] == "Hamburg"
    assert facts["kp_needed"] == "5.5"
    # 16:10 UTC is 18:10 in Berlin (CEST).
    assert facts["local_time"] == "18:10"
    assert len(facts["nights"]) == 3
    tonight = facts["nights"][0]
    assert tonight["night"] == "tonight"
    assert tonight["chance"].endswith("%")
    assert "note" in facts["nights"][2]  # cloud forecast does not reach night 3
    assert facts["weeks_confidence"] == "low"


def test_number_guard(forecast):
    facts = briefing_facts(forecast, BERLIN)
    tonight = facts["nights"][0]
    ok = f"Tonight the chance is {tonight['chance']}, best around {tonight['best_hour']}."
    assert numbers_grounded(ok, facts)
    assert not numbers_grounded("Tonight the chance is 87%.", facts)


def test_template_mentions_every_horizon(forecast):
    facts = briefing_facts(forecast, BERLIN)
    text = template_briefing(facts)
    assert "Tonight" in text and "Following nights" in text and "Coming weeks" in text
    assert numbers_grounded(text, facts)


def test_template_reports_outages(forecast):
    broken = {**forecast, "now": None, "errors": {"solar_wind": "boom"}}
    text = template_briefing(briefing_facts(broken, BERLIN))
    assert "Unavailable right now: real-time solar wind" in text


def test_no_client_uses_template(tmp_path, forecast):
    b = Briefer(
        client=None, cache=BriefingCache(tmp_path / "b.sqlite"), client_factory=lambda: None
    )
    assert b.brief(forecast, BERLIN).source == "template"


def test_llm_reply_is_cached(tmp_path, forecast):
    facts = briefing_facts(forecast, BERLIN)
    reply = f"Tonight looks quiet, around {facts['nights'][0]['chance']}."
    llm = FakeLLM(reply)
    briefer = make_briefer(tmp_path, llm)

    first = briefer.brief(forecast, BERLIN)
    second = briefer.brief(forecast, BERLIN)
    assert (first.source, first.text) == ("llm", reply)
    assert second.source == "llm-cached" and second.text == reply
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert "Every number you write must appear in the facts" in call["system"]

    # A different time zone is a different briefing (local times change).
    briefer.brief(forecast, ZoneInfo("Europe/London"))
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    "llm",
    [
        FakeLLM("Kp will reach 8 tonight with a 64% chance."),  # invented numbers
        FakeLLM(fail=True),
        FakeLLM("Tonight", stop_reason="max_tokens"),
        FakeLLM("   "),
    ],
    ids=["ungrounded", "api-error", "truncated", "empty"],
)
def test_bad_llm_reply_falls_back_to_template(tmp_path, forecast, llm):
    briefer = make_briefer(tmp_path, llm)
    result = briefer.brief(forecast, BERLIN)
    assert result.source == "template"
    assert briefer.cache.calls_today() == 0  # failures are not cached


def test_daily_budget(tmp_path, forecast):
    facts = briefing_facts(forecast, BERLIN)
    llm = FakeLLM(f"Best hour {facts['nights'][0]['best_hour']}.")
    briefer = make_briefer(tmp_path, llm, daily_limit=1)
    assert briefer.brief(forecast, BERLIN).source == "llm"
    assert briefer.brief(forecast, ZoneInfo("UTC")).source == "template"
    assert len(llm.calls) == 1
