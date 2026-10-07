"""Daily briefing: a few sentences that phrase the forecast for one location.

The LLM (Claude Haiku 4.5) only phrases facts. ``briefing_facts`` turns the
/api/forecast response into pre-formatted strings in the viewer's time zone, the
model is told to copy them, and ``numbers_grounded`` rejects any reply that
contains a number the facts do not. Every failure path (no API key, API error,
daily budget used up, ungrounded reply) falls back to ``template_briefing``, so
the panel never breaks because of the LLM.

LLM replies are cached in SQLite per (0.1 degree cell, time zone, local date,
forecast run) for CACHE_TTL. The cell matches the cloud cache, so the briefing
uses the same numbers as the panels next to it.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from aurora import config

log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 400
CACHE_TTL = 3 * 3600.0  # seconds
CELL = 0.1  # degrees, same as the cloud cache in app/services.py
DEFAULT_DAILY_LIMIT = 300  # LLM calls per UTC day; beyond that the template is used
DEFAULT_DB = config.DATA_DIR / "briefings.sqlite"

SYSTEM_PROMPT = """\
You write the daily aurora briefing for one location in a space-weather dashboard.

You receive a JSON object of facts. Write 3 to 5 short sentences of plain text for
someone deciding whether to go outside tonight.

Rules:
- Use only the facts given. Every number you write must appear in the facts, copied
  exactly as written there (same rounding, same % sign, same clock format). Do not
  compute, convert, round or estimate new numbers.
- If a fact is missing or null, leave that topic out. If "unavailable" lists a
  source, say briefly that this part of the forecast is unavailable.
- Order: the next hours, then tonight and the following nights, then the coming
  weeks if "weeks" has entries.
- Make the confidence clear: the next hours are the most reliable, the nights less
  so (say "between X and Y" when a range is given), the weeks only a rough hint.
- "chance" combines geomagnetic activity, clouds, darkness and the moon. "if_clear"
  is the chance ignoring clouds. "kp_needed" is the Kp index needed to see aurora
  low on the northern horizon at this location.
- Be plain and factual, like a weather service. No hype, no exclamation marks, no
  emoji, no markdown, no greeting, no sign-off.
"""


# --- facts --------------------------------------------------------------------


def pct(p: float | None) -> str | None:
    """Probability as display text: '<1%', '4%', '37%'. Same rule as the frontend."""
    if p is None:
        return None
    if p < 0.01:
        return "<1%"
    return f"{round(p * 100):d}%"


def pct_range(lo: float, hi: float) -> str:
    lo_s, hi_s = pct(lo), pct(hi)
    if lo_s == hi_s:
        return lo_s
    if lo_s == "<1%":
        return f"up to {hi_s}"
    return f"{lo_s.rstrip('%')}–{hi_s}"


def coords_label(lat: float, lon: float) -> str:
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return f"{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}"


def resolve_tz(name: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(name) if name else ZoneInfo("UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _local(ts: str, tz: ZoneInfo) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert(tz)


def _clock(ts: str, tz: ZoneInfo) -> str:
    return _local(ts, tz).strftime("%H:%M")


def _day(ts: str, tz: ZoneInfo) -> str:
    t = _local(ts, tz)
    return f"{t.strftime('%a')} {t.day} {t.strftime('%b')}"


_SOURCE_NAMES = {
    "solar_wind": "real-time solar wind (next hours)",
    "kp": "observed Kp",
    "noaa_3day": "NOAA 3-day forecast (nights)",
    "noaa_27day": "NOAA 27-day outlook (weeks)",
    "clouds": "cloud forecast",
}


def briefing_facts(forecast: dict, tz: ZoneInfo, place: str | None = None) -> dict:
    """Pre-formatted, LLM-ready facts from an /api/forecast response."""
    loc = forecast["location"]
    facts: dict = {
        "location": place or coords_label(loc["lat"], loc["lon"]),
        "today": _day(forecast["generated"], tz),
        "local_time": _clock(forecast["generated"], tz),
        "kp_needed": f"{loc['kp_needed']:.1f}",
    }

    now = forecast.get("now")
    if now and now["hours"]:
        hours = now["hours"]
        has_clouds = all(h["chance"] is not None for h in hours)
        key = "chance" if has_clouds else "chance_if_clear"
        best = max(hours, key=lambda h: h[key])
        dark = [h for h in hours if h["darkness"] > 0]
        facts["next_hours"] = {
            "confidence": "high (1 h) to medium (6 h)",
            "kp_last_observed": f"{now['kp_forecast']['kp_last']:.1f}",
            "until": _clock(str(pd.Timestamp(hours[-1]["time"]) + pd.Timedelta("1h")), tz),
            "best_hour": _clock(best["time"], tz),
            "best_chance": pct(best["chance"]) if has_clouds else None,
            "best_if_clear": pct(best["chance_if_clear"]),
            "p_kp_needed_max": pct(max(h["p_kp"] for h in hours)),
            "dark_from": _clock(dark[0]["time"], tz) if dark else None,
            "cloud_cover_best_hour": pct(best["cloud_cover"])
            if best["cloud_cover"] is not None
            else None,
        }

    nights = (forecast.get("nights") or {}).get("nights") or []
    if nights:
        facts["nights"] = []
        for i, n in enumerate(nights):
            row = {
                "night": "tonight" if i == 0 else f"night starting {_day(n['start'], tz)}",
                "dark": f"{_clock(n['start'], tz)}–{_clock(n['end'], tz)}",
                "best_hour": _clock(n["best_hour"], tz),
                "chance": pct_range(*n["chance_range"]) if n["chance_range"] else None,
                "if_clear": pct(n["chance_if_clear"]),
                "cloud_cover": pct(n["cloud_cover_mean"]),
                "moon_illuminated": pct(n["moon_illumination"]),
                "noaa_kp_max": f"{n['noaa_kp_max']:.1f}" if n["noaa_kp_max"] is not None else None,
            }
            if not n["cloud_forecast_complete"]:
                row["note"] = "cloud forecast does not reach this night yet"
            facts["nights"].append(row)

    weeks = forecast.get("weeks")
    if weeks:
        facts["weeks"] = [
            {
                "date": _day(d["date"] + "T12:00:00+00:00", ZoneInfo("UTC")),
                "noaa_outlook_kp_max": str(d["outlook_kp_max"]),
                "enough_for_this_location": d["reaches_kp_needed"],
            }
            for d in weeks["possible_activity"]
        ]
        facts["weeks_confidence"] = "low"

    unavailable = [_SOURCE_NAMES.get(k, k) for k in forecast.get("errors", {})]
    if unavailable:
        facts["unavailable"] = unavailable
    return facts


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_grounded(text: str, facts: dict) -> bool:
    """True if every number in ``text`` also appears in the facts."""
    allowed = set(_NUMBER.findall(json.dumps(facts, ensure_ascii=False)))
    return set(_NUMBER.findall(text)) <= allowed


# --- template fallback --------------------------------------------------------


def template_briefing(facts: dict) -> str:
    parts = []
    h = facts.get("next_hours")
    if h:
        if h["best_chance"] is not None:
            parts.append(
                f"Until {h['until']}, the best chance is {h['best_chance']} at "
                f"{h['best_hour']} (Kp {facts['kp_needed']} needed here, "
                f"last observed Kp {h['kp_last_observed']})."
            )
        else:
            parts.append(
                f"Until {h['until']}, the chance under a clear sky peaks at "
                f"{h['best_if_clear']} at {h['best_hour']}; no cloud forecast is available."
            )
    nights = facts.get("nights") or []
    for n in nights[:1]:
        if n["chance"] is not None:
            parts.append(
                f"Tonight ({n['dark']}): chance {n['chance']}, best around {n['best_hour']}, "
                f"with {n['cloud_cover']} cloud cover on average."
            )
        else:
            parts.append(f"Tonight: up to {n['if_clear']} if the sky is clear.")
    later = []
    for n in nights[1:]:
        name = n["night"].removeprefix("night starting ")
        if n["chance"] is not None:
            later.append(f"{name}: {n['chance']} ({n['if_clear']} if clear)")
        elif n["if_clear"]:
            later.append(f"{name}: {n['if_clear']} if clear, no cloud forecast yet")
    if later:
        parts.append(f"Following nights (lower confidence): {'; '.join(later)}.")
    weeks = facts.get("weeks")
    if weeks:
        strong = [w for w in weeks if w["enough_for_this_location"]]
        pick = strong or weeks
        dates = ", ".join(w["date"] for w in pick[:3])
        parts.append(f"Coming weeks (low confidence): possible activity around {dates}.")
    if facts.get("unavailable"):
        parts.append("Unavailable right now: " + ", ".join(facts["unavailable"]) + ".")
    return " ".join(parts) or "No forecast data is available right now."


# --- cache --------------------------------------------------------------------


class BriefingCache:
    """SQLite store of LLM briefings; also counts today's LLM calls for the budget."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS briefings ("
                "key TEXT PRIMARY KEY, created REAL NOT NULL, text TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def get(self, key: str, ttl: float) -> str | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT created, text FROM briefings WHERE key = ?", (key,)).fetchone()
        if row and time.time() - row[0] < ttl:
            return row[1]
        return None

    def put(self, key: str, text: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO briefings (key, created, text) VALUES (?, ?, ?)",
                (key, time.time(), text),
            )

    def calls_today(self) -> int:
        midnight = pd.Timestamp.now(tz="UTC").floor("D").timestamp()
        with self._lock, self._connect() as db:
            return db.execute(
                "SELECT COUNT(*) FROM briefings WHERE created >= ?", (midnight,)
            ).fetchone()[0]


# --- briefer ------------------------------------------------------------------


def _anthropic_client():
    """Client if credentials are configured, else None (template only)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    return anthropic.Anthropic(timeout=20.0, max_retries=1)


@dataclass
class Briefing:
    text: str
    source: str  # "llm", "llm-cached" or "template"
    model: str | None = None

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "model": self.model}


class Briefer:
    def __init__(
        self,
        client=None,
        cache: BriefingCache | None = None,
        daily_limit: int | None = None,
        client_factory: Callable = _anthropic_client,
    ):
        self.client = client if client is not None else client_factory()
        self.cache = cache or BriefingCache(Path(os.environ.get("BRIEFING_DB", DEFAULT_DB)))
        self.daily_limit = (
            daily_limit
            if daily_limit is not None
            else int(os.environ.get("BRIEFING_DAILY_LIMIT", DEFAULT_DAILY_LIMIT))
        )

    @staticmethod
    def cache_key(forecast: dict, tz: ZoneInfo, place: str | None) -> str:
        loc = forecast["location"]
        now = forecast.get("now") or {}
        run = (now.get("kp_forecast") or {}).get("issued") or forecast["generated"][:13]
        local_date = _local(forecast["generated"], tz).date().isoformat()
        cell = f"{round(loc['lat'] / CELL)}:{round(loc['lon'] / CELL)}"
        return "|".join([cell, str(tz), local_date, run, place or ""])

    def brief(self, forecast: dict, tz: ZoneInfo, place: str | None = None) -> Briefing:
        facts = briefing_facts(forecast, tz, place)
        if self.client is None:
            return Briefing(template_briefing(facts), "template")

        key = self.cache_key(forecast, tz, place)
        cached = self.cache.get(key, CACHE_TTL)
        if cached is not None:
            return Briefing(cached, "llm-cached", MODEL)
        if self.cache.calls_today() >= self.daily_limit:
            log.warning("briefing budget of %d calls used up, using template", self.daily_limit)
            return Briefing(template_briefing(facts), "template")

        try:
            text = self._ask(facts)
        except Exception as err:  # noqa: BLE001 - any API failure falls back to the template
            log.warning("briefing LLM call failed: %r", err)
            return Briefing(template_briefing(facts), "template")
        if not numbers_grounded(text, facts):
            log.warning("briefing rejected, numbers not in facts: %r", text)
            return Briefing(template_briefing(facts), "template")
        self.cache.put(key, text)
        return Briefing(text, "llm", MODEL)

    def _ask(self, facts: dict) -> str:
        # No cache_control: the system prompt is far below Haiku 4.5's minimum cacheable
        # prefix, so prompt caching would silently do nothing. The SQLite cache saves calls.
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(facts, ensure_ascii=False, indent=1)}],
        )
        if response.stop_reason != "end_turn":
            raise RuntimeError(f"stop_reason={response.stop_reason}")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise RuntimeError("empty reply")
        return text
