"""Background jobs (APScheduler, in-process, single worker):

- every 15 min: log the current Kp forecast (one row per horizon; repeats of the
  same issue time are skipped, so the log has one entry per hourly issue)
- every hour: store observed Kp for the last ~30 days of completed intervals (GFZ
  nowcast, NOAA where GFZ is late), overwriting earlier values
"""

import logging
from dataclasses import dataclass, field

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.engine import Engine

from app import db
from app.services import ForecastService, utc_now

log = logging.getLogger(__name__)

FORECAST_EVERY_MIN = 15
VERIFY_EVERY_MIN = 60


@dataclass
class Jobs:
    service: ForecastService
    engine: Engine
    last_run: dict[str, dict] = field(default_factory=dict)

    def _record(self, name: str, ok: bool, detail: str) -> None:
        self.last_run[name] = {"time": utc_now().isoformat(), "ok": ok, "detail": detail}

    def log_forecast(self) -> int:
        now = self.service.clock()
        g = self.service.global_inputs()
        if g.kp_forecast is None:
            self._record("log_forecast", False, f"no forecast: {g.errors}")
            return 0
        try:
            added = db.log_forecast(self.engine, g.kp_forecast, now)
        except Exception as err:  # noqa: BLE001 - a DB outage must not stop the scheduler
            log.exception("logging forecast failed")
            self._record("log_forecast", False, repr(err))
            return 0
        self._record("log_forecast", True, f"{added} rows, issued {g.kp_forecast.issued}")
        return added

    def verify(self) -> int:
        now = self.service.clock()
        g = self.service.global_inputs()
        if g.kp_history is None:
            self._record("verify", False, f"no observed Kp: {g.errors}")
            return 0
        try:
            # The whole fetched history (~30 days, ~240 rows): small, and it fills the
            # observed-Kp chart right after a fresh deploy.
            written = db.upsert_observations(self.engine, g.kp_history, now)
        except Exception as err:  # noqa: BLE001
            log.exception("storing observations failed")
            self._record("verify", False, repr(err))
            return 0
        self._record("verify", True, f"{written} intervals")
        return written


def start(jobs: Jobs) -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone="UTC", job_defaults={"coalesce": True, "max_instances": 1})
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    sched.add_job(jobs.log_forecast, "interval", minutes=FORECAST_EVERY_MIN, next_run_time=now)
    sched.add_job(
        jobs.verify,
        "interval",
        minutes=VERIFY_EVERY_MIN,
        next_run_time=now + pd.Timedelta("1min").to_pytimedelta(),
    )
    sched.start()
    log.info("scheduler started")
    return sched
