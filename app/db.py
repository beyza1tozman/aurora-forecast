"""Forecast log and observed Kp, for /monitoring.

The host's disk is wiped on every deploy and restart, so production points DATABASE_URL at an
external Postgres (free tier). Locally and in tests it defaults to SQLite.

Tables
------
forecasts     one row per (issue time, horizon): class probabilities + kp_last,
              so the model and persistence can be verified on the same rows.
observations  one row per 3-h Kp interval. Re-fetched every hour and overwritten,
              because GFZ nowcast values can still change after first publication.
"""

import os

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

from aurora import config
from aurora.dataset import CLASS_LABELS
from aurora.live import KpForecast

metadata = MetaData()
PROB_COLUMNS = [f"p_{i}" for i in range(len(CLASS_LABELS))]  # p_0 = P(<=3) ... p_4 = P(7+)

forecasts = Table(
    "forecasts",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("issued", DateTime(timezone=True), nullable=False),
    Column("horizon_h", Integer, nullable=False),
    Column("interval_start", DateTime(timezone=True), nullable=False, index=True),
    Column("kp_last", Float),
    *(Column(c, Float, nullable=False) for c in PROB_COLUMNS),
    Column("model_version", String(32)),
    Column("logged_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("issued", "horizon_h", name="uq_forecast_issue_horizon"),
)

observations = Table(
    "observations",
    metadata,
    Column("interval_start", DateTime(timezone=True), primary_key=True),
    Column("kp", Float, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

DEFAULT_URL = f"sqlite:///{(config.DATA_DIR / 'aurora.sqlite').as_posix()}"


def normalize_url(url: str) -> str:
    """Neon/Supabase hand out postgres:// or postgresql:// URLs; use the psycopg 3 driver."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def make_engine(url: str | None = None) -> Engine:
    url = normalize_url(url or os.environ.get("DATABASE_URL") or DEFAULT_URL)
    if url.startswith("sqlite:///"):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, pool_pre_ping=True)
    metadata.create_all(engine)
    return engine


def _utc(ts: pd.Timestamp):
    return pd.Timestamp(ts).tz_convert("UTC").to_pydatetime()


def _ts(value) -> pd.Timestamp:
    """SQLite returns naive datetimes; everything stored is UTC."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def log_forecast(engine: Engine, fc: KpForecast, now: pd.Timestamp, version: str = "") -> int:
    """Insert one row per horizon; an issue time already logged is skipped. Returns rows added."""
    issued = _utc(fc.issued)
    with engine.begin() as conn:
        known = set(
            conn.execute(
                select(forecasts.c.horizon_h).where(forecasts.c.issued == issued)
            ).scalars()
        )
        rows = [
            {
                "issued": issued,
                "horizon_h": h.horizon_h,
                "interval_start": _utc(h.interval_start),
                "kp_last": fc.kp_last,
                **{c: float(p) for c, p in zip(PROB_COLUMNS, h.class_probs, strict=True)},
                "model_version": version,
                "logged_at": _utc(now),
            }
            for h in fc.horizons
            if h.horizon_h not in known
        ]
        if rows:
            conn.execute(forecasts.insert(), rows)
    return len(rows)


def upsert_observations(engine: Engine, kp: pd.Series, now: pd.Timestamp) -> int:
    """Insert or overwrite observed Kp per interval start (NaN skipped). Returns rows written."""
    kp = kp.dropna()
    if kp.empty:
        return 0
    starts = [_utc(t) for t in kp.index]
    with engine.begin() as conn:
        conn.execute(observations.delete().where(observations.c.interval_start.in_(starts)))
        conn.execute(
            observations.insert(),
            [
                {"interval_start": s, "kp": float(v), "updated_at": _utc(now)}
                for s, v in zip(starts, kp.to_numpy(), strict=True)
            ],
        )
    return len(kp)


def verified_pairs(engine: Engine, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Logged forecasts joined to the observed Kp of their target interval."""
    q = select(
        forecasts.c.issued,
        forecasts.c.horizon_h,
        forecasts.c.interval_start,
        forecasts.c.kp_last,
        *(forecasts.c[c] for c in PROB_COLUMNS),
        observations.c.kp.label("kp_observed"),
    ).join(observations, observations.c.interval_start == forecasts.c.interval_start)
    if since is not None:
        q = q.where(forecasts.c.interval_start >= _utc(since))
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(q).mappings().all())
    if df.empty:
        return df
    for col in ("issued", "interval_start"):
        df[col] = df[col].map(_ts)
    return df.sort_values(["interval_start", "horizon_h"]).reset_index(drop=True)


def counts(engine: Engine) -> dict:
    with engine.connect() as conn:
        n_fc = conn.execute(select(func.count()).select_from(forecasts)).scalar_one()
        n_obs = conn.execute(select(func.count()).select_from(observations)).scalar_one()
        first = conn.execute(select(func.min(forecasts.c.issued))).scalar_one()
        last = conn.execute(select(func.max(forecasts.c.issued))).scalar_one()
    return {
        "forecasts_logged": int(n_fc),
        "observations": int(n_obs),
        "first_issued": _ts(first) if first else None,
        "last_issued": _ts(last) if last else None,
    }


def observed_series(engine: Engine, since: pd.Timestamp) -> pd.Series:
    with engine.connect() as conn:
        rows = conn.execute(
            select(observations.c.interval_start, observations.c.kp)
            .where(observations.c.interval_start >= _utc(since))
            .order_by(observations.c.interval_start)
        ).all()
    return pd.Series(
        [r.kp for r in rows],
        index=pd.DatetimeIndex([_ts(r.interval_start) for r in rows]),
        dtype=float,
    )


def is_postgres(engine: Engine) -> bool:
    return engine.dialect.name == "postgresql"
