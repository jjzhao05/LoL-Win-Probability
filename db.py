"""Shared Postgres connection for the pipeline.

Every script that reads or writes match data goes through `get_engine()`
here, so there is exactly one place that knows how to connect.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

# Every raw match-minute snapshot collected from the Riot API lives in this
# one wide table (one row per match per timeline frame). It replaces the old
# per-rank parquet folders + full_dataset.parquet combine step: `rank` is
# just a normal column now, so there's nothing to combine.
MATCH_SNAPSHOTS_TABLE = "match_snapshots"

_engine = None


def get_engine():
    """Create (or reuse) the Postgres engine. DATABASE_URL is checked here,
    not at import time, so importing this module doesn't require a database
    connection -- only actually calling get_engine() does. That matters for
    app.py, which imports MATCH_SNAPSHOTS_TABLE from here but only needs a
    live connection for one optional feature (real-match event annotations)."""
    global _engine
    if _engine is None:
        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeError(
                "Missing DATABASE_URL. Add it to your .env file, e.g.\n"
                "DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/lol_win_prob"
            )
        _engine = create_engine(database_url)
    return _engine
