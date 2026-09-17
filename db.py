"""Shared Postgres connection for the pipeline."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

MATCH_SNAPSHOTS_TABLE = "match_snapshots"

_engine = None


def get_engine():
    """Create (or reuse) the Postgres engine. DATABASE_URL is checked here,
    not at import time, so importing this module alone doesn't require a
    database connection."""
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
