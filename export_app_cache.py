"""Caches the tower/objective columns app.py needs for event annotations"""

from pathlib import Path

import pandas as pd

from common import TEAMS, TOWER_PARTS
from db import get_engine, MATCH_SNAPSHOTS_TABLE
from logging_utils import run_with_file_logging

XGB_PREDICTIONS_FILE = Path("results/xgb_predictions.parquet")
LOGREG_PREDICTIONS_FILE = Path("results/logreg_predictions.parquet")
OUTPUT_FILE = Path("results/match_events_cache.parquet")

LOG_DIR = Path("logs")

OBJECTIVES = ["dragons", "heralds", "barons", "elders"]


def needed_match_ids():
    ids = set()
    for path in (XGB_PREDICTIONS_FILE, LOGREG_PREDICTIONS_FILE):
        if path.exists():
            ids |= set(pd.read_parquet(path, columns=["match_id"])["match_id"].astype(str))
    return sorted(ids)


def main():
    match_ids = needed_match_ids()
    if not match_ids:
        raise SystemExit(
            "No match ids found in results/xgb_predictions.parquet or "
            "results/logreg_predictions.parquet. Run the training scripts first."
        )

    print(f"Caching event data for {len(match_ids)} test matches...")

    tower_cols = [f"{part}_{team}_destroyed" for part in TOWER_PARTS for team in TEAMS]
    obj_cols = [f"{obj}_{team}" for obj in OBJECTIVES for team in TEAMS]
    columns = ["match_id", "timestamp_sec"] + tower_cols + obj_cols

    id_list = ", ".join(f"'{mid}'" for mid in match_ids)
    query = (
        f"SELECT {', '.join(columns)} FROM {MATCH_SNAPSHOTS_TABLE} "
        f"WHERE match_id IN ({id_list})"
    )
    df = pd.read_sql_query(query, get_engine())
    df["match_id"] = df["match_id"].astype(str)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} rows across {df['match_id'].nunique()} matches to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_with_file_logging(LOG_DIR, "export_app_cache", main)
