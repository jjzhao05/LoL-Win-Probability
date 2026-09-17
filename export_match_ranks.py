from pathlib import Path

import pandas as pd

from db import get_engine, MATCH_SNAPSHOTS_TABLE
from logging_utils import run_with_file_logging


# One-off/manual export, the same way collect_data.py is. Everything else
# that wants a match's rank (rank_breakdown.py, ablation.py's rank
# breakdown) reads this CSV instead of hitting Postgres itself, so those
# scripts keep working against just the already-materialized parquet/split
# files on disk. Re-run this after collect_data.py pulls new matches.
OUTPUT_FILE = Path("results/match_ranks.csv")

LOG_DIR = Path("logs")


def main():
    print(f"Reading distinct (match_id, rank) pairs from: {MATCH_SNAPSHOTS_TABLE}")

    query = f"SELECT DISTINCT match_id, rank FROM {MATCH_SNAPSHOTS_TABLE}"
    df = pd.read_sql_query(query, get_engine())

    df["match_id"] = df["match_id"].astype(str)
    df["rank"] = df["rank"].astype(str).str.strip().str.upper()

    dupes = df["match_id"].duplicated().sum()
    if dupes:
        print(
            f"Warning: {dupes} match_id(s) tagged with more than one rank value, "
            "keeping the first."
        )
        df = df.drop_duplicates(subset="match_id", keep="first")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Matches:", len(df))
    print("Rank counts:")
    print(df["rank"].value_counts())
    print()
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    run_with_file_logging(LOG_DIR, "export_match_ranks", main)
