from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import MODEL_COLORS, compute_metrics
from logging_utils import run_with_file_logging


XGB_PREDICTIONS_FILE = Path("results/xgb_predictions.parquet")
LOGREG_PREDICTIONS_FILE = Path("results/logreg_predictions.parquet")
MATCH_RANKS_FILE = Path("results/match_ranks.csv")

OUTPUT_CSV = Path("results/rank_breakdown.csv")
OUTPUT_PLOT = Path("figures/rank_breakdown.png")

LOG_DIR = Path("logs")

RANK_ORDER = [
    "IRON",
    "BRONZE",
    "SILVER",
    "GOLD",
    "PLATINUM",
    "EMERALD",
    "DIAMOND",
    "MASTER",
    "GRANDMASTER",
    "CHALLENGER",
]

# Below this many rows, an AUC estimate for that rank/model is too noisy to
# report on its own (also true if only one class is present at all).
MIN_ROWS_FOR_AUC = 30


def load_predictions(path, rename_to=None):
    df = pd.read_parquet(path)

    required = {"match_id", "minute", "target", "pred_prob_team_100_win"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {sorted(missing)}")

    df["match_id"] = df["match_id"].astype(str)

    if rename_to:
        df = df.rename(columns={"pred_prob_team_100_win": rename_to})

    return df


def load_match_ranks():
    """One row per match_id with its rank tier, from export_match_ranks.py's CSV."""
    if not MATCH_RANKS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {MATCH_RANKS_FILE}. Run export_match_ranks.py first "
            "so match-to-rank lookups don't need a live database connection."
        )

    df = pd.read_csv(MATCH_RANKS_FILE, dtype=str)
    df["rank"] = df["rank"].str.strip().str.upper()

    return df


def order_ranks(ranks):
    known = [r for r in RANK_ORDER if r in ranks]
    unknown = sorted(r for r in ranks if r not in RANK_ORDER)
    return known + unknown


def evaluate_by_rank(df, prob_col, target_col="target", rank_col="rank"):
    rows = []

    for rank in order_ranks(df[rank_col].unique()):
        g = df[df[rank_col] == rank]

        n_rows = len(g)
        n_matches = g["match_id"].nunique()
        targets = g[target_col].to_numpy()

        if n_rows < MIN_ROWS_FOR_AUC or len(np.unique(targets)) < 2:
            rows.append({
                "rank": rank,
                "rows": n_rows,
                "matches": n_matches,
                "auc": float("nan"),
                "accuracy": float("nan"),
                "brier": float("nan"),
            })
            continue

        metrics = compute_metrics(targets, g[prob_col].to_numpy())

        rows.append({
            "rank": rank,
            "rows": n_rows,
            "matches": n_matches,
            "auc": metrics["auc"],
            "accuracy": metrics["accuracy"],
            "brier": metrics["brier"],
        })

    return pd.DataFrame(rows)


def print_breakdown(name, breakdown_df):
    print()
    print(name)
    for _, row in breakdown_df.iterrows():
        auc_text = "n/a (too few rows)" if np.isnan(row["auc"]) else f"{row['auc']:.4f}"
        print(
            f"  {row['rank']:<12} rows={int(row['rows']):>7}  "
            f"matches={int(row['matches']):>5}  auc={auc_text}"
        )


def save_plot(xgb_df, logreg_df, out_path):
    ranks = order_ranks(set(xgb_df["rank"]) | set(logreg_df["rank"]))

    xgb_by_rank = xgb_df.set_index("rank")["auc"]
    logreg_by_rank = logreg_df.set_index("rank")["auc"]

    x = np.arange(len(ranks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(ranks)), 5))

    ax.bar(x - width / 2, [xgb_by_rank.get(r, np.nan) for r in ranks], width, label="XGBoost", color=MODEL_COLORS["XGBoost"])
    ax.bar(x + width / 2, [logreg_by_rank.get(r, np.nan) for r in ranks], width, label="Logistic Regression", color=MODEL_COLORS["LogisticRegression"])

    ax.set_xticks(x)
    ax.set_xticklabels(ranks, rotation=30, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("Test AUC by rank")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    xgb_df = load_predictions(XGB_PREDICTIONS_FILE)
    logreg_df = load_predictions(LOGREG_PREDICTIONS_FILE)
    ranks_df = load_match_ranks()

    xgb_df = xgb_df.merge(ranks_df, on="match_id", how="left")
    logreg_df = logreg_df.merge(ranks_df, on="match_id", how="left")

    missing_xgb = xgb_df["rank"].isna().sum()
    missing_logreg = logreg_df["rank"].isna().sum()

    if missing_xgb or missing_logreg:
        print(
            f"Warning: {missing_xgb} XGBoost rows and {missing_logreg} logistic "
            "regression rows had no matching rank (dropped)."
        )
        xgb_df = xgb_df.dropna(subset=["rank"])
        logreg_df = logreg_df.dropna(subset=["rank"])

    xgb_breakdown = evaluate_by_rank(xgb_df, prob_col="pred_prob_team_100_win")
    logreg_breakdown = evaluate_by_rank(logreg_df, prob_col="pred_prob_team_100_win")

    print_breakdown("XGBoost test AUC by rank", xgb_breakdown)
    print_breakdown("Logistic regression test AUC by rank", logreg_breakdown)

    combined = pd.concat(
        [
            xgb_breakdown.assign(model="XGBoost"),
            logreg_breakdown.assign(model="LogisticRegression"),
        ],
        ignore_index=True,
    )[["model", "rank", "rows", "matches", "auc", "accuracy", "brier"]]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    print()
    print("Saved rank breakdown to:", OUTPUT_CSV)

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    save_plot(xgb_breakdown, logreg_breakdown, OUTPUT_PLOT)
    print("Saved rank breakdown chart to:", OUTPUT_PLOT)


if __name__ == "__main__":
    run_with_file_logging(LOG_DIR, "rank_breakdown", main)
