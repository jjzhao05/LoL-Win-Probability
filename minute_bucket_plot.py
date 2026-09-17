from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import MINUTE_BUCKETS, evaluate_by_minute_bucket
from logging_utils import run_with_file_logging


XGB_PREDICTIONS_FILE = Path("results/xgb_predictions.parquet")
LOGREG_PREDICTIONS_FILE = Path("results/logreg_predictions.parquet")

OUTPUT_CSV = Path("results/minute_bucket_breakdown.csv")
OUTPUT_PLOT = Path("figures/auc_by_minute.png")

LOG_DIR = Path("logs")

MODEL_COLORS = {"XGBoost": "#1f77b4", "LogisticRegression": "#ff7f0e"}

BUCKET_LABELS = [f"{start}-{end}" for start, end in MINUTE_BUCKETS]


def load_predictions(path):
    df = pd.read_parquet(path)

    required = {"minute", "target", "pred_prob_team_100_win"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {sorted(missing)}")

    return df


def compute_bucket_table(pred_df, model_name):
    rows = evaluate_by_minute_bucket(pred_df, prob_col="pred_prob_team_100_win", buckets=MINUTE_BUCKETS)
    df = pd.DataFrame(rows)
    df.insert(0, "model", model_name)
    return df


def print_table(model_name, df):
    print(f"\n{model_name} test metrics by minute bucket:")
    for _, row in df.iterrows():
        print(
            f"  {row['minutes']:<8} | rows: {row['rows']:>6} | auc: {row['auc']:.4f} "
            f"| log_loss: {row['log_loss']:.4f} | accuracy: {row['accuracy']:.4f} | brier: {row['brier']:.4f}"
        )


def save_plot(combined, out_path=OUTPUT_PLOT):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model_name, group in combined.groupby("model"):
        group = group.set_index("minutes").reindex(BUCKET_LABELS)
        ax.plot(
            BUCKET_LABELS, group["auc"],
            marker="o", label=model_name, color=MODEL_COLORS.get(model_name), linewidth=2,
        )

    ax.set_xlabel("Game minute bucket")
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Test AUC by minute bucket")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    xgb_pred = load_predictions(XGB_PREDICTIONS_FILE)
    logreg_pred = load_predictions(LOGREG_PREDICTIONS_FILE)

    xgb_df = compute_bucket_table(xgb_pred, "XGBoost")
    logreg_df = compute_bucket_table(logreg_pred, "LogisticRegression")

    print_table("XGBoost", xgb_df)
    print_table("LogisticRegression", logreg_df)

    combined = pd.concat([xgb_df, logreg_df], ignore_index=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.round(4).to_csv(OUTPUT_CSV, index=False)
    print("\nSaved:", OUTPUT_CSV)

    save_plot(combined)
    print("Saved:", OUTPUT_PLOT)


if __name__ == "__main__":
    run_with_file_logging(LOG_DIR, "minute_bucket_plot", main)
