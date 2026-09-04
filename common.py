from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


ROLES = ["top", "jg", "mid", "bot", "sup"]
TEAMS = [100, 200]

BASIC_STATS = [
    "current_gold",
    "total_gold",
    "level",
    "xp",
    "minions_killed",
    "jungle_minions_killed",
    "kills",
    "deaths",
    "assists",
    "solo_kills",
    "wards_placed",
    "wards_killed",
    "control_wards_placed",
]

DAMAGE_STATS = [
    "magic_damage_done",
    "magic_damage_done_to_champions",
    "magic_damage_taken",
    "physical_damage_done",
    "physical_damage_done_to_champions",
    "physical_damage_taken",
    "true_damage_done",
    "true_damage_done_to_champions",
    "true_damage_taken",
]

OBJECTIVES = ["plates", "dragons", "heralds", "barons", "elders"]

TOWER_PARTS = [
    "top_outer",
    "top_inner",
    "top_base",
    "mid_outer",
    "mid_inner",
    "mid_base",
    "bot_outer",
    "bot_inner",
    "bot_base",
    "nexus_tower_1",
    "nexus_tower_2",
]

MINUTE_BUCKETS = [
    (1, 5),
    (6, 10),
    (11, 15),
    (16, 20),
    (21, 25),
    (26, 30),
    (31, 45),
]

RANDOM_STATE = 101705
VAL_SIZE = 0.20

SPLIT_FILE = Path("results/shared_split_ids.npz")


def compute_metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_true, probs),
        "log_loss": log_loss(y_true, probs, labels=[0, 1]),
        "accuracy": accuracy_score(y_true, preds),
        "brier": brier_score_loss(y_true, probs),
        "rows": len(y_true),
    }


def print_metrics(name, metrics):
    print()
    print(name)
    print("AUC:", round(float(metrics["auc"]), 4))
    print("Log loss:", round(float(metrics["log_loss"]), 4))
    print("Accuracy:", round(float(metrics["accuracy"]), 4))
    print("Brier score:", round(float(metrics["brier"]), 4))
    if "rows" in metrics:
        print("Rows:", int(metrics["rows"]))


def evaluate_by_minute_bucket(pred_df, prob_col, target_col="target", minute_col="minute",
                               buckets=MINUTE_BUCKETS):
    rows = []

    for start, end in buckets:
        g = pred_df[(pred_df[minute_col] >= start) & (pred_df[minute_col] <= end)]

        if g.empty:
            continue

        probs = g[prob_col].to_numpy()
        targets = g[target_col].to_numpy()
        preds = (probs >= 0.5).astype(int)

        if len(np.unique(targets)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(targets, probs)

        rows.append({
            "minutes": f"{start}-{end}",
            "rows": len(g),
            "auc": auc,
            "log_loss": log_loss(targets, probs, labels=[0, 1]),
            "accuracy": accuracy_score(targets, preds),
            "brier": brier_score_loss(targets, probs),
        })

    return rows


def print_bucket_rows(rows):
    for row in rows:
        auc_value = row["auc"]
        auc_text = "nan" if (isinstance(auc_value, float) and np.isnan(auc_value)) else round(float(auc_value), 4)

        print(
            row["minutes"],
            "| rows:", row["rows"],
            "| auc:", auc_text,
            "| log_loss:", round(float(row["log_loss"]), 4),
            "| accuracy:", round(float(row["accuracy"]), 4),
            "| brier:", round(float(row["brier"]), 4),
        )


def save_calibration_plot(y_true, probs, out_path, title, n_bins=10):
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    brier = brier_score_loss(y_true, probs)
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title} (Brier = {brier:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return brier


def infer_monotone_constraints(feature_names):
    constraints = []
    for col in feature_names:
        if "_200_" in col and "_100_" not in col:
            constraints.append(-1)
        else:
            constraints.append(1)
    return tuple(constraints)
