from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import RANDOM_STATE, TEAMS, TOWER_PARTS


XGB_PREDICTIONS_FILE = Path("results/xgb_predictions.parquet")
LSTM_PREDICTIONS_FILE = Path("results/lstm_predictions.npz")
FULL_DATA_FILE = Path("data/full_dataset.parquet")

OUTPUT_DIR = Path("figures/xgb_vs_lstm_labeled_plots")

N_MATCHES = 30

# Deliberately excludes "plates" (unlike common.OBJECTIVES): plate gold is a
# minor economic tick, not a game-shaping objective worth annotating on the
# probability chart.
OBJECTIVES = ["dragons", "heralds", "barons", "elders"]


def load_xgb_predictions():
    df = pd.read_parquet(XGB_PREDICTIONS_FILE)

    required = {
        "match_id",
        "minute",
        "target",
        "pred_prob_team_100_win",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing XGBoost prediction columns: {sorted(missing)}")

    df["match_id"] = df["match_id"].astype(str)

    return df


def load_lstm_predictions():
    data = np.load(LSTM_PREDICTIONS_FILE, allow_pickle=True)

    required = {
        "probs_test",
        "y_test",
        "mask_test",
        "match_ids_test",
    }

    missing = required - set(data.files)

    if missing:
        raise ValueError(f"Missing LSTM prediction arrays: {sorted(missing)}")

    probs = data["probs_test"]
    y = data["y_test"]
    mask = data["mask_test"]
    match_ids = data["match_ids_test"].astype(str)

    rows = []

    for i, match_id in enumerate(match_ids):
        for t in range(probs.shape[1]):
            if mask[i, t] == 1:
                rows.append({
                    "match_id": match_id,
                    "minute": t + 1,
                    "target": int(y[i]),
                    "lstm_prob_team_100_win": float(probs[i, t]),
                })

    return pd.DataFrame(rows)


def load_full_data():
    df = pd.read_parquet(FULL_DATA_FILE)

    required = {
        "match_id",
        "timestamp_sec",
        "team_100_win",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing full-data columns: {sorted(missing)}")

    df["match_id"] = df["match_id"].astype(str)
    df["minute"] = (df["timestamp_sec"] / 60).round().astype(int)

    return df.sort_values(["match_id", "timestamp_sec"])


def add_tower_totals(df):
    df = df.copy()

    for team in TEAMS:
        tower_cols = [
            f"{part}_{team}_destroyed"
            for part in TOWER_PARTS
            if f"{part}_{team}_destroyed" in df.columns
        ]

        if tower_cols:
            df[f"team_{team}_towers"] = df[tower_cols].sum(axis=1)
        else:
            df[f"team_{team}_towers"] = 0

    return df


def clean_objective_name(obj):
    names = {
        "dragons": "dragon",
        "heralds": "herald",
        "barons": "baron",
        "elders": "elder",
    }

    return names.get(obj, obj)


def get_event_rows(full_df, match_id):
    g = full_df[full_df["match_id"] == match_id].copy()

    if g.empty:
        return pd.DataFrame()

    g = add_tower_totals(g)
    g = g.sort_values("timestamp_sec").copy()

    event_rows = []

    for team in TEAMS:
        team_name = "Blue" if team == 100 else "Red"

        tower_col = f"team_{team}_towers"
        tower_delta_col = f"{tower_col}_delta"

        g[tower_delta_col] = (
            g[tower_col]
            .diff()
            .fillna(g[tower_col])
            .clip(lower=0)
        )

        for _, row in g[g[tower_delta_col] > 0].iterrows():
            count = int(row[tower_delta_col])

            if count == 1:
                label = f"{team_name} tower"
            else:
                label = f"{team_name} {count} towers"

            event_rows.append({
                "minute": int(row["minute"]),
                "team": team,
                "type": "tower",
                "label": label,
            })

        for obj in OBJECTIVES:
            col = f"{obj}_{team}"

            if col not in g.columns:
                continue

            delta_col = f"{col}_delta"

            g[delta_col] = (
                g[col]
                .diff()
                .fillna(g[col])
                .clip(lower=0)
            )

            for _, row in g[g[delta_col] > 0].iterrows():
                count = int(row[delta_col])
                obj_name = clean_objective_name(obj)

                if count == 1:
                    label = f"{team_name} {obj_name}"
                else:
                    label = f"{team_name} {count} {obj}"

                event_rows.append({
                    "minute": int(row["minute"]),
                    "team": team,
                    "type": obj,
                    "label": label,
                })

    events = pd.DataFrame(event_rows)

    if events.empty:
        return events

    events = (
        events
        .groupby(["minute", "team"], as_index=False)
        .agg({"label": lambda x: ", ".join(x)})
        .sort_values(["minute", "team"])
    )

    return events


def merge_predictions(xgb_df, lstm_df):
    merged = xgb_df.merge(
        lstm_df[["match_id", "minute", "lstm_prob_team_100_win"]],
        on=["match_id", "minute"],
        how="inner",
    )

    return merged.sort_values(["match_id", "minute"])


def annotate_events(ax, events):
    for _, event in events.iterrows():
        minute = int(event["minute"])
        team = int(event["team"])
        label = event["label"]

        if team == 100:
            line_color = "blue"
            text_y = 0.97
            va = "top"
        else:
            line_color = "red"
            text_y = 0.03
            va = "bottom"

        ax.axvline(
            minute,
            linewidth=0.9,
            alpha=0.25,
            color=line_color,
        )

        ax.text(
            minute,
            text_y,
            label,
            rotation=90,
            fontsize=8,
            ha="center",
            va=va,
            color=line_color,
            alpha=0.9,
        )


def plot_match(match_df, full_df, match_id):
    g = match_df[match_df["match_id"] == match_id].copy()

    if g.empty:
        return

    winner = "Blue" if int(g["target"].iloc[0]) == 1 else "Red"

    minutes = g["minute"].to_numpy()
    xgb_probs = g["pred_prob_team_100_win"].to_numpy()
    lstm_probs = g["lstm_prob_team_100_win"].to_numpy()

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        minutes,
        xgb_probs,
        linewidth=2,
        label="XGBoost",
        color="blue",
    )

    ax.plot(
        minutes,
        lstm_probs,
        linewidth=2,
        linestyle="--",
        label="LSTM",
        color="orange",
    )

    ax.fill_between(
        minutes,
        0.5,
        xgb_probs,
        where=xgb_probs >= 0.5,
        alpha=0.20,
        interpolate=True,
        color="blue",
    )

    ax.fill_between(
        minutes,
        0.5,
        xgb_probs,
        where=xgb_probs < 0.5,
        alpha=0.20,
        interpolate=True,
        color="red",
    )

    ax.axhline(
        0.5,
        linewidth=1,
        color="black",
    )

    events = get_event_rows(full_df, match_id)

    if not events.empty:
        annotate_events(ax, events)

    ax.set_title(
        f"XGBoost vs LSTM Win Probability | Match {match_id} | Winner: {winner}",
        fontsize=14,
    )

    ax.set_xlabel("Game minute")
    ax.set_ylabel("Blue-side win probability")

    ax.set_ylim(0, 1)

    ax.set_yticks([1.0, 0.75, 0.5, 0.25, 0.0])
    ax.set_yticklabels([
        "Blue 100%",
        "Blue 75%",
        "50/50",
        "Red 75%",
        "Red 100%",
    ])

    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    plt.tight_layout()

    out_file = OUTPUT_DIR / f"xgb_vs_lstm_labeled_{match_id}.png"

    plt.savefig(out_file, dpi=150)
    plt.close()

    print("Saved:", out_file)


def clear_output_dir():
    """Remove PNGs from a previous run before writing this run's sample.

    Without this, re-running with a different random sample of matches (e.g.
    because the underlying predictions changed) leaves stale plots from the
    old sample sitting alongside the new ones, so the folder silently
    accumulates more files than N_MATCHES actually asks for.
    """
    if not OUTPUT_DIR.exists():
        return

    removed = 0
    for path in OUTPUT_DIR.glob("xgb_vs_lstm_labeled_*.png"):
        path.unlink()
        removed += 1

    if removed:
        print(f"Removed {removed} stale plot(s) from a previous run.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_output_dir()

    xgb_df = load_xgb_predictions()
    lstm_df = load_lstm_predictions()
    full_df = load_full_data()

    merged = merge_predictions(xgb_df, lstm_df)

    common_matches = sorted(merged["match_id"].unique())

    if not common_matches:
        raise ValueError("No common matches found between XGBoost and LSTM predictions.")

    rng = np.random.default_rng(RANDOM_STATE)

    sampled_matches = rng.choice(
        common_matches,
        size=min(N_MATCHES, len(common_matches)),
        replace=False,
    )

    print("Common matches:", len(common_matches))
    print("Plotting matches:", len(sampled_matches))

    for match_id in sampled_matches:
        plot_match(merged, full_df, match_id)


if __name__ == "__main__":
    main()