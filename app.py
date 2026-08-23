"""
League of Legends Win-Probability Model -- interactive companion app.

Three pages, picked from the sidebar:
  1. Match Replay Explorer -- pick a held-out test match, watch XGBoost's and
     the LSTM's win-probability curves over the course of the game, with
     objective/tower events annotated (an interactive version of
     probability_plot.py).
  2. Model Performance Dashboard -- calibration curves, per-minute-bucket
     metrics, feature importances, and hyperparameter grid search results for
     both models.
  3. Live What-If Predictor -- move a handful of macro sliders (gold/xp/cs
     lead, objectives, towers) and get a live XGBoost win-probability read
     out, useful for building intuition about which signals move the model.

Run with:  streamlit run app.py
(from the project root, after main.py / the individual train scripts have
been run at least once so the files below exist.)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from common import (
    MINUTE_BUCKETS,
    TEAMS,
    TOWER_PARTS,
    evaluate_by_minute_bucket,
)

# --------------------------------------------------------------------------
# File locations (all local to the project folder -- this app is meant to be
# run from the repo root, reading the artifacts the pipeline scripts wrote).
# --------------------------------------------------------------------------

XGB_PREDICTIONS_FILE = Path("results/xgb_predictions.parquet")
LSTM_PREDICTIONS_FILE = Path("results/lstm_predictions.npz")
FULL_DATA_FILE = Path("data/full_dataset.parquet")
XGB_MODEL_FILE = Path("models/xgb_model.json")
XGB_FEATURE_IMPORTANCE_CSV = Path("results/xgb_feature_importances.csv")
XGB_CALIBRATION_IMG = Path("figures/xgb_calibration_curve.png")
LSTM_CALIBRATION_IMG = Path("figures/lstm_calibration_curve.png")
XGB_GRID_RESULTS = Path("results/xgb_grid_search_results.csv")
LSTM_GRID_RESULTS = Path("results/lstm_grid_search_results.csv")
RUN_TIMES_FILE = Path("results/run_times.csv")

OBJECTIVES = ["dragons", "heralds", "barons", "elders"]


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------

@st.cache_data
def load_xgb_predictions():
    if not XGB_PREDICTIONS_FILE.exists():
        return None
    df = pd.read_parquet(XGB_PREDICTIONS_FILE)
    df["match_id"] = df["match_id"].astype(str)
    # xgboost_train.py's "minute" comes from timestamp_sec / 60 and is
    # occasionally fractional for a match's final frame (e.g. 73.13). The
    # LSTM side's minute is always an integer sequence position, so an
    # unrounded merge on "minute" silently drops those trailing frames.
    df["minute"] = df["minute"].round().astype(int)
    return df


@st.cache_data
def load_lstm_predictions():
    if not LSTM_PREDICTIONS_FILE.exists():
        return None

    data = np.load(LSTM_PREDICTIONS_FILE, allow_pickle=True)
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


@st.cache_data
def load_full_data():
    if not FULL_DATA_FILE.exists():
        return None

    df = pd.read_parquet(FULL_DATA_FILE)
    df["match_id"] = df["match_id"].astype(str)
    df["minute"] = (df["timestamp_sec"] / 60).round().astype(int)
    return df.sort_values(["match_id", "timestamp_sec"])


@st.cache_resource
def load_xgb_model():
    if not XGB_MODEL_FILE.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(str(XGB_MODEL_FILE))
    return booster


@st.cache_data
def load_csv_if_exists(path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


def merge_predictions(xgb_df, lstm_df):
    merged = xgb_df.merge(
        lstm_df[["match_id", "minute", "lstm_prob_team_100_win"]],
        on=["match_id", "minute"],
        how="inner",
    )
    return merged.sort_values(["match_id", "minute"])


# --------------------------------------------------------------------------
# Page 1: Match Replay Explorer
# --------------------------------------------------------------------------

def add_tower_totals(df):
    df = df.copy()
    for team in TEAMS:
        tower_cols = [
            f"{part}_{team}_destroyed"
            for part in TOWER_PARTS
            if f"{part}_{team}_destroyed" in df.columns
        ]
        df[f"team_{team}_towers"] = df[tower_cols].sum(axis=1) if tower_cols else 0
    return df


def clean_objective_name(obj):
    return {"dragons": "dragon", "heralds": "herald", "barons": "baron", "elders": "elder"}.get(obj, obj)


def get_event_rows(full_df, match_id):
    g = full_df[full_df["match_id"] == match_id].copy()
    if g.empty:
        return pd.DataFrame()

    g = add_tower_totals(g).sort_values("timestamp_sec").copy()
    event_rows = []

    for team in TEAMS:
        team_name = "Blue" if team == 100 else "Red"
        tower_col = f"team_{team}_towers"
        delta_col = f"{tower_col}_delta"
        g[delta_col] = g[tower_col].diff().fillna(g[tower_col]).clip(lower=0)

        for _, row in g[g[delta_col] > 0].iterrows():
            count = int(row[delta_col])
            label = f"{team_name} tower" if count == 1 else f"{team_name} {count} towers"
            event_rows.append({"minute": int(row["minute"]), "team": team, "label": label})

        for obj in OBJECTIVES:
            col = f"{obj}_{team}"
            if col not in g.columns:
                continue
            dcol = f"{col}_delta"
            g[dcol] = g[col].diff().fillna(g[col]).clip(lower=0)
            for _, row in g[g[dcol] > 0].iterrows():
                count = int(row[dcol])
                name = clean_objective_name(obj)
                label = f"{team_name} {name}" if count == 1 else f"{team_name} {count} {obj}"
                event_rows.append({"minute": int(row["minute"]), "team": team, "label": label})

    events = pd.DataFrame(event_rows)
    if events.empty:
        return events

    return (
        events.groupby(["minute", "team"], as_index=False)
        .agg({"label": lambda x: ", ".join(x)})
        .sort_values(["minute", "team"])
    )


def render_match_replay():
    st.header("Match Replay Explorer")
    st.caption(
        "Pick a held-out test match and watch how each model's predicted win "
        "probability moved over the course of the game."
    )

    xgb_pred = load_xgb_predictions()
    lstm_pred = load_lstm_predictions()

    if xgb_pred is None or lstm_pred is None:
        st.warning(
            "Missing `xgb_predictions.parquet` and/or `lstm_predictions.npz`. "
            "Run `xgboost_train.py` and `lstm_train.py` first."
        )
        return

    merged = merge_predictions(xgb_pred, lstm_pred)
    common_matches = sorted(merged["match_id"].unique())

    if not common_matches:
        st.error("No matches are common to both the XGBoost and LSTM test predictions.")
        return

    match_id = st.selectbox("Test match", common_matches)

    g = merged[merged["match_id"] == match_id].sort_values("minute")
    if g.empty:
        st.warning("No rows for this match.")
        return

    winner = "Blue" if int(g["target"].iloc[0]) == 1 else "Red"
    st.markdown(f"**Winner:** {winner}-side (Team {'100' if winner == 'Blue' else '200'})")

    full_df = load_full_data()
    events = get_event_rows(full_df, match_id) if full_df is not None else pd.DataFrame()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["minute"], y=g["pred_prob_team_100_win"],
        mode="lines", name="XGBoost", line=dict(color="royalblue", width=3),
        hovertemplate="minute %{x}<br>XGBoost: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=g["minute"], y=g["lstm_prob_team_100_win"],
        mode="lines", name="LSTM", line=dict(color="orange", width=3, dash="dash"),
        hovertemplate="minute %{x}<br>LSTM: %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_color="black", line_width=1)

    if not events.empty:
        for _, ev in events.iterrows():
            color = "blue" if ev["team"] == 100 else "red"
            fig.add_vline(x=ev["minute"], line_width=1, line_color=color, opacity=0.3)
            fig.add_annotation(
                x=ev["minute"], y=1.0 if ev["team"] == 100 else 0.0,
                text=ev["label"], showarrow=False, textangle=-90,
                font=dict(size=9, color=color), yshift=6 if ev["team"] == 100 else -6,
            )
    elif full_df is None:
        st.info(
            "No `data/full_dataset.parquet` found -- showing win probability without "
            "objective/tower event annotations."
        )

    fig.update_layout(
        yaxis=dict(
            range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["Red 100%", "Red 75%", "50/50", "Blue 75%", "Blue 100%"],
            title="Blue-side win probability",
        ),
        xaxis_title="Game minute",
        title=f"XGBoost vs LSTM | Match {match_id} | Winner: {winner}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        height=520,
        margin=dict(t=80),
    )

    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# Page 2: Model Performance Dashboard
# --------------------------------------------------------------------------

def render_dashboard():
    st.header("Model Performance Dashboard")

    xgb_pred = load_xgb_predictions()
    lstm_pred = load_lstm_predictions()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("XGBoost calibration")
        if XGB_CALIBRATION_IMG.exists():
            st.image(str(XGB_CALIBRATION_IMG), width="stretch")
        else:
            st.info("Run `xgboost_train.py` to generate `xgb_calibration_curve.png`.")
    with col2:
        st.subheader("LSTM calibration")
        if LSTM_CALIBRATION_IMG.exists():
            st.image(str(LSTM_CALIBRATION_IMG), width="stretch")
        else:
            st.info("Run `lstm_train.py` to generate `lstm_calibration_curve.png`.")

    st.subheader("Performance by minute bucket")
    if xgb_pred is not None:
        xgb_buckets = evaluate_by_minute_bucket(
            xgb_pred, prob_col="pred_prob_team_100_win", buckets=MINUTE_BUCKETS
        )
        xgb_bucket_df = pd.DataFrame(xgb_buckets).rename(columns=lambda c: f"xgb_{c}" if c != "minutes" else c)
    else:
        xgb_bucket_df = None

    if lstm_pred is not None:
        lstm_buckets = evaluate_by_minute_bucket(
            lstm_pred, prob_col="lstm_prob_team_100_win", buckets=MINUTE_BUCKETS
        )
        lstm_bucket_df = pd.DataFrame(lstm_buckets).rename(columns=lambda c: f"lstm_{c}" if c != "minutes" else c)
    else:
        lstm_bucket_df = None

    if xgb_bucket_df is not None and lstm_bucket_df is not None:
        combined = xgb_bucket_df.merge(lstm_bucket_df, on="minutes", how="outer")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined["minutes"], y=combined["xgb_auc"], name="XGBoost AUC", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=combined["minutes"], y=combined["lstm_auc"], name="LSTM AUC", mode="lines+markers"))
        fig.update_layout(yaxis_title="AUC", xaxis_title="Game minute bucket", height=380)
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            combined[["minutes", "xgb_rows", "xgb_auc", "xgb_log_loss", "xgb_accuracy", "xgb_brier",
                      "lstm_rows", "lstm_auc", "lstm_log_loss", "lstm_accuracy", "lstm_brier"]]
            .round(4),
            width="stretch",
        )
    else:
        st.info("Need both `xgb_predictions.parquet` and `lstm_predictions.npz` for this comparison.")

    st.subheader("XGBoost feature importance")
    imp_df = load_csv_if_exists(XGB_FEATURE_IMPORTANCE_CSV)
    if imp_df is not None:
        top_n = st.slider("Show top N features", 5, 50, 20)
        top = imp_df.sort_values("gain", ascending=False).head(top_n).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["gain"], y=top["feature"], orientation="h"))
        fig.update_layout(
            xaxis_title="Gain (avg loss reduction per split)",
            height=max(400, 22 * len(top)),
            margin=dict(l=220),
        )
        st.plotly_chart(fig, width="stretch")
        with st.expander("Full feature importance table"):
            st.dataframe(imp_df.sort_values("gain", ascending=False), width="stretch")
    else:
        st.info("Run `xgboost_train.py` (current version) to generate `xgb_feature_importances.csv`.")

    st.subheader("Hyperparameter grid search")
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.markdown("**XGBoost**")
        xgb_grid = load_csv_if_exists(XGB_GRID_RESULTS)
        if xgb_grid is not None:
            st.dataframe(
                xgb_grid.sort_values("val_log_loss").reset_index(drop=True),
                width="stretch",
            )
        else:
            st.info("`xgb_grid_search_results.csv` not found.")
    with gcol2:
        st.markdown("**LSTM**")
        lstm_grid = load_csv_if_exists(LSTM_GRID_RESULTS)
        if lstm_grid is not None:
            st.dataframe(
                lstm_grid.sort_values("val_log_loss").reset_index(drop=True),
                width="stretch",
            )
        else:
            st.info("`lstm_grid_search_results.csv` not found.")

    run_times = load_csv_if_exists(RUN_TIMES_FILE)
    if run_times is not None:
        st.subheader("Last pipeline run times")
        st.dataframe(run_times, width="stretch")


# --------------------------------------------------------------------------
# Page 3: Live What-If Predictor
# --------------------------------------------------------------------------

def build_feature_vector(booster, inputs):
    """Build a full feature row for the model from a small set of macro
    "what-if" sliders. Every feature the model was NOT given a slider for
    (all per-role breakdowns, all 1-/3-minute momentum deltas, damage/vision
    detail) is left at 0 -- i.e. "no additional information beyond the macro
    team-level snapshot entered here". This is a deliberate simplification
    for exploration, not a claim that those signals don't matter.
    """
    feature_names = booster.feature_names
    row = pd.Series(0.0, index=feature_names)

    minute = inputs["minute"]
    gold_diff = inputs["gold_diff"]
    xp_diff = inputs["xp_diff"]
    cs_diff = inputs["cs_diff"]

    if "minute" in row.index:
        row["minute"] = minute
    if "timestamp_sec" in row.index:
        row["timestamp_sec"] = minute * 60

    if "team_total_gold_diff" in row.index:
        row["team_total_gold_diff"] = gold_diff
    if "gold_diff_per_min" in row.index:
        row["gold_diff_per_min"] = gold_diff / max(minute, 1)

    if "team_xp_diff" in row.index:
        row["team_xp_diff"] = xp_diff
    if "xp_diff_per_min" in row.index:
        row["xp_diff_per_min"] = xp_diff / max(minute, 1)

    if "team_minions_killed_diff" in row.index:
        row["team_minions_killed_diff"] = cs_diff
    if "cs_diff_per_min" in row.index:
        row["cs_diff_per_min"] = cs_diff / max(minute, 1)

    # team_100_gold_share has no absolute-gold inputs available here, so it's
    # approximated from the gold diff against a rough combined-team-gold
    # baseline (~1600 gold/min combined at even game states). This is a
    # simplification for the demo, not a recomputation of the real feature.
    if "team_100_gold_share" in row.index:
        baseline_total_gold = max(2 * 2500 + 1600 * minute, 1)
        row["team_100_gold_share"] = float(np.clip(0.5 + gold_diff / (2 * baseline_total_gold), 0.0, 1.0))

    if "team_kills_diff" in row.index:
        row["team_kills_diff"] = inputs["kills_diff"]
    if "team_deaths_diff" in row.index:
        row["team_deaths_diff"] = -inputs["kills_diff"]
    if "team_assists_diff" in row.index:
        row["team_assists_diff"] = inputs["kills_diff"]

    towers_100 = inputs["towers_100"]
    towers_200 = inputs["towers_200"]
    if "team_100_towers_destroyed" in row.index:
        row["team_100_towers_destroyed"] = towers_100
    if "team_200_towers_destroyed" in row.index:
        row["team_200_towers_destroyed"] = towers_200
    if "tower_diff" in row.index:
        row["tower_diff"] = towers_100 - towers_200

    for obj, key in [("dragons", "dragons_diff"), ("heralds", "heralds_diff"),
                      ("barons", "barons_diff"), ("elders", "elders_diff"),
                      ("plates", "plates_diff")]:
        if key in row.index:
            row[key] = inputs.get(key, 0)

    fb = inputs["first_blood"]  # "Blue", "Red", "Neither"
    if fb != "Neither":
        if "first_blood_diff" in row.index:
            row["first_blood_diff"] = 1 if fb == "Blue" else -1
        if fb == "Blue" and "team_100_first_blood_so_far" in row.index:
            row["team_100_first_blood_so_far"] = 1
        if fb == "Red" and "team_200_first_blood_so_far" in row.index:
            row["team_200_first_blood_so_far"] = 1

    ft = inputs["first_tower"]
    if ft != "Neither":
        if "first_tower_diff" in row.index:
            row["first_tower_diff"] = 1 if ft == "Blue" else -1
        if ft == "Blue" and "team_100_first_tower_so_far" in row.index:
            row["team_100_first_tower_so_far"] = 1
        if ft == "Red" and "team_200_first_tower_so_far" in row.index:
            row["team_200_first_tower_so_far"] = 1

    return row


def render_predictor():
    st.header("Live What-If Predictor")
    st.caption(
        "Set a macro game state and get a live XGBoost win-probability read. Only "
        "the aggregate team-level signals below are controlled -- every per-role, "
        "per-minute-momentum, and vision/damage-detail feature the model also uses "
        "is held at a neutral 0, so treat this as a simplified sketch of the model's "
        "behavior, not a full replica of the trained pipeline's prediction on a real game."
    )
    st.info(
        "**A real finding, not a UI bug:** this model leans overwhelmingly on the gold/XP "
        "differential. With gold and XP lead left at 0, moving towers, objectives, or first "
        "blood alone barely changes the prediction -- gold/XP dominate hard enough that "
        "secondary signals mostly matter *through* the gold lead they typically come with, "
        "not independently of it. To see towers/objectives move the needle, pair them with a "
        "modest gold/XP lead (which is also what a real game with those events looks like)."
    )

    booster = load_xgb_model()
    if booster is None:
        st.warning("`xgb_model.json` not found. Run `xgboost_train.py` first.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        minute = st.slider("Game minute", 1, 45, 15)
        gold_diff = st.slider("Blue gold lead", -15000, 15000, 0, step=250)
        xp_diff = st.slider("Blue XP lead", -15000, 15000, 0, step=250)
        cs_diff = st.slider("Blue CS lead", -100, 100, 0, step=1)
    with c2:
        kills_diff = st.slider("Blue net kills (kills - deaths)", -20, 20, 0)
        towers_100 = st.slider("Blue towers destroyed", 0, 11, 0)
        towers_200 = st.slider("Red towers destroyed", 0, 11, 0)
        dragons_diff = st.slider("Blue dragon advantage", -4, 4, 0)
    with c3:
        heralds_diff = st.slider("Blue herald advantage", -2, 2, 0)
        barons_diff = st.slider("Blue baron advantage", -3, 3, 0)
        elders_diff = st.slider("Blue elder dragon advantage", -2, 2, 0)
        first_blood = st.radio("First blood", ["Neither", "Blue", "Red"], horizontal=True)
        first_tower = st.radio("First tower", ["Neither", "Blue", "Red"], horizontal=True)

    inputs = {
        "minute": minute,
        "gold_diff": gold_diff,
        "xp_diff": xp_diff,
        "cs_diff": cs_diff,
        "kills_diff": kills_diff,
        "towers_100": towers_100,
        "towers_200": towers_200,
        "dragons_diff": dragons_diff,
        "heralds_diff": heralds_diff,
        "barons_diff": barons_diff,
        "elders_diff": elders_diff,
        "first_blood": first_blood,
        "first_tower": first_tower,
    }

    row = build_feature_vector(booster, inputs)
    dmat = xgb.DMatrix(row.to_frame().T, feature_names=booster.feature_names)
    prob_blue = float(booster.predict(dmat)[0])

    st.divider()
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("Blue win probability", f"{prob_blue:.1%}")
        st.metric("Red win probability", f"{1 - prob_blue:.1%}")
    with m2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_blue * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "royalblue"},
                "steps": [
                    {"range": [0, 50], "color": "#fde0e0"},
                    {"range": [50, 100], "color": "#dbe7fb"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
            },
            title={"text": "Blue-side win probability"},
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="LoL Win Probability Model", layout="wide")
    st.title("League of Legends Win-Probability Model")

    page = st.sidebar.radio(
        "Page",
        ["Match Replay Explorer", "Model Performance Dashboard", "Live What-If Predictor"],
    )

    if page == "Match Replay Explorer":
        render_match_replay()
    elif page == "Model Performance Dashboard":
        render_dashboard()
    else:
        render_predictor()


if __name__ == "__main__":
    main()
