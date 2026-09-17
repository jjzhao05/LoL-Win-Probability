import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from common import (
    MINUTE_BUCKETS,
    MODEL_COLORS,
    TEAMS,
    TOWER_PARTS,
    evaluate_by_minute_bucket,
)
from db import get_engine, MATCH_SNAPSHOTS_TABLE


def show_chart(fig, **kwargs):
    fig.update_layout(dragmode=False)
    kwargs.setdefault("width", "stretch")
    st.plotly_chart(fig, **kwargs)


XGB_PREDICTIONS_FILE = Path("results/xgb_predictions.parquet")
LOGREG_PREDICTIONS_FILE = Path("results/logreg_predictions.parquet")

XGB_MODEL_FILE = Path("models/xgb_model.json")
LOGREG_MODEL_FILE = Path("models/logreg_model.joblib")

XGB_FEATURE_IMPORTANCE_CSV = Path("results/xgb_feature_importances.csv")
LOGREG_FEATURE_IMPORTANCE_CSV = Path("results/logreg_feature_importances.csv")

XGB_CALIBRATION_IMG = Path("figures/xgb_calibration_curve.png")
LOGREG_CALIBRATION_IMG = Path("figures/logreg_calibration_curve.png")

XGB_GRID_RESULTS = Path("results/xgb_grid_search_results.csv")
LOGREG_GRID_RESULTS = Path("results/logreg_grid_search_results.csv")

ABLATION_RESULTS_FILE = Path("results/ablation_results.csv")
ABLATION_BOOTSTRAP_FILE = Path("results/ablation_bootstrap_cis.csv")
ABLATION_CLOSENESS_FILE = Path("results/ablation_closeness_breakdown.csv")
ABLATION_RANK_FILE = Path("results/ablation_rank_breakdown.csv")

OBJECTIVES = ["dragons", "heralds", "barons", "elders"]

RANK_ORDER = [
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM",
    "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER",
]

SIGNIFICANT_ALPHA = 1.0
NOT_SIGNIFICANT_ALPHA = 0.35


def order_ranks(ranks):
    known = [r for r in RANK_ORDER if r in ranks]
    unknown = sorted(r for r in ranks if r not in RANK_ORDER)
    return known + unknown


@st.cache_data
def load_xgb_predictions():
    if not XGB_PREDICTIONS_FILE.exists():
        return None
    df = pd.read_parquet(XGB_PREDICTIONS_FILE)
    df["match_id"] = df["match_id"].astype(str)
    df["minute"] = df["minute"].round().astype(int)
    return df


@st.cache_data
def load_logreg_predictions():
    if not LOGREG_PREDICTIONS_FILE.exists():
        return None
    df = pd.read_parquet(LOGREG_PREDICTIONS_FILE)
    df["match_id"] = df["match_id"].astype(str)
    df["minute"] = df["minute"].round().astype(int)
    return df


@st.cache_data
def load_full_data():
    try:
        df = pd.read_sql_table(MATCH_SNAPSHOTS_TABLE, get_engine())
    except Exception:
        return None
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


@st.cache_resource
def load_logreg_model():
    if not LOGREG_MODEL_FILE.exists():
        return None
    return joblib.load(LOGREG_MODEL_FILE)


@st.cache_data
def load_csv_if_exists(path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


def merge_predictions(xgb_df, logreg_df):
    logreg_df = logreg_df.rename(columns={"pred_prob_team_100_win": "logreg_prob_team_100_win"})
    merged = xgb_df.merge(
        logreg_df[["match_id", "minute", "logreg_prob_team_100_win"]],
        on=["match_id", "minute"],
        how="inner",
    )
    return merged.sort_values(["match_id", "minute"])


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


def _pick_random_match(matches):
    st.session_state.selected_match = random.choice(matches)


def render_real_match_tab():
    xgb_pred = load_xgb_predictions()
    logreg_pred = load_logreg_predictions()

    if xgb_pred is None or logreg_pred is None:
        st.warning(
            "Missing `xgb_predictions.parquet` and/or `logreg_predictions.parquet`. "
            "Run `xgboost_train.py` and `logistic_regression_train.py` first."
        )
        return

    merged = merge_predictions(xgb_pred, logreg_pred)
    common_matches = sorted(merged["match_id"].unique())

    if not common_matches:
        st.error("No matches are common to both the XGBoost and logistic regression test predictions.")
        return

    if "selected_match" not in st.session_state or st.session_state.selected_match not in common_matches:
        st.session_state.selected_match = common_matches[0]

    sel_col, btn_col = st.columns([5, 1])
    with sel_col:
        match_id = st.selectbox("Test match", common_matches, key="selected_match")
    with btn_col:
        st.write("")
        st.write("")
        st.button("Random match", on_click=_pick_random_match, args=(common_matches,))

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
        mode="lines", name="XGBoost", line=dict(color=MODEL_COLORS["XGBoost"], width=3),
        hovertemplate="minute %{x}<br>XGBoost: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=g["minute"], y=g["logreg_prob_team_100_win"],
        mode="lines", name="Logistic Regression", line=dict(color=MODEL_COLORS["LogisticRegression"], width=3, dash="dash"),
        hovertemplate="minute %{x}<br>Logistic Regression: %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_color="white", line_width=3)

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
        st.info(f"Couldn't load `{MATCH_SNAPSHOTS_TABLE}` from the database. No event annotations shown.")

    fig.update_layout(
        yaxis=dict(
            range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["Red 100%", "Red 75%", "50/50", "Blue 75%", "Blue 100%"],
            title="Blue-side win probability",
        ),
        xaxis_title="Game minute",
        title=f"XGBoost vs Logistic Regression | Match {match_id} | Winner: {winner}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        height=520,
        margin=dict(t=80),
    )

    show_chart(fig, width="stretch")


def build_feature_vector(feature_names, inputs):
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

    fb = inputs["first_blood"]
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


def render_gauge(prob_blue, title, bar_color=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_blue * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color or MODEL_COLORS["XGBoost"]},
            "steps": [
                {"range": [0, 50], "color": "#fde0e0"},
                {"range": [50, 100], "color": "#dbe7fb"},
            ],
            "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
        },
        title={"text": title},
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10))
    return fig


def render_custom_scenario_tab():
    st.caption(
        "Set a macro game state and see each model's win probability. Momentum features are "
        "held at 0, so this is a simplified view, not a full replica of a real game."
    )

    booster = load_xgb_model()
    logreg = load_logreg_model()

    if booster is None and logreg is None:
        st.warning("No trained models found. Run `xgboost_train.py` and `logistic_regression_train.py` first.")
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
        heralds_diff = st.slider("Blue herald advantage", -1, 1, 0)
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

    st.divider()
    m1, m2 = st.columns(2)

    if booster is not None:
        row = build_feature_vector(booster.feature_names, inputs)
        dmat = xgb.DMatrix(row.to_frame().T, feature_names=booster.feature_names)
        prob_blue_xgb = float(booster.predict(dmat)[0])
        with m1:
            st.subheader("XGBoost")
            st.metric("Blue win probability", f"{prob_blue_xgb:.1%}")
            show_chart(render_gauge(prob_blue_xgb, "XGBoost", MODEL_COLORS["XGBoost"]), width="stretch")
    else:
        with m1:
            st.warning("`xgb_model.json` not found. Run `xgboost_train.py` first.")

    if logreg is not None:
        feature_names = list(logreg["scaler"].feature_names_in_)
        row = build_feature_vector(feature_names, inputs)
        X_row = row.to_frame().T[feature_names]

        imputer = logreg.get("imputer")
        if imputer is not None:
            X_row = pd.DataFrame(imputer.transform(X_row), columns=feature_names)

        scaled = logreg["scaler"].transform(X_row)
        prob_blue_logreg = float(logreg["model"].predict_proba(scaled)[0, 1])
        with m2:
            st.subheader("Logistic Regression")
            st.metric("Blue win probability", f"{prob_blue_logreg:.1%}")
            show_chart(render_gauge(prob_blue_logreg, "Logistic Regression", MODEL_COLORS["LogisticRegression"]), width="stretch")
    else:
        with m2:
            st.warning("`logreg_model.joblib` not found. Run `logistic_regression_train.py` first.")


def render_explore_game():
    st.header("Explore a Game")
    real_tab, custom_tab = st.tabs(["Real Match", "Custom Scenario"])
    with real_tab:
        render_real_match_tab()
    with custom_tab:
        render_custom_scenario_tab()


ABLATION_VARIANT_LABELS = {
    "full": "Full features",
    "no_economy": "No gold/XP/level",
    "no_objectives": "No objectives/towers",
    "no_structures": "No towers/plates",
    "no_epic_monsters": "No dragons/heralds/barons/elders",
    "only_economy": "Only gold/XP/level",
    "only_objectives": "Only objectives/towers",
    "only_structures": "Only towers/plates",
    "only_epic_monsters": "Only dragons/heralds/barons/elders",
}


def render_ablation_headline():
    st.subheader("Does the win-probability signal come from gold, or objectives?")
    results = load_csv_if_exists(ABLATION_RESULTS_FILE)
    bootstrap = load_csv_if_exists(ABLATION_BOOTSTRAP_FILE)

    if results is None:
        st.info("Run `ablation.py` to generate `ablation_results.csv`.")
        return

    variant_order = ["full", "no_economy", "no_objectives", "no_structures", "no_epic_monsters"]
    bootstrap_by_key = (
        {(r["model"], r["ablation"]): r for _, r in bootstrap.iterrows()} if bootstrap is not None else {}
    )

    fig = go.Figure()
    for model_name, group in results.groupby("model"):
        group = group.set_index("variant").reindex(variant_order)
        alphas = []
        for v in variant_order:
            b = bootstrap_by_key.get((model_name, v))
            significant = v == "full" or b is None or not (b["ci_low"] <= 0 <= b["ci_high"])
            alphas.append(SIGNIFICANT_ALPHA if significant else NOT_SIGNIFICANT_ALPHA)
        fig.add_trace(go.Bar(
            x=[ABLATION_VARIANT_LABELS[v] for v in variant_order],
            y=group["auc"],
            name=model_name,
            text=group["auc"].map(lambda v: f"{v:.4f}" if pd.notna(v) else ""),
            textposition="outside",
            marker=dict(color=MODEL_COLORS.get(model_name), opacity=alphas),
        ))
    fig.update_layout(
        barmode="group", yaxis_title="Test AUC",
        yaxis=dict(range=[0.6, 0.9]),
        title="Dropping one feature group at a time (faded = not distinguishable from full model at 95% CI)",
        height=440,
        margin=dict(t=90),
    )
    show_chart(fig, width="stretch")

    only_variant_order = ["only_economy", "only_objectives", "only_structures", "only_epic_monsters"]
    if set(only_variant_order) & set(results["variant"].unique()):
        st.markdown("**How well does each feature group predict on its own?** *(axis zoomed to the data)*")
        fig2 = go.Figure()
        for model_name, group in results.groupby("model"):
            group = group.set_index("variant").reindex(only_variant_order)
            fig2.add_trace(go.Bar(
                x=[ABLATION_VARIANT_LABELS[v] for v in only_variant_order],
                y=group["auc"],
                name=model_name,
                text=group["auc"].map(lambda v: f"{v:.4f}" if pd.notna(v) else ""),
                textposition="outside",
                marker_color=MODEL_COLORS.get(model_name),
            ))
        fig2.update_layout(
            barmode="group", yaxis_title="Test AUC",
            yaxis=dict(range=[0.6, 0.9]),
            height=400,
            margin=dict(t=50),
        )
        show_chart(fig2, width="stretch")

    if bootstrap is not None:
        display = bootstrap.copy()
        display["ablation"] = display["ablation"].map(ABLATION_VARIANT_LABELS).fillna(display["ablation"])
        display["95% CI"] = display.apply(lambda r: f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1)
        with st.expander("Match-level bootstrap AUC gap from each ablation vs. the full model (95% CI)"):
            st.dataframe(
                display[["model", "ablation", "mean_gap", "95% CI"]].rename(
                    columns={"model": "Model", "ablation": "Variant", "mean_gap": "Mean AUC gap"}
                ),
                width="stretch",
            )


def render_rank_breakdown():
    st.subheader("Does objective control matter more at some ranks than others?")
    rank_df = load_csv_if_exists(ABLATION_RANK_FILE)
    if rank_df is None:
        st.info("Run `ablation.py` to generate `ablation_rank_breakdown.csv`.")
        return

    available = [v for v in ("no_objectives", "only_objectives") if v in rank_df["ablation"].unique()]
    if not available:
        st.info("No recognized ablation labels found in `ablation_rank_breakdown.csv`.")
        return

    chosen = st.radio(
        "Variant",
        available,
        format_func=lambda v: ABLATION_VARIANT_LABELS.get(v, v),
        horizontal=True,
        key="rank_breakdown_variant",
    )
    subset = rank_df[rank_df["ablation"] == chosen]
    ranks = order_ranks(subset["rank"].unique())

    MODEL_SYMBOLS = {"XGBoost": "circle", "LogisticRegression": "square"}
    MODEL_DASH = {"XGBoost": "solid", "LogisticRegression": "dash"}

    fig = go.Figure()
    for model_name, group in subset.groupby("model"):
        group = group.set_index("rank").reindex(ranks)
        not_sig = (group["ci_low"] <= 0) & (0 <= group["ci_high"])
        color = MODEL_COLORS.get(model_name)
        symbol = MODEL_SYMBOLS.get(model_name, "circle")

        fig.add_trace(go.Scatter(
            x=ranks, y=group["auc_gap"], mode="lines", name=model_name,
            line=dict(color=color, width=1.5, dash=MODEL_DASH.get(model_name, "solid")),
            opacity=0.6, showlegend=False, hoverinfo="skip",
        ))
        marker_opacity = [NOT_SIGNIFICANT_ALPHA if ns else SIGNIFICANT_ALPHA for ns in not_sig]
        fig.add_trace(go.Scatter(
            x=ranks, y=group["auc_gap"], mode="markers", name=model_name,
            marker=dict(color=color, size=10, symbol=symbol, opacity=marker_opacity,
                        line=dict(color=color, width=1)),
            hovertemplate="%{x}<br>AUC gap: %{y:.4f}<extra>" + model_name + "</extra>",
        ))

    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        xaxis_title="Rank",
        yaxis_title=f"AUC gap from removing {ABLATION_VARIANT_LABELS.get(chosen, chosen).lower()}"
        if chosen.startswith("no_") else f"AUC gap ({ABLATION_VARIANT_LABELS.get(chosen, chosen).lower()})",
        yaxis=dict(tickformat=".3f"),
        title="Faded/hollow-looking points: 95% CI includes zero. Circle = XGBoost, square = Logistic Regression.",
        height=440,
    )
    show_chart(fig, width="stretch")
    with st.expander("Full rank breakdown table"):
        st.dataframe(subset, width="stretch")


CLOSENESS_ABLATION_LABELS = {
    "no_objectives": "objectives/towers",
    "no_structures": "towers/plates (structures / map control)",
}


def render_closeness_breakdown():
    st.subheader("Does objective control matter more in close games?")
    closeness = load_csv_if_exists(ABLATION_CLOSENESS_FILE)
    if closeness is None:
        st.info("Run `ablation.py` to generate `ablation_closeness_breakdown.csv`.")
        return

    available = [v for v in CLOSENESS_ABLATION_LABELS if v in closeness["ablation"].unique()]
    if not available:
        st.info("No recognized ablation labels found in `ablation_closeness_breakdown.csv`.")
        return

    bucket_order = ["close", "medium", "blowout"]

    for variant in available:
        subset = closeness[closeness["ablation"] == variant]

        st.markdown(f"**Feature group removed: {CLOSENESS_ABLATION_LABELS[variant]}**")

        fig = go.Figure()
        for model_name, group in subset.groupby("model"):
            group = group.set_index("bucket").reindex(bucket_order)
            fig.add_trace(go.Bar(
                x=bucket_order, y=group["auc_gap"], name=model_name,
                marker_color=MODEL_COLORS.get(model_name),
                text=group["auc_gap"].map(lambda v: f"{v:+.4f}" if pd.notna(v) else ""),
                textposition="outside",
            ))
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(
            barmode="group",
            xaxis_title="Game closeness (close <2.5k, medium 2.5k-7.5k, blowout >7.5k gold)",
            yaxis_title="AUC gap",
            yaxis=dict(tickformat=".3f", automargin=True),
            xaxis=dict(automargin=True),
            height=360,
            margin=dict(l=60, r=20, t=20, b=60),
        )
        show_chart(fig, width="stretch")

    with st.expander("Full closeness breakdown table"):
        st.dataframe(closeness, width="stretch")


def render_calibration_and_importance():
    st.subheader("Calibration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost**")
        if XGB_CALIBRATION_IMG.exists():
            st.image(str(XGB_CALIBRATION_IMG), width="stretch")
        else:
            st.info("Run `xgboost_train.py` to generate `xgb_calibration_curve.png`.")
    with col2:
        st.markdown("**Logistic Regression**")
        if LOGREG_CALIBRATION_IMG.exists():
            st.image(str(LOGREG_CALIBRATION_IMG), width="stretch")
        else:
            st.info("Run `logistic_regression_train.py` to generate `logreg_calibration_curve.png`.")

    st.subheader("Feature importance")
    top_n = st.slider("Show top N features", 5, 50, 20)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost (gain)**")
        imp_df = load_csv_if_exists(XGB_FEATURE_IMPORTANCE_CSV)
        if imp_df is not None:
            top = imp_df.sort_values("gain", ascending=False).head(top_n).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["gain"], y=top["feature"], orientation="h", marker_color=MODEL_COLORS["XGBoost"]))
            fig.update_layout(height=max(400, 22 * len(top)), margin=dict(l=220))
            show_chart(fig, width="stretch")
        else:
            st.info("Run `xgboost_train.py` to generate `xgb_feature_importances.csv`.")
    with col2:
        st.markdown("**Logistic Regression (|coefficient|)**")
        imp_df = load_csv_if_exists(LOGREG_FEATURE_IMPORTANCE_CSV)
        if imp_df is not None:
            top = imp_df.sort_values("abs_coefficient", ascending=False).head(top_n).iloc[::-1]
            colors = ["#2ca02c" if c >= 0 else "#d62728" for c in top["coefficient"]]
            fig = go.Figure(go.Bar(x=top["coefficient"], y=top["feature"], orientation="h", marker_color=colors))
            fig.update_layout(height=max(400, 22 * len(top)), margin=dict(l=220))
            show_chart(fig, width="stretch")
        else:
            st.info("Run `logistic_regression_train.py` to generate `logreg_feature_importances.csv`.")


def render_minute_bucket_comparison():
    st.subheader("Performance by minute bucket")
    xgb_pred = load_xgb_predictions()
    logreg_pred = load_logreg_predictions()

    if xgb_pred is None or logreg_pred is None:
        st.info("Need both `xgb_predictions.parquet` and `logreg_predictions.parquet` for this comparison.")
        return

    xgb_buckets = evaluate_by_minute_bucket(xgb_pred, prob_col="pred_prob_team_100_win", buckets=MINUTE_BUCKETS)
    xgb_bucket_df = pd.DataFrame(xgb_buckets).rename(columns=lambda c: f"xgb_{c}" if c != "minutes" else c)

    logreg_buckets = evaluate_by_minute_bucket(logreg_pred, prob_col="pred_prob_team_100_win", buckets=MINUTE_BUCKETS)
    logreg_bucket_df = pd.DataFrame(logreg_buckets).rename(columns=lambda c: f"logreg_{c}" if c != "minutes" else c)

    combined = xgb_bucket_df.merge(logreg_bucket_df, on="minutes", how="outer")

    bucket_labels = [f"{start}-{end}" for start, end in MINUTE_BUCKETS]
    combined = combined.set_index("minutes").reindex(bucket_labels).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined["minutes"], y=combined["xgb_auc"], name="XGBoost AUC", mode="lines+markers",
        line=dict(color=MODEL_COLORS["XGBoost"]),
        marker=dict(symbol="circle", size=9, color=MODEL_COLORS["XGBoost"]),
        hovertemplate="%{x}<br>XGBoost AUC: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=combined["minutes"], y=combined["logreg_auc"], name="Logistic Regression AUC", mode="lines+markers",
        line=dict(color=MODEL_COLORS["LogisticRegression"], dash="dash"),
        marker=dict(symbol="square", size=9, color=MODEL_COLORS["LogisticRegression"]),
        hovertemplate="%{x}<br>Logistic Regression AUC: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="AUC", xaxis_title="Game minute bucket",
        yaxis=dict(range=[0.6, 0.95], tickformat=".3f"),
        xaxis=dict(type="category", categoryorder="array", categoryarray=bucket_labels),
        height=400,
    )
    show_chart(fig, width="stretch")

    with st.expander("Full minute-bucket breakdown table"):
        st.dataframe(
            combined[["minutes", "xgb_rows", "xgb_auc", "xgb_log_loss", "xgb_accuracy", "xgb_brier",
                      "logreg_rows", "logreg_auc", "logreg_log_loss", "logreg_accuracy", "logreg_brier"]]
            .round(4),
            width="stretch",
        )


def render_pipeline_details():
    with st.expander("Pipeline details: hyperparameter search"):
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("**XGBoost**")
            xgb_grid = load_csv_if_exists(XGB_GRID_RESULTS)
            if xgb_grid is not None:
                st.dataframe(xgb_grid.sort_values("val_log_loss").reset_index(drop=True), width="stretch")
            else:
                st.info("`xgb_grid_search_results.csv` not found.")
        with gcol2:
            st.markdown("**Logistic Regression**")
            logreg_grid = load_csv_if_exists(LOGREG_GRID_RESULTS)
            if logreg_grid is not None:
                st.dataframe(logreg_grid.sort_values("val_log_loss").reset_index(drop=True), width="stretch")
            else:
                st.info("`logreg_grid_search_results.csv` not found.")


def render_findings():
    st.header("Findings")
    render_ablation_headline()
    st.divider()
    render_closeness_breakdown()
    st.divider()
    render_rank_breakdown()
    st.divider()
    render_calibration_and_importance()
    st.divider()
    render_minute_bucket_comparison()
    st.divider()
    render_pipeline_details()


def main():
    st.set_page_config(page_title="LoL Win Probability Model", layout="wide")
    st.title("League of Legends Win-Probability Model")

    page = st.sidebar.radio("Page", ["Explore a Game", "Findings"])

    if page == "Explore a Game":
        render_explore_game()
    else:
        render_findings()


if __name__ == "__main__":
    main()
