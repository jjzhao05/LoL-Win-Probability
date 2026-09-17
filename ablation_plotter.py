import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import MINUTE_BUCKETS, MODEL_COLORS, evaluate_by_minute_bucket


# Reads the CSVs ablation.py already writes -- no xgboost import, no
# retraining, no database. Run this after ablation.py, and again any time
# you just want to re-draw the figures (e.g. after tweaking a title)
# without repeating the ~expensive training run. The minute-bucket plot
# additionally reads the two training scripts' saved prediction files and
# needs common.py (and therefore sklearn) to recompute per-bucket metrics
# from them -- everything else here stays pure-CSV.
RESULTS_FILE = Path("results/ablation_results.csv")
BOOTSTRAP_RESULTS_FILE = Path("results/ablation_bootstrap_cis.csv")
CLOSENESS_RESULTS_FILE = Path("results/ablation_closeness_breakdown.csv")
RANK_RESULTS_FILE = Path("results/ablation_rank_breakdown.csv")

XGB_PREDICTIONS_FILE = Path("results/xgb_predictions.parquet")
LOGREG_PREDICTIONS_FILE = Path("results/logreg_predictions.parquet")
MINUTE_BUCKET_CSV = Path("results/minute_bucket_breakdown.csv")

FIGURES_DIR = Path("figures")
AUC_BY_VARIANT_PLOT = FIGURES_DIR / "ablation_auc_by_variant.png"
CLOSENESS_PLOT = FIGURES_DIR / "ablation_closeness_breakdown.png"
RANK_PLOT = FIGURES_DIR / "ablation_rank_breakdown.png"
MINUTE_BUCKET_PLOT = FIGURES_DIR / "auc_by_minute.png"

LOG_DIR = Path("logs")

MINUTE_BUCKET_LABELS = [f"{start}-{end}" for start, end in MINUTE_BUCKETS]

CLOSENESS_BUCKET_LABELS = ["close", "medium", "blowout"]
# Matches ablation.py's fixed CLOSENESS_BINS ([0, 2500, 7500, inf]) -- used
# only to label the x-axis with the actual thresholds.
CLOSENESS_BUCKET_TICK_LABELS = ["Close\n(<2.5k gold)", "Medium\n(2.5k-7.5k gold)", "Blowout\n(>7.5k gold)"]

# Riot's ranked tiers, low to high, matching collect_data.py's ten skill
# brackets. Duplicated from ablation.py rather than imported, so this
# script never pulls in xgboost/sklearn just to draw a chart.
RANK_ORDER = [
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM",
    "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER",
]

# A gap whose 95% CI is drawn faded/hollow rather than solid -- it means
# the CI on that gap includes zero, i.e. it isn't distinguishable from no
# difference at all at this sample size. This project's ablation gaps are
# almost all small in absolute terms (well under 0.02 AUC out of a 0.5-1.0
# scale), so it's easy for a bar chart or a connected line to make a
# noise-level wiggle look like a real effect. Fading/hollowing those points
# instead of just coloring every point the same is a deliberate choice to
# not overstate them.
SIGNIFICANT_ALPHA = 1.0
NOT_SIGNIFICANT_ALPHA = 0.35

# The variants the "economy vs objectives" headline plot shows -- full plus
# the four no_X/only_X variants the report text discusses. The other
# variants (no_structures, no_epic_monsters, only_structures,
# only_epic_monsters) are in ablation_results.csv but left off this
# particular chart to keep it readable.
AUC_PLOT_VARIANTS = ["full", "no_economy", "no_objectives", "only_economy", "only_objectives"]
AUC_PLOT_LABELS = {
    "full": "Full features",
    "no_economy": "No gold/XP/level",
    "no_objectives": "No objectives/towers",
    "only_economy": "Only gold/XP/level",
    "only_objectives": "Only objectives/towers",
}

CLOSENESS_PLOT_VARIANTS = ("no_objectives", "only_objectives")
RANK_PLOT_VARIANTS = ("no_objectives", "only_objectives")


def order_ranks(ranks):
    known = [r for r in RANK_ORDER if r in ranks]
    unknown = sorted(r for r in ranks if r not in RANK_ORDER)
    return known + unknown


def load_csv(path, required_for):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run ablation.py first to generate it "
            f"({required_for})."
        )
    return pd.read_csv(path)


def save_auc_by_variant_plot(results_df, bootstrap_df, out_path=AUC_BY_VARIANT_PLOT):
    """Grouped bar chart of raw test AUC for the full feature set plus the
    no_economy/no_objectives/only_economy/only_objectives variants. Error
    bars show each ablated bar's 95% bootstrap CI (translated from the CI
    on the full-vs-ablated AUC gap onto the ablated AUC itself); a bar is
    drawn faded when that CI includes zero, i.e. that variant's AUC isn't
    reliably different from the full model's."""
    by_model = {
        model: group.set_index("variant") for model, group in results_df.groupby("model")
    }
    bootstrap_by_key = {
        (row["model"], row["ablation"]): row for _, row in bootstrap_df.iterrows()
    }

    variants = [v for v in AUC_PLOT_VARIANTS if any(v in vs.index for vs in by_model.values())]
    models = list(by_model.keys())

    x = np.arange(len(variants))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(models):
        full_auc = by_model[model_name].loc["full", "auc"] if "full" in by_model[model_name].index else None
        offsets = x + (i - (len(models) - 1) / 2) * width

        aucs, yerr_low, yerr_high, alphas = [], [], [], []
        for v in variants:
            row = by_model[model_name].loc[v] if v in by_model[model_name].index else None
            auc = row["auc"] if row is not None else np.nan
            aucs.append(auc)

            b = bootstrap_by_key.get((model_name, v))
            significant = True
            if b is not None and v != "full" and full_auc is not None:
                ci_low_auc = full_auc - b["ci_high"]
                ci_high_auc = full_auc - b["ci_low"]
                yerr_low.append(max(0.0, auc - ci_low_auc))
                yerr_high.append(max(0.0, ci_high_auc - auc))
                significant = not (b["ci_low"] <= 0 <= b["ci_high"])
            else:
                yerr_low.append(0.0)
                yerr_high.append(0.0)
            alphas.append(SIGNIFICANT_ALPHA if significant else NOT_SIGNIFICANT_ALPHA)

        bars = ax.bar(offsets, aucs, width, label=model_name, color=MODEL_COLORS.get(model_name))
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
        ax.errorbar(offsets, aucs, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", elinewidth=1, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([AUC_PLOT_LABELS[v] for v in variants], rotation=20, ha="right")
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Test AUC by feature group (error bars: 95% bootstrap CI vs. full features)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_closeness_plot(closeness_df, out_path=CLOSENESS_PLOT):
    """One panel per variant in CLOSENESS_PLOT_VARIANTS: AUC gap from full
    features, grouped by model, across close/medium/blowout buckets. A bar
    is faded when its own bucket-level bootstrap CI includes zero."""
    if closeness_df.empty:
        return

    variants = [v for v in CLOSENESS_PLOT_VARIANTS if v in closeness_df["ablation"].unique()]
    if not variants:
        return

    fig, axes = plt.subplots(1, len(variants), figsize=(6 * len(variants), 5.5), sharey=False)
    if len(variants) == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        subset = closeness_df[closeness_df["ablation"] == variant]
        models = subset["model"].unique()

        x = np.arange(len(CLOSENESS_BUCKET_LABELS))
        width = 0.8 / max(len(models), 1)

        for i, model_name in enumerate(models):
            group = subset[subset["model"] == model_name].set_index("bucket").reindex(CLOSENESS_BUCKET_LABELS)
            offsets = x + (i - (len(models) - 1) / 2) * width

            yerr_low = (group["auc_gap"] - group["ci_low"]).clip(lower=0)
            yerr_high = (group["ci_high"] - group["auc_gap"]).clip(lower=0)
            not_sig = (group["ci_low"] <= 0) & (0 <= group["ci_high"])

            bars = ax.bar(offsets, group["auc_gap"], width, label=model_name, color=MODEL_COLORS.get(model_name))
            for bar, is_not_sig in zip(bars, not_sig):
                bar.set_alpha(NOT_SIGNIFICANT_ALPHA if is_not_sig else SIGNIFICANT_ALPHA)
            ax.errorbar(offsets, group["auc_gap"], yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", elinewidth=1, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(CLOSENESS_BUCKET_TICK_LABELS)
        ax.set_xlabel("Game closeness (fixed gold-lead thresholds)")
        ax.set_ylabel("AUC gap from full features")
        ax.set_title(AUC_PLOT_LABELS.get(variant, variant))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

    fig.suptitle("AUC gap by game closeness (with 95% bootstrap CI per bucket)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_rank_plot(rank_df, out_path=RANK_PLOT):
    """One panel per variant in RANK_PLOT_VARIANTS: AUC gap from full
    features, one line per model, across rank tiers Iron through
    Challenger. Points whose own rank-level bootstrap CI includes zero are
    drawn as small hollow markers instead of solid filled ones, so a
    noise-level wiggle at a given rank doesn't read the same as a real
    difference."""
    if rank_df.empty:
        return

    variants = [v for v in RANK_PLOT_VARIANTS if v in rank_df["ablation"].unique()]
    if not variants:
        return

    fig, axes = plt.subplots(1, len(variants), figsize=(7.5 * len(variants), 5.5), sharex=False)
    if len(variants) == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        subset = rank_df[rank_df["ablation"] == variant]
        ranks = order_ranks(subset["rank"].unique())

        for model_name, group in subset.groupby("model"):
            group = group.set_index("rank").reindex(ranks)
            color = MODEL_COLORS.get(model_name)

            ax.plot(ranks, group["auc_gap"], linewidth=1.5, alpha=0.6, color=color, zorder=1)

            not_sig = (group["ci_low"] <= 0) & (0 <= group["ci_high"])
            sig_mask = ~not_sig.to_numpy()

            xs = np.arange(len(ranks))
            ax.scatter(xs[sig_mask], group["auc_gap"].to_numpy()[sig_mask], color=color, marker="o", s=45,
                       label=model_name, zorder=3)
            ax.scatter(xs[~sig_mask], group["auc_gap"].to_numpy()[~sig_mask], facecolors="none", edgecolors=color,
                       marker="o", s=45, zorder=3)

        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels(ranks, rotation=30, ha="right")
        ax.set_ylabel("AUC gap from full features")
        ax.set_title(AUC_PLOT_LABELS.get(variant, variant))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle("AUC gap by rank (hollow points: 95% CI includes zero)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run xgboost_train.py / logistic_regression_train.py "
            "first to generate it (the minute-bucket plot)."
        )

    df = pd.read_parquet(path)

    required = {"minute", "target", "pred_prob_team_100_win"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {sorted(missing)}")

    return df


def compute_minute_bucket_table(pred_df, model_name):
    rows = evaluate_by_minute_bucket(pred_df, prob_col="pred_prob_team_100_win", buckets=MINUTE_BUCKETS)
    df = pd.DataFrame(rows)
    df.insert(0, "model", model_name)
    return df


def save_minute_bucket_plot(combined_df, out_path=MINUTE_BUCKET_PLOT):
    """Line chart of test AUC by minute bucket, one line per model. Unlike
    the ablation gap plots above, this is a plot of the metric itself (it
    swings from ~0.6 to ~0.91 across buckets), not a small difference
    between two conditions, so it doesn't need the same
    faded/not-distinguishable-from-zero treatment -- the fixed 0.5-1.0
    y-axis keeps it on the same honest scale as the other AUC figures."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model_name, group in combined_df.groupby("model"):
        group = group.set_index("minutes").reindex(MINUTE_BUCKET_LABELS)
        ax.plot(
            MINUTE_BUCKET_LABELS, group["auc"],
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
    results_df = load_csv(RESULTS_FILE, "the AUC-by-variant plot")
    bootstrap_df = load_csv(BOOTSTRAP_RESULTS_FILE, "the AUC-by-variant plot")
    closeness_df = load_csv(CLOSENESS_RESULTS_FILE, "the closeness plot")
    rank_df = load_csv(RANK_RESULTS_FILE, "the rank plot")

    if "ci_low" not in closeness_df.columns or "ci_low" not in rank_df.columns:
        raise ValueError(
            f"{CLOSENESS_RESULTS_FILE} / {RANK_RESULTS_FILE} are missing per-bucket/per-rank "
            "confidence interval columns. Re-run ablation.py to regenerate them with the "
            "current version of that script before plotting."
        )

    save_auc_by_variant_plot(results_df, bootstrap_df)
    print("Saved:", AUC_BY_VARIANT_PLOT)

    save_closeness_plot(closeness_df)
    print("Saved:", CLOSENESS_PLOT)

    save_rank_plot(rank_df)
    print("Saved:", RANK_PLOT)

    xgb_pred = load_predictions(XGB_PREDICTIONS_FILE)
    logreg_pred = load_predictions(LOGREG_PREDICTIONS_FILE)

    minute_bucket_df = pd.concat(
        [
            compute_minute_bucket_table(xgb_pred, "XGBoost"),
            compute_minute_bucket_table(logreg_pred, "LogisticRegression"),
        ],
        ignore_index=True,
    )

    MINUTE_BUCKET_CSV.parent.mkdir(parents=True, exist_ok=True)
    minute_bucket_df.round(4).to_csv(MINUTE_BUCKET_CSV, index=False)
    print("Saved:", MINUTE_BUCKET_CSV)

    save_minute_bucket_plot(minute_bucket_df)
    print("Saved:", MINUTE_BUCKET_PLOT)


class Tee:
    """Mirrors writes to every stream it wraps (e.g. the real console plus
    a log file), so redirecting sys.stdout/sys.stderr through one of these
    logs a full run without touching any of the print() calls above."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"ablation_plotter_run_{datetime.now():%Y%m%d_%H%M%S}.log"

    real_stdout, real_stderr = sys.stdout, sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_f:
        sys.stdout = Tee(real_stdout, log_f)
        sys.stderr = Tee(real_stderr, log_f)

        try:
            print(f"Logging full run output to: {log_path}")
            main()
        finally:
            sys.stdout = real_stdout
            sys.stderr = real_stderr
