import csv
import itertools
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import (
    RANDOM_STATE,
    SPLIT_FILE,
    VAL_SIZE,
    compute_metrics,
    evaluate_by_minute_bucket,
    print_bucket_rows,
    print_metrics,
    save_calibration_plot,
)


INPUT_FILE = Path("xgb_engineered/xgb_clean_dataset.parquet")

MODEL_FILE = Path("models/logreg_model.joblib")
PREDICTIONS_FILE = Path("results/logreg_predictions.parquet")
CALIBRATION_PLOT_FILE = Path("figures/logreg_calibration_curve.png")
RESULTS_FILE = Path("results/logreg_grid_search_results.csv")
FEATURE_IMPORTANCE_CSV = Path("results/logreg_feature_importances.csv")
FEATURE_IMPORTANCE_PLOT = Path("figures/logreg_feature_importance.png")
TOP_N_FEATURES_PLOTTED = 25

TEST_SIZE = 0.20
MAX_ITER = 1000

GRID = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
}
PENALTY = "l2"


def load_or_create_split(match_ids):
    match_ids = np.array(sorted(pd.Series(match_ids).astype(str).unique()))

    if SPLIT_FILE.exists():
        print("Using existing split:", SPLIT_FILE)

        data = np.load(SPLIT_FILE, allow_pickle=True)

        train_ids = data["train_ids"].astype(str)
        test_ids = data["test_ids"].astype(str)

        return train_ids, test_ids

    print("Creating split:", SPLIT_FILE)

    train_ids, test_ids = train_test_split(
        match_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        SPLIT_FILE,
        train_ids=train_ids,
        test_ids=test_ids,
    )

    return train_ids, test_ids


def get_xy(df):
    y = df["target"].astype(int)

    drop_cols = [
        "match_id",
        "target",
        "timestamp_sec",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y


def config_iterator():
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def train_one_config(config, X_fit, y_fit, X_val, y_val, scaler):
    X_fit_scaled = scaler.transform(X_fit)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        penalty=PENALTY,
        C=config["C"],
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_fit_scaled, y_fit)

    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    val_metrics = compute_metrics(y_val, val_probs)

    return {
        "model": model,
        "val_auc": val_metrics["auc"],
        "val_log_loss": val_metrics["log_loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_brier": val_metrics["brier"],
    }


def save_results_csv(results):
    if not results:
        return

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def save_feature_importances(model, feature_names, csv_path, plot_path, top_n=TOP_N_FEATURES_PLOTTED):
    coefs = model.coef_[0]

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    imp_df.to_csv(csv_path, index=False)

    top = imp_df.head(top_n).iloc[::-1]
    colors = ["#1f77b4" if c >= 0 else "#d62728" for c in top["coefficient"]]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(top))))
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.set_xlabel("Standardized coefficient (blue = raises P(team 100 wins), red = lowers it)")
    ax.set_title(f"Top {len(top)} logistic regression coefficients (by magnitude)")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return imp_df


def main():
    print("Reading:", INPUT_FILE)

    df = pd.read_parquet(INPUT_FILE)

    required = {"match_id", "target", "minute"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["match_id"] = df["match_id"].astype(str)

    train_ids, test_ids = load_or_create_split(df["match_id"])

    fit_ids, val_ids = train_test_split(
        train_ids,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    fit_ids = set(fit_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)

    fit_df = df[df["match_id"].isin(fit_ids)].copy()
    val_df = df[df["match_id"].isin(val_ids)].copy()
    test_df = df[df["match_id"].isin(test_ids)].copy()

    if fit_df.empty:
        raise ValueError("Training data is empty.")

    if val_df.empty:
        raise ValueError("Validation data is empty.")

    if test_df.empty:
        raise ValueError("Test data is empty.")

    X_fit, y_fit = get_xy(fit_df)
    X_val, y_val = get_xy(val_df)
    X_test, y_test = get_xy(test_df)

    print("Rows:", len(df))
    print("Matches:", df["match_id"].nunique())
    print("Train matches:", fit_df["match_id"].nunique())
    print("Val matches:", val_df["match_id"].nunique())
    print("Test matches:", test_df["match_id"].nunique())
    print("Features:", X_fit.shape[1])

    baseline = max(y_fit.mean(), 1 - y_fit.mean())

    print()
    print("Majority-class baseline accuracy:", round(float(baseline), 4))

    scaler = StandardScaler()
    scaler.fit(X_fit)

    configs = list(config_iterator())
    print()
    print("Total configs:", len(configs))

    results = []

    best_model = None
    best_config = None
    best_val_log_loss = float("inf")

    for i, config in enumerate(configs, start=1):
        print("\n" + "=" * 80)
        print(f"Config {i}/{len(configs)}")
        print(config)

        result = train_one_config(config, X_fit, y_fit, X_val, y_val, scaler)

        print(
            f"val_auc={result['val_auc']:.4f} "
            f"val_log_loss={result['val_log_loss']:.4f} "
            f"val_accuracy={result['val_accuracy']:.4f} "
            f"val_brier={result['val_brier']:.4f}"
        )

        row = {
            **config,
            "penalty": PENALTY,
            "val_auc": result["val_auc"],
            "val_log_loss": result["val_log_loss"],
            "val_accuracy": result["val_accuracy"],
            "val_brier": result["val_brier"],
        }

        results.append(row)
        save_results_csv(results)

        if result["val_log_loss"] < best_val_log_loss:
            best_val_log_loss = result["val_log_loss"]
            best_model = result["model"]
            best_config = config

    print("\n" + "=" * 80)
    print("Best config")
    print(best_config)
    print("Best validation log loss:", round(float(best_val_log_loss), 4))
    print("Saved grid search results to:", RESULTS_FILE)

    print("\nFinal evaluation")

    importances = save_feature_importances(
        best_model, X_fit.columns, FEATURE_IMPORTANCE_CSV, FEATURE_IMPORTANCE_PLOT
    )
    print("Saved feature importances to:", FEATURE_IMPORTANCE_CSV)
    print("Saved feature importance chart to:", FEATURE_IMPORTANCE_PLOT)
    print()
    print(f"Top {min(10, len(importances))} features by |coefficient|:")
    for _, row in importances.head(10).iterrows():
        print(f"  {row['feature']:<45} coef={row['coefficient']:+.4f}")

    X_test_scaled = scaler.transform(X_test)
    probs = best_model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    overall = compute_metrics(y_test, probs)

    print_metrics("Test metrics across all valid minutes", overall)

    brier = save_calibration_plot(y_test, probs, CALIBRATION_PLOT_FILE, "Logistic regression win probability")
    print("Saved calibration curve to:", CALIBRATION_PLOT_FILE, f"(Brier={brier:.4f})")

    pred_df = test_df[["match_id", "minute", "target"]].copy()
    pred_df["pred_prob_team_100_win"] = probs
    pred_df["pred_label"] = preds
    pred_df = pred_df.sort_values(["match_id", "minute"])

    print()
    print("Test metrics by minute bucket")
    bucket_rows = evaluate_by_minute_bucket(pred_df, prob_col="pred_prob_team_100_win")
    print_bucket_rows(bucket_rows)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": best_model}, MODEL_FILE)
    pred_df.to_parquet(PREDICTIONS_FILE, index=False)

    print()
    print("Saved model to:", MODEL_FILE)
    print("Saved predictions to:", PREDICTIONS_FILE)
    print("Saved shared split to:", SPLIT_FILE)


if __name__ == "__main__":
    main()
