import csv
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import (
    RANDOM_STATE,
    SPLIT_FILE,
    VAL_SIZE,
    compute_metrics,
    evaluate_by_minute_bucket,
    infer_monotone_constraints,
    print_bucket_rows,
    print_metrics,
)


ECONOMY_KEYWORDS = ["gold", "xp", "level"]
OBJECTIVE_KEYWORDS = ["dragon", "herald", "baron", "elder", "plate", "tower", "destroyed"]
STRUCTURE_KEYWORDS = ["tower", "destroyed", "plate"]
EPIC_MONSTER_KEYWORDS = ["dragon", "herald", "baron", "elder"]

XGB_INPUT_FILE = Path("xgb_engineered/xgb_clean_dataset.parquet")

RESULTS_FILE = Path("results/ablation_results.csv")
BOOTSTRAP_RESULTS_FILE = Path("results/ablation_bootstrap_cis.csv")
CLOSENESS_RESULTS_FILE = Path("results/ablation_closeness_breakdown.csv")

CLOSENESS_FEATURE = "team_total_gold_diff"
CLOSENESS_BUCKET_LABELS = ["close", "medium", "blowout"]

N_BOOTSTRAP = 2000

XGB_CONFIG = {"max_depth": 6, "learning_rate": 0.03, "subsample": 1.0}
XGB_COLSAMPLE_BYTREE = 0.85
XGB_MAX_ESTIMATORS = 600
XGB_EARLY_STOPPING_ROUNDS = 30

LOGREG_CONFIG = {"C": 1.0}
LOGREG_PENALTY = "l2"
LOGREG_MAX_ITER = 1000


def is_economy_col(name):
    lname = name.lower()
    return any(kw in lname for kw in ECONOMY_KEYWORDS)


def is_objective_col(name):
    lname = name.lower()
    return any(kw in lname for kw in OBJECTIVE_KEYWORDS)


def is_structure_col(name):
    lname = name.lower()
    return any(kw in lname for kw in STRUCTURE_KEYWORDS)


def is_epic_monster_col(name):
    lname = name.lower()
    return any(kw in lname for kw in EPIC_MONSTER_KEYWORDS)


VARIANTS = {
    "full": lambda name: False,
    "no_economy": is_economy_col,
    "no_objectives": is_objective_col,
    "no_structures": is_structure_col,
    "no_epic_monsters": is_epic_monster_col,
}

ABLATION_DROP_LABELS = {
    "no_economy": "gold/XP/level",
    "no_objectives": "objectives/towers",
    "no_structures": "towers/plates",
    "no_epic_monsters": "dragons/heralds/barons/elders",
}

CLOSENESS_VARIANTS = ("no_objectives", "no_structures")


def load_split(match_ids):
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SPLIT_FILE}. Run xgboost_train.py (or "
            "logistic_regression_train.py) first so the shared split exists "
            "-- this ablation reuses it rather than creating a new one, so "
            "its results stay comparable to the main report."
        )

    data = np.load(SPLIT_FILE, allow_pickle=True)
    return data["train_ids"].astype(str), data["test_ids"].astype(str)


def get_xy(df, drop_cols):
    y = df["target"].astype(int)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y


def load_splits_for_variant(drop_predicate):
    df = pd.read_parquet(XGB_INPUT_FILE)
    df["match_id"] = df["match_id"].astype(str)

    train_ids, test_ids = load_split(df["match_id"])

    fit_ids, val_ids = train_test_split(
        train_ids, test_size=VAL_SIZE, random_state=RANDOM_STATE,
    )
    fit_ids, val_ids, test_ids = set(fit_ids), set(val_ids), set(test_ids)

    fit_df = df[df["match_id"].isin(fit_ids)].copy()
    val_df = df[df["match_id"].isin(val_ids)].copy()
    test_df = df[df["match_id"].isin(test_ids)].copy()

    drop_cols = ["match_id", "target"] + [c for c in df.columns if drop_predicate(c)]

    X_fit, y_fit = get_xy(fit_df, drop_cols)
    X_val, y_val = get_xy(val_df, drop_cols)
    X_test, y_test = get_xy(test_df, drop_cols)

    return (X_fit, y_fit), (X_val, y_val), (X_test, y_test, test_df)


def build_pred_df(test_df, probs):
    pred_df = test_df[["match_id", "minute", "target"]].copy()
    pred_df["prob"] = probs
    pred_df["abs_gold_diff"] = (
        test_df[CLOSENESS_FEATURE].abs().to_numpy()
        if CLOSENESS_FEATURE in test_df.columns else np.nan
    )
    return pred_df


def fit_xgb(X_fit, y_fit, X_val, y_val, X_test):
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=XGB_MAX_ESTIMATORS,
        max_depth=XGB_CONFIG["max_depth"],
        learning_rate=XGB_CONFIG["learning_rate"],
        subsample=XGB_CONFIG["subsample"],
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        monotone_constraints=infer_monotone_constraints(X_fit.columns),
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    probs = model.predict_proba(X_test)[:, 1]
    return probs, {"best_iteration": model.best_iteration}


def fit_logreg(X_fit, y_fit, X_val, y_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_fit)

    model = LogisticRegression(
        penalty=LOGREG_PENALTY,
        C=LOGREG_CONFIG["C"],
        max_iter=LOGREG_MAX_ITER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(scaler.transform(X_fit), y_fit)
    probs = model.predict_proba(scaler.transform(X_test))[:, 1]
    return probs, {}


MODEL_FITTERS = {
    "XGBoost": fit_xgb,
    "LogisticRegression": fit_logreg,
}


def run_variant(model_name, label, drop_predicate):
    print("\n" + "=" * 80)
    print(f"{model_name} -- {label}")
    print("=" * 80)

    (X_fit, y_fit), (X_val, y_val), (X_test, y_test, test_df) = load_splits_for_variant(drop_predicate)
    print("Features used:", X_fit.shape[1])

    probs, extra = MODEL_FITTERS[model_name](X_fit, y_fit, X_val, y_val, X_test)

    metrics = compute_metrics(y_test, probs)
    print_metrics(f"{model_name} test metrics ({label})", metrics)

    pred_df = build_pred_df(test_df, probs)
    bucket_rows = evaluate_by_minute_bucket(pred_df, prob_col="prob")
    print(f"\nTest metrics by minute bucket ({label})")
    print_bucket_rows(bucket_rows)

    baseline = max(y_fit.mean(), 1 - y_fit.mean())

    return {
        "model": model_name,
        "variant": label,
        "n_features": X_fit.shape[1],
        "baseline_accuracy": round(float(baseline), 4),
        **{k: round(float(v), 4) for k, v in metrics.items() if k != "rows"},
        "bucket_rows": bucket_rows,
        "predictions": pred_df,
        **extra,
    }


def bootstrap_auc_gap(merged, prob_col_full, prob_col_ablated, target_col="target",
                       match_col="match_id", n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    merged = merged.reset_index(drop=True)
    groups = merged.groupby(match_col).indices
    match_list = np.array(list(groups.keys()))

    full_vals = merged[prob_col_full].to_numpy()
    ablated_vals = merged[prob_col_ablated].to_numpy()
    targets = merged[target_col].to_numpy()

    rng = np.random.default_rng(seed)
    gaps = []

    for _ in range(n_boot):
        sampled_matches = rng.choice(match_list, size=len(match_list), replace=True)
        idx = np.concatenate([groups[m] for m in sampled_matches])

        y = targets[idx]
        if len(np.unique(y)) < 2:
            continue

        gaps.append(roc_auc_score(y, full_vals[idx]) - roc_auc_score(y, ablated_vals[idx]))

    gaps = np.array(gaps)

    return {
        "mean_gap": float(gaps.mean()),
        "ci_low": float(np.percentile(gaps, 2.5)),
        "ci_high": float(np.percentile(gaps, 97.5)),
        "n_boot": int(len(gaps)),
    }


def closeness_breakdown(merged, prob_col_full, prob_col_ablated, ablation_label, target_col="target",
                         closeness_col="abs_gold_diff", labels=CLOSENESS_BUCKET_LABELS):
    merged = merged.dropna(subset=[closeness_col]).copy()
    if merged.empty:
        return []

    merged["closeness_bucket"] = pd.qcut(merged[closeness_col], q=len(labels), labels=labels, duplicates="drop")

    rows = []
    for label in labels:
        g = merged[merged["closeness_bucket"] == label]
        y = g[target_col].to_numpy()

        if g.empty or len(np.unique(y)) < 2:
            continue

        auc_full = roc_auc_score(y, g[prob_col_full].to_numpy())
        auc_ablated = roc_auc_score(y, g[prob_col_ablated].to_numpy())

        rows.append({
            "ablation": ablation_label,
            "bucket": label,
            "rows": len(g),
            "median_abs_gold_diff": round(float(g[closeness_col].median()), 1),
            "auc_full": round(float(auc_full), 4),
            "auc_ablated": round(float(auc_ablated), 4),
            "auc_gap": round(float(auc_full - auc_ablated), 4),
            "auc_retained_pct": round(float(auc_ablated / auc_full), 4) if auc_full else float("nan"),
        })

    return rows


def save_results_csv(results):
    rows = [{k: v for k, v in r.items() if k != "bucket_rows"} for r in results]
    fieldnames = sorted({k for row in rows for k in row})
    ordered = ["model", "variant", "n_features"] + [f for f in fieldnames if f not in ("model", "variant", "n_features")]

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def save_dict_rows_csv(rows, path):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results):
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY: full feature set vs. gold/XP/level removed vs. objectives removed")
    print("=" * 80)

    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], {})[r["variant"]] = r

    for model_name, variants in by_model.items():
        full = variants.get("full")
        if not full:
            continue

        print(f"\n{model_name}:")
        print(f"  Baseline accuracy (majority class): {full['baseline_accuracy']:.4f}")
        print(f"  Full features     ({full['n_features']:>3} feats): AUC={full['auc']:.4f}  log_loss={full['log_loss']:.4f}  accuracy={full['accuracy']:.4f}  brier={full['brier']:.4f}")

        for variant_key, drop_label in ABLATION_DROP_LABELS.items():
            ablated = variants.get(variant_key)
            if not ablated:
                continue

            print(f"  {variant_key:<17} ({ablated['n_features']:>3} feats): AUC={ablated['auc']:.4f}  log_loss={ablated['log_loss']:.4f}  accuracy={ablated['accuracy']:.4f}  brier={ablated['brier']:.4f}")
            print(f"    AUC retained without {drop_label}: {ablated['auc'] / full['auc']:.1%}  (gap: {full['auc'] - ablated['auc']:.4f})")


def print_closeness_rows(model_name, variant_label, rows):
    print(f"\n{model_name} -- {variant_label} AUC gap by game closeness (|{CLOSENESS_FEATURE}|):")
    for row in rows:
        print(
            f"  {row['bucket']:<8} | rows: {row['rows']:>6} | median |gold diff|: {row['median_abs_gold_diff']:>8} "
            f"| auc_full: {row['auc_full']:.4f} | auc_ablated: {row['auc_ablated']:.4f} "
            f"| gap: {row['auc_gap']:.4f} | retained: {row['auc_retained_pct']:.1%}"
        )


def main():
    results = []
    predictions = {}

    for model_name in MODEL_FITTERS:
        for variant_label, predicate in VARIANTS.items():
            r = run_variant(model_name, variant_label, predicate)
            predictions[(model_name, variant_label)] = r.pop("predictions")
            results.append(r)

    save_results_csv(results)
    print("\nSaved:", RESULTS_FILE)
    print_summary(results)

    print("\n" + "=" * 80)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS (match-level, n_boot={N_BOOTSTRAP})")
    print("=" * 80)

    bootstrap_rows = []
    closeness_rows = []

    for model_name in MODEL_FITTERS:
        full_pred = predictions[(model_name, "full")]

        for variant_label, drop_label in ABLATION_DROP_LABELS.items():
            ablated_pred = predictions[(model_name, variant_label)]

            merged = full_pred.merge(
                ablated_pred[["match_id", "minute", "prob"]],
                on=["match_id", "minute"],
                suffixes=("_full", "_ablated"),
            )

            ci = bootstrap_auc_gap(merged, "prob_full", "prob_ablated")
            print(
                f"\n{model_name} -- AUC gap from removing {drop_label}: "
                f"{ci['mean_gap']:.4f}  (95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}], n_boot={ci['n_boot']})"
            )
            bootstrap_rows.append({"model": model_name, "ablation": variant_label, **ci})

            if variant_label in CLOSENESS_VARIANTS:
                rows = closeness_breakdown(merged, "prob_full", "prob_ablated", variant_label)
                print_closeness_rows(model_name, variant_label, rows)
                closeness_rows.extend({"model": model_name, **row} for row in rows)

    save_dict_rows_csv(bootstrap_rows, BOOTSTRAP_RESULTS_FILE)
    print("\nSaved:", BOOTSTRAP_RESULTS_FILE)

    save_dict_rows_csv(closeness_rows, CLOSENESS_RESULTS_FILE)
    print("Saved:", CLOSENESS_RESULTS_FILE)


if __name__ == "__main__":
    main()
