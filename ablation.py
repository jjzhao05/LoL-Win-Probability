"""Ablation study: how much does the model actually depend on gold/XP/level?

Motivation: XGBoost's gain-based feature importance (see xgboost_train.py's
save_feature_importances) is dominated by gold/XP/level differentials -- the
top 10 features by gain are ~51% of total gain and are almost all economy
stats. That's a fair thing to be suspicious of: gain-based importance can't
tell "this feature matters" apart from "this feature is redundant with a
cleaner, more direct feature" -- and since almost everything in League of
Legends (kills, towers, objectives) converts into gold/XP eventually, a tree
model handed the gold/XP diff directly has little reason to also lean on the
noisier derived signals.

This script answers the honest version of that question directly: retrain
both models with every gold/XP/level feature removed, and see how much
predictive power survives on objectives/towers/kills/momentum alone. If
accuracy collapses to near-baseline, the model really is "just a gold
graph." If meaningful AUC survives, the non-economy features carry real
independent information that the full-feature importance ranking hides
through collinearity.

Both models are retrained twice each (full feature set vs. economy features
removed) using the same shared train/val/test split and the same
hyperparameters selected by the existing grid searches (xgboost_train.py /
lstm_train.py), so any performance gap is attributable to the ablation
itself and not to a different train/val split or a different config being
picked.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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


# Any feature whose name contains one of these substrings is considered an
# "economy" feature and is dropped in the no_economy ablation. This covers
# team and per-role gold/XP/level diffs, the derived per-minute rate
# features, and their 1-/3-minute momentum deltas (XGBoost side only -- the
# LSTM feature set has no momentum deltas).
ECONOMY_KEYWORDS = ["gold", "xp", "level"]

# Any feature whose name contains one of these substrings is considered an
# "objective" feature and is dropped in the no_objectives ablation: dragons,
# heralds, barons, elders, plates, every tower/lane-tower destroyed counter
# (including the nexus towers), the team-level tower totals/diff, and
# first_tower_diff/so_far (first_blood is deliberately NOT included here --
# a kill, not an objective).
OBJECTIVE_KEYWORDS = ["dragon", "herald", "baron", "elder", "plate", "tower", "destroyed"]

XGB_INPUT_FILE = Path("xgb_engineered/xgb_clean_dataset.parquet")
LSTM_INPUT_FILE = Path("results/lstm_data.npz")

RESULTS_FILE = Path("results/ablation_results.csv")

# Best configs selected by the existing grid searches (xgb_grid_search_results.csv
# / lstm_grid_search_results.csv) -- reused here rather than re-searching, so
# the ablation isolates "which features" rather than also varying "which
# hyperparameters."
XGB_CONFIG = {"max_depth": 6, "learning_rate": 0.03, "subsample": 1.0}
XGB_COLSAMPLE_BYTREE = 0.85
XGB_MAX_ESTIMATORS = 600
XGB_EARLY_STOPPING_ROUNDS = 30

LSTM_CONFIG = {"hidden_size": 128, "num_layers": 2, "dropout": 0.10, "weight_decay": 1e-4}
LSTM_BATCH_SIZE = 128
LSTM_MAX_EPOCHS = 15
LSTM_PATIENCE = 3
LSTM_LEARNING_RATE = 0.001


def is_economy_col(name):
    lname = name.lower()
    return any(kw in lname for kw in ECONOMY_KEYWORDS)


def is_objective_col(name):
    lname = name.lower()
    return any(kw in lname for kw in OBJECTIVE_KEYWORDS)


# variant label -> predicate deciding whether a feature is DROPPED for that variant
VARIANTS = {
    "full": lambda name: False,
    "no_economy": is_economy_col,
    "no_objectives": is_objective_col,
}


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def load_xgb_split(match_ids):
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SPLIT_FILE}. Run xgboost_train.py first so the shared "
            "split exists -- this ablation reuses it rather than creating a "
            "new one, so its results stay comparable to the main report."
        )

    data = np.load(SPLIT_FILE, allow_pickle=True)
    train_ids = data["train_ids"].astype(str)
    test_ids = data["test_ids"].astype(str)

    return train_ids, test_ids


def get_xy(df, drop_cols):
    y = df["target"].astype(int)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y


def run_xgb(label, drop_predicate):
    print("\n" + "=" * 80)
    print(f"XGBoost -- {label}")
    print("=" * 80)

    df = pd.read_parquet(XGB_INPUT_FILE)
    df["match_id"] = df["match_id"].astype(str)

    train_ids, test_ids = load_xgb_split(df["match_id"])

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

    print("Features used:", X_fit.shape[1])

    monotone_constraints = infer_monotone_constraints(X_fit.columns)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=XGB_MAX_ESTIMATORS,
        max_depth=XGB_CONFIG["max_depth"],
        learning_rate=XGB_CONFIG["learning_rate"],
        subsample=XGB_CONFIG["subsample"],
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        monotone_constraints=monotone_constraints,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    probs = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, probs)
    print_metrics(f"XGBoost test metrics ({label})", metrics)

    pred_df = test_df[["match_id", "minute", "target"]].copy()
    pred_df["prob"] = probs
    bucket_rows = evaluate_by_minute_bucket(pred_df, prob_col="prob")

    print(f"\nTest metrics by minute bucket ({label})")
    print_bucket_rows(bucket_rows)

    baseline = max(y_fit.mean(), 1 - y_fit.mean())

    return {
        "model": "XGBoost",
        "variant": label,
        "n_features": X_fit.shape[1],
        "best_iteration": model.best_iteration,
        "baseline_accuracy": round(float(baseline), 4),
        **{k: round(float(v), 4) for k, v in metrics.items() if k != "rows"},
        "bucket_rows": bucket_rows,
    }


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMEveryMinute(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


def make_loader(X, y, mask, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=shuffle)


def masked_loss(logits, y, mask, loss_fn):
    y_seq = y.unsqueeze(1).expand_as(logits)
    raw_loss = loss_fn(logits, y_seq)
    return (raw_loss * mask).sum() / mask.sum().clamp(min=1)


def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for X, y, mask in loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            probs = torch.sigmoid(model(X))
            y_seq = y.unsqueeze(1).expand_as(probs)

            all_probs.append(probs[mask == 1].cpu().numpy())
            all_targets.append(y_seq[mask == 1].cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_targets)


def evaluate(model, loader, device):
    probs, targets = collect_predictions(model, loader, device)
    return compute_metrics(targets, probs)


def evaluate_by_minute_bucket_lstm(model, X, y, mask, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        probs_all = torch.sigmoid(model(X_t)).cpu().numpy()

    n_matches, max_len = mask.shape
    minutes = np.tile(np.arange(1, max_len + 1), (n_matches, 1))
    targets = np.repeat(y[:, None], max_len, axis=1)
    valid = mask == 1

    pred_df = pd.DataFrame({
        "minute": minutes[valid],
        "target": targets[valid],
        "prob": probs_all[valid],
    })

    return evaluate_by_minute_bucket(pred_df, prob_col="prob")


def run_lstm(label, drop_predicate):
    print("\n" + "=" * 80)
    print(f"LSTM -- {label}")
    print("=" * 80)

    data = np.load(LSTM_INPUT_FILE, allow_pickle=True)
    feature_names = list(data["feature_names"])

    keep_idx = [i for i, f in enumerate(feature_names) if not drop_predicate(f)]

    print("Features used:", len(keep_idx), "of", len(feature_names))

    X_train = data["X_train"][:, :, keep_idx]
    y_train = data["y_train"]
    mask_train = data["mask_train"]

    X_val = data["X_val"][:, :, keep_idx]
    y_val = data["y_val"]
    mask_val = data["mask_val"]

    X_test = data["X_test"][:, :, keep_idx]
    y_test = data["y_test"]
    mask_test = data["mask_test"]

    input_size = X_train.shape[2]

    train_loader = make_loader(X_train, y_train, mask_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, mask_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, mask_test, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    model = LSTMEveryMinute(
        input_size=input_size,
        hidden_size=LSTM_CONFIG["hidden_size"],
        num_layers=LSTM_CONFIG["num_layers"],
        dropout=LSTM_CONFIG["dropout"],
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LSTM_LEARNING_RATE, weight_decay=LSTM_CONFIG["weight_decay"],
    )

    best_val_log_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_state = None

    for epoch in range(1, LSTM_MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for X, y, mask in train_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = masked_loss(model(X), y, mask, loss_fn)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device)

        print(
            f"epoch={epoch:02d} train_loss={np.mean(train_losses):.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_log_loss={val_metrics['log_loss']:.4f}"
        )

        if val_metrics["log_loss"] < best_val_log_loss:
            best_val_log_loss = val_metrics["log_loss"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if bad_epochs >= LSTM_PATIENCE:
            break

    model.load_state_dict(best_state)

    test_probs, test_targets = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(test_targets, test_probs)
    print_metrics(f"LSTM test metrics ({label})", metrics)

    bucket_rows = evaluate_by_minute_bucket_lstm(model, X_test, y_test, mask_test, device)

    print(f"\nTest metrics by minute bucket ({label})")
    print_bucket_rows(bucket_rows)

    baseline = max(y_train.mean(), 1 - y_train.mean())

    return {
        "model": "LSTM",
        "variant": label,
        "n_features": input_size,
        "best_epoch": best_epoch,
        "baseline_accuracy": round(float(baseline), 4),
        **{k: round(float(v), 4) for k, v in metrics.items() if k != "rows"},
        "bucket_rows": bucket_rows,
    }


# ---------------------------------------------------------------------------

def save_results_csv(results):
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "bucket_rows"}
        rows.append(row)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    # Keep a readable column order with the identifying fields first.
    ordered = ["model", "variant", "n_features"] + [f for f in fieldnames if f not in ("model", "variant", "n_features")]

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
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

        for variant_key, drop_label in [("no_economy", "gold/XP/level"), ("no_objectives", "objectives/towers")]:
            ablated = variants.get(variant_key)
            if not ablated:
                continue

            print(f"  {variant_key:<17} ({ablated['n_features']:>3} feats): AUC={ablated['auc']:.4f}  log_loss={ablated['log_loss']:.4f}  accuracy={ablated['accuracy']:.4f}  brier={ablated['brier']:.4f}")
            print(f"    AUC retained without {drop_label}: {ablated['auc'] / full['auc']:.1%}  (gap: {full['auc'] - ablated['auc']:.4f})")


def main():
    results = []

    for variant_label, predicate in VARIANTS.items():
        results.append(run_xgb(variant_label, predicate))

    for variant_label, predicate in VARIANTS.items():
        results.append(run_lstm(variant_label, predicate))

    save_results_csv(results)
    print("\nSaved:", RESULTS_FILE)

    print_summary(results)


if __name__ == "__main__":
    main()
