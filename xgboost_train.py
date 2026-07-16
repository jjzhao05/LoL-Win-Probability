from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


INPUT_FILE = Path("xgb_engineered/xgb_clean_dataset.parquet")

SPLIT_FILE = Path("shared_split_ids.npz")

MODEL_FILE = Path("xgb_model.json")
PREDICTIONS_FILE = Path("xgb_predictions.parquet")

TEST_SIZE = 0.20
RANDOM_STATE = 101705


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
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y


def get_metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_true, probs),
        "log_loss": log_loss(y_true, probs, labels=[0, 1]),
        "accuracy": accuracy_score(y_true, preds),
        "rows": len(y_true),
    }


def print_metrics(name, metrics):
    print()
    print(name)
    print("AUC:", round(float(metrics["auc"]), 4))
    print("Log loss:", round(float(metrics["log_loss"]), 4))
    print("Accuracy:", round(float(metrics["accuracy"]), 4))
    print("Rows:", int(metrics["rows"]))


def evaluate_by_minute_bucket(pred_df):
    buckets = [
        (1, 5),
        (6, 10),
        (11, 15),
        (16, 20),
        (21, 25),
        (26, 30),
        (31, 45),
    ]

    print()
    print("Test metrics by minute bucket")

    for start, end in buckets:
        g = pred_df[
            (pred_df["minute"] >= start)
            & (pred_df["minute"] <= end)
        ].copy()

        if g.empty:
            continue

        probs = g["pred_prob_team_100_win"]
        preds = (probs >= 0.5).astype(int)

        if g["target"].nunique() < 2:
            auc_text = "nan"
        else:
            auc_text = round(float(roc_auc_score(g["target"], probs)), 4)

        bucket_log_loss = log_loss(g["target"], probs, labels=[0, 1])
        bucket_accuracy = accuracy_score(g["target"], preds)

        print(
            f"{start}-{end}",
            "| rows:", len(g),
            "| auc:", auc_text,
            "| log_loss:", round(float(bucket_log_loss), 4),
            "| accuracy:", round(float(bucket_accuracy), 4),
        )


def main():
    print("Reading:", INPUT_FILE)

    df = pd.read_parquet(INPUT_FILE)

    required = {"match_id", "target", "minute"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["match_id"] = df["match_id"].astype(str)

    train_ids, test_ids = load_or_create_split(df["match_id"])

    train_ids = set(train_ids)
    test_ids = set(test_ids)

    train_df = df[df["match_id"].isin(train_ids)].copy()
    test_df = df[df["match_id"].isin(test_ids)].copy()

    if train_df.empty:
        raise ValueError("Training data is empty.")

    if test_df.empty:
        raise ValueError("Test data is empty.")

    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)

    print("Rows:", len(df))
    print("Matches:", df["match_id"].nunique())
    print("Train matches:", train_df["match_id"].nunique())
    print("Test matches:", test_df["match_id"].nunique())
    print("Features:", X_train.shape[1])

    print()
    print("Feature columns:")
    for col in X_train.columns:
        print(col)

    baseline = max(y_train.mean(), 1 - y_train.mean())

    print()
    print("Majority-class baseline accuracy:", round(float(baseline), 4))

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print()
    print("Training XGBoost")

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    overall = get_metrics(y_test, probs)

    print_metrics("Test metrics across all valid minutes", overall)

    pred_df = test_df[["match_id", "minute", "target"]].copy()
    pred_df["pred_prob_team_100_win"] = probs
    pred_df["pred_label"] = preds
    pred_df = pred_df.sort_values(["match_id", "minute"])

    evaluate_by_minute_bucket(pred_df)

    model.save_model(MODEL_FILE)
    pred_df.to_parquet(PREDICTIONS_FILE, index=False)

    print()
    print("Saved model to:", MODEL_FILE)
    print("Saved predictions to:", PREDICTIONS_FILE)
    print("Saved shared split to:", SPLIT_FILE)


if __name__ == "__main__":
    main()