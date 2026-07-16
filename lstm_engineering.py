from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INPUT_FILE = Path("data/full_dataset.parquet")
OUTPUT_FILE = Path("lstm_data.npz")
SPLIT_FILE = Path("shared_split_ids.npz")

MAX_LEN = 45
VAL_SIZE = 0.2
RANDOM_STATE = 101705

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

OBJECTIVES = ["plates", "dragons", "heralds", "barons", "elders"]


def team_total(df, team, stat):
    cols = [
        f"{role}_{team}_{stat}"
        for role in ROLES
        if f"{role}_{team}_{stat}" in df.columns
    ]

    if not cols:
        return None

    return df[cols].sum(axis=1)


def add_features(df):
    df = df.copy()

    df["minute"] = df["timestamp_sec"] / 60

    for stat in BASIC_STATS:
        total_100 = team_total(df, 100, stat)
        total_200 = team_total(df, 200, stat)

        if total_100 is not None and total_200 is not None:
            df[f"team_{stat}_diff"] = total_100 - total_200

    for role in ROLES:
        for stat in BASIC_STATS:
            c100 = f"{role}_100_{stat}"
            c200 = f"{role}_200_{stat}"

            if c100 in df.columns and c200 in df.columns:
                df[f"{role}_{stat}_diff"] = df[c100] - df[c200]

    for obj in OBJECTIVES:
        c100 = f"{obj}_100"
        c200 = f"{obj}_200"

        if c100 in df.columns and c200 in df.columns:
            df[f"{obj}_diff"] = df[c100] - df[c200]

    if "team_total_gold_diff" in df.columns:
        df["gold_diff_per_min"] = df["team_total_gold_diff"] / df["minute"].clip(lower=1)

    if "team_xp_diff" in df.columns:
        df["xp_diff_per_min"] = df["team_xp_diff"] / df["minute"].clip(lower=1)

    if "team_minions_killed_diff" in df.columns and "team_jungle_minions_killed_diff" in df.columns:
        df["cs_diff_per_min"] = (
            df["team_minions_killed_diff"] + df["team_jungle_minions_killed_diff"]
        ) / df["minute"].clip(lower=1)

    tower_parts = [
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

    for team in TEAMS:
        tower_cols = [
            f"{part}_{team}_destroyed"
            for part in tower_parts
            if f"{part}_{team}_destroyed" in df.columns
        ]

        if tower_cols:
            df[f"team_{team}_towers_destroyed"] = df[tower_cols].sum(axis=1)

    if "team_100_towers_destroyed" in df.columns and "team_200_towers_destroyed" in df.columns:
        df["tower_diff"] = df["team_100_towers_destroyed"] - df["team_200_towers_destroyed"]

    if "first_blood_team" in df.columns and "first_blood_time_sec" in df.columns:
        df["first_blood_diff"] = 0

        df.loc[
            (df["first_blood_time_sec"] <= df["timestamp_sec"])
            & (df["first_blood_team"] == 100),
            "first_blood_diff",
        ] = 1

        df.loc[
            (df["first_blood_time_sec"] <= df["timestamp_sec"])
            & (df["first_blood_team"] == 200),
            "first_blood_diff",
        ] = -1

    if "first_tower_team" in df.columns and "first_tower_time_sec" in df.columns:
        df["first_tower_diff"] = 0

        df.loc[
            (df["first_tower_time_sec"] <= df["timestamp_sec"])
            & (df["first_tower_team"] == 100),
            "first_tower_diff",
        ] = 1

        df.loc[
            (df["first_tower_time_sec"] <= df["timestamp_sec"])
            & (df["first_tower_team"] == 200),
            "first_tower_diff",
        ] = -1

    return df


def get_feature_cols(df):
    feature_cols = []

    for col in df.columns:
        if col == "minute":
            feature_cols.append(col)
        elif col.endswith("_diff"):
            feature_cols.append(col)
        elif col.endswith("_per_min"):
            feature_cols.append(col)
        elif col.endswith("_destroyed"):
            feature_cols.append(col)

    return sorted(feature_cols)


def load_shared_split(all_match_ids):
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SPLIT_FILE}. Run xgboost_train.py first so it creates the shared split."
        )

    split = np.load(SPLIT_FILE, allow_pickle=True)

    train_ids = split["train_ids"].astype(str)
    test_ids = split["test_ids"].astype(str)

    all_match_ids = pd.Series(all_match_ids).astype(str).drop_duplicates()

    train_ids = np.array([m for m in train_ids if m in set(all_match_ids)])
    test_ids = np.array([m for m in test_ids if m in set(all_match_ids)])

    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    return train_ids, val_ids, test_ids


def make_arrays(df, match_ids, feature_cols):
    X = np.zeros((len(match_ids), MAX_LEN, len(feature_cols)), dtype=np.float32)
    y = np.zeros(len(match_ids), dtype=np.float32)
    mask = np.zeros((len(match_ids), MAX_LEN), dtype=np.float32)

    y_by_match = df.groupby("match_id")["team_100_win"].first()

    for i, match_id in enumerate(match_ids):
        g = df[df["match_id"] == match_id].sort_values("timestamp_sec")

        values = g[feature_cols].to_numpy(dtype=np.float32)
        length = min(len(values), MAX_LEN)

        X[i, :length, :] = values[:length]
        mask[i, :length] = 1
        y[i] = y_by_match.loc[match_id]

    return X, y, mask


def main():
    df = pd.read_parquet(INPUT_FILE)

    required = {"match_id", "timestamp_sec", "team_100_win"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["match_id"] = df["match_id"].astype(str)

    df = df.sort_values(["match_id", "timestamp_sec"]).copy()
    df = add_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_cols = get_feature_cols(df)

    train_ids, val_ids, test_ids = load_shared_split(df["match_id"].unique())

    scaler = StandardScaler()
    scaler.fit(df[df["match_id"].isin(train_ids)][feature_cols])

    df[feature_cols] = scaler.transform(df[feature_cols])

    X_train, y_train, mask_train = make_arrays(df, train_ids, feature_cols)
    X_val, y_val, mask_val = make_arrays(df, val_ids, feature_cols)
    X_test, y_test, mask_test = make_arrays(df, test_ids, feature_cols)

    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train,
        y_train=y_train,
        mask_train=mask_train,
        X_val=X_val,
        y_val=y_val,
        mask_val=mask_val,
        X_test=X_test,
        y_test=y_test,
        mask_test=mask_test,
        feature_names=np.array(feature_cols),
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        match_ids_test=test_ids,
    )

    print("Saved:", OUTPUT_FILE)
    print("Features:", len(feature_cols))
    print("Train matches:", len(train_ids))
    print("Val matches:", len(val_ids))
    print("Test matches:", len(test_ids))
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)


if __name__ == "__main__":
    main()