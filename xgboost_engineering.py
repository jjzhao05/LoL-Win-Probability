from pathlib import Path

import polars as pl

from common import BASIC_STATS, DAMAGE_STATS, OBJECTIVES, ROLES, TOWER_PARTS


INPUT_FILE = Path("data/full_dataset.parquet")
OUTPUT_FILE = Path("xgb_engineered/xgb_clean_dataset.parquet")


def cols(lf):
    return set(lf.collect_schema().names())


def safe_col(existing, name):
    if name in existing:
        return pl.col(name)
    return pl.lit(0)


def team_total(existing, team, stat):
    expr = pl.lit(0)

    for role in ROLES:
        col = f"{role}_{team}_{stat}"
        if col in existing:
            expr = expr + pl.col(col)

    return expr


def add_features(lf):
    existing = cols(lf)

    exprs = [
        (pl.col("timestamp_sec") / 60).alias("minute"),
        pl.col("team_100_win").cast(pl.Int8).alias("target"),
    ]

    for stat in BASIC_STATS + DAMAGE_STATS:
        team_100 = team_total(existing, 100, stat)
        team_200 = team_total(existing, 200, stat)

        exprs.append(team_100.alias(f"team_100_{stat}"))
        exprs.append(team_200.alias(f"team_200_{stat}"))
        exprs.append((team_100 - team_200).alias(f"team_{stat}_diff"))

    for role in ROLES:
        for stat in BASIC_STATS + DAMAGE_STATS:
            c100 = f"{role}_100_{stat}"
            c200 = f"{role}_200_{stat}"

            if c100 in existing and c200 in existing:
                exprs.append((pl.col(c100) - pl.col(c200)).alias(f"{role}_{stat}_diff"))

    for obj in OBJECTIVES:
        c100 = f"{obj}_100"
        c200 = f"{obj}_200"

        if c100 in existing and c200 in existing:
            exprs.append((pl.col(c100) - pl.col(c200)).alias(f"{obj}_diff"))

    lf = lf.with_columns(exprs)

    lf = lf.with_columns(
        [
            (
                pl.col("team_100_total_gold")
                / (pl.col("team_100_total_gold") + pl.col("team_200_total_gold") + 1e-9)
            ).alias("team_100_gold_share"),

            (
                pl.col("team_total_gold_diff")
                / pl.max_horizontal(pl.col("minute"), pl.lit(1.0))
            ).alias("gold_diff_per_min"),

            (
                pl.col("team_xp_diff")
                / pl.max_horizontal(pl.col("minute"), pl.lit(1.0))
            ).alias("xp_diff_per_min"),

            (
                (
                    pl.col("team_minions_killed_diff")
                    + pl.col("team_jungle_minions_killed_diff")
                )
                / pl.max_horizontal(pl.col("minute"), pl.lit(1.0))
            ).alias("cs_diff_per_min"),
        ]
    )

    return lf


def add_tower_features(lf):
    existing = cols(lf)

    def tower_sum(team):
        expr = pl.lit(0)

        for part in TOWER_PARTS:
            col = f"{part}_{team}_destroyed"
            if col in existing:
                expr = expr + pl.col(col)

        return expr

    lf = lf.with_columns(
        [
            tower_sum(100).alias("team_100_towers_destroyed"),
            tower_sum(200).alias("team_200_towers_destroyed"),
        ]
    )

    lf = lf.with_columns(
        (
            pl.col("team_100_towers_destroyed")
            - pl.col("team_200_towers_destroyed")
        ).alias("tower_diff")
    )

    return lf


def add_event_features(lf):
    existing = cols(lf)

    exprs = []

    if "first_blood_team" in existing and "first_blood_time_sec" in existing:
        exprs.extend(
            [
                (
                    (pl.col("first_blood_time_sec") <= pl.col("timestamp_sec"))
                    & (pl.col("first_blood_team") == 100)
                ).cast(pl.Int8).alias("team_100_first_blood_so_far"),

                (
                    (pl.col("first_blood_time_sec") <= pl.col("timestamp_sec"))
                    & (pl.col("first_blood_team") == 200)
                ).cast(pl.Int8).alias("team_200_first_blood_so_far"),
            ]
        )

    if "first_tower_team" in existing and "first_tower_time_sec" in existing:
        exprs.extend(
            [
                (
                    (pl.col("first_tower_time_sec") <= pl.col("timestamp_sec"))
                    & (pl.col("first_tower_team") == 100)
                ).cast(pl.Int8).alias("team_100_first_tower_so_far"),

                (
                    (pl.col("first_tower_time_sec") <= pl.col("timestamp_sec"))
                    & (pl.col("first_tower_team") == 200)
                ).cast(pl.Int8).alias("team_200_first_tower_so_far"),
            ]
        )

    if exprs:
        lf = lf.with_columns(exprs)

    existing = cols(lf)

    more = []

    if "team_100_first_blood_so_far" in existing:
        more.append(
            (
                pl.col("team_100_first_blood_so_far")
                - pl.col("team_200_first_blood_so_far")
            ).alias("first_blood_diff")
        )

    if "team_100_first_tower_so_far" in existing:
        more.append(
            (
                pl.col("team_100_first_tower_so_far")
                - pl.col("team_200_first_tower_so_far")
            ).alias("first_tower_diff")
        )

    if more:
        lf = lf.with_columns(more)

    return lf


def add_momentum_features(lf):
    feature_cols = [
        c for c in cols(lf)
        if c.endswith("_diff")
        or c.endswith("_share")
        or c.endswith("_per_min")
        or c.endswith("_destroyed")
    ]

    lf = lf.sort(["match_id", "timestamp_sec"])

    exprs = []

    for c in feature_cols:
        exprs.append(
            (pl.col(c) - pl.col(c).shift(1).over("match_id"))
            .fill_null(0)
            .alias(f"{c}_delta_1min")
        )

        exprs.append(
            (pl.col(c) - pl.col(c).shift(3).over("match_id"))
            .fill_null(0)
            .alias(f"{c}_delta_3min")
        )

    return lf.with_columns(exprs)


def select_output_columns(lf):
    existing = cols(lf)

    keep = [
        "match_id",
        "timestamp_sec",
        "minute",
        "target",
    ]

    feature_cols = [
        c for c in existing
        if c.endswith("_diff")
        or c.endswith("_share")
        or c.endswith("_per_min")
        or c.endswith("_destroyed")
        or c.endswith("_so_far")
        or c.endswith("_delta_1min")
        or c.endswith("_delta_3min")
    ]

    keep = keep + sorted(feature_cols)

    return lf.select([c for c in keep if c in existing])


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    print(f"Reading: {INPUT_FILE}")

    lf = pl.scan_parquet(INPUT_FILE)

    required = {"match_id", "timestamp_sec", "team_100_win"}
    missing = required - cols(lf)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    lf = add_features(lf)
    lf = add_tower_features(lf)
    lf = add_event_features(lf)
    lf = add_momentum_features(lf)
    lf = select_output_columns(lf)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing: {OUTPUT_FILE}")

    lf.collect(streaming=True).write_parquet(
        OUTPUT_FILE,
        compression="zstd",
        statistics=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()