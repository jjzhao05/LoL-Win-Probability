import json
from pathlib import Path

import pandas as pd

from db import get_engine, MATCH_SNAPSHOTS_TABLE


ROLES = ["top", "jg", "mid", "bot", "sup"]
TEAMS = ["100", "200"]
SNOWBALL_TIME_SEC = 900

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOOKUP_DIR = BASE_DIR / "lol_data"
OUTPUT_DIR = DATA_DIR


def _load_map(filename: str) -> dict:
    path = LOOKUP_DIR / filename

    if not path.exists():
        print(f"Warning: lookup file not found: {path}")
        print(f"Using empty mapping for {filename}")
        return {}

    with open(path, encoding="utf-8") as f:
        return json.load(f)


CHAMPION_MAP = _load_map("champion_map.json")
SPELL_MAP = _load_map("summoner_spell_map.json")
STYLE_MAP = _load_map("rune_style_map.json")
PERK_MAP = _load_map("perk_map.json")


def champ_name(champ_id):
    if pd.isna(champ_id):
        return "Unknown"

    champ_id = int(champ_id)
    return CHAMPION_MAP.get(str(champ_id), f"Unknown({champ_id})")


def spell_name(spell_id):
    if pd.isna(spell_id):
        return "Unknown"

    spell_id = int(spell_id)
    return SPELL_MAP.get(str(spell_id), f"Unknown({spell_id})")


def style_name(style_id):
    if pd.isna(style_id):
        return "Unknown"

    style_id = int(style_id)
    return STYLE_MAP.get(str(style_id), f"Unknown({style_id})")


def perk_name(perk_id):
    if pd.isna(perk_id):
        return "Unknown"

    perk_id = int(perk_id)
    return PERK_MAP.get(str(perk_id), f"Unknown({perk_id})")


def load_dataset() -> pd.DataFrame:
    engine = get_engine()
    raw_df = pd.read_sql_table(MATCH_SNAPSHOTS_TABLE, engine)

    print(f"Loaded {len(raw_df)} rows from '{MATCH_SNAPSHOTS_TABLE}'")
    print()

    return raw_df


def win_rate_when(df, cond_100_has_it, cond_200_has_it):
    only_100 = cond_100_has_it & ~cond_200_has_it
    only_200 = cond_200_has_it & ~cond_100_has_it

    wins_100 = df.loc[only_100, "team_100_win"].sum()
    wins_200 = (1 - df.loc[only_200, "team_100_win"]).sum()

    total = only_100.sum() + only_200.sum()

    if total == 0:
        return None, 0

    return round((wins_100 + wins_200) / total, 4), int(total)


ROLE_GOLD_DIFF_COLS = [f"{role}_gold_diff" for role in ROLES]


def feature_correlations(raw_df):
    gold_diff = raw_df[ROLE_GOLD_DIFF_COLS].sum(axis=1)
    tower_diff = raw_df["towers_100"] - raw_df["towers_200"]
    elder_diff = raw_df["elders_100"] - raw_df["elders_200"]

    candidates = {
        "kill_diff": raw_df["kill_diff"],
        "dragon_diff": raw_df["dragon_diff"],
        "herald_diff": raw_df["herald_diff"],
        "baron_diff": raw_df["baron_diff"],
        "elder_diff": elder_diff,
        "plate_diff": raw_df["plate_diff"],
        "tower_diff": tower_diff,
    }

    rows = []
    for name, series in candidates.items():
        rows.append({
            "feature": name,
            "pearson_corr_with_gold_diff": round(float(gold_diff.corr(series)), 4),
            "n_rows": int(min(gold_diff.notna().sum(), series.notna().sum())),
        })

    return pd.DataFrame(rows).sort_values("pearson_corr_with_gold_diff", ascending=False, key=abs)


def main():
    raw_df = load_dataset()

    df = raw_df.sort_values("timestamp_sec").drop_duplicates("match_id", keep="last").copy()

    n_games = len(df)
    print(f"Deduplicated to {n_games} unique games using final snapshot per match")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    longest = df.loc[df["game_duration_sec"].idxmax()]
    shortest = df.loc[df["game_duration_sec"].idxmin()]

    win_rate_100 = df["team_100_win"].mean()

    champion_pick_cols = [
        df[f"{role}_{team}_champion_id"]
        for role in ROLES
        for team in TEAMS
        if f"{role}_{team}_champion_id" in df.columns
    ]

    rows = [
        ("num_timeline_snapshot_rows", len(raw_df)),
        ("num_games", n_games),
        ("num_unique_patches", df["patch"].nunique()),
        ("num_unique_champions_picked", pd.concat(champion_pick_cols).nunique()),
        ("avg_game_duration_min", round(df["game_duration_sec"].mean() / 60, 2)),
        ("median_game_duration_min", round(df["game_duration_sec"].median() / 60, 2)),
        ("longest_game_min", round(longest["game_duration_sec"] / 60, 2)),
        ("longest_game_match_id", longest["match_id"]),
        ("shortest_game_min", round(shortest["game_duration_sec"] / 60, 2)),
        ("shortest_game_match_id", shortest["match_id"]),
        ("blue_side_team_100_win_rate", round(win_rate_100, 4)),
        ("red_side_team_200_win_rate", round(1 - win_rate_100, 4)),
        ("avg_first_blood_time_sec", round(df["first_blood_time_sec"].mean(), 1)),
        ("avg_first_tower_time_sec", round(df["first_tower_time_sec"].mean(), 1)),
        ("avg_total_dragons_per_game", round((df["dragons_100"] + df["dragons_200"]).mean(), 2)),
        ("avg_total_barons_per_game", round((df["barons_100"] + df["barons_200"]).mean(), 2)),
        ("avg_total_heralds_per_game", round((df["heralds_100"] + df["heralds_200"]).mean(), 2)),
        ("pct_games_with_elder_dragon", round(((df["elders_100"] + df["elders_200"]) > 0).mean() * 100, 1)),
        ("pct_games_with_baron", round(((df["barons_100"] + df["barons_200"]) > 0).mean() * 100, 1)),
    ]

    obj_rows = []

    wr, n = win_rate_when(
        df,
        df["first_blood_team"] == 100,
        df["first_blood_team"] == 200,
    )
    obj_rows.append(("first_blood", wr, n))

    wr, n = win_rate_when(
        df,
        df["first_tower_team"] == 100,
        df["first_tower_team"] == 200,
    )
    obj_rows.append(("first_tower", wr, n))

    wr, n = win_rate_when(
        df,
        df["dragons_100"] > df["dragons_200"],
        df["dragons_200"] > df["dragons_100"],
    )
    obj_rows.append(("dragon_advantage", wr, n))

    wr, n = win_rate_when(
        df,
        df["barons_100"] > df["barons_200"],
        df["barons_200"] > df["barons_100"],
    )
    obj_rows.append(("baron_advantage", wr, n))

    wr, n = win_rate_when(
        df,
        df["heralds_100"] > df["heralds_200"],
        df["heralds_200"] > df["heralds_100"],
    )
    obj_rows.append(("herald_advantage", wr, n))

    objective_impact = pd.DataFrame(
        obj_rows,
        columns=["objective", "win_rate_for_team_that_secured_it", "games_with_clear_advantage"],
    )
    objective_impact.to_csv(OUTPUT_DIR / "objective_impact.csv", index=False)

    corr_df = feature_correlations(raw_df)
    corr_df.to_csv(OUTPUT_DIR / "feature_correlations.csv", index=False)

    raw_df = raw_df.copy()
    raw_df["dist_to_15min"] = (raw_df["timestamp_sec"] - SNOWBALL_TIME_SEC).abs()

    snap15 = raw_df.sort_values("dist_to_15min").drop_duplicates("match_id", keep="first").copy()

    gold_100_cols = [f"{role}_100_total_gold" for role in ROLES]
    gold_200_cols = [f"{role}_200_total_gold" for role in ROLES]

    snap15["team_100_gold_15min"] = snap15[gold_100_cols].sum(axis=1)
    snap15["team_200_gold_15min"] = snap15[gold_200_cols].sum(axis=1)
    snap15["gold_lead_100"] = snap15["team_100_gold_15min"] - snap15["team_200_gold_15min"]

    snap15 = snap15.merge(
        df[["match_id", "team_100_win"]],
        on="match_id",
        suffixes=("", "_final"),
    )

    leads_100 = snap15["gold_lead_100"] > 500
    leads_200 = snap15["gold_lead_100"] < -500
    n_leads = int(leads_100.sum() + leads_200.sum())

    if n_leads > 0:
        snowball_wr_100 = snap15.loc[leads_100, "team_100_win_final"].mean()
        snowball_wr_200 = 1 - snap15.loc[leads_200, "team_100_win_final"].mean()

        snowball_wr = round(
            (
                snowball_wr_100 * leads_100.sum()
                + snowball_wr_200 * leads_200.sum()
            )
            / n_leads,
            4,
        )
    else:
        snowball_wr = None

    avg_gold_lead_mag = snap15["gold_lead_100"].abs().mean()

    rows.append(("win_rate_with_500plus_gold_lead_at_15min", snowball_wr))
    rows.append(("avg_gold_lead_magnitude_at_15min", round(avg_gold_lead_mag, 0)))

    snap15["team_100_was_behind"] = snap15["gold_lead_100"] < -1000
    snap15["team_200_was_behind"] = snap15["gold_lead_100"] > 1000

    comeback_100 = snap15["team_100_was_behind"] & (snap15["team_100_win_final"] == 1)
    comeback_200 = snap15["team_200_was_behind"] & (snap15["team_100_win_final"] == 0)

    n_comebacks = int(comeback_100.sum() + comeback_200.sum())

    rows.append(("num_comeback_wins_from_1000plus_gold_deficit_at_15min", n_comebacks))
    rows.append(("pct_games_that_were_comebacks", round(n_comebacks / n_games * 100, 1) if n_games else None))

    comeback_candidates = snap15.copy()

    def comeback_margin(row):
        team_100_behind_and_won = row["gold_lead_100"] < 0 and row["team_100_win_final"] == 1
        team_200_behind_and_won = row["gold_lead_100"] > 0 and row["team_100_win_final"] == 0

        if team_100_behind_and_won or team_200_behind_and_won:
            return abs(row["gold_lead_100"])

        return 0

    comeback_candidates["comeback_margin"] = comeback_candidates.apply(comeback_margin, axis=1)
    biggest_comeback = comeback_candidates.loc[comeback_candidates["comeback_margin"].idxmax()]

    notable = pd.DataFrame(
        [
            {
                "category": "longest_game",
                "match_id": longest["match_id"],
                "detail": f"{round(longest['game_duration_sec'] / 60, 1)} min",
            },
            {
                "category": "shortest_game",
                "match_id": shortest["match_id"],
                "detail": f"{round(shortest['game_duration_sec'] / 60, 1)} min",
            },
            {
                "category": "biggest_15min_comeback",
                "match_id": biggest_comeback["match_id"],
                "detail": f"overcame a {int(biggest_comeback['comeback_margin'])}g deficit at 15min to win",
            },
        ]
    )
    notable.to_csv(OUTPUT_DIR / "notable_games.csv", index=False)

    summary_df = pd.DataFrame(rows, columns=["stat", "value"])
    summary_df.to_csv(OUTPUT_DIR / "summary_stats.csv", index=False)

    champ_records = []

    for role in ROLES:
        for team in TEAMS:
            sub = df[[f"{role}_{team}_champion_id", "team_100_win"]].copy()
            sub.columns = ["champion_id", "team_100_win"]

            if team == "100":
                sub["win"] = sub["team_100_win"]
            else:
                sub["win"] = 1 - sub["team_100_win"]

            sub["role"] = role
            champ_records.append(sub[["champion_id", "role", "win"]])

    champ_df = pd.concat(champ_records, ignore_index=True)

    champ_stats = (
        champ_df.groupby("champion_id")
        .agg(
            picks=("win", "size"),
            win_rate=("win", "mean"),
        )
        .reset_index()
        .sort_values("picks", ascending=False)
    )

    champ_stats["win_rate"] = champ_stats["win_rate"].round(4)
    champ_stats.insert(1, "champion_name", champ_stats["champion_id"].apply(champ_name))
    champ_stats.to_csv(OUTPUT_DIR / "champion_stats.csv", index=False)

    ban_ids = pd.concat(
        [
            df["team_100_bans"].astype(str).str.split("|").explode(),
            df["team_200_bans"].astype(str).str.split("|").explode(),
        ]
    )

    ban_ids = pd.to_numeric(ban_ids, errors="coerce").dropna()

    ban_stats = ban_ids.value_counts().reset_index()
    ban_stats.columns = ["champion_id", "times_banned"]
    ban_stats.insert(1, "champion_name", ban_stats["champion_id"].apply(champ_name))
    ban_stats.to_csv(OUTPUT_DIR / "ban_stats.csv", index=False)

    role_rows = []

    for role in ROLES:
        kills = pd.concat([df[f"{role}_100_kills"], df[f"{role}_200_kills"]])
        deaths = pd.concat([df[f"{role}_100_deaths"], df[f"{role}_200_deaths"]])
        assists = pd.concat([df[f"{role}_100_assists"], df[f"{role}_200_assists"]])

        cs = pd.concat(
            [
                df[f"{role}_100_minions_killed"] + df[f"{role}_100_jungle_minions_killed"],
                df[f"{role}_200_minions_killed"] + df[f"{role}_200_jungle_minions_killed"],
            ]
        )

        gold = pd.concat([df[f"{role}_100_total_gold"], df[f"{role}_200_total_gold"]])
        kda = (kills.sum() + assists.sum()) / max(deaths.sum(), 1)

        role_rows.append(
            {
                "role": role,
                "avg_kills": round(kills.mean(), 2),
                "avg_deaths": round(deaths.mean(), 2),
                "avg_assists": round(assists.mean(), 2),
                "kda_ratio": round(kda, 2),
                "avg_cs": round(cs.mean(), 1),
                "avg_gold": round(gold.mean(), 0),
            }
        )

    role_stats = pd.DataFrame(role_rows)
    role_stats.to_csv(OUTPUT_DIR / "role_stats.csv", index=False)

    spell_ids = pd.concat(
        [df[f"{role}_{team}_summoner1_id"] for role in ROLES for team in TEAMS]
        + [df[f"{role}_{team}_summoner2_id"] for role in ROLES for team in TEAMS]
    )

    spell_stats = spell_ids.value_counts().reset_index()
    spell_stats.columns = ["spell_id", "times_taken"]
    spell_stats.insert(1, "spell_name", spell_stats["spell_id"].apply(spell_name))
    spell_stats.to_csv(OUTPUT_DIR / "summoner_spell_stats.csv", index=False)

    style_ids = pd.concat([df[f"{role}_{team}_primary_style"] for role in ROLES for team in TEAMS])

    style_stats = style_ids.value_counts().reset_index()
    style_stats.columns = ["style_id", "times_taken"]
    style_stats.insert(1, "style_name", style_stats["style_id"].apply(style_name))
    style_stats.to_csv(OUTPUT_DIR / "rune_style_stats.csv", index=False)

    keystone_ids = pd.concat([df[f"{role}_{team}_keystone"] for role in ROLES for team in TEAMS])

    keystone_stats = keystone_ids.value_counts().reset_index()
    keystone_stats.columns = ["keystone_id", "times_taken"]
    keystone_stats.insert(1, "keystone_name", keystone_stats["keystone_id"].apply(perk_name))
    keystone_stats.to_csv(OUTPUT_DIR / "keystone_stats.csv", index=False)

    patch_stats = (
        df.groupby("patch")
        .agg(
            games=("match_id", "count"),
            team_100_win_rate=("team_100_win", "mean"),
            avg_duration_min=("game_duration_sec", lambda x: round(x.mean() / 60, 1)),
        )
        .reset_index()
        .sort_values("patch")
    )

    patch_stats["team_100_win_rate"] = patch_stats["team_100_win_rate"].round(4)
    patch_stats.to_csv(OUTPUT_DIR / "patch_stats.csv", index=False)

    print("=" * 60)
    print("OVERALL SUMMARY STATS")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    print()
    print("=" * 60)
    print("DOES SECURING AN OBJECTIVE ACTUALLY WIN YOU THE GAME?")
    print("=" * 60)
    print(objective_impact.to_string(index=False))

    print()
    print("=" * 60)
    print("CORRELATION OF EACH FEATURE GROUP WITH GOLD DIFFERENCE")
    print("=" * 60)
    print(corr_df.to_string(index=False))

    print()
    print("=" * 60)
    print("NOTABLE GAMES")
    print("=" * 60)
    print(notable.to_string(index=False))

    print()
    print("=" * 60)
    print("ROLE STATS")
    print("=" * 60)
    print(role_stats.to_string(index=False))

    print()
    print("=" * 60)
    print("PATCH BREAKDOWN")
    print("=" * 60)
    print(patch_stats.to_string(index=False))

    print()
    print("=" * 60)
    print("TOP 15 MOST PICKED CHAMPIONS")
    print("=" * 60)
    print(champ_stats.head(15).to_string(index=False))

    print()
    print("=" * 60)
    print("TOP 15 MOST BANNED CHAMPIONS")
    print("=" * 60)
    print(ban_stats.head(15).to_string(index=False))

    print()
    print("=" * 60)
    print("SUMMONER SPELL POPULARITY")
    print("=" * 60)
    print(spell_stats.to_string(index=False))

    print()
    print("=" * 60)
    print("RUNE TREE STYLE POPULARITY")
    print("=" * 60)
    print(style_stats.to_string(index=False))

    print()
    print("=" * 60)
    print("TOP 10 KEYSTONE RUNES")
    print("=" * 60)
    print(keystone_stats.head(10).to_string(index=False))

    print()
    print("Saved files to:")
    print(OUTPUT_DIR)
    print()
    print(
        "Saved: summary_stats.csv, champion_stats.csv, ban_stats.csv, "
        "role_stats.csv, objective_impact.csv, feature_correlations.csv, patch_stats.csv, "
        "notable_games.csv, summoner_spell_stats.csv, rune_style_stats.csv, keystone_stats.csv"
    )


if __name__ == "__main__":
    main()