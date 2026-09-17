import os
import json
import time
import random
import threading
from collections import deque
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

from db import get_engine, MATCH_SNAPSHOTS_TABLE
from logging_utils import run_with_file_logging

load_dotenv()

LOG_DIR = Path("logs")

# Same value as common.RANDOM_STATE, kept local since this script otherwise
# has no reason to import common.py. Seeds the puuid-pool shuffles below.
RANDOM_STATE = 101705
random.seed(RANDOM_STATE)


API_KEY = os.getenv("RIOT_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("Missing RIOT_API_KEY. Add it to your .env file.")

HEADERS = {"X-Riot-Token": API_KEY}

PLATFORM = "na1"
REGION = "americas"

QUEUE = "RANKED_SOLO_5x5"
QUEUE_ID = 420


TARGET_GAMES_PER_BRACKET = 1000

MATCHES_PER_PLAYER = 5

TARGET_PATCH = "16.17"

MIN_GAME_DURATION_SEC = 900

MATCH_CHUNK_SIZE = 25

# How many not-yet-queried puuids to pull match ids from per batch, and how
# much to grow a bracket's puuid pool by once every pool puuid is queried.
QUERIED_BATCH_SIZE = 200
POOL_GROWTH_STEP = 400

RATE_LIMITS = [
    (20, 1),
    (100, 120),
]

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

BRACKETS = {
    tier: [(tier, d) for d in ("I", "II", "III", "IV")]
    for tier in ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND")
}

for apex_tier in ("MASTER", "GRANDMASTER", "CHALLENGER"):
    BRACKETS[apex_tier] = [apex_tier]

LEAGUE_HOST = f"https://{PLATFORM}.api.riotgames.com"
MATCH_HOST = f"https://{REGION}.api.riotgames.com"


class RateLimiter:
    def __init__(self, limits):
        self.limits = limits
        self.calls = [deque() for _ in limits]
        self.lock = threading.Lock()

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                sleep_for = 0.0
                for i, (max_calls, window) in enumerate(self.limits):
                    dq = self.calls[i]
                    while dq and now - dq[0] > window:
                        dq.popleft()
                    if len(dq) >= max_calls:
                        sleep_for = max(sleep_for, window - (now - dq[0]) + 0.05)
                if sleep_for <= 0:
                    for dq in self.calls:
                        dq.append(now)
                    return
            time.sleep(sleep_for)


limiter = RateLimiter(RATE_LIMITS)

def riot_get(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        limiter.wait()
        resp = requests.get(url, headers=HEADERS, params=params, timeout=20)

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 401:
            raise RuntimeError(
                "401 Unauthorized. API key is invalid or expired, "
                "generate a new one at https://developer.riotgames.com"
            )
        if resp.status_code == 403:
            raise RuntimeError(
                "403 Forbidden. Key missing required permissions/expired."
            )
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "5"))
            print(f"  429 rate limited, sleeping {retry_after}s...")
            time.sleep(retry_after + 1)
            continue
        if resp.status_code >= 500:
            wait = 2 ** attempt
            print(f"  {resp.status_code} server error, retrying in {wait}s...")
            time.sleep(wait)
            continue

        resp.raise_for_status()

    raise RuntimeError(f"Exceeded retries for {url}")


def get_league_entries(tier, division, page=1):
    url = f"{LEAGUE_HOST}/lol/league/v4/entries/{QUEUE}/{tier}/{division}"
    return riot_get(url, params={"page": page})


def get_apex_league(tier):
    endpoint = {
        "MASTER": "masterleagues",
        "GRANDMASTER": "grandmasterleagues",
        "CHALLENGER": "challengerleagues",
    }[tier]
    url = f"{LEAGUE_HOST}/lol/league/v4/{endpoint}/by-queue/{QUEUE}"
    data = riot_get(url)
    return data.get("entries", []) if data else []


def get_summoner_puuid(encrypted_summoner_id):
    url = f"{LEAGUE_HOST}/lol/summoner/v4/summoners/{encrypted_summoner_id}"
    data = riot_get(url)
    return data.get("puuid") if data else None


def entry_to_puuid(entry):
    puuid = entry.get("puuid")
    if puuid:
        return puuid
    summoner_id = entry.get("summonerId")
    if summoner_id:
        return get_summoner_puuid(summoner_id)
    return None


def collect_puuid_pool(bracket_name, target_pool_size, cache_path):
    if cache_path.exists():
        pool = json.loads(cache_path.read_text())
        print(f"[{bracket_name}] loaded cached pool of {len(pool)} puuids")
        if len(pool) >= target_pool_size:
            return pool
    else:
        pool = []

    seen = set(pool)
    spec = BRACKETS[bracket_name]

    APEX_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER"}

    if bracket_name in APEX_TIERS:
        for tier in spec:
            if len(pool) >= target_pool_size:
                break
            print(f"[{bracket_name}] fetching apex league {tier}...")
            entries = get_apex_league(tier)
            random.shuffle(entries)
            for entry in entries:
                if len(pool) >= target_pool_size:
                    break
                puuid = entry_to_puuid(entry)
                if puuid and puuid not in seen:
                    seen.add(puuid)
                    pool.append(puuid)
    else:
        divisions = list(spec)
        random.shuffle(divisions)
        per_division_cap = -(-target_pool_size // len(divisions))
        pages = {d: 1 for d in divisions}
        exhausted = set()
        division_counts = {d: 0 for d in divisions}

        while len(pool) < target_pool_size and len(exhausted) < len(divisions):
            for tier, division in divisions:
                if (tier, division) in exhausted:
                    continue
                if division_counts[(tier, division)] >= per_division_cap:
                    exhausted.add((tier, division))
                    continue
                if len(pool) >= target_pool_size:
                    break

                entries = get_league_entries(tier, division, page=pages[(tier, division)])
                if not entries:
                    exhausted.add((tier, division))
                    continue

                for entry in entries:
                    if division_counts[(tier, division)] >= per_division_cap:
                        break
                    puuid = entry_to_puuid(entry)
                    if puuid and puuid not in seen:
                        seen.add(puuid)
                        pool.append(puuid)
                        division_counts[(tier, division)] += 1
                pages[(tier, division)] += 1

        for (tier, division), count in division_counts.items():
            print(f"[{bracket_name}] {tier} {division}: {count} puuids")

    cache_path.write_text(json.dumps(pool))
    print(f"[{bracket_name}] final pool: {len(pool)} puuids")
    return pool


def get_ranked_match_ids(puuid, count):
    url = f"{MATCH_HOST}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"queue": QUEUE_ID, "type": "ranked", "start": 0, "count": count}
    return riot_get(url, params=params) or []


def get_match(match_id):
    url = f"{MATCH_HOST}/lol/match/v5/matches/{match_id}"
    return riot_get(url)


def get_timeline(match_id):
    url = f"{MATCH_HOST}/lol/match/v5/matches/{match_id}/timeline"
    return riot_get(url)


def collect_match_ids(bracket_name, puuid_batch, processed_ids, global_seen):
    """Pull recent ranked match ids for every puuid in the batch. The
    caller (process_bracket) decides when enough matches have been
    collected and whether to pull another batch."""
    match_ids = []
    seen = set(processed_ids) | set(global_seen)
    for puuid in puuid_batch:
        try:
            ids = get_ranked_match_ids(puuid, MATCHES_PER_PLAYER)
        except RuntimeError as e:
            print(f"  warning: failed to get match ids for a puuid: {e}")
            continue
        for mid in ids:
            if mid not in seen:
                seen.add(mid)
                match_ids.append(mid)
    print(f"[{bracket_name}] collected {len(match_ids)} new candidate match ids from {len(puuid_batch)} puuids")
    return match_ids


ROLE_CODE = {
    "TOP": "top", "JUNGLE": "jg", "MIDDLE": "mid",
    "BOTTOM": "bot", "UTILITY": "sup",
}

ROLES = ("top", "jg", "mid", "bot", "sup")
TEAMS = (100, 200)

RUNE_PERK_FIELDS = (
    "primary_style",
    "secondary_style",
    "keystone",
    "primary_rune_1",
    "primary_rune_2",
    "primary_rune_3",
    "secondary_rune_1",
    "secondary_rune_2",
    "stat_shard_offense",
    "stat_shard_flex",
    "stat_shard_defense",
    "all_runes",
)

PLAYER_FRAME_FIELDS = (
    "currentGold",
    "totalGold",
    "level",
    "xp",
    "minionsKilled",
    "jungleMinionsKilled",
    "timeEnemySpentControlled",
)

DAMAGE_STAT_FIELDS = (
    "magicDamageDone",
    "magicDamageDoneToChampions",
    "magicDamageTaken",
    "physicalDamageDone",
    "physicalDamageDoneToChampions",
    "physicalDamageTaken",
    "totalDamageDone",
    "totalDamageDoneToChampions",
    "totalDamageTaken",
    "trueDamageDone",
    "trueDamageDoneToChampions",
    "trueDamageTaken",
)

CHAMPION_STAT_FIELDS = (
    "abilityPower",
    "armor",
    "armorPen",
    "armorPenPercent",
    "attackDamage",
    "attackSpeed",
    "bonusArmorPenPercent",
    "bonusMagicPenPercent",
    "ccReduction",
    "cooldownReduction",
    "health",
    "healthMax",
    "healthRegen",
    "lifesteal",
    "magicPen",
    "magicPenPercent",
    "magicResist",
    "movementSpeed",
    "omnivamp",
    "physicalVamp",
    "power",
    "powerMax",
    "powerRegen",
    "spellVamp",
)

PLAYER_EVENT_FIELDS = (
    "kills",
    "deaths",
    "assists",
    "kda",
    "kill_participation",
    "solo_kills",
    "wards_placed",
    "wards_killed",
    "control_wards_placed",
)

def _snake(name):
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _role_map(participants, team_ids):
    result = {}
    for p in participants:
        if p["participantId"] in team_ids:
            role = ROLE_CODE.get(p.get("teamPosition") or p.get("individualPosition"))
            if role:
                result[role] = p["participantId"]
    return result


def _participant_lookup(participants):
    return {p["participantId"]: p for p in participants}


def _rune_setup(p):
    result = {field: "" for field in RUNE_PERK_FIELDS}
    perks = p.get("perks") or {}
    styles = perks.get("styles") or []

    if len(styles) >= 1:
        primary = styles[0] or {}
        result["primary_style"] = primary.get("style", "")
        selections = primary.get("selections") or []
        for i, sel in enumerate(selections[:4]):
            field = "keystone" if i == 0 else f"primary_rune_{i}"
            result[field] = sel.get("perk", "")

    if len(styles) >= 2:
        secondary = styles[1] or {}
        result["secondary_style"] = secondary.get("style", "")
        selections = secondary.get("selections") or []
        for i, sel in enumerate(selections[:2], start=1):
            result[f"secondary_rune_{i}"] = sel.get("perk", "")

    stat_perks = perks.get("statPerks") or {}
    result["stat_shard_offense"] = stat_perks.get("offense", "")
    result["stat_shard_flex"] = stat_perks.get("flex", "")
    result["stat_shard_defense"] = stat_perks.get("defense", "")

    ordered = [str(result[field]) for field in (
        "keystone", "primary_rune_1", "primary_rune_2", "primary_rune_3",
        "secondary_rune_1", "secondary_rune_2",
        "stat_shard_offense", "stat_shard_flex", "stat_shard_defense",
    ) if result[field] != ""]
    result["all_runes"] = "|".join(ordered)
    return result


def _team_rune_field(team_id, participants, rune_field):
    return "|".join(
        str(_rune_setup(participant).get(rune_field, ""))
        for participant in participants
        if participant["teamId"] == team_id
    )


def _match_context(match):
    info = match["info"]
    participants = info["participants"]
    version_parts = info.get("gameVersion", "0.0").split(".")
    patch = ".".join(version_parts[:2])

    def team_field(team_id, fn):
        return "|".join(fn(p) for p in participants if p["teamId"] == team_id)

    champs_100 = team_field(100, lambda p: p["championName"])
    champs_200 = team_field(200, lambda p: p["championName"])
    summs_100 = team_field(100, lambda p: f'{p["summoner1Id"]}-{p["summoner2Id"]}')
    summs_200 = team_field(200, lambda p: f'{p["summoner1Id"]}-{p["summoner2Id"]}')
    rune_context = {}
    for rune_field in RUNE_PERK_FIELDS:
        rune_context[f"team_100_{rune_field}s"] = _team_rune_field(100, participants, rune_field)
        rune_context[f"team_200_{rune_field}s"] = _team_rune_field(200, participants, rune_field)

    bans_100 = bans_200 = ""
    for t in info.get("teams", []):
        ban_ids = "|".join(str(b["championId"]) for b in t.get("bans", []) if b.get("championId", -1) != -1)
        if t["teamId"] == 100:
            bans_100 = ban_ids
        elif t["teamId"] == 200:
            bans_200 = ban_ids

    return {
        "patch": patch,
        "team_100_champs": champs_100, "team_200_champs": champs_200,
        "team_100_summs": summs_100, "team_200_summs": summs_200,
        **rune_context,
        "team_100_bans": bans_100, "team_200_bans": bans_200,
    }


EVENT_TYPES = ("kill", "tower", "plate", "dragon", "herald", "baron", "elder",
               "ward_placed", "ward_killed", "control_ward")


LANE_TOWER_PARTS = (
    ("top", "outer"), ("top", "inner"), ("top", "base"),
    ("mid", "outer"), ("mid", "inner"), ("mid", "base"),
    ("bot", "outer"), ("bot", "inner"), ("bot", "base"),
)

INDIVIDUAL_TOWER_STATE_FIELDS = []
for team_id in TEAMS:
    for lane, tier in LANE_TOWER_PARTS:
        INDIVIDUAL_TOWER_STATE_FIELDS.append(f"{lane}_{tier}_{team_id}_destroyed")
    INDIVIDUAL_TOWER_STATE_FIELDS.extend([
        f"nexus_tower_1_{team_id}_destroyed",
        f"nexus_tower_2_{team_id}_destroyed",
    ])

TOWER_SUMMARY_FIELDS = [
    "outer_towers_100_destroyed",
    "inner_towers_100_destroyed",
    "base_towers_100_destroyed",
    "nexus_towers_100_destroyed",
    "outer_towers_200_destroyed",
    "inner_towers_200_destroyed",
    "base_towers_200_destroyed",
    "nexus_towers_200_destroyed",
    "outer_tower_destroyed_diff",
    "inner_tower_destroyed_diff",
    "base_tower_destroyed_diff",
    "nexus_tower_destroyed_diff",
    "top_towers_100_destroyed",
    "mid_towers_100_destroyed",
    "bot_towers_100_destroyed",
    "top_towers_200_destroyed",
    "mid_towers_200_destroyed",
    "bot_towers_200_destroyed",
    "top_tower_destroyed_diff",
    "mid_tower_destroyed_diff",
    "bot_tower_destroyed_diff",
]

TOWER_STATE_FIELDS = INDIVIDUAL_TOWER_STATE_FIELDS + TOWER_SUMMARY_FIELDS

INHIB_LANES = ("top", "mid", "bot")

INHIB_STATE_FIELDS = [
    f"{lane}_inhib_{team_id}_destroyed"
    for team_id in TEAMS
    for lane in INHIB_LANES
]


def _tower_lane(lane_type):
    return {
        "TOP_LANE": "top",
        "MID_LANE": "mid",
        "BOT_LANE": "bot",
    }.get(lane_type)


def _tower_tier(tower_type):
    return {
        "OUTER_TURRET": "outer",
        "INNER_TURRET": "inner",
        "BASE_TURRET": "base",
        "NEXUS_TURRET": "nexus",
    }.get(tower_type)


def _build_inhib_destroy_log(frames):
    """Like _build_tower_destroy_log, but for inhibitors. Inhibitors can
    respawn and be destroyed again, but the state features below only track
    whether a lane's inhibitor has EVER been destroyed by a given snapshot,
    the same simplification already used for towers -- it keeps the feature
    a plain 0/1 rather than needing to model respawn timers."""
    inhib_events = []
    for frame in frames:
        for ev in frame.get("events", []):
            if ev.get("type") != "BUILDING_KILL":
                continue
            if ev.get("buildingType") != "INHIBITOR_BUILDING":
                continue
            ts = ev.get("timestamp")
            destroyed_team = ev.get("teamId")
            lane = _tower_lane(ev.get("laneType"))
            if ts is None or destroyed_team not in TEAMS or lane not in INHIB_LANES:
                continue
            inhib_events.append((ts, destroyed_team, lane))
    inhib_events.sort(key=lambda x: x[0])
    return inhib_events


def _inhib_state_features(inhib_destroy_log, up_to_ts):
    state = {field: 0 for field in INHIB_STATE_FIELDS}
    for ts, destroyed_team, lane in inhib_destroy_log:
        if ts > up_to_ts:
            break
        state[f"{lane}_inhib_{destroyed_team}_destroyed"] = 1
    return state


def _build_tower_destroy_log(frames):
    tower_events = []
    for frame in frames:
        for ev in frame.get("events", []):
            if ev.get("type") != "BUILDING_KILL":
                continue
            if ev.get("buildingType") != "TOWER_BUILDING":
                continue
            ts = ev.get("timestamp")
            destroyed_team = ev.get("teamId")
            tier = _tower_tier(ev.get("towerType"))
            lane = _tower_lane(ev.get("laneType"))
            if ts is None or destroyed_team not in TEAMS or not tier:
                continue
            tower_events.append((ts, destroyed_team, lane, tier))
    tower_events.sort(key=lambda x: x[0])
    return tower_events


def _tower_state_features(tower_destroy_log, up_to_ts):
    state = {field: 0 for field in TOWER_STATE_FIELDS}
    assigned_nexus = {100: 0, 200: 0}
    seen_lane_towers = set()

    for ts, destroyed_team, lane, tier in tower_destroy_log:
        if ts > up_to_ts:
            break
        if tier == "nexus":
            if assigned_nexus[destroyed_team] < 2:
                assigned_nexus[destroyed_team] += 1
                key = f"nexus_tower_{assigned_nexus[destroyed_team]}_{destroyed_team}_destroyed"
                state[key] = 1
            continue
        if lane not in {"top", "mid", "bot"} or tier not in {"outer", "inner", "base"}:
            continue
        key = f"{lane}_{tier}_{destroyed_team}_destroyed"
        if key not in seen_lane_towers:
            seen_lane_towers.add(key)
            state[key] = 1

    for team_id in TEAMS:
        state[f"outer_towers_{team_id}_destroyed"] = sum(
            state[f"{lane}_outer_{team_id}_destroyed"] for lane in ("top", "mid", "bot")
        )
        state[f"inner_towers_{team_id}_destroyed"] = sum(
            state[f"{lane}_inner_{team_id}_destroyed"] for lane in ("top", "mid", "bot")
        )
        state[f"base_towers_{team_id}_destroyed"] = sum(
            state[f"{lane}_base_{team_id}_destroyed"] for lane in ("top", "mid", "bot")
        )
        state[f"nexus_towers_{team_id}_destroyed"] = (
            state[f"nexus_tower_1_{team_id}_destroyed"] + state[f"nexus_tower_2_{team_id}_destroyed"]
        )
        for lane in ("top", "mid", "bot"):
            state[f"{lane}_towers_{team_id}_destroyed"] = sum(
                state[f"{lane}_{tier}_{team_id}_destroyed"] for tier in ("outer", "inner", "base")
            )

    for tier in ("outer", "inner", "base", "nexus"):
        state[f"{tier}_tower_destroyed_diff"] = (
            state[f"{tier}_towers_200_destroyed"] - state[f"{tier}_towers_100_destroyed"]
        )
    for lane in ("top", "mid", "bot"):
        state[f"{lane}_tower_destroyed_diff"] = (
            state[f"{lane}_towers_200_destroyed"] - state[f"{lane}_towers_100_destroyed"]
        )

    return state


def _build_event_log(frames, team100_ids, team200_ids):
    team_event_log = []
    player_event_log = []
    first_blood = (None, None)
    first_tower = (None, None)

    def team_of(pid):
        if pid in team100_ids:
            return 100
        if pid in team200_ids:
            return 200
        return None

    for frame in frames:
        for ev in frame.get("events", []):
            etype = ev.get("type")
            ts = ev.get("timestamp")
            if ts is None:
                continue

            if etype == "CHAMPION_KILL":
                killer = ev.get("killerId", 0)
                victim = ev.get("victimId", 0)
                assists = ev.get("assistingParticipantIds") or []
                team = team_of(killer)
                victim_team = team_of(victim)

                if team:
                    team_event_log.append((ts, team, "kill"))
                    player_event_log.append((ts, killer, "kills", 1))
                    if not assists:
                        player_event_log.append((ts, killer, "solo_kills", 1))
                    if first_blood[0] is None:
                        first_blood = (team, ts)
                if victim_team:
                    player_event_log.append((ts, victim, "deaths", 1))
                for assist_pid in assists:
                    if team_of(assist_pid):
                        player_event_log.append((ts, assist_pid, "assists", 1))

            elif etype == "BUILDING_KILL":
                killer = ev.get("killerId", 0)
                team = team_of(killer)
                if team:
                    team_event_log.append((ts, team, "tower"))
                    if first_tower[0] is None:
                        first_tower = (team, ts)

            elif etype == "TURRET_PLATE_DESTROYED":
                killer = ev.get("killerId", 0)
                team = team_of(killer)
                if team:
                    team_event_log.append((ts, team, "plate"))

            elif etype == "ELITE_MONSTER_KILL":
                killer = ev.get("killerId", 0)
                team = team_of(killer)
                monster = ev.get("monsterType")
                if team and monster == "DRAGON":
                    subtype = ev.get("monsterSubType", "")
                    event_name = "elder" if subtype == "ELDER_DRAGON" else "dragon"
                    team_event_log.append((ts, team, event_name))
                elif team and monster == "RIFTHERALD":
                    team_event_log.append((ts, team, "herald"))
                elif team and monster == "BARON_NASHOR":
                    team_event_log.append((ts, team, "baron"))

            elif etype == "WARD_PLACED":
                creator = ev.get("creatorId", 0)
                team = team_of(creator)
                if team:
                    team_event_log.append((ts, team, "ward_placed"))
                    player_event_log.append((ts, creator, "wards_placed", 1))
                    if ev.get("wardType") == "CONTROL_WARD":
                        team_event_log.append((ts, team, "control_ward"))
                        player_event_log.append((ts, creator, "control_wards_placed", 1))

            elif etype == "WARD_KILL":
                killer = ev.get("killerId", 0)
                team = team_of(killer)
                if team:
                    team_event_log.append((ts, team, "ward_killed"))
                    player_event_log.append((ts, killer, "wards_killed", 1))

    return team_event_log, player_event_log, first_blood, first_tower


def _empty_player_event_counts(participant_ids):
    keys = [k for k in PLAYER_EVENT_FIELDS if k not in ("kda", "kill_participation")]
    return {pid: {key: 0 for key in keys} for pid in participant_ids}


def _make_player_counts_advancer(player_event_log, participant_ids):
    event_log = sorted(player_event_log, key=lambda item: item[0])
    counts = _empty_player_event_counts(participant_ids)
    team_kills = {100: 0, 200: 0}
    idx = 0
    n = len(event_log)

    def advance(up_to_ts):
        nonlocal idx
        while idx < n and event_log[idx][0] <= up_to_ts:
            ts, pid, stat, amount = event_log[idx]
            if pid in counts and stat in counts[pid]:
                counts[pid][stat] += amount
                if stat == "kills":
                    team = 100 if 1 <= pid <= 5 else 200
                    team_kills[team] += amount
            idx += 1

        result = {}
        for pid, c in counts.items():
            team = 100 if 1 <= pid <= 5 else 200
            kills = c.get("kills", 0)
            deaths = c.get("deaths", 0)
            assists = c.get("assists", 0)
            derived = dict(c)
            derived["kda"] = round((kills + assists) / max(1, deaths), 3)
            derived["kill_participation"] = (
                round((kills + assists) / team_kills[team], 3) if team_kills[team] else 0.0
            )
            result[pid] = derived
        return result

    return advance


def _player_snapshot_values(pf, event_counts):
    if not pf:
        values = {}
        for field in PLAYER_FRAME_FIELDS:
            values[_snake(field)] = ""
        values["cs"] = ""
        for field in DAMAGE_STAT_FIELDS:
            values[_snake(field)] = ""
        for field in CHAMPION_STAT_FIELDS:
            values[_snake(field)] = ""
        values["hp_pct"] = ""
        values["power_pct"] = ""
        values["x"] = ""
        values["y"] = ""
        for field in PLAYER_EVENT_FIELDS:
            values[field] = ""
        return values

    values = {}
    for field in PLAYER_FRAME_FIELDS:
        values[_snake(field)] = pf.get(field, 0)

    minions = pf.get("minionsKilled", 0)
    jungle_minions = pf.get("jungleMinionsKilled", 0)
    values["cs"] = minions + jungle_minions

    damage_stats = pf.get("damageStats", {})
    for field in DAMAGE_STAT_FIELDS:
        values[_snake(field)] = damage_stats.get(field, 0)

    champion_stats = pf.get("championStats", {})
    for field in CHAMPION_STAT_FIELDS:
        values[_snake(field)] = champion_stats.get(field, 0)

    health = champion_stats.get("health", 0)
    health_max = champion_stats.get("healthMax", 0)
    power = champion_stats.get("power", 0)
    power_max = champion_stats.get("powerMax", 0)
    values["hp_pct"] = round(health / health_max * 100, 1) if health_max else 0.0
    values["power_pct"] = round(power / power_max * 100, 1) if power_max else 0.0

    pos = pf.get("position") or {}
    values["x"] = pos.get("x", "")
    values["y"] = pos.get("y", "")

    for field in PLAYER_EVENT_FIELDS:
        values[field] = event_counts.get(field, 0)

    return values


def _add_role_static_context(row, role, team_id, participant):
    prefix = f"{role}_{team_id}"
    if participant:
        row[f"{prefix}_participant_id"] = participant.get("participantId", "")
        row[f"{prefix}_champ"] = participant.get("championName", "")
        row[f"{prefix}_champion_id"] = participant.get("championId", "")
        row[f"{prefix}_summoner1_id"] = participant.get("summoner1Id", "")
        row[f"{prefix}_summoner2_id"] = participant.get("summoner2Id", "")
        rune_setup = _rune_setup(participant)
        for rune_field in RUNE_PERK_FIELDS:
            row[f"{prefix}_{rune_field}"] = rune_setup.get(rune_field, "")
    else:
        row[f"{prefix}_participant_id"] = ""
        row[f"{prefix}_champ"] = ""
        row[f"{prefix}_champion_id"] = ""
        row[f"{prefix}_summoner1_id"] = ""
        row[f"{prefix}_summoner2_id"] = ""
        for rune_field in RUNE_PERK_FIELDS:
            row[f"{prefix}_{rune_field}"] = ""


def _add_role_player_snapshot(row, role, team_id, values):
    prefix = f"{role}_{team_id}"
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


def extract_snapshots(match, timeline):
    info = match["info"]
    participants = info["participants"]
    participant_by_id = _participant_lookup(participants)
    team100_ids = [p["participantId"] for p in participants if p["teamId"] == 100]
    team200_ids = [p["participantId"] for p in participants if p["teamId"] == 200]
    all_participant_ids = team100_ids + team200_ids

    team100_win = next((t["win"] for t in info["teams"] if t["teamId"] == 100), None)
    if team100_win is None:
        return []

    frames = timeline["info"]["frames"]
    if not frames:
        return []

    game_duration_ms = info.get("gameDuration", 0) * (
        1000 if info.get("gameDuration", 0) < 100000 else 1
    )
    game_duration_ms = max(game_duration_ms, frames[-1]["timestamp"])

    context = _match_context(match)
    role_map_100 = _role_map(participants, team100_ids)
    role_map_200 = _role_map(participants, team200_ids)
    common_roles = set(role_map_100) & set(role_map_200)

    team_event_log, player_event_log, first_blood, first_tower = _build_event_log(frames, team100_ids, team200_ids)
    tower_destroy_log = _build_tower_destroy_log(frames)
    inhib_destroy_log = _build_inhib_destroy_log(frames)

    sorted_team_event_log = sorted(team_event_log, key=lambda item: item[0])
    team_counts_state = {100: {k: 0 for k in EVENT_TYPES}, 200: {k: 0 for k in EVENT_TYPES}}
    team_event_idx = 0
    team_event_n = len(sorted_team_event_log)

    def cumulative_counts(up_to_ts):
        nonlocal team_event_idx
        while team_event_idx < team_event_n and sorted_team_event_log[team_event_idx][0] <= up_to_ts:
            _, team, etype = sorted_team_event_log[team_event_idx]
            team_counts_state[team][etype] += 1
            team_event_idx += 1
        return team_counts_state

    player_counts_advancer = _make_player_counts_advancer(player_event_log, all_participant_ids)

    match_id = match["metadata"]["matchId"]
    rows = []

    for frame in frames:
        t = frame["timestamp"]
        pf = frame.get("participantFrames", {})

        counts = cumulative_counts(t)
        c100, c200 = counts[100], counts[200]
        player_counts = player_counts_advancer(t)

        row = {
            "match_id": match_id,
            "timestamp_sec": t // 1000,
            "game_duration_sec": game_duration_ms // 1000,
            "team_100_win": int(team100_win),

            **context,
            "first_blood_team": first_blood[0] or "",
            "first_blood_time_sec": (first_blood[1] // 1000) if first_blood[1] else "",
            "first_tower_team": first_tower[0] or "",
            "first_tower_time_sec": (first_tower[1] // 1000) if first_tower[1] else "",

            "plates_100": c100["plate"], "plates_200": c200["plate"],
            "dragons_100": c100["dragon"], "dragons_200": c200["dragon"],
            "heralds_100": c100["herald"], "heralds_200": c200["herald"],
            "barons_100": c100["baron"], "barons_200": c200["baron"],
            "elders_100": c100["elder"], "elders_200": c200["elder"],
        }
        row.update(_tower_state_features(tower_destroy_log, t))
        row.update(_inhib_state_features(inhib_destroy_log, t))

        for role in ROLES:
            pid100 = role_map_100.get(role)
            pid200 = role_map_200.get(role)
            p100 = participant_by_id.get(pid100) if pid100 else None
            p200 = participant_by_id.get(pid200) if pid200 else None
            _add_role_static_context(row, role, 100, p100)
            _add_role_static_context(row, role, 200, p200)

            if role in common_roles:
                pf100 = pf.get(str(pid100), {})
                pf200 = pf.get(str(pid200), {})
                vals100 = _player_snapshot_values(pf100, player_counts.get(pid100, {}))
                vals200 = _player_snapshot_values(pf200, player_counts.get(pid200, {}))
            else:
                vals100 = _player_snapshot_values({}, {})
                vals200 = _player_snapshot_values({}, {})

            _add_role_player_snapshot(row, role, 100, vals100)
            _add_role_player_snapshot(row, role, 200, vals200)

        rows.append(row)

    return rows


# The columns of the final dataset. Not everything extract_snapshots
# computes is kept: identifiers/strings that don't help a model, values
# already implied by other kept columns (all_runes, cs, hp_pct/power_pct,
# total_damage_done(_to_champions)), and kda/kill_participation, which are
# derived stats better computed from kills/deaths/assists downstream.

GLOBAL_FIELDS = (
    "match_id", "timestamp_sec", "game_duration_sec", "team_100_win", "patch",
    "team_100_bans", "team_200_bans",
    "first_blood_team", "first_blood_time_sec", "first_tower_team", "first_tower_time_sec",
    "plates_100", "plates_200",
    "dragons_100", "dragons_200",
    "heralds_100", "heralds_200",
    "barons_100", "barons_200",
    "elders_100", "elders_200",
    "nexus_tower_1_100_destroyed", "nexus_tower_2_100_destroyed",
    "nexus_tower_1_200_destroyed", "nexus_tower_2_200_destroyed",
    *INHIB_STATE_FIELDS,
)

# Only top/mid/bot have an individual lane with its own towers to track.
LANE_ROLES = ("top", "mid", "bot")
TOWER_LANE_TIERS = ("outer", "inner", "base")

# Kept from each role's static per-game context (participant_id, champ name and
# all_runes are dropped; champion_id is the numeric id we actually model on).
ROLE_STATIC_FIELDS = (
    "champion_id", "summoner1_id", "summoner2_id",
    "primary_style", "secondary_style", "keystone",
    "primary_rune_1", "primary_rune_2", "primary_rune_3",
    "secondary_rune_1", "secondary_rune_2",
    "stat_shard_offense", "stat_shard_flex", "stat_shard_defense",
)

# Kept from each role's per-frame snapshot.
ROLE_SNAPSHOT_FIELDS = (
    *(_snake(field) for field in PLAYER_FRAME_FIELDS),
    *(_snake(field) for field in DAMAGE_STAT_FIELDS if not _snake(field).startswith("total_damage")),
    *(_snake(field) for field in CHAMPION_STAT_FIELDS),
    "x", "y",
    "kills", "deaths", "assists", "solo_kills",
    "wards_placed", "wards_killed", "control_wards_placed",
)


def _build_csv_fields():
    fields = ["rank"]
    fields.extend(GLOBAL_FIELDS)
    for role in ROLES:
        if role in LANE_ROLES:
            for tier in TOWER_LANE_TIERS:
                for team in TEAMS:
                    fields.append(f"{role}_{tier}_{team}_destroyed")
        for team in TEAMS:
            fields.extend(f"{role}_{team}_{field}" for field in ROLE_STATIC_FIELDS)
            fields.extend(f"{role}_{team}_{field}" for field in ROLE_SNAPSHOT_FIELDS)
    return fields


CSV_FIELDS = _build_csv_fields()


GLOBAL_PROCESSED_PATH = OUTPUT_DIR / "all_processed_matches.json"


def load_global_processed():
    if GLOBAL_PROCESSED_PATH.exists():
        return set(json.loads(GLOBAL_PROCESSED_PATH.read_text()))
    return set()


def save_global_processed(global_processed):
    GLOBAL_PROCESSED_PATH.write_text(json.dumps(list(global_processed)))


STRING_COLUMNS = {"rank", "match_id", "patch", "team_100_bans", "team_200_bans"}


def _is_string_column(column_name):
    return column_name in STRING_COLUMNS


def _clean_rows_for_db(rows, rank):
    clean_rows = []
    for row in rows:
        row = {**row, "rank": rank}
        clean_rows.append({field: (None if row.get(field, None) == "" else row.get(field, None)) for field in CSV_FIELDS})

    df = pd.DataFrame(clean_rows, columns=CSV_FIELDS)
    for column in CSV_FIELDS:
        if _is_string_column(column):
            df[column] = df[column].astype("string")
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _write_snapshot_rows(rows, rank, engine):
    if not rows:
        return 0
    df = _clean_rows_for_db(rows, rank)
    df.to_sql(
        MATCH_SNAPSHOTS_TABLE,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    return len(df)


def process_bracket(bracket_name, global_processed, engine):
    print(f"\n=== Tier: {bracket_name} ===")
    pool_cache = OUTPUT_DIR / f"{bracket_name}_puuid_pool.json"
    processed_path = OUTPUT_DIR / f"{bracket_name}_processed_matches.json"
    # Tracks which puuids we've already pulled match ids for, so a rerun
    # doesn't waste API calls re-querying the same puuid.
    queried_path = OUTPUT_DIR / f"{bracket_name}_queried_puuids.json"

    processed = set(json.loads(processed_path.read_text())) if processed_path.exists() else set()
    queried = set(json.loads(queried_path.read_text())) if queried_path.exists() else set()
    print(f"[{bracket_name}] already processed: {len(processed)} matches")

    if len(processed) >= TARGET_GAMES_PER_BRACKET:
        print(f"[{bracket_name}] target already met, skipping.")
        return

    pool_target = max(TARGET_GAMES_PER_BRACKET * 2 // MATCHES_PER_PLAYER, 200)
    pool = collect_puuid_pool(bracket_name, pool_target, pool_cache)

    buffered_rows = []
    buffered_match_ids = []

    def flush_buffer():
        nonlocal buffered_rows, buffered_match_ids
        if not buffered_rows:
            return
        n_rows = _write_snapshot_rows(buffered_rows, bracket_name, engine)
        for mid in buffered_match_ids:
            processed.add(mid)
            global_processed.add(mid)
        processed_path.write_text(json.dumps(list(processed)))
        save_global_processed(global_processed)
        print(
            f"[{bracket_name}] wrote {len(buffered_match_ids)} matches / "
            f"{n_rows} rows -> {MATCH_SNAPSHOTS_TABLE}"
        )
        print(f"[{bracket_name}] processed {len(processed)}/{TARGET_GAMES_PER_BRACKET} matches")
        buffered_rows = []
        buffered_match_ids = []

    # Loop over batches of not-yet-queried puuids, growing the pool once
    # it's exhausted, until the bracket hits its target or runs dry.
    while len(processed) + len(buffered_match_ids) < TARGET_GAMES_PER_BRACKET:
        unqueried = [p for p in pool if p not in queried]

        if not unqueried:
            new_pool_target = len(pool) + POOL_GROWTH_STEP
            print(f"[{bracket_name}] pool exhausted, growing to {new_pool_target} puuids...")
            pool = collect_puuid_pool(bracket_name, new_pool_target, pool_cache)
            unqueried = [p for p in pool if p not in queried]
            if not unqueried:
                print(
                    f"[{bracket_name}] no more puuids available in this bracket, "
                    f"stopping short at {len(processed)}/{TARGET_GAMES_PER_BRACKET}."
                )
                break

        random.shuffle(unqueried)
        batch = unqueried[:QUERIED_BATCH_SIZE]

        candidate_ids = collect_match_ids(bracket_name, batch, processed, global_processed)

        queried.update(batch)
        queried_path.write_text(json.dumps(list(queried)))

        for match_id in candidate_ids:
            if len(processed) + len(buffered_match_ids) >= TARGET_GAMES_PER_BRACKET:
                break
            if match_id in global_processed or match_id in processed or match_id in buffered_match_ids:
                continue
            try:
                # Check the cheap fields on `match` (queue, patch, duration)
                # before paying for a `get_timeline` call.
                match = get_match(match_id)
                if not match:
                    continue
                if match["info"].get("queueId") != QUEUE_ID:
                    continue
                game_version = match["info"].get("gameVersion", "")
                patch = ".".join(game_version.split(".")[:2])
                if patch != TARGET_PATCH:
                    continue
                game_duration_sec = match["info"].get("gameDuration", 0)
                if game_duration_sec >= 100000:
                    game_duration_sec = game_duration_sec // 1000
                if game_duration_sec < MIN_GAME_DURATION_SEC:
                    continue
                timeline = get_timeline(match_id)
                if not timeline:
                    continue
                rows = extract_snapshots(match, timeline)
                if not rows:
                    continue
                buffered_rows.extend(rows)
                buffered_match_ids.append(match_id)
                if len(buffered_match_ids) >= MATCH_CHUNK_SIZE:
                    flush_buffer()
            except RuntimeError as e:
                print(f"  warning: skipping {match_id}: {e}")
                continue

    flush_buffer()

    processed_path.write_text(json.dumps(list(processed)))
    save_global_processed(global_processed)
    print(f"[{bracket_name}] done: {len(processed)} matches -> {MATCH_SNAPSHOTS_TABLE}")


def main():
    engine = get_engine()
    global_processed = load_global_processed()
    for bracket_name in BRACKETS:
        process_bracket(bracket_name, global_processed, engine)
    print("\nAll brackets complete.")


if __name__ == "__main__":
    run_with_file_logging(LOG_DIR, "collect_data", main)