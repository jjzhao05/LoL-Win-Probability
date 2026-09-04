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

load_dotenv()


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

TARGET_PATCH = "14.18"

PARQUET_MATCH_CHUNK_SIZE = 25

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


def collect_match_ids(bracket_name, puuid_pool, target_games, processed_ids, global_seen):
    match_ids = []
    seen = set(processed_ids) | set(global_seen)
    for puuid in puuid_pool:
        if len(match_ids) + len(processed_ids) >= target_games:
            break
        try:
            ids = get_ranked_match_ids(puuid, MATCHES_PER_PLAYER)
        except RuntimeError as e:
            print(f"  warning: failed to get match ids for a puuid: {e}")
            continue
        for mid in ids:
            if mid not in seen:
                seen.add(mid)
                match_ids.append(mid)
    print(f"[{bracket_name}] collected {len(match_ids)} new candidate match ids")
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

ROLE_DIFF_FIELDS = (
    "currentGold",
    "totalGold",
    "level",
    "xp",
    "cs",
    "minionsKilled",
    "jungleMinionsKilled",
    "timeEnemySpentControlled",
    *DAMAGE_STAT_FIELDS,
    *CHAMPION_STAT_FIELDS,
    *PLAYER_EVENT_FIELDS,
)


def _snake(name):
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _team_farm_totals(participant_frames, ids):
    gold = xp = cs = 0
    for pid in ids:
        pf = participant_frames.get(str(pid))
        if not pf:
            continue
        gold += pf.get("totalGold", 0)
        xp += pf.get("xp", 0)
        cs += pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)
    return gold, xp, cs


def _team_combat_totals(participant_frames, ids):
    dmg_to_champs = health = health_max = armor = mr = ad = ap = 0
    for pid in ids:
        pf = participant_frames.get(str(pid))
        if not pf:
            continue
        dmg_to_champs += pf.get("damageStats", {}).get("totalDamageDoneToChampions", 0)
        stats = pf.get("championStats", {})
        health += stats.get("health", 0)
        health_max += stats.get("healthMax", 0)
        armor += stats.get("armor", 0)
        mr += stats.get("magicResist", 0)
        ad += stats.get("attackDamage", 0)
        ap += stats.get("abilityPower", 0)
    hp_pct = (health / health_max * 100) if health_max else 0
    return dmg_to_champs, hp_pct, armor, mr, ad, ap


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


def _keystone(p):
    return str(_rune_setup(p).get("keystone", ""))


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


def _add_role_diffs(row, role, vals100, vals200):
    for field in ROLE_DIFF_FIELDS:
        key = _snake(field)
        v100 = vals100.get(key, "")
        v200 = vals200.get(key, "")
        if v100 == "" or v200 == "":
            row[f"{role}_{key}_diff"] = ""
        else:
            row[f"{role}_{key}_diff"] = round(v100 - v200, 3) if isinstance(v100, float) or isinstance(v200, float) else v100 - v200

    row[f"{role}_gold_diff"] = row[f"{role}_total_gold_diff"]
    row[f"{role}_dmg_diff"] = row[f"{role}_total_damage_done_to_champions_diff"]


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

        gold100, xp100, cs100 = _team_farm_totals(pf, team100_ids)
        gold200, xp200, cs200 = _team_farm_totals(pf, team200_ids)

        dmg100, hp100, armor100, mr100, ad100, ap100 = _team_combat_totals(pf, team100_ids)
        dmg200, hp200, armor200, mr200, ad200, ap200 = _team_combat_totals(pf, team200_ids)

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

            "gold_100": gold100, "gold_200": gold200, "gold_diff": gold100 - gold200,
            "xp_100": xp100, "xp_200": xp200, "xp_diff": xp100 - xp200,
            "cs_100": cs100, "cs_200": cs200, "cs_diff": cs100 - cs200,

            "dmg_to_champs_100": dmg100, "dmg_to_champs_200": dmg200,
            "dmg_to_champs_diff": dmg100 - dmg200,
            "hp_pct_100": round(hp100, 1), "hp_pct_200": round(hp200, 1),
            "hp_pct_diff": round(hp100 - hp200, 1),
            "armor_100": armor100, "armor_200": armor200,
            "mr_100": mr100, "mr_200": mr200,
            "ad_100": ad100, "ad_200": ad200,
            "ap_100": ap100, "ap_200": ap200,

            "kills_100": c100["kill"], "kills_200": c200["kill"], "kill_diff": c100["kill"] - c200["kill"],
            "towers_100": c100["tower"], "towers_200": c200["tower"], "tower_diff": c100["tower"] - c200["tower"],
            "plates_100": c100["plate"], "plates_200": c200["plate"], "plate_diff": c100["plate"] - c200["plate"],
            "dragons_100": c100["dragon"], "dragons_200": c200["dragon"], "dragon_diff": c100["dragon"] - c200["dragon"],
            "heralds_100": c100["herald"], "heralds_200": c200["herald"], "herald_diff": c100["herald"] - c200["herald"],
            "barons_100": c100["baron"], "barons_200": c200["baron"], "baron_diff": c100["baron"] - c200["baron"],
            "elders_100": c100["elder"], "elders_200": c200["elder"],

            "wards_placed_100": c100["ward_placed"], "wards_placed_200": c200["ward_placed"],
            "wards_killed_100": c100["ward_killed"], "wards_killed_200": c200["ward_killed"],
            "control_wards_100": c100["control_ward"], "control_wards_200": c200["control_ward"],
        }
        row.update(_tower_state_features(tower_destroy_log, t))

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
            _add_role_diffs(row, role, vals100, vals200)

        rows.append(row)

    return rows


CSV_FIELDS = ['match_id', 'timestamp_sec', 'game_duration_sec', 'team_100_win', 'patch', 'team_100_bans', 'team_200_bans', 'first_blood_team', 'first_blood_time_sec', 'first_tower_team', 'first_tower_time_sec', 'kills_100', 'kills_200', 'kill_diff', 'towers_100', 'towers_200', 'plates_100', 'plates_200', 'plate_diff', 'dragons_100', 'dragons_200', 'dragon_diff', 'heralds_100', 'heralds_200', 'herald_diff', 'barons_100', 'barons_200', 'baron_diff', 'elders_100', 'elders_200', 'wards_placed_100', 'wards_placed_200', 'wards_killed_100', 'wards_killed_200', 'control_wards_100', 'control_wards_200', 'top_outer_100_destroyed', 'top_inner_100_destroyed', 'top_base_100_destroyed', 'mid_outer_100_destroyed', 'mid_inner_100_destroyed', 'mid_base_100_destroyed', 'bot_outer_100_destroyed', 'bot_inner_100_destroyed', 'bot_base_100_destroyed', 'nexus_tower_1_100_destroyed', 'nexus_tower_2_100_destroyed', 'top_outer_200_destroyed', 'top_inner_200_destroyed', 'top_base_200_destroyed', 'mid_outer_200_destroyed', 'mid_inner_200_destroyed', 'mid_base_200_destroyed', 'bot_outer_200_destroyed', 'bot_inner_200_destroyed', 'bot_base_200_destroyed', 'nexus_tower_1_200_destroyed', 'nexus_tower_2_200_destroyed', 'outer_towers_100_destroyed', 'inner_towers_100_destroyed', 'base_towers_100_destroyed', 'nexus_towers_100_destroyed', 'outer_towers_200_destroyed', 'inner_towers_200_destroyed', 'base_towers_200_destroyed', 'nexus_towers_200_destroyed', 'top_towers_100_destroyed', 'mid_towers_100_destroyed', 'bot_towers_100_destroyed', 'top_towers_200_destroyed', 'mid_towers_200_destroyed', 'bot_towers_200_destroyed', 'top_100_participant_id', 'top_100_champ', 'top_100_champion_id', 'top_100_summoner1_id', 'top_100_summoner2_id', 'top_100_primary_style', 'top_100_secondary_style', 'top_100_keystone', 'top_100_primary_rune_1', 'top_100_primary_rune_2', 'top_100_primary_rune_3', 'top_100_secondary_rune_1', 'top_100_secondary_rune_2', 'top_100_stat_shard_offense', 'top_100_stat_shard_flex', 'top_100_stat_shard_defense', 'top_100_all_runes', 'top_200_participant_id', 'top_200_champ', 'top_200_champion_id', 'top_200_summoner1_id', 'top_200_summoner2_id', 'top_200_primary_style', 'top_200_secondary_style', 'top_200_keystone', 'top_200_primary_rune_1', 'top_200_primary_rune_2', 'top_200_primary_rune_3', 'top_200_secondary_rune_1', 'top_200_secondary_rune_2', 'top_200_stat_shard_offense', 'top_200_stat_shard_flex', 'top_200_stat_shard_defense', 'top_200_all_runes', 'top_100_current_gold', 'top_100_total_gold', 'top_100_level', 'top_100_xp', 'top_100_minions_killed', 'top_100_jungle_minions_killed', 'top_100_time_enemy_spent_controlled', 'top_100_cs', 'top_100_magic_damage_done', 'top_100_magic_damage_done_to_champions', 'top_100_magic_damage_taken', 'top_100_physical_damage_done', 'top_100_physical_damage_done_to_champions', 'top_100_physical_damage_taken', 'top_100_total_damage_done', 'top_100_total_damage_done_to_champions', 'top_100_total_damage_taken', 'top_100_true_damage_done', 'top_100_true_damage_done_to_champions', 'top_100_true_damage_taken', 'top_100_ability_power', 'top_100_armor', 'top_100_armor_pen', 'top_100_armor_pen_percent', 'top_100_attack_damage', 'top_100_attack_speed', 'top_100_bonus_armor_pen_percent', 'top_100_bonus_magic_pen_percent', 'top_100_cc_reduction', 'top_100_cooldown_reduction', 'top_100_health', 'top_100_health_max', 'top_100_health_regen', 'top_100_lifesteal', 'top_100_magic_pen', 'top_100_magic_pen_percent', 'top_100_magic_resist', 'top_100_movement_speed', 'top_100_omnivamp', 'top_100_physical_vamp', 'top_100_power', 'top_100_power_max', 'top_100_power_regen', 'top_100_spell_vamp', 'top_100_hp_pct', 'top_100_power_pct', 'top_100_x', 'top_100_y', 'top_100_kills', 'top_100_deaths', 'top_100_assists', 'top_100_kda', 'top_100_kill_participation', 'top_100_solo_kills', 'top_100_wards_placed', 'top_100_wards_killed', 'top_100_control_wards_placed', 'top_200_current_gold', 'top_200_total_gold', 'top_200_level', 'top_200_xp', 'top_200_minions_killed', 'top_200_jungle_minions_killed', 'top_200_time_enemy_spent_controlled', 'top_200_cs', 'top_200_magic_damage_done', 'top_200_magic_damage_done_to_champions', 'top_200_magic_damage_taken', 'top_200_physical_damage_done', 'top_200_physical_damage_done_to_champions', 'top_200_physical_damage_taken', 'top_200_total_damage_done', 'top_200_total_damage_done_to_champions', 'top_200_total_damage_taken', 'top_200_true_damage_done', 'top_200_true_damage_done_to_champions', 'top_200_true_damage_taken', 'top_200_ability_power', 'top_200_armor', 'top_200_armor_pen', 'top_200_armor_pen_percent', 'top_200_attack_damage', 'top_200_attack_speed', 'top_200_bonus_armor_pen_percent', 'top_200_bonus_magic_pen_percent', 'top_200_cc_reduction', 'top_200_cooldown_reduction', 'top_200_health', 'top_200_health_max', 'top_200_health_regen', 'top_200_lifesteal', 'top_200_magic_pen', 'top_200_magic_pen_percent', 'top_200_magic_resist', 'top_200_movement_speed', 'top_200_omnivamp', 'top_200_physical_vamp', 'top_200_power', 'top_200_power_max', 'top_200_power_regen', 'top_200_spell_vamp', 'top_200_hp_pct', 'top_200_power_pct', 'top_200_x', 'top_200_y', 'top_200_kills', 'top_200_deaths', 'top_200_assists', 'top_200_kda', 'top_200_kill_participation', 'top_200_solo_kills', 'top_200_wards_placed', 'top_200_wards_killed', 'top_200_control_wards_placed', 'top_current_gold_diff', 'top_total_gold_diff', 'top_level_diff', 'top_xp_diff', 'top_cs_diff', 'top_minions_killed_diff', 'top_jungle_minions_killed_diff', 'top_time_enemy_spent_controlled_diff', 'top_magic_damage_done_diff', 'top_magic_damage_done_to_champions_diff', 'top_magic_damage_taken_diff', 'top_physical_damage_done_diff', 'top_physical_damage_done_to_champions_diff', 'top_physical_damage_taken_diff', 'top_total_damage_done_diff', 'top_total_damage_done_to_champions_diff', 'top_total_damage_taken_diff', 'top_true_damage_done_diff', 'top_true_damage_done_to_champions_diff', 'top_true_damage_taken_diff', 'top_ability_power_diff', 'top_armor_diff', 'top_armor_pen_diff', 'top_armor_pen_percent_diff', 'top_attack_damage_diff', 'top_attack_speed_diff', 'top_bonus_armor_pen_percent_diff', 'top_bonus_magic_pen_percent_diff', 'top_cc_reduction_diff', 'top_cooldown_reduction_diff', 'top_health_diff', 'top_health_max_diff', 'top_health_regen_diff', 'top_lifesteal_diff', 'top_magic_pen_diff', 'top_magic_pen_percent_diff', 'top_magic_resist_diff', 'top_movement_speed_diff', 'top_omnivamp_diff', 'top_physical_vamp_diff', 'top_power_diff', 'top_power_max_diff', 'top_power_regen_diff', 'top_spell_vamp_diff', 'top_kills_diff', 'top_deaths_diff', 'top_assists_diff', 'top_kda_diff', 'top_kill_participation_diff', 'top_solo_kills_diff', 'top_wards_placed_diff', 'top_wards_killed_diff', 'top_control_wards_placed_diff', 'top_gold_diff', 'top_dmg_diff', 'jg_100_participant_id', 'jg_100_champ', 'jg_100_champion_id', 'jg_100_summoner1_id', 'jg_100_summoner2_id', 'jg_100_primary_style', 'jg_100_secondary_style', 'jg_100_keystone', 'jg_100_primary_rune_1', 'jg_100_primary_rune_2', 'jg_100_primary_rune_3', 'jg_100_secondary_rune_1', 'jg_100_secondary_rune_2', 'jg_100_stat_shard_offense', 'jg_100_stat_shard_flex', 'jg_100_stat_shard_defense', 'jg_100_all_runes', 'jg_200_participant_id', 'jg_200_champ', 'jg_200_champion_id', 'jg_200_summoner1_id', 'jg_200_summoner2_id', 'jg_200_primary_style', 'jg_200_secondary_style', 'jg_200_keystone', 'jg_200_primary_rune_1', 'jg_200_primary_rune_2', 'jg_200_primary_rune_3', 'jg_200_secondary_rune_1', 'jg_200_secondary_rune_2', 'jg_200_stat_shard_offense', 'jg_200_stat_shard_flex', 'jg_200_stat_shard_defense', 'jg_200_all_runes', 'jg_100_current_gold', 'jg_100_total_gold', 'jg_100_level', 'jg_100_xp', 'jg_100_minions_killed', 'jg_100_jungle_minions_killed', 'jg_100_time_enemy_spent_controlled', 'jg_100_cs', 'jg_100_magic_damage_done', 'jg_100_magic_damage_done_to_champions', 'jg_100_magic_damage_taken', 'jg_100_physical_damage_done', 'jg_100_physical_damage_done_to_champions', 'jg_100_physical_damage_taken', 'jg_100_total_damage_done', 'jg_100_total_damage_done_to_champions', 'jg_100_total_damage_taken', 'jg_100_true_damage_done', 'jg_100_true_damage_done_to_champions', 'jg_100_true_damage_taken', 'jg_100_ability_power', 'jg_100_armor', 'jg_100_armor_pen', 'jg_100_armor_pen_percent', 'jg_100_attack_damage', 'jg_100_attack_speed', 'jg_100_bonus_armor_pen_percent', 'jg_100_bonus_magic_pen_percent', 'jg_100_cc_reduction', 'jg_100_cooldown_reduction', 'jg_100_health', 'jg_100_health_max', 'jg_100_health_regen', 'jg_100_lifesteal', 'jg_100_magic_pen', 'jg_100_magic_pen_percent', 'jg_100_magic_resist', 'jg_100_movement_speed', 'jg_100_omnivamp', 'jg_100_physical_vamp', 'jg_100_power', 'jg_100_power_max', 'jg_100_power_regen', 'jg_100_spell_vamp', 'jg_100_hp_pct', 'jg_100_power_pct', 'jg_100_x', 'jg_100_y', 'jg_100_kills', 'jg_100_deaths', 'jg_100_assists', 'jg_100_kda', 'jg_100_kill_participation', 'jg_100_solo_kills', 'jg_100_wards_placed', 'jg_100_wards_killed', 'jg_100_control_wards_placed', 'jg_200_current_gold', 'jg_200_total_gold', 'jg_200_level', 'jg_200_xp', 'jg_200_minions_killed', 'jg_200_jungle_minions_killed', 'jg_200_time_enemy_spent_controlled', 'jg_200_cs', 'jg_200_magic_damage_done', 'jg_200_magic_damage_done_to_champions', 'jg_200_magic_damage_taken', 'jg_200_physical_damage_done', 'jg_200_physical_damage_done_to_champions', 'jg_200_physical_damage_taken', 'jg_200_total_damage_done', 'jg_200_total_damage_done_to_champions', 'jg_200_total_damage_taken', 'jg_200_true_damage_done', 'jg_200_true_damage_done_to_champions', 'jg_200_true_damage_taken', 'jg_200_ability_power', 'jg_200_armor', 'jg_200_armor_pen', 'jg_200_armor_pen_percent', 'jg_200_attack_damage', 'jg_200_attack_speed', 'jg_200_bonus_armor_pen_percent', 'jg_200_bonus_magic_pen_percent', 'jg_200_cc_reduction', 'jg_200_cooldown_reduction', 'jg_200_health', 'jg_200_health_max', 'jg_200_health_regen', 'jg_200_lifesteal', 'jg_200_magic_pen', 'jg_200_magic_pen_percent', 'jg_200_magic_resist', 'jg_200_movement_speed', 'jg_200_omnivamp', 'jg_200_physical_vamp', 'jg_200_power', 'jg_200_power_max', 'jg_200_power_regen', 'jg_200_spell_vamp', 'jg_200_hp_pct', 'jg_200_power_pct', 'jg_200_x', 'jg_200_y', 'jg_200_kills', 'jg_200_deaths', 'jg_200_assists', 'jg_200_kda', 'jg_200_kill_participation', 'jg_200_solo_kills', 'jg_200_wards_placed', 'jg_200_wards_killed', 'jg_200_control_wards_placed', 'jg_current_gold_diff', 'jg_total_gold_diff', 'jg_level_diff', 'jg_xp_diff', 'jg_cs_diff', 'jg_minions_killed_diff', 'jg_jungle_minions_killed_diff', 'jg_time_enemy_spent_controlled_diff', 'jg_magic_damage_done_diff', 'jg_magic_damage_done_to_champions_diff', 'jg_magic_damage_taken_diff', 'jg_physical_damage_done_diff', 'jg_physical_damage_done_to_champions_diff', 'jg_physical_damage_taken_diff', 'jg_total_damage_done_diff', 'jg_total_damage_done_to_champions_diff', 'jg_total_damage_taken_diff', 'jg_true_damage_done_diff', 'jg_true_damage_done_to_champions_diff', 'jg_true_damage_taken_diff', 'jg_ability_power_diff', 'jg_armor_diff', 'jg_armor_pen_diff', 'jg_armor_pen_percent_diff', 'jg_attack_damage_diff', 'jg_attack_speed_diff', 'jg_bonus_armor_pen_percent_diff', 'jg_bonus_magic_pen_percent_diff', 'jg_cc_reduction_diff', 'jg_cooldown_reduction_diff', 'jg_health_diff', 'jg_health_max_diff', 'jg_health_regen_diff', 'jg_lifesteal_diff', 'jg_magic_pen_diff', 'jg_magic_pen_percent_diff', 'jg_magic_resist_diff', 'jg_movement_speed_diff', 'jg_omnivamp_diff', 'jg_physical_vamp_diff', 'jg_power_diff', 'jg_power_max_diff', 'jg_power_regen_diff', 'jg_spell_vamp_diff', 'jg_kills_diff', 'jg_deaths_diff', 'jg_assists_diff', 'jg_kda_diff', 'jg_kill_participation_diff', 'jg_solo_kills_diff', 'jg_wards_placed_diff', 'jg_wards_killed_diff', 'jg_control_wards_placed_diff', 'jg_gold_diff', 'jg_dmg_diff', 'mid_100_participant_id', 'mid_100_champ', 'mid_100_champion_id', 'mid_100_summoner1_id', 'mid_100_summoner2_id', 'mid_100_primary_style', 'mid_100_secondary_style', 'mid_100_keystone', 'mid_100_primary_rune_1', 'mid_100_primary_rune_2', 'mid_100_primary_rune_3', 'mid_100_secondary_rune_1', 'mid_100_secondary_rune_2', 'mid_100_stat_shard_offense', 'mid_100_stat_shard_flex', 'mid_100_stat_shard_defense', 'mid_100_all_runes', 'mid_200_participant_id', 'mid_200_champ', 'mid_200_champion_id', 'mid_200_summoner1_id', 'mid_200_summoner2_id', 'mid_200_primary_style', 'mid_200_secondary_style', 'mid_200_keystone', 'mid_200_primary_rune_1', 'mid_200_primary_rune_2', 'mid_200_primary_rune_3', 'mid_200_secondary_rune_1', 'mid_200_secondary_rune_2', 'mid_200_stat_shard_offense', 'mid_200_stat_shard_flex', 'mid_200_stat_shard_defense', 'mid_200_all_runes', 'mid_100_current_gold', 'mid_100_total_gold', 'mid_100_level', 'mid_100_xp', 'mid_100_minions_killed', 'mid_100_jungle_minions_killed', 'mid_100_time_enemy_spent_controlled', 'mid_100_cs', 'mid_100_magic_damage_done', 'mid_100_magic_damage_done_to_champions', 'mid_100_magic_damage_taken', 'mid_100_physical_damage_done', 'mid_100_physical_damage_done_to_champions', 'mid_100_physical_damage_taken', 'mid_100_total_damage_done', 'mid_100_total_damage_done_to_champions', 'mid_100_total_damage_taken', 'mid_100_true_damage_done', 'mid_100_true_damage_done_to_champions', 'mid_100_true_damage_taken', 'mid_100_ability_power', 'mid_100_armor', 'mid_100_armor_pen', 'mid_100_armor_pen_percent', 'mid_100_attack_damage', 'mid_100_attack_speed', 'mid_100_bonus_armor_pen_percent', 'mid_100_bonus_magic_pen_percent', 'mid_100_cc_reduction', 'mid_100_cooldown_reduction', 'mid_100_health', 'mid_100_health_max', 'mid_100_health_regen', 'mid_100_lifesteal', 'mid_100_magic_pen', 'mid_100_magic_pen_percent', 'mid_100_magic_resist', 'mid_100_movement_speed', 'mid_100_omnivamp', 'mid_100_physical_vamp', 'mid_100_power', 'mid_100_power_max', 'mid_100_power_regen', 'mid_100_spell_vamp', 'mid_100_hp_pct', 'mid_100_power_pct', 'mid_100_x', 'mid_100_y', 'mid_100_kills', 'mid_100_deaths', 'mid_100_assists', 'mid_100_kda', 'mid_100_kill_participation', 'mid_100_solo_kills', 'mid_100_wards_placed', 'mid_100_wards_killed', 'mid_100_control_wards_placed', 'mid_200_current_gold', 'mid_200_total_gold', 'mid_200_level', 'mid_200_xp', 'mid_200_minions_killed', 'mid_200_jungle_minions_killed', 'mid_200_time_enemy_spent_controlled', 'mid_200_cs', 'mid_200_magic_damage_done', 'mid_200_magic_damage_done_to_champions', 'mid_200_magic_damage_taken', 'mid_200_physical_damage_done', 'mid_200_physical_damage_done_to_champions', 'mid_200_physical_damage_taken', 'mid_200_total_damage_done', 'mid_200_total_damage_done_to_champions', 'mid_200_total_damage_taken', 'mid_200_true_damage_done', 'mid_200_true_damage_done_to_champions', 'mid_200_true_damage_taken', 'mid_200_ability_power', 'mid_200_armor', 'mid_200_armor_pen', 'mid_200_armor_pen_percent', 'mid_200_attack_damage', 'mid_200_attack_speed', 'mid_200_bonus_armor_pen_percent', 'mid_200_bonus_magic_pen_percent', 'mid_200_cc_reduction', 'mid_200_cooldown_reduction', 'mid_200_health', 'mid_200_health_max', 'mid_200_health_regen', 'mid_200_lifesteal', 'mid_200_magic_pen', 'mid_200_magic_pen_percent', 'mid_200_magic_resist', 'mid_200_movement_speed', 'mid_200_omnivamp', 'mid_200_physical_vamp', 'mid_200_power', 'mid_200_power_max', 'mid_200_power_regen', 'mid_200_spell_vamp', 'mid_200_hp_pct', 'mid_200_power_pct', 'mid_200_x', 'mid_200_y', 'mid_200_kills', 'mid_200_deaths', 'mid_200_assists', 'mid_200_kda', 'mid_200_kill_participation', 'mid_200_solo_kills', 'mid_200_wards_placed', 'mid_200_wards_killed', 'mid_200_control_wards_placed', 'mid_current_gold_diff', 'mid_total_gold_diff', 'mid_level_diff', 'mid_xp_diff', 'mid_cs_diff', 'mid_minions_killed_diff', 'mid_jungle_minions_killed_diff', 'mid_time_enemy_spent_controlled_diff', 'mid_magic_damage_done_diff', 'mid_magic_damage_done_to_champions_diff', 'mid_magic_damage_taken_diff', 'mid_physical_damage_done_diff', 'mid_physical_damage_done_to_champions_diff', 'mid_physical_damage_taken_diff', 'mid_total_damage_done_diff', 'mid_total_damage_done_to_champions_diff', 'mid_total_damage_taken_diff', 'mid_true_damage_done_diff', 'mid_true_damage_done_to_champions_diff', 'mid_true_damage_taken_diff', 'mid_ability_power_diff', 'mid_armor_diff', 'mid_armor_pen_diff', 'mid_armor_pen_percent_diff', 'mid_attack_damage_diff', 'mid_attack_speed_diff', 'mid_bonus_armor_pen_percent_diff', 'mid_bonus_magic_pen_percent_diff', 'mid_cc_reduction_diff', 'mid_cooldown_reduction_diff', 'mid_health_diff', 'mid_health_max_diff', 'mid_health_regen_diff', 'mid_lifesteal_diff', 'mid_magic_pen_diff', 'mid_magic_pen_percent_diff', 'mid_magic_resist_diff', 'mid_movement_speed_diff', 'mid_omnivamp_diff', 'mid_physical_vamp_diff', 'mid_power_diff', 'mid_power_max_diff', 'mid_power_regen_diff', 'mid_spell_vamp_diff', 'mid_kills_diff', 'mid_deaths_diff', 'mid_assists_diff', 'mid_kda_diff', 'mid_kill_participation_diff', 'mid_solo_kills_diff', 'mid_wards_placed_diff', 'mid_wards_killed_diff', 'mid_control_wards_placed_diff', 'mid_gold_diff', 'mid_dmg_diff', 'bot_100_participant_id', 'bot_100_champ', 'bot_100_champion_id', 'bot_100_summoner1_id', 'bot_100_summoner2_id', 'bot_100_primary_style', 'bot_100_secondary_style', 'bot_100_keystone', 'bot_100_primary_rune_1', 'bot_100_primary_rune_2', 'bot_100_primary_rune_3', 'bot_100_secondary_rune_1', 'bot_100_secondary_rune_2', 'bot_100_stat_shard_offense', 'bot_100_stat_shard_flex', 'bot_100_stat_shard_defense', 'bot_100_all_runes', 'bot_200_participant_id', 'bot_200_champ', 'bot_200_champion_id', 'bot_200_summoner1_id', 'bot_200_summoner2_id', 'bot_200_primary_style', 'bot_200_secondary_style', 'bot_200_keystone', 'bot_200_primary_rune_1', 'bot_200_primary_rune_2', 'bot_200_primary_rune_3', 'bot_200_secondary_rune_1', 'bot_200_secondary_rune_2', 'bot_200_stat_shard_offense', 'bot_200_stat_shard_flex', 'bot_200_stat_shard_defense', 'bot_200_all_runes', 'bot_100_current_gold', 'bot_100_total_gold', 'bot_100_level', 'bot_100_xp', 'bot_100_minions_killed', 'bot_100_jungle_minions_killed', 'bot_100_time_enemy_spent_controlled', 'bot_100_cs', 'bot_100_magic_damage_done', 'bot_100_magic_damage_done_to_champions', 'bot_100_magic_damage_taken', 'bot_100_physical_damage_done', 'bot_100_physical_damage_done_to_champions', 'bot_100_physical_damage_taken', 'bot_100_total_damage_done', 'bot_100_total_damage_done_to_champions', 'bot_100_total_damage_taken', 'bot_100_true_damage_done', 'bot_100_true_damage_done_to_champions', 'bot_100_true_damage_taken', 'bot_100_ability_power', 'bot_100_armor', 'bot_100_armor_pen', 'bot_100_armor_pen_percent', 'bot_100_attack_damage', 'bot_100_attack_speed', 'bot_100_bonus_armor_pen_percent', 'bot_100_bonus_magic_pen_percent', 'bot_100_cc_reduction', 'bot_100_cooldown_reduction', 'bot_100_health', 'bot_100_health_max', 'bot_100_health_regen', 'bot_100_lifesteal', 'bot_100_magic_pen', 'bot_100_magic_pen_percent', 'bot_100_magic_resist', 'bot_100_movement_speed', 'bot_100_omnivamp', 'bot_100_physical_vamp', 'bot_100_power', 'bot_100_power_max', 'bot_100_power_regen', 'bot_100_spell_vamp', 'bot_100_hp_pct', 'bot_100_power_pct', 'bot_100_x', 'bot_100_y', 'bot_100_kills', 'bot_100_deaths', 'bot_100_assists', 'bot_100_kda', 'bot_100_kill_participation', 'bot_100_solo_kills', 'bot_100_wards_placed', 'bot_100_wards_killed', 'bot_100_control_wards_placed', 'bot_200_current_gold', 'bot_200_total_gold', 'bot_200_level', 'bot_200_xp', 'bot_200_minions_killed', 'bot_200_jungle_minions_killed', 'bot_200_time_enemy_spent_controlled', 'bot_200_cs', 'bot_200_magic_damage_done', 'bot_200_magic_damage_done_to_champions', 'bot_200_magic_damage_taken', 'bot_200_physical_damage_done', 'bot_200_physical_damage_done_to_champions', 'bot_200_physical_damage_taken', 'bot_200_total_damage_done', 'bot_200_total_damage_done_to_champions', 'bot_200_total_damage_taken', 'bot_200_true_damage_done', 'bot_200_true_damage_done_to_champions', 'bot_200_true_damage_taken', 'bot_200_ability_power', 'bot_200_armor', 'bot_200_armor_pen', 'bot_200_armor_pen_percent', 'bot_200_attack_damage', 'bot_200_attack_speed', 'bot_200_bonus_armor_pen_percent', 'bot_200_bonus_magic_pen_percent', 'bot_200_cc_reduction', 'bot_200_cooldown_reduction', 'bot_200_health', 'bot_200_health_max', 'bot_200_health_regen', 'bot_200_lifesteal', 'bot_200_magic_pen', 'bot_200_magic_pen_percent', 'bot_200_magic_resist', 'bot_200_movement_speed', 'bot_200_omnivamp', 'bot_200_physical_vamp', 'bot_200_power', 'bot_200_power_max', 'bot_200_power_regen', 'bot_200_spell_vamp', 'bot_200_hp_pct', 'bot_200_power_pct', 'bot_200_x', 'bot_200_y', 'bot_200_kills', 'bot_200_deaths', 'bot_200_assists', 'bot_200_kda', 'bot_200_kill_participation', 'bot_200_solo_kills', 'bot_200_wards_placed', 'bot_200_wards_killed', 'bot_200_control_wards_placed', 'bot_current_gold_diff', 'bot_total_gold_diff', 'bot_level_diff', 'bot_xp_diff', 'bot_cs_diff', 'bot_minions_killed_diff', 'bot_jungle_minions_killed_diff', 'bot_time_enemy_spent_controlled_diff', 'bot_magic_damage_done_diff', 'bot_magic_damage_done_to_champions_diff', 'bot_magic_damage_taken_diff', 'bot_physical_damage_done_diff', 'bot_physical_damage_done_to_champions_diff', 'bot_physical_damage_taken_diff', 'bot_total_damage_done_diff', 'bot_total_damage_done_to_champions_diff', 'bot_total_damage_taken_diff', 'bot_true_damage_done_diff', 'bot_true_damage_done_to_champions_diff', 'bot_true_damage_taken_diff', 'bot_ability_power_diff', 'bot_armor_diff', 'bot_armor_pen_diff', 'bot_armor_pen_percent_diff', 'bot_attack_damage_diff', 'bot_attack_speed_diff', 'bot_bonus_armor_pen_percent_diff', 'bot_bonus_magic_pen_percent_diff', 'bot_cc_reduction_diff', 'bot_cooldown_reduction_diff', 'bot_health_diff', 'bot_health_max_diff', 'bot_health_regen_diff', 'bot_lifesteal_diff', 'bot_magic_pen_diff', 'bot_magic_pen_percent_diff', 'bot_magic_resist_diff', 'bot_movement_speed_diff', 'bot_omnivamp_diff', 'bot_physical_vamp_diff', 'bot_power_diff', 'bot_power_max_diff', 'bot_power_regen_diff', 'bot_spell_vamp_diff', 'bot_kills_diff', 'bot_deaths_diff', 'bot_assists_diff', 'bot_kda_diff', 'bot_kill_participation_diff', 'bot_solo_kills_diff', 'bot_wards_placed_diff', 'bot_wards_killed_diff', 'bot_control_wards_placed_diff', 'bot_gold_diff', 'bot_dmg_diff', 'sup_100_participant_id', 'sup_100_champ', 'sup_100_champion_id', 'sup_100_summoner1_id', 'sup_100_summoner2_id', 'sup_100_primary_style', 'sup_100_secondary_style', 'sup_100_keystone', 'sup_100_primary_rune_1', 'sup_100_primary_rune_2', 'sup_100_primary_rune_3', 'sup_100_secondary_rune_1', 'sup_100_secondary_rune_2', 'sup_100_stat_shard_offense', 'sup_100_stat_shard_flex', 'sup_100_stat_shard_defense', 'sup_100_all_runes', 'sup_200_participant_id', 'sup_200_champ', 'sup_200_champion_id', 'sup_200_summoner1_id', 'sup_200_summoner2_id', 'sup_200_primary_style', 'sup_200_secondary_style', 'sup_200_keystone', 'sup_200_primary_rune_1', 'sup_200_primary_rune_2', 'sup_200_primary_rune_3', 'sup_200_secondary_rune_1', 'sup_200_secondary_rune_2', 'sup_200_stat_shard_offense', 'sup_200_stat_shard_flex', 'sup_200_stat_shard_defense', 'sup_200_all_runes', 'sup_100_current_gold', 'sup_100_total_gold', 'sup_100_level', 'sup_100_xp', 'sup_100_minions_killed', 'sup_100_jungle_minions_killed', 'sup_100_time_enemy_spent_controlled', 'sup_100_cs', 'sup_100_magic_damage_done', 'sup_100_magic_damage_done_to_champions', 'sup_100_magic_damage_taken', 'sup_100_physical_damage_done', 'sup_100_physical_damage_done_to_champions', 'sup_100_physical_damage_taken', 'sup_100_total_damage_done', 'sup_100_total_damage_done_to_champions', 'sup_100_total_damage_taken', 'sup_100_true_damage_done', 'sup_100_true_damage_done_to_champions', 'sup_100_true_damage_taken', 'sup_100_ability_power', 'sup_100_armor', 'sup_100_armor_pen', 'sup_100_armor_pen_percent', 'sup_100_attack_damage', 'sup_100_attack_speed', 'sup_100_bonus_armor_pen_percent', 'sup_100_bonus_magic_pen_percent', 'sup_100_cc_reduction', 'sup_100_cooldown_reduction', 'sup_100_health', 'sup_100_health_max', 'sup_100_health_regen', 'sup_100_lifesteal', 'sup_100_magic_pen', 'sup_100_magic_pen_percent', 'sup_100_magic_resist', 'sup_100_movement_speed', 'sup_100_omnivamp', 'sup_100_physical_vamp', 'sup_100_power', 'sup_100_power_max', 'sup_100_power_regen', 'sup_100_spell_vamp', 'sup_100_hp_pct', 'sup_100_power_pct', 'sup_100_x', 'sup_100_y', 'sup_100_kills', 'sup_100_deaths', 'sup_100_assists', 'sup_100_kda', 'sup_100_kill_participation', 'sup_100_solo_kills', 'sup_100_wards_placed', 'sup_100_wards_killed', 'sup_100_control_wards_placed', 'sup_200_current_gold', 'sup_200_total_gold', 'sup_200_level', 'sup_200_xp', 'sup_200_minions_killed', 'sup_200_jungle_minions_killed', 'sup_200_time_enemy_spent_controlled', 'sup_200_cs', 'sup_200_magic_damage_done', 'sup_200_magic_damage_done_to_champions', 'sup_200_magic_damage_taken', 'sup_200_physical_damage_done', 'sup_200_physical_damage_done_to_champions', 'sup_200_physical_damage_taken', 'sup_200_total_damage_done', 'sup_200_total_damage_done_to_champions', 'sup_200_total_damage_taken', 'sup_200_true_damage_done', 'sup_200_true_damage_done_to_champions', 'sup_200_true_damage_taken', 'sup_200_ability_power', 'sup_200_armor', 'sup_200_armor_pen', 'sup_200_armor_pen_percent', 'sup_200_attack_damage', 'sup_200_attack_speed', 'sup_200_bonus_armor_pen_percent', 'sup_200_bonus_magic_pen_percent', 'sup_200_cc_reduction', 'sup_200_cooldown_reduction', 'sup_200_health', 'sup_200_health_max', 'sup_200_health_regen', 'sup_200_lifesteal', 'sup_200_magic_pen', 'sup_200_magic_pen_percent', 'sup_200_magic_resist', 'sup_200_movement_speed', 'sup_200_omnivamp', 'sup_200_physical_vamp', 'sup_200_power', 'sup_200_power_max', 'sup_200_power_regen', 'sup_200_spell_vamp', 'sup_200_hp_pct', 'sup_200_power_pct', 'sup_200_x', 'sup_200_y', 'sup_200_kills', 'sup_200_deaths', 'sup_200_assists', 'sup_200_kda', 'sup_200_kill_participation', 'sup_200_solo_kills', 'sup_200_wards_placed', 'sup_200_wards_killed', 'sup_200_control_wards_placed', 'sup_current_gold_diff', 'sup_total_gold_diff', 'sup_level_diff', 'sup_xp_diff', 'sup_cs_diff', 'sup_minions_killed_diff', 'sup_jungle_minions_killed_diff', 'sup_time_enemy_spent_controlled_diff', 'sup_magic_damage_done_diff', 'sup_magic_damage_done_to_champions_diff', 'sup_magic_damage_taken_diff', 'sup_physical_damage_done_diff', 'sup_physical_damage_done_to_champions_diff', 'sup_physical_damage_taken_diff', 'sup_total_damage_done_diff', 'sup_total_damage_done_to_champions_diff', 'sup_total_damage_taken_diff', 'sup_true_damage_done_diff', 'sup_true_damage_done_to_champions_diff', 'sup_true_damage_taken_diff', 'sup_ability_power_diff', 'sup_armor_diff', 'sup_armor_pen_diff', 'sup_armor_pen_percent_diff', 'sup_attack_damage_diff', 'sup_attack_speed_diff', 'sup_bonus_armor_pen_percent_diff', 'sup_bonus_magic_pen_percent_diff', 'sup_cc_reduction_diff', 'sup_cooldown_reduction_diff', 'sup_health_diff', 'sup_health_max_diff', 'sup_health_regen_diff', 'sup_lifesteal_diff', 'sup_magic_pen_diff', 'sup_magic_pen_percent_diff', 'sup_magic_resist_diff', 'sup_movement_speed_diff', 'sup_omnivamp_diff', 'sup_physical_vamp_diff', 'sup_power_diff', 'sup_power_max_diff', 'sup_power_regen_diff', 'sup_spell_vamp_diff', 'sup_kills_diff', 'sup_deaths_diff', 'sup_assists_diff', 'sup_kda_diff', 'sup_kill_participation_diff', 'sup_solo_kills_diff', 'sup_wards_placed_diff', 'sup_wards_killed_diff', 'sup_control_wards_placed_diff', 'sup_gold_diff', 'sup_dmg_diff']


REDUNDANT_FIELDS_REMOVED = ['baron_diff', 'base_towers_100_destroyed', 'base_towers_200_destroyed', 'bot_100_all_runes', 'bot_100_champ', 'bot_100_cs', 'bot_100_hp_pct', 'bot_100_kda', 'bot_100_kill_participation', 'bot_100_participant_id', 'bot_100_power_pct', 'bot_100_total_damage_done', 'bot_100_total_damage_done_to_champions', 'bot_100_total_damage_taken', 'bot_200_all_runes', 'bot_200_champ', 'bot_200_cs', 'bot_200_hp_pct', 'bot_200_kda', 'bot_200_kill_participation', 'bot_200_participant_id', 'bot_200_power_pct', 'bot_200_total_damage_done', 'bot_200_total_damage_done_to_champions', 'bot_200_total_damage_taken', 'bot_ability_power_diff', 'bot_armor_diff', 'bot_armor_pen_diff', 'bot_armor_pen_percent_diff', 'bot_assists_diff', 'bot_attack_damage_diff', 'bot_attack_speed_diff', 'bot_bonus_armor_pen_percent_diff', 'bot_bonus_magic_pen_percent_diff', 'bot_cc_reduction_diff', 'bot_control_wards_placed_diff', 'bot_cooldown_reduction_diff', 'bot_cs_diff', 'bot_current_gold_diff', 'bot_deaths_diff', 'bot_dmg_diff', 'bot_gold_diff', 'bot_health_diff', 'bot_health_max_diff', 'bot_health_regen_diff', 'bot_jungle_minions_killed_diff', 'bot_kda_diff', 'bot_kill_participation_diff', 'bot_kills_diff', 'bot_level_diff', 'bot_lifesteal_diff', 'bot_magic_damage_done_diff', 'bot_magic_damage_done_to_champions_diff', 'bot_magic_damage_taken_diff', 'bot_magic_pen_diff', 'bot_magic_pen_percent_diff', 'bot_magic_resist_diff', 'bot_minions_killed_diff', 'bot_movement_speed_diff', 'bot_omnivamp_diff', 'bot_physical_damage_done_diff', 'bot_physical_damage_done_to_champions_diff', 'bot_physical_damage_taken_diff', 'bot_physical_vamp_diff', 'bot_power_diff', 'bot_power_max_diff', 'bot_power_regen_diff', 'bot_solo_kills_diff', 'bot_spell_vamp_diff', 'bot_time_enemy_spent_controlled_diff', 'bot_total_damage_done_diff', 'bot_total_damage_done_to_champions_diff', 'bot_total_damage_taken_diff', 'bot_total_gold_diff', 'bot_towers_100_destroyed', 'bot_towers_200_destroyed', 'bot_true_damage_done_diff', 'bot_true_damage_done_to_champions_diff', 'bot_true_damage_taken_diff', 'bot_wards_killed_diff', 'bot_wards_placed_diff', 'bot_xp_diff', 'control_wards_100', 'control_wards_200', 'dragon_diff', 'herald_diff', 'inner_towers_100_destroyed', 'inner_towers_200_destroyed', 'jg_100_all_runes', 'jg_100_champ', 'jg_100_cs', 'jg_100_hp_pct', 'jg_100_kda', 'jg_100_kill_participation', 'jg_100_participant_id', 'jg_100_power_pct', 'jg_100_total_damage_done', 'jg_100_total_damage_done_to_champions', 'jg_100_total_damage_taken', 'jg_200_all_runes', 'jg_200_champ', 'jg_200_cs', 'jg_200_hp_pct', 'jg_200_kda', 'jg_200_kill_participation', 'jg_200_participant_id', 'jg_200_power_pct', 'jg_200_total_damage_done', 'jg_200_total_damage_done_to_champions', 'jg_200_total_damage_taken', 'jg_ability_power_diff', 'jg_armor_diff', 'jg_armor_pen_diff', 'jg_armor_pen_percent_diff', 'jg_assists_diff', 'jg_attack_damage_diff', 'jg_attack_speed_diff', 'jg_bonus_armor_pen_percent_diff', 'jg_bonus_magic_pen_percent_diff', 'jg_cc_reduction_diff', 'jg_control_wards_placed_diff', 'jg_cooldown_reduction_diff', 'jg_cs_diff', 'jg_current_gold_diff', 'jg_deaths_diff', 'jg_dmg_diff', 'jg_gold_diff', 'jg_health_diff', 'jg_health_max_diff', 'jg_health_regen_diff', 'jg_jungle_minions_killed_diff', 'jg_kda_diff', 'jg_kill_participation_diff', 'jg_kills_diff', 'jg_level_diff', 'jg_lifesteal_diff', 'jg_magic_damage_done_diff', 'jg_magic_damage_done_to_champions_diff', 'jg_magic_damage_taken_diff', 'jg_magic_pen_diff', 'jg_magic_pen_percent_diff', 'jg_magic_resist_diff', 'jg_minions_killed_diff', 'jg_movement_speed_diff', 'jg_omnivamp_diff', 'jg_physical_damage_done_diff', 'jg_physical_damage_done_to_champions_diff', 'jg_physical_damage_taken_diff', 'jg_physical_vamp_diff', 'jg_power_diff', 'jg_power_max_diff', 'jg_power_regen_diff', 'jg_solo_kills_diff', 'jg_spell_vamp_diff', 'jg_time_enemy_spent_controlled_diff', 'jg_total_damage_done_diff', 'jg_total_damage_done_to_champions_diff', 'jg_total_damage_taken_diff', 'jg_total_gold_diff', 'jg_true_damage_done_diff', 'jg_true_damage_done_to_champions_diff', 'jg_true_damage_taken_diff', 'jg_wards_killed_diff', 'jg_wards_placed_diff', 'jg_xp_diff', 'kill_diff', 'kills_100', 'kills_200', 'mid_100_all_runes', 'mid_100_champ', 'mid_100_cs', 'mid_100_hp_pct', 'mid_100_kda', 'mid_100_kill_participation', 'mid_100_participant_id', 'mid_100_power_pct', 'mid_100_total_damage_done', 'mid_100_total_damage_done_to_champions', 'mid_100_total_damage_taken', 'mid_200_all_runes', 'mid_200_champ', 'mid_200_cs', 'mid_200_hp_pct', 'mid_200_kda', 'mid_200_kill_participation', 'mid_200_participant_id', 'mid_200_power_pct', 'mid_200_total_damage_done', 'mid_200_total_damage_done_to_champions', 'mid_200_total_damage_taken', 'mid_ability_power_diff', 'mid_armor_diff', 'mid_armor_pen_diff', 'mid_armor_pen_percent_diff', 'mid_assists_diff', 'mid_attack_damage_diff', 'mid_attack_speed_diff', 'mid_bonus_armor_pen_percent_diff', 'mid_bonus_magic_pen_percent_diff', 'mid_cc_reduction_diff', 'mid_control_wards_placed_diff', 'mid_cooldown_reduction_diff', 'mid_cs_diff', 'mid_current_gold_diff', 'mid_deaths_diff', 'mid_dmg_diff', 'mid_gold_diff', 'mid_health_diff', 'mid_health_max_diff', 'mid_health_regen_diff', 'mid_jungle_minions_killed_diff', 'mid_kda_diff', 'mid_kill_participation_diff', 'mid_kills_diff', 'mid_level_diff', 'mid_lifesteal_diff', 'mid_magic_damage_done_diff', 'mid_magic_damage_done_to_champions_diff', 'mid_magic_damage_taken_diff', 'mid_magic_pen_diff', 'mid_magic_pen_percent_diff', 'mid_magic_resist_diff', 'mid_minions_killed_diff', 'mid_movement_speed_diff', 'mid_omnivamp_diff', 'mid_physical_damage_done_diff', 'mid_physical_damage_done_to_champions_diff', 'mid_physical_damage_taken_diff', 'mid_physical_vamp_diff', 'mid_power_diff', 'mid_power_max_diff', 'mid_power_regen_diff', 'mid_solo_kills_diff', 'mid_spell_vamp_diff', 'mid_time_enemy_spent_controlled_diff', 'mid_total_damage_done_diff', 'mid_total_damage_done_to_champions_diff', 'mid_total_damage_taken_diff', 'mid_total_gold_diff', 'mid_towers_100_destroyed', 'mid_towers_200_destroyed', 'mid_true_damage_done_diff', 'mid_true_damage_done_to_champions_diff', 'mid_true_damage_taken_diff', 'mid_wards_killed_diff', 'mid_wards_placed_diff', 'mid_xp_diff', 'nexus_towers_100_destroyed', 'nexus_towers_200_destroyed', 'outer_towers_100_destroyed', 'outer_towers_200_destroyed', 'plate_diff', 'sup_100_all_runes', 'sup_100_champ', 'sup_100_cs', 'sup_100_hp_pct', 'sup_100_kda', 'sup_100_kill_participation', 'sup_100_participant_id', 'sup_100_power_pct', 'sup_100_total_damage_done', 'sup_100_total_damage_done_to_champions', 'sup_100_total_damage_taken', 'sup_200_all_runes', 'sup_200_champ', 'sup_200_cs', 'sup_200_hp_pct', 'sup_200_kda', 'sup_200_kill_participation', 'sup_200_participant_id', 'sup_200_power_pct', 'sup_200_total_damage_done', 'sup_200_total_damage_done_to_champions', 'sup_200_total_damage_taken', 'sup_ability_power_diff', 'sup_armor_diff', 'sup_armor_pen_diff', 'sup_armor_pen_percent_diff', 'sup_assists_diff', 'sup_attack_damage_diff', 'sup_attack_speed_diff', 'sup_bonus_armor_pen_percent_diff', 'sup_bonus_magic_pen_percent_diff', 'sup_cc_reduction_diff', 'sup_control_wards_placed_diff', 'sup_cooldown_reduction_diff', 'sup_cs_diff', 'sup_current_gold_diff', 'sup_deaths_diff', 'sup_dmg_diff', 'sup_gold_diff', 'sup_health_diff', 'sup_health_max_diff', 'sup_health_regen_diff', 'sup_jungle_minions_killed_diff', 'sup_kda_diff', 'sup_kill_participation_diff', 'sup_kills_diff', 'sup_level_diff', 'sup_lifesteal_diff', 'sup_magic_damage_done_diff', 'sup_magic_damage_done_to_champions_diff', 'sup_magic_damage_taken_diff', 'sup_magic_pen_diff', 'sup_magic_pen_percent_diff', 'sup_magic_resist_diff', 'sup_minions_killed_diff', 'sup_movement_speed_diff', 'sup_omnivamp_diff', 'sup_physical_damage_done_diff', 'sup_physical_damage_done_to_champions_diff', 'sup_physical_damage_taken_diff', 'sup_physical_vamp_diff', 'sup_power_diff', 'sup_power_max_diff', 'sup_power_regen_diff', 'sup_solo_kills_diff', 'sup_spell_vamp_diff', 'sup_time_enemy_spent_controlled_diff', 'sup_total_damage_done_diff', 'sup_total_damage_done_to_champions_diff', 'sup_total_damage_taken_diff', 'sup_total_gold_diff', 'sup_true_damage_done_diff', 'sup_true_damage_done_to_champions_diff', 'sup_true_damage_taken_diff', 'sup_wards_killed_diff', 'sup_wards_placed_diff', 'sup_xp_diff', 'top_100_all_runes', 'top_100_champ', 'top_100_cs', 'top_100_hp_pct', 'top_100_kda', 'top_100_kill_participation', 'top_100_participant_id', 'top_100_power_pct', 'top_100_total_damage_done', 'top_100_total_damage_done_to_champions', 'top_100_total_damage_taken', 'top_200_all_runes', 'top_200_champ', 'top_200_cs', 'top_200_hp_pct', 'top_200_kda', 'top_200_kill_participation', 'top_200_participant_id', 'top_200_power_pct', 'top_200_total_damage_done', 'top_200_total_damage_done_to_champions', 'top_200_total_damage_taken', 'top_ability_power_diff', 'top_armor_diff', 'top_armor_pen_diff', 'top_armor_pen_percent_diff', 'top_assists_diff', 'top_attack_damage_diff', 'top_attack_speed_diff', 'top_bonus_armor_pen_percent_diff', 'top_bonus_magic_pen_percent_diff', 'top_cc_reduction_diff', 'top_control_wards_placed_diff', 'top_cooldown_reduction_diff', 'top_cs_diff', 'top_current_gold_diff', 'top_deaths_diff', 'top_dmg_diff', 'top_gold_diff', 'top_health_diff', 'top_health_max_diff', 'top_health_regen_diff', 'top_jungle_minions_killed_diff', 'top_kda_diff', 'top_kill_participation_diff', 'top_kills_diff', 'top_level_diff', 'top_lifesteal_diff', 'top_magic_damage_done_diff', 'top_magic_damage_done_to_champions_diff', 'top_magic_damage_taken_diff', 'top_magic_pen_diff', 'top_magic_pen_percent_diff', 'top_magic_resist_diff', 'top_minions_killed_diff', 'top_movement_speed_diff', 'top_omnivamp_diff', 'top_physical_damage_done_diff', 'top_physical_damage_done_to_champions_diff', 'top_physical_damage_taken_diff', 'top_physical_vamp_diff', 'top_power_diff', 'top_power_max_diff', 'top_power_regen_diff', 'top_solo_kills_diff', 'top_spell_vamp_diff', 'top_time_enemy_spent_controlled_diff', 'top_total_damage_done_diff', 'top_total_damage_done_to_champions_diff', 'top_total_damage_taken_diff', 'top_total_gold_diff', 'top_towers_100_destroyed', 'top_towers_200_destroyed', 'top_true_damage_done_diff', 'top_true_damage_done_to_champions_diff', 'top_true_damage_taken_diff', 'top_wards_killed_diff', 'top_wards_placed_diff', 'top_xp_diff', 'towers_100', 'towers_200', 'wards_killed_100', 'wards_killed_200', 'wards_placed_100', 'wards_placed_200']

CSV_FIELDS = [field for field in CSV_FIELDS if field not in set(REDUNDANT_FIELDS_REMOVED)]


GLOBAL_PROCESSED_PATH = OUTPUT_DIR / "all_processed_matches.json"


def load_global_processed():
    if GLOBAL_PROCESSED_PATH.exists():
        return set(json.loads(GLOBAL_PROCESSED_PATH.read_text()))
    return set()


def save_global_processed(global_processed):
    GLOBAL_PROCESSED_PATH.write_text(json.dumps(list(global_processed)))


STRING_COLUMNS = {"match_id", "patch", "team_100_bans", "team_200_bans"}
STRING_SUFFIXES = ("_champ", "_all_runes")


def _is_string_column(column_name):
    return column_name in STRING_COLUMNS or column_name.endswith(STRING_SUFFIXES)


def _clean_rows_for_parquet(rows):
    clean_rows = []
    for row in rows:
        clean_rows.append({field: (None if row.get(field, None) == "" else row.get(field, None)) for field in CSV_FIELDS})

    df = pd.DataFrame(clean_rows, columns=CSV_FIELDS)
    for column in CSV_FIELDS:
        if _is_string_column(column):
            df[column] = df[column].astype("string")
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _next_parquet_part_index(parquet_dir):
    existing = sorted(parquet_dir.glob("part_*.parquet"))
    if not existing:
        return 0
    max_index = -1
    for path in existing:
        try:
            max_index = max(max_index, int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max_index + 1


def _write_parquet_part(rows, parquet_dir, part_index):
    if not rows:
        return None
    parquet_dir.mkdir(parents=True, exist_ok=True)
    df = _clean_rows_for_parquet(rows)
    out_path = parquet_dir / f"part_{part_index:05d}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    return out_path


def process_bracket(bracket_name, global_processed):
    print(f"\n=== Tier: {bracket_name} ===")
    pool_cache = OUTPUT_DIR / f"{bracket_name}_puuid_pool.json"
    processed_path = OUTPUT_DIR / f"{bracket_name}_processed_matches.json"
    parquet_dir = OUTPUT_DIR / f"{bracket_name}_snapshots_parquet"

    processed = set(json.loads(processed_path.read_text())) if processed_path.exists() else set()
    print(f"[{bracket_name}] already processed: {len(processed)} matches")

    if len(processed) >= TARGET_GAMES_PER_BRACKET:
        print(f"[{bracket_name}] target already met, skipping.")
        return

    pool_target = max(TARGET_GAMES_PER_BRACKET * 2 // MATCHES_PER_PLAYER, 200)
    pool = collect_puuid_pool(bracket_name, pool_target, pool_cache)
    random.shuffle(pool)

    candidate_ids = collect_match_ids(
        bracket_name, pool, TARGET_GAMES_PER_BRACKET, processed, global_processed
    )

    part_index = _next_parquet_part_index(parquet_dir)
    buffered_rows = []
    buffered_match_ids = []

    def flush_buffer():
        nonlocal part_index, buffered_rows, buffered_match_ids
        if not buffered_rows:
            return
        out_path = _write_parquet_part(buffered_rows, parquet_dir, part_index)
        part_index += 1
        for mid in buffered_match_ids:
            processed.add(mid)
            global_processed.add(mid)
        processed_path.write_text(json.dumps(list(processed)))
        save_global_processed(global_processed)
        print(
            f"[{bracket_name}] wrote {len(buffered_match_ids)} matches / "
            f"{len(buffered_rows)} rows -> {out_path}"
        )
        print(f"[{bracket_name}] processed {len(processed)}/{TARGET_GAMES_PER_BRACKET} matches")
        buffered_rows = []
        buffered_match_ids = []

    for match_id in candidate_ids:
        if len(processed) + len(buffered_match_ids) >= TARGET_GAMES_PER_BRACKET:
            break
        if match_id in global_processed or match_id in processed or match_id in buffered_match_ids:
            continue
        try:
            match = get_match(match_id)
            timeline = get_timeline(match_id)
            if not match or not timeline:
                continue
            if match["info"].get("queueId") != QUEUE_ID:
                continue
            game_version = match["info"].get("gameVersion", "")
            patch = ".".join(game_version.split(".")[:2])
            if patch != TARGET_PATCH:
                continue
            rows = extract_snapshots(match, timeline)
            if not rows:
                continue
            buffered_rows.extend(rows)
            buffered_match_ids.append(match_id)
            if len(buffered_match_ids) >= PARQUET_MATCH_CHUNK_SIZE:
                flush_buffer()
        except RuntimeError as e:
            print(f"  warning: skipping {match_id}: {e}")
            continue

    flush_buffer()

    processed_path.write_text(json.dumps(list(processed)))
    save_global_processed(global_processed)
    print(f"[{bracket_name}] done: {len(processed)} matches -> {parquet_dir}")


def main():
    global_processed = load_global_processed()
    for bracket_name in BRACKETS:
        process_bracket(bracket_name, global_processed)
    print("\nAll brackets complete.")


if __name__ == "__main__":
    main()