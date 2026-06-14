"""HR Oracle - main prediction pipeline.

Pitcher data is fetched from the MLB Stats API and passed through to
compute_model().  Previously every pitcher field was hard-coded to None
or 'R', which meant platoon splits, K/BB matchup adjustments, and the
pitcher-fatigue factor were always bypassed.
"""
import requests
import json
from datetime import datetime, date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from factors import compute_model

MLB_API = "https://statsapi.mlb.com/api/v1"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

# -- helpers ------------------------------------------------------------------

def fetch_json(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[WARN] Fetch error {url}: {e}")
        return None


def _safe_float(d, key, default=0.0):
    """Return float from dict; return default on missing/null/non-numeric."""
    v = d.get(key) if d else None
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(d, key, default=0):
    v = d.get(key) if d else None
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# -- pitcher fetching ---------------------------------------------------------

def fetch_pitcher_data(pitcher_id, season):
    """Fetch season pitching stats and last-3-game log for *pitcher_id*.

    Returns (pitcher_stat_dict, pitcher_l3_list, pitcher_throws_str, ip_season_float).
    All values fall back to safe defaults if the API call fails so that
    the rest of the pipeline always receives well-typed inputs.
    """
    if not pitcher_id:
        print("[WARN] No pitcher_id supplied - using league-average pitcher stats.")
        return None, None, "R", None

    # Season totals
    stats_url = (
        f"{MLB_API}/people/{pitcher_id}/stats"
        f"?stats=season&group=pitching&season={season}&sportId=1"
    )
    stats_data = fetch_json(stats_url)
    pitcher_stat = None
    ip_season = None
    throws = "R"  # safe default

    if stats_data:
        splits = (
            stats_data.get("stats", [{}])[0].get("splits", [])
            if stats_data.get("stats") else []
        )
        if splits:
            raw = splits[0].get("stat", {})
            ip_season = _safe_float(raw, "inningsPitched")
            pitcher_stat = {
                "hr9":   _safe_float(raw, "homeRunsPer9"),
                "kPct":  _safe_float(raw, "strikeoutsPer9") / 27.0,
                "bbPct": _safe_float(raw, "walksPer9")    / 27.0,
                "fbPct": 0.38,   # not in basic API; use league average
                "era":   _safe_float(raw, "era"),
                "ip":    ip_season,
            }
            if pitcher_stat["hr9"] == 0.0 and pitcher_stat["era"] == 0.0:
                print(f"[WARN] Pitcher {pitcher_id}: all-zero stat block - using None.")
                pitcher_stat = None

    # Handedness from people endpoint
    people_url = f"{MLB_API}/people/{pitcher_id}"
    people_data = fetch_json(people_url)
    if people_data:
        people_list = people_data.get("people", [])
        if people_list:
            throws = people_list[0].get("pitchHand", {}).get("code", "R") or "R"

    # Last-3-game log
    log_url = (
        f"{MLB_API}/people/{pitcher_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}&sportId=1"
    )
    log_data = fetch_json(log_url)
    pitcher_l3 = None
    if log_data:
        log_splits = (
            log_data.get("stats", [{}])[0].get("splits", [])
            if log_data.get("stats") else []
        )
        recent = log_splits[-3:] if len(log_splits) >= 3 else log_splits
        if recent:
            pitcher_l3 = [
                {
                    "ip":  _safe_float(s.get("stat"), "inningsPitched"),
                    "hr":  _safe_int(s.get("stat"),   "homeRuns"),
                    "k":   _safe_int(s.get("stat"),   "strikeOuts"),
                    "bb":  _safe_int(s.get("stat"),   "baseOnBalls"),
                }
                for s in recent
            ]

    if pitcher_stat:
        print(
            f"[INFO] Pitcher {pitcher_id} ({throws}): "
            f"HR/9={pitcher_stat['hr9']:.2f}  "
            f"K%={pitcher_stat['kPct']:.3f}  "
            f"IP={ip_season}"
        )
    else:
        print(f"[WARN] Pitcher {pitcher_id}: stat fetch failed - model will use league averages.")

    return pitcher_stat, pitcher_l3, throws, ip_season


# -- weather ------------------------------------------------------------------

def fetch_weather(lat, lon):
    url = (
        f"{WEATHER_API}?latitude={lat}&longitude={lon}"
        "&hourly=wind_speed_10m,wind_direction_10m,temperature_2m"
        "&forecast_days=1&timezone=auto"
    )
    data = fetch_json(url)
    if not data:
        return None
    try:
        hourly = data["hourly"]
        idx = 13  # Use midday hour (1 PM)
        return {
            "wind_speed": hourly["wind_speed_10m"][idx],
            "wind_dir":   hourly["wind_direction_10m"][idx],
            "temp_c":     hourly["temperature_2m"][idx],
        }
    except (KeyError, IndexError):
        return None


# -- venue helpers ------------------------------------------------------------

# Minimal park factor table (neutral = 1.0).  Extend as needed.
PARK_FACTORS = {
    # park_id: (overall, lhf, rhf)
    2392: (1.05, 1.07, 1.03),  # Coors Field
    2395: (0.94, 0.93, 0.95),  # Petco Park
    2394: (1.02, 1.02, 1.02),  # Chase Field
    2680: (0.96, 0.96, 0.96),  # T-Mobile Park
    15:   (1.01, 1.01, 1.01),  # Fenway Park
}

# Rough lat/lon for weather lookups keyed by park_id.
PARK_COORDS = {
    2392: (39.7559, -104.9942),
    2395: (32.7073, -117.1566),
    2394: (33.4453, -112.0667),
    2680: (47.5914, -122.3325),
    15:   (42.3467, -71.0972),
}

# CF bearing (degrees) per park for wind calculation.
PARK_CF_BEARING = {
    2392: 25,
    2395: 300,
    2394: 30,
    2680: 15,
    15:   55,
}


def get_park_info(venue_id):
    pf_tuple = PARK_FACTORS.get(venue_id, (1.0, 1.0, 1.0))
    coords    = PARK_COORDS.get(venue_id)
    cf        = PARK_CF_BEARING.get(venue_id, 0)
    return pf_tuple, coords, cf


# -- main ---------------------------------------------------------------------

def main():
    print("Starting HR Oracle model run...")
    today  = date.today().isoformat()
    season = date.today().year

    schedule_url = f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=probablePitcher,venue"
    schedule_data = fetch_json(schedule_url)
    if not schedule_data or not schedule_data.get("dates"):
        print("No games today or API error")
        return

    games = []
    for dt in schedule_data["dates"]:
        games.extend(dt.get("games", []))

    if not games:
        print("No games scheduled")
        return

    print(f"Found {len(games)} games")

    # Cache pitcher fetches so each pitcher is only looked up once per run.
    _pitcher_cache = {}

    def get_pitcher(pid):
        if pid not in _pitcher_cache:
            _pitcher_cache[pid] = fetch_pitcher_data(pid, season)
        return _pitcher_cache[pid]

    predictions = []

    for game in games:
        game_pk = game.get("gamePk")
        if not game_pk:
            continue

        # -- venue / park info -----------------------------------------------
        venue_id   = game.get("venue", {}).get("id", 0)
        pf_tuple, coords, cf_bearing = get_park_info(venue_id)
        pf_overall, pf_lhf, pf_rhf = pf_tuple

        weather = None
        if coords:
            weather = fetch_weather(*coords)
            if weather:
                weather["cf_bearing"] = cf_bearing

        # -- probable pitchers -----------------------------------------------
        teams_sched = game.get("teams", {})
        away_pitcher_id = (
            teams_sched.get("away", {}).get("probablePitcher", {}).get("id")
        )
        home_pitcher_id = (
            teams_sched.get("home", {}).get("probablePitcher", {}).get("id")
        )

        # Pitcher seen by away batters = home team's starter, and vice-versa.
        away_pit_stat, away_pit_l3, away_pit_throws, away_pit_ip = get_pitcher(home_pitcher_id)
        home_pit_stat, home_pit_l3, home_pit_throws, home_pit_ip = get_pitcher(away_pitcher_id)

        # -- boxscore for lineup ---------------------------------------------
        boxscore_url = f"{MLB_API}/game/{game_pk}/boxscore"
        box = fetch_json(boxscore_url)
        if not box:
            continue

        teams = box.get("teams", {})

        for side in ["away", "home"]:
            opp_side  = "home" if side == "away" else "away"
            team_data = teams.get(side, {})
            opp_data  = teams.get(opp_side, {})

            if side == "away":
                pit_stat, pit_l3, pit_throws, pit_ip = away_pit_stat, away_pit_l3, away_pit_throws, away_pit_ip
                opp_pitcher_id = home_pitcher_id
            else:
                pit_stat, pit_l3, pit_throws, pit_ip = home_pit_stat, home_pit_l3, home_pit_throws, home_pit_ip
                opp_pitcher_id = away_pitcher_id

            team_abbr  = team_data.get("team", {}).get("abbreviation", "")
            opp_abbr   = opp_data.get("team",  {}).get("abbreviation", "")

            pitcher_name = ""
            if opp_pitcher_id:
                pd = fetch_json(f"{MLB_API}/people/{opp_pitcher_id}")
                if pd and pd.get("people"):
                    pitcher_name = pd["people"][0].get("fullName", "")

            # Validate pitcher data before processing batters.
            if pit_stat is None:
                print(
                    f"[WARN] {side.upper()} batters vs game {game_pk}: "
                    "pitcher stat unavailable - using league-average adjustments."
                )

            batters = team_data.get("batters", [])
            batter_order_map = {bid: idx + 1 for idx, bid in enumerate(batters)}

            # Estimate season_day (1-180)
            season_start = date(season, 3, 28)
            season_day   = max(1, (date.today() - season_start).days)

            for batter_id in batters:
                batter_stats = team_data.get("players", {}).get(f"ID{batter_id}", {})
                if not batter_stats:
                    continue

                person       = batter_stats.get("person", {})
                season_stats = batter_stats.get("seasonStats", {}).get("batting", {})

                # Map MLB API camelCase batting stats to the field names factors.py expects.
                batter_stat = {
                    "pa":  _safe_int(season_stats,   "plateAppearances"),
                    "ab":  _safe_int(season_stats,   "atBats"),
                    "hr":  _safe_int(season_stats,   "homeRuns"),
                    "bb":  _safe_int(season_stats,   "baseOnBalls"),
                    "k":   _safe_int(season_stats,   "strikeOuts"),
                    "sb":  _safe_int(season_stats,   "stolenBases"),
                    "dbl": _safe_int(season_stats,   "doubles"),
                    "trp": _safe_int(season_stats,   "triples"),
                    "avg": _safe_float(season_stats, "avg"),
                    "obp": _safe_float(season_stats, "obp"),
                    "slg": _safe_float(season_stats, "slg"),
                    "ops": _safe_float(season_stats, "ops"),
                }

                batting_order = batter_order_map.get(batter_id, 5)

                game_ctx = {
                    "team_abbr": team_abbr,
                    "opp_abbr":  opp_abbr,
                    "isAway":    side == "away",
                    "pitThrows": pit_throws,
                    "order":     batting_order,
                    "dayNight":  game.get("dayNight", "night"),
                    "pitcher":   pitcher_name,
                    "park_lhf":  pf_lhf,
                    "park_rhf":  pf_rhf,
                }

                result = compute_model(
                    batter=person,
                    batter_stat=batter_stat,
                    batter_l7=None,
                    batter_l14=None,
                    vs_hand=None,
                    h2h=None,
                    pitcher_stat=pit_stat,
                    pitcher_l3=pit_l3,
                    game=game_ctx,
                    weather=weather,
                    park_factor=pf_overall,
                    season_day=season_day,
                    pitcher_ip_season=pit_ip,
                )

                if result:
                    predictions.append(result)

    predictions.sort(key=lambda x: x["gameProb"], reverse=True)

    # -- validation summary --------------------------------------------------
    missing_pitcher = sum(1 for p in predictions if not p.get("pitcher"))
    if missing_pitcher:
        print(f"[WARN] {missing_pitcher}/{len(predictions)} predictions have no pitcher name.")

    output = {
        "updatedAt": today,
        "season":    season,
        "predictions": predictions,
    }

    Path("data").mkdir(exist_ok=True)
    Path("data/history").mkdir(exist_ok=True)

    with open("data/predictions.json", "w") as f:
        json.dump(output, f, indent=2)

    with open(f"data/history/{today}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(predictions)} predictions to data/predictions.json")


if __name__ == "__main__":
    main()
