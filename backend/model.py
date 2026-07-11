"""HR Oracle - main prediction pipeline.

Fetches real season stats, recent form, platoon splits, head-to-head
history, and pitcher stats from the MLB Stats API for every batter in
today's slate, then feeds them all into compute_model().  Earlier
versions of this pipeline fetched pitcher stats but never actually used
them (pitcher_hr_mult/pitcher_k_mult were hard-coded to 1.0), never
computed ISO (so iso_mult was always exactly 1.0), never fetched batter
recent-form/splits/H2H (always None), and only had park factors for 5 of
30 stadiums -- which is why predicted probabilities barely varied
between players.
"""
import requests
import json
from datetime import datetime, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.insert(0, str(Path(__file__).parent))
from factors import compute_model

MLB_API = "https://statsapi.mlb.com/api/v1"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"
BATTER_FETCH_WORKERS = 8

# -- helpers ----------------------------------------------------------

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


# -- pitcher fetching -------------------------------------------------

def fetch_pitcher_data(pitcher_id, season):
    """Fetch season pitching stats and last-3-game log for *pitcher_id*.

    Returns (pitcher_stat_dict, pitcher_l3_list, pitcher_throws_str,
    ip_season_float, full_name_str). All values fall back to safe
    defaults if the API call fails so the rest of the pipeline always
    receives well-typed inputs.
    """
    if not pitcher_id:
        print("[WARN] No pitcher_id supplied - using league-average pitcher stats.")
        return None, None, "R", None, ""

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
            bf = _safe_int(raw, "battersFaced")
            k = _safe_int(raw, "strikeOuts")
            bb = _safe_int(raw, "baseOnBalls")
            # groundOutsToAirouts is a GB:FB-ish ratio; lower ratio means more
            # fly balls, which correlates with more home runs allowed.
            gb_ratio = _safe_float(raw, "groundOutsToAirouts", 1.0) or 1.0
            fb_pct = max(0.30, min(0.72, 0.42 + 0.12 * (1 / max(gb_ratio, 0.5))))
            pitcher_stat = {
                "hr9":   _safe_float(raw, "homeRunsPer9"),
                "kPct":  (k / bf) if bf > 0 else _safe_float(raw, "strikeoutsPer9") / 27.0,
                "bbPct": (bb / bf) if bf > 0 else _safe_float(raw, "walksPer9") / 27.0,
                "fbPct": fb_pct,
                "era":   _safe_float(raw, "era"),
                "whip":  _safe_float(raw, "whip", 1.30) or 1.30,
                "bf":    bf,
                "ip":    ip_season,
            }
            if pitcher_stat["hr9"] == 0.0 and pitcher_stat["era"] == 0.0:
                print(f"[WARN] Pitcher {pitcher_id}: all-zero stat block - using None.")
                pitcher_stat = None

    # Handedness + full name from people endpoint
    people_url = f"{MLB_API}/people/{pitcher_id}"
    people_data = fetch_json(people_url)
    full_name = ""
    if people_data:
        people_list = people_data.get("people", [])
        if people_list:
            throws = people_list[0].get("pitchHand", {}).get("code", "R") or "R"
            full_name = people_list[0].get("fullName", "")

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

    return pitcher_stat, pitcher_l3, throws, ip_season, full_name


def fetch_team_bullpen_hr9(team_id, season):
    """Team-level season pitching HR/9 -- used as a proxy for the quality of
    the pitching staff the batter's team will face over the whole game
    (there is no free, simple bullpen-only endpoint, so the team total,
    which includes the probable starter, is the best available signal)."""
    if not team_id:
        return None
    url = (
        f"{MLB_API}/teams/{team_id}/stats"
        f"?stats=season&group=pitching&season={season}&sportId=1"
    )
    data = fetch_json(url)
    if not data:
        return None
    splits = data.get("stats", [{}])[0].get("splits", []) if data.get("stats") else []
    if not splits:
        return None
    raw = splits[0].get("stat", {})
    ip = _safe_float(raw, "inningsPitched")
    hr = _safe_int(raw, "homeRuns")
    if ip <= 0:
        return None
    return hr / ip * 9


# -- batter enrichment fetching ----------------------------------------

def fetch_batter_recent(batter_id, season, limit):
    """Aggregate the last *limit* games of hitting for a batter."""
    url = (
        f"{MLB_API}/people/{batter_id}/stats?stats=lastXGames&group=hitting"
        f"&season={season}&gameType=R&limit={limit}"
    )
    data = fetch_json(url)
    if not data:
        return None
    splits = data.get("stats", [{}])[0].get("splits", []) if data.get("stats") else []
    if not splits:
        return None
    agg = {"pa": 0, "hr": 0, "k": 0, "bb": 0, "ab": 0, "h": 0, "dbl": 0, "trp": 0}
    for s in splits:
        st = s.get("stat", {}) or {}
        agg["pa"]  += _safe_int(st, "plateAppearances")
        agg["hr"]  += _safe_int(st, "homeRuns")
        agg["k"]   += _safe_int(st, "strikeOuts")
        agg["bb"]  += _safe_int(st, "baseOnBalls")
        agg["ab"]  += _safe_int(st, "atBats")
        agg["h"]   += _safe_int(st, "hits")
        agg["dbl"] += _safe_int(st, "doubles")
        agg["trp"] += _safe_int(st, "triples")
    singles = max(agg["h"] - agg["dbl"] - agg["trp"] - agg["hr"], 0)
    total_bases = singles + 2 * agg["dbl"] + 3 * agg["trp"] + 4 * agg["hr"]
    agg["slg"] = round(total_bases / agg["ab"], 3) if agg["ab"] > 0 else None
    return agg


def fetch_batter_split(batter_id, season, opp_pitcher_hand):
    """Season stats vs LHP or vs RHP -- whichever hand today's actual
    starter throws, so this reflects the real matchup rather than a
    generic 'vs opposite hand' guess."""
    code = "vl" if opp_pitcher_hand == "L" else "vr"
    url = (
        f"{MLB_API}/people/{batter_id}/stats?stats=statSplits&group=hitting"
        f"&season={season}&gameType=R&sitCodes={code}"
    )
    data = fetch_json(url)
    if not data:
        return None
    splits = data.get("stats", [{}])[0].get("splits", []) if data.get("stats") else []
    for sp in splits:
        if sp.get("split", {}).get("code") == code:
            st = sp.get("stat", {}) or {}
            pa = _safe_int(st, "plateAppearances") or _safe_int(st, "atBats")
            return {
                "pa": pa,
                "hr": _safe_int(st, "homeRuns"),
                # obp/slg are unused by the HR model's vs_mult factor but are
                # reused by game_model.py to build a lineup-level OPS-vs-hand
                # signal for the moneyline model, so both share one fetch.
                "obp": _safe_float(st, "obp"),
                "slg": _safe_float(st, "slg"),
            }
    return None


def fetch_h2h(batter_id, pitcher_id):
    """Career at-bats for this batter against this specific pitcher."""
    if not batter_id or not pitcher_id:
        return None
    url = (
        f"{MLB_API}/people/{batter_id}/stats?stats=vsPlayer&group=hitting"
        f"&opposingPlayerId={pitcher_id}"
    )
    data = fetch_json(url)
    if not data:
        return None
    splits = data.get("stats", [{}])[0].get("splits", []) if data.get("stats") else []
    if not splits:
        return None
    st = splits[0].get("stat", {}) or {}
    pa = _safe_int(st, "plateAppearances") or _safe_int(st, "atBats")
    return {"pa": pa, "hr": _safe_int(st, "homeRuns")}


def fetch_batter_person(batter_id):
    data = fetch_json(f"{MLB_API}/people/{batter_id}")
    if not data or not data.get("people"):
        return None
    p = data["people"][0]
    return {
        "fullName": p.get("fullName", ""),
        "bats": p.get("batSide", {}).get("code", "R") or "R",
    }


def fetch_batter_extra(batter_id, opp_pitcher_id, opp_pitcher_hand, season):
    """Everything needed for the recent-form/splits/H2H factors, bundled so
    it can be dispatched to a thread pool (this is the O(hundreds) fetch
    stage of the run)."""
    return {
        "person":  fetch_batter_person(batter_id),
        "l7":      fetch_batter_recent(batter_id, season, 7),
        "l14":     fetch_batter_recent(batter_id, season, 14),
        "vs_hand": fetch_batter_split(batter_id, season, opp_pitcher_hand),
        "h2h":     fetch_h2h(batter_id, opp_pitcher_id),
    }


# -- weather ----------------------------------------------------------

def fetch_weather(lat, lon):
    url = (
        f"{WEATHER_API}?latitude={lat}&longitude={lon}"
        "&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,precipitation_probability"
        "&temperature_unit=fahrenheit&wind_speed_unit=mph"
        "&forecast_days=1&timezone=auto"
    )
    data = fetch_json(url)
    if not data:
        return None
    try:
        hourly = data["hourly"]
        idx = 19  # ~7 PM local, typical first pitch window
        return {
            "windSpeed": hourly["wind_speed_10m"][idx],
            "windDir":   hourly["wind_direction_10m"][idx],
            "temp":      hourly["temperature_2m"][idx],
            "precipProb": hourly.get("precipitation_probability", [0] * (idx + 1))[idx],
        }
    except (KeyError, IndexError):
        return None


# -- venue / park helpers ----------------------------------------------

# Real park factors, hand-specific factors, coordinates, CF bearing and
# altitude for every MLB stadium, keyed by the *home team's* team id (the
# same real, verified numbers already used by the client-side model in
# index.html -- kept as a single source of truth instead of the previous
# 5-stadium table that left 25 teams defaulting to a flat neutral factor
# and no weather data at all).
PARK_FACTORS = {
    108: {"name": "Angel Stadium",        "factor": 0.96, "lhf": 0.93, "rhf": 0.99, "lat": 33.800, "lon": -117.883, "cfBearing": 5,   "alt": 150},
    109: {"name": "Chase Field",          "factor": 1.08, "lhf": 1.10, "rhf": 1.06, "lat": 33.446, "lon": -112.067, "cfBearing": 355, "alt": 1100, "retractable": True},
    110: {"name": "Camden Yards",         "factor": 1.10, "lhf": 1.12, "rhf": 1.08, "lat": 39.284, "lon": -76.622,  "cfBearing": 340, "alt": 20},
    111: {"name": "Fenway Park",          "factor": 1.04, "lhf": 1.06, "rhf": 1.02, "lat": 42.347, "lon": -71.097,  "cfBearing": 330, "alt": 20},
    112: {"name": "Wrigley Field",        "factor": 1.02, "lhf": 1.04, "rhf": 1.00, "lat": 41.948, "lon": -87.655,  "cfBearing": 20,  "alt": 595},
    113: {"name": "Great American BP",    "factor": 1.16, "lhf": 1.18, "rhf": 1.14, "lat": 39.098, "lon": -84.507,  "cfBearing": 10,  "alt": 490},
    114: {"name": "Progressive Field",    "factor": 0.93, "lhf": 0.91, "rhf": 0.95, "lat": 41.496, "lon": -81.685,  "cfBearing": 15,  "alt": 660},
    115: {"name": "Coors Field",          "factor": 1.38, "lhf": 1.40, "rhf": 1.36, "lat": 39.756, "lon": -104.994, "cfBearing": 5,   "alt": 5280},
    116: {"name": "Comerica Park",        "factor": 0.94, "lhf": 0.92, "rhf": 0.96, "lat": 42.339, "lon": -83.049,  "cfBearing": 355, "alt": 583},
    117: {"name": "Minute Maid Park",     "factor": 1.09, "lhf": 1.12, "rhf": 1.06, "lat": 29.757, "lon": -95.355,  "cfBearing": 355, "alt": 22,  "retractable": True},
    118: {"name": "Kauffman Stadium",     "factor": 0.95, "lhf": 0.93, "rhf": 0.97, "lat": 39.051, "lon": -94.480,  "cfBearing": 5,   "alt": 750},
    119: {"name": "Dodger Stadium",       "factor": 0.95, "lhf": 0.93, "rhf": 0.97, "lat": 34.074, "lon": -118.240, "cfBearing": 20,  "alt": 515},
    120: {"name": "Nationals Park",       "factor": 1.03, "lhf": 1.04, "rhf": 1.02, "lat": 38.873, "lon": -77.008,  "cfBearing": 5,   "alt": 25},
    121: {"name": "Citi Field",           "factor": 0.95, "lhf": 0.93, "rhf": 0.97, "lat": 40.757, "lon": -73.846,  "cfBearing": 15,  "alt": 20},
    133: {"name": "Sutter Health Park",   "factor": 0.94, "lhf": 0.92, "rhf": 0.96, "lat": 38.576, "lon": -121.508, "cfBearing": 340, "alt": 30},
    134: {"name": "PNC Park",             "factor": 0.96, "lhf": 0.94, "rhf": 0.98, "lat": 40.447, "lon": -80.006,  "cfBearing": 330, "alt": 730},
    135: {"name": "Petco Park",           "factor": 0.89, "lhf": 0.87, "rhf": 0.91, "lat": 32.707, "lon": -117.157, "cfBearing": 20,  "alt": 62},
    136: {"name": "T-Mobile Park",        "factor": 0.91, "lhf": 0.89, "rhf": 0.93, "lat": 47.591, "lon": -122.332, "cfBearing": 350, "alt": 13,  "retractable": True},
    137: {"name": "Oracle Park",          "factor": 0.87, "lhf": 0.85, "rhf": 0.89, "lat": 37.778, "lon": -122.389, "cfBearing": 65,  "alt": 10},
    138: {"name": "Busch Stadium",        "factor": 0.96, "lhf": 0.94, "rhf": 0.98, "lat": 38.623, "lon": -90.193,  "cfBearing": 350, "alt": 465},
    139: {"name": "Tropicana Field",      "factor": 0.96, "lhf": 0.94, "rhf": 0.98, "lat": 27.768, "lon": -82.653,  "cfBearing": 5,   "alt": 10,  "dome": True},
    140: {"name": "Globe Life Field",     "factor": 1.04, "lhf": 1.06, "rhf": 1.02, "lat": 32.747, "lon": -97.082,  "cfBearing": 25,  "alt": 551, "retractable": True},
    141: {"name": "Rogers Centre",        "factor": 1.03, "lhf": 1.04, "rhf": 1.02, "lat": 43.641, "lon": -79.389,  "cfBearing": 350, "alt": 287, "retractable": True},
    142: {"name": "Target Field",         "factor": 0.97, "lhf": 0.96, "rhf": 0.98, "lat": 44.982, "lon": -93.278,  "cfBearing": 350, "alt": 840},
    143: {"name": "Citizens Bank Park",   "factor": 1.11, "lhf": 1.13, "rhf": 1.09, "lat": 39.906, "lon": -75.166,  "cfBearing": 5,   "alt": 20},
    144: {"name": "Truist Park",          "factor": 1.03, "lhf": 1.04, "rhf": 1.02, "lat": 33.891, "lon": -84.468,  "cfBearing": 355, "alt": 1050},
    145: {"name": "Guaranteed Rate Field","factor": 1.07, "lhf": 1.09, "rhf": 1.05, "lat": 41.830, "lon": -87.634,  "cfBearing": 10,  "alt": 595},
    146: {"name": "loanDepot Park",       "factor": 0.95, "lhf": 0.94, "rhf": 0.96, "lat": 25.778, "lon": -80.220,  "cfBearing": 10,  "alt": 6,   "retractable": True},
    147: {"name": "Yankee Stadium",       "factor": 1.15, "lhf": 1.22, "rhf": 1.08, "lat": 40.829, "lon": -73.926,  "cfBearing": 45,  "alt": 55},
    158: {"name": "American Family Field","factor": 1.05, "lhf": 1.07, "rhf": 1.03, "lat": 43.028, "lon": -87.971,  "cfBearing": 5,   "alt": 635, "retractable": True},
}


def get_park_info(team_id):
    """Returns (park_factor_dict, (lat, lon) | None, cf_bearing, enclosed)."""
    p = PARK_FACTORS.get(team_id)
    if not p:
        return {"lhf": 1.0, "rhf": 1.0, "cfBearing": 0}, None, 0, False
    park_factor = {"lhf": p["lhf"], "rhf": p["rhf"], "cfBearing": p["cfBearing"]}
    coords = (p["lat"], p["lon"])
    enclosed = bool(p.get("dome") or p.get("retractable"))
    return park_factor, coords, p["cfBearing"], enclosed


# -- main -------------------------------------------------------------

def main():
    print("Starting HR Oracle model run...")
    today  = date.today().isoformat()
    season = date.today().year

    schedule_url = f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=probablePitcher,venue,team"
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

    _bullpen_cache = {}

    def get_bullpen(team_id):
        if team_id not in _bullpen_cache:
            _bullpen_cache[team_id] = fetch_team_bullpen_hr9(team_id, season)
        return _bullpen_cache[team_id]

    # -- Pass 1: walk every game/side/batter and build a prediction context.
    # Recent-form, splits and H2H are deferred to a concurrent fetch stage
    # below since that's O(hundreds of batters) and would be far too slow
    # fetched one at a time.
    contexts = []
    weather_cache = {}

    for game in games:
        game_pk = game.get("gamePk")
        if not game_pk:
            continue

        teams_sched = game.get("teams", {})
        home_team_id = teams_sched.get("home", {}).get("team", {}).get("id")
        away_team_id = teams_sched.get("away", {}).get("team", {}).get("id")

        # -- venue / park info (keyed by home team, covers all 30 parks) --
        park_factor, coords, cf_bearing, enclosed = get_park_info(home_team_id)

        weather = None
        if coords and not enclosed:
            if coords not in weather_cache:
                weather_cache[coords] = fetch_weather(*coords)
            weather = weather_cache[coords]

        # -- probable pitchers ----------------------------------------
        away_pitcher_id = teams_sched.get("away", {}).get("probablePitcher", {}).get("id")
        home_pitcher_id = teams_sched.get("home", {}).get("probablePitcher", {}).get("id")

        # Pitcher seen by away batters = home team's starter, and vice-versa.
        away_pit_stat, away_pit_l3, away_pit_throws, away_pit_ip, away_pit_name = get_pitcher(home_pitcher_id)
        home_pit_stat, home_pit_l3, home_pit_throws, home_pit_ip, home_pit_name = get_pitcher(away_pitcher_id)

        away_bullpen_hr9 = get_bullpen(home_team_id)
        home_bullpen_hr9 = get_bullpen(away_team_id)

        # -- boxscore for lineup --------------------------------------
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
                pit_stat, pit_l3, pit_throws, pit_ip, pit_name = away_pit_stat, away_pit_l3, away_pit_throws, away_pit_ip, away_pit_name
                opp_pitcher_id = home_pitcher_id
                bullpen_hr9 = away_bullpen_hr9
            else:
                pit_stat, pit_l3, pit_throws, pit_ip, pit_name = home_pit_stat, home_pit_l3, home_pit_throws, home_pit_ip, home_pit_name
                opp_pitcher_id = away_pitcher_id
                bullpen_hr9 = home_bullpen_hr9

            team_abbr = team_data.get("team", {}).get("abbreviation", "")
            opp_abbr  = opp_data.get("team",  {}).get("abbreviation", "")

            if pit_stat is None:
                print(
                    f"[WARN] {side.upper()} batters vs game {game_pk}: "
                    "pitcher stat unavailable - using league-average adjustments."
                )

            batters = team_data.get("batters", [])
            batter_order_map = {bid: idx + 1 for idx, bid in enumerate(batters)}

            season_start = date(season, 3, 28)
            season_day   = max(1, (date.today() - season_start).days)

            for batter_id in batters:
                batter_stats = team_data.get("players", {}).get(f"ID{batter_id}", {})
                if not batter_stats:
                    continue

                person       = batter_stats.get("person", {})
                season_stats = batter_stats.get("seasonStats", {}).get("batting", {})

                pa = _safe_int(season_stats, "plateAppearances")
                if pa < 40:
                    continue  # compute_model requires >=40 PA; skip the extra fetches

                slg = _safe_float(season_stats, "slg")
                avg = _safe_float(season_stats, "avg")
                batter_stat = {
                    "pa":  pa,
                    "ab":  _safe_int(season_stats,   "atBats"),
                    "hr":  _safe_int(season_stats,   "homeRuns"),
                    "bb":  _safe_int(season_stats,   "baseOnBalls"),
                    "k":   _safe_int(season_stats,   "strikeOuts"),
                    "sb":  _safe_int(season_stats,   "stolenBases"),
                    "dbl": _safe_int(season_stats,   "doubles"),
                    "trp": _safe_int(season_stats,   "triples"),
                    "avg": avg,
                    "obp": _safe_float(season_stats, "obp"),
                    "slg": slg,
                    "ops": _safe_float(season_stats, "ops"),
                    "iso": round(max(0.02, min(0.45, slg - avg)), 3),
                }

                batting_order = batter_order_map.get(batter_id, 5)

                game_ctx = {
                    "team_abbr": team_abbr,
                    "opp_abbr":  opp_abbr,
                    "isAway":    side == "away",
                    "pitThrows": pit_throws,
                    "order":     batting_order,
                    "dayNight":  game.get("dayNight", "night"),
                    "pitcher":   pit_name,
                }

                contexts.append({
                    "batter_id": batter_id,
                    "person": person,
                    "batter_stat": batter_stat,
                    "game_ctx": game_ctx,
                    "pitcher_stat": pit_stat,
                    "pitcher_l3": pit_l3,
                    "weather": weather,
                    "park_factor": park_factor,
                    "season_day": season_day,
                    "bullpen_hr9": bullpen_hr9,
                    "pitcher_ip": pit_ip,
                    "opp_pitcher_id": opp_pitcher_id,
                    "opp_pitcher_hand": pit_throws,
                })

    if not contexts:
        print("No batters met the minimum PA threshold")
        return

    # -- Pass 2: concurrently fetch recent-form/splits/H2H for every unique
    # batter (deduping so a batter appearing more than once is only fetched
    # once).
    unique_batters = {}
    for ctx in contexts:
        unique_batters.setdefault(
            ctx["batter_id"], (ctx["opp_pitcher_id"], ctx["opp_pitcher_hand"])
        )

    print(f"Fetching recent form/splits/H2H for {len(unique_batters)} batters...")
    batter_extra = {}
    with ThreadPoolExecutor(max_workers=BATTER_FETCH_WORKERS) as ex:
        futures = {
            ex.submit(fetch_batter_extra, bid, opp_pid, opp_hand, season): bid
            for bid, (opp_pid, opp_hand) in unique_batters.items()
        }
        for fut in as_completed(futures):
            bid = futures[fut]
            try:
                batter_extra[bid] = fut.result()
            except Exception as e:
                print(f"[WARN] Batter {bid} enrichment fetch failed: {e}")
                batter_extra[bid] = {}

    # -- Pass 3: compute predictions with the full, real dataset ----------
    predictions = []
    for ctx in contexts:
        extra = batter_extra.get(ctx["batter_id"], {})
        person_full = extra.get("person") or {}
        batter = dict(ctx["person"])
        if person_full.get("fullName"):
            batter["fullName"] = person_full["fullName"]
        if person_full.get("bats"):
            batter["bats"] = person_full["bats"]

        result = compute_model(
            batter=batter,
            batter_stat=ctx["batter_stat"],
            batter_l7=extra.get("l7"),
            batter_l14=extra.get("l14"),
            vs_hand=extra.get("vs_hand"),
            h2h=extra.get("h2h"),
            pitcher_stat=ctx["pitcher_stat"],
            pitcher_l3=ctx["pitcher_l3"],
            game=ctx["game_ctx"],
            weather=ctx["weather"],
            park_factor=ctx["park_factor"],
            season_day=ctx["season_day"],
            bullpen_hr9=ctx["bullpen_hr9"],
            pitcher_ip_season=ctx["pitcher_ip"],
        )

        if result:
            predictions.append(result)

    predictions.sort(key=lambda x: x["gameProb"], reverse=True)

    # -- validation summary -------------------------------------------
    missing_pitcher = sum(1 for p in predictions if not p.get("pitcherName"))
    if missing_pitcher:
        print(f"[WARN] {missing_pitcher}/{len(predictions)} predictions have no pitcher name.")
    missing_name = sum(1 for p in predictions if not p.get("name"))
    if missing_name:
        print(f"[WARN] {missing_name}/{len(predictions)} predictions have no batter name.")

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
