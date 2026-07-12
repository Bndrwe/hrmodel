"""HR Oracle - game-level prediction engine.

Extends the HR-prop model with whole-game MLB predictions: moneylines,
run-line/alternate spreads, totals (over/under), first-5-innings (F5)
lines, NRFI/YRFI, and team total hits. These are the model's own
fair-value lines (no sportsbook odds feed is configured for this repo),
derived from real data:

  1. Each team's runs-scored/hits and runs-allowed per game (season).
  2. Today's probable starters, quality-graded by a regressed blend of
     ERA (actual run prevention) and FIP (defense-independent "true
     talent": HR/BB/K rate), so a single lucky or unlucky start doesn't
     dominate.
  3. Today's bullpen quality: every *other* pitcher on the active
     roster (i.e. not today's starter) weighted by innings pitched,
     blended the same ERA/FIP way -- a genuine bullpen-specific number
     instead of reusing team-wide totals, which would double-count the
     starter's own stat line.
  4. A batter-vs-pitcher-hand lineup matchup signal: the confirmed
     lineup's actual OPS against the opposing starter's throwing hand
     when it's posted (fetched per batter, concurrently), falling back
     to the team's own season-long vs-hand split when it isn't.
  5. The home park's run-scoring factor and a standard home-field bump.
  6. A single-game Pythagorean-win-expectation formula (exponent 1.83)
     applied to the two teams' fully adjusted expected runs, which
     gives the moneyline probability directly.
  7. A Normal approximation over the resulting run differential/total
     for run-line, alternate-spread, and total probabilities; F5 scales
     the same run model down to 5 innings; NRFI uses a Poisson
     zero-run probability on each team's first-inning run expectation.
"""
import json
import math
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from model import fetch_json, _safe_float, _safe_int, PARK_FACTORS, MLB_API, fetch_batter_split
from factors import clamp

LG_ERA = 4.15
FIP_CONSTANT = 3.10  # standard modern-day FIP constant
LG_RUNS_PER_GAME = 4.50
LG_HITS_PER_GAME = 8.50
LG_OPS_VS_HAND = 0.722  # roughly LG obp(.314) + slg(.408), either hand
PYTHAG_EXP = 1.83
# Empirically-typical standard deviations for single-game MLB run
# differential / total runs / team hits; used as a Normal approximation
# since we don't run per-game Monte Carlo simulation for team scoring
# (unlike the HR model, which does run one client-side).
STD_MARGIN = 4.4
STD_TOTAL = 4.6
STD_HITS = 3.0
HOME_FIELD_BUMP = 0.02  # ~2% scoring bump for the home team
STARTER_INNINGS_SHARE = 0.56  # starters average ~5 of 9 innings
F5_INNING_SHARE = 5 / 9
FIRST_INNING_SHARE = 1 / 9
FIRST_INNING_BUMP = 1.05  # the 1-2-3 hitters bat first; well-documented NRFI skew
GAME_FETCH_WORKERS = 4
SUB_FETCH_WORKERS = 8


def _load_game_model_weights():
    path = Path(__file__).resolve().parent.parent / "data" / "game_model_weights.json"
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

_GAME_MODEL_WEIGHTS = _load_game_model_weights()
_ML_CALIBRATION_BUCKETS = _GAME_MODEL_WEIGHTS.get("moneylineCalibration", [])


def calibrate_moneyline(p_home):
    """Nudges the Pythagorean home-win probability toward what's actually
    been observed for picks at this confidence level. grade_game_predictions.py
    runs every day, bucketing every moneyline pick by its predicted
    probability and tracking whether the favorite actually won -- this is
    what closes the loop, instead of the model running the same formula
    forever regardless of how it's actually performed. Only trusts buckets
    with >=20 tracked picks, and caps the correction at +/-8 points so a
    short bad or good streak can't swing it hard."""
    fav_prob = max(p_home, 1 - p_home)
    for b in _ML_CALIBRATION_BUCKETS:
        lo, hi, total = b.get("lo", 0), b.get("hi", 1), b.get("total", 0)
        if lo <= fav_prob < hi and total >= 20:
            observed = b.get("hits", 0) / total
            expected = b.get("sumExpected", total * (lo + hi) / 2) / total
            drift = observed - expected
            adjustment = clamp(drift * 0.4, -0.08, 0.08)
            fav_prob = clamp(fav_prob + adjustment, 0.51, 0.99)
            break
    return fav_prob if p_home >= 0.5 else 1 - fav_prob


def norm_cdf(x, mean, std):
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))


def prob_to_american_odds(p):
    p = clamp(p, 0.01, 0.99)
    if p >= 0.5:
        return -round(100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def _fip(hr, bb, k, ip):
    if ip <= 0:
        return None
    return (13 * hr + 3 * bb - 2 * k) / ip + FIP_CONSTANT


def get_team_run_environment(team_id, season):
    """Real season-to-date runs-scored/game, hits/game, and runs-allowed/game
    for a team, falling back to league average on any missing/failed data."""
    rs_per_game = LG_RUNS_PER_GAME
    hits_per_game = LG_HITS_PER_GAME
    ra_per_game = LG_RUNS_PER_GAME

    hitting = fetch_json(
        f"{MLB_API}/teams/{team_id}/stats?stats=season&group=hitting&season={season}&sportId=1"
    )
    games_played = 0
    if hitting:
        splits = hitting.get("stats", [{}])[0].get("splits", []) if hitting.get("stats") else []
        if splits:
            raw = splits[0].get("stat", {})
            games_played = _safe_int(raw, "gamesPlayed")
            runs = _safe_int(raw, "runs")
            hits = _safe_int(raw, "hits")
            if games_played > 0:
                rs_per_game = runs / games_played
                hits_per_game = hits / games_played

    pitching = fetch_json(
        f"{MLB_API}/teams/{team_id}/stats?stats=season&group=pitching&season={season}&sportId=1"
    )
    if pitching:
        splits = pitching.get("stats", [{}])[0].get("splits", []) if pitching.get("stats") else []
        if splits:
            raw = splits[0].get("stat", {})
            gp = _safe_int(raw, "gamesPlayed") or games_played
            runs_allowed = _safe_int(raw, "runs")
            if gp > 0:
                ra_per_game = runs_allowed / gp

    return rs_per_game, hits_per_game, ra_per_game


def fetch_pitcher_hand(pitcher_id):
    if not pitcher_id:
        return "R"
    data = fetch_json(f"{MLB_API}/people/{pitcher_id}")
    if not data or not data.get("people"):
        return "R"
    return data["people"][0].get("pitchHand", {}).get("code", "R") or "R"


def fetch_starter_quality(pitcher_id, season):
    """Regressed ERA/FIP blend for a probable starter (league-average when
    unknown or on a tiny sample). FIP isolates the pitcher's own strikeout,
    walk and home-run rates from defense/luck, so blending it with ERA is
    more predictive of *future* run prevention than ERA alone."""
    if not pitcher_id:
        return LG_ERA
    data = fetch_json(
        f"{MLB_API}/people/{pitcher_id}/stats?stats=season&group=pitching&season={season}&sportId=1"
    )
    if not data:
        return LG_ERA
    splits = data.get("stats", [{}])[0].get("splits", []) if data.get("stats") else []
    if not splits:
        return LG_ERA
    raw = splits[0].get("stat", {})
    era = _safe_float(raw, "era", LG_ERA) or LG_ERA
    ip = _safe_float(raw, "inningsPitched")
    fip = _fip(_safe_int(raw, "homeRuns"), _safe_int(raw, "baseOnBalls"), _safe_int(raw, "strikeOuts"), ip)
    blended = 0.6 * era + 0.4 * fip if fip is not None else era
    bf = _safe_int(raw, "battersFaced")
    prw = clamp(bf / 300, 0.10, 0.85)
    return prw * blended + (1 - prw) * LG_ERA


def fetch_bullpen_quality(team_id, starter_id, season):
    """Innings-weighted ERA/FIP blend across every *other* pitcher on the
    active roster (today's starter excluded), so the bullpen's actual
    current quality is isolated instead of reusing team-wide pitching
    totals, which bake the starter's own line back in. Returns None if the
    roster can't be fetched (falls back to team-wide numbers upstream)."""
    roster = fetch_json(f"{MLB_API}/teams/{team_id}/roster/Active?season={season}")
    if not roster or not roster.get("roster"):
        return None
    pitcher_ids = [
        p["person"]["id"] for p in roster["roster"]
        if p.get("position", {}).get("abbreviation") in ("P", "SP", "RP")
        and p.get("person", {}).get("id") != starter_id
    ]
    if not pitcher_ids:
        return None

    def _fetch_one(pid):
        return fetch_json(f"{MLB_API}/people/{pid}/stats?stats=season&group=pitching&season={season}&sportId=1")

    total_ip = 0.0
    weighted = 0.0
    with ThreadPoolExecutor(max_workers=SUB_FETCH_WORKERS) as ex:
        for data in ex.map(_fetch_one, pitcher_ids):
            if not data:
                continue
            splits = data.get("stats", [{}])[0].get("splits", []) if data.get("stats") else []
            if not splits:
                continue
            raw = splits[0].get("stat", {})
            ip = _safe_float(raw, "inningsPitched")
            if ip < 5:  # skip token/rehab-stint appearances
                continue
            era = _safe_float(raw, "era", LG_ERA) or LG_ERA
            fip = _fip(_safe_int(raw, "homeRuns"), _safe_int(raw, "baseOnBalls"), _safe_int(raw, "strikeOuts"), ip)
            blended = 0.6 * era + 0.4 * fip if fip is not None else era
            weighted += blended * ip
            total_ip += ip

    if total_ip < 20:
        return None
    return weighted / total_ip


def combined_pitching_quality(starter_quality, pen_quality):
    if pen_quality is None:
        return starter_quality
    return STARTER_INNINGS_SHARE * starter_quality + (1 - STARTER_INNINGS_SHARE) * pen_quality


def fetch_team_ops_vs_hand(team_id, season, pitcher_hand):
    """Team-level season OPS vs LHP/RHP -- the aggregate of exactly the
    batters who actually play for this team, used as the fallback signal
    when today's specific lineup hasn't posted yet."""
    code = "vl" if pitcher_hand == "L" else "vr"
    data = fetch_json(
        f"{MLB_API}/teams/{team_id}/stats?stats=statSplits&group=hitting"
        f"&season={season}&gameType=R&sitCodes={code}"
    )
    if not data:
        return None
    splits = data.get("stats", [{}])[0].get("splits", []) if data.get("stats") else []
    for sp in splits:
        if sp.get("split", {}).get("code") == code:
            st = sp.get("stat", {}) or {}
            obp = _safe_float(st, "obp")
            slg = _safe_float(st, "slg")
            if obp or slg:
                return obp + slg
    return None


def fetch_lineup_ops_vs_hand(game_pk, side, pitcher_hand, season):
    """If today's lineup is already posted in the boxscore, fetch each
    confirmed batter's actual season split vs this pitcher hand
    (concurrently) and return the PA-weighted OPS. This is per-batter,
    matchup-specific analysis -- who is *actually* in the lineup today,
    not just the team's average -- but only available once lineups post
    (usually 1-3 hours before first pitch), so callers should fall back to
    fetch_team_ops_vs_hand() when this returns None."""
    box = fetch_json(f"{MLB_API}/game/{game_pk}/boxscore")
    if not box:
        return None
    batters = box.get("teams", {}).get(side, {}).get("batters", [])
    if not batters:
        return None

    with ThreadPoolExecutor(max_workers=SUB_FETCH_WORKERS) as ex:
        results = list(ex.map(lambda bid: fetch_batter_split(bid, season, pitcher_hand), batters))

    total_pa = 0
    weighted_ops = 0.0
    for r in results:
        if r and r.get("pa", 0) >= 15 and (r.get("obp") or r.get("slg")):
            ops = (r.get("obp") or 0) + (r.get("slg") or 0)
            weighted_ops += ops * r["pa"]
            total_pa += r["pa"]

    if total_pa < 80:  # not enough combined sample across the lineup yet
        return None
    return weighted_ops / total_pa


def get_lineup_matchup_mult(game_pk, side, team_id, pitcher_hand, season):
    """Batter-vs-pitcher-hand matchup multiplier for a team's offense today.
    Real per-batter tendencies against this specific pitcher hand when the
    lineup is posted, otherwise the team's own season vs-hand tendency --
    either way, real MLB data, not a generic platoon guess."""
    ops_vs_hand = fetch_lineup_ops_vs_hand(game_pk, side, pitcher_hand, season)
    source = "lineup"
    if ops_vs_hand is None:
        ops_vs_hand = fetch_team_ops_vs_hand(team_id, season, pitcher_hand)
        source = "teamSplit"
    if not ops_vs_hand:
        return 1.0, "none"
    ratio = ops_vs_hand / LG_OPS_VS_HAND
    # Dampened to 60% strength: this rides on top of the team's overall
    # run rate, which already reflects a roughly average mix of matchups
    # over the season, so the adjustment should be a nudge, not a redo.
    return clamp(1 + (ratio - 1) * 0.6, 0.85, 1.20), source


def build_game_prediction(game, season):
    game_pk = game.get("gamePk")
    teams = game.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})
    home_team = home.get("team", {})
    away_team = away.get("team", {})
    home_id = home_team.get("id")
    away_id = away_team.get("id")
    if not home_id or not away_id:
        return None

    home_pitcher = home.get("probablePitcher", {})
    away_pitcher = away.get("probablePitcher", {})
    home_pitcher_id = home_pitcher.get("id")
    away_pitcher_id = away_pitcher.get("id")

    home_rs, home_hits_pg, home_ra = get_team_run_environment(home_id, season)
    away_rs, away_hits_pg, away_ra = get_team_run_environment(away_id, season)

    home_starter_q = fetch_starter_quality(home_pitcher_id, season)
    away_starter_q = fetch_starter_quality(away_pitcher_id, season)
    home_pen_q = fetch_bullpen_quality(home_id, home_pitcher_id, season)
    away_pen_q = fetch_bullpen_quality(away_id, away_pitcher_id, season)
    home_pitching_q = combined_pitching_quality(home_starter_q, home_pen_q)
    away_pitching_q = combined_pitching_quality(away_starter_q, away_pen_q)

    home_pitcher_hand = fetch_pitcher_hand(home_pitcher_id)
    away_pitcher_hand = fetch_pitcher_hand(away_pitcher_id)

    # Home batters face the AWAY starter's hand, and vice-versa.
    home_matchup_mult, home_matchup_src = get_lineup_matchup_mult(game_pk, "home", home_id, away_pitcher_hand, season)
    away_matchup_mult, away_matchup_src = get_lineup_matchup_mult(game_pk, "away", away_id, home_pitcher_hand, season)

    park = PARK_FACTORS.get(home_id, {})
    # The power (HR) park factor is reused as an approximate run/hits-scoring
    # factor -- hitter-friendly power parks are, in practice, usually also
    # higher-scoring parks, but home runs move far more than total runs do,
    # so the effect is dampened to half strength (e.g. Coors' +38% HR factor
    # becomes a more realistic +19% run-scoring bump) rather than applied
    # 1:1, since maintaining an entirely separate verified run-factor table
    # isn't practical here.
    hr_park_factor = park.get("factor", 1.0)
    park_run_factor = 1.0 + (hr_park_factor - 1.0) * 0.5

    # Pitching strength facing each offense = blend of that opponent's
    # overall season run-prevention (their defense/team-wide numbers) and
    # today's specific starter+bullpen quality (real, matchup-specific arms).
    away_pitching_ratio = 0.45 * (away_ra / LG_RUNS_PER_GAME) + 0.55 * (away_pitching_q / LG_ERA)
    home_pitching_ratio = 0.45 * (home_ra / LG_RUNS_PER_GAME) + 0.55 * (home_pitching_q / LG_ERA)

    exp_runs_home = home_rs * away_pitching_ratio * home_matchup_mult * park_run_factor * (1 + HOME_FIELD_BUMP)
    exp_runs_away = away_rs * home_pitching_ratio * away_matchup_mult * park_run_factor * (1 - HOME_FIELD_BUMP)
    exp_runs_home = clamp(exp_runs_home, 1.5, 10.0)
    exp_runs_away = clamp(exp_runs_away, 1.5, 10.0)

    p_home = (exp_runs_home ** PYTHAG_EXP) / (exp_runs_home ** PYTHAG_EXP + exp_runs_away ** PYTHAG_EXP)
    p_home = calibrate_moneyline(p_home)
    p_away = 1 - p_home

    mean_margin = exp_runs_home - exp_runs_away
    mean_total = exp_runs_home + exp_runs_away

    def run_line(line):
        # line is the home team's line (negative = home favored by that much)
        home_cover = 1 - norm_cdf(-line, mean_margin, STD_MARGIN)
        return {
            "homeLine": round(line, 1),
            "awayLine": round(-line, 1),
            "homeCoverProb": round(home_cover, 4),
            "awayCoverProb": round(1 - home_cover, 4),
        }

    model_total = round(mean_total * 2) / 2  # nearest half-run
    total_lines = []
    for offset in (-1.5, -0.5, 0.5, 1.5):
        line = round(model_total + offset, 1)
        under_prob = norm_cdf(line, mean_total, STD_TOTAL)
        total_lines.append({
            "line": line,
            "overProb": round(1 - under_prob, 4),
            "underProb": round(under_prob, 4),
        })

    # -- F5 (first 5 innings): scale the same run model down to 5 innings.
    # Variance scales with time-in-game, so std is scaled by sqrt(innings
    # share) rather than carried over 1:1 from the full-game model.
    exp_runs_home_f5 = exp_runs_home * F5_INNING_SHARE
    exp_runs_away_f5 = exp_runs_away * F5_INNING_SHARE
    std_margin_f5 = STD_MARGIN * math.sqrt(F5_INNING_SHARE)
    std_total_f5 = STD_TOTAL * math.sqrt(F5_INNING_SHARE)
    p_home_f5 = (exp_runs_home_f5 ** PYTHAG_EXP) / (exp_runs_home_f5 ** PYTHAG_EXP + exp_runs_away_f5 ** PYTHAG_EXP)
    mean_margin_f5 = exp_runs_home_f5 - exp_runs_away_f5
    mean_total_f5 = exp_runs_home_f5 + exp_runs_away_f5
    f5_home_cover = 1 - norm_cdf(0.0, mean_margin_f5, std_margin_f5)
    f5_model_total = round(mean_total_f5 * 2) / 2
    f5 = {
        "expectedRuns": {"home": round(exp_runs_home_f5, 2), "away": round(exp_runs_away_f5, 2)},
        "moneyline": {
            # F5 can end tied (ties aren't modeled by this continuous
            # approximation -- treat as a simplification of the real
            # win/loss/push market).
            "homeProb": round(p_home_f5, 4),
            "awayProb": round(1 - p_home_f5, 4),
            "homeOdds": prob_to_american_odds(p_home_f5),
            "awayOdds": prob_to_american_odds(1 - p_home_f5),
        },
        "runLine": {
            "homeLine": -0.5, "awayLine": 0.5,
            "homeCoverProb": round(f5_home_cover, 4),
            "awayCoverProb": round(1 - f5_home_cover, 4),
        },
        "total": {
            "modelTotal": f5_model_total,
            "overProb": round(1 - norm_cdf(f5_model_total, mean_total_f5, std_total_f5), 4),
            "underProb": round(norm_cdf(f5_model_total, mean_total_f5, std_total_f5), 4),
        },
    }

    # -- NRFI/YRFI: Poisson zero-run probability on each team's 1st-inning
    # run expectation (their full-game rate divided across 9 innings, with
    # a modest bump since the best hitters bat 1-2-3 to lead off the game).
    lam_home_1st = exp_runs_home * FIRST_INNING_SHARE * FIRST_INNING_BUMP
    lam_away_1st = exp_runs_away * FIRST_INNING_SHARE * FIRST_INNING_BUMP
    nrfi_prob = math.exp(-lam_home_1st) * math.exp(-lam_away_1st)
    nrfi = {
        "expectedFirstInningRuns": {"home": round(lam_home_1st, 3), "away": round(lam_away_1st, 3)},
        "nrfiProb": round(nrfi_prob, 4),
        "yrfiProb": round(1 - nrfi_prob, 4),
    }

    # -- Team total hits (over/under): each team's hits/game rate adjusted
    # by the opposing pitching staff's run-prevention ratio (a real proxy
    # for baserunners allowed) and park.
    exp_hits_home = clamp(home_hits_pg * away_pitching_ratio * park_run_factor, 4.0, 14.0)
    exp_hits_away = clamp(away_hits_pg * home_pitching_ratio * park_run_factor, 4.0, 14.0)
    hits_total = {}
    for label, exp_hits in (("home", exp_hits_home), ("away", exp_hits_away)):
        model_hits_line = round(exp_hits * 2) / 2
        under = norm_cdf(model_hits_line, exp_hits, STD_HITS)
        hits_total[label] = {
            "team": home_team.get("abbreviation", "") if label == "home" else away_team.get("abbreviation", ""),
            "expectedHits": round(exp_hits, 2),
            "modelLine": model_hits_line,
            "overProb": round(1 - under, 4),
            "underProb": round(under, 4),
        }

    return {
        "gamePk": game_pk,
        "gameDate": game.get("gameDate"),
        "venue": game.get("venue", {}).get("name", park.get("name", "")),
        "homeTeam": home_team.get("abbreviation", ""),
        "awayTeam": away_team.get("abbreviation", ""),
        "homeTeamId": home_id,
        "awayTeamId": away_id,
        "homePitcher": home_pitcher.get("fullName", "TBD"),
        "awayPitcher": away_pitcher.get("fullName", "TBD"),
        "expectedRuns": {"home": round(exp_runs_home, 2), "away": round(exp_runs_away, 2)},
        "pitchingQuality": {
            "homeStarter": round(home_starter_q, 2), "awayStarter": round(away_starter_q, 2),
            "homePen": round(home_pen_q, 2) if home_pen_q else None,
            "awayPen": round(away_pen_q, 2) if away_pen_q else None,
        },
        "lineupMatchup": {
            "home": {"mult": round(home_matchup_mult, 3), "source": home_matchup_src},
            "away": {"mult": round(away_matchup_mult, 3), "source": away_matchup_src},
        },
        "moneyline": {
            "homeProb": round(p_home, 4),
            "awayProb": round(p_away, 4),
            "homeOdds": prob_to_american_odds(p_home),
            "awayOdds": prob_to_american_odds(p_away),
        },
        "runLine": {
            "standard": run_line(-1.5),
            "alternates": [run_line(-1.0), run_line(-2.5)],
        },
        "total": {
            "modelTotal": model_total,
            "lines": total_lines,
        },
        "f5": f5,
        "nrfi": nrfi,
        "teamHits": hits_total,
    }


def main():
    print("Starting HR Oracle game-prediction model run...")
    today = date.today().isoformat()
    season = date.today().year

    schedule_url = f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=probablePitcher,team,venue,linescore"
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

    predictions = []
    with ThreadPoolExecutor(max_workers=GAME_FETCH_WORKERS) as ex:
        futures = {ex.submit(build_game_prediction, game, season): game for game in games}
        for fut in as_completed(futures):
            game = futures[fut]
            try:
                pred = fut.result()
            except Exception as e:
                print(f"[WARN] Failed to build prediction for game {game.get('gamePk')}: {e}")
                pred = None
            if pred:
                predictions.append(pred)

    if not predictions:
        print("No game predictions produced")
        return

    predictions.sort(key=lambda p: p.get("gameDate") or "")

    output = {
        "updatedAt": today,
        "season": season,
        "games": predictions,
    }

    Path("data").mkdir(exist_ok=True)
    Path("data/history").mkdir(exist_ok=True)

    with open("data/game_predictions.json", "w") as f:
        json.dump(output, f, indent=2)

    with open(f"data/history/games_{today}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(predictions)} game predictions to data/game_predictions.json")


if __name__ == "__main__":
    main()
