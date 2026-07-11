"""HR Oracle - game-level prediction engine.

Extends the HR-prop model with whole-game MLB predictions: moneylines,
run-line/alternate spreads, and totals (over/under). These are the
model's own fair-value lines (no sportsbook odds feed is configured for
this repo), derived from real team run environments and real probable
starter quality:

  1. Each team's runs-scored and runs-allowed per game (season-to-date).
  2. Today's probable starters' ERA, regressed toward league average by
     sample size (battersFaced), applied against the *opposing* team's
     runs-scored expectation.
  3. The home park's run-scoring factor and a standard home-field bump.
  4. A single-game Pythagorean-win-expectation formula (exponent 1.83)
     applied to the two teams' adjusted expected runs for this specific
     game, which gives the moneyline probability directly.
  5. A Normal approximation over the resulting run differential / total
     (with empirically-typical MLB single-game standard deviations) for
     run-line, alternate-spread, and total probabilities.
"""
import json
import math
from datetime import date
from pathlib import Path

from model import fetch_json, _safe_float, _safe_int, PARK_FACTORS, MLB_API
from factors import clamp

LG_ERA = 4.15
LG_RUNS_PER_GAME = 4.50
PYTHAG_EXP = 1.83
# Empirically-typical standard deviations for single-game MLB run
# differential / total runs; used as a Normal approximation since we don't
# have per-game Monte Carlo simulation for team scoring (unlike the HR
# model, which does run one client-side).
STD_MARGIN = 4.4
STD_TOTAL = 4.6
HOME_FIELD_BUMP = 0.02  # ~2% scoring bump for the home team


def norm_cdf(x, mean, std):
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))


def prob_to_american_odds(p):
    p = clamp(p, 0.01, 0.99)
    if p >= 0.5:
        return -round(100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def get_team_run_environment(team_id, season):
    """Real season-to-date runs-scored/game and runs-allowed/game for a team,
    falling back to league average on any missing/failed data."""
    rs_per_game = LG_RUNS_PER_GAME
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
            if games_played > 0:
                rs_per_game = runs / games_played

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

    return rs_per_game, ra_per_game


def fetch_starter_quality(pitcher_id, season):
    """Regressed ERA for a probable starter (league-average when unknown or
    on a tiny sample)."""
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
    bf = _safe_int(raw, "battersFaced")
    prw = clamp(bf / 300, 0.10, 0.85)
    return prw * era + (1 - prw) * LG_ERA


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

    home_rs, home_ra = get_team_run_environment(home_id, season)
    away_rs, away_ra = get_team_run_environment(away_id, season)

    home_starter_era = fetch_starter_quality(home_pitcher.get("id"), season)
    away_starter_era = fetch_starter_quality(away_pitcher.get("id"), season)

    park = PARK_FACTORS.get(home_id, {})
    # The power (HR) park factor is reused as an approximate run-scoring
    # factor -- hitter-friendly power parks are, in practice, usually also
    # higher-scoring parks, but home runs move far more than total runs do,
    # so the effect is dampened to half strength (e.g. Coors' +38% HR factor
    # becomes a more realistic +19% run-scoring bump) rather than applied
    # 1:1, since maintaining an entirely separate verified run-factor table
    # isn't practical here.
    hr_park_factor = park.get("factor", 1.0)
    park_run_factor = 1.0 + (hr_park_factor - 1.0) * 0.5

    # Each team's expected runs today = their own scoring rate, adjusted by
    # how the OPPOSING starter compares to league average, times park,
    # times a small home-field scoring bump.
    exp_runs_home = home_rs * (away_starter_era / LG_ERA) * park_run_factor * (1 + HOME_FIELD_BUMP)
    exp_runs_away = away_rs * (home_starter_era / LG_ERA) * park_run_factor * (1 - HOME_FIELD_BUMP)
    exp_runs_home = clamp(exp_runs_home, 1.5, 10.0)
    exp_runs_away = clamp(exp_runs_away, 1.5, 10.0)

    p_home = (exp_runs_home ** PYTHAG_EXP) / (exp_runs_home ** PYTHAG_EXP + exp_runs_away ** PYTHAG_EXP)
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
    for game in games:
        try:
            pred = build_game_prediction(game, season)
        except Exception as e:
            print(f"[WARN] Failed to build prediction for game {game.get('gamePk')}: {e}")
            pred = None
        if pred:
            predictions.append(pred)

    if not predictions:
        print("No game predictions produced")
        return

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
