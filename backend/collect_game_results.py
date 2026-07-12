"""Collect actual final scores for yesterday's games.

This is the data grade_game_predictions.py needs to check backend/
game_model.py's moneyline/run-line/total/F5/NRFI/team-hits predictions
against what actually happened -- final score, hits, first-inning runs,
and first-5-innings runs, straight from the MLB Stats API's linescore.
"""
import json
from datetime import date, timedelta
from pathlib import Path

from model import fetch_json, _safe_int, MLB_API


def fetch_game_result(game_pk):
    data = fetch_json(f"{MLB_API}/game/{game_pk}/linescore")
    if not data:
        return None
    innings = data.get("innings", [])
    if not innings:
        return None

    teams = data.get("teams", {})
    home_runs = _safe_int(teams.get("home", {}), "runs")
    away_runs = _safe_int(teams.get("away", {}), "runs")
    home_hits = _safe_int(teams.get("home", {}), "hits")
    away_hits = _safe_int(teams.get("away", {}), "hits")

    first = innings[0]
    first_inning_home = _safe_int(first.get("home", {}), "runs")
    first_inning_away = _safe_int(first.get("away", {}), "runs")

    f5_innings = innings[:5]
    f5_home = sum(_safe_int(i.get("home", {}), "runs") for i in f5_innings)
    f5_away = sum(_safe_int(i.get("away", {}), "runs") for i in f5_innings)

    return {
        "homeRuns": home_runs, "awayRuns": away_runs,
        "homeHits": home_hits, "awayHits": away_hits,
        "firstInningRuns": {"home": first_inning_home, "away": first_inning_away},
        "f5Runs": {"home": f5_home, "away": f5_away},
        "f5Complete": len(innings) >= 5,
    }


def main():
    print("Collecting game-line results...")
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    schedule = fetch_json(f"{MLB_API}/schedule?sportId=1&date={yesterday}")
    if not schedule or not schedule.get("dates"):
        print(f"No schedule for {yesterday}")
        return

    results = []
    for dt in schedule["dates"]:
        for game in dt.get("games", []):
            if game.get("status", {}).get("detailedState") != "Final":
                continue
            game_pk = game.get("gamePk")
            if not game_pk:
                continue
            result = fetch_game_result(game_pk)
            if result:
                result["gamePk"] = game_pk
                results.append(result)

    if not results:
        print("No completed games found")
        return

    Path("data/game_results").mkdir(parents=True, exist_ok=True)
    output = {"date": yesterday, "games": results}
    with open(f"data/game_results/results_{yesterday}.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(results)} game results to data/game_results/results_{yesterday}.json")


if __name__ == "__main__":
    main()
