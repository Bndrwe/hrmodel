import requests
import json
from datetime import datetime, date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from factors import compute_model

MLB_API = "https://statsapi.mlb.com/api/v1"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

def fetch_json(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Fetch error {url}: {e}")
        return None

def main():
    print("Starting HR Oracle model run...")
    today = date.today().isoformat()
    
    schedule_url = f"{MLB_API}/schedule?sportId=1&date={today}"
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
        game_pk = game.get("gamePk")
        if not game_pk:
            continue
        
        boxscore_url = f"{MLB_API}/game/{game_pk}/boxscore"
        box = fetch_json(boxscore_url)
        if not box:
            continue
        
        teams = box.get("teams", {})
        for side in ["away", "home"]:
            team_data = teams.get(side, {})
            batters = team_data.get("batters", [])
            
            for batter_id in batters:
                batter_stats = team_data.get("players", {}).get(f"ID{batter_id}", {})
                if not batter_stats:
                    continue
                
                person = batter_stats.get("person", {})
                season_stats = batter_stats.get("seasonStats", {}).get("batting", {})
                
                result = compute_model(
                    batter=person,
                    batter_stat=season_stats,
                    batter_l7=None,
                    batter_l14=None,
                    vs_hand=None,
                    h2h=None,
                    pitcher_stat=None,
                    pitcher_l3=None,
                    game={"team_abbr": team_data.get("team", {}).get("abbreviation", ""),
                          "opp_abbr": "",
                          "isAway": side == "away",
                          "pitThrows": "R",
                          "order": 5,
                          "dayNight": "night",
                          "pitcher": ""},
                    weather=None,
                    park_factor=1.0,
                    season_day=50
                )
                
                if result:
                    predictions.append(result)
    
    predictions.sort(key=lambda x: x["gameProb"], reverse=True)
    
    output = {
        "updatedAt": today,
        "season": 2026,
        "predictions": predictions
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
