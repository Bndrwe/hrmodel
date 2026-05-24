"""Collect actual game results for model training and evaluation."""
import requests
import json
from datetime import datetime, date, timedelta
from pathlib import Path

# MLB Stats API
MLB_API = "https://statsapi.mlb.com/api/v1"

def fetch_json(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Fetch error {url}: {e}")
        return None

def collect_yesterdays_results():
    """Collect HR results from yesterday's games."""
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    schedule_url = f"{MLB_API}/schedule?sportId=1&date={yesterday}"
    schedule = fetch_json(schedule_url)
    
    if not schedule or 'dates' not in schedule:
        print(f"No schedule data for {yesterday}")
    results = [][]
        
    print(f"Collecting results for {yesterday}...")
    
    for dt in schedule.get('dates', []):
        for game in dt.get('games', []):
            game_pk = game['gamePk']
            if game.get('status', {}).get('detailedState') != 'Final':
                continue
            
            boxscore_url = f"{MLB_API}/game/{game_pk}/boxscore"
            boxscore = fetch_json(boxscore_url)
            
            if not boxscore:
                continue
            
            for side in ['away', 'home']:
                batters = boxscore.get('teams', {}).get(side, {}).get('batters', [])
                players_data = boxscore.get('teams', {}).get(side, {}).get('players', {})
                
                for batter_id in batters:
                    player_key = f"ID{batter_id}"
                    if player_key not in players_data:
                        continue
                    
                    player = players_data[player_key]
                    name = player.get('person', {}).get('fullName', '')
                    stats = player.get('stats', {}).get('batting', {})
                    
                    home_runs = stats.get('homeRuns', 0)
                    
                    results.append({
                        'date': yesterday,
                        'player_id': batter_id,
                        'player_name': name,
                        'home_runs': home_runs,
                        'hits': stats.get('hits', 0),
                        'at_bats': stats.get('atBats', 0),
                        'game_pk': game_pk
                    })
    
    return results

def save_results(results):
    """Save results to JSON files."""
    if not results:
        print("No results to save")
        return
    
    # Create results directory
    results_dir = Path('data/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    filename = results_dir / f"results_{yesterday}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'date': yesterday,
            'collected_at': datetime.now().isoformat(),
            'total_players': len(results),
            'total_hrs': sum(r['home_runs'] for r in results),
            'results': results
        }, f, indent=2)
    
    print(f"Saved {len(results)} player results to {filename}")

def main():
    print("Starting result collection...")
    results = collect_yesterdays_results()
    save_results(results)
    print("Result collection complete!")

if __name__ == "__main__":
    main()