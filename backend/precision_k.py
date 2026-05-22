import json
from pathlib import Path
from datetime import datetime, timedelta

def precision_at_k(predictions, actuals, k=15):
    if not predictions or k <= 0:
        return 0.0
    top_k = predictions[:k]
    hits = sum(actuals.get(p["pid"], 0) for p in top_k)
    return hits / k

def evaluate_slate(date_str):
    hist_file = Path(f"data/history/{date_str}.json")
    if not hist_file.exists():
        print(f"No history file for {date_str}")
        return None
    with open(hist_file) as f:
        data = json.load(f)
    predictions = data.get("predictions", [])
    if not predictions:
        return None
    actuals = {}
    p15 = precision_at_k(predictions, actuals, 15)
    p20 = precision_at_k(predictions, actuals, 20)
    return {"date": date_str, "precision@15": p15, "precision@20": p20}

def season_summary():
    history_dir = Path("data/history")
    if not history_dir.exists():
        print("No history directory")
        return
    results = []
    for file in sorted(history_dir.glob("*.json")):
        date_str = file.stem
        result = evaluate_slate(date_str)
        if result:
            results.append(result)
    if not results:
        print("No evaluation results")
        return
    avg_p15 = sum(r["precision@15"] for r in results) / len(results)
    avg_p20 = sum(r["precision@20"] for r in results) / len(results)
    print(f"Season Summary ({len(results)} slates):")
    print(f"  Average Precision@15: {avg_p15:.3f}")
    print(f"  Average Precision@20: {avg_p20:.3f}")

if __name__ == "__main__":
    yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
    result = evaluate_slate(yesterday)
    if result:
        print(f"Yesterday ({yesterday}):")
        print(f"  Precision@15: {result['precision@15']:.3f}")
        print(f"  Precision@20: {result['precision@20']:.3f}")
