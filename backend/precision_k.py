import json
from pathlib import Path
from datetime import datetime, timedelta

# Probability tiers used for bucketed model calibration -- mirrors the
# client-side model's tier-calibration buckets in index.html.
TIER_BUCKETS = [
    (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
    (0.20, 0.25), (0.25, 0.30), (0.30, 0.40), (0.40, 1.01),
]


def _load_actuals(date_str):
    """Load actual HR outcomes from data/results/results_<date>.json.

    Returns dict mapping str(player_id) -> home_runs (int).
    Returns {} when no results file exists yet; precision will be 0 for
    that date but the pipeline won't crash.
    """
    results_file = Path(f"data/results/results_{date_str}.json")
    if not results_file.exists():
        print(f"[WARN] No results file for {date_str} - precision will be 0.")
        return {}
    with open(results_file) as f:
        rows = json.load(f)
    # rows may be a list or a dict with a "results" key
    if isinstance(rows, dict):
        rows = rows.get("results", [])
    return {
        str(r["player_id"]): r.get("home_runs", 0)
        for r in rows
        if "player_id" in r
    }


def precision_at_k(predictions, actuals, k=15):
    if not predictions or k <= 0:
        return 0.0
    top_k = predictions[:k]
    # p["pid"] is the canonical key written by compute_model(); fall back
    # to "id" in case older history files used that field name.
    hits = sum(
        1
        for p in top_k
        if actuals.get(str(p.get("pid", p.get("id", ""))), 0) > 0
    )
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
    actuals = _load_actuals(date_str)   # was always {} before - now reads real results
    p10 = precision_at_k(predictions, actuals, 10)
    p15 = precision_at_k(predictions, actuals, 15)
    p20 = precision_at_k(predictions, actuals, 20)
    return {"date": date_str, "precision@10": p10, "precision@15": p15, "precision@20": p20}


def update_tier_calibration():
    """Walk every tracked slate and bucket predicted-vs-actual HR outcomes by
    probability tier, persisting the result to data/model_weights.json so
    factors.py's tier_calibration_mult() can apply a bounded, evidence-based
    correction instead of running the same static weights forever."""
    history_dir = Path("data/history")
    if not history_dir.exists():
        print("No history directory - skipping tier calibration")
        return

    buckets = [{"lo": lo, "hi": hi, "total": 0, "hits": 0, "sumExpected": 0.0} for lo, hi in TIER_BUCKETS]

    for file in sorted(history_dir.glob("*.json")):
        if file.name.startswith("games_"):
            continue  # game-line history, not HR-prop predictions
        with open(file) as f:
            data = json.load(f)
        predictions = data.get("predictions", [])
        if not predictions:
            continue
        actuals = _load_actuals(file.stem)
        for p in predictions:
            prob = p.get("gameProb", 0)
            hit = actuals.get(str(p.get("pid", p.get("id", ""))), 0) > 0
            for b in buckets:
                if b["lo"] <= prob < b["hi"]:
                    b["total"] += 1
                    if hit:
                        b["hits"] += 1
                    b["sumExpected"] += prob
                    break

    weights_file = Path("data/model_weights.json")
    weights = {}
    if weights_file.exists():
        with open(weights_file) as f:
            weights = json.load(f)
    weights["tierCalibration"] = buckets
    weights["last_updated"] = datetime.now().isoformat()
    with open(weights_file, "w") as f:
        json.dump(weights, f, indent=2)

    total_tracked = sum(b["total"] for b in buckets)
    print(f"Updated tier calibration across {total_tracked} tracked picks")


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
    avg_p10 = sum(r["precision@10"] for r in results) / len(results)
    avg_p15 = sum(r["precision@15"] for r in results) / len(results)
    avg_p20 = sum(r["precision@20"] for r in results) / len(results)
    print(f"Season Summary ({len(results)} slates):")
    print(f"  Average Precision@10: {avg_p10:.3f}")
    print(f"  Average Precision@15: {avg_p15:.3f}")
    print(f"  Average Precision@20: {avg_p20:.3f}")


if __name__ == "__main__":
    yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
    result = evaluate_slate(yesterday)
    if result:
        print(f"Yesterday ({yesterday}):")
        print(f"  Precision@10: {result['precision@10']:.3f}")
        print(f"  Precision@15: {result['precision@15']:.3f}")
        print(f"  Precision@20: {result['precision@20']:.3f}")
    update_tier_calibration()
