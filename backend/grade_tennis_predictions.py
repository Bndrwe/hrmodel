"""Grade yesterday's tennis picks against actual results and close the
self-learning loop -- same pattern as grade_game_predictions.py for MLB.
"""
import json
from datetime import date, timedelta
from pathlib import Path

CONFIDENCE_BUCKETS = [
    (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
    (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.01),
]


def _load(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def grade_day(date_str):
    preds = _load(f"data/history/tennis_{date_str}.json")
    results = _load(f"data/tennis_results/results_{date_str}.json")
    if not preds or not results:
        print(f"Missing tennis predictions or results for {date_str} -- skipping")
        return None

    results_by_id = {r["matchId"]: r for r in results.get("results", [])}
    graded = []
    for m in preds.get("matches", []):
        r = results_by_id.get(m.get("matchId"))
        if not r:
            continue
        prob = max(m["modelProb"]["player1"], m["modelProb"]["player2"])
        hit = m["pick"] == r["winner"]
        graded.append({
            "matchId": m.get("matchId"),
            "matchup": f"{m['player1']['name']} vs {m['player2']['name']}",
            "tour": m.get("tour"),
            "tournament": m.get("tournament"),
            "pick": m["pick"],
            "winner": r["winner"],
            "prob": round(prob, 4),
            "source": m.get("source"),
            "hit": hit,
        })
    return graded


def update_accuracy_log(date_str, graded):
    log_path = Path("data/tennis_accuracy_log.json")
    log = _load(log_path) or {"sessions": []}

    hits = sum(1 for g in graded if g["hit"])
    session = {
        "date": date_str,
        "matchesGraded": len(graded),
        "hits": hits,
        "total": len(graded),
        "detail": graded,
    }
    log["sessions"] = [s for s in log.get("sessions", []) if s.get("date") != date_str]
    log["sessions"].append(session)
    log["sessions"].sort(key=lambda s: s["date"])
    log["sessions"] = log["sessions"][-60:]

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Graded {len(graded)} tennis matches for {date_str}: {hits}/{len(graded)} hit")
    return log


def update_calibration(log):
    buckets = [{"lo": lo, "hi": hi, "total": 0, "hits": 0, "sumExpected": 0.0}
               for lo, hi in CONFIDENCE_BUCKETS]
    for session in log.get("sessions", []):
        for g in session.get("detail", []):
            prob, hit = g.get("prob"), g.get("hit")
            if prob is None or hit is None:
                continue
            for b in buckets:
                if b["lo"] <= prob < b["hi"]:
                    b["total"] += 1
                    if hit:
                        b["hits"] += 1
                    b["sumExpected"] += prob
                    break

    weights_path = Path("data/tennis_model_weights.json")
    weights = _load(weights_path) or {}
    weights["calibrationBuckets"] = buckets
    weights["accuracySummary"] = {
        "hits": sum(b["hits"] for b in buckets),
        "total": sum(b["total"] for b in buckets),
    }
    weights["lastUpdated"] = date.today().isoformat()
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Updated tennis calibration across {sum(b['total'] for b in buckets)} tracked picks")


def main():
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    graded = grade_day(yesterday)
    if not graded:
        return
    log = update_accuracy_log(yesterday, graded)
    update_calibration(log)


if __name__ == "__main__":
    main()
