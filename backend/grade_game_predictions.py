"""Grade yesterday's game-line predictions against what actually happened.

Every market backend/game_model.py predicts (moneyline, run line, total,
F5, NRFI/YRFI, team hits) gets checked hit-or-miss against the real final
score collected by collect_game_results.py. Results are appended to
data/game_accuracy_log.json (last 60 tracked days, with full per-game
detail for transparency), and the moneyline picks are bucketed by
predicted-confidence into data/game_model_weights.json so game_model.py
can apply a bounded, evidence-based self-correction -- the actual
"learn and fix itself" step, not just a log nobody reads.
"""
import json
from datetime import date, timedelta
from pathlib import Path

# Confidence tiers for the moneyline favorite's predicted win probability.
MONEYLINE_BUCKETS = [
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
    preds = _load(f"data/history/games_{date_str}.json")
    results = _load(f"data/game_results/results_{date_str}.json")
    if not preds or not results:
        print(f"Missing predictions or results for {date_str} -- skipping grading")
        return None

    results_by_pk = {r["gamePk"]: r for r in results.get("games", [])}
    graded = []

    for g in preds.get("games", []):
        r = results_by_pk.get(g.get("gamePk"))
        if not r:
            continue

        home_runs, away_runs = r["homeRuns"], r["awayRuns"]
        actual_margin = home_runs - away_runs
        actual_total = home_runs + away_runs

        ml = g.get("moneyline", {}) or {}
        ml_pick = "home" if ml.get("homeProb", 0.5) >= ml.get("awayProb", 0.5) else "away"
        ml_prob = max(ml.get("homeProb", 0.5), ml.get("awayProb", 0.5))
        if home_runs == away_runs:
            ml_hit = None  # shouldn't happen for a completed MLB game, but be safe
        else:
            ml_winner = "home" if home_runs > away_runs else "away"
            ml_hit = (ml_pick == ml_winner)

        rl = (g.get("runLine") or {}).get("standard", {}) or {}
        rl_line = rl.get("homeLine", -1.5)
        rl_pick = "home" if rl.get("homeCoverProb", 0.5) >= 0.5 else "away"
        rl_hit = (actual_margin > -rl_line) if rl_pick == "home" else (actual_margin < -rl_line)

        tot = g.get("total", {}) or {}
        model_total = tot.get("modelTotal")
        tot_pick, tot_hit = None, None
        if model_total is not None:
            over_line = next((l for l in tot.get("lines", []) if l.get("line") == model_total), None)
            if over_line:
                tot_pick = "over" if over_line.get("overProb", 0.5) >= 0.5 else "under"
                tot_hit = (actual_total > model_total) if tot_pick == "over" else (actual_total < model_total)

        f5 = g.get("f5", {}) or {}
        f5_runs = r.get("f5Runs")
        f5_pick, f5_hit = None, None
        if f5_runs and r.get("f5Complete", True):
            f5_ml = f5.get("moneyline", {}) or {}
            f5_pick = "home" if f5_ml.get("homeProb", 0.5) >= f5_ml.get("awayProb", 0.5) else "away"
            f5_margin = f5_runs["home"] - f5_runs["away"]
            if f5_margin == 0:
                f5_hit = None  # F5 can genuinely push
            elif f5_pick == "home":
                f5_hit = f5_margin > 0
            else:
                f5_hit = f5_margin < 0

        nrfi = g.get("nrfi", {}) or {}
        fi = r.get("firstInningRuns", {}) or {}
        nrfi_actual = (fi.get("home", 0) == 0 and fi.get("away", 0) == 0)
        nrfi_pick = "nrfi" if nrfi.get("nrfiProb", 0.5) >= 0.5 else "yrfi"
        nrfi_hit = (nrfi_pick == "nrfi") == nrfi_actual

        hits = g.get("teamHits", {}) or {}
        hits_graded = {}
        for side, actual_hits in (("home", r.get("homeHits")), ("away", r.get("awayHits"))):
            h = hits.get(side)
            if h and actual_hits is not None and h.get("modelLine") is not None:
                pick = "over" if h.get("overProb", 0.5) >= 0.5 else "under"
                hit = (actual_hits > h["modelLine"]) if pick == "over" else (actual_hits < h["modelLine"])
                hits_graded[side] = {"pick": pick, "line": h["modelLine"], "actual": actual_hits, "hit": hit}

        graded.append({
            "gamePk": g.get("gamePk"),
            "matchup": f"{g.get('awayTeam')} @ {g.get('homeTeam')}",
            "actualScore": f"{away_runs}-{home_runs}",
            "moneyline": {"pick": ml_pick, "prob": round(ml_prob, 4), "hit": ml_hit},
            "runLine": {"pick": rl_pick, "line": rl_line, "hit": rl_hit},
            "total": {"pick": tot_pick, "line": model_total, "hit": tot_hit},
            "f5": {"pick": f5_pick, "hit": f5_hit},
            "nrfi": {"pick": nrfi_pick, "hit": nrfi_hit},
            "teamHits": hits_graded,
        })

    return graded


def _tally(graded, key):
    hits = sum(1 for g in graded if g[key].get("hit") is True)
    total = sum(1 for g in graded if g[key].get("hit") is not None)
    return hits, total


def update_accuracy_log(date_str, graded):
    log_path = Path("data/game_accuracy_log.json")
    log = _load(log_path) or {"sessions": []}

    ml_hits, ml_total = _tally(graded, "moneyline")
    rl_hits, rl_total = _tally(graded, "runLine")
    tot_hits, tot_total = _tally(graded, "total")
    f5_hits, f5_total = _tally(graded, "f5")
    nrfi_hits, nrfi_total = _tally(graded, "nrfi")
    hits_hits = sum(1 for g in graded for v in g["teamHits"].values() if v.get("hit"))
    hits_total = sum(1 for g in graded for v in g["teamHits"].values())

    session = {
        "date": date_str,
        "gamesGraded": len(graded),
        "moneyline": {"hits": ml_hits, "total": ml_total},
        "runLine": {"hits": rl_hits, "total": rl_total},
        "total": {"hits": tot_hits, "total": tot_total},
        "f5": {"hits": f5_hits, "total": f5_total},
        "nrfi": {"hits": nrfi_hits, "total": nrfi_total},
        "teamHits": {"hits": hits_hits, "total": hits_total},
        "detail": graded,
    }
    log["sessions"].append(session)
    log["sessions"] = log["sessions"][-60:]  # keep last 60 tracked days

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(
        f"Graded {len(graded)} games for {date_str}: "
        f"ML {ml_hits}/{ml_total}, RL {rl_hits}/{rl_total}, "
        f"Total {tot_hits}/{tot_total}, F5 {f5_hits}/{f5_total}, "
        f"NRFI {nrfi_hits}/{nrfi_total}, Hits {hits_hits}/{hits_total}"
    )
    return log


def update_moneyline_calibration(log):
    """Bucket every tracked moneyline pick by its predicted confidence and
    persist observed-vs-expected win rates. This is what closes the loop --
    game_model.py reads this file back and nudges its win probability
    toward reality when a confidence tier has a real, sustained bias."""
    buckets = [{"lo": lo, "hi": hi, "total": 0, "hits": 0, "sumExpected": 0.0} for lo, hi in MONEYLINE_BUCKETS]
    for session in log.get("sessions", []):
        for g in session.get("detail", []):
            ml = g.get("moneyline", {})
            prob, hit = ml.get("prob"), ml.get("hit")
            if prob is None or hit is None:
                continue
            for b in buckets:
                if b["lo"] <= prob < b["hi"]:
                    b["total"] += 1
                    if hit:
                        b["hits"] += 1
                    b["sumExpected"] += prob
                    break

    # Simple aggregate bias for totals/NRFI (these are roughly symmetric
    # around a coin flip by construction, so a single global observed-rate
    # is meaningful without needing separate confidence buckets).
    tot_hits = sum(s["total"]["hits"] for s in log.get("sessions", []))
    tot_total = sum(s["total"]["total"] for s in log.get("sessions", []))
    nrfi_hits = sum(s["nrfi"]["hits"] for s in log.get("sessions", []))
    nrfi_total = sum(s["nrfi"]["total"] for s in log.get("sessions", []))
    rl_hits = sum(s["runLine"]["hits"] for s in log.get("sessions", []))
    rl_total = sum(s["runLine"]["total"] for s in log.get("sessions", []))
    f5_hits = sum(s["f5"]["hits"] for s in log.get("sessions", []))
    f5_total = sum(s["f5"]["total"] for s in log.get("sessions", []))
    hits_hits = sum(s["teamHits"]["hits"] for s in log.get("sessions", []))
    hits_total = sum(s["teamHits"]["total"] for s in log.get("sessions", []))

    weights_path = Path("data/game_model_weights.json")
    weights = _load(weights_path) or {}
    weights["moneylineCalibration"] = buckets
    weights["accuracySummary"] = {
        "moneyline": {"hits": sum(b["hits"] for b in buckets), "total": sum(b["total"] for b in buckets)},
        "runLine": {"hits": rl_hits, "total": rl_total},
        "total": {"hits": tot_hits, "total": tot_total},
        "f5": {"hits": f5_hits, "total": f5_total},
        "nrfi": {"hits": nrfi_hits, "total": nrfi_total},
        "teamHits": {"hits": hits_hits, "total": hits_total},
    }
    weights["lastUpdated"] = date.today().isoformat()
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Updated moneyline calibration across {sum(b['total'] for b in buckets)} tracked picks")


def main():
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    graded = grade_day(yesterday)
    if not graded:
        return
    log = update_accuracy_log(yesterday, graded)
    update_moneyline_calibration(log)


if __name__ == "__main__":
    main()
