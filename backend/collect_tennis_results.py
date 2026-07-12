"""Collect actual tennis match outcomes for a past day.

tennisexplorer.com's schedule page is stateful by date -- refetching the same
URL for a day that has already happened returns the same rows but with the
score cells populated. Winner is determined by comparing set-by-set score
cells rather than trusting the ambiguous "result" marker cell, since the
marker's exact win/loss encoding hasn't been observed against a real
completed match yet (this sandbox can't reach the site directly to check;
only GitHub Actions can). If that assumption turns out wrong, matches simply
fail to grade instead of grading incorrectly -- same fail-closed posture as
the rest of this pipeline.
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tennis_model import BASE, fetch, parse_schedule  # noqa: E402


def determine_winner(m):
    p1s, p2s = m.get("player1Scores", []), m.get("player2Scores", [])
    sets1 = sets2 = 0
    for a, b in zip(p1s, p2s):
        try:
            av, bv = int(a), int(b)
        except (ValueError, TypeError):
            continue
        if av > bv:
            sets1 += 1
        elif bv > av:
            sets2 += 1
    if sets1 == sets2:
        return None
    return "player1" if sets1 > sets2 else "player2"


def determine_first_set_winner(m):
    """Whoever won more games in the first score column won set 1. Ties
    (game counts equal, e.g. a retirement before the set finished) can't
    be resolved from game counts alone, so those are left ungraded rather
    than guessed."""
    p1s, p2s = m.get("player1Scores", []), m.get("player2Scores", [])
    if not p1s or not p2s:
        return None
    try:
        a, b = int(p1s[0]), int(p2s[0])
    except (ValueError, TypeError):
        return None
    if a == b:
        return None
    return "player1" if a > b else "player2"


def main():
    yesterday = date.today() - timedelta(days=1)
    try:
        html = fetch(
            f"{BASE}/matches/?type=all&year={yesterday.year}"
            f"&month={yesterday.month:02d}&day={yesterday.day:02d}"
        )
        matches = parse_schedule(html)
    except Exception as e:
        print(f"Result collection failed: {type(e).__name__}: {e}")
        matches = []

    results = []
    for m in matches:
        if not m.get("matchId"):
            continue
        winner = determine_winner(m)
        if winner is None:
            continue
        results.append({
            "matchId": m["matchId"],
            "winner": winner,
            "firstSetWinner": determine_first_set_winner(m),
            "player1Scores": m.get("player1Scores", []),
            "player2Scores": m.get("player2Scores", []),
        })

    out_dir = Path("data/tennis_results")
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(out_dir / f"results_{yesterday.isoformat()}.json", "w") as f:
        json.dump({"date": yesterday.isoformat(), "results": results}, f, indent=2)

    print(f"Collected {len(results)} completed tennis results for {yesterday.isoformat()}")


if __name__ == "__main__":
    main()
