"""Edge IQ - tennis prediction pipeline (ATP + WTA, all tour-level matches).

Data source: tennisexplorer.com's public match-schedule and ranking pages.
There is no free, keyless, official ATP/WTA API (verified: RapidAPI's tennis
listing requires payment despite advertising a free tier, Jeff Sackmann's
historical GitHub datasets are currently unreachable, and the unofficial
SofaScore API blocks GitHub Actions' IP ranges outright with HTTP 403).
tennisexplorer.com's schedule/ranking pages are plain server-rendered HTML,
reachable from GitHub Actions, and already include bookmaker odds per match --
so instead of hand-rolling a ranking-only win-probability model, the primary
signal is the devigged market-implied probability, blended with a
ranking-points-based fallback for matches odds don't cover yet.

This is inherently a scrape of an unofficial, undocumented page structure: if
tennisexplorer changes their markup, this degrades to an empty matches list
for that day rather than crashing (same "check back soon" pattern already
used for the MLB game-lines pipeline when its data isn't ready).
"""
import json
import math
import re
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from bs4 import BeautifulSoup

BASE = "https://www.tennisexplorer.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
}
TOURS = {"ATP": "atp-men", "WTA": "wta-women"}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _text(td):
    return td.get_text(strip=True).replace("\xa0", "") if td is not None else ""


def _match_time(td):
    """Extract just the HH:MM kickoff time. High-profile TV matches embed
    a hidden livestream-provider tooltip inside this same cell, which a
    plain get_text() would concatenate onto the time (e.g.
    "17:00Livestreams1xBetbet365Unibet...") -- so match the leading
    HH:MM pattern instead of trusting the full cell text."""
    if td is None:
        return None
    m = re.match(r"\s*(\d{1,2}:\d{2})", _text(td))
    return m.group(1) if m else None


def _slug(href):
    return [s for s in (href or "").split("/") if s][-1] if href else None


def _parse_odds(td):
    txt = _text(td)
    try:
        return float(txt)
    except ValueError:
        return None


def _tour_from_href(href):
    slug = _slug(href) or ""
    if "wta" in slug:
        return "WTA"
    if "atp" in slug:
        return "ATP"
    return "Other"


def parse_schedule(html):
    """Parse a tennisexplorer /matches/ page into a flat list of match dicts.

    Rows come in pairs sharing an id ("sN" / "sNb") because the site uses
    rowspan=2 on the columns that don't change per-player (time, odds, match
    link) -- row 1 (player 1) carries those, row 2 (player 2) only carries
    the name/result/score cells.
    """
    soup = BeautifulSoup(html, "lxml")
    matches = []
    current_tournament, current_tour = None, None

    for table in soup.find_all("table", class_="result"):
        pending = {}
        for tr in table.find_all("tr"):
            classes = tr.get("class") or []
            if "head" in classes:
                name_td = tr.find("td", class_="t-name")
                a = name_td.find("a") if name_td else None
                if a:
                    current_tournament = a.get_text(strip=True)
                    current_tour = _tour_from_href(a.get("href"))
                continue

            row_id = tr.get("id")
            if not row_id:
                continue

            if row_id.endswith("b"):
                base_id = row_id[:-1]
                if base_id not in pending:
                    continue
                name_td = tr.find("td", class_="t-name")
                a = name_td.find("a") if name_td else None
                if not a:
                    continue
                pending[base_id]["player2"] = a.get_text(strip=True)
                pending[base_id]["player2Slug"] = _slug(a.get("href"))
                pending[base_id]["player2Scores"] = [
                    _text(td) for td in tr.find_all("td", class_="score")
                ]
            else:
                if current_tour not in ("ATP", "WTA"):
                    continue
                name_td = tr.find("td", class_="t-name")
                a = name_td.find("a") if name_td else None
                if not a:
                    continue
                time_td = tr.find("td", class_="first")
                match_id = None
                for link in tr.find_all("a"):
                    href = link.get("href") or ""
                    if "match-detail" in href:
                        m = re.search(r"id=(\d+)", href)
                        if m:
                            match_id = m.group(1)
                pending[row_id] = {
                    "matchId": match_id,
                    "tournament": current_tournament,
                    "tour": current_tour,
                    "time": _match_time(time_td),
                    "player1": a.get_text(strip=True),
                    "player1Slug": _slug(a.get("href")),
                    "oddsHome": _parse_odds(tr.find("td", class_="coursew")),
                    "oddsAway": _parse_odds(tr.find("td", class_="course")),
                    "player1Scores": [
                        _text(td) for td in tr.find_all("td", class_="score")
                    ],
                }
        matches.extend(m for m in pending.values() if m.get("player2"))
    return matches


def fetch_rankings(tour_path):
    html = fetch(f"{BASE}/ranking/{tour_path}/")
    soup = BeautifulSoup(html, "lxml")
    out = {}
    # The page has more than one <table class="result"> -- the first is
    # the date/name/country search filter form, not the ranking list --
    # so scan every <tr> in the whole document and keep the ones that
    # actually carry a rank cell, rather than trusting table position.
    for tr in soup.find_all("tr"):
        rank_td = tr.find("td", class_="rank")
        name_td = tr.find("td", class_="t-name")
        pts_td = tr.find("td", class_="long-point")
        if not (rank_td and name_td and pts_td):
            continue
        a = name_td.find("a")
        if not a:
            continue
        slug = _slug(a.get("href"))
        if not slug:
            continue
        try:
            rank = int(_text(rank_td).rstrip("."))
        except ValueError:
            continue
        try:
            pts = int(_text(pts_td).replace(",", ""))
        except ValueError:
            pts = None
        out[slug] = {"rank": rank, "points": pts, "name": a.get_text(strip=True)}
    return out


def implied_prob(odds_a, odds_b):
    """Market-implied win probability for player A. Devigged when both
    sides are priced; falls back to the single available price (still a
    real signal, just includes the bookmaker's overround) rather than
    discarding the match entirely -- this matters most for exactly the
    highest-profile matches, where one side's price is occasionally
    missing from the scraped page but the other is a strong signal on
    its own (e.g. a heavy 1.15 favorite in a Slam quarterfinal)."""
    a_ok, b_ok = bool(odds_a) and odds_a > 1, bool(odds_b) and odds_b > 1
    if a_ok and b_ok:
        ra, rb = 1.0 / odds_a, 1.0 / odds_b
        total = ra + rb
        return ra / total if total > 0 else None
    if a_ok:
        return clamp(1.0 / odds_a, 0.05, 0.95)
    if b_ok:
        return clamp(1.0 - 1.0 / odds_b, 0.05, 0.95)
    return None


def ranking_prob(points_a, points_b):
    """Bradley-Terry-style share on log-compressed ranking points -- used
    only as a fallback when no market odds are posted yet."""
    if not points_a or not points_b:
        return 0.5
    diff = math.log(max(points_a, 1)) - math.log(max(points_b, 1))
    p = 1.0 / (1.0 + math.exp(-2.2 * diff))
    return clamp(p, 0.05, 0.95)


def blend_prob(odds_p, rank_p):
    if odds_p is None:
        return clamp(rank_p, 0.05, 0.95), "ranking-only"
    return clamp(0.65 * odds_p + 0.35 * rank_p, 0.03, 0.97), "odds+ranking"


MAJORS = ("wimbledon", "us open", "french open", "roland garros", "australian open")


def best_of_for(tournament, tour):
    """Men's Slams are best-of-5; every other ATP/WTA tour-level match
    (including all WTA, even at Slams) is best-of-3."""
    name = (tournament or "").lower()
    if tour == "ATP" and any(m in name for m in MAJORS):
        return 5
    return 3


def _match_prob_from_set_prob(p, best_of):
    """P(win the match) given a constant per-set win probability p,
    treating sets as i.i.d. -- the standard best-of-N series formula."""
    if best_of == 5:
        return p**3 * (1 + 3 * (1 - p) + 6 * (1 - p) ** 2)
    return p**2 * (3 - 2 * p)


def set_prob_from_match_prob(match_prob, best_of):
    """Invert _match_prob_from_set_prob via bisection (it's monotonic
    increasing in p on [0,1], so this converges cleanly)."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _match_prob_from_set_prob(mid, best_of) < match_prob:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def first_set_prob(match_prob1, tournament, tour):
    """First-set win probability for player 1, derived from the same
    match-win probability the model already computed -- not a separate
    signal, just what that probability implies about a single set once
    you assume sets are i.i.d. Bernoulli trials (the standard simplifying
    assumption in tennis analytics; real matches have some serve-order/
    momentum correlation this ignores, so treat it as a reasonable
    estimate, not a precise one)."""
    bo = best_of_for(tournament, tour)
    if match_prob1 >= 0.5:
        return clamp(set_prob_from_match_prob(match_prob1, bo), 0.05, 0.95)
    return clamp(1 - set_prob_from_match_prob(1 - match_prob1, bo), 0.05, 0.95)


def _load_weights():
    path = Path(__file__).resolve().parent.parent / "data" / "tennis_model_weights.json"
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def calibrate(prob, weights, bucket_key="calibrationBuckets"):
    """Bounded, evidence-gated self-correction -- same shape as the MLB
    moneyline calibration in game_model.py. Only engages once a confidence
    tier has enough tracked picks (>=20) to say anything meaningful.
    bucket_key lets the match-winner and first-set markets each learn
    from their own tracked history rather than sharing one signal."""
    fav_prob = max(prob, 1 - prob)
    for b in weights.get(bucket_key, []):
        lo, hi, total = b.get("lo", 0), b.get("hi", 1), b.get("total", 0)
        if lo <= fav_prob < hi and total >= 20:
            observed = b.get("hits", 0) / total
            expected = b.get("sumExpected", total * (lo + hi) / 2) / total
            drift = observed - expected
            adjustment = clamp(drift * 0.4, -0.08, 0.08)
            fav_prob = clamp(fav_prob + adjustment, 0.51, 0.99)
            break
    return fav_prob if prob >= 0.5 else 1 - fav_prob


def build_predictions(target_date):
    html = fetch(
        f"{BASE}/matches/?type=all&year={target_date.year}"
        f"&month={target_date.month:02d}&day={target_date.day:02d}"
    )
    raw_matches = parse_schedule(html)

    rankings = {}
    for tour, path in TOURS.items():
        try:
            rankings[tour] = fetch_rankings(path)
        except Exception as e:
            print(f"Ranking fetch failed for {tour}: {e}")
            rankings[tour] = {}

    weights = _load_weights()
    out_matches = []
    for m in raw_matches:
        ranks = rankings.get(m["tour"], {})
        r1 = ranks.get(m["player1Slug"], {})
        r2 = ranks.get(m["player2Slug"], {})
        odds_p = implied_prob(m.get("oddsHome"), m.get("oddsAway"))
        rank_p = ranking_prob(r1.get("points"), r2.get("points"))
        prob1, source = blend_prob(odds_p, rank_p)
        prob1 = calibrate(prob1, weights)
        pick = "player1" if prob1 >= 0.5 else "player2"

        fs_prob1 = first_set_prob(prob1, m.get("tournament"), m.get("tour"))
        fs_prob1 = calibrate(fs_prob1, weights, bucket_key="firstSetCalibrationBuckets")
        fs_pick = "player1" if fs_prob1 >= 0.5 else "player2"

        out_matches.append({
            "matchId": m.get("matchId"),
            "tournament": m.get("tournament"),
            "tour": m.get("tour"),
            "time": m.get("time"),
            "player1": {"name": m["player1"], "slug": m["player1Slug"],
                        "rank": r1.get("rank"), "points": r1.get("points")},
            "player2": {"name": m["player2"], "slug": m["player2Slug"],
                        "rank": r2.get("rank"), "points": r2.get("points")},
            "odds": {"player1": m.get("oddsHome"), "player2": m.get("oddsAway")},
            "modelProb": {"player1": round(prob1, 4), "player2": round(1 - prob1, 4)},
            "pick": pick,
            "source": source,
            "firstSet": {
                "player1": round(fs_prob1, 4),
                "player2": round(1 - fs_prob1, 4),
                "pick": fs_pick,
            },
        })
    return out_matches


def main():
    today = date.today()
    try:
        matches = build_predictions(today)
    except Exception as e:
        print(f"Tennis schedule fetch failed: {type(e).__name__}: {e}")
        matches = []

    out = {"updatedAt": today.isoformat(), "matches": matches}

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "tennis_predictions.json", "w") as f:
        json.dump(out, f, indent=2)

    hist_dir = data_dir / "history"
    hist_dir.mkdir(exist_ok=True, parents=True)
    with open(hist_dir / f"tennis_{today.isoformat()}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(matches)} ATP/WTA matches for {today.isoformat()}")


if __name__ == "__main__":
    main()
