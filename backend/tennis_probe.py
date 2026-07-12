"""One-off diagnostic: confirm the real shape of SofaScore's unofficial tennis
endpoints before building the real prediction pipeline on top of them. This
sandbox can't reach third-party APIs directly, so this runs inside GitHub
Actions (which has normal internet access) and dumps what it actually gets
back to the job log for inspection. Not part of the regular daily pipeline --
triggered manually, once, then deleted.
"""
import json
import urllib.request
from datetime import date

BASE = "https://api.sofascore.com/api/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}


def fetch(path):
    req = urllib.request.Request(BASE + path, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def main():
    today = date.today().isoformat()
    print(f"=== scheduled-events for {today} ===")
    try:
        data = fetch(f"/sport/tennis/scheduled-events/{today}")
        events = data.get("events", [])
        print(f"event count: {len(events)}")
        if events:
            e = events[0]
            print("--- sample event (full) ---")
            print(json.dumps(e, indent=2)[:4000])
            print("--- all top-level keys across first 20 events ---")
            keys = set()
            for ev in events[:20]:
                keys |= set(ev.keys())
            print(sorted(keys))
            print("--- tournament categories seen ---")
            cats = set()
            for ev in events:
                t = ev.get("tournament", {})
                cat = t.get("category", {})
                cats.add((cat.get("name"), t.get("name"), t.get("uniqueTournament", {}).get("name")))
            for c in sorted(cats, key=lambda x: str(x)):
                print(c)
            print("--- homeTeam/awayTeam sample keys ---")
            print(sorted(e.get("homeTeam", {}).keys()))
            print(json.dumps(e.get("homeTeam", {}), indent=2))
    except Exception as ex:
        print(f"scheduled-events FAILED: {type(ex).__name__}: {ex}")
        events = []

    if events:
        pid = events[0].get("homeTeam", {}).get("id")
        eid = events[0].get("id")

        print(f"\n=== /team/{pid} (player detail) ===")
        try:
            d = fetch(f"/team/{pid}")
            print(json.dumps(d, indent=2)[:3000])
        except Exception as ex:
            print(f"/team/{{id}} FAILED: {type(ex).__name__}: {ex}")

        print(f"\n=== /team/{pid}/events/last/0 (recent form) ===")
        try:
            d = fetch(f"/team/{pid}/events/last/0")
            print(json.dumps(d, indent=2)[:2000])
        except Exception as ex:
            print(f"/team/{{id}}/events/last/0 FAILED: {type(ex).__name__}: {ex}")

        print(f"\n=== /event/{eid}/h2h (head to head) ===")
        try:
            d = fetch(f"/event/{eid}/h2h")
            print(json.dumps(d, indent=2)[:2000])
        except Exception as ex:
            print(f"/event/{{id}}/h2h FAILED: {type(ex).__name__}: {ex}")

        print(f"\n=== /rankings/type/1 (rankings, guess) ===")
        for rtype in ["1", "2"]:
            try:
                d = fetch(f"/rankings/type/{rtype}")
                print(f"type {rtype} OK:")
                print(json.dumps(d, indent=2)[:1500])
            except Exception as ex:
                print(f"/rankings/type/{rtype} FAILED: {type(ex).__name__}: {ex}")


def probe_alternates():
    candidates = [
        ("tennis-data.co.uk 2026 xlsx", "https://www.tennis-data.co.uk/2026/2026.xlsx"),
        ("tennis-data.co.uk alldata page", "http://www.tennis-data.co.uk/alldata.php"),
        ("api-tennis.com root", "https://api.api-tennis.com/tennis/"),
        ("ultimatetennisstatistics ping", "https://www.ultimatetennisstatistics.com/api/ranking/ATP?date=2026-07-06"),
        ("atptour.com rankings ajax", "https://www.atptour.com/en/-/www/rankings/singles?rankRange=1-20"),
        ("flashscore tennis", "https://www.flashscore.com/x/feed/proxy-tennis"),
        ("tennisexplorer schedule", "https://www.tennisexplorer.com/matches/?type=all"),
    ]
    for label, url in candidates:
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read()
                print(f"{label} -> HTTP {resp.status}, {len(body)} bytes")
                print(body[:300])
        except Exception as ex:
            print(f"{label} -> FAILED: {type(ex).__name__}: {ex}")
        print()


def dump_page(label, url, markers=("<table", "result", "ranking-table"), window=3000):
    req = urllib.request.Request(url, headers=HEADERS)
    print(f"\n=== STRUCTURE DUMP: {label} ({url}) ===")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"HTTP {resp.status}, {len(body)} chars total")
            seen = set()
            for m in markers:
                idx = body.find(m)
                if idx == -1:
                    print(f"-- marker {m!r} not found --")
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                print(f"-- window around first {m!r} at offset {idx} --")
                print(body[max(0, idx - 200):idx + window])
                print("-- end window --\n")
    except Exception as ex:
        print(f"FAILED: {type(ex).__name__}: {ex}")


def probe_tennisexplorer_deep():
    dump_page("today's matches (all levels)", "https://www.tennisexplorer.com/matches/?type=all",
              markers=("<table", "class=\"result", "id=\"quick"))
    dump_page("ATP ranking", "https://www.tennisexplorer.com/ranking/atp-men/",
              markers=("<table", "flag", "class=\"t-name"))


if __name__ == "__main__":
    main()
    print("\n\n=== ALTERNATE SOURCE REACHABILITY ===")
    probe_alternates()
    print("\n\n=== TENNISEXPLORER DEEP DUMP ===")
    probe_tennisexplorer_deep()
