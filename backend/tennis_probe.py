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


if __name__ == "__main__":
    main()
