import numpy as np
import pandas as pd
import json
from pathlib import Path

LG = {
    "avg": 0.244, "obp": 0.314, "slg": 0.408, "ops": 0.722,
    "iso": 0.155, "kPct": 0.228, "bbPct": 0.085, "hrPA": 0.0307,
    "fbPct": 0.38, "pullPct": 0.38, "linePct": 0.21,
    "pitHR9": 1.28, "pitKPct": 0.228, "pitBBPct": 0.085, "pitFBPct": 0.38
}

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def est_barrel(iso):
    return clamp((iso / LG["iso"]) * 0.078, 0.04, 0.16)

def est_pull_rate(hr, ab):
    hr_rate = hr / max(ab, 1)
    return clamp(0.35 + 0.10 * (hr_rate / LG["hrPA"]), 0.25, 0.50)

def wind_effect(wind_speed, wind_dir, cf_bearing):
    if wind_speed < 2:
        return 1.0
    wind_to = (wind_dir + 180) % 360
    diff = abs(wind_to - cf_bearing)
    if diff > 180:
        diff = 360 - diff
    component = np.cos(np.radians(diff))
    return clamp(1 + component * wind_speed * 0.0048, 0.78, 1.22)

def lineup_pa(order):
    return [5.2, 5.1, 5.0, 4.9, 4.7, 4.5, 4.2, 3.9, 3.7][order - 1] if 1 <= order <= 9 else 4.5

def count_advantage(k_pct, bb_pct, pit_k_pct, pit_bb_pct):
    batter_disc = bb_pct - k_pct
    pitcher_disc = pit_bb_pct - pit_k_pct
    return clamp(1 + 0.08 * (batter_disc - pitcher_disc), 0.92, 1.12)

def compute_model(batter, batter_stat, batter_l7, batter_l14, vs_hand, h2h,
                  pitcher_stat, pitcher_l3, game, weather, park_factor, season_day):
    pid = batter.get("id")
    name = f"{batter.get('first', '')} {batter.get('last', '')}".strip()
    team = game.get("team_abbr", "")
    opp_team = game.get("opp_abbr", "")
    game_matchup = f"{team} @ {opp_team}" if game.get("isAway") else f"{team} vs {opp_team}"
    bats = batter.get("bats", "R")
    pit_throws = game.get("pitThrows", "R")
    platoon = (bats == "L" and pit_throws == "R") or (bats == "R" and pit_throws == "L")

    pa = batter_stat.get("pa", 0) if batter_stat else 0
    if pa < 40:
        return None

    ab = batter_stat.get("ab", 0) if batter_stat else 0
    hr = batter_stat.get("hr", 0) if batter_stat else 0
    bb = batter_stat.get("bb", 0) if batter_stat else 0
    k = batter_stat.get("k", 0) if batter_stat else 0
    avg = batter_stat.get("avg", LG["avg"]) if batter_stat else LG["avg"]
    slg = batter_stat.get("slg", LG["slg"]) if batter_stat else LG["slg"]
    iso = batter_stat.get("iso", LG["iso"]) if batter_stat else LG["iso"]
    k_pct = k / pa if pa > 0 else LG["kPct"]
    bb_pct = bb / pa if pa > 0 else LG["bbPct"]

    # Flatten pitcher skill to league average
    pitcher_hr_mult = 1.0
    pitcher_k_mult = 1.0
    pitcher_fb_pct = LG["pitFBPct"]
    pit_k_pct = LG["pitKPct"]
    pit_bb_pct = LG["pitBBPct"]

    # Base rate
    base_rate = LG["hrPA"] * 1.0

    # Factor 1: iso mult
    iso_mult = clamp(iso / LG["iso"], 0.45, 2.10)

    # Factor 2: platoon
    platoon_mult = 1.15 if platoon else 0.96

    # Factor 3: park
    park_mult = park_factor if park_factor else 1.0

    # Factor 4: pitcher HR (flattened)
    # pitcher_hr_mult = 1.0 already

    # Factor 5: pitcher K (flattened)
    # pitcher_k_mult = 1.0 already

    # Factor 6: weather wind
    cf_bearing = park.get("cfBearing", 0) if isinstance(park_factor, dict) else 0
    wind_speed = weather.get("windSpeed", 0) if weather else 0
    wind_dir = weather.get("windDir", 0) if weather else 0
    wind_mult = wind_effect(wind_speed, wind_dir, cf_bearing)

    # Factor 7: temperature
    temp = weather.get("temp", 70) if weather else 70
    temp_mult = clamp(0.93 + 0.01 * (temp - 70) / 10, 0.88, 1.14)

    # Factor 8: precip risk
    precip = weather.get("precipProb", 0) if weather else 0
    precip_mult = clamp(1 - 0.001 * precip, 0.93, 1.0)

    # Factor 9: season phase
    season_mult = 1.0
    if season_day < 30:
        season_mult = 0.93
    elif season_day > 140:
        season_mult = 1.05

    # Factor 10: lineup slot
    order = game.get("order", 5)
    lineup_mult = clamp(0.90 + 0.025 * (5 - order), 0.85, 1.10)

    # Factor 11: day/night
    is_day = game.get("dayNight", "night") == "day"
    day_night_mult = 0.97 if is_day else 1.02

    # Factor 12: H2H
    h2h_mult = 1.0
    if h2h and h2h.get("pa", 0) >= 10:
        h2h_hr = h2h.get("hr", 0)
        h2h_pa = h2h.get("pa", 1)
        h2h_rate = h2h_hr / h2h_pa
        h2h_mult = clamp(1 + 0.50 * (h2h_rate / LG["hrPA"] - 1), 0.80, 1.35)

    # Factor 13: vs hand
    vs_mult = 1.0
    if vs_hand and vs_hand.get("pa", 0) >= 50:
        vs_hr = vs_hand.get("hr", 0)
        vs_pa = vs_hand.get("pa", 1)
        vs_rate = vs_hr / vs_pa
        vs_mult = clamp(vs_rate / LG["hrPA"], 0.60, 1.55)

    # Factor 14: K alignment (flattened)
    k_align_mult = 1.0

    # Factor 15: count advantage (flattened)
    count_adv_mult = count_advantage(k_pct, bb_pct, pit_k_pct, pit_bb_pct)

    # Factor 16: barrel proxy
    barrel = est_barrel(iso)
    barrel_mult = clamp(1 + 0.90 * (barrel / 0.078 - 1), 0.75, 1.40)

    # Factor 17: zone match score
    zone_score = 50 + (iso / LG["iso"] - 1) * 14 - (pit_k_pct / LG["pitKPct"] - 1) * 11
    if platoon:
        zone_score += 8
    zone_mult = clamp(zone_score / 50, 0.75, 1.35)

    # Factor 18: hard contact
    dbl = batter_stat.get("dbl", 0) if batter_stat else 0
    trp = batter_stat.get("trp", 0) if batter_stat else 0
    xbh = (dbl + trp + hr) / pa if pa > 0 else 0
    hard_score = clamp(0.55 * ((slg - avg) / LG["iso"]) + 0.45 * (xbh / 0.081), 0.35, 2.2)
    hard_mult = clamp(0.88 + 0.12 * hard_score, 0.88, 1.16)

    # Factor 19: pull tendency
    pull_rate = est_pull_rate(hr, ab)
    pull_mult = clamp(0.92 + 0.18 * (pull_rate / LG["pullPct"]), 0.85, 1.18)

    # Factor 20: FB alignment (flattened)
    fb_mult = 1.0

    # Factor 21: pitcher trend (last 3 starts - flattened)
    pit_trend_mult = 1.0

    # Factor 22: pitcher fatigue (flattened)
    pit_fatigue_mult = 1.0

    # Factor 23: slugging trend (L7 vs L14)
    slug_trend_mult = 1.0
    if batter_l7 and batter_l14:
        pa_l7 = batter_l7.get("pa", 0)
        pa_l14 = batter_l14.get("pa", 0)
        if pa_l7 >= 10 and pa_l14 >= 20:
            slg_l7 = batter_l7.get("slg", slg)
            slg_l14 = batter_l14.get("slg", slg)
            trend = (slg_l7 - slg_l14) / max(slg_l14, 0.001)
            slug_trend_mult = clamp(1 + trend * 0.55, 0.90, 1.12)

    # Factor 24: lineup protection
    lineup_prot_mult = 1.0
    if 3 <= order <= 5:
        lineup_prot_mult = 1.04

    # Factor 25-29: advanced factors (not in JS, placeholders)
    air_density_mult = 1.0
    umpire_mult = 1.0
    travel_fatigue_mult = 1.0
    bullpen_vuln_mult = 1.0
    catcher_framing_mult = 1.0

    # Self-calibration (placeholder - use ledger later)
    self_calib_mult = 1.0

    # Combine all
    game_prob = (
        base_rate *
        iso_mult * platoon_mult * park_mult * pitcher_hr_mult * pitcher_k_mult *
        wind_mult * temp_mult * precip_mult * season_mult * lineup_mult *
        day_night_mult * h2h_mult * vs_mult * k_align_mult * count_adv_mult *
        barrel_mult * zone_mult * hard_mult * pull_mult * fb_mult *
        pit_trend_mult * pit_fatigue_mult * slug_trend_mult * lineup_prot_mult *
        air_density_mult * umpire_mult * travel_fatigue_mult * bullpen_vuln_mult * catcher_framing_mult *
        self_calib_mult
    )

    game_prob = clamp(game_prob, 0.001, 0.40)

    grade = "D"
    if game_prob >= 0.16:
        grade = "A"
    elif game_prob >= 0.12:
        grade = "B"
    elif game_prob >= 0.08:
        grade = "C"

    return {
        "pid": pid,
        "name": name,
        "team": team,
        "gameProb": round(game_prob, 4),
        "gameMatchup": game_matchup,
        "battingOrder": order,
        "pitcherName": game.get("pitcher", ""),
        "grade": grade,
        "factors": {
            "iso": round(iso_mult, 3),
            "platoon": round(platoon_mult, 3),
            "park": round(park_mult, 3),
            "wind": round(wind_mult, 3),
            "temp": round(temp_mult, 3)
        }
    }
