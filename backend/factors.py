import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

LG = {
    "avg": 0.244, "obp": 0.314, "slg": 0.408, "ops": 0.722,
    "iso": 0.155, "kPct": 0.228, "bbPct": 0.085, "hrPA": 0.0307,
    "fbPct": 0.38, "pullPct": 0.38, "linePct": 0.21,
    "pitHR9": 1.28, "pitKPct": 0.228, "pitBBPct": 0.085, "pitFBPct": 0.38,
    "xbhPA": 0.081, "hardHitPct": 0.38
}

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def regression_weight(pa):
    """Dynamic regression weight based on sample size"""
    if pa < 50:
        return 0.85  # Heavy regression for tiny samples
    elif pa < 100:
        return 0.70
    elif pa < 200:
        return 0.50
    elif pa < 400:
        return 0.30
    else:
        return 0.15  # Light regression for large samples

def est_barrel(iso):
    return clamp((iso / LG["iso"]) * 0.078, 0.04, 0.16)

def est_pull_rate(hr, ab):
    hr_rate = hr / max(ab, 1)
    return clamp(0.35 + 0.10 * (hr_rate / LG["hrPA"]), 0.25, 0.50)

def est_hard_contact(slg, avg, xbh, pa):
    """NEW FACTOR 1: Estimate hard contact rate from SLG-AVG and XBH rate"""
    if pa < 20:
        return LG["hardHitPct"]
    slg_iso = slg - avg
    xbh_rate = xbh / pa if pa > 0 else LG["xbhPA"]
    hard_proxy = 0.60 * (slg_iso / LG["iso"]) + 0.40 * (xbh_rate / LG["xbhPA"])
    return clamp(hard_proxy * LG["hardHitPct"], 0.25, 0.55)

def est_sprint_speed_boost(sb, pa):
    """NEW FACTOR 2: Sprint speed proxy from stolen bases (not HR relevant but for completeness)"""
    # Faster runners might hit more triples, but less HR power typically
    # We'll use this inversely: more SB = slight HR penalty
    if pa < 50:
        return 1.0
    sb_rate = sb / pa if pa > 0 else 0
    # High SB rate (>0.15) suggests speed over power
    if sb_rate > 0.15:
        return 0.96  # Slight penalty
    return 1.0

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

def k_rate_trend(k_recent, k_season, pa_recent):
    """NEW FACTOR 3: Recent K-rate improvement"""
    if pa_recent < 20:
        return 1.0
    k_diff = k_season - k_recent  # Positive if recent K% is lower (better)
    # Lower K% recently = better contact = HR boost
    return clamp(1 + k_diff * 0.35, 0.90, 1.12)

def bullpen_vulnerability(team_bullpen_hr9):
    """NEW FACTOR 4: Opposing bullpen HR/9 rate"""
    if team_bullpen_hr9 is None:
        return 1.0
    mult = team_bullpen_hr9 / LG["pitHR9"]
    return clamp(0.92 + 0.16 * (mult - 1), 0.85, 1.18)

def pitcher_fatigue_factor(pitcher_ip_season):
    """NEW FACTOR 5: Pitcher fatigue from cumulative innings pitched"""
    if pitcher_ip_season is None or pitcher_ip_season < 30:
        return 1.0
    # Pitchers degrade after ~160 IP in a season
    if pitcher_ip_season > 160:
        fatigue = (pitcher_ip_season - 160) / 40  # Every 40 IP past 160 = +2.5% HR rate
        return clamp(1 + fatigue * 0.025, 1.0, 1.10)
    return 1.0

def park_hand_factor(park_lhf, park_rhf, batter_hand):
    """Improved: Use hand-specific park factors"""
    if batter_hand == "L":
        return park_lhf if park_lhf else 1.0
    elif batter_hand == "R":
        return park_rhf if park_rhf else 1.0
    else:  # Switch hitter
        return (park_lhf + park_rhf) / 2 if (park_lhf and park_rhf) else 1.0

def compute_model(batter, batter_stat, batter_l7, batter_l14, vs_hand, h2h,
                 pitcher_stat, pitcher_l3, game, weather, park_factor, season_day,
                 bullpen_hr9=None, pitcher_ip_season=None):
    
    pid = batter.get("id")
    name = f"{batter.get('first', '')} {batter.get('last', '')}".strip()
    team = game.get("team_abbr", "")
    opp_team = game.get("opp_abbr", "")
    game_matchup = f"{team} @ {opp_team}" if game.get("isAway") else f"{team} vs {opp_team}"
    
    bats = batter.get("bats", "R")
    pit_throws = game.get("pitThrows", "R")
    platoon = (bats == "L" and pit_throws == "R") or (bats == "R" and pit_throws == "L")
    
    # Require minimum PA
    pa = batter_stat.get("pa", 0) if batter_stat else 0
    if pa < 40:
        return None
    
    ab = batter_stat.get("ab", 0) if batter_stat else 0
    hr = batter_stat.get("hr", 0) if batter_stat else 0
    bb = batter_stat.get("bb", 0) if batter_stat else 0
    k = batter_stat.get("k", 0) if batter_stat else 0
    sb = batter_stat.get("sb", 0) if batter_stat else 0
    dbl = batter_stat.get("dbl", 0) if batter_stat else 0
    trp = batter_stat.get("trp", 0) if batter_stat else 0
    
    avg = batter_stat.get("avg", LG["avg"]) if batter_stat else LG["avg"]
    slg = batter_stat.get("slg", LG["slg"]) if batter_stat else LG["slg"]
    iso = batter_stat.get("iso", LG["iso"]) if batter_stat else LG["iso"]
    
    k_pct = k / pa if pa > 0 else LG["kPct"]
    bb_pct = bb / pa if pa > 0 else LG["bbPct"]
    xbh = dbl + trp + hr
    
    # Regression-weighted base rate
    rw = regression_weight(pa)
    raw_rate = hr / pa if pa > 0 else LG["hrPA"]
    base_rate = rw * LG["hrPA"] + (1 - rw) * raw_rate
    
    # === EXISTING FACTORS (1-29) ===
    
    # Factor 1: ISO multiplier
    iso_mult = clamp(iso / LG["iso"], 0.45, 2.10)
    
    # Factor 2: Platoon advantage
    platoon_mult = 1.15 if platoon else 0.96
    
    # Factor 3: Park factor (now hand-specific)
    if isinstance(park_factor, dict):
        park_lhf = park_factor.get("lhf", 1.0)
        park_rhf = park_factor.get("rhf", 1.0)
        park_mult = park_hand_factor(park_lhf, park_rhf, bats)
    else:
        park_mult = park_factor if park_factor else 1.0
    
    # Factor 4-15: (Placeholder - already in original, keeping structure)
    pitcher_hr_mult = 1.0  # Would use pitcher HR/9 if available
    pitcher_k_mult = 1.0
    
    # Factor 6: Weather wind
    cf_bearing = park_factor.get("cfBearing", 0) if isinstance(park_factor, dict) else 0
    wind_speed = weather.get("windSpeed", 0) if weather else 0
    wind_dir = weather.get("windDir", 0) if weather else 0
    wind_mult = wind_effect(wind_speed, wind_dir, cf_bearing)
    
    # Factor 7: Temperature
    temp = weather.get("temp", 70) if weather else 70
    temp_mult = clamp(0.93 + 0.01 * (temp - 70) / 10, 0.88, 1.14)
    
    # Factor 8: Precipitation risk
    precip = weather.get("precipProb", 0) if weather else 0
    precip_mult = clamp(1 - 0.001 * precip, 0.93, 1.0)
    
    # Factor 9: Season phase
    season_mult = 1.0
    if season_day < 30:
        season_mult = 0.93
    elif season_day > 140:
        season_mult = 1.05
    
    # Factor 10: Lineup slot
    order = game.get("order", 5)
    lineup_mult = clamp(0.90 + 0.025 * (5 - order), 0.85, 1.10)
    
    # Factor 11: Day/night
    is_day = game.get("dayNight", "night") == "day"
    day_night_mult = 0.97 if is_day else 1.02
    
    # Factor 12: H2H history
    h2h_mult = 1.0
    if h2h and h2h.get("pa", 0) >= 10:
        h2h_hr = h2h.get("hr", 0)
        h2h_pa = h2h.get("pa", 1)
        h2h_rate = h2h_hr / h2h_pa
        h2h_mult = clamp(1 + 0.50 * (h2h_rate / LG["hrPA"] - 1), 0.80, 1.35)
    
    # Factor 13: Vs hand splits
    vs_mult = 1.0
    if vs_hand and vs_hand.get("pa", 0) >= 50:
        vs_hr = vs_hand.get("hr", 0)
        vs_pa = vs_hand.get("pa", 1)
        vs_rate = vs_hr / vs_pa
        vs_mult = clamp(vs_rate / LG["hrPA"], 0.60, 1.55)
    
    # Factor 14-15: Count advantage
    pit_k_pct = LG["pitKPct"]
    pit_bb_pct = LG["pitBBPct"]
    count_adv_mult = count_advantage(k_pct, bb_pct, pit_k_pct, pit_bb_pct)
    
    # Factor 16: Barrel estimate
    barrel = est_barrel(iso)
    barrel_mult = clamp(1 + 0.90 * (barrel / 0.078 - 1), 0.75, 1.40)
    
    # Factor 17: Zone match
    zone_score = 50 + (iso / LG["iso"] - 1) * 14 - (pit_k_pct / LG["pitKPct"] - 1) * 11
    if platoon:
        zone_score += 8
    zone_mult = clamp(zone_score / 50, 0.75, 1.35)
    
    # Factor 18: Hard contact (IMPROVED with new function)
    hard_rate = est_hard_contact(slg, avg, xbh, pa)
    hard_mult = clamp(0.88 + 0.24 * (hard_rate / LG["hardHitPct"]), 0.88, 1.20)
    
    # Factor 19: Pull tendency
    pull_rate = est_pull_rate(hr, ab)
    pull_mult = clamp(0.92 + 0.18 * (pull_rate / LG["pullPct"]), 0.85, 1.18)
    
    # Factor 20-22: (Placeholders - pitcher trends, fatigue, FB%)
    pit_trend_mult = 1.0
    fb_mult = 1.0
    
    # Factor 23: Slugging trend (L7 vs L14)
    slug_trend_mult = 1.0
    if batter_l7 and batter_l14:
        pa_l7 = batter_l7.get("pa", 0)
        pa_l14 = batter_l14.get("pa", 0)
        if pa_l7 >= 10 and pa_l14 >= 20:
            slg_l7 = batter_l7.get("slg", slg)
            slg_l14 = batter_l14.get("slg", slg)
            trend = (slg_l7 - slg_l14) / max(slg_l14, 0.001)
            slug_trend_mult = clamp(1 + trend * 0.55, 0.90, 1.12)
    
    # Factor 24: Lineup protection
    lineup_prot_mult = 1.0
    if 3 <= order <= 5:
        lineup_prot_mult = 1.04
    
    # Factor 25-29: Advanced placeholders
    air_density_mult = 1.0
    umpire_mult = 1.0
    travel_fatigue_mult = 1.0
    catcher_framing_mult = 1.0
    
    # === NEW FACTORS (30-34) ===
    
    # Factor 30: NEW - K-rate trend
    k_recent_pct = batter_l7.get("k", 0) / batter_l7.get("pa", 1) if batter_l7 and batter_l7.get("pa", 0) >= 15 else k_pct
    k_trend_mult = k_rate_trend(k_recent_pct, k_pct, batter_l7.get("pa", 0) if batter_l7 else 0)
    
    # Factor 31: NEW - Sprint speed (inverse for power)
    sprint_mult = est_sprint_speed_boost(sb, pa)
    
    # Factor 32: NEW - Bullpen vulnerability
    bullpen_mult = bullpen_vulnerability(bullpen_hr9)
    
    # Factor 33: NEW - Pitcher fatigue (IP cumulative)
    pit_fatigue_mult = pitcher_fatigue_factor(pitcher_ip_season)
    
    # Factor 34: Self-calibration (placeholder - would use historical accuracy)
    self_calib_mult = 1.0
    
    # === CONVERGENCE BONUS (IMPROVED) ===
    # Count strong signals (factors > 1.08 or < 0.94)
    strong_signals = 0
    factors_list = [
        iso_mult, platoon_mult, park_mult, wind_mult, temp_mult,
        h2h_mult, vs_mult, barrel_mult, zone_mult, hard_mult,
        pull_mult, slug_trend_mult, lineup_prot_mult, k_trend_mult,
        bullpen_mult, pit_fatigue_mult
    ]
    
    for f in factors_list:
        if f > 1.08 or f < 0.94:
            strong_signals += 1
    
    # Improved convergence: 0-2 signals = 1.0x, 3-4 = 1.03x, 5-6 = 1.06x, 7+ = 1.10x
    if strong_signals >= 7:
        convergence_mult = 1.10
    elif strong_signals >= 5:
        convergence_mult = 1.06
    elif strong_signals >= 3:
        convergence_mult = 1.03
    else:
        convergence_mult = 1.0
    
    # === COMBINE ALL FACTORS ===
    game_prob = (
        base_rate *
        iso_mult * platoon_mult * park_mult * pitcher_hr_mult * pitcher_k_mult *
        wind_mult * temp_mult * precip_mult * season_mult * lineup_mult *
        day_night_mult * h2h_mult * vs_mult * count_adv_mult *
        barrel_mult * zone_mult * hard_mult * pull_mult * fb_mult *
        pit_trend_mult * slug_trend_mult * lineup_prot_mult *
        air_density_mult * umpire_mult * travel_fatigue_mult *
        catcher_framing_mult * k_trend_mult * sprint_mult *
        bullpen_mult * pit_fatigue_mult * self_calib_mult *
        convergence_mult
    )
    
    game_prob = clamp(game_prob, 0.001, 0.40)
    
    # Grading
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
            "temp": round(temp_mult, 3),
            "kTrend": round(k_trend_mult, 3),
            "bullpen": round(bullpen_mult, 3),
            "pitFatigue": round(pit_fatigue_mult, 3),
            "convergence": round(convergence_mult, 3)
        }
    }