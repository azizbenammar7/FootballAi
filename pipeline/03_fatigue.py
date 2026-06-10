"""
pipeline/03_fatigue.py
======================
Step 3 – Derive fatigue indicators and compute an injury-risk score per player.

Inputs : data/processed/player_stats.parquet
         data/processed/player_summary.json
Output : data/processed/risk_scores.json

Fatigue model
-------------
Four independent indicators, each scored 0–25 (total 0–100):

1. Speed decay (0–25 pts)
   Fit a linear regression through mean_speed per 15-min block.
   A negative slope = declining pace = fatigue.
   Reference: Bradley et al. (2010) show ~12% speed decrease over 90 min.
   Normalisation: slope of –0.10 m/s per block maps to 25 pts.

2. Sprint frequency drop (0–25 pts)
   (H1 sprints – H2 sprints) / H1 sprints  as a ratio.
   Rationale: Mohr et al. (2005) show significant 2nd-half sprint reduction.
   25 pts = 100% drop (no sprints in 2nd half).

3. Distance drop-off (0–25 pts)
   (H1 distance – H2 distance) / H1 distance.
   25 pts = player covers 60 % less ground in 2nd half.

4. High-speed running (HSR) load (0–25 pts)
   Players in top quartile of total distance accumulate highest muscle strain.
   25 pts = top-quartile player.  Rationale: Malone et al. (2017) link HSR volume
   to next-day injury risk.

Risk flag
---------
  LOW          : total score < 40
  MEDIUM       : 40–69
  HIGH         : ≥ 70
  INSUFFICIENT : track too short / not present in both halves to judge fatigue

Why INSUFFICIENT matters
------------------------
ByteTrack produces ID switches, so a 90-min match yields many short-lived track
IDs, not 22 clean players. The half-vs-half indicators (sprint drop, distance
drop) would otherwise flag every substituted player or ID fragment as HIGH risk
simply because they appear in only one half. We therefore only assign a fatigue
score to tracks that are present in both halves with enough match coverage; the
rest are surfaced honestly as INSUFFICIENT rather than as false positives.

A transparent heuristic is deliberately chosen over a black box so coaches can
understand and challenge each flag.

Usage:
    python pipeline/03_fatigue.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DATA_DIR = Path("data/processed")
HSR_THRESHOLD_MS = 5.5    # m/s – same as sprint threshold in 02_stats
MIN_COVERAGE_FRAC = 0.05  # track must span ≥5% of the match to be scored


def speed_decay_score(speed_timeline: dict[str, float]) -> tuple[float, float]:
    """
    Returns (score 0-25, slope m/s per block).
    slope < 0  means speed is declining (fatigue signal).
    """
    if len(speed_timeline) < 3:
        return 0.0, 0.0
    blocks = sorted(int(k) for k in speed_timeline)
    speeds = [speed_timeline[str(b)] for b in blocks]
    if len(set(speeds)) < 2:
        return 0.0, 0.0
    slope, *_ = scipy_stats.linregress(blocks, speeds)
    # –0.10 m/s per block = full 25 pts; positive slope = no fatigue
    score = min(25.0, max(0.0, -slope / 0.10 * 25.0))
    return float(score), float(slope)


def sprint_drop_score(h1_sprints: int, h2_sprints: int) -> tuple[float, float]:
    """Returns (score 0-25, drop ratio 0-1)."""
    if h1_sprints <= 0:
        return 0.0, 0.0
    drop = max(0.0, (h1_sprints - h2_sprints) / h1_sprints)
    return float(min(25.0, drop * 25.0)), float(drop)


def dist_drop_score(h1_dist: float, h2_dist: float) -> tuple[float, float]:
    """Returns (score 0-25, drop ratio 0-1).  60 % drop = 25 pts."""
    if h1_dist <= 0:
        return 0.0, 0.0
    drop = max(0.0, (h1_dist - h2_dist) / h1_dist)
    return float(min(25.0, drop / 0.60 * 25.0)), float(drop)


def hsr_load_score(total_dist_m: float, p75_dist: float) -> float:
    """
    Players above the 75th percentile in total distance get 25 pts.
    Below p75: proportional.
    """
    if p75_dist <= 0:
        return 0.0
    return float(min(25.0, total_dist_m / p75_dist * 25.0))


def main():
    stats_df = pd.read_parquet(DATA_DIR / "player_stats.parquet")
    with open(DATA_DIR / "player_summary.json") as f:
        summary = json.load(f)

    players = summary["players"]

    # Population-level threshold for HSR load scoring — computed only over
    # full-match tracks so partial fragments don't distort the percentile.
    scorable = [
        p for p in players.values()
        if p.get("present_h1") and p.get("present_h2")
        and p.get("coverage_frac", 0) >= MIN_COVERAGE_FRAC
    ]
    pool = scorable if len(scorable) >= 4 else list(players.values())
    all_dists = [p["total_distance_m"] for p in pool]
    p75_dist = float(np.percentile(all_dists, 75)) if len(all_dists) >= 4 else max(all_dists, default=1.0)

    risk_data: dict = {}
    n_insufficient = 0

    for tid_str, player in players.items():
        pstats = stats_df[stats_df["track_id"] == int(tid_str)]
        h1 = pstats[pstats["half"] == 0]
        h2 = pstats[pstats["half"] == 1]

        h1_sprints = int(h1["sprint_count"].sum()) if len(h1) > 0 else 0
        h2_sprints = int(h2["sprint_count"].sum()) if len(h2) > 0 else 0
        h1_dist = float(h1["distance_m"].sum()) if len(h1) > 0 else 0.0
        h2_dist = float(h2["distance_m"].sum()) if len(h2) > 0 else 0.0

        # Gate: only judge fatigue for tracks that actually span both halves.
        is_scorable = (
            player.get("present_h1") and player.get("present_h2")
            and player.get("coverage_frac", 0) >= MIN_COVERAGE_FRAC
        )

        if not is_scorable:
            n_insufficient += 1
            risk_data[tid_str] = {
                "track_id": int(tid_str),
                "risk_score": None,
                "risk_flag": "INSUFFICIENT",
                "coverage_frac": player.get("coverage_frac", 0),
                "reason": "partial track (ID switch / sub) — not present across both halves",
                "fatigue_indicators": {
                    "h1_sprints": h1_sprints,
                    "h2_sprints": h2_sprints,
                    "h1_distance_m": round(h1_dist, 1),
                    "h2_distance_m": round(h2_dist, 1),
                },
                "score_breakdown": {},
            }
            print(f"Player {tid_str:>3s} | INSUFFICIENT (coverage {player.get('coverage_frac', 0):.0%})")
            continue

        s_decay, slope = speed_decay_score(player["speed_timeline"])
        s_sprint, sprint_drop = sprint_drop_score(h1_sprints, h2_sprints)
        s_dist, dist_drop = dist_drop_score(h1_dist, h2_dist)
        s_hsr = hsr_load_score(player["total_distance_m"], p75_dist)

        total = int(round(s_decay + s_sprint + s_dist + s_hsr))
        total = min(100, max(0, total))

        flag = "HIGH" if total >= 70 else ("MEDIUM" if total >= 40 else "LOW")

        risk_data[tid_str] = {
            "track_id": int(tid_str),
            "risk_score": total,
            "risk_flag": flag,
            "coverage_frac": player.get("coverage_frac", 1.0),
            "fatigue_indicators": {
                "speed_decay_slope_ms_per_block": round(slope, 4),
                "sprint_drop_pct": round(sprint_drop * 100, 1),
                "distance_drop_pct": round(dist_drop * 100, 1),
                "h1_sprints": h1_sprints,
                "h2_sprints": h2_sprints,
                "h1_distance_m": round(h1_dist, 1),
                "h2_distance_m": round(h2_dist, 1),
            },
            "score_breakdown": {
                "speed_decay":   round(s_decay, 1),
                "sprint_drop":   round(s_sprint, 1),
                "distance_drop": round(s_dist, 1),
                "hsr_load":      round(s_hsr, 1),
            },
        }

        print(f"Player {tid_str:>3s} | {flag:<6s} {total:>3d}/100 | "
              f"decay={slope:+.3f}  sprint_drop={sprint_drop:.0%}  "
              f"dist_drop={dist_drop:.0%}  hsr={s_hsr:.1f}")

    out_path = DATA_DIR / "risk_scores.json"
    with open(out_path, "w") as f:
        json.dump(risk_data, f, indent=2)
    print(f"\nSaved risk scores for {len(risk_data)} players → {out_path}")

    high = sum(1 for r in risk_data.values() if r["risk_flag"] == "HIGH")
    med  = sum(1 for r in risk_data.values() if r["risk_flag"] == "MEDIUM")
    low  = sum(1 for r in risk_data.values() if r["risk_flag"] == "LOW")
    print(f"Summary: HIGH={high}  MEDIUM={med}  LOW={low}  "
          f"INSUFFICIENT={n_insufficient}")
    print(f"({len(risk_data) - n_insufficient} full-match tracks scored, "
          f"{n_insufficient} partial tracks held out)")


if __name__ == "__main__":
    main()
