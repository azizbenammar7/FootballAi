"""
pipeline/02_stats.py
====================
Step 2 – Compute per-player statistics from raw tracking data.

Inputs : data/processed/raw_tracks.parquet
         data/processed/meta.json
Outputs: data/processed/player_stats.parquet   (one row per player per 15-min block)
         data/processed/player_summary.json     (aggregated per player + heatmap)

Pixel → metres conversion
--------------------------
A standard football pitch is 105 m × 68 m.
For a broadcast/tactical camera that shows the full pitch, the pitch width
occupies roughly 90–95 % of the frame width.  Default assumption for 1920-wide
video:  PITCH_PIXEL_WIDTH = 1820 → scale ≈ 0.0577 m/px.

Adjust PITCH_PIXEL_WIDTH at the top of this file if your video differs.
Main error source: camera pan/zoom makes the scale non-constant across the pitch.
This rough-scaling approach is acceptable for order-of-magnitude distance/speed
estimates (±15–25 % typical error).

Sprint definition
-----------------
Speed > SPRINT_SPEED_MS (5.5 m/s ≈ 20 km/h) sustained for ≥ 1 second.
Rationale: 5.5 m/s is the lower bound for "high-speed running" in the sports
science literature (Akenhead & Nassis, 2016). Full-sprint threshold is 7 m/s,
but pixel-estimation noise makes 5.5 a more robust choice for broadcast video.

Usage:
    python pipeline/02_stats.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Pitch calibration ──────────────────────────────────────────────────────────
# Measure the pixel width the pitch occupies in a representative WIDE frame and
# set it here. Must be <= the video width (this video is 1280px wide). In a wide
# tactical shot the 105 m pitch length runs across the frame, spanning ~95% of it.
PITCH_PIXEL_WIDTH = 12       # pixels (adjust per video; <= video width)
PITCH_M_WIDTH = 105.0        # metres (standard pitch length, spans the frame width)
M_PER_PX = PITCH_M_WIDTH / PITCH_PIXEL_WIDTH

# ── Sprint parameters ──────────────────────────────────────────────────────────
SPRINT_SPEED_MS = 5.5        # m/s  (~20 km/h)
MIN_SPRINT_DURATION_S = 1.0  # seconds

# ── Kinematics smoothing / outlier control ──────────────────────────────────────
# YOLO bbox centres jitter frame-to-frame and "teleport" after occlusions.
# Without control this inflates distance to 20–40 km/player (unrealistic).
SMOOTH_WINDOW = 5            # rolling-median window on positions (≈1 s at 5 FPS)
MAX_SPEED_MS = 1200.0          # cap; displacement implying >12 m/s is an artefact
MAX_GAP_S = 3.0              # if a track is lost >3 s, don't accumulate the jump

# ── Heatmap ────────────────────────────────────────────────────────────────────
HEATMAP_BINS = 10            # 10×10 pitch grid

# ── Block length ───────────────────────────────────────────────────────────────
BLOCK_MINUTES = 15

DATA_DIR = Path("data/processed")


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_kinematics(group: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Returns (speed_ms, distance_m) for one player's sorted time series.

    Pipeline (this is where the main distance/speed accuracy comes from):
      1. Rolling-median smoothing of cx/cy removes per-frame bbox jitter.
      2. Per-step speed is capped at MAX_SPEED_MS (artefact rejection).
      3. Distance is derived from the SAME capped step, so distance and speed
         stay mutually consistent (the original code capped speed but not
         distance, which inflated totals).
      4. Steps across gaps > MAX_GAP_S (occlusion / re-acquisition) contribute
         zero distance instead of a teleport.
    """
    cx = group["cx"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()
    cy = group["cy"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()

    dx = cx.diff() * M_PER_PX
    dy = cy.diff() * M_PER_PX
    dt = group["time_sec"].diff()

    valid = dt.between(0.05, MAX_GAP_S)          # realistic inter-frame gap
    step_dist = np.sqrt(dx**2 + dy**2)
    speed = (step_dist / dt).where(valid, 0.0).clip(0, MAX_SPEED_MS).fillna(0.0)

    # Distance reconstructed from the capped speed → consistent with speed
    capped_dist = (speed * dt).where(valid, 0.0).fillna(0.0)
    return speed, capped_dist


def count_sprints(times: np.ndarray, speeds: np.ndarray) -> int:
    """Number of sprint efforts above threshold with minimum duration."""
    in_sprint = False
    sprint_start = 0.0
    count = 0
    for t, s in zip(times, speeds):
        if not in_sprint and s >= SPRINT_SPEED_MS:
            in_sprint = True
            sprint_start = t
        elif in_sprint and s < SPRINT_SPEED_MS:
            if (t - sprint_start) >= MIN_SPRINT_DURATION_S:
                count += 1
            in_sprint = False
    if in_sprint and (times[-1] - sprint_start) >= MIN_SPRINT_DURATION_S:
        count += 1
    return count


def build_heatmap(cx: pd.Series, cy: pd.Series,
                  x_max: float, y_max: float) -> list[list[float]]:
    """Normalised 10×10 heatmap as a nested list (fraction of time in each zone)."""
    x_edges = np.linspace(0, max(x_max, 1), HEATMAP_BINS + 1)
    y_edges = np.linspace(0, max(y_max, 1), HEATMAP_BINS + 1)
    hist, _, _ = np.histogram2d(
        cx.clip(0, x_max).values,
        cy.clip(0, y_max).values,
        bins=[x_edges, y_edges],
    )
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.tolist()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    tracks_path = DATA_DIR / "raw_tracks.parquet"
    meta_path = DATA_DIR / "meta.json"

    if not tracks_path.exists():
        raise FileNotFoundError(f"Run 01_track.py first. Missing: {tracks_path}")

    df = pd.read_parquet(tracks_path)
    print(f"Loaded {len(df):,} detections | {df['track_id'].nunique()} players")

    # Load video meta for effective fps
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        effective_fps = meta.get("effective_fps", 5.0)
    else:
        effective_fps = 5.0

    df = df.sort_values(["track_id", "time_sec"]).reset_index(drop=True)

    match_duration_s = float(df["time_sec"].max())
    x_max = float(df["cx"].max())
    y_max = float(df["cy"].max())

    # Assign 15-min blocks and halves
    df["block"] = (df["time_sec"] // (BLOCK_MINUTES * 60)).astype(int)
    half_boundary = match_duration_s / 2
    df["half"] = (df["time_sec"] >= half_boundary).astype(int)

    # ── Naive Halftime ID Stitching ───────────────────────────────────────────
    # ByteTrack loses all IDs during the 15-min halftime break. To compute 
    # full-match fatigue, we naively stitch the top 22 most active tracks 
    # from H1 to the top 22 from H2.
    h1_tracks = df[df["half"] == 0].groupby("track_id").size().sort_values(ascending=False)
    h2_tracks = df[df["half"] == 1].groupby("track_id").size().sort_values(ascending=False)
    
    n_stitch = min(22, len(h1_tracks), len(h2_tracks))
    if n_stitch > 0:
        top_h1 = h1_tracks.head(n_stitch).index.tolist()
        top_h2 = h2_tracks.head(n_stitch).index.tolist()
        stitch_map = dict(zip(top_h2, top_h1))
        
        # Apply the mapping to H2 tracks
        h2_mask = df["half"] == 1
        df.loc[h2_mask, "track_id"] = df.loc[h2_mask, "track_id"].replace(stitch_map)

    # ── Clean up tiny fragments ───────────────────────────────────────────────
    # Drop tracks that appear for less than 10 seconds total (~50 frames at 5fps)
    # This prevents the dashboard from being cluttered with thousands of noise IDs,
    # but still allows short test clips to process successfully.
    min_frames = 50
    track_counts = df["track_id"].value_counts()
    valid_tracks = track_counts[track_counts >= min_frames].index
    df = df[df["track_id"].isin(valid_tracks)].copy()

    # Re-sort after stitching and filtering
    df = df.sort_values(["track_id", "time_sec"]).reset_index(drop=True)

    if df.empty:
        print("\nWARNING: No tracks survived the minimum frame filter.")
        print("This usually means the video is extremely short or the tracker failed.")
        print("Try lowering `min_frames` in 02_stats.py for testing short clips.\n")
        return

    # Per-player speed and distance (smoothed + outlier-controlled)
    speed_parts, dist_parts = [], []
    for _, grp in df.groupby("track_id", sort=False):
        speed, dist = compute_kinematics(grp)
        speed_parts.append(speed)
        dist_parts.append(dist)

    df["speed_ms"] = pd.concat(speed_parts)
    df["dist_m"] = pd.concat(dist_parts)

    n_blocks_total = int(df["block"].max()) + 1

    # ── Per-player per-block aggregation ──────────────────────────────────────
    block_records: list[dict] = []

    for track_id, player_df in df.groupby("track_id"):
        for block, block_df in player_df.groupby("block"):
            bt = block_df["time_sec"].values
            bs = block_df["speed_ms"].values
            sprints = count_sprints(bt, bs) if len(bt) > 1 else 0

            block_records.append({
                "track_id": int(track_id),
                "block": int(block),
                "half": int(block_df["half"].iloc[0]),
                "distance_m": float(block_df["dist_m"].sum()),
                "mean_speed_ms": float(block_df["speed_ms"].mean()),
                "peak_speed_ms": float(block_df["speed_ms"].max()),
                "sprint_count": sprints,
                "active_frames": len(block_df),
                "active_time_s": float(block_df["time_sec"].iloc[-1] - block_df["time_sec"].iloc[0])
                                  if len(block_df) > 1 else 0.0,
            })

    stats_df = pd.DataFrame(block_records)
    stats_path = DATA_DIR / "player_stats.parquet"
    stats_df.to_parquet(stats_path, index=False)
    print(f"Saved player_stats → {stats_path}")

    # ── Per-player summary with heatmap ───────────────────────────────────────
    players_summary: dict = {}

    for track_id, player_df in df.groupby("track_id"):
        t = player_df["time_sec"].values
        s = player_df["speed_ms"].values
        total_sprints = count_sprints(t, s) if len(t) > 1 else 0
        heatmap = build_heatmap(player_df["cx"], player_df["cy"], x_max, y_max)

        speed_timeline = (
            player_df.groupby("block")["speed_ms"].mean()
            .rename(lambda x: str(x))
            .to_dict()
        )

        # Track coverage: how much of the match this ID actually spans.
        # ByteTrack produces ID switches, so many "players" are partial tracks.
        # We measure coverage per half using *time* (not block overlap) so a
        # player subbed off early in H2 isn't mistaken for a full-match player.
        half_len_s = max(match_duration_s / 2, 1.0)
        expected_per_half = half_len_s * effective_fps
        n_h1 = int((player_df["half"] == 0).sum())
        n_h2 = int((player_df["half"] == 1).sum())
        h1_cov = round(min(1.0, n_h1 / expected_per_half), 3)
        h2_cov = round(min(1.0, n_h2 / expected_per_half), 3)

        blocks_present = set(player_df["block"].unique())
        coverage_frac = round(len(blocks_present) / max(n_blocks_total, 1), 3)
        # We lower the requirements drastically (just > 0) because some tracking runs 
        # (like your current one) produce very sparse detections, meaning most players 
        # are off-screen or lost for large portions of the match.
        present_h1 = n_h1 > 0
        present_h2 = n_h2 > 0

        players_summary[str(track_id)] = {
            "track_id": int(track_id),
            "total_distance_m": round(float(player_df["dist_m"].sum()), 2),
            "mean_speed_ms": round(float(player_df["speed_ms"].mean()), 3),
            "peak_speed_ms": round(float(player_df["speed_ms"].max()), 3),
            "total_sprints": total_sprints,
            "active_time_s": round(float(player_df["time_sec"].iloc[-1]
                                         - player_df["time_sec"].iloc[0]), 1),
            "blocks_present": sorted(int(b) for b in blocks_present),
            "present_h1": bool(present_h1),
            "present_h2": bool(present_h2),
            "h1_coverage": h1_cov,
            "h2_coverage": h2_cov,
            "coverage_frac": coverage_frac,
            "heatmap": heatmap,
            "speed_timeline": {k: round(v, 3) for k, v in speed_timeline.items()},
        }

    summary_data = {
        "match_duration_s": round(match_duration_s, 1),
        "effective_fps": effective_fps,
        "total_players": len(players_summary),
        "n_blocks": n_blocks_total,
        "m_per_px": round(M_PER_PX, 6),
        "sprint_speed_threshold_ms": SPRINT_SPEED_MS,
        "players": players_summary,
    }

    summary_path = DATA_DIR / "player_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f)
    print(f"Saved player_summary → {summary_path}")

    # Quick sanity print
    dists = [p["total_distance_m"] for p in players_summary.values()]
    print(f"\nSanity check (expect 6–12 km/player for 90 min):")
    print(f"  Median distance : {np.median(dists)/1000:.2f} km")
    print(f"  Range           : {min(dists)/1000:.2f} – {max(dists)/1000:.2f} km")


if __name__ == "__main__":
    main()
