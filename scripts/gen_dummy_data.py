"""
scripts/gen_dummy_data.py
=========================
Generates realistic synthetic match data so you can develop and demo the
dashboard before (or without) running the full CV pipeline.

Simulates 22 players over a 90-minute match at 5 FPS effective rate.
Players have realistic fatigue: speed and sprint rate decline in the 2nd half.

Output:
    data/processed/meta.json
    data/processed/raw_tracks.parquet
    data/processed/player_stats.parquet
    data/processed/player_summary.json
    data/processed/risk_scores.json

Usage:
    python scripts/gen_dummy_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so pipeline modules are importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

N_PLAYERS      = 22
MATCH_DURATION = 90 * 60        # seconds
EFFECTIVE_FPS  = 5.0
VIDEO_W, VIDEO_H = 1920, 1080
PITCH_W_PX     = 1820           # assumed pitch width in pixels (matches 02_stats.py)
BLOCK_S        = 15 * 60        # 15 min in seconds

M_PER_PX = 105.0 / PITCH_W_PX


# ── Simulate raw tracks ────────────────────────────────────────────────────────

def simulate_player(track_id: int,
                    t_start: float = 0.0,
                    t_end: float = MATCH_DURATION) -> pd.DataFrame:
    """
    Random walk on the pitch with a realistic fatigue profile.
    t_start / t_end let us model substitutions (a track that only spans part
    of the match), which exercises the INSUFFICIENT path in 03_fatigue.py.
    """
    times = np.arange(t_start, t_end, 1.0 / EFFECTIVE_FPS)

    # Base speed (m/s) varies by "role": ~2.0–2.8 m/s average across match
    # (lands total distance around 8–11 km/player after smoothing & drops)
    base_speed_ms = RNG.uniform(2.0, 2.8)

    # Fatigue factor: speed declines from first to last block
    fatigue_slope = RNG.uniform(-0.05, -0.01)  # m/s per block

    cx = np.zeros(len(times))
    cy = np.zeros(len(times))
    cx[0] = RNG.uniform(0.1 * VIDEO_W, 0.9 * VIDEO_W)
    cy[0] = RNG.uniform(0.2 * VIDEO_H, 0.8 * VIDEO_H)

    for i in range(1, len(times)):
        t = times[i]
        block = int(t // BLOCK_S)
        speed_ms = max(0.3, base_speed_ms + fatigue_slope * block)

        # Occasional burst (sprint ~10% of time)
        if RNG.random() < 0.10:
            speed_ms *= RNG.uniform(2.5, 4.5)

        # Convert speed to pixel displacement
        dist_px = speed_ms / M_PER_PX / EFFECTIVE_FPS
        angle   = RNG.uniform(0, 2 * np.pi)
        dx      = dist_px * np.cos(angle)
        dy      = dist_px * np.sin(angle)

        # Bounce off pitch edges
        cx[i] = np.clip(cx[i-1] + dx, 0.05 * VIDEO_W, 0.95 * VIDEO_W)
        cy[i] = np.clip(cy[i-1] + dy, 0.15 * VIDEO_H, 0.85 * VIDEO_H)

    # Drop ~8% of frames (occlusion / tracking loss)
    mask = RNG.random(len(times)) > 0.08

    w = RNG.uniform(30, 55, size=mask.sum())
    h = w * RNG.uniform(2.0, 2.8, size=mask.sum())

    return pd.DataFrame({
        "frame_idx": (times[mask] * 25).astype(int),   # original 25 FPS frame
        "time_sec":  np.round(times[mask], 3),
        "track_id":  track_id,
        "cx":        cx[mask],
        "cy":        cy[mask],
        "w":         w,
        "h":         h,
        "conf":      RNG.uniform(0.55, 0.99, size=mask.sum()).round(3),
    })


def main():
    print("Generating dummy match data…")

    # ── raw_tracks.parquet ────────────────────────────────────────────────────
    # 22 full-match players …
    all_tracks = [simulate_player(i + 1) for i in range(N_PLAYERS)]
    # … plus 3 substitutions (partial tracks) to exercise the INSUFFICIENT path:
    #   - two players subbed OFF at the hour mark (present H1 only)
    #   - one player subbed ON at the hour mark (present H2 only)
    all_tracks.append(simulate_player(N_PLAYERS + 1, t_start=0,        t_end=60 * 60))
    all_tracks.append(simulate_player(N_PLAYERS + 2, t_start=0,        t_end=65 * 60))
    all_tracks.append(simulate_player(N_PLAYERS + 3, t_start=60 * 60,  t_end=MATCH_DURATION))
    tracks_df = pd.concat(all_tracks, ignore_index=True)
    tracks_df.to_parquet(PROCESSED / "raw_tracks.parquet", index=False)
    print(f"  raw_tracks.parquet   {len(tracks_df):,} rows")

    # ── meta.json ─────────────────────────────────────────────────────────────
    meta = {
        "src_fps": 25.0,
        "total_frames": int(MATCH_DURATION * 25),
        "width": VIDEO_W,
        "height": VIDEO_H,
        "duration_s": float(MATCH_DURATION),
        "stride": 5,
        "effective_fps": EFFECTIVE_FPS,
    }
    with open(PROCESSED / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Run stats and fatigue scripts ─────────────────────────────────────────
    # Modules with numeric prefixes can't be imported via normal import;
    # use importlib to load them by file path.
    import importlib.util

    def run_script(path: Path):
        spec = importlib.util.spec_from_file_location("_mod", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.main()

    print("  Running 02_stats.py …")
    run_script(ROOT / "pipeline" / "02_stats.py")
    print("  Running 03_fatigue.py …")
    run_script(ROOT / "pipeline" / "03_fatigue.py")

    print("\nDone! Launch dashboard:")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
