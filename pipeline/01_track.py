"""
pipeline/01_track.py
====================
Step 1 – Detect and track players using YOLOv8n + ByteTrack.

Output: data/processed/raw_tracks.parquet
        data/processed/meta.json  (video metadata)

Throughput strategy for a full 90-min match on MacBook Air M4:
  - YOLOv8n  (fastest YOLO variant, ~60 FPS on M4 MPS)
  - vid_stride=5 samples every 5th frame → effective 5 FPS from a 25 FPS source
  - MPS device (Apple Metal)  → GPU acceleration without CUDA
  - 90 min × 60 s × 5 FPS = 27 000 frames  ≈ 25–45 min wall-clock time

Usage:
    python pipeline/01_track.py --video data/raw/match.mp4
    python pipeline/01_track.py --video data/raw/match.mp4 --fps 3  # slower but faster processing
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = "yolov8m.pt"   # medium model: far better recall on small/distant players
TARGET_FPS = 5              # effective FPS after downsampling
CONF = 0.20                 # lower threshold catches more small players on the pitch
IMGSZ = 1280                # inference resolution; match native width so players stay detectable
DEVICE = "mps"              # "mps" for M-series Mac, "cuda" for Nvidia, "cpu" fallback
CHECKPOINT_EVERY = 3000     # flush partial results to disk every N processed frames
TRACKER = "pipeline/bytetrack_custom.yaml"  # custom tracker: longer buffer = fewer ID switches
OUTPUT_DIR = Path("data/processed")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to match video")
    p.add_argument("--fps", type=float, default=TARGET_FPS,
                   help=f"Target processing FPS (default {TARGET_FPS})")
    p.add_argument("--output_dir", default=str(OUTPUT_DIR))
    return p.parse_args()


def get_video_meta(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    meta = {
        "src_fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    meta["duration_s"] = meta["total_frames"] / meta["src_fps"] if meta["src_fps"] > 0 else 0
    cap.release()
    return meta


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    meta = get_video_meta(str(video_path))
    src_fps = meta["src_fps"]
    stride = max(1, round(src_fps / args.fps))
    effective_fps = src_fps / stride
    n_frames_to_process = meta["total_frames"] // stride

    print(f"Video : {video_path.name}")
    print(f"Source: {src_fps:.1f} FPS  {meta['width']}×{meta['height']}  "
          f"{meta['duration_s']/60:.1f} min")
    print(f"Processing every {stride} frames → {effective_fps:.1f} effective FPS")
    print(f"Frames to process: {n_frames_to_process:,}  "
          f"(estimated {n_frames_to_process/60:.0f} min wall-clock @ 60 it/s)")

    meta["stride"] = stride
    meta["effective_fps"] = effective_fps
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    model = YOLO(MODEL_NAME)

    # Robustness: stream rows to a CSV checkpoint as we go, so a crash at
    # minute 85 of a 90-min match doesn't discard the whole run. The CSV is
    # converted to the final parquet at the end and then removed.
    checkpoint_path = output_dir / "raw_tracks_checkpoint.csv"
    columns = ["frame_idx", "time_sec", "track_id", "cx", "cy", "w", "h", "conf"]
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    records: list[dict] = []
    frame_counter = 0
    rows_written = 0

    def flush(buffer: list[dict]):
        nonlocal rows_written
        if not buffer:
            return
        chunk = pd.DataFrame(buffer, columns=columns)
        chunk.to_csv(
            checkpoint_path, mode="a", header=not rows_written, index=False
        )
        rows_written += len(chunk)

    tracker_cfg = TRACKER if Path(TRACKER).exists() else "bytetrack.yaml"

    results_gen = model.track(
        source=str(video_path),
        tracker=tracker_cfg,
        classes=[0],           # class 0 = person
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        vid_stride=stride,
        stream=True,
        verbose=False,
        persist=True,          # keep tracker state across calls
    )

    try:
        for result in tqdm(results_gen, total=n_frames_to_process,
                           desc="Tracking", unit="frame"):
            real_frame = frame_counter * stride
            time_sec = real_frame / src_fps

            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().tolist()
                xyxy = result.boxes.xyxy.tolist()
                confs = result.boxes.conf.tolist()

                for track_id, box, conf in zip(ids, xyxy, confs):
                    x1, y1, x2, y2 = box
                    records.append({
                        "frame_idx": real_frame,
                        "time_sec": round(time_sec, 3),
                        "track_id": track_id,
                        "cx": (x1 + x2) / 2,
                        "cy": (y1 + y2) / 2,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "conf": round(conf, 3),
                    })

            frame_counter += 1

            if frame_counter % CHECKPOINT_EVERY == 0:
                flush(records)
                records = []
    except KeyboardInterrupt:
        print("\nInterrupted — flushing partial results so far…")
    finally:
        flush(records)

    if rows_written == 0:
        print("WARNING: No detections found. Check video path and model.")
        return

    df = pd.read_csv(checkpoint_path)

    # Remove tracks that appear in fewer than 10 frames (noise / referee / spectator edge)
    track_counts = df["track_id"].value_counts()
    valid_tracks = track_counts[track_counts >= 10].index
    df = df[df["track_id"].isin(valid_tracks)].copy()

    out_path = output_dir / "raw_tracks.parquet"
    df.to_parquet(out_path, index=False)

    # Remove the CSV checkpoint now that the final parquet is written.
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nDone. {len(df):,} detections across {df['track_id'].nunique()} tracks")
    print(f"Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
