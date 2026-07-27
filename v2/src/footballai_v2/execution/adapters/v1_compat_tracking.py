"""V2-controlled execution of the preserved V1 YOLOv8 + ByteTrack stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COLUMNS = ["frame_idx", "time_sec", "track_id", "cx", "cy", "w", "h", "conf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--device", required=True, choices=("mps", "cpu", "cuda"))
    parser.add_argument("--target-fps", required=True, type=float)
    parser.add_argument("--image-size", required=True, type=int)
    parser.add_argument("--confidence", required=True, type=float)
    return parser.parse_args()


def main() -> None:
    # Optional imports remain isolated from demo_fast and bounded CI.
    import cv2
    import pandas as pd
    from tqdm import tqdm
    from ultralytics import YOLO

    args = parse_args()
    video_path = Path(args.video).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model).resolve(strict=True)
    tracker_path = Path(args.tracker).resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if src_fps <= 0 or total_frames <= 0:
        raise RuntimeError("The uploaded video has no processable frames.")
    stride = max(1, round(src_fps / args.target_fps))
    effective_fps = src_fps / stride
    expected_frames = max(1, (total_frames + stride - 1) // stride)
    meta = {
        "src_fps": src_fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_s": total_frames / src_fps,
        "stride": stride,
        "effective_fps": effective_fps,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # The absolute, prevalidated local path is the crucial no-download boundary.
    model = YOLO(str(model_path))
    records: list[dict] = []
    frame_counter = 0
    results = model.track(
        source=str(video_path), tracker=str(tracker_path), classes=[0],
        conf=args.confidence, imgsz=args.image_size, device=args.device,
        vid_stride=stride, stream=True, verbose=False, persist=True,
    )
    for result in tqdm(results, total=expected_frames, desc="Tracking", unit="frame"):
        real_frame = frame_counter * stride
        time_sec = real_frame / src_fps
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            boxes = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()
            for track_id, box, confidence in zip(ids, boxes, confidences):
                x1, y1, x2, y2 = box
                records.append({
                    "frame_idx": real_frame,
                    "time_sec": round(time_sec, 3),
                    "track_id": track_id,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": round(confidence, 3),
                })
        frame_counter += 1

    frame = pd.DataFrame(records, columns=COLUMNS)
    if not frame.empty:
        counts = frame["track_id"].value_counts()
        frame = frame[frame["track_id"].isin(counts[counts >= 10].index)].copy()
    frame.to_parquet(output_dir / "raw_tracks.parquet", index=False)
    counts = frame["track_id"].value_counts() if not frame.empty else None
    summary = {
        "frames_processed": frame_counter,
        "detection_rows": len(frame),
        "tracked_ids": int(frame["track_id"].nunique()) if not frame.empty else 0,
        "max_track_observations": int(counts.max()) if counts is not None and not counts.empty else 0,
        "empty_after_v1_filters": frame.empty or int(counts.max()) < 50,
    }
    (output_dir / "tracking_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
