"""CLI for replayable ByteTrack tracking over an AI1 detection cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from footballai_v2.execution.ai1.artifacts import (
    detections_from_parquet_bytes,
    sha256_bytes,
    tracklets_to_parquet_bytes,
)
from footballai_v2.execution.ai1.diagnostics import compute_fragmentation_diagnostics
from footballai_v2.execution.ai1.tracker import ByteTrackTracker


COLUMNS = ["frame_idx", "time_sec", "track_id", "cx", "cy", "w", "h", "conf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track cached detections without detector inference.")
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--source-run-id", required=True)
    return parser.parse_args()


def _tracker_configuration(path: Path) -> dict:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "tracker_type", "track_high_thresh", "track_low_thresh",
        "new_track_thresh", "track_buffer", "match_thresh", "fuse_score",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("tracker_type") != "bytetrack":
        raise ValueError("ByteTrack configuration is incomplete or unsupported")
    return value


def main() -> None:
    import pandas as pd
    from ultralytics import __version__ as ultralytics_version

    args = parse_args()
    detections_path = Path(args.detections).resolve(strict=True)
    tracker_path = Path(args.tracker).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_bytes = detections_path.read_bytes()
    detections = detections_from_parquet_bytes(detection_bytes)
    configuration = _tracker_configuration(tracker_path)
    started = time.perf_counter()
    tracklets = ByteTrackTracker(
        configuration=configuration,
        detections_artifact_id="detections",
        detections_sha256=sha256_bytes(detection_bytes),
        source_run_id=args.source_run_id,
        tracker_version=f"ultralytics-{ultralytics_version}",
    ).track(detections)
    tracking_seconds = time.perf_counter() - started
    tracklet_bytes = tracklets_to_parquet_bytes(tracklets)
    (output_dir / "tracklets.parquet").write_bytes(tracklet_bytes)

    # Compatibility adapter only: retain the historical >=10-observation input
    # contract for 02_stats.py without discarding short tracklets from AI1.
    records = []
    for tracklet in tracklets.tracklets:
        if tracklet.observation_count < 10:
            continue
        for item in tracklet.observations:
            records.append({
                "frame_idx": item.frame_index,
                "time_sec": round(item.timestamp_seconds, 3),
                "track_id": tracklet.track_id,
                "cx": item.bbox.center_x,
                "cy": item.bbox.center_y,
                "w": item.bbox.x2 - item.bbox.x1,
                "h": item.bbox.y2 - item.bbox.y1,
                "conf": round(item.confidence, 3),
            })
    frame = pd.DataFrame(records, columns=COLUMNS)
    frame.to_parquet(output_dir / "raw_tracks.parquet", index=False)
    counts = frame["track_id"].value_counts() if not frame.empty else None

    diagnostics_started = time.perf_counter()
    diagnostics = compute_fragmentation_diagnostics(detections, tracklets)
    diagnostics_seconds = time.perf_counter() - diagnostics_started
    diagnostics["stage_timings"] = {
        "tracking_seconds": round(tracking_seconds, 6),
        "diagnostic_computation_seconds": round(diagnostics_seconds, 6),
    }
    (output_dir / "fragmentation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "frames_processed": detections.frames_processed,
        "detection_rows": len(frame),
        "tracked_ids": int(frame["track_id"].nunique()) if not frame.empty else 0,
        "max_track_observations": int(counts.max()) if counts is not None and not counts.empty else 0,
        "empty_after_v1_filters": frame.empty or int(counts.max()) < 50,
        "total_detections": len(detections.detections),
        "total_tracklets": len(tracklets.tracklets),
        "tracking_seconds": round(tracking_seconds, 6),
        "diagnostic_computation_seconds": round(diagnostics_seconds, 6),
        "tracklets_byte_size": len(tracklet_bytes),
        "tracklets_sha256": sha256_bytes(tracklet_bytes),
    }
    (output_dir / "tracking_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
