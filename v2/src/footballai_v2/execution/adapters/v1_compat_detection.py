"""CLI for the AI1 YOLO-only detection stage."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from footballai_v2.execution.ai1.artifacts import detections_to_parquet_bytes, sha256_bytes
from footballai_v2.execution.ai1.detector import YoloDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache deterministic YOLO detections without tracking.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--device", required=True, choices=("mps", "cpu", "cuda"))
    parser.add_argument("--target-fps", required=True, type=float)
    parser.add_argument("--image-size", required=True, type=int)
    parser.add_argument("--confidence", required=True, type=float)
    parser.add_argument("--source-video-sha256", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--pipeline-version", required=True)
    return parser.parse_args()


def main() -> None:
    from ultralytics import __version__ as ultralytics_version

    args = parse_args()
    video_path = Path(args.video).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model).resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    artifact = YoloDetector(
        model_path=model_path,
        model_sha256=args.model_sha256,
        model_version=f"ultralytics-{ultralytics_version}",
        confidence=args.confidence,
        image_size=args.image_size,
        target_fps=args.target_fps,
        device=args.device,
        source_video_sha256=args.source_video_sha256,
        code_revision=args.code_revision,
        pipeline_version=args.pipeline_version,
    ).detect(video_path)
    elapsed = time.perf_counter() - started
    content = detections_to_parquet_bytes(artifact)
    (output_dir / "detections.parquet").write_bytes(content)
    meta = {
        "src_fps": artifact.provenance.source_fps,
        "total_frames": artifact.provenance.source_total_frames,
        "width": artifact.provenance.source_width,
        "height": artifact.provenance.source_height,
        "duration_s": artifact.provenance.source_duration_seconds,
        "stride": artifact.provenance.sampling_stride,
        "effective_fps": artifact.provenance.effective_fps,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    summary = {
        "frames_processed": artifact.frames_processed,
        "detection_count": len(artifact.detections),
        "video_decoding_detection_seconds": round(elapsed, 6),
        "detections_byte_size": len(content),
        "detections_sha256": sha256_bytes(content),
    }
    (output_dir / "detection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
