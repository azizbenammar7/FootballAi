"""Synthetic AI1 artifacts and landmark annotations with a known pitch ground truth.

Shared by the AI3-B stage, worker and API tests. Pinhole cameras, ideal boxes;
nothing here represents real broadcast footage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random

from footballai_v2.execution.ai1.artifacts import (
    DETECTIONS_SCHEMA,
    TRACKLETS_SCHEMA,
    BoundingBox,
    Detection,
    DetectionProvenance,
    DetectionsArtifact,
    SampledFrame,
    TrackerProvenance,
    Tracklet,
    TrackletsArtifact,
    TrackObservation,
    detections_to_parquet_bytes,
    sha256_bytes,
    tracklets_to_parquet_bytes,
)
from footballai_v2.execution.ai3.annotations import (
    PITCH_ANNOTATIONS_SCHEMA,
    AnnotationOrigin,
    AnnotationSourceInfo,
    CameraMotion,
    PitchAnnotations,
    ShotSegment,
)
from footballai_v2.execution.ai3.geometry import IFAB_LAW_1_MARKINGS, CameraSide, PitchGeometry
from footballai_v2.execution.ai3.synthetic import SyntheticCamera, synthetic_frame


PITCH = PitchGeometry(105.0, 68.0, IFAB_LAW_1_MARKINGS, "test configuration")
CAMERA = SyntheticCamera(52.5, 98.0, 20.0, 40.0, 34.0, 1400.0)
OTHER_CAMERA = SyntheticCamera(52.5, 98.0, 20.0, 70.0, 30.0, 1400.0)
FPS = 5.0
VIDEO_BYTES = b"bounded-video-bytes"
VIDEO_SHA = hashlib.sha256(VIDEO_BYTES).hexdigest()
PLAYERS = [(20.0, 20.0), (35.0, 50.0), (45.0, 30.0)]
RUN_ID = "11111111-1111-4111-8111-111111111111"


def provenance(video_sha=VIDEO_SHA, width=1920, height=1080) -> DetectionProvenance:
    return DetectionProvenance(
        model_family="Ultralytics YOLO", model_name="yolov8m.pt", model_version="YOLOv8m", model_sha256="a" * 64,
        confidence_threshold=0.2, image_size=1280, target_fps=FPS, effective_fps=FPS, device="cpu",
        source_video_sha256=video_sha, source_width=width, source_height=height, source_fps=25.0,
        source_total_frames=1500, sampling_stride=5, source_duration_seconds=60.0, code_revision="c" * 40,
        pipeline_version="v1_compat/2.2.0",
    )


def ai1_bytes(frames=range(0, 41), *, camera=CAMERA, video_sha=VIDEO_SHA, ball=True, shift_timestamp=False):
    """Synthetic AI1 detections + tracklets whose foot points are known pitch positions."""
    frames = list(frames)
    detections, tracklets = [], []
    boxes: dict[int, list[tuple[int, BoundingBox]]] = {}
    for track_id, (x, y) in enumerate(PLAYERS, start=1):
        u, v = camera.project(x, y)
        box = BoundingBox(u - 12, v - 70, u + 12, v)
        for index in frames:
            boxes.setdefault(index, []).append((track_id, box))
    for index in frames:
        for _, box in boxes[index]:
            detections.append(Detection(index, index / FPS, 0, "person", 0.9, box))
        if ball:
            detections.append(Detection(index, index / FPS, 32, "sports ball", 0.5, BoundingBox(900, 600, 910, 610)))
    det = DetectionsArtifact(
        DETECTIONS_SCHEMA, provenance(video_sha), tuple(detections),
        tuple(SampledFrame(index, index / FPS) for index in frames),
    )
    det_bytes = detections_to_parquet_bytes(det)
    for track_id in range(1, len(PLAYERS) + 1):
        observations = []
        for index in frames:
            box = next(b for t, b in boxes[index] if t == track_id)
            stamp = index / FPS + (0.05 if shift_timestamp and index == frames[-1] else 0.0)
            observations.append(TrackObservation(index, stamp, box, 0.9))
        tracklets.append(Tracklet(track_id, 0, "person", tuple(observations)))
    if ball:
        tracklets.append(Tracklet(99, 32, "sports ball", tuple(
            TrackObservation(index, index / FPS, BoundingBox(900, 600, 910, 610), 0.5) for index in frames
        )))
    trk = TrackletsArtifact(
        TRACKLETS_SCHEMA, TrackerProvenance("ByteTrack", "test", {}, "detections", sha256_bytes(det_bytes), RUN_ID),
        tuple(tracklets),
    )
    return det_bytes, tracklets_to_parquet_bytes(trk)


def notes(keyframes=(10, 30), *, origin=AnnotationOrigin.SYNTHETIC, video_sha=VIDEO_SHA, motion=CameraMotion.STATIC,
          cameras=None, size=(1920, 1080), timestamp_offset=0.0) -> PitchAnnotations:
    cameras = cameras or {}
    frames = tuple(
        dataclasses.replace(
            synthetic_frame(
                PITCH, cameras.get(index, CAMERA), frame_index=index, timestamp_seconds=index / FPS, shot_id="s1",
                rng=random.Random(index + 17), noise_px=0.5,
            ),
            timestamp_seconds=index / FPS + timestamp_offset,
        )
        for index in keyframes
    )
    return PitchAnnotations(
        PITCH_ANNOTATIONS_SCHEMA,
        AnnotationSourceInfo("test-clip", origin, size[0], size[1], video_sha, None, None),
        PITCH, (ShotSegment("s1", 0, 100, motion, CameraSide.BOTTOM_TOUCHLINE),), frames,
    )
