"""Detector-neutral tracker abstraction with ByteTrack as the AI1 reference."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

from footballai_v2.execution.ai1.artifacts import (
    TRACKLETS_SCHEMA,
    BoundingBox,
    DetectionsArtifact,
    TrackObservation,
    Tracklet,
    TrackerProvenance,
    TrackletsArtifact,
)


class _DetectionBatch:
    """Minimal results object accepted by the Ultralytics ByteTrack API."""

    def __init__(self, xyxy: np.ndarray, confidence: np.ndarray, classes: np.ndarray) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape((-1, 4))
        self.conf = np.asarray(confidence, dtype=np.float32)
        self.cls = np.asarray(classes, dtype=np.float32)
        if len(self.xyxy):
            centers = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
            sizes = self.xyxy[:, 2:] - self.xyxy[:, :2]
            self.xywh = np.concatenate((centers, sizes), axis=1)
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> "_DetectionBatch":
        return _DetectionBatch(self.xyxy[index], self.conf[index], self.cls[index])


class ByteTrackTracker:
    """Consume cached detections; this class has no detector/model dependency."""

    def __init__(
        self,
        *,
        configuration: Mapping[str, Any],
        detections_artifact_id: str,
        detections_sha256: str,
        source_run_id: str,
        tracker_version: str,
        tracker_factory: Callable[[SimpleNamespace], object] | None = None,
    ) -> None:
        self.configuration = dict(configuration)
        self.detections_artifact_id = detections_artifact_id
        self.detections_sha256 = detections_sha256
        self.source_run_id = source_run_id
        self.tracker_version = tracker_version
        self.tracker_factory = tracker_factory

    def track(self, detections: DetectionsArtifact) -> TrackletsArtifact:
        configuration = SimpleNamespace(**self.configuration)
        if self.tracker_factory is None:
            from ultralytics.trackers.byte_tracker import BYTETracker

            tracker = BYTETracker(configuration)
        else:
            tracker = self.tracker_factory(configuration)
        by_frame = defaultdict(list)
        for item in detections.detections:
            by_frame[item.frame_index].append(item)
        observations: dict[int, list[TrackObservation]] = defaultdict(list)
        track_classes: dict[int, tuple[int, str]] = {}
        for sampled in detections.sampled_frames:
            frame_detections = by_frame[sampled.frame_index]
            batch = _DetectionBatch(
                np.asarray([[item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2] for item in frame_detections]),
                np.asarray([item.confidence for item in frame_detections]),
                np.asarray([item.class_id for item in frame_detections]),
            )
            output = np.asarray(tracker.update(batch), dtype=np.float32)
            if output.size == 0:
                continue
            output = output.reshape((-1, 8))
            for row in output:
                x1, y1, x2, y2, raw_track_id, confidence, raw_class_id, _detection_index = row
                x1 = min(max(float(x1), 0.0), float(detections.provenance.source_width))
                y1 = min(max(float(y1), 0.0), float(detections.provenance.source_height))
                x2 = min(max(float(x2), 0.0), float(detections.provenance.source_width))
                y2 = min(max(float(y2), 0.0), float(detections.provenance.source_height))
                if x2 <= x1 or y2 <= y1:
                    continue
                track_id = int(raw_track_id)
                class_id = int(raw_class_id)
                semantic_class = next(
                    (item.semantic_class for item in frame_detections if item.class_id == class_id),
                    "person",
                )
                track_classes.setdefault(track_id, (class_id, semantic_class))
                observations[track_id].append(TrackObservation(
                    frame_index=sampled.frame_index,
                    timestamp_seconds=sampled.timestamp_seconds,
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=float(confidence),
                ))
        tracklets = tuple(
            Tracklet(track_id, *track_classes[track_id], tuple(items))
            for track_id, items in sorted(observations.items())
        )
        provenance = TrackerProvenance(
            tracker_name="ByteTrack",
            tracker_version=self.tracker_version,
            configuration=self.configuration,
            detections_artifact_id=self.detections_artifact_id,
            detections_sha256=self.detections_sha256,
            source_run_id=self.source_run_id,
        )
        return TrackletsArtifact(TRACKLETS_SCHEMA, provenance, tracklets)
