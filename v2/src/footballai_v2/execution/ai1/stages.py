"""Small explicit detector and tracker stage interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from footballai_v2.execution.ai1.artifacts import DetectionsArtifact, TrackletsArtifact


class Detector(Protocol):
    def detect(self, video_path: Path) -> DetectionsArtifact: ...


class Tracker(Protocol):
    def track(self, detections: DetectionsArtifact) -> TrackletsArtifact: ...


@dataclass(frozen=True, slots=True)
class DetectionStage:
    detector: Detector

    def run(self, video_path: Path) -> DetectionsArtifact:
        return self.detector.detect(video_path)


@dataclass(frozen=True, slots=True)
class TrackingStage:
    tracker: Tracker

    def run(self, detections: DetectionsArtifact) -> TrackletsArtifact:
        return self.tracker.track(detections)
