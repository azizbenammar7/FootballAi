"""Ultralytics detector adapter that performs inference without tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from footballai_v2.execution.ai1.artifacts import (
    DETECTIONS_SCHEMA,
    BoundingBox,
    Detection,
    DetectionProvenance,
    DetectionsArtifact,
    SampledFrame,
)


class YoloDetector:
    """Run YOLO once and return the complete sampled-frame detection cache."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_sha256: str,
        model_version: str,
        confidence: float,
        image_size: int,
        target_fps: float,
        device: str,
        source_video_sha256: str,
        code_revision: str,
        pipeline_version: str,
        model_factory: Callable[[str], object] | None = None,
    ) -> None:
        self.model_path = model_path
        self.model_sha256 = model_sha256
        self.model_version = model_version
        self.confidence = confidence
        self.image_size = image_size
        self.target_fps = target_fps
        self.device = device
        self.source_video_sha256 = source_video_sha256
        self.code_revision = code_revision
        self.pipeline_version = pipeline_version
        self.model_factory = model_factory

    def detect(self, video_path: Path) -> DetectionsArtifact:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if source_fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("The uploaded video has no processable frames.")
        stride = max(1, round(source_fps / self.target_fps))
        effective_fps = source_fps / stride
        if self.model_factory is None:
            from ultralytics import YOLO

            factory: Callable[[str], object] = YOLO
        else:
            factory = self.model_factory
        # The caller supplies an already validated absolute weight file and sets
        # YOLO_OFFLINE. No model alias is ever passed across this boundary.
        model = factory(str(self.model_path.resolve(strict=True)))
        results = model.predict(
            source=str(video_path), classes=[0], conf=self.confidence,
            imgsz=self.image_size, device=self.device, vid_stride=stride,
            stream=True, verbose=False,
        )
        sampled_frames = []
        detections = []
        for sample_index, result in enumerate(results):
            frame_index = sample_index * stride
            timestamp = frame_index / source_fps
            sampled_frames.append(SampledFrame(frame_index, timestamp))
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            names = getattr(result, "names", {})
            for box, confidence, class_id in zip(xyxy, confidences, classes):
                detections.append(Detection(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    class_id=int(class_id),
                    semantic_class=str(names.get(int(class_id), "person")),
                    confidence=float(confidence),
                    bbox=BoundingBox(*(float(value) for value in box)),
                ))
        provenance = DetectionProvenance(
            model_family="Ultralytics YOLO",
            model_name=self.model_path.name,
            model_version=self.model_version,
            model_sha256=self.model_sha256,
            confidence_threshold=self.confidence,
            image_size=self.image_size,
            target_fps=self.target_fps,
            effective_fps=effective_fps,
            device=self.device,
            source_video_sha256=self.source_video_sha256,
            source_width=width,
            source_height=height,
            source_fps=source_fps,
            source_total_frames=total_frames,
            sampling_stride=stride,
            source_duration_seconds=total_frames / source_fps,
            code_revision=self.code_revision,
            pipeline_version=self.pipeline_version,
        )
        return DetectionsArtifact(
            DETECTIONS_SCHEMA, provenance, tuple(detections), tuple(sampled_frames)
        )
