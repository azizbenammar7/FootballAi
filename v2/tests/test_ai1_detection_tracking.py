"""AI1 detection/tracking decoupling and fragmentation diagnostics tests."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from footballai_v2.execution.ai1.artifacts import (
    DETECTIONS_SCHEMA,
    TRACKLETS_SCHEMA,
    BoundingBox,
    Detection,
    DetectionProvenance,
    DetectionsArtifact,
    SampledFrame,
    TrackObservation,
    Tracklet,
    TrackerProvenance,
    TrackletsArtifact,
    detections_from_parquet_bytes,
    detections_to_parquet_bytes,
    tracklets_from_parquet_bytes,
    tracklets_to_parquet_bytes,
)
from footballai_v2.execution.ai1.diagnostics import compute_fragmentation_diagnostics
from footballai_v2.execution.ai1.stages import DetectionStage, TrackingStage
from footballai_v2.execution.ai1.tracker import ByteTrackTracker
from footballai_v2.storage.object_storage.memory import InMemoryObjectStorage
from footballai_v2.contracts.v1 import ArtifactCategory, StageName
from footballai_v2.execution.coordinator import AnalysisCoordinator


def detection_provenance() -> DetectionProvenance:
    return DetectionProvenance(
        model_family="Ultralytics YOLO",
        model_name="yolov8m.pt",
        model_version="YOLOv8m",
        model_sha256="a" * 64,
        confidence_threshold=0.2,
        image_size=1280,
        target_fps=5.0,
        effective_fps=5.0,
        device="cpu",
        source_video_sha256="b" * 64,
        source_width=1920,
        source_height=1080,
        source_fps=25.0,
        source_total_frames=1500,
        sampling_stride=5,
        source_duration_seconds=60.0,
        code_revision="c" * 40,
        pipeline_version="v1_compat/2.0.0",
    )


def detection(frame: int, *, confidence: float = 0.9) -> Detection:
    return Detection(
        frame_index=frame,
        timestamp_seconds=frame / 5,
        class_id=0,
        semantic_class="person",
        confidence=confidence,
        bbox=BoundingBox(10, 20, 30, 60),
    )


def detections(*items: Detection) -> DetectionsArtifact:
    return DetectionsArtifact(DETECTIONS_SCHEMA, detection_provenance(), tuple(items))


def tracker_provenance() -> TrackerProvenance:
    return TrackerProvenance(
        tracker_name="ByteTrack",
        tracker_version="ultralytics-8.4.107",
        configuration={"track_buffer": 90, "match_thresh": 0.85},
        detections_artifact_id="detections",
        detections_sha256="d" * 64,
        source_run_id="00000000-0000-4000-8000-000000000001",
    )


def observation(frame: int, *, confidence: float = 0.9) -> TrackObservation:
    return TrackObservation(
        frame_index=frame,
        timestamp_seconds=frame / 5,
        bbox=BoundingBox(10, 20, 30, 60),
        confidence=confidence,
    )


def tracklet(track_id: int, frames: range) -> Tracklet:
    return Tracklet(
        track_id=track_id,
        class_id=0,
        semantic_class="person",
        observations=tuple(observation(frame) for frame in frames),
    )


def test_detections_parquet_roundtrip_is_lossless_and_versioned():
    artifact = detections(detection(0), detection(1, confidence=0.8))
    encoded = detections_to_parquet_bytes(artifact)
    decoded = detections_from_parquet_bytes(encoded)
    assert decoded == artifact
    assert decoded.schema == "footballai.detections/v1"
    assert b"/private/" not in encoded


def test_detections_roundtrip_preserves_sampled_frames_without_detections():
    artifact = DetectionsArtifact(
        DETECTIONS_SCHEMA,
        detection_provenance(),
        (detection(0), detection(2)),
        (SampledFrame(0, 0.0), SampledFrame(1, 0.2), SampledFrame(2, 0.4)),
    )
    decoded = detections_from_parquet_bytes(detections_to_parquet_bytes(artifact))
    assert decoded == artifact
    assert decoded.sampled_frames[1] == SampledFrame(1, 0.2)


def test_cached_detections_pass_existing_integrity_boundary():
    artifact = detections(detection(0))
    encoded = detections_to_parquet_bytes(artifact)
    storage = InMemoryObjectStorage()
    reference = storage.write_artifact(
        "00000000-0000-4000-8000-000000000002",
        artifact_id="detections",
        name="Cached detections",
        category=ArtifactCategory.OTHER,
        relative_path="artifacts/detections.parquet",
        content=encoded,
        media_type="application/vnd.apache.parquet",
        schema_version=DETECTIONS_SCHEMA,
    )
    assert storage.artifact_reference_integrity(
        "00000000-0000-4000-8000-000000000002", reference
    )


def test_cached_detections_can_be_tracked_repeatedly_without_detector(tmp_path):
    calls = {"detector": 0, "tracker": 0}
    cached = detections(detection(0), detection(1))

    class FakeDetector:
        def detect(self, video_path: Path) -> DetectionsArtifact:
            calls["detector"] += 1
            assert video_path == tmp_path / "clip.mp4"
            return cached

    class FakeTracker:
        def track(self, artifact: DetectionsArtifact) -> TrackletsArtifact:
            calls["tracker"] += 1
            assert artifact is cached
            return TrackletsArtifact(
                TRACKLETS_SCHEMA,
                tracker_provenance(),
                (tracklet(calls["tracker"], range(2)),),
            )

    cached_once = DetectionStage(FakeDetector()).run(tmp_path / "clip.mp4")
    stage = TrackingStage(FakeTracker())
    first = stage.run(cached_once)
    second = stage.run(cached_once)
    assert calls == {"detector": 1, "tracker": 2}
    assert first.tracklets[0].track_id != second.tracklets[0].track_id


def test_tracker_configuration_and_detection_link_roundtrip():
    artifact = TrackletsArtifact(
        TRACKLETS_SCHEMA,
        tracker_provenance(),
        (tracklet(7, range(5)),),
    )
    decoded = tracklets_from_parquet_bytes(tracklets_to_parquet_bytes(artifact))
    assert decoded == artifact
    assert decoded.provenance.configuration["track_buffer"] == 90
    assert decoded.provenance.detections_artifact_id == "detections"
    assert decoded.tracklets[0].confidence_summary == {"min": .9, "mean": .9, "max": .9}


def test_bytetrack_adapter_consumes_every_cached_frame_including_empty_frames():
    cached = DetectionsArtifact(
        DETECTIONS_SCHEMA,
        detection_provenance(),
        (detection(0), detection(2)),
        (SampledFrame(0, 0.0), SampledFrame(1, 0.2), SampledFrame(2, 0.4)),
    )
    batches = []

    class FakeByteTracker:
        def update(self, batch):
            batches.append(len(batch))
            if len(batch) == 0:
                return []
            box = batch.xyxy[0]
            return [[*box, 7, batch.conf[0], batch.cls[0], 0]]

    tracker = ByteTrackTracker(
        configuration={"track_buffer": 90},
        detections_artifact_id="detections",
        detections_sha256="d" * 64,
        source_run_id="00000000-0000-4000-8000-000000000001",
        tracker_version="test",
        tracker_factory=lambda _configuration: FakeByteTracker(),
    )
    result = tracker.track(cached)
    assert batches == [1, 0, 1]
    assert result.tracklets[0].track_id == 7
    assert [item.frame_index for item in result.tracklets[0].observations] == [0, 2]


def test_bytetrack_predictions_are_clipped_to_source_dimensions():
    cached = detections(detection(0))

    class FakeByteTracker:
        def update(self, batch):
            return [[-2, -3, 2000, 1200, 8, .8, 0, 0]]

    result = ByteTrackTracker(
        configuration={"track_buffer": 90}, detections_artifact_id="detections",
        detections_sha256="d" * 64,
        source_run_id="00000000-0000-4000-8000-000000000001",
        tracker_version="test", tracker_factory=lambda _configuration: FakeByteTracker(),
    ).track(cached)
    assert result.tracklets[0].observations[0].bbox == BoundingBox(0, 0, 1920, 1080)


def test_real_stage_sources_keep_yolo_out_of_tracking_and_never_download_weights():
    root = Path(__file__).resolve().parents[2]
    detector_source = (root / "v2/src/footballai_v2/execution/ai1/detector.py").read_text()
    tracker_source = (root / "v2/src/footballai_v2/execution/ai1/tracker.py").read_text()
    assert "model.predict(" in detector_source
    assert ".track(" not in detector_source
    assert "YOLO" not in tracker_source
    assert "yolov8m.pt" not in tracker_source
    assert "download" not in detector_source.lower() + tracker_source.lower()


@pytest.mark.parametrize(
    "bbox",
    [
        BoundingBox(-1, 0, 10, 10),
        BoundingBox(10, 0, 10, 10),
        BoundingBox(10, 20, 9, 30),
        BoundingBox(0, 0, 1921, 10),
        BoundingBox(0, 0, math.inf, 10),
    ],
)
def test_invalid_detection_bounding_boxes_are_rejected(bbox):
    item = replace(detection(0), bbox=bbox)
    with pytest.raises(ValueError, match="bounding box"):
        detections(item)


def test_invalid_detection_and_observation_ordering_is_rejected():
    with pytest.raises(ValueError, match="frame order"):
        detections(detection(2), detection(1))
    later_frame_earlier_time = replace(detection(2), timestamp_seconds=0.0)
    with pytest.raises(ValueError, match="timestamp order"):
        detections(detection(1), later_frame_earlier_time)
    with pytest.raises(ValueError, match="strictly increasing"):
        Tracklet(1, 0, "person", (observation(2), observation(1)))


def test_empty_diagnostics_are_safe_deterministic_and_finite():
    result = compute_fragmentation_diagnostics(detections(), TrackletsArtifact(
        TRACKLETS_SCHEMA, tracker_provenance(), ()
    ))
    assert result == compute_fragmentation_diagnostics(detections(), TrackletsArtifact(
        TRACKLETS_SCHEMA, tracker_provenance(), ()
    ))
    assert result["observed_metrics"]["total_detections"] == 0
    assert result["observed_metrics"]["total_tracklets"] == 0
    assert_finite_numbers(result)


def test_one_long_continuous_tracklet_diagnostics_are_correct():
    cached = detections(*(detection(frame) for frame in range(101)))
    tracked = TrackletsArtifact(
        TRACKLETS_SCHEMA, tracker_provenance(), (tracklet(1, range(101)),)
    )
    result = compute_fragmentation_diagnostics(cached, tracked)
    metrics = result["observed_metrics"]
    assert metrics["total_detections"] == 101
    assert metrics["total_tracklets"] == 1
    assert metrics["tracklet_duration_seconds"] == {
        "min": 20.0, "median": 20.0, "mean": 20.0,
        "p90": 20.0, "p95": 20.0, "max": 20.0,
    }
    assert metrics["insufficient_tracklets"]["count"] == 0
    assert metrics["detection_assignment_fraction"] == 1.0


def test_many_fragmented_tracklets_have_documented_short_and_insufficient_rates():
    cached = detections(*(detection(frame) for frame in range(20)))
    tracked = TrackletsArtifact(
        TRACKLETS_SCHEMA,
        tracker_provenance(),
        tuple(tracklet(index + 1, range(index * 2, index * 2 + 2)) for index in range(10)),
    )
    result = compute_fragmentation_diagnostics(cached, tracked)
    metrics = result["observed_metrics"]
    assert metrics["total_tracklets"] == 10
    assert metrics["short_tracklets"]["1_seconds"]["count"] == 10
    assert metrics["short_tracklets"]["1_seconds"]["percentage"] == 100.0
    assert metrics["insufficient_tracklets"] == {
        "criterion": "observation_count < 50",
        "count": 10,
        "percentage": 100.0,
    }
    assert metrics["track_start_rate_per_minute"] == 10.0
    assert metrics["track_end_rate_per_minute"] == 10.0
    assert result["interpretation"]["status"] == "descriptive_only"
    assert_finite_numbers(result)


def test_diagnostics_do_not_claim_labelled_identity_metrics():
    result = compute_fragmentation_diagnostics(
        detections(detection(0)),
        TrackletsArtifact(TRACKLETS_SCHEMA, tracker_provenance(), (tracklet(1, range(1)),)),
    )
    serialized = str(result).lower()
    for forbidden in ("hota", "idf1", "id switches", "id_switches"):
        assert forbidden not in serialized
    assert result["ground_truth_metrics"]["status"] == "not_measured"


def test_v1_analysis_stage_contract_is_preserved_with_future_stages_skipped():
    stages = AnalysisCoordinator._queued_stages(1, "v1_compat")
    assert [stage.stage_name for stage in stages] == list(StageName)
    required = {stage.stage_name: stage.required for stage in stages}
    assert required[StageName.IDENTITY_RESOLUTION] is False
    assert required[StageName.PITCH_CALIBRATION] is False
    assert all(required[name] for name in StageName if name not in {
        StageName.IDENTITY_RESOLUTION, StageName.PITCH_CALIBRATION,
    })


def test_normal_ci_does_not_install_or_launch_full_inference():
    root = Path(__file__).resolve().parents[2]
    workflows = "\n".join(path.read_text() for path in (root / ".github/workflows").glob("*.yml"))
    assert "requirements-v1-compat.txt" not in workflows
    assert "v1_compat_detection" not in workflows
    assert "v2-v1-compat-smoke" not in workflows


def assert_finite_numbers(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            assert_finite_numbers(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite_numbers(child)
    elif isinstance(value, float):
        assert math.isfinite(value)
