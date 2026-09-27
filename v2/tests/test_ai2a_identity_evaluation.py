"""AI2A identity-evaluation foundation contracts.

These tests deliberately use synthetic AI1 artifacts. Normal CI must never need
YOLO, model weights, or a video download to exercise the evaluation layer.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from footballai_v2.execution.ai1.artifacts import (
    TRACKLETS_SCHEMA,
    BoundingBox,
    TrackObservation,
    TrackerProvenance,
    Tracklet,
    TrackletsArtifact,
    tracklets_from_parquet_bytes,
    tracklets_to_parquet_bytes,
)
from footballai_v2.execution.ai2a.association import (
    TeamEvidence,
    TrackletAppearance,
    build_candidate_graph,
    classify_team,
    generate_temporal_cannot_links,
)
from footballai_v2.execution.ai2a.metrics import (
    candidate_pair_recall,
    identity_observation_coverage,
    pairwise_association_metrics,
    team_classification_metrics,
)
from footballai_v2.execution.ai2a.evaluation import (
    EVALUATION_SCHEMA,
    build_evaluation_report,
    evaluation_to_json_bytes,
)
from footballai_v2.execution.ai2a.model import (
    GROUND_TRUTH_SCHEMA,
    AnnotationStatus,
    IdentityGroundTruthArtifact,
    PlayerIdentityId,
    ReviewState,
    SourceReference,
    TeamLabel,
    TrackletAnnotation,
    TrackletId,
    ground_truth_from_json_bytes,
    ground_truth_to_json_bytes,
    tracklet_id,
)
from footballai_v2.execution.ai2a.review import (
    REVIEW_SAMPLES_SCHEMA,
    ReviewSample,
    ReviewSamplesArtifact,
    review_samples_from_json_bytes,
    review_samples_to_json_bytes,
    select_representative_observations,
)


def observation(frame: int, x: float, *, confidence: float = 0.9) -> TrackObservation:
    return TrackObservation(
        frame_index=frame,
        timestamp_seconds=frame / 5,
        bbox=BoundingBox(x, 10, x + 20, 70),
        confidence=confidence,
    )


def make_tracklet(track_id: int, frames: tuple[int, ...], x: float = 10) -> Tracklet:
    return Tracklet(
        track_id=track_id,
        class_id=0,
        semantic_class="person",
        observations=tuple(observation(frame, x) for frame in frames),
    )


def make_tracklets(*items: Tracklet) -> TrackletsArtifact:
    return TrackletsArtifact(
        TRACKLETS_SCHEMA,
        TrackerProvenance(
            tracker_name="ByteTrack",
            tracker_version="test",
            configuration={"track_buffer": 30},
            detections_artifact_id="detections",
            detections_sha256="d" * 64,
            source_run_id="bounded-test",
        ),
        tuple(items),
    )


def source() -> SourceReference:
    return SourceReference(
        source_id="bounded-clip",
        video_sha256="a" * 64,
        duration_seconds=20,
        detections_sha256="b" * 64,
        tracklets_sha256="c" * 64,
    )


def annotation(
    track_id: int,
    identity: str | None,
    *,
    team: TeamLabel = TeamLabel.TEAM_A,
    start: float = 0,
    end: float = 1,
    status: AnnotationStatus = AnnotationStatus.CONFIRMED,
) -> TrackletAnnotation:
    return TrackletAnnotation(
        tracklet_id=tracklet_id(track_id),
        ground_truth_player_id=PlayerIdentityId(identity) if identity else None,
        team_label=team,
        role_label="PLAYER",
        start_frame=round(start * 5),
        end_frame=round(end * 5),
        start_time_seconds=start,
        end_time_seconds=end,
        annotation_confidence=0.9 if identity else None,
        status=status,
        annotator="test-reviewer",
        source="manual-review",
        notes="",
        review_state=ReviewState.REVIEWED,
    )


def test_annotation_round_trip_is_canonical_and_validated_against_ai1_tracklets():
    tracklets = make_tracklets(make_tracklet(1, (0, 5)))
    artifact = IdentityGroundTruthArtifact(
        schema=GROUND_TRUTH_SCHEMA,
        source=source(),
        annotations=(annotation(1, "TEAM_A_01"),),
    )

    encoded = ground_truth_to_json_bytes(artifact)
    decoded = ground_truth_from_json_bytes(encoded, tracklets=tracklets)

    assert decoded == artifact
    assert encoded == ground_truth_to_json_bytes(decoded)


def test_invalid_tracklet_reference_is_rejected():
    artifact = IdentityGroundTruthArtifact(
        schema=GROUND_TRUTH_SCHEMA,
        source=source(),
        annotations=(annotation(99, "TEAM_A_01"),),
    )

    with pytest.raises(ValueError, match="unknown tracklet"):
        ground_truth_from_json_bytes(
            ground_truth_to_json_bytes(artifact),
            tracklets=make_tracklets(make_tracklet(1, (0, 5))),
        )


def test_duplicate_identity_is_allowed_across_non_overlapping_tracklets():
    artifact = IdentityGroundTruthArtifact(
        schema=GROUND_TRUTH_SCHEMA,
        source=source(),
        annotations=(
            annotation(1, "TEAM_A_01", start=0, end=1),
            annotation(2, "TEAM_A_01", start=2, end=3),
        ),
    )

    artifact.validate_identity_assignments()


def test_duplicate_identity_is_rejected_for_overlapping_tracklets():
    artifact = IdentityGroundTruthArtifact(
        schema=GROUND_TRUTH_SCHEMA,
        source=source(),
        annotations=(
            annotation(1, "TEAM_A_01", start=0, end=2),
            annotation(2, "TEAM_A_01", start=1, end=3),
        ),
    )

    tracklets = make_tracklets(
        make_tracklet(1, tuple(range(0, 11)), x=0),
        make_tracklet(2, tuple(range(5, 16)), x=100),
    )

    with pytest.raises(ValueError, match="cannot-link"):
        ground_truth_from_json_bytes(ground_truth_to_json_bytes(artifact), tracklets=tracklets)


def test_unknown_and_uncertain_annotations_are_supported():
    unknown = annotation(
        1,
        None,
        team=TeamLabel.UNKNOWN,
        status=AnnotationStatus.UNKNOWN,
    )
    uncertain = annotation(
        2,
        "TEAM_B_02",
        team=TeamLabel.TEAM_B,
        status=AnnotationStatus.UNCERTAIN,
    )

    artifact = IdentityGroundTruthArtifact(GROUND_TRUTH_SCHEMA, source(), (unknown, uncertain))

    assert artifact.annotations[0].ground_truth_player_id is None
    assert artifact.annotations[1].status is AnnotationStatus.UNCERTAIN


def test_representative_crop_selection_is_bounded_and_deterministic():
    item = make_tracklet(1, tuple(range(10)))

    first = select_representative_observations(item)
    second = select_representative_observations(item)

    assert first == second
    assert [(sample.label, sample.observation.frame_index) for sample in first] == [
        ("early", 0),
        ("middle", 4),
        ("late", 9),
    ]
    assert len(first) <= 3


def test_representative_crop_selection_deduplicates_short_tracklets():
    item = make_tracklet(1, (4, 5))

    assert [sample.label for sample in select_representative_observations(item)] == ["early", "late"]


def test_review_sample_artifact_round_trip_retains_integrity_metadata():
    artifact = ReviewSamplesArtifact(
        REVIEW_SAMPLES_SCHEMA,
        source(),
        "early_middle_late/v1",
        (
            ReviewSample(
                tracklet_id(1),
                "early",
                5,
                1.0,
                (1.0, 2.0, 11.0, 22.0),
                "crops/tracklet_000001/early-f000005.jpg",
                "e" * 64,
            ),
        ),
    )

    encoded = review_samples_to_json_bytes(artifact)

    assert review_samples_from_json_bytes(encoded) == artifact
    assert encoded == review_samples_to_json_bytes(review_samples_from_json_bytes(encoded))


def test_temporal_overlap_generates_cannot_link_only_for_spatially_distinct_people():
    tracklets = make_tracklets(
        make_tracklet(1, (0, 1, 2), x=0),
        make_tracklet(2, (1, 2, 3), x=100),
        make_tracklet(3, (4, 5), x=200),
    )

    constraints = generate_temporal_cannot_links(tracklets)

    assert [(value.left.value, value.right.value) for value in constraints] == [
        ("tracklet_000001", "tracklet_000002")
    ]
    assert constraints[0].reason == "simultaneous_spatially_distinct"


def test_same_tracklet_is_never_compared_with_itself():
    tracklets = make_tracklets(make_tracklet(1, (0, 1)), make_tracklet(2, (2, 3)))
    evidence = {
        tracklet_id(1): TeamEvidence(TeamLabel.UNKNOWN, 0, "jersey_color_v1"),
        tracklet_id(2): TeamEvidence(TeamLabel.UNKNOWN, 0, "jersey_color_v1"),
    }

    graph = build_candidate_graph(tracklets, evidence, {}, ())

    assert all(edge.left != edge.right for edge in graph.edges)


def test_team_unknown_is_a_rejection_not_a_forced_assignment():
    appearance = TrackletAppearance(tracklet_id(1), hue=50, saturation=5, value=120)

    evidence = classify_team(
        appearance,
        {TeamLabel.TEAM_A: (0, 220, 180), TeamLabel.TEAM_B: (110, 220, 180)},
    )

    assert evidence.team is TeamLabel.UNKNOWN
    assert evidence.confidence == 0


def test_single_team_prototype_supports_clips_with_no_second_team():
    appearance = TrackletAppearance(tracklet_id(1), hue=5, saturation=200, value=160)

    evidence = classify_team(appearance, {TeamLabel.TEAM_A: (5, 200, 160)})

    assert evidence.team is TeamLabel.TEAM_A
    assert evidence.confidence == 1


def test_candidate_graph_is_deterministic_and_respects_cannot_links():
    tracklets = make_tracklets(
        make_tracklet(3, (6, 7), x=0),
        make_tracklet(1, (0, 1), x=0),
        make_tracklet(2, (1, 2), x=100),
    )
    teams = {
        tracklet_id(1): TeamEvidence(TeamLabel.TEAM_A, 0.8, "jersey_color_v1"),
        tracklet_id(2): TeamEvidence(TeamLabel.TEAM_A, 0.8, "jersey_color_v1"),
        tracklet_id(3): TeamEvidence(TeamLabel.TEAM_A, 0.8, "jersey_color_v1"),
    }
    appearance = {
        tracklet_id(1): TrackletAppearance(tracklet_id(1), 10, 200, 150),
        tracklet_id(2): TrackletAppearance(tracklet_id(2), 12, 200, 150),
        tracklet_id(3): TrackletAppearance(tracklet_id(3), 11, 200, 150),
    }
    constraints = generate_temporal_cannot_links(tracklets)

    first = build_candidate_graph(tracklets, teams, appearance, constraints)
    second = build_candidate_graph(tracklets, teams, appearance, constraints)

    assert first == second
    assert [(edge.left.value, edge.right.value) for edge in first.edges] == [
        ("tracklet_000001", "tracklet_000003"),
        ("tracklet_000002", "tracklet_000003"),
    ]


def test_pairwise_and_candidate_metrics_on_tiny_ground_truth():
    annotations = (
        annotation(1, "TEAM_A_01", start=0, end=1),
        annotation(2, "TEAM_A_01", start=2, end=3),
        annotation(3, "TEAM_A_02", start=4, end=5),
    )
    predicted = {
        frozenset((tracklet_id(1), tracklet_id(2))),
        frozenset((tracklet_id(2), tracklet_id(3))),
    }

    metrics = pairwise_association_metrics(annotations, predicted)

    assert metrics == {"precision": 0.5, "recall": 1.0, "f1": pytest.approx(2 / 3), "support": 1}
    assert candidate_pair_recall(annotations, predicted) == {"recall": 1.0, "support": 1}


def test_metrics_are_zero_division_safe():
    assert pairwise_association_metrics((), set()) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }
    assert candidate_pair_recall((), set()) == {"recall": 0.0, "support": 0}
    assert team_classification_metrics((), {}) == {
        "accuracy": 0.0,
        "rejection_rate": 0.0,
        "evaluated": 0,
        "correct": 0,
        "unknown_predictions": 0,
    }


def test_team_metrics_measure_accuracy_and_rejection_separately():
    annotations = (
        annotation(1, "TEAM_A_01", team=TeamLabel.TEAM_A),
        annotation(2, "TEAM_B_01", team=TeamLabel.TEAM_B),
        annotation(3, None, team=TeamLabel.UNKNOWN, status=AnnotationStatus.UNKNOWN),
    )
    evidence = {
        tracklet_id(1): TeamEvidence(TeamLabel.TEAM_A, 0.9, "jersey_color_v1"),
        tracklet_id(2): TeamEvidence(TeamLabel.UNKNOWN, 0.0, "jersey_color_v1"),
        tracklet_id(3): TeamEvidence(TeamLabel.TEAM_A, 0.6, "jersey_color_v1"),
    }

    assert team_classification_metrics(annotations, evidence) == {
        "accuracy": pytest.approx(1 / 3),
        "rejection_rate": pytest.approx(1 / 3),
        "evaluated": 3,
        "correct": 1,
        "unknown_predictions": 1,
    }


def test_identity_observation_coverage_weights_tracklet_observations():
    annotations = (
        annotation(1, "TEAM_A_01"),
        annotation(2, "TEAM_A_02"),
    )
    tracklets = make_tracklets(make_tracklet(1, (0, 1, 2)), make_tracklet(2, (3,)))
    predicted = {tracklet_id(1): PlayerIdentityId("TEAM_A_01")}

    assert identity_observation_coverage(annotations, predicted, tracklets) == {
        "coverage": 0.75,
        "correct_observations": 3,
        "annotated_observations": 4,
    }


def test_ai1_tracklet_artifacts_remain_readable_by_ai2a_inputs():
    original = make_tracklets(make_tracklet(1, (0, 1, 2)))

    decoded = tracklets_from_parquet_bytes(tracklets_to_parquet_bytes(original))

    assert decoded == original
    assert TrackletId.from_ai1_track_id(decoded.tracklets[0].track_id) == tracklet_id(1)


def test_normal_ai2a_import_does_not_load_yolo_or_ultralytics():
    code = (
        "import sys; "
        "import footballai_v2.execution.ai2a.model; "
        "import footballai_v2.execution.ai2a.association; "
        "import footballai_v2.execution.ai2a.review; "
        "assert 'ultralytics' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


def test_unsupported_tracking_identity_metrics_are_not_exposed():
    from footballai_v2.execution.ai2a.metrics import SUPPORTED_METRICS, UNSUPPORTED_METRICS

    assert {"pairwise_precision", "pairwise_recall", "pairwise_f1", "candidate_pair_recall"} <= SUPPORTED_METRICS
    assert {"HOTA", "IDF1", "ID_switches"} <= UNSUPPORTED_METRICS
    assert not SUPPORTED_METRICS & UNSUPPORTED_METRICS


def test_evaluation_artifact_marks_resolver_metrics_not_measured():
    tracklets = make_tracklets(make_tracklet(1, (0, 1)), make_tracklet(2, (2, 3)))
    ground_truth = IdentityGroundTruthArtifact(
        GROUND_TRUTH_SCHEMA,
        source(),
        (
            annotation(1, "TEAM_A_01", start=0, end=0.2),
            annotation(2, "TEAM_A_01", start=0.4, end=0.6),
        ),
    )
    teams = {
        tracklet_id(1): TeamEvidence(TeamLabel.TEAM_A, 0.8, "jersey_color_v1"),
        tracklet_id(2): TeamEvidence(TeamLabel.TEAM_A, 0.8, "jersey_color_v1"),
    }
    graph = build_candidate_graph(tracklets, teams, {}, ())

    report = build_evaluation_report(
        ground_truth,
        tracklets,
        teams,
        (),
        graph,
        team_prototype_tracklets={"TEAM_A": tracklet_id(1), "TEAM_B": tracklet_id(2)},
        annotation_effort_minutes=1,
    )

    assert report["schema"] == EVALUATION_SCHEMA
    assert report["scope"]["persistent_identity_resolver"] is False
    assert "HOTA" not in report["evaluation_metrics"]["measured"]
    assert "IDF1" not in report["evaluation_metrics"]["measured"]
    assert report["evaluation_metrics"]["not_measured"]["identity_observation_coverage"]
    assert evaluation_to_json_bytes(report).endswith(b"\n")
