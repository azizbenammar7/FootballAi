"""AI3 pitch calibration: geometry, annotations, estimator, gates, temporal policy,
artifact integrity, AI2C projection and CLI.

All fixtures are synthetic pinhole cameras with a known ground truth. Passing
these tests proves the mathematics and the safety gates, not accuracy on real
broadcast footage.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from footballai_v2.execution.ai1.artifacts import (
    TRACKLETS_SCHEMA,
    BoundingBox,
    TrackerProvenance,
    Tracklet,
    TrackletsArtifact,
    TrackObservation,
    sha256_bytes,
    tracklets_to_parquet_bytes,
)
from footballai_v2.execution.ai2c.artifact import AssignmentStatus, player_identities_sha256
from footballai_v2.execution.ai2c.inputs import source_for_tracklets, summaries_from_tracklets
from footballai_v2.execution.ai2c.observations import PlayerObservation, iter_player_observations
from footballai_v2.execution.ai2c.resolver import resolve_identities
from footballai_v2.execution.ai3 import homography as hg
from footballai_v2.execution.ai3.annotations import (
    PITCH_ANNOTATIONS_SCHEMA,
    AnnotatedFrame,
    AnnotationOrigin,
    AnnotationQuality,
    AnnotationSourceInfo,
    CameraMotion,
    CorrespondenceRole,
    LandmarkAnnotation,
    PitchAnnotations,
    ReviewStatus,
    ShotSegment,
    annotations_from_json_bytes,
    annotations_to_json_bytes,
    check_source_bindings,
)
from footballai_v2.execution.ai3.artifact import (
    PitchCalibrationArtifact,
    calibrate,
    pitch_calibration_from_json_bytes,
    pitch_calibration_sha256,
    pitch_calibration_to_json_bytes,
)
from footballai_v2.execution.ai3.calibration import (
    CalibrationMethod,
    CalibrationPolicy,
    CalibrationStatus,
    Reason,
    Usage,
    calibrate_keyframe,
)
from footballai_v2.execution.ai3.cuts import analysed_frames, suspected_cuts_from_tracklets
from footballai_v2.execution.ai3.geometry import (
    COORDINATE_SYSTEM_DEFINITION,
    IFAB_LAW_1_MARKINGS,
    CameraSide,
    PitchGeometry,
    PitchMarkings,
)
from footballai_v2.execution.ai3.projection import (
    ProjectedPlayerObservation,
    ProjectionStatus,
    project_player_observations,
)
from footballai_v2.execution.ai3.synthetic import SyntheticCamera, field_error_m, synthetic_frame, visible_landmarks


REPOSITORY = Path(__file__).resolve().parents[2]
PITCH = PitchGeometry(105.0, 68.0, IFAB_LAW_1_MARKINGS, "test configuration")
POLICY = CalibrationPolicy()
FRAME = (1920, 1080)
CAMERA = SyntheticCamera(52.5, 98.0, 20.0, 40.0, 34.0, 1400.0)
TRACKLETS_SHA = "a" * 64
FPS = 5.0


def shot(shot_id="s", start=0, end=100, motion=CameraMotion.STATIC, side=CameraSide.BOTTOM_TOUCHLINE) -> ShotSegment:
    return ShotSegment(shot_id, start, end, motion, side)


def frame(camera=CAMERA, index=0, shot_id="s", noise=0.5, **kwargs) -> AnnotatedFrame:
    return synthetic_frame(
        PITCH, camera, frame_index=index, timestamp_seconds=index / FPS, shot_id=shot_id,
        rng=random.Random(index + 17), noise_px=noise, **kwargs,
    )


def annotations(frames, shots=None, *, origin=AnnotationOrigin.SYNTHETIC, tracklets_sha=TRACKLETS_SHA, pitch=PITCH):
    return PitchAnnotations(
        PITCH_ANNOTATIONS_SCHEMA,
        AnnotationSourceInfo("test-clip", origin, *FRAME, None, None, tracklets_sha),
        pitch, tuple(shots or (shot(),)), tuple(frames),
    )


def run(notes: PitchAnnotations, frames=(), cuts=(), policy=POLICY) -> PitchCalibrationArtifact:
    """Calibrate as if the declared tracklets file had been supplied and verified."""
    return calibrate(
        notes, annotations_sha256=sha256_bytes(annotations_to_json_bytes(notes)), policy=policy, frames=frames,
        suspected_cut_frames=cuts, suspected_cuts_source="test" if cuts else None,
        tracklets_sha256=notes.source.tracklets_sha256,
    )


def keyframe(annotated: AnnotatedFrame, segment: ShotSegment | None = None, policy=POLICY):
    return calibrate_keyframe(annotated, segment or shot(), PITCH, FRAME, policy)


def landmark(landmark_id, pixel, *, role=CorrespondenceRole.FIT, review=ReviewStatus.ACCEPTED,
             quality=AnnotationQuality.CLEAR, pitch=PITCH) -> LandmarkAnnotation:
    x, y = pitch.landmarks()[landmark_id]
    return LandmarkAnnotation(landmark_id, pixel[0], pixel[1], x, y, role, quality, review, "tester")


def with_landmarks(annotated: AnnotatedFrame, items) -> AnnotatedFrame:
    return dataclasses.replace(annotated, landmarks=tuple(items))


# --- pitch geometry -------------------------------------------------------------------


def test_pitch_dimensions_have_no_default_and_are_validated():
    with pytest.raises(TypeError):
        PitchGeometry()  # type: ignore[call-arg]
    for length, width in ((105, 0), (-105, 68), (math.nan, 68), (68, 105), (60, 68), (40, 30)):
        with pytest.raises(ValueError):
            PitchGeometry(length, width, IFAB_LAW_1_MARKINGS, "test")
    with pytest.raises(ValueError, match="dimensions_source"):
        PitchGeometry(105, 68, IFAB_LAW_1_MARKINGS, " ")


def test_origin_axes_and_core_landmarks_follow_the_plan_convention():
    points = PITCH.landmarks()
    assert points["corner_top_left"] == (0.0, 0.0)
    assert points["corner_top_right"] == (105.0, 0.0)
    assert points["corner_bottom_left"] == (0.0, 68.0)
    assert points["corner_bottom_right"] == (105.0, 68.0)
    assert points["halfway_top"] == (52.5, 0.0)
    assert points["centre_spot"] == (52.5, 34.0)
    assert points["penalty_mark_left"] == (11.0, 34.0)
    assert points["penalty_mark_right"] == (94.0, 34.0)
    assert points["penalty_area_left_top_corner"] == pytest.approx((16.5, 34.0 - 20.16))
    assert points["goal_area_right_bottom_goal_line"] == pytest.approx((105.0, 34.0 + 9.16))
    assert COORDINATE_SYSTEM_DEFINITION["origin"].startswith("corner where the LEFT goal line meets the TOP")
    assert all(PITCH.contains(x, y) for x, y in points.values())


def test_landmarks_are_mirror_symmetric_and_ordered():
    points = PITCH.landmarks()
    assert list(points) == sorted(points)
    for name, (x, y) in points.items():
        mirror = name.replace("left", "@").replace("right", "left").replace("@", "right")
        assert points[mirror] == pytest.approx((105.0 - x, y))
        flipped = name.replace("top", "@").replace("bottom", "top").replace("@", "bottom")
        assert points[flipped] == pytest.approx((x, 68.0 - y))


def test_penalty_arc_landmarks_lie_on_the_arc_and_the_area_line():
    points = PITCH.landmarks()
    mark = np.array(points["penalty_mark_left"])
    for end in ("top", "bottom"):
        point = np.array(points[f"penalty_arc_left_{end}"])
        assert point[0] == pytest.approx(16.5)
        assert np.linalg.norm(point - mark) == pytest.approx(9.15)


def test_non_standard_pitch_moves_dimension_dependent_landmarks_only():
    small = PitchGeometry(100.0, 64.0, IFAB_LAW_1_MARKINGS, "measured")
    assert small.landmarks()["corner_bottom_right"] == (100.0, 64.0)
    assert small.landmarks()["penalty_mark_right"] == (89.0, 32.0)
    assert small.within_law_1_range and not PitchGeometry(80, 50, IFAB_LAW_1_MARKINGS, "youth").within_law_1_range


def test_inconsistent_markings_are_rejected():
    values = IFAB_LAW_1_MARKINGS.to_dict()
    for change in ({"goal_area_depth_m": 20.0}, {"penalty_mark_distance_m": 17.0}, {"goal_width_m": 30.0},
                   {"penalty_arc_radius_m": 4.0}, {"centre_circle_radius_m": -1.0}):
        with pytest.raises(ValueError):
            PitchMarkings(**{**values, **change})
    with pytest.raises(ValueError, match="penalty area is wider"):
        PitchGeometry(105, 40, IFAB_LAW_1_MARKINGS, "test")


def test_geometry_round_trips_and_rejects_foreign_coordinate_systems():
    value = PITCH.to_dict()
    assert PitchGeometry.from_dict(value) == PITCH
    with pytest.raises(ValueError, match="coordinate system"):
        PitchGeometry.from_dict({**value, "coordinate_system": {**value["coordinate_system"], "origin": "centre spot"}})
    with pytest.raises(ValueError, match="within_law_1_range"):
        PitchGeometry.from_dict({**value, "within_law_1_range": False})


# --- annotation input -------------------------------------------------------------------


def test_annotations_round_trip_byte_for_byte():
    notes = annotations([frame()])
    content = annotations_to_json_bytes(notes)
    assert annotations_from_json_bytes(content) == notes
    assert annotations_to_json_bytes(annotations_from_json_bytes(content)) == content


@pytest.mark.parametrize("mutate, message", [
    (lambda f: f["landmarks"].append(dict(f["landmarks"][0])), "duplicate landmark"),
    (lambda f: f["landmarks"][0].update(landmark_id="corner_flag_post"), "unknown pitch landmark"),
    (lambda f: f["landmarks"][0].update(pitch_x_m=f["landmarks"][0]["pitch_x_m"] + 0.5), "conflict with the configured pitch"),
    (lambda f: f["landmarks"][0].update(pixel_x=1920.5), "outside the 1920"),
    (lambda f: f["landmarks"][0].update(pixel_y=-1.0), "outside the 1920"),
    (lambda f: f["landmarks"][1].update(pixel_x=f["landmarks"][0]["pixel_x"] + 0.2, pixel_y=f["landmarks"][0]["pixel_y"]),
     "at the same pixel"),
    (lambda f: f.update(frame_index=500), "not inside its shot"),
    (lambda f: f["landmarks"][0].update(role="training"), "training"),
    (lambda f: f["landmarks"][0].update(extra=1), "unknown fields"),
    (lambda f: f["landmarks"][0].update(pixel_x=float("nan")), "finite|JSON"),
])
def test_invalid_annotations_are_rejected(mutate, message):
    value = json.loads(annotations_to_json_bytes(annotations([frame()])))
    mutate(value["frames"][0])
    content = json.dumps(value).encode()
    with pytest.raises(ValueError, match=message):
        annotations_from_json_bytes(content)


def test_annotations_for_another_pitch_size_are_rejected():
    value = json.loads(annotations_to_json_bytes(annotations([frame()])))
    value["pitch"]["length_m"] = 100.0
    with pytest.raises(ValueError, match="conflict with the configured pitch geometry"):
        annotations_from_json_bytes(json.dumps(value).encode())


def test_overlapping_shots_and_unordered_frames_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        annotations([], [shot("a", 0, 10), shot("b", 10, 20)])
    with pytest.raises(ValueError, match="strictly increasing"):
        annotations([frame(index=5), frame(index=3)])


def test_source_checksum_and_frame_size_mismatches_are_rejected():
    notes = annotations([frame()])
    check_source_bindings(notes, tracklets_sha256=TRACKLETS_SHA, frame_size=FRAME)
    with pytest.raises(ValueError, match="source checksum mismatch"):
        check_source_bindings(notes, tracklets_sha256="b" * 64)
    with pytest.raises(ValueError, match="frame size mismatch"):
        check_source_bindings(notes, frame_size=(1280, 720))
    with pytest.raises(ValueError, match="source checksum mismatch"):
        calibrate(notes, annotations_sha256="c" * 64, policy=POLICY, tracklets_sha256="b" * 64)


# --- homography estimation ------------------------------------------------------------------


def ideal_points(camera=CAMERA):
    visible = visible_landmarks(PITCH, camera)
    return np.array([item[1] for item in visible]), np.array([item[2] for item in visible])


def assert_same_homography(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    if a.ravel() @ b.ravel() < 0:
        b = -b
    assert np.allclose(a, b, atol=1e-8)


def test_dlt_recovers_the_known_transform_from_ideal_correspondences():
    pixels, pitch = ideal_points()
    solution = hg.dlt(pixels, pitch)
    assert_same_homography(solution.matrix, CAMERA.image_to_pitch())
    assert solution.singular_ratio > 1e-6


def test_ransac_rejects_outliers_and_reports_them():
    pixels, pitch = ideal_points()
    corrupted = pixels.copy()
    corrupted[[2, 7]] += np.array([[140.0, -60.0], [-90.0, 120.0]])
    result = hg.ransac(corrupted, pitch, threshold_m=1.0, max_subsets=5000, seed=0, min_triangle_ratio=0.02, max_refinements=5)
    assert result is not None and result.exhaustive
    assert list(np.flatnonzero(~result.inliers)) == [2, 7]
    assert_same_homography(result.matrix, CAMERA.image_to_pitch())


def test_ransac_is_deterministic_in_exhaustive_and_sampled_modes():
    pixels, pitch = ideal_points()
    noisy = pixels + np.random.default_rng(3).normal(0, 1.0, pixels.shape)
    kwargs = dict(threshold_m=1.0, seed=11, min_triangle_ratio=0.02, max_refinements=5)
    for max_subsets in (5000, 50):
        first = hg.ransac(noisy, pitch, max_subsets=max_subsets, **kwargs)
        second = hg.ransac(noisy, pitch, max_subsets=max_subsets, **kwargs)
        assert first.exhaustive is (max_subsets == 5000)
        assert np.array_equal(first.matrix, second.matrix) and np.array_equal(first.inliers, second.inliers)


def test_collinear_and_insufficient_points_cannot_determine_a_homography():
    on_line = np.array([[0.0, 0.0], [10.0, 0.0], [52.5, 0.0], [80.0, 0.0], [105.0, 0.0]])
    pixels = np.array([[100.0, 900.0], [200.0, 880.0], [600.0, 800.0], [900.0, 760.0], [1200.0, 720.0]])
    assert not hg.has_general_position_subset(pixels, on_line, 0.02)
    assert hg.ransac(pixels, on_line, threshold_m=1, max_subsets=100, seed=0, min_triangle_ratio=0.02, max_refinements=3) is None
    assert hg.ransac(pixels[:3], on_line[:3], threshold_m=1, max_subsets=100, seed=0, min_triangle_ratio=0.02, max_refinements=3) is None
    almost = np.array([[0.0, 0.0], [50.0, 0.3], [105.0, 0.0], [70.0, 0.2]])
    assert hg.sample_is_degenerate(pixels[:4], almost, 0.02)


def test_extreme_perspective_is_still_recovered_exactly():
    grazing = SyntheticCamera(52.5, 75.0, 4.0, 30.0, 30.0, 2200.0)
    pixels, pitch = ideal_points(grazing)
    assert len(pixels) >= 5
    assert_same_homography(hg.dlt(pixels, pitch).matrix, grazing.image_to_pitch())


def test_orientation_is_preserved_for_real_cameras_and_reversed_by_mirroring():
    pixels, pitch = ideal_points()
    assert np.all(hg.jacobian_determinant(CAMERA.image_to_pitch(), pixels) > 0)
    mirrored = pitch.copy()
    mirrored[:, 0] = 105.0 - mirrored[:, 0]
    assert np.all(hg.jacobian_determinant(hg.dlt(pixels, mirrored).matrix, pixels) < 0)


def test_estimator_agrees_with_opencv_when_available():
    cv2 = pytest.importorskip("cv2")
    pixels, pitch = ideal_points()
    noisy = pixels + np.random.default_rng(5).normal(0, 0.5, pixels.shape)
    reference, _ = cv2.findHomography(noisy, pitch, 0)
    ours = hg.dlt(noisy, pitch).matrix
    projected_ours, _ = hg.apply(ours, noisy)
    projected_cv, _ = hg.apply(reference, noisy)
    assert np.max(np.linalg.norm(projected_ours - projected_cv, axis=1)) < 0.05


# --- keyframe quality gates ------------------------------------------------------------------


def test_clean_keyframe_is_calibrated_with_independent_validation():
    result = keyframe(frame())
    assert result.status is CalibrationStatus.CALIBRATED and result.reason is None
    metrics = result.metrics
    assert metrics["validation_points"] >= 2 and metrics["fit_points"] >= 4
    assert metrics["validation_max_error_m"] <= POLICY.max_validation_error_m
    assert all(value in ("pass", "not_checked") for value in metrics["checks"].values())
    validation = [item for item in result.correspondences if item.usage is Usage.VALIDATION]
    assert all(item.ransac_inlier is None and item.error_m is not None for item in validation)
    errors = field_error_m(hg.as_matrix(result.homography), CAMERA, PITCH)
    assert np.percentile(errors, 95) < 0.5


def test_validation_points_never_influence_the_fit():
    base = frame()
    moved = with_landmarks(base, [
        dataclasses.replace(item, pixel_x=min(item.pixel_x + 25, 1920)) if item.role is CorrespondenceRole.VALIDATION else item
        for item in base.landmarks
    ])
    original, shifted = keyframe(base), keyframe(moved)
    fit_residuals = [
        [(item.landmark_id, item.error_m, item.ransac_inlier) for item in result.correspondences if item.usage is Usage.FIT]
        for result in (original, shifted)
    ]
    assert fit_residuals[0] == fit_residuals[1]  # identical fit ⇒ identical homography
    assert original.status is CalibrationStatus.CALIBRATED
    assert shifted.homography is None and shifted.status is CalibrationStatus.INVALIDATED
    assert shifted.reason is Reason.VALIDATION_ERROR_EXCEEDED


def test_fit_outlier_is_excluded_by_ransac_and_reported():
    result = keyframe(frame(fit_outliers=1))
    assert result.status is CalibrationStatus.CALIBRATED
    assert result.metrics["fit_outliers"] == 1
    assert sum(item.ransac_inlier is False for item in result.correspondences) == 1


def test_too_few_fit_or_validation_points_leave_the_frame_uncalibrated():
    base = frame()
    fit = [item for item in base.landmarks if item.role is CorrespondenceRole.FIT]
    validation = [item for item in base.landmarks if item.role is CorrespondenceRole.VALIDATION]
    few_fit = keyframe(with_landmarks(base, fit[:3] + validation))
    assert (few_fit.status, few_fit.reason) == (CalibrationStatus.UNCALIBRATED, Reason.INSUFFICIENT_FIT_POINTS)
    no_validation = keyframe(with_landmarks(base, fit + validation[:1]))
    assert (no_validation.status, no_validation.reason) == (CalibrationStatus.UNCALIBRATED, Reason.INSUFFICIENT_VALIDATION_POINTS)
    assert few_fit.homography is None and no_validation.homography is None


def test_unreviewed_and_unclear_points_are_excluded():
    base = frame()
    pending = keyframe(with_landmarks(base, [dataclasses.replace(item, review_status=ReviewStatus.PENDING) for item in base.landmarks]))
    assert pending.reason is Reason.INSUFFICIENT_FIT_POINTS
    assert {item.exclusion for item in pending.correspondences} == {"review_pending"}
    estimated = keyframe(with_landmarks(base, [
        dataclasses.replace(item, quality=AnnotationQuality.ESTIMATED) if item.role is CorrespondenceRole.VALIDATION else item
        for item in base.landmarks
    ]))
    assert estimated.reason is Reason.INSUFFICIENT_VALIDATION_POINTS


def test_collinear_annotation_layout_is_uncalibrated():
    project = CAMERA.project
    on_lines = ["corner_top_left", "penalty_area_left_top_goal_line", "goal_area_left_top_goal_line",
                "goal_area_left_bottom_goal_line", "halfway_top"]
    items = [landmark(name, project(*PITCH.landmarks()[name])) for name in on_lines[:4]]
    items += [landmark(name, project(*PITCH.landmarks()[name]), role=CorrespondenceRole.VALIDATION)
              for name in ("centre_spot", "penalty_mark_left")]
    result = keyframe(with_landmarks(frame(), items))
    assert (result.status, result.reason) == (CalibrationStatus.UNCALIBRATED, Reason.DEGENERATE_GEOMETRY)


def test_mirrored_annotation_is_invalidated_as_a_reflection():
    result = keyframe(frame(swap_left_right=True))
    assert (result.status, result.reason) == (CalibrationStatus.INVALIDATED, Reason.REFLECTED_ORIENTATION)
    assert result.homography is None


def test_declared_camera_side_must_match_the_geometry():
    result = keyframe(frame(), shot(side=CameraSide.TOP_TOUCHLINE))
    assert (result.status, result.reason) == (CalibrationStatus.INVALIDATED, Reason.CAMERA_SIDE_INCONSISTENT)
    opposite = SyntheticCamera(52.5, -30.0, 20.0, 60.0, 34.0, 1400.0)
    assert keyframe(frame(opposite), shot(side=CameraSide.TOP_TOUCHLINE)).status is CalibrationStatus.CALIBRATED
    assert keyframe(frame(), shot(side=CameraSide.UNKNOWN)).metrics["checks"]["camera_side_consistent"] == "not_checked"


def test_policy_is_versioned_validated_and_round_trips():
    assert CalibrationPolicy.from_dict(POLICY.to_dict()) == POLICY
    for change in ({"min_fit_points": 3}, {"max_validation_error_m": 0.1}, {"accepted_review_statuses": ("rejected",)},
                   {"version": "ai3-calibration-policy/v0"}, {"ransac_threshold_m": math.inf}):
        with pytest.raises(ValueError):
            dataclasses.replace(POLICY, **change)


# --- camera cuts and temporal validity ------------------------------------------------------------


def frames_between(start, end):
    return [(index, index / FPS) for index in range(start, end + 1)]


def test_static_shot_propagates_only_between_two_agreeing_keyframes():
    artifact = run(annotations([frame(index=10), frame(index=30)]), frames_between(0, 50))
    states = {item.frame_index: item for item in artifact.frames}
    assert states[10].method is CalibrationMethod.MANUAL_LANDMARKS
    assert states[19].method is CalibrationMethod.STATIC_SHOT_PROPAGATION and states[19].source_keyframe == 10
    assert states[21].source_keyframe == 30 and states[20].source_keyframe == 10
    for index in (*range(0, 10), *range(31, 51)):
        assert states[index].status is CalibrationStatus.UNCALIBRATED
        assert states[index].reason is Reason.OUTSIDE_VERIFIED_INTERVAL and states[index].homography is None


def test_moving_camera_never_propagates():
    notes = annotations([frame(index=10), frame(index=30)], [shot(motion=CameraMotion.MOVING)])
    artifact = run(notes, frames_between(0, 50))
    calibrated = [item.frame_index for item in artifact.frames if item.status is CalibrationStatus.CALIBRATED]
    assert calibrated == [10, 30]
    assert artifact.frame(20).reason is Reason.CAMERA_NOT_DECLARED_STATIC


def test_disagreeing_static_keyframes_invalidate_the_interval():
    panned = SyntheticCamera(52.5, 98.0, 20.0, 55.0, 34.0, 1400.0)
    artifact = run(annotations([frame(index=10), frame(panned, index=30)]), frames_between(0, 50))
    assert artifact.frame(20).status is CalibrationStatus.INVALIDATED
    assert artifact.frame(20).reason is Reason.KEYFRAMES_DISAGREE
    assert artifact.frame(10).status is artifact.frame(30).status is CalibrationStatus.CALIBRATED


def test_suspected_cut_invalidates_propagation_across_it():
    artifact = run(annotations([frame(index=10), frame(index=30)]), frames_between(0, 50), cuts=(25,))
    assert {artifact.frame(index).reason for index in range(11, 30)} == {Reason.SUSPECTED_CAMERA_CUT}
    assert artifact.frame(10).status is CalibrationStatus.CALIBRATED


def test_shot_boundary_stops_calibration_and_gaps_stay_uncalibrated():
    shots = [shot("a", 0, 19), shot("b", 25, 60)]
    artifact = run(annotations([frame(index=10, shot_id="a"), frame(index=19, shot_id="a")], shots), frames_between(0, 40))
    assert artifact.frame(22).reason is Reason.NO_SHOT_SEGMENT
    assert artifact.frame(30).reason is Reason.OUTSIDE_VERIFIED_INTERVAL
    assert artifact.summary["frames_by_status"]["calibrated"] == 10


def test_disabling_propagation_keeps_only_exact_keyframes():
    policy = dataclasses.replace(POLICY, propagate_static_shots=False)
    artifact = run(annotations([frame(index=10), frame(index=30)]), frames_between(0, 50), policy=policy)
    assert [item.frame_index for item in artifact.frames if item.status is CalibrationStatus.CALIBRATED] == [10, 30]
    assert artifact.frame(20).reason is Reason.NO_KEYFRAME_FOR_FRAME


def test_uncalibrated_frames_never_fall_back_to_a_pixel_scale():
    artifact = run(annotations([frame(index=10), frame(index=12, fit_outliers=0, swap_left_right=True)]), frames_between(0, 20))
    for item in artifact.frames:
        assert (item.homography is not None) == (item.status is CalibrationStatus.CALIBRATED)
    serialized = json.loads(pitch_calibration_to_json_bytes(artifact))
    assert all(entry["homography_image_to_pitch"] is None for entry in serialized["frames"] if entry["status"] != "calibrated")
    assert "metres_per_pixel_scale" not in json.dumps(serialized)


def test_birth_burst_suspected_cuts_from_tracklets():
    def track(track_id, start):
        return Tracklet(track_id, 0, "person", (TrackObservation(start, start / FPS, BoundingBox(10, 10, 20, 40), 0.9),))

    tracklets = TrackletsArtifact(
        TRACKLETS_SCHEMA, TrackerProvenance("t", "1", {}, "d", "d" * 64, "r"),
        tuple(track(i, 0) for i in range(1, 10)) + tuple(track(i, 40) for i in range(10, 19)) + (track(30, 50),),
    )
    assert suspected_cuts_from_tracklets(tracklets) == (40,)
    assert suspected_cuts_from_tracklets(tracklets, min_births=20) == ()


def test_annotated_timestamp_must_match_the_analysed_frame():
    with pytest.raises(ValueError, match="frame conventions differ"):
        run(annotations([frame(index=10)]), [(10, 9.99)])


# --- artifact contract and integrity ------------------------------------------------------------


def sample_artifact():
    return run(annotations([frame(index=10), frame(index=30), frame(index=40, swap_left_right=True)]), frames_between(0, 45), cuts=(42,))


def test_artifact_round_trips_and_is_deterministic():
    first, second = sample_artifact(), sample_artifact()
    content = pitch_calibration_to_json_bytes(first)
    assert content == pitch_calibration_to_json_bytes(second)
    loaded = pitch_calibration_from_json_bytes(content)
    assert pitch_calibration_to_json_bytes(loaded) == content
    value = json.loads(content)
    assert value["schema"] == "footballai.pitch-calibration/v1"
    assert value["pipeline"]["software"]["numpy"]
    assert value["source"]["annotations_sha256"] and value["source"]["tracklets_sha256"] == TRACKLETS_SHA
    assert value["pitch"]["coordinate_system"]["name"] == "footballai.pitch-plan/v1"
    assert value["summary"]["frames_by_status"] == {"calibrated": 21, "uncalibrated": 24, "invalidated": 1}


def reseal(value):
    from footballai_v2.execution.ai3.artifact import _canonical

    body = {key: item for key, item in value.items() if key != "integrity"}
    value["integrity"]["content_sha256"] = sha256_bytes(_canonical(body))
    return json.dumps(value).encode()


@pytest.mark.parametrize("tamper", [
    lambda v: v["keyframes"][0]["homography_image_to_pitch"].__setitem__(2, v["keyframes"][0]["homography_image_to_pitch"][2] + 0.01),
    lambda v: v["frames"][25].update(homography_image_to_pitch=v["frames"][10]["homography_image_to_pitch"]),
    lambda v: v["frames"][0].update(status="calibrated", reason=None, method="manual_landmark_homography",
                                    source_keyframe=10, homography_image_to_pitch=v["frames"][10]["homography_image_to_pitch"]),
    lambda v: v["keyframes"][2].update(status="calibrated", reason=None),
    lambda v: v["keyframes"][0]["correspondences"][0].update(pixel_xy=[5.0, 5.0]),
    lambda v: v["policy"].update(min_validation_points=50),
    lambda v: v["summary"].update(calibrated_fraction=1.0),
    lambda v: v.update(suspected_cut_frames=[20, 42]),
])
def test_tampered_artifacts_fail_re_derivation_even_with_a_valid_seal(tamper):
    value = json.loads(pitch_calibration_to_json_bytes(sample_artifact()))
    tamper(value)
    with pytest.raises(ValueError):
        pitch_calibration_from_json_bytes(reseal(value))


def test_loading_keeps_recorded_values_and_reserialises_identically():
    content = pitch_calibration_to_json_bytes(sample_artifact())
    value = json.loads(content)
    entry = value["keyframes"][0]["homography_image_to_pitch"]
    entry[0] = entry[0] * (1 + 1e-12)  # last-digit noise, as from another BLAS
    for item in value["frames"]:
        if item["source_keyframe"] == value["keyframes"][0]["frame_index"]:
            item["homography_image_to_pitch"] = entry
    sealed = reseal(value)
    loaded = pitch_calibration_from_json_bytes(sealed)
    assert pitch_calibration_to_json_bytes(loaded) == (json.dumps(json.loads(sealed), indent=2, sort_keys=True) + "\n").encode()
    assert loaded.keyframes[0].homography[0] == entry[0]


@pytest.mark.parametrize("tamper", [
    lambda v: v["keyframes"][0].pop("shot_id"),
    lambda v: v["keyframes"][0]["correspondences"][0].update(pixel_xy=[1.0]),
    lambda v: v["frames"][0].update(frame_index="0"),
    lambda v: v.update(shots={}),
    lambda v: v["keyframes"][0].update(metrics=[]),
])
def test_malformed_artifacts_raise_value_error(tamper):
    value = json.loads(pitch_calibration_to_json_bytes(sample_artifact()))
    tamper(value)
    with pytest.raises(ValueError):
        pitch_calibration_from_json_bytes(reseal(value))


def test_declared_but_unverified_tracklets_binding_is_not_recorded():
    notes = annotations([frame(index=10)])
    artifact = calibrate(notes, annotations_sha256="c" * 64, policy=POLICY)
    assert notes.source.tracklets_sha256 == TRACKLETS_SHA
    assert artifact.source.tracklets_sha256 is None


def test_broken_integrity_seal_is_rejected():
    value = json.loads(pitch_calibration_to_json_bytes(sample_artifact()))
    value["limitations"].append("edited")
    with pytest.raises(ValueError, match="integrity"):
        pitch_calibration_from_json_bytes(json.dumps(value).encode())


def test_forged_in_memory_artifact_is_rejected():
    artifact = sample_artifact()
    forged_frames = tuple(
        dataclasses.replace(item, status=CalibrationStatus.CALIBRATED, method=CalibrationMethod.MANUAL_LANDMARKS,
                            reason=None, source_keyframe=10, homography=artifact.frame(10).homography)
        if item.frame_index == 5 else item
        for item in artifact.frames
    )
    with pytest.raises(ValueError):
        dataclasses.replace(artifact, frames=forged_frames)


# --- projection and AI2C compatibility ------------------------------------------------------------


PLAYERS = [(20.0, 20.0), (35.0, 50.0), (45.0, 30.0)]


def player_tracklets(frames, camera=CAMERA):
    tracklets = []
    for track_id, (x, y) in enumerate(PLAYERS, start=1):
        u, v = camera.project(x, y)
        tracklets.append(Tracklet(track_id, 0, "person", tuple(
            TrackObservation(index, index / FPS, BoundingBox(u - 12, v - 70, u + 12, v), 0.9) for index in frames
        )))
    return TrackletsArtifact(TRACKLETS_SCHEMA, TrackerProvenance("ByteTrack", "test", {}, "det", "d" * 64, "run"), tuple(tracklets))


def ai2c_observations(tracklets, content):
    digest = sha256_bytes(content)
    identities = resolve_identities(
        summaries_from_tracklets(tracklets), source_for_tracklets(content, tracklets, source_id="test-clip"),
    )
    return list(iter_player_observations(
        tracklets, digest, identities, player_identities_sha256(identities), include_provisional=True, include_unresolved=True,
    )), digest


def test_validated_calibration_projects_ai2c_observations_and_preserves_identity():
    tracklets = player_tracklets(range(0, 41))
    content = tracklets_to_parquet_bytes(tracklets)
    observations, digest = ai2c_observations(tracklets, content)
    notes = annotations([frame(index=10), frame(index=30)], tracklets_sha=digest)
    artifact = run(notes, analysed_frames(tracklets))
    projected = list(project_player_observations(observations, artifact, pitch_calibration_sha256(artifact), allow_synthetic=True))

    assert len(projected) == len(observations)
    by_status = {}
    for item in projected:
        by_status.setdefault(item.projection_status, []).append(item)
        assert item.observation.pitch_xy_m is None and item.observation.pitch_calibration == "uncalibrated"
        assert item.observation.identity_status is AssignmentStatus.UNRESOLVED
        assert item.observation.player_identity_id is None
    calibrated_frames = {item.frame_index for item in artifact.frames if item.status is CalibrationStatus.CALIBRATED}
    assert {item.observation.frame_index for item in by_status[ProjectionStatus.PROJECTED]} == calibrated_frames
    for item in by_status[ProjectionStatus.PROJECTED]:
        truth = PLAYERS[int(item.observation.tracklet_id.value[-6:]) - 1]
        assert math.dist(item.pitch_xy_m, truth) < 0.5
        assert item.inside_pitch_lines is True and item.source_calibration_sha256 == pitch_calibration_sha256(artifact)
        assert item.observation.foot_pixel_xy == (item.observation.bbox.center_x, item.observation.bbox.y2)
    assert all(item.pitch_xy_m is None for item in by_status[ProjectionStatus.FRAME_UNCALIBRATED])


def test_ai2c_observation_guard_is_unchanged():
    tracklets = player_tracklets(range(0, 3))
    observations, _ = ai2c_observations(tracklets, tracklets_to_parquet_bytes(tracklets))
    with pytest.raises(ValueError, match="validated calibration"):
        dataclasses.replace(observations[0], pitch_xy_m=(1.0, 2.0))
    with pytest.raises(ValueError, match="validated calibration"):
        dataclasses.replace(observations[0], pitch_calibration="calibrated")


def projection_fixture(frames=range(0, 41), camera=CAMERA, keyframes=(10, 30), motion=CameraMotion.STATIC):
    tracklets = player_tracklets(frames, camera)
    content = tracklets_to_parquet_bytes(tracklets)
    observations, digest = ai2c_observations(tracklets, content)
    notes = annotations([frame(index=index) for index in keyframes], [shot(motion=motion)], tracklets_sha=digest)
    artifact = run(notes, analysed_frames(tracklets))
    return observations, artifact, pitch_calibration_sha256(artifact)


def test_projected_observations_cannot_carry_injected_coordinates():
    observations, artifact, digest = projection_fixture()
    good = next(item for item in project_player_observations(observations, artifact, digest, allow_synthetic=True)
                if item.projection_status is ProjectionStatus.PROJECTED)
    with pytest.raises(ValueError, match="homography projection"):
        dataclasses.replace(good, pitch_xy_m=(good.pitch_xy_m[0] + 3.0, good.pitch_xy_m[1]))
    with pytest.raises(ValueError, match="calibrated frame"):
        dataclasses.replace(good, calibration_status=CalibrationStatus.UNCALIBRATED)
    with pytest.raises(ValueError, match="only a projected observation"):
        dataclasses.replace(good, projection_status=ProjectionStatus.FRAME_UNCALIBRATED)
    with pytest.raises(ValueError, match="AI2C PlayerObservation"):
        dataclasses.replace(good, observation=object())
    with pytest.raises(ValueError, match="checksum"):
        dataclasses.replace(good, source_calibration_sha256="0" * 64)
    # A self-consistent forgery: a different homography that maps the foot point
    # exactly where the forger wants, with matching pitch_xy_m.
    forged = np.asarray(good.homography, dtype=float).reshape(3, 3).copy()
    forged[0, 2] += 0.001
    forged_matrix = tuple(float(value) for value in forged.ravel())
    point, _ = hg.apply(forged, np.asarray([good.observation.foot_pixel_xy]))
    with pytest.raises(ValueError, match="not the calibration's homography"):
        dataclasses.replace(good, homography=forged_matrix, pitch_xy_m=(float(point[0, 0]), float(point[0, 1])))


def test_projection_refuses_mismatched_or_unsafe_inputs():
    observations, artifact, digest = projection_fixture()
    with pytest.raises(ValueError, match="calibration_sha256"):
        list(project_player_observations(observations, artifact, "f" * 64, allow_synthetic=True))
    with pytest.raises(ValueError, match="synthetic"):
        list(project_player_observations(observations, artifact, digest))
    other = dataclasses.replace(observations[0], source_tracklets_sha256="9" * 64)
    with pytest.raises(ValueError, match="different AI1 tracklets"):
        list(project_player_observations([other], artifact, digest, allow_synthetic=True))
    shifted = dataclasses.replace(observations[0], timestamp_seconds=observations[0].timestamp_seconds + 0.1)
    with pytest.raises(ValueError, match="frame conventions"):
        list(project_player_observations([shifted], artifact, digest, allow_synthetic=True))
    relaxed = dataclasses.replace(POLICY, max_validation_median_error_m=50.0, max_validation_error_m=100.0)
    lenient = run(annotations([frame(index=10)], tracklets_sha=artifact.source.tracklets_sha256), policy=relaxed)
    with pytest.raises(ValueError, match="different policy"):
        list(project_player_observations(observations, lenient, pitch_calibration_sha256(lenient), allow_synthetic=True))
    assert list(project_player_observations(
        observations[:1], lenient, pitch_calibration_sha256(lenient), expected_policy=relaxed, allow_synthetic=True,
    ))
    unbound = run(annotations([frame(index=10)], tracklets_sha=None))
    with pytest.raises(ValueError, match="not bound"):
        list(project_player_observations(observations, unbound, pitch_calibration_sha256(unbound), allow_synthetic=True))


def test_projection_reports_out_of_image_out_of_pitch_and_unknown_frames():
    observations, artifact, digest = projection_fixture()
    at_keyframe = next(item for item in observations if item.frame_index == 10)
    outside_image = dataclasses.replace(at_keyframe, foot_pixel_xy=(2500.0, 500.0))
    horizon = CAMERA.project(52.5, 250.0) or (960.0, 5.0)
    far = dataclasses.replace(at_keyframe, foot_pixel_xy=(960.0, 1.0))
    unknown = dataclasses.replace(at_keyframe, frame_index=999, timestamp_seconds=999 / FPS)
    results = list(project_player_observations([outside_image, far, unknown], artifact, digest, allow_synthetic=True))
    assert results[0].projection_status is ProjectionStatus.FOOT_POINT_OUTSIDE_IMAGE
    assert results[1].projection_status in (ProjectionStatus.OUTSIDE_PITCH_MARGIN, ProjectionStatus.BEYOND_HORIZON)
    assert results[2].projection_status is ProjectionStatus.FRAME_NOT_IN_CALIBRATION
    assert all(item.pitch_xy_m is None for item in results)
    assert horizon is not None


def test_camera_pan_without_keyframes_invents_no_movement():
    panning = [SyntheticCamera(52.5, 98.0, 20.0, 30.0 + index, 34.0, 1400.0) for index in range(21)]
    tracklets = TrackletsArtifact(TRACKLETS_SCHEMA, TrackerProvenance("t", "1", {}, "d", "d" * 64, "r"), (
        Tracklet(1, 0, "person", tuple(
            TrackObservation(index, index / FPS, BoundingBox(u - 12, v - 70, u + 12, v), 0.9)
            for index, (u, v) in ((index, panning[index].project(30.0, 30.0)) for index in range(21))
        )),
    ))
    content = tracklets_to_parquet_bytes(tracklets)
    observations, digest = ai2c_observations(tracklets, content)
    notes = annotations([frame(panning[0], index=0)], [shot(motion=CameraMotion.MOVING)], tracklets_sha=digest)
    artifact = run(notes, analysed_frames(tracklets))
    projected = list(project_player_observations(observations, artifact, pitch_calibration_sha256(artifact), allow_synthetic=True))
    positions = [item.pitch_xy_m for item in projected if item.pitch_xy_m is not None]
    assert len(positions) == 1 and math.dist(positions[0], (30.0, 30.0)) < 0.5
    pixels = [item.observation.foot_pixel_xy[0] for item in projected]
    assert max(pixels) - min(pixels) > 100  # the player moved in the image, not on the pitch


# --- CLI, benchmark and isolation ------------------------------------------------------------


def test_cli_validates_and_calibrates_and_refuses_to_overwrite(tmp_path, capsys):
    from footballai_v2.execution.ai3.cli import main

    tracklets = player_tracklets(range(0, 41))
    tracklets_path = tmp_path / "tracklets.parquet"
    tracklets_path.write_bytes(tracklets_to_parquet_bytes(tracklets))
    digest = sha256_bytes(tracklets_path.read_bytes())
    notes_path = tmp_path / "annotations.json"
    notes_path.write_bytes(annotations_to_json_bytes(annotations([frame(index=10), frame(index=30)], tracklets_sha=digest)))
    assert main(["validate", "--annotations", str(notes_path), "--tracklets", str(tracklets_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [item["status"] for item in report["keyframes"]] == ["calibrated", "calibrated"]
    output = tmp_path / "calibration.json"
    args = ["calibrate", "--annotations", str(notes_path), "--tracklets", str(tracklets_path),
            "--suspected-cuts-from-tracklets", "--output", str(output)]
    assert main(args) == 0
    artifact = pitch_calibration_from_json_bytes(output.read_bytes())
    assert artifact.source.annotations_sha256 == sha256_bytes(notes_path.read_bytes())
    assert artifact.source.suspected_cuts_source == "tracklet_birth_burst/v1:min_births=8"
    assert artifact.summary["frames_total"] == 41
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(args)
    wrong = tmp_path / "other.parquet"
    wrong.write_bytes(tracklets_to_parquet_bytes(player_tracklets(range(0, 5))))
    with pytest.raises(ValueError, match="source checksum mismatch"):
        main(["calibrate", "--annotations", str(notes_path), "--tracklets", str(wrong), "--output", str(tmp_path / "x.json")])


def test_cli_template_and_landmarks(tmp_path, capsys):
    from footballai_v2.execution.ai3.cli import main

    pitch_args = ["--length-m", "100", "--width-m", "64", "--markings", "ifab-law-1", "--dimensions-source", "measured"]
    svg = tmp_path / "plan.svg"
    assert main(["landmarks", *pitch_args, "--svg", str(svg)]) == 0
    table = json.loads(capsys.readouterr().out)
    assert len(table["landmarks"]) == 31 and svg.read_text().startswith("<svg")
    output = tmp_path / "template.json"
    assert main(["template", *pitch_args, "--source-id", "clip", "--frame-width", "1920", "--frame-height", "1080",
                 "--shot", "s1:0:99:static:bottom_touchline", "--keyframe", "9:1.8", "--output", str(output)]) == 0
    template = annotations_from_json_bytes(output.read_bytes())
    assert template.source.origin is AnnotationOrigin.MANUAL and template.frames[0].landmarks == ()
    with pytest.raises(SystemExit):
        main(["landmarks", "--length-m", "105", "--width-m", "68", "--markings", "ifab-law-1"])


def test_cli_refuses_to_write_frames_inside_the_repository():
    from footballai_v2.execution.ai3.cli import main

    with pytest.raises(SystemExit, match="inside a Git working tree"):
        main(["annotate-frame", "--length-m", "105", "--width-m", "68", "--markings", "ifab-law-1",
              "--dimensions-source", "x", "--video", "missing.mp4", "--frame-index", "0", "--timestamp", "0",
              "--shot-id", "s", "--output-dir", str(REPOSITORY / "frames")])


def test_annotation_page_is_self_contained_and_escapes_script_breakouts():
    from footballai_v2.execution.ai3.annotation_page import annotation_page_html

    page = annotation_page_html(image_png=b"\x89PNG", frame_index=3, timestamp_seconds=0.6,
                                shot_id="</script><script>alert(1)</script>", frame_image_sha256="0" * 64, geometry=PITCH)
    assert page.count("</script>") == 2
    assert "http://" not in page and "https://" not in page
    assert '"review_status": "pending"' not in page and 'review_status: "pending"' in page


def test_temporal_benchmark_scenario_meets_every_expectation():
    from footballai_v2.execution.ai3.benchmark import temporal_scenario

    result = temporal_scenario(POLICY)
    assert result["all_state_checks_passed"]
    assert result["projected_observations_during_pan_without_keyframe"] == 0
    assert result["projected_player_error_m"]["max"] < 1.0


def test_committed_benchmark_is_synthetic_consistent_and_honest():
    value = json.loads((REPOSITORY / "evaluation/ai3/synthetic-benchmark.json").read_bytes())
    assert value["schema"] == "footballai.pitch-calibration-benchmark/v1"
    assert value["evidence_kind"] == "synthetic"
    assert value["real_world_evaluation"] == {
        "status": "pending", "annotated_real_frames": 0,
        "reason": "no independently annotated real pitch frames exist in the repository or session",
    }
    assert value["policy"] == POLICY.to_dict()
    assert value["temporal"]["all_state_checks_passed"]
    for partition in ("development", "evaluation"):
        for scenario, summary in value[partition]["scenarios"].items():
            assert summary["calibrated"] + summary["uncalibrated"] + summary["invalidated"] == summary["frames"]
    assert value["evaluation"]["scenarios"]["mirrored_left_right"]["calibrated"] == 0


def test_small_benchmark_partition_is_reproducible():
    from footballai_v2.execution.ai3.benchmark import _partition

    assert _partition((0, 1), POLICY) == _partition((0, 1), POLICY)


def test_ai3_import_does_not_load_detectors_or_opencv():
    code = (
        "import sys; import footballai_v2.execution.ai3.cli; "
        "print(sorted(m for m in ('torch','ultralytics','cv2') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                            env={"PYTHONPATH": str(REPOSITORY / "v2/src")})
    assert result.stdout.strip() == "[]"
