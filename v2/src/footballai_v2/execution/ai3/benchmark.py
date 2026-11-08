"""Offline synthetic benchmark for AI3 (``footballai.pitch-calibration-benchmark/v1``).

Synthetic only. Each scene is a pinhole camera on the BOTTOM-touchline side
of an explicitly configured 105 m × 68 m pitch with random position, pan and
zoom; landmarks are "clicked" with Gaussian noise. Because the true camera is
known, the benchmark can measure what real data never can: the *true* error
across the visible pitch, and therefore whether the gates accept bad
calibrations.

Partitions: development seeds were used while building the pipeline;
evaluation seeds were scored once at the end. Thresholds are the
``CalibrationPolicy`` defaults, fixed before either partition was run.

The result says nothing about real broadcast footage (lens distortion, line
blur, occlusion, real click error, non-planar pitches).
"""

from __future__ import annotations

import math
import random
import time
import tracemalloc
from collections import Counter
from typing import Any, Iterable

import numpy as np

from footballai_v2.execution.ai1.artifacts import BoundingBox
from footballai_v2.execution.ai2a.model import TeamLabel, tracklet_id
from footballai_v2.execution.ai2c.artifact import AssignmentStatus
from footballai_v2.execution.ai2c.observations import PlayerObservation
from footballai_v2.execution.ai3 import homography as hg
from footballai_v2.execution.ai3.annotations import (
    PITCH_ANNOTATIONS_SCHEMA,
    AnnotationOrigin,
    AnnotationSourceInfo,
    CameraMotion,
    PitchAnnotations,
    ShotSegment,
    annotations_to_json_bytes,
)
from footballai_v2.execution.ai3.artifact import calibrate, pitch_calibration_sha256
from footballai_v2.execution.ai3.calibration import CalibrationPolicy, CalibrationStatus, calibrate_keyframe
from footballai_v2.execution.ai3.geometry import IFAB_LAW_1_MARKINGS, CameraSide, PitchGeometry
from footballai_v2.execution.ai3.projection import ProjectionStatus, project_player_observations
from footballai_v2.execution.ai3.synthetic import SyntheticCamera, broadcast_camera, field_error_m, synthetic_frame
from footballai_v2.execution.ai1.artifacts import sha256_bytes


BENCHMARK_SCHEMA = "footballai.pitch-calibration-benchmark/v1"
DEVELOPMENT_SEEDS = tuple(range(0, 40))
EVALUATION_SEEDS = tuple(range(1000, 1200))
UNSAFE_ACCEPTANCE_P95_M = 2.0
"""A calibrated frame whose true p95 field error exceeds this is an unsafe acceptance (declared up front)."""
SYNTHETIC_PITCH = PitchGeometry(105.0, 68.0, IFAB_LAW_1_MARKINGS, "synthetic benchmark configuration (explicit)")
WRONG_SIZE_PITCH = PitchGeometry(100.0, 64.0, IFAB_LAW_1_MARKINGS, "synthetic: true pitch differs from configuration")
FRAME_SIZE = (1920, 1080)

VALID_SCENARIOS = ("clean_1px", "noisy_3px", "one_fit_outlier")
INVALID_SCENARIOS = ("mirrored_left_right", "wrong_pitch_dimensions")


def _quantiles(values: Iterable[float]) -> dict[str, Any]:
    data = sorted(float(item) for item in values)
    if not data:
        return {"n": 0, "median": None, "p90": None, "max": None}
    array = np.asarray(data)
    return {
        "n": len(data),
        "median": round(float(np.median(array)), 4),
        "p90": round(float(np.quantile(array, 0.9, method="higher")), 4) if len(data) >= 10 else None,
        "max": round(data[-1], 4) if math.isfinite(data[-1]) else "inf",
    }


def _scene(seed: int, scenario: str, policy: CalibrationPolicy) -> dict[str, Any]:
    rng = random.Random(seed * 7919 + hash_scenario(scenario))
    true_pitch = WRONG_SIZE_PITCH if scenario == "wrong_pitch_dimensions" else SYNTHETIC_PITCH
    camera = broadcast_camera(true_pitch, rng)
    frame = synthetic_frame(
        true_pitch, camera, frame_index=0, timestamp_seconds=0.0, shot_id="shot", rng=rng,
        noise_px=3.0 if scenario == "noisy_3px" else 1.0,
        fit_outliers=1 if scenario == "one_fit_outlier" else 0,
        swap_left_right=scenario == "mirrored_left_right",
    )
    if scenario == "wrong_pitch_dimensions":
        configured = SYNTHETIC_PITCH.landmarks()
        frame = type(frame)(frame.frame_index, frame.timestamp_seconds, frame.shot_id, tuple(
            type(item)(
                item.landmark_id, item.pixel_x, item.pixel_y, *configured[item.landmark_id],
                item.role, item.quality, item.review_status, item.annotator,
            ) for item in frame.landmarks
        ))
    shot = ShotSegment("shot", 0, 0, CameraMotion.STATIC, CameraSide.BOTTOM_TOUCHLINE)
    keyframe = calibrate_keyframe(frame, shot, SYNTHETIC_PITCH, FRAME_SIZE, policy)
    row: dict[str, Any] = {
        "seed": seed,
        "scenario": scenario,
        "visible_landmarks": len(frame.landmarks),
        "status": keyframe.status.value,
        "reason": keyframe.reason.value if keyframe.reason else None,
    }
    if keyframe.homography is not None:
        if scenario == "wrong_pitch_dimensions":
            # The configured pitch is not the filmed one, so there is no single
            # ground-truth frame to score against; the acceptance itself is the finding.
            row["true_field_error_p95_m"] = None
        else:
            errors = field_error_m(hg.as_matrix(keyframe.homography), camera, SYNTHETIC_PITCH)
            row["true_field_error_p95_m"] = round(float(np.percentile(errors, 95)), 4)
            row["true_field_error_max_m"] = round(float(errors.max()), 4)
        row["validation_median_error_m"] = keyframe.metrics["validation_median_error_m"]
        row["validation_max_error_m"] = keyframe.metrics["validation_max_error_m"]
    return row


def hash_scenario(name: str) -> int:
    return int(sha256_bytes(name.encode())[:8], 16)


def _partition(seeds: Iterable[int], policy: CalibrationPolicy) -> dict[str, Any]:
    rows = [_scene(seed, scenario, policy) for seed in seeds for scenario in VALID_SCENARIOS + INVALID_SCENARIOS]
    scenarios = {}
    for scenario in VALID_SCENARIOS + INVALID_SCENARIOS:
        items = [row for row in rows if row["scenario"] == scenario]
        statuses = Counter(row["status"] for row in items)
        calibrated = [row for row in items if row["status"] == "calibrated"]
        summary: dict[str, Any] = {
            "frames": len(items),
            "calibrated": statuses.get("calibrated", 0),
            "uncalibrated": statuses.get("uncalibrated", 0),
            "invalidated": statuses.get("invalidated", 0),
            "calibrated_fraction": round(len(calibrated) / len(items), 4) if items else None,
            "reasons": dict(sorted(Counter(row["reason"] for row in items if row["reason"]).items())),
        }
        if scenario in VALID_SCENARIOS:
            summary["held_out_validation_error_m"] = _quantiles(
                row["validation_max_error_m"] for row in calibrated
            ) | {"statistic": "per-frame maximum held-out validation error"}
            summary["true_field_error_p95_m"] = _quantiles(row["true_field_error_p95_m"] for row in calibrated) | {
                "statistic": "per-frame 95th percentile of true error over the visible pitch (2 m grid)",
            }
            summary["unsafe_acceptances"] = sum(
                row["true_field_error_p95_m"] > UNSAFE_ACCEPTANCE_P95_M for row in calibrated
            )
        else:
            summary["rejection_rate"] = round(1 - len(calibrated) / len(items), 4) if items else None
        scenarios[scenario] = summary
    return {"seeds": [min(seeds), max(seeds)] if seeds else [], "scenes": len(rows), "scenarios": scenarios}


def _player(frame_index: int, timestamp: float, foot: tuple[float, float], tracklets_sha: str, track: int) -> PlayerObservation:
    x, y = foot
    return PlayerObservation(
        frame_index=frame_index, timestamp_seconds=timestamp, tracklet_id=tracklet_id(track),
        player_identity_id=None, identity_status=AssignmentStatus.UNRESOLVED, identity_association_score=None,
        team=TeamLabel.UNKNOWN, bbox=BoundingBox(x - 10, y - 60, x + 10, y), foot_pixel_xy=(x, y),
        tracking_confidence=0.9, pitch_xy_m=None, pitch_calibration="uncalibrated",
        source_tracklets_sha256=tracklets_sha, source_identities_sha256="0" * 64,
    )


def temporal_scenario(policy: CalibrationPolicy) -> dict[str, Any]:
    """Shots, pans, a false 'static' claim and a suspected cut, with stationary players."""
    pitch = SYNTHETIC_PITCH
    fps = 5.0
    base = SyntheticCamera(52.5, 98.0, 20.0, 40.0, 34.0, 1400.0)

    def camera_at(frame: int) -> SyntheticCamera:
        if 50 <= frame <= 99:  # genuine pan, declared moving
            return SyntheticCamera(52.5, 98.0, 20.0, 30.0 + 0.6 * (frame - 50), 34.0, 1400.0)
        if 100 <= frame <= 149:  # pans, but declared static (annotation mistake)
            return SyntheticCamera(52.5, 98.0, 20.0, 40.0 + 0.5 * (frame - 100), 34.0, 1400.0)
        return base

    shots = (
        ShotSegment("A_static", 0, 49, CameraMotion.STATIC, CameraSide.BOTTOM_TOUCHLINE),
        ShotSegment("B_pan", 50, 99, CameraMotion.MOVING, CameraSide.BOTTOM_TOUCHLINE),
        ShotSegment("C_claimed_static", 100, 149, CameraMotion.STATIC, CameraSide.BOTTOM_TOUCHLINE),
        ShotSegment("D_static_with_cut", 150, 199, CameraMotion.STATIC, CameraSide.BOTTOM_TOUCHLINE),
    )
    keyframes = {0: "A_static", 40: "A_static", 50: "B_pan", 100: "C_claimed_static", 140: "C_claimed_static",
                 150: "D_static_with_cut", 190: "D_static_with_cut"}
    rng = random.Random(20261107)
    annotated = tuple(
        synthetic_frame(pitch, camera_at(frame), frame_index=frame, timestamp_seconds=frame / fps,
                        shot_id=shot, rng=rng, noise_px=1.0)
        for frame, shot in sorted(keyframes.items())
    )
    annotations = PitchAnnotations(
        PITCH_ANNOTATIONS_SCHEMA,
        AnnotationSourceInfo("synthetic-temporal", AnnotationOrigin.SYNTHETIC, *FRAME_SIZE, None, None, "e" * 64),
        pitch, shots, annotated,
    )
    frames = [(index, index / fps) for index in range(0, 210)]
    artifact = calibrate(
        annotations, annotations_sha256=sha256_bytes(annotations_to_json_bytes(annotations)), policy=policy,
        frames=frames, suspected_cut_frames=(170,), suspected_cuts_source="synthetic scenario: declared cut at 170",
        tracklets_sha256="e" * 64,
    )
    expected: dict[str, tuple[range, set[str]]] = {
        "A keyframe pair 0..40 propagates": (range(0, 41), {"calibrated"}),
        "A after last keyframe stays uncalibrated": (range(41, 50), {"uncalibrated"}),
        "B pan: only keyframe 50 calibrated": (range(51, 100), {"uncalibrated"}),
        "C false static claim: keyframes disagree": (range(101, 140), {"invalidated"}),
        "D suspected cut between keyframes": (range(151, 190), {"invalidated"}),
        "gap after last shot": (range(200, 210), {"uncalibrated"}),
    }
    checks = {}
    for name, (frames_range, allowed) in expected.items():
        observed = Counter(artifact.frame(index).status.value for index in frames_range)
        checks[name] = {"frames": len(frames_range), "observed": dict(sorted(observed.items())),
                        "as_expected": set(observed) <= allowed}
    # Stationary players (true pitch positions fixed) seen through the moving camera.
    players = [(20.0 + 10 * i, 20.0 + 5 * (i % 3)) for i in range(6)]
    tracklets_sha = "e" * 64
    observations = []
    truth = []
    for frame in range(0, 150):
        camera = camera_at(frame)
        for index, (x, y) in enumerate(players):
            pixel = camera.project(x, y)
            if camera.in_frame(pixel):
                observations.append(_player(frame, frame / fps, pixel, tracklets_sha, index + 1))
                truth.append((x, y))
    digest = pitch_calibration_sha256(artifact)
    projected = list(project_player_observations(observations, artifact, digest, allow_synthetic=True))
    statuses = Counter(item.projection_status.value for item in projected)
    errors = [
        math.hypot(item.pitch_xy_m[0] - x, item.pitch_xy_m[1] - y)
        for item, (x, y) in zip(projected, truth) if item.projection_status is ProjectionStatus.PROJECTED
    ]
    pan_projected = sum(
        item.projection_status is ProjectionStatus.PROJECTED for item in projected if 51 <= item.observation.frame_index <= 99
    )
    return {
        "frames": len(frames),
        "frame_states": dict(artifact.summary["frames_by_status"]),
        "frame_reasons": dict(artifact.summary["frames_by_reason"]),
        "expected_state_checks": checks,
        "all_state_checks_passed": all(item["as_expected"] for item in checks.values()),
        "player_observations": len(projected),
        "projection_statuses": dict(sorted(statuses.items())),
        "projected_player_error_m": _quantiles(errors) | {
            "statistic": "true error of projected stationary-player foot points (exact pixels; synthetic)",
        },
        "projected_observations_during_pan_without_keyframe": pan_projected,
    }


def run_benchmark(policy: CalibrationPolicy | None = None, *, measure_resources: bool = True) -> dict[str, Any]:
    policy = policy or CalibrationPolicy()
    if measure_resources:
        tracemalloc.start()
    started = time.perf_counter()
    development = _partition(DEVELOPMENT_SEEDS, policy)
    evaluation = _partition(EVALUATION_SEEDS, policy)
    temporal = temporal_scenario(policy)
    elapsed = time.perf_counter() - started
    peak = None
    if measure_resources:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    keyframes = development["scenes"] + evaluation["scenes"] + 7
    return {
        "schema": BENCHMARK_SCHEMA,
        "evidence_kind": "synthetic",
        "statement": (
            "Synthetic pinhole scenes with known ground truth. Validates the mathematics and the quality gates; "
            "it does not measure calibration accuracy on real football footage."
        ),
        "policy": policy.to_dict(),
        "configuration": {
            "pitch": SYNTHETIC_PITCH.to_dict(),
            "frame_size": list(FRAME_SIZE),
            "valid_scenarios": list(VALID_SCENARIOS),
            "invalid_scenarios": list(INVALID_SCENARIOS),
            "unsafe_acceptance_true_p95_m": UNSAFE_ACCEPTANCE_P95_M,
            "camera": "bottom-touchline pinhole, 20–45 m behind the line, 12–28 m high, focal 900–2600 px, random pan",
            "validation_split": "every third visible landmark (sorted by ID, offset 1) is a held-out validation point",
        },
        "development": development,
        "evaluation": evaluation,
        "temporal": temporal,
        "real_world_evaluation": {
            "status": "pending",
            "annotated_real_frames": 0,
            "reason": "no independently annotated real pitch frames exist in the repository or session",
        },
        "unmeasured": [
            "accuracy on real broadcast frames (no annotated real frames)",
            "human click error and inter-annotator agreement",
            "lens distortion, pitch non-planarity, line blur and occlusion",
            "foot-point bias of bounding-box bottoms (feet are not the box bottom centre)",
            "automatic shot-cut detection quality (birth-burst rule is unevaluated)",
            "cross-camera or cross-stadium generalisation",
        ],
        "runtime": {
            "keyframes_calibrated_or_rejected": keyframes,
            "elapsed_seconds": round(elapsed, 2),
            "peak_traced_memory_mb": round(peak / 1e6, 1) if peak is not None else None,
            "note": "wall-clock and memory vary by machine; excluded from determinism checks",
        },
    }
