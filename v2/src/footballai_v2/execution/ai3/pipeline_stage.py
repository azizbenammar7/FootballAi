"""AI3-B: the ``pitch_calibration`` run stage over verified AI1 artifacts.

The asynchronous worker calls these functions after YOLO/ByteTrack (AI1) has
produced ``detections.parquet`` and ``tracklets.parquet`` for a run::

    verified AI1 artifacts (checksums, video binding, frame/time alignment)
      -> identity_resolution: AI2C in review_only mode (no automatic association;
         every player tracklet stays UNKNOWN unless a human confirms it)
      -> pitch_calibration: AI3 homographies from the landmark annotations
         attached to the run, a state for every AI1 sampled frame
      -> foot-point projection on CALIBRATED frames only
      -> versioned artifacts:
           footballai.player-identities/v1               (AI2C, JSON)
           footballai.pitch-calibration/v1               (AI3, JSON; only with annotations)
           footballai.projected-player-observations/v1   (Parquet; only with annotations)
           footballai.pitch-space-diagnostic/v1          (JSON; always)

Nothing here estimates a homography without manual landmark evidence, and
nothing falls back to a fixed pixel→metre scale. A run without annotations
publishes a diagnostic in which every analysed frame is ``uncalibrated``
(``no_landmark_annotations``) and no pitch coordinate exists.

Landmark annotations are attached to a run as a parameter
(:func:`pitch_annotations_parameter`). They must declare the SHA-256 of the
uploaded video, so the AI1 frame indices and timestamps they reference belong
to that video. The canonical bytes are pinned by their own SHA-256 and
re-verified by the worker before calibration.

All outputs are deterministic functions of their inputs: no wall-clock time,
no randomness beyond AI3's seeded estimator. A redelivered job therefore
republishes identical bytes.
"""

from __future__ import annotations

import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from footballai_v2.execution.ai1.artifacts import (
    DetectionsArtifact,
    TrackletsArtifact,
    artifact_metadata,
    detections_from_parquet_bytes,
    read_artifact_metadata,
    sha256_bytes,
    tracklets_from_parquet_bytes,
)
from footballai_v2.execution.ai2c.artifact import (
    AssignmentStatus,
    PlayerIdentitiesArtifact,
    player_identities_to_json_bytes,
)
from footballai_v2.execution.ai2c.inputs import source_for_tracklets, summaries_from_tracklets
from footballai_v2.execution.ai2c.observations import iter_player_observations
from footballai_v2.execution.ai2c.resolver import ResolverConfig, ResolverMode, resolve_identities
from footballai_v2.execution.ai3.annotations import (
    PITCH_ANNOTATIONS_SCHEMA,
    AnnotationOrigin,
    PitchAnnotations,
    annotations_from_json_bytes,
    annotations_to_json_bytes,
    check_source_bindings,
)
from footballai_v2.execution.ai3.artifact import (
    PitchCalibrationArtifact,
    calibrate,
    pitch_calibration_to_json_bytes,
)
from footballai_v2.execution.ai3.calibration import CALIBRATION_POLICY_VERSION, CalibrationPolicy, CalibrationStatus
from footballai_v2.execution.ai3.cuts import birth_burst_source, suspected_cuts_from_tracklets
from footballai_v2.execution.ai3.geometry import COORDINATE_SYSTEM_DEFINITION
from footballai_v2.execution.ai3.projection import (
    DEFAULT_PITCH_MARGIN_M,
    PROJECTED_OBSERVATION_CONTRACT,
    ProjectionStatus,
    project_player_observations,
)


STAGE_NAME = "footballai.ai3b.pitch-calibration-stage"
STAGE_VERSION = "1.0.0"
PITCH_ANNOTATIONS_PARAMETER = "pitch_annotations"
MAX_PITCH_ANNOTATIONS_BYTES = 1024 * 1024
PROJECTED_OBSERVATIONS_SCHEMA = "footballai.projected-player-observations/v1"
PITCH_SPACE_DIAGNOSTIC_SCHEMA = "footballai.pitch-space-diagnostic/v1"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
SUSPECTED_CUT_MIN_BIRTHS = 8
NO_ANNOTATIONS_REASON = "no_landmark_annotations"
_TIMESTAMP_TOLERANCE_S = 1e-6

DIAGNOSTIC_LABEL = (
    "Tracklet-level pitch-space diagnostic. This is NOT a verified individual player heatmap: "
    "tracklets are temporary and fragment, every identity is UNKNOWN unless a human confirmed it, "
    "referees and staff are not separated from players, and no team assignment exists."
)
LIMITATIONS = (
    "Positions exist only for frames whose homography is backed by manually annotated, independently validated "
    "pitch landmarks; every other frame is reported as uncalibrated or invalidated, never estimated.",
    "Foot points are the bottom-centre of tracker boxes; box-bottom bias (about 0.5 m) is not corrected.",
    "Projections outside the keyframe's fit-landmark hull are flagged as extrapolated; held-out landmark error "
    "does not bound their error.",
    "Tracklets are not players: one player may appear as several tracklets, and referees, staff and other persons "
    "are included as UNKNOWN tracklets.",
    "Identity association ran in review_only mode: no automatic association was made and no team is assigned.",
    "Lens distortion and pitch non-planarity are not modelled.",
)
_IDENTITY_CONFIG = ResolverConfig(mode=ResolverMode.REVIEW_ONLY)


class SpatialStageError(ValueError):
    """A safe, coded refusal. ``message`` never contains raw bytes or paths."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --- landmark annotations attached to a run --------------------------------------------


_PARAMETER_KEYS = frozenset({"schema", "sha256", "uploaded_sha256", "size_bytes", "origin", "keyframes", "document"})


def pitch_annotations_parameter(content: bytes, *, video_sha256: str, data_origin: str) -> dict[str, Any]:
    """Validate uploaded annotation bytes and return the run parameter that pins them.

    Raises :class:`SpatialStageError` with a stable code on any refusal.
    """
    if not isinstance(content, bytes) or not content:
        raise SpatialStageError("pitch_annotations_empty", "The pitch annotation file is empty.")
    if len(content) > MAX_PITCH_ANNOTATIONS_BYTES:
        raise SpatialStageError("pitch_annotations_too_large", "The pitch annotation file exceeds 1 MiB.")
    try:
        annotations = annotations_from_json_bytes(content)
    except (ValueError, TypeError, KeyError) as exc:
        detail = str(exc).replace("\n", " ")[:240]
        raise SpatialStageError(
            "invalid_pitch_annotations", f"The pitch annotation file is not valid {PITCH_ANNOTATIONS_SCHEMA}: {detail}"
        ) from exc
    source = annotations.source
    if source.video_sha256 is None:
        raise SpatialStageError(
            "pitch_annotations_unbound",
            "Pitch annotations must declare source.video_sha256, the SHA-256 of the uploaded video.",
        )
    if source.video_sha256 != video_sha256:
        raise SpatialStageError(
            "pitch_annotations_video_mismatch",
            "Pitch annotations were made for a different video (source.video_sha256 does not match the upload).",
        )
    if source.detections_sha256 is not None or source.tracklets_sha256 is not None:
        # Every attempt writes new AI1 bytes (tracklets name their run), so an
        # annotation pinned to earlier AI1 checksums could never match.
        raise SpatialStageError(
            "pitch_annotations_bound_to_ai1",
            "Pitch annotations attached to an analysis must be bound to the video only "
            "(create them with `template --bind video`); AI1 checksums change for every attempt.",
        )
    if source.origin is AnnotationOrigin.SYNTHETIC and data_origin != "synthetic":
        raise SpatialStageError(
            "pitch_annotations_synthetic",
            "Synthetic pitch annotations can only be attached to a synthetic-origin analysis.",
        )
    if not annotations.frames:
        raise SpatialStageError("pitch_annotations_no_keyframes", "Pitch annotations contain no annotated keyframes.")
    canonical = annotations_to_json_bytes(annotations)
    return {
        "schema": PITCH_ANNOTATIONS_SCHEMA,
        "sha256": sha256_bytes(canonical),
        "uploaded_sha256": sha256_bytes(content),
        "size_bytes": len(canonical),
        "origin": source.origin.value,
        "keyframes": len(annotations.frames),
        "document": json.loads(canonical),
    }


def load_pitch_annotations_parameter(value: Any) -> tuple[PitchAnnotations, str]:
    """Re-read the pinned annotations from a run parameter and verify their SHA-256."""
    if not isinstance(value, Mapping) or set(value) != _PARAMETER_KEYS or value.get("schema") != PITCH_ANNOTATIONS_SCHEMA:
        raise SpatialStageError("pitch_annotations_parameter_invalid", "The run's pitch annotation record is malformed.")
    try:
        document = (json.dumps(value["document"], allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
        annotations = annotations_from_json_bytes(document)
    except (ValueError, TypeError, KeyError) as exc:
        raise SpatialStageError(
            "pitch_annotations_parameter_invalid", "The run's pitch annotation record is malformed."
        ) from exc
    canonical = annotations_to_json_bytes(annotations)
    if sha256_bytes(canonical) != value["sha256"]:
        raise SpatialStageError(
            "pitch_annotations_checksum_mismatch", "The run's pitch annotations do not match their recorded SHA-256."
        )
    return annotations, value["sha256"]


# --- verified AI1 inputs -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerifiedAI1:
    detections: DetectionsArtifact
    tracklets: TrackletsArtifact
    detections_sha256: str
    tracklets_sha256: str
    video_sha256: str

    @property
    def frames(self) -> tuple[tuple[int, float], ...]:
        """Every AI1 sampled frame, including frames without detections."""
        return tuple((item.frame_index, item.timestamp_seconds) for item in self.detections.sampled_frames)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.detections.provenance.source_width, self.detections.provenance.source_height


def verify_ai1_artifacts(
    detections_bytes: bytes,
    tracklets_bytes: bytes,
    *,
    video_sha256: str,
    recorded_tracklets_sha256: str | None = None,
) -> VerifiedAI1:
    """Parse AI1 artifacts and check every binding the spatial stage relies on.

    - the tracklets bytes are the ones AI1 recorded (when a checksum was recorded),
    - the detections were computed from this run's video,
    - the tracklets were built from exactly these detections,
    - every tracklet observation lies on an AI1 sampled frame with the same timestamp.
    """
    detections_sha = sha256_bytes(detections_bytes)
    tracklets_sha = sha256_bytes(tracklets_bytes)
    if recorded_tracklets_sha256 is not None and recorded_tracklets_sha256 != tracklets_sha:
        raise SpatialStageError("ai1_tracklets_checksum_mismatch", "AI1 tracklets do not match their recorded SHA-256.")
    try:
        detections = detections_from_parquet_bytes(detections_bytes)
        tracklets = tracklets_from_parquet_bytes(tracklets_bytes)
    except Exception as exc:  # pyarrow raises its own exception types for corrupt files
        raise SpatialStageError("ai1_artifact_unreadable", "AI1 tracking artifacts could not be read.") from exc
    if detections.provenance.source_video_sha256 != video_sha256:
        raise SpatialStageError("ai1_video_binding_mismatch", "AI1 detections were computed from a different video.")
    if tracklets.provenance.detections_sha256 != detections_sha:
        raise SpatialStageError("ai1_tracklets_binding_mismatch", "AI1 tracklets were built from different detections.")
    sampled = {item.frame_index: item.timestamp_seconds for item in detections.sampled_frames}
    for tracklet in tracklets.tracklets:
        for item in tracklet.observations:
            expected = sampled.get(item.frame_index)
            if expected is None or abs(expected - item.timestamp_seconds) > _TIMESTAMP_TOLERANCE_S:
                raise SpatialStageError(
                    "ai1_frame_alignment_mismatch",
                    "An AI1 tracklet observation does not lie on a sampled frame with the same timestamp.",
                )
    return VerifiedAI1(detections, tracklets, detections_sha, tracklets_sha, video_sha256)


# --- identity_resolution (review_only) ------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdentityStageResult:
    artifact: PlayerIdentitiesArtifact
    content: bytes
    sha256: str

    def counts(self) -> dict[str, int]:
        statuses = Counter(item.status.value for item in self.artifact.assignments)
        return {status.value: statuses.get(status.value, 0) for status in AssignmentStatus}


def run_identity_stage(ai1: VerifiedAI1, tracklets_bytes: bytes, *, source_id: str) -> IdentityStageResult:
    """AI2C in ``review_only`` mode: no automatic association, so no identity is invented."""
    if sha256_bytes(tracklets_bytes) != ai1.tracklets_sha256:
        raise SpatialStageError("ai1_tracklets_checksum_mismatch", "AI1 tracklets changed after verification.")
    artifact = resolve_identities(
        summaries_from_tracklets(ai1.tracklets),
        source_for_tracklets(tracklets_bytes, ai1.tracklets, source_id=source_id, video_sha256=ai1.video_sha256),
        config=_IDENTITY_CONFIG,
    )
    # player_identities_sha256 is defined as the SHA-256 of these canonical bytes.
    content = player_identities_to_json_bytes(artifact)
    return IdentityStageResult(artifact, content, sha256_bytes(content))


# --- pitch_calibration -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpatialStageResult:
    calibration: PitchCalibrationArtifact | None
    calibration_bytes: bytes | None
    observations_bytes: bytes | None
    diagnostic: dict[str, Any]

    @property
    def calibrated_frames(self) -> int:
        return int(self.diagnostic["frames"]["by_status"]["calibrated"])

    @property
    def projected_observations(self) -> int:
        return int(self.diagnostic["observations"]["projected"])


_COLUMNS = (
    "frame_index", "timestamp_seconds", "tracklet_id", "identity_status", "player_identity_id", "team",
    "foot_pixel_x", "foot_pixel_y", "tracking_confidence", "projection_status", "pitch_x_m", "pitch_y_m",
    "inside_pitch_lines", "extrapolated_beyond_landmarks", "calibration_status", "calibration_method",
    "calibration_reason", "calibration_keyframe",
)


def _observations_to_parquet(columns: Mapping[str, list[Any]], provenance: Mapping[str, Any]) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("frame_index", pa.int64()), ("timestamp_seconds", pa.float64()), ("tracklet_id", pa.string()),
        ("identity_status", pa.string()), ("player_identity_id", pa.string()), ("team", pa.string()),
        ("foot_pixel_x", pa.float64()), ("foot_pixel_y", pa.float64()), ("tracking_confidence", pa.float64()),
        ("projection_status", pa.string()), ("pitch_x_m", pa.float64()), ("pitch_y_m", pa.float64()),
        ("inside_pitch_lines", pa.bool_()), ("extrapolated_beyond_landmarks", pa.bool_()),
        ("calibration_status", pa.string()), ("calibration_method", pa.string()),
        ("calibration_reason", pa.string()), ("calibration_keyframe", pa.int64()),
    ], metadata=artifact_metadata(PROJECTED_OBSERVATIONS_SCHEMA, provenance))
    # Columnar build (no per-row objects), then a stable (frame, tracklet) order.
    table = pa.Table.from_pydict(dict(columns), schema=schema).sort_by(
        [("frame_index", "ascending"), ("tracklet_id", "ascending")]
    )
    output = io.BytesIO()
    pq.write_table(table, output, compression="zstd", version="2.6")
    return output.getvalue()


_VIEW_COLUMNS = (
    "frame_index", "timestamp_seconds", "tracklet_id", "identity_status", "projection_status", "pitch_x_m",
    "pitch_y_m", "inside_pitch_lines", "extrapolated_beyond_landmarks", "calibration_keyframe",
)


def read_projected_observations(content: bytes, *, columns: Sequence[str] | None = None) -> tuple[dict[str, Any], Any]:
    """Return ``(provenance, pyarrow.Table)`` of a projected-observations artifact."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(io.BytesIO(content))
    if tuple(schema.names) != _COLUMNS:
        raise ValueError("projected observations columns do not match the contract")
    provenance = read_artifact_metadata(schema.metadata, PROJECTED_OBSERVATIONS_SCHEMA)
    table = pq.read_table(io.BytesIO(content), columns=list(columns) if columns is not None else None)
    return provenance, table


def _frame_states(
    ai1: VerifiedAI1, calibration: PitchCalibrationArtifact | None
) -> tuple[list[int], list[float], list[str], list[str | None]]:
    indices, times, statuses, reasons = [], [], [], []
    for frame_index, timestamp in ai1.frames:
        indices.append(frame_index)
        times.append(timestamp)
        state = calibration.frame(frame_index) if calibration is not None else None
        if state is None:
            statuses.append(CalibrationStatus.UNCALIBRATED.value)
            reasons.append(NO_ANNOTATIONS_REASON)
        else:
            statuses.append(state.status.value)
            reasons.append(state.reason.value if state.reason else None)
    return indices, times, statuses, reasons


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    return dict(sorted(Counter(item for item in values if item is not None).items()))


def run_pitch_calibration_stage(
    ai1: VerifiedAI1,
    identities: IdentityStageResult,
    *,
    run_id: str,
    data_origin: str,
    annotations: tuple[PitchAnnotations, str] | None,
    policy: CalibrationPolicy | None = None,
) -> SpatialStageResult:
    """Calibrate the run's AI1 frames and project AI2C observations onto the pitch."""
    policy = policy or CalibrationPolicy()
    if identities.artifact.source.tracklets_sha256 != ai1.tracklets_sha256:
        raise SpatialStageError("identity_binding_mismatch", "The identity artifact belongs to different tracklets.")
    calibration = calibration_bytes = observations_bytes = None
    calibration_sha = annotations_sha = None
    origin: AnnotationOrigin | None = None
    projection_counts: Counter[str] = Counter()
    projected_tracklets: set[str] = set()
    inside = extrapolated = 0
    observations = iter_player_observations(
        ai1.tracklets, ai1.tracklets_sha256, identities.artifact, identities.sha256,
        include_provisional=False, include_unresolved=True,
    )
    keyframes_outside = 0
    if annotations is not None:
        notes, annotations_sha = annotations
        origin = notes.source.origin
        if origin is AnnotationOrigin.SYNTHETIC and data_origin != "synthetic":
            raise SpatialStageError(
                "pitch_annotations_synthetic", "Synthetic pitch annotations cannot calibrate a non-synthetic analysis."
            )
        try:
            check_source_bindings(
                notes, tracklets_sha256=ai1.tracklets_sha256, detections_sha256=ai1.detections_sha256,
                video_sha256=ai1.video_sha256, frame_size=ai1.frame_size,
            )
            cuts = suspected_cuts_from_tracklets(ai1.tracklets, min_births=SUSPECTED_CUT_MIN_BIRTHS)
            calibration = calibrate(
                notes, annotations_sha256=annotations_sha, policy=policy, frames=ai1.frames,
                suspected_cut_frames=cuts, suspected_cuts_source=birth_burst_source(SUSPECTED_CUT_MIN_BIRTHS),
                tracklets_sha256=ai1.tracklets_sha256, detections_sha256=ai1.detections_sha256,
                video_sha256=ai1.video_sha256,
            )
        except ValueError as exc:
            raise SpatialStageError(
                "pitch_annotations_incompatible",
                "The attached pitch annotations do not match this run's AI1 frames, frame size or checksums.",
            ) from exc
        calibration_bytes = pitch_calibration_to_json_bytes(calibration)
        calibration_sha = calibration.sha256()
        sampled = {index for index, _ in ai1.frames}
        keyframes_outside = sum(item.frame_index not in sampled for item in calibration.keyframes)
        columns: dict[str, list[Any]] = {name: [] for name in _COLUMNS}
        for item in project_player_observations(
            observations, calibration, calibration_sha, pitch_margin_m=DEFAULT_PITCH_MARGIN_M,
            expected_policy=policy, allow_synthetic=origin is AnnotationOrigin.SYNTHETIC,
        ):
            record = item.observation
            status = item.projection_status.value
            projection_counts[status] += 1
            if item.projection_status is ProjectionStatus.PROJECTED:
                projected_tracklets.add(record.tracklet_id.value)
                inside += bool(item.inside_pitch_lines)
                extrapolated += bool(item.extrapolated_beyond_landmarks)
            for name, value in (
                ("frame_index", record.frame_index),
                ("timestamp_seconds", record.timestamp_seconds),
                ("tracklet_id", record.tracklet_id.value),
                ("identity_status", record.identity_status.value),
                ("player_identity_id", record.player_identity_id.value if record.player_identity_id else None),
                ("team", record.team.value),
                ("foot_pixel_x", record.foot_pixel_xy[0]),
                ("foot_pixel_y", record.foot_pixel_xy[1]),
                ("tracking_confidence", record.tracking_confidence),
                ("projection_status", status),
                ("pitch_x_m", item.pitch_xy_m[0] if item.pitch_xy_m else None),
                ("pitch_y_m", item.pitch_xy_m[1] if item.pitch_xy_m else None),
                ("inside_pitch_lines", item.inside_pitch_lines),
                ("extrapolated_beyond_landmarks", item.extrapolated_beyond_landmarks),
                ("calibration_status", item.calibration_status.value if item.calibration_status else None),
                ("calibration_method", item.calibration_method.value if item.calibration_method else None),
                ("calibration_reason", item.calibration_reason),
                ("calibration_keyframe", item.calibration_keyframe),
            ):
                columns[name].append(value)
        observations_bytes = _observations_to_parquet(columns, {
            "record_contract": PROJECTED_OBSERVATION_CONTRACT,
            "stage": {"name": STAGE_NAME, "version": STAGE_VERSION},
            "run_id": run_id,
            "source_video_sha256": ai1.video_sha256,
            "source_detections_sha256": ai1.detections_sha256,
            "source_tracklets_sha256": ai1.tracklets_sha256,
            "source_identities_sha256": identities.sha256,
            "source_annotations_sha256": annotations_sha,
            "source_calibration_sha256": calibration_sha,
            "calibration_origin": origin.value,
            "pitch_margin_m": DEFAULT_PITCH_MARGIN_M,
            "identity_filter": {"include_provisional": False, "include_unresolved": True},
            "level": "tracklet",
        })
    else:
        # No landmark evidence: every analysed frame is uncalibrated and no
        # observation is projected. The observations are still counted so the
        # diagnostic states what is missing.
        for _ in observations:
            projection_counts[ProjectionStatus.FRAME_UNCALIBRATED.value] += 1

    indices, times, statuses, reasons = _frame_states(ai1, calibration)
    by_status = Counter(statuses)
    frames_total = len(indices)
    if calibration is None:
        calibration_state = "not_attempted"
    elif by_status.get(CalibrationStatus.CALIBRATED.value, 0):
        calibration_state = "calibrated_frames_available"
    else:
        calibration_state = "no_calibrated_frames"
    if origin is None:
        evidence_status = "no_annotations"
    elif origin is AnnotationOrigin.SYNTHETIC:
        evidence_status = "synthetic_annotations"
    else:
        evidence_status = "manual_annotations"
    excluded_classes = Counter(
        item.semantic_class for item in ai1.tracklets.tracklets
        if item.semantic_class not in _IDENTITY_CONFIG.eligible_semantic_classes
    )
    projected = projection_counts.get(ProjectionStatus.PROJECTED.value, 0)
    diagnostic: dict[str, Any] = {
        "schema": PITCH_SPACE_DIAGNOSTIC_SCHEMA,
        "label": DIAGNOSTIC_LABEL,
        "level": "tracklet",
        "stage": {"name": STAGE_NAME, "version": STAGE_VERSION},
        "run_id": run_id,
        "data_origin": data_origin,
        "evidence_status": evidence_status,
        "calibration": {
            "state": calibration_state,
            "reason": NO_ANNOTATIONS_REASON if calibration is None else None,
            "policy_version": CALIBRATION_POLICY_VERSION,
            "summary": dict(calibration.summary) if calibration else None,
            "keyframes": [
                {
                    "frame_index": item.frame_index,
                    "timestamp_seconds": item.timestamp_seconds,
                    "shot_id": item.shot_id,
                    "status": item.status.value,
                    "reason": item.reason.value if item.reason else None,
                    "metrics": dict(item.metrics),
                }
                for item in (calibration.keyframes if calibration else ())
            ],
            "shots": [item.to_dict() for item in (calibration.shots if calibration else ())],
            "suspected_cut_frames": list(calibration.suspected_cut_frames) if calibration else [],
        },
        "pitch": calibration.pitch.to_dict() if calibration else None,
        "coordinate_system": dict(COORDINATE_SYSTEM_DEFINITION) if calibration else None,
        "frames": {
            "total": frames_total,
            "by_status": {status.value: by_status.get(status.value, 0) for status in CalibrationStatus},
            "by_reason": _counts(reasons),
            "calibrated_fraction": (
                round(by_status.get(CalibrationStatus.CALIBRATED.value, 0) / frames_total, 6) if frames_total else None
            ),
            "keyframes_outside_ai1_sampling": keyframes_outside,
        },
        "frame_states": {
            "frame_index": indices, "timestamp_seconds": times, "status": statuses, "reason": reasons,
        },
        "observations": {
            "total": sum(projection_counts.values()),
            "by_projection_status": dict(sorted(projection_counts.items())),
            "projected": projected,
            "projected_inside_pitch_lines": inside,
            "projected_extrapolated_beyond_landmarks": extrapolated,
            "distinct_tracklets_projected": len(projected_tracklets),
        },
        "excluded_tracklets_by_class": dict(sorted(excluded_classes.items())),
        "identity": {
            "resolver_mode": _IDENTITY_CONFIG.mode.value,
            "assignments_by_status": identities.counts(),
            "team_assignment": "not_available",
        },
        "evidence": {
            "video_sha256": ai1.video_sha256,
            "detections_sha256": ai1.detections_sha256,
            "tracklets_sha256": ai1.tracklets_sha256,
            "identities_sha256": identities.sha256,
            "annotations_sha256": annotations_sha,
            "calibration_sha256": calibration_sha,
            "projected_observations_sha256": sha256_bytes(observations_bytes) if observations_bytes else None,
        },
        "limitations": list(LIMITATIONS),
    }
    return SpatialStageResult(calibration, calibration_bytes, observations_bytes, diagnostic)


def diagnostic_to_json_bytes(diagnostic: Mapping[str, Any]) -> bytes:
    return (json.dumps(diagnostic, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


# --- read model: interval-filtered pitch-space view --------------------------------------


MAX_VIEW_POINTS = 5000
DEFAULT_CELL_SIZE_M = 5.0


def _finite_bound(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise SpatialStageError("invalid_interval", f"{name} must be a finite, non-negative number of seconds.")
    return result


def pitch_space_view(
    diagnostic_bytes: bytes,
    observations_bytes: bytes | None,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    max_points: int = 2000,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> dict[str, Any]:
    """Interval-filtered view over the stage's verified artifacts.

    Coverage is counted over AI1 sampled frames in ``[start_seconds, end_seconds]``.
    Occupancy and points use only ``projected`` observations in that interval.
    """
    start = _finite_bound(start_seconds, "start_seconds")
    end = _finite_bound(end_seconds, "end_seconds")
    if start is not None and end is not None and end < start:
        raise SpatialStageError("invalid_interval", "end_seconds must not be before start_seconds.")
    if not isinstance(max_points, int) or isinstance(max_points, bool) or not 0 <= max_points <= MAX_VIEW_POINTS:
        raise SpatialStageError("invalid_interval", f"max_points must be between 0 and {MAX_VIEW_POINTS}.")
    if not math.isfinite(cell_size_m) or not 1.0 <= cell_size_m <= 20.0:
        raise SpatialStageError("invalid_interval", "cell_size_m must be between 1 and 20 metres.")
    diagnostic = json.loads(diagnostic_bytes)
    if diagnostic.get("schema") != PITCH_SPACE_DIAGNOSTIC_SCHEMA:
        raise ValueError("artifact is not footballai.pitch-space-diagnostic/v1")

    def within(t: float) -> bool:
        return (start is None or t >= start) and (end is None or t <= end)

    states = diagnostic["frame_states"]
    selected = [
        (index, t, status, reason)
        for index, t, status, reason in zip(
            states["frame_index"], states["timestamp_seconds"], states["status"], states["reason"]
        )
        if within(t)
    ]
    frame_status = Counter(item[2] for item in selected)
    segments: list[dict[str, Any]] = []
    for index, t, status, reason in selected:
        if segments and segments[-1]["status"] == status and segments[-1]["reason"] == reason:
            segment = segments[-1]
            segment["end_frame"], segment["end_seconds"] = index, t
            segment["frames"] += 1
        else:
            segments.append({
                "start_frame": index, "end_frame": index, "start_seconds": t, "end_seconds": t,
                "status": status, "reason": reason, "frames": 1,
            })
    pitch = diagnostic.get("pitch")
    view: dict[str, Any] = {
        "schema": "footballai.pitch-space-view/v1",
        "label": diagnostic["label"],
        "level": diagnostic["level"],
        "data_origin": diagnostic["data_origin"],
        "evidence_status": diagnostic["evidence_status"],
        "calibration_state": diagnostic["calibration"]["state"],
        "calibration_reason": diagnostic["calibration"]["reason"],
        "pitch": pitch,
        "interval": {
            "start_seconds": start, "end_seconds": end,
            "analysed_frames": len(selected),
            "frames_by_status": {status.value: frame_status.get(status.value, 0) for status in CalibrationStatus},
            "frames_by_reason": _counts(item[3] for item in selected),
            "calibrated_fraction": (
                round(frame_status.get(CalibrationStatus.CALIBRATED.value, 0) / len(selected), 6) if selected else None
            ),
            "segments": segments,
        },
        "observations": None,
        "occupancy": None,
        "points": [],
        "points_truncated": False,
        "evidence": diagnostic["evidence"],
        "identity": diagnostic["identity"],
        "limitations": diagnostic["limitations"],
    }
    if observations_bytes is None:
        # Without a calibration no observation was projected. Per-observation
        # rows were not published, so interval totals are only known unfiltered.
        unfiltered = start is None and end is None
        view["observations"] = {
            "total": diagnostic["observations"]["total"] if unfiltered else None,
            "projected": 0,
            "by_projection_status": diagnostic["observations"]["by_projection_status"] if unfiltered else {},
            "distinct_tracklets_projected": 0,
            "projected_extrapolated_beyond_landmarks": 0,
            "projected_inside_pitch_lines": 0,
        }
        return view
    if sha256_bytes(observations_bytes) != diagnostic["evidence"]["projected_observations_sha256"]:
        raise ValueError("projected observations are not the artifact the diagnostic names")
    import pyarrow.compute as pc

    provenance, table = read_projected_observations(observations_bytes, columns=_VIEW_COLUMNS)
    if provenance.get("source_calibration_sha256") != diagnostic["evidence"]["calibration_sha256"]:
        raise ValueError("projected observations and diagnostic name different calibrations")
    stamps = table["timestamp_seconds"]
    mask = None
    if start is not None:
        mask = pc.greater_equal(stamps, start)
    if end is not None:
        upper = pc.less_equal(stamps, end)
        mask = upper if mask is None else pc.and_(mask, upper)
    selected_table = table.filter(mask) if mask is not None else table
    status_counts = Counter(selected_table["projection_status"].to_pylist())
    projected_table = selected_table.filter(pc.equal(selected_table["projection_status"], ProjectionStatus.PROJECTED.value))
    columns = projected_table.to_pydict()
    projected_rows = list(range(projected_table.num_rows))
    length, width = float(pitch["length_m"]), float(pitch["width_m"])
    margin = float(provenance.get("pitch_margin_m", DEFAULT_PITCH_MARGIN_M))
    cols = math.ceil((length + 2 * margin) / cell_size_m)
    rows_n = math.ceil((width + 2 * margin) / cell_size_m)
    grid = [[0] * cols for _ in range(rows_n)]
    for i in projected_rows:
        cx = min(cols - 1, max(0, int((columns["pitch_x_m"][i] + margin) // cell_size_m)))
        cy = min(rows_n - 1, max(0, int((columns["pitch_y_m"][i] + margin) // cell_size_m)))
        grid[cy][cx] += 1
    view["observations"] = {
        "total": selected_table.num_rows,
        "projected": len(projected_rows),
        "by_projection_status": dict(sorted(status_counts.items())),
        "distinct_tracklets_projected": len({columns["tracklet_id"][i] for i in projected_rows}),
        "projected_extrapolated_beyond_landmarks": sum(
            bool(columns["extrapolated_beyond_landmarks"][i]) for i in projected_rows
        ),
        "projected_inside_pitch_lines": sum(bool(columns["inside_pitch_lines"][i]) for i in projected_rows),
    }
    view["occupancy"] = {
        "unit": "projected tracklet observations per cell",
        "cell_size_m": cell_size_m,
        "origin_m": [-margin, -margin],
        "columns": cols,
        "rows": rows_n,
        "counts": grid,
    }
    if max_points and projected_rows:
        total = len(projected_rows)
        # Evenly spaced, deterministic subsample in time order.
        chosen = projected_rows if total <= max_points else [
            projected_rows[(k * total) // max_points] for k in range(max_points)
        ]
        view["points_truncated"] = len(chosen) < len(projected_rows)
        view["points"] = [
            {
                "frame_index": columns["frame_index"][i],
                "timestamp_seconds": columns["timestamp_seconds"][i],
                "tracklet_id": columns["tracklet_id"][i],
                "identity_status": columns["identity_status"][i],
                "x_m": columns["pitch_x_m"][i],
                "y_m": columns["pitch_y_m"][i],
                "inside_pitch_lines": columns["inside_pitch_lines"][i],
                "extrapolated_beyond_landmarks": columns["extrapolated_beyond_landmarks"][i],
                "calibration_keyframe": columns["calibration_keyframe"][i],
            }
            for i in chosen
        ]
    return view


__all__ = [
    "DIAGNOSTIC_LABEL",
    "MAX_PITCH_ANNOTATIONS_BYTES",
    "NO_ANNOTATIONS_REASON",
    "PITCH_ANNOTATIONS_PARAMETER",
    "PITCH_SPACE_DIAGNOSTIC_SCHEMA",
    "PROJECTED_OBSERVATIONS_SCHEMA",
    "IdentityStageResult",
    "SpatialStageError",
    "SpatialStageResult",
    "VerifiedAI1",
    "diagnostic_to_json_bytes",
    "load_pitch_annotations_parameter",
    "pitch_annotations_parameter",
    "pitch_space_view",
    "read_projected_observations",
    "run_identity_stage",
    "run_pitch_calibration_stage",
    "verify_ai1_artifacts",
]
