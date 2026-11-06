"""Bounded per-tracklet summaries consumed by the AI2C resolver.

The resolver needs only each tracklet's temporal extent and its first and last
pixel boxes, so its memory is proportional to the number of tracklets rather
than the number of observations. Three input kinds are supported and recorded
in provenance; they are never silently interchangeable:

``footballai.tracklets/v1``
    The authoritative AI1 artifact. Every field is exact.
``footballai.tracklet-review-samples/v1``
    AI2A review samples. ``early``/``late`` are the first/last observations
    (see ``select_representative_observations``), so extents and endpoint boxes
    are exact; observation counts are known only for one/two-observation
    tracklets. Used to evaluate against committed benchmarks whose Parquet
    caches are intentionally not in Git.
``legacy.v1-raw-tracks``
    The historical V1 ``raw_tracks.parquet`` (already filtered to tracks with at
    least ten observations, legacy frame labels, no semantic class). Diagnostic
    scale input only; it has no identity ground truth.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from footballai_v2.execution.ai1.artifacts import BoundingBox, TrackletsArtifact, sha256_bytes
from footballai_v2.execution.ai2a.model import TrackletId, tracklet_id
from footballai_v2.execution.ai2a.review import ReviewSamplesArtifact


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class InputKind(str, Enum):
    TRACKLETS = "footballai.tracklets/v1"
    REVIEW_SAMPLES = "footballai.tracklet-review-samples/v1"
    LEGACY_RAW_TRACKS = "legacy.v1-raw-tracks"


def _check_sha(value: str | None, name: str, *, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    """Provenance of the tracklets an identity artifact refers to."""

    source_id: str
    input_kind: InputKind
    input_sha256: str
    tracklets_sha256: str | None
    detections_sha256: str | None
    video_sha256: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not _SOURCE_ID_RE.fullmatch(self.source_id):
            raise ValueError("source_id must be a short safe identifier")
        if not isinstance(self.input_kind, InputKind):
            raise ValueError("input_kind must be typed")
        _check_sha(self.input_sha256, "input_sha256", required=True)
        _check_sha(
            self.tracklets_sha256,
            "tracklets_sha256",
            required=self.input_kind is not InputKind.LEGACY_RAW_TRACKS,
        )
        _check_sha(self.detections_sha256, "detections_sha256", required=False)
        _check_sha(self.video_sha256, "video_sha256", required=False)
        if self.input_kind is InputKind.TRACKLETS and self.input_sha256 != self.tracklets_sha256:
            raise ValueError("an AI1 tracklets input must be the tracklets artifact itself")

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "input_kind": self.input_kind.value,
            "input_sha256": self.input_sha256,
            "tracklets_sha256": self.tracklets_sha256,
            "detections_sha256": self.detections_sha256,
            "video_sha256": self.video_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "SourceDescriptor":
        return cls(
            source_id=value["source_id"],
            input_kind=InputKind(value["input_kind"]),
            input_sha256=value["input_sha256"],
            tracklets_sha256=value.get("tracklets_sha256"),
            detections_sha256=value.get("detections_sha256"),
            video_sha256=value.get("video_sha256"),
        )


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class TrackletSummary:
    tracklet_id: TrackletId
    semantic_class: str | None
    start_frame: int
    end_frame: int
    start_time_seconds: float
    end_time_seconds: float
    first_bbox: BoundingBox
    last_bbox: BoundingBox
    observation_count: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.tracklet_id, TrackletId):
            raise ValueError("tracklet_id must be typed")
        if self.semantic_class is not None and (
            not isinstance(self.semantic_class, str) or not self.semantic_class.strip()
        ):
            raise ValueError("semantic_class must be non-empty when supplied")
        for name in ("start_frame", "end_frame"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.end_frame < self.start_frame:
            raise ValueError("tracklet frame extent is invalid")
        start = _finite(self.start_time_seconds, "start_time_seconds")
        end = _finite(self.end_time_seconds, "end_time_seconds")
        if start < 0 or end < start:
            raise ValueError("tracklet time extent is invalid")
        if (self.start_frame == self.end_frame) != (start == end):
            raise ValueError("tracklet frame and time extents disagree")
        for box in (self.first_bbox, self.last_bbox):
            box.validate()
        if self.observation_count is not None:
            if (
                not isinstance(self.observation_count, int)
                or isinstance(self.observation_count, bool)
                or self.observation_count < 1
            ):
                raise ValueError("observation_count must be a positive integer when known")
            if self.observation_count == 1 and self.start_frame != self.end_frame:
                raise ValueError("a single observation cannot span several frames")

    @property
    def duration_seconds(self) -> float:
        return self.end_time_seconds - self.start_time_seconds


def _sorted_unique(items: Iterable[TrackletSummary]) -> tuple[TrackletSummary, ...]:
    result = tuple(sorted(items, key=lambda item: item.tracklet_id))
    ids = [item.tracklet_id for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("tracklet summaries must be unique")
    return result


def summaries_from_tracklets(artifact: TrackletsArtifact) -> tuple[TrackletSummary, ...]:
    return _sorted_unique(
        TrackletSummary(
            tracklet_id=tracklet_id(item.track_id),
            semantic_class=item.semantic_class,
            start_frame=item.start_frame,
            end_frame=item.end_frame,
            start_time_seconds=item.start_time_seconds,
            end_time_seconds=item.end_time_seconds,
            first_bbox=item.observations[0].bbox,
            last_bbox=item.observations[-1].bbox,
            observation_count=item.observation_count,
        )
        for item in artifact.tracklets
    )


def source_for_tracklets(
    content: bytes, artifact: TrackletsArtifact, *, source_id: str, video_sha256: str | None = None
) -> SourceDescriptor:
    digest = sha256_bytes(content)
    return SourceDescriptor(
        source_id=source_id,
        input_kind=InputKind.TRACKLETS,
        input_sha256=digest,
        tracklets_sha256=digest,
        detections_sha256=artifact.provenance.detections_sha256,
        video_sha256=video_sha256,
    )


def summaries_from_review_samples(artifact: ReviewSamplesArtifact) -> tuple[TrackletSummary, ...]:
    """Reconstruct exact extents and endpoint boxes from early/middle/late samples."""
    grouped: dict[TrackletId, dict[str, object]] = {}
    for sample in artifact.samples:
        grouped.setdefault(sample.tracklet_id, {})[sample.label] = sample
    result = []
    for identifier, samples in grouped.items():
        early = samples.get("early")
        if early is None:
            raise ValueError(f"{identifier.value} has no early (first-observation) sample")
        late = samples.get("late", early)
        if "late" not in samples and "middle" in samples:
            raise ValueError(f"{identifier.value} has a middle sample but no late sample")
        if late.frame_index < early.frame_index:
            raise ValueError(f"{identifier.value} samples are out of order")
        if "late" not in samples:
            count: int | None = 1
        elif "middle" not in samples:
            count = 2
        else:
            count = None
        result.append(TrackletSummary(
            tracklet_id=identifier,
            semantic_class=None,
            start_frame=early.frame_index,
            end_frame=late.frame_index,
            start_time_seconds=early.timestamp_seconds,
            end_time_seconds=late.timestamp_seconds,
            first_bbox=BoundingBox(*early.bbox_xyxy),
            last_bbox=BoundingBox(*late.bbox_xyxy),
            observation_count=count,
        ))
    return _sorted_unique(result)


def source_for_review_samples(content: bytes, artifact: ReviewSamplesArtifact) -> SourceDescriptor:
    return SourceDescriptor(
        source_id=artifact.source.source_id,
        input_kind=InputKind.REVIEW_SAMPLES,
        input_sha256=sha256_bytes(content),
        tracklets_sha256=artifact.source.tracklets_sha256,
        detections_sha256=artifact.source.detections_sha256,
        video_sha256=artifact.source.video_sha256,
    )


_LEGACY_COLUMNS = ("frame_idx", "time_sec", "track_id", "cx", "cy", "w", "h")


def _legacy_box(cx: float, cy: float, w: float, h: float) -> BoundingBox:
    return BoundingBox(max(0.0, cx - w / 2), max(0.0, cy - h / 2), cx + w / 2, cy + h / 2)


def summaries_from_legacy_raw_tracks(
    content: bytes, *, batch_size: int = 65_536
) -> tuple[TrackletSummary, ...]:
    """Stream the historical V1 table, keeping only first/last rows per track."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(io.BytesIO(content))
    missing = set(_LEGACY_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"legacy raw tracks are missing columns: {sorted(missing)}")
    state: dict[int, list] = {}
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(_LEGACY_COLUMNS)):
        columns = [batch.column(name).to_pylist() for name in _LEGACY_COLUMNS]
        for frame, time, track, cx, cy, w, h in zip(*columns):
            row = (int(frame), float(time), float(cx), float(cy), float(w), float(h))
            current = state.get(int(track))
            if current is None:
                state[int(track)] = [row, row, 1]
                continue
            if row[0] < current[0][0]:
                current[0] = row
            if row[0] > current[1][0]:
                current[1] = row
            current[2] += 1
    return _sorted_unique(
        TrackletSummary(
            tracklet_id=tracklet_id(track),
            semantic_class=None,
            start_frame=first[0],
            end_frame=last[0],
            start_time_seconds=first[1],
            end_time_seconds=last[1],
            first_bbox=_legacy_box(*first[2:]),
            last_bbox=_legacy_box(*last[2:]),
            observation_count=count,
        )
        for track, (first, last, count) in state.items()
    )


def source_for_legacy_raw_tracks(content: bytes, *, source_id: str) -> SourceDescriptor:
    return SourceDescriptor(
        source_id=source_id,
        input_kind=InputKind.LEGACY_RAW_TRACKS,
        input_sha256=sha256_bytes(content),
        tracklets_sha256=None,
        detections_sha256=None,
        video_sha256=None,
    )
