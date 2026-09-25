"""Versioned, detector-neutral AI1 detection and tracklet artifacts.

The wire format is Parquet for bounded size and typed columns. Public metadata
is stored in the Parquet schema rather than in filesystem-dependent sidecars.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping


DETECTIONS_SCHEMA = "footballai.detections/v1"
TRACKLETS_SCHEMA = "footballai.tracklets/v1"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_METADATA_KEY = b"footballai.artifact"


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _hash(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _safe_json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    try:
        encoded = json.dumps(dict(value), allow_nan=False, sort_keys=True, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be an object")
    return decoded


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def validate(self, *, width: int | None = None, height: int | None = None) -> None:
        values = tuple(_finite(value, "bounding box coordinate") for value in asdict(self).values())
        x1, y1, x2, y2 = values
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bounding box must have non-negative origin and positive area")
        if width is not None and x2 > width:
            raise ValueError("bounding box exceeds source width")
        if height is not None and y2 > height:
            raise ValueError("bounding box exceeds source height")

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass(frozen=True, slots=True)
class Detection:
    frame_index: int
    timestamp_seconds: float
    class_id: int
    semantic_class: str
    confidence: float
    bbox: BoundingBox

    def validate(self, *, width: int, height: int) -> None:
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if _finite(self.timestamp_seconds, "timestamp_seconds") < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        if not isinstance(self.class_id, int) or isinstance(self.class_id, bool) or self.class_id < 0:
            raise ValueError("class_id must be a non-negative integer")
        if not isinstance(self.semantic_class, str) or not self.semantic_class.strip():
            raise ValueError("semantic_class must be non-empty")
        confidence = _finite(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        self.bbox.validate(width=width, height=height)


@dataclass(frozen=True, slots=True)
class DetectionProvenance:
    model_family: str
    model_name: str
    model_version: str
    model_sha256: str
    confidence_threshold: float
    image_size: int
    target_fps: float
    effective_fps: float
    device: str
    source_video_sha256: str
    source_width: int
    source_height: int
    source_fps: float
    source_total_frames: int
    sampling_stride: int
    source_duration_seconds: float
    code_revision: str
    pipeline_version: str

    def __post_init__(self) -> None:
        for name in ("model_family", "model_name", "model_version", "device", "pipeline_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "/" in value and name == "model_name":
                raise ValueError(f"{name} must be a safe non-empty name")
        _hash(self.model_sha256, "model_sha256")
        _hash(self.source_video_sha256, "source_video_sha256")
        if not _REVISION_RE.fullmatch(self.code_revision):
            raise ValueError("code_revision must be a lowercase Git revision")
        threshold = _finite(self.confidence_threshold, "confidence_threshold")
        if not 0 < threshold <= 1:
            raise ValueError("confidence_threshold must be between zero and one")
        for name in ("image_size", "source_width", "source_height", "source_total_frames", "sampling_stride"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _positive(self.target_fps, "target_fps")
        _positive(self.effective_fps, "effective_fps")
        _positive(self.source_fps, "source_fps")
        _positive(self.source_duration_seconds, "source_duration_seconds")


@dataclass(frozen=True, slots=True)
class SampledFrame:
    frame_index: int
    timestamp_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("sampled frame index must be a non-negative integer")
        if _finite(self.timestamp_seconds, "sampled frame timestamp") < 0:
            raise ValueError("sampled frame timestamp must be non-negative")


@dataclass(frozen=True, slots=True)
class DetectionsArtifact:
    schema: str
    provenance: DetectionProvenance
    detections: tuple[Detection, ...]
    sampled_frames: tuple[SampledFrame, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != DETECTIONS_SCHEMA:
            raise ValueError(f"schema must be {DETECTIONS_SCHEMA}")
        if not isinstance(self.detections, tuple):
            raise ValueError("detections must be a tuple")
        if not isinstance(self.sampled_frames, tuple):
            raise ValueError("sampled_frames must be a tuple")
        if not self.sampled_frames and self.detections:
            inferred: dict[int, float] = {}
            for item in self.detections:
                inferred.setdefault(item.frame_index, item.timestamp_seconds)
            object.__setattr__(self, "sampled_frames", tuple(
                SampledFrame(frame_index, timestamp) for frame_index, timestamp in inferred.items()
            ))
        previous_sampled_frame = -1
        previous_sampled_timestamp = -1.0
        sampled = {}
        for item in self.sampled_frames:
            if item.frame_index <= previous_sampled_frame or item.timestamp_seconds <= previous_sampled_timestamp:
                raise ValueError("sampled frame order and timestamp order must be strictly increasing")
            sampled[item.frame_index] = item.timestamp_seconds
            previous_sampled_frame = item.frame_index
            previous_sampled_timestamp = item.timestamp_seconds
        previous_frame = -1
        previous_timestamp = -1.0
        frame_timestamps: dict[int, float] = {}
        for item in self.detections:
            item.validate(width=self.provenance.source_width, height=self.provenance.source_height)
            if item.frame_index < previous_frame:
                raise ValueError("detections must be in non-decreasing frame order")
            if item.timestamp_seconds < previous_timestamp:
                raise ValueError("detections must be in non-decreasing timestamp order")
            existing = frame_timestamps.setdefault(item.frame_index, item.timestamp_seconds)
            if existing != item.timestamp_seconds:
                raise ValueError("detections in one frame must share a timestamp")
            if sampled.get(item.frame_index) != item.timestamp_seconds:
                raise ValueError("detection frame is absent from sampled frame sequence")
            previous_frame = item.frame_index
            previous_timestamp = item.timestamp_seconds

    @property
    def frames_processed(self) -> int:
        return len(self.sampled_frames)


@dataclass(frozen=True, slots=True)
class TrackObservation:
    frame_index: int
    timestamp_seconds: float
    bbox: BoundingBox
    confidence: float

    def validate(self) -> None:
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if _finite(self.timestamp_seconds, "timestamp_seconds") < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        confidence = _finite(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        self.bbox.validate()


@dataclass(frozen=True, slots=True)
class Tracklet:
    track_id: int
    class_id: int
    semantic_class: str
    observations: tuple[TrackObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, int) or isinstance(self.track_id, bool) or self.track_id < 1:
            raise ValueError("track_id must be a positive integer")
        if not isinstance(self.class_id, int) or isinstance(self.class_id, bool) or self.class_id < 0:
            raise ValueError("class_id must be a non-negative integer")
        if not isinstance(self.semantic_class, str) or not self.semantic_class.strip():
            raise ValueError("semantic_class must be non-empty")
        if not isinstance(self.observations, tuple) or not self.observations:
            raise ValueError("tracklet observations must be a non-empty tuple")
        previous_frame = -1
        previous_timestamp = -1.0
        for item in self.observations:
            item.validate()
            if item.frame_index <= previous_frame or item.timestamp_seconds <= previous_timestamp:
                raise ValueError("tracklet observations must be strictly increasing")
            previous_frame = item.frame_index
            previous_timestamp = item.timestamp_seconds

    @property
    def start_frame(self) -> int:
        return self.observations[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame_index

    @property
    def start_time_seconds(self) -> float:
        return self.observations[0].timestamp_seconds

    @property
    def end_time_seconds(self) -> float:
        return self.observations[-1].timestamp_seconds

    @property
    def duration_seconds(self) -> float:
        return self.end_time_seconds - self.start_time_seconds

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def mean_confidence(self) -> float:
        return sum(item.confidence for item in self.observations) / self.observation_count

    @property
    def confidence_summary(self) -> dict[str, float]:
        values = [item.confidence for item in self.observations]
        return {"min": min(values), "mean": self.mean_confidence, "max": max(values)}


@dataclass(frozen=True, slots=True)
class TrackerProvenance:
    tracker_name: str
    tracker_version: str
    configuration: Mapping[str, Any]
    detections_artifact_id: str
    detections_sha256: str
    source_run_id: str

    def __post_init__(self) -> None:
        for name in ("tracker_name", "tracker_version", "detections_artifact_id", "source_run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        _hash(self.detections_sha256, "detections_sha256")
        object.__setattr__(self, "configuration", MappingProxyType(_safe_json_mapping(self.configuration, "configuration")))


@dataclass(frozen=True, slots=True)
class TrackletsArtifact:
    schema: str
    provenance: TrackerProvenance
    tracklets: tuple[Tracklet, ...]

    def __post_init__(self) -> None:
        if self.schema != TRACKLETS_SCHEMA:
            raise ValueError(f"schema must be {TRACKLETS_SCHEMA}")
        if not isinstance(self.tracklets, tuple):
            raise ValueError("tracklets must be a tuple")
        ids = [item.track_id for item in self.tracklets]
        if len(ids) != len(set(ids)):
            raise ValueError("tracklet IDs must be unique")

    @property
    def observation_count(self) -> int:
        return sum(item.observation_count for item in self.tracklets)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _metadata(schema: str, provenance: Mapping[str, Any]) -> dict[bytes, bytes]:
    value = {"schema": schema, "provenance": provenance}
    return {_METADATA_KEY: json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()}


def _read_metadata(table: Any, expected_schema: str) -> dict[str, Any]:
    raw = (table.schema.metadata or {}).get(_METADATA_KEY)
    if raw is None:
        raise ValueError("Parquet artifact metadata is missing")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Parquet artifact metadata is invalid") from exc
    if value.get("schema") != expected_schema or not isinstance(value.get("provenance"), dict):
        raise ValueError("Parquet artifact schema does not match")
    return value["provenance"]


def detections_to_parquet_bytes(artifact: DetectionsArtifact) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("frame_index", pa.int64()), ("timestamp_seconds", pa.float64()), ("has_detection", pa.bool_()),
        ("class_id", pa.int32()), ("semantic_class", pa.string()),
        ("confidence", pa.float64()), ("x1", pa.float64()), ("y1", pa.float64()),
        ("x2", pa.float64()), ("y2", pa.float64()),
    ], metadata=_metadata(DETECTIONS_SCHEMA, asdict(artifact.provenance)))
    by_frame: dict[int, list[Detection]] = {}
    for item in artifact.detections:
        by_frame.setdefault(item.frame_index, []).append(item)
    rows = []
    for sampled in artifact.sampled_frames:
        frame_detections = by_frame.get(sampled.frame_index, [])
        if not frame_detections:
            rows.append({
                "frame_index": sampled.frame_index, "timestamp_seconds": sampled.timestamp_seconds,
                "has_detection": False, "class_id": None, "semantic_class": None, "confidence": None,
                "x1": None, "y1": None, "x2": None, "y2": None,
            })
        for item in frame_detections:
            rows.append({
                "frame_index": item.frame_index, "timestamp_seconds": item.timestamp_seconds,
                "has_detection": True, "class_id": item.class_id, "semantic_class": item.semantic_class,
                "confidence": item.confidence, "x1": item.bbox.x1, "y1": item.bbox.y1,
                "x2": item.bbox.x2, "y2": item.bbox.y2,
            })
    table = pa.Table.from_pylist(rows, schema=schema)
    output = io.BytesIO()
    pq.write_table(table, output, compression="zstd", version="2.6")
    return output.getvalue()


def detections_from_parquet_bytes(content: bytes) -> DetectionsArtifact:
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(content))
    provenance = DetectionProvenance(**_read_metadata(table, DETECTIONS_SCHEMA))
    rows = table.to_pylist()
    items = tuple(Detection(
        frame_index=row["frame_index"], timestamp_seconds=row["timestamp_seconds"],
        class_id=row["class_id"], semantic_class=row["semantic_class"], confidence=row["confidence"],
        bbox=BoundingBox(row["x1"], row["y1"], row["x2"], row["y2"]),
    ) for row in rows if row["has_detection"])
    sampled: dict[int, float] = {}
    for row in rows:
        sampled.setdefault(row["frame_index"], row["timestamp_seconds"])
    sampled_frames = tuple(SampledFrame(frame, timestamp) for frame, timestamp in sampled.items())
    return DetectionsArtifact(DETECTIONS_SCHEMA, provenance, items, sampled_frames)


def tracklets_to_parquet_bytes(artifact: TrackletsArtifact) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    provenance = {
        "tracker_name": artifact.provenance.tracker_name,
        "tracker_version": artifact.provenance.tracker_version,
        "configuration": dict(artifact.provenance.configuration),
        "detections_artifact_id": artifact.provenance.detections_artifact_id,
        "detections_sha256": artifact.provenance.detections_sha256,
        "source_run_id": artifact.provenance.source_run_id,
    }
    schema = pa.schema([
        ("track_id", pa.int64()), ("class_id", pa.int32()), ("semantic_class", pa.string()),
        ("frame_index", pa.int64()), ("timestamp_seconds", pa.float64()),
        ("confidence", pa.float64()), ("x1", pa.float64()), ("y1", pa.float64()),
        ("x2", pa.float64()), ("y2", pa.float64()),
    ], metadata=_metadata(TRACKLETS_SCHEMA, provenance))
    rows = []
    for tracklet in artifact.tracklets:
        for item in tracklet.observations:
            rows.append({
                "track_id": tracklet.track_id, "class_id": tracklet.class_id,
                "semantic_class": tracklet.semantic_class, "frame_index": item.frame_index,
                "timestamp_seconds": item.timestamp_seconds, "confidence": item.confidence,
                "x1": item.bbox.x1, "y1": item.bbox.y1, "x2": item.bbox.x2, "y2": item.bbox.y2,
            })
    table = pa.Table.from_pylist(rows, schema=schema)
    output = io.BytesIO()
    pq.write_table(table, output, compression="zstd", version="2.6")
    return output.getvalue()


def tracklets_from_parquet_bytes(content: bytes) -> TrackletsArtifact:
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(content))
    provenance = TrackerProvenance(**_read_metadata(table, TRACKLETS_SCHEMA))
    grouped: dict[int, dict[str, Any]] = {}
    for row in table.to_pylist():
        group = grouped.setdefault(row["track_id"], {
            "class_id": row["class_id"], "semantic_class": row["semantic_class"], "observations": [],
        })
        if group["class_id"] != row["class_id"] or group["semantic_class"] != row["semantic_class"]:
            raise ValueError("tracklet class changed within the artifact")
        group["observations"].append(TrackObservation(
            frame_index=row["frame_index"], timestamp_seconds=row["timestamp_seconds"],
            confidence=row["confidence"], bbox=BoundingBox(row["x1"], row["y1"], row["x2"], row["y2"]),
        ))
    tracklets = tuple(Tracklet(
        track_id=track_id, class_id=value["class_id"], semantic_class=value["semantic_class"],
        observations=tuple(value["observations"]),
    ) for track_id, value in sorted(grouped.items()))
    return TrackletsArtifact(TRACKLETS_SCHEMA, provenance, tracklets)
