"""Versioned appearance evidence artifacts for bounded identity evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from footballai_v2.execution.ai2a.model import SourceReference, TrackletId
from footballai_v2.execution.ai2a.review import ReviewSamplesArtifact


APPEARANCE_EMBEDDINGS_SCHEMA = "footballai.appearance-embeddings/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_embedding(values: Iterable[float]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding must contain finite numeric values") from exc
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding must contain finite numeric values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("embedding must have a non-zero finite norm")
    return tuple(value / norm for value in vector)


def embedding_sha256(values: Iterable[float]) -> str:
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding must contain finite numeric values")
    content = json.dumps(vector, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_relative_path(value: str, name: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a bounded relative path")


def _validate_normalized(values: tuple[float, ...]) -> None:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("embedding must contain finite numeric values")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("embedding must be L2-normalized")


@dataclass(frozen=True, slots=True)
class AppearanceModelProvenance:
    model_name: str
    model_version: str
    model_sha256: str
    preprocessing_version: str
    embedding_dimension: int
    purpose: str
    license: str
    device: str

    def __post_init__(self) -> None:
        for name in (
            "model_name", "model_version", "preprocessing_version", "purpose", "license", "device"
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        _validate_sha256(self.model_sha256, "model_sha256")
        if not isinstance(self.embedding_dimension, int) or self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")


@dataclass(frozen=True, slots=True)
class CropEmbedding:
    tracklet_id: TrackletId
    crop_reference: str
    crop_sha256: str
    embedding: tuple[float, ...]
    embedding_sha256: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.crop_reference, "crop_reference")
        _validate_sha256(self.crop_sha256, "crop_sha256")
        _validate_normalized(self.embedding)
        _validate_sha256(self.embedding_sha256, "embedding_sha256")
        if embedding_sha256(self.embedding) != self.embedding_sha256:
            raise ValueError("embedding checksum mismatch")


@dataclass(frozen=True, slots=True)
class CropRejection:
    tracklet_id: TrackletId
    crop_reference: str
    crop_sha256: str
    reason: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.crop_reference, "crop_reference")
        _validate_sha256(self.crop_sha256, "crop_sha256")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("crop rejection reason must be non-empty")


@dataclass(frozen=True, slots=True)
class TrackletEmbedding:
    tracklet_id: TrackletId
    embedding: tuple[float, ...]
    embedding_sha256: str
    usable_crop_count: int
    rejected_crop_count: int

    def __post_init__(self) -> None:
        _validate_normalized(self.embedding)
        _validate_sha256(self.embedding_sha256, "embedding_sha256")
        if embedding_sha256(self.embedding) != self.embedding_sha256:
            raise ValueError("embedding checksum mismatch")
        if not isinstance(self.usable_crop_count, int) or self.usable_crop_count <= 0:
            raise ValueError("usable_crop_count must be positive")
        if not isinstance(self.rejected_crop_count, int) or self.rejected_crop_count < 0:
            raise ValueError("rejected_crop_count must be non-negative")


@dataclass(frozen=True, slots=True)
class AppearanceEmbeddingsArtifact:
    schema: str
    source: SourceReference
    model: AppearanceModelProvenance
    crop_embeddings: tuple[CropEmbedding, ...]
    tracklet_embeddings: tuple[TrackletEmbedding, ...]
    rejections: tuple[CropRejection, ...]

    def __post_init__(self) -> None:
        if self.schema != APPEARANCE_EMBEDDINGS_SCHEMA:
            raise ValueError(f"schema must be {APPEARANCE_EMBEDDINGS_SCHEMA}")
        for name in ("crop_embeddings", "tracklet_embeddings", "rejections"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f"{name} must be a tuple")
        dimension = self.model.embedding_dimension
        if any(len(item.embedding) != dimension for item in self.crop_embeddings):
            raise ValueError("crop embedding dimension does not match model provenance")
        if any(len(item.embedding) != dimension for item in self.tracklet_embeddings):
            raise ValueError("tracklet embedding dimension does not match model provenance")
        crop_keys = [(item.tracklet_id, item.crop_reference) for item in self.crop_embeddings]
        rejection_keys = [(item.tracklet_id, item.crop_reference) for item in self.rejections]
        if len(crop_keys) != len(set(crop_keys)) or len(rejection_keys) != len(set(rejection_keys)):
            raise ValueError("crop evidence keys must be unique")
        if set(crop_keys) & set(rejection_keys):
            raise ValueError("one crop cannot be both usable and rejected")
        tracklet_ids = [item.tracklet_id for item in self.tracklet_embeddings]
        if len(tracklet_ids) != len(set(tracklet_ids)):
            raise ValueError("tracklet embeddings must be unique")
        usable_counts: dict[TrackletId, int] = {}
        rejected_counts: dict[TrackletId, int] = {}
        for item in self.crop_embeddings:
            usable_counts[item.tracklet_id] = usable_counts.get(item.tracklet_id, 0) + 1
        for item in self.rejections:
            rejected_counts[item.tracklet_id] = rejected_counts.get(item.tracklet_id, 0) + 1
        for item in self.tracklet_embeddings:
            if item.usable_crop_count != usable_counts.get(item.tracklet_id, 0):
                raise ValueError("tracklet usable crop count does not match crop embeddings")
            if item.rejected_crop_count != rejected_counts.get(item.tracklet_id, 0):
                raise ValueError("tracklet rejected crop count does not match rejections")

    def validate_review_samples(self, samples: ReviewSamplesArtifact) -> None:
        if self.source != samples.source:
            raise ValueError("appearance artifact and review sample source linkage mismatch")
        expected = {
            (item.tracklet_id, item.relative_path): item.sha256 for item in samples.samples
        }
        for item in (*self.crop_embeddings, *self.rejections):
            if expected.get((item.tracklet_id, item.crop_reference)) != item.crop_sha256:
                raise ValueError("appearance crop-tracklet linkage mismatch")


def appearance_embeddings_to_json_bytes(artifact: AppearanceEmbeddingsArtifact) -> bytes:
    value = {
        "schema": artifact.schema,
        "source": asdict(artifact.source),
        "model": asdict(artifact.model),
        "crop_embeddings": [
            {**asdict(item), "tracklet_id": item.tracklet_id.value, "embedding": list(item.embedding)}
            for item in artifact.crop_embeddings
        ],
        "tracklet_embeddings": [
            {**asdict(item), "tracklet_id": item.tracklet_id.value, "embedding": list(item.embedding)}
            for item in artifact.tracklet_embeddings
        ],
        "rejections": [
            {**asdict(item), "tracklet_id": item.tracklet_id.value} for item in artifact.rejections
        ],
    }
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def appearance_embeddings_from_json_bytes(content: bytes) -> AppearanceEmbeddingsArtifact:
    try:
        value = json.loads(content)
        return AppearanceEmbeddingsArtifact(
            schema=value["schema"],
            source=SourceReference(**value["source"]),
            model=AppearanceModelProvenance(**value["model"]),
            crop_embeddings=tuple(
                CropEmbedding(
                    tracklet_id=TrackletId(item["tracklet_id"]),
                    crop_reference=item["crop_reference"],
                    crop_sha256=item["crop_sha256"],
                    embedding=tuple(item["embedding"]),
                    embedding_sha256=item["embedding_sha256"],
                )
                for item in value["crop_embeddings"]
            ),
            tracklet_embeddings=tuple(
                TrackletEmbedding(
                    tracklet_id=TrackletId(item["tracklet_id"]),
                    embedding=tuple(item["embedding"]),
                    embedding_sha256=item["embedding_sha256"],
                    usable_crop_count=item["usable_crop_count"],
                    rejected_crop_count=item["rejected_crop_count"],
                )
                for item in value["tracklet_embeddings"]
            ),
            rejections=tuple(
                CropRejection(
                    tracklet_id=TrackletId(item["tracklet_id"]),
                    crop_reference=item["crop_reference"],
                    crop_sha256=item["crop_sha256"],
                    reason=item["reason"],
                )
                for item in value["rejections"]
            ),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid appearance-embeddings artifact") from exc
