"""Deterministic multi-crop appearance extraction over AI2A review samples."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from footballai_v2.execution.ai2a.model import TrackletId
from footballai_v2.execution.ai2a.review import ReviewSamplesArtifact
from footballai_v2.execution.ai2b.appearance import AppearanceEncoder
from footballai_v2.execution.ai2b.artifact import (
    APPEARANCE_EMBEDDINGS_SCHEMA,
    AppearanceEmbeddingsArtifact,
    CropEmbedding,
    CropRejection,
    TrackletEmbedding,
    embedding_sha256,
    normalize_embedding,
)


def build_appearance_embeddings(
    samples: ReviewSamplesArtifact,
    artifact_root: Path,
    encoder: AppearanceEncoder,
    *,
    minimum_width: int = 24,
    minimum_height: int = 48,
) -> AppearanceEmbeddingsArtifact:
    """Encode valid early/middle/late crops and aggregate a normalized mean per tracklet."""
    import cv2

    if minimum_width < 1 or minimum_height < 1:
        raise ValueError("minimum crop dimensions must be positive")
    crop_embeddings: list[CropEmbedding] = []
    rejections: list[CropRejection] = []
    grouped_vectors: dict[TrackletId, list[tuple[float, ...]]] = defaultdict(list)
    rejected_counts: dict[TrackletId, int] = defaultdict(int)
    ordered = sorted(samples.samples, key=lambda item: (item.tracklet_id, item.frame_index, item.label))
    for sample in ordered:
        path = artifact_root / sample.relative_path
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != sample.sha256:
            raise ValueError(f"review crop checksum mismatch: {sample.relative_path}")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            reason = "crop_decode_failed"
        elif image.shape[1] < minimum_width or image.shape[0] < minimum_height:
            reason = "crop_too_small"
        else:
            reason = None
        if reason is not None:
            rejections.append(CropRejection(sample.tracklet_id, sample.relative_path, digest, reason))
            rejected_counts[sample.tracklet_id] += 1
            continue
        vector = normalize_embedding(encoder.encode(image))
        if len(vector) != encoder.provenance.embedding_dimension:
            raise ValueError("encoder output dimension does not match model provenance")
        crop_embeddings.append(
            CropEmbedding(
                sample.tracklet_id,
                sample.relative_path,
                digest,
                vector,
                embedding_sha256(vector),
            )
        )
        grouped_vectors[sample.tracklet_id].append(vector)
    tracklet_embeddings: list[TrackletEmbedding] = []
    for identifier, vectors in sorted(grouped_vectors.items()):
        mean = tuple(sum(items) / len(items) for items in zip(*vectors))
        vector = normalize_embedding(mean)
        tracklet_embeddings.append(
            TrackletEmbedding(
                identifier,
                vector,
                embedding_sha256(vector),
                len(vectors),
                rejected_counts.get(identifier, 0),
            )
        )
    artifact = AppearanceEmbeddingsArtifact(
        APPEARANCE_EMBEDDINGS_SCHEMA,
        samples.source,
        encoder.provenance,
        tuple(crop_embeddings),
        tuple(tracklet_embeddings),
        tuple(rejections),
    )
    artifact.validate_review_samples(samples)
    return artifact
