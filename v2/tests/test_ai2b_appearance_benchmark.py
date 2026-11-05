"""AI2B appearance evidence contracts; never a production identity resolver."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from footballai_v2.execution.ai2a.association import CannotLink, TeamEvidence
from footballai_v2.execution.ai2a.model import SourceReference, TeamLabel, tracklet_id
from footballai_v2.execution.ai2a.review import (
    REVIEW_SAMPLES_SCHEMA,
    ReviewSample,
    ReviewSamplesArtifact,
)
from footballai_v2.execution.ai2b.appearance import preprocess_crop
from footballai_v2.execution.ai2b.artifact import (
    APPEARANCE_EMBEDDINGS_SCHEMA,
    AppearanceEmbeddingsArtifact,
    AppearanceModelProvenance,
    CropEmbedding,
    TrackletEmbedding,
    appearance_embeddings_from_json_bytes,
    appearance_embeddings_to_json_bytes,
    embedding_sha256,
    normalize_embedding,
)
from footballai_v2.execution.ai2b.association import (
    cosine_similarity,
    score_candidate_pair,
    select_similarity_threshold,
)
from footballai_v2.execution.ai2b.metrics import binary_pair_metrics, team_classification_report
from footballai_v2.execution.ai2b.pipeline import build_appearance_embeddings


def source() -> SourceReference:
    return SourceReference("bounded", "a" * 64, 60.0, "b" * 64, "c" * 64)


def provenance(*, dimension: int = 3) -> AppearanceModelProvenance:
    return AppearanceModelProvenance(
        model_name="mock-appearance",
        model_version="test-v1",
        model_sha256="d" * 64,
        preprocessing_version="rgb-center-crop-224/v1",
        embedding_dimension=dimension,
        purpose="experimental general visual control",
        license="BSD-3-Clause",
        device="cpu",
    )


def write_crop(root: Path, *, name: str = "early-f000000.jpg", size=(80, 40)) -> ReviewSample:
    image = np.full((size[0], size[1], 3), (10, 80, 220), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    content = encoded.tobytes()
    relative = f"crops/tracklet_000001/{name}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ReviewSample(
        tracklet_id(1), "early", 0, 0.0, (0.0, 0.0, float(size[1]), float(size[0])),
        relative, hashlib.sha256(content).hexdigest(),
    )


class MockEncoder:
    provenance = provenance()

    def encode(self, crop: np.ndarray) -> tuple[float, ...]:
        assert crop.ndim == 3
        return normalize_embedding((float(crop[..., 0].mean()), 2.0, 3.0))


def test_preprocessing_is_deterministic_and_channel_first():
    crop = np.arange(60 * 30 * 3, dtype=np.uint8).reshape((60, 30, 3))

    first = preprocess_crop(crop)
    second = preprocess_crop(crop.copy())

    assert first.shape == (3, 224, 224)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)


def test_mocked_encoder_pipeline_output_is_deterministic(tmp_path: Path):
    sample = write_crop(tmp_path)
    reviews = ReviewSamplesArtifact(REVIEW_SAMPLES_SCHEMA, source(), "early_middle_late/v1", (sample,))

    first = build_appearance_embeddings(reviews, tmp_path, MockEncoder())
    second = build_appearance_embeddings(reviews, tmp_path, MockEncoder())

    assert first == second
    assert first.tracklet_embeddings[0].usable_crop_count == 1


def test_embedding_is_l2_normalized():
    value = normalize_embedding((3.0, 4.0))

    assert value == pytest.approx((0.6, 0.8))
    assert math.sqrt(sum(item * item for item in value)) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [(0.0, 0.0), (math.nan, 1.0), (math.inf, 1.0), ()])
def test_zero_or_invalid_embedding_is_rejected(value):
    with pytest.raises(ValueError, match="embedding"):
        normalize_embedding(value)


def test_appearance_artifact_round_trip_is_canonical():
    crop_vector = normalize_embedding((1.0, 2.0, 3.0))
    tracklet_vector = normalize_embedding((3.0, 2.0, 1.0))
    artifact = AppearanceEmbeddingsArtifact(
        APPEARANCE_EMBEDDINGS_SCHEMA,
        source(),
        provenance(),
        (CropEmbedding(tracklet_id(1), "crops/tracklet_000001/early-f000000.jpg", "e" * 64,
                       crop_vector, embedding_sha256(crop_vector)),),
        (TrackletEmbedding(tracklet_id(1), tracklet_vector, embedding_sha256(tracklet_vector), 1, 0),),
        (),
    )

    encoded = appearance_embeddings_to_json_bytes(artifact)
    decoded = appearance_embeddings_from_json_bytes(encoded)

    assert decoded == artifact
    assert appearance_embeddings_to_json_bytes(decoded) == encoded


def test_crop_tracklet_linkage_is_validated(tmp_path: Path):
    sample = write_crop(tmp_path)
    reviews = ReviewSamplesArtifact(REVIEW_SAMPLES_SCHEMA, source(), "early_middle_late/v1", (sample,))
    artifact = build_appearance_embeddings(reviews, tmp_path, MockEncoder())
    bad_reviews = ReviewSamplesArtifact(
        REVIEW_SAMPLES_SCHEMA, source(), "early_middle_late/v1",
        (ReviewSample(tracklet_id(2), sample.label, sample.frame_index, sample.timestamp_seconds,
                      sample.bbox_xyxy, sample.relative_path, sample.sha256),),
    )

    with pytest.raises(ValueError, match="linkage"):
        artifact.validate_review_samples(bad_reviews)


def test_confident_cross_team_pair_is_rejected():
    decision = score_candidate_pair(
        tracklet_id(1), tracklet_id(2),
        {tracklet_id(1): (1.0, 0.0), tracklet_id(2): (1.0, 0.0)},
        (),
        {
            tracklet_id(1): TeamEvidence(TeamLabel.TEAM_A, 0.95, "test"),
            tracklet_id(2): TeamEvidence(TeamLabel.TEAM_B, 0.95, "test"),
        },
        appearance_threshold=0.5,
    )

    assert not decision.candidate_match
    assert decision.rejection_reason == "confident_team_conflict"


def test_unknown_team_does_not_force_rejection():
    decision = score_candidate_pair(
        tracklet_id(1), tracklet_id(2),
        {tracklet_id(1): (1.0, 0.0), tracklet_id(2): (1.0, 0.0)}, (),
        {
            tracklet_id(1): TeamEvidence(TeamLabel.UNKNOWN, 0.0, "test"),
            tracklet_id(2): TeamEvidence(TeamLabel.TEAM_B, 0.95, "test"),
        }, appearance_threshold=0.5,
    )

    assert decision.team_compatible
    assert decision.candidate_match


def test_temporal_cannot_link_always_wins_over_identical_embedding():
    cannot_link = CannotLink(tracklet_id(1), tracklet_id(2), "simultaneous", 10, 2.0)

    decision = score_candidate_pair(
        tracklet_id(1), tracklet_id(2),
        {tracklet_id(1): (1.0, 0.0), tracklet_id(2): (1.0, 0.0)},
        (cannot_link,), {}, appearance_threshold=-1.0,
    )

    assert not decision.temporal_compatible
    assert not decision.candidate_match
    assert decision.rejection_reason == "temporal_cannot_link"


def test_cosine_similarity_is_deterministic():
    assert cosine_similarity((1.0, 0.0), (0.5, 0.5)) == cosine_similarity(
        (1.0, 0.0), (0.5, 0.5)
    )
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)


def test_tracklet_aggregation_is_normalized_and_deterministic(tmp_path: Path):
    first = write_crop(tmp_path, name="early-f000000.jpg")
    second = write_crop(tmp_path, name="late-f000010.jpg")
    second = ReviewSample(second.tracklet_id, "late", 10, 2.0, second.bbox_xyxy,
                          second.relative_path, second.sha256)
    reviews = ReviewSamplesArtifact(
        REVIEW_SAMPLES_SCHEMA, source(), "early_middle_late/v1", (first, second)
    )

    artifact = build_appearance_embeddings(reviews, tmp_path, MockEncoder())
    vector = artifact.tracklet_embeddings[0].embedding

    assert artifact.tracklet_embeddings[0].usable_crop_count == 2
    assert sum(value * value for value in vector) == pytest.approx(1.0)
    assert vector == build_appearance_embeddings(reviews, tmp_path, MockEncoder()).tracklet_embeddings[0].embedding


def test_unusable_tiny_crop_is_rejected_without_inventing_features(tmp_path: Path):
    sample = write_crop(tmp_path, size=(12, 8))
    reviews = ReviewSamplesArtifact(REVIEW_SAMPLES_SCHEMA, source(), "early_middle_late/v1", (sample,))

    artifact = build_appearance_embeddings(reviews, tmp_path, MockEncoder())

    assert not artifact.crop_embeddings
    assert not artifact.tracklet_embeddings
    assert artifact.rejections[0].reason == "crop_too_small"


def test_candidate_scoring_is_deterministic():
    kwargs = dict(
        left=tracklet_id(1), right=tracklet_id(2),
        embeddings={tracklet_id(1): (1.0, 0.0), tracklet_id(2): (0.8, 0.2)},
        cannot_links=(), team_evidence={}, appearance_threshold=0.8,
    )

    assert score_candidate_pair(**kwargs) == score_candidate_pair(**kwargs)


def test_threshold_selection_uses_labelled_development_pairs():
    selection = select_similarity_threshold(((0.9, True), (0.8, True), (0.7, False), (0.1, False)))

    assert selection.threshold == pytest.approx(0.8)
    assert selection.metrics["f1"] == pytest.approx(1.0)
    assert selection.metrics["positive_support"] == 2
    assert selection.metrics["negative_support"] == 2


def test_pairwise_metric_values_and_support_counts_are_correct():
    metrics = binary_pair_metrics((True, True, False, False), (True, False, True, False))

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["positive_support"] == 2
    assert metrics["negative_support"] == 2


def test_team_metrics_report_each_class_and_unknown_rejection():
    truth = {
        tracklet_id(1): TeamLabel.TEAM_A,
        tracklet_id(2): TeamLabel.TEAM_B,
        tracklet_id(3): TeamLabel.UNKNOWN,
    }
    evidence = {
        tracklet_id(1): TeamEvidence(TeamLabel.TEAM_A, 0.9, "test"),
        tracklet_id(2): TeamEvidence(TeamLabel.UNKNOWN, 0.0, "test"),
        tracklet_id(3): TeamEvidence(TeamLabel.UNKNOWN, 0.0, "test"),
    }

    metrics = team_classification_report(truth, evidence)

    assert metrics["TEAM_A"]["precision"] == pytest.approx(1.0)
    assert metrics["TEAM_A"]["recall"] == pytest.approx(1.0)
    assert metrics["TEAM_B"]["recall"] == pytest.approx(0.0)
    assert metrics["UNKNOWN"]["rejection_rate"] == pytest.approx(1.0)
    assert metrics["UNKNOWN"]["support"] == 1


def test_artifact_is_evidence_not_production_identity_output():
    assert "identity" not in APPEARANCE_EMBEDDINGS_SCHEMA
    assert not hasattr(AppearanceEmbeddingsArtifact, "player_identities")


def test_normal_ci_source_has_no_weight_download_or_yolo_import():
    root = Path(__file__).resolve().parents[2]
    sources = "".join(
        path.read_text(encoding="utf-8")
        for path in (root / "v2/src/footballai_v2/execution/ai2b").glob("*.py")
    )

    assert "load_state_dict_from_url" not in sources
    assert "urlretrieve" not in sources
    assert "requests.get" not in sources
    assert "from ultralytics" not in sources
    assert "import ultralytics" not in sources


def test_committed_benchmark_passes_dataset_gate_and_keeps_split_identity_disjoint():
    root = Path(__file__).resolve().parents[2]
    report = json.loads(
        (root / "evaluation/ai2b/belgium-japan-wide-90s/appearance-evaluation.json")
        .read_text(encoding="utf-8")
    )
    dataset = report["dataset"]

    assert dataset["gate"] == "PASS"
    assert dataset["known_identities"] >= 8
    assert dataset["team_a_identities"] > 0
    assert dataset["team_b_identities"] > 0
    assert dataset["positive_pair_support"] >= 20
    assert dataset["negative_pair_support"] > 0
    assert report["cached_pipeline"]["detector_executions"] == 1
    assert report["scope"]["persistent_identity_resolver"] is False
    assert set(report["split"]["development_identities"]).isdisjoint(
        report["split"]["evaluation_identities"]
    )
    for comparison in report["association_comparison"].values():
        assert comparison["positive_support"] > 0
        assert comparison["negative_support"] > 0


def test_committed_crop_references_are_relative_and_contain_no_pixels():
    root = Path(__file__).resolve().parents[2]
    value = json.loads(
        (root / "evaluation/ai2b/belgium-japan-wide-90s/review-samples.json")
        .read_text(encoding="utf-8")
    )

    assert value["samples"]
    assert all(not Path(item["relative_path"]).is_absolute() for item in value["samples"])
    assert not list((root / "evaluation/ai2b/belgium-japan-wide-90s").glob("*.mp4"))
    assert not list((root / "evaluation/ai2b/belgium-japan-wide-90s/crops").glob("**/*"))
