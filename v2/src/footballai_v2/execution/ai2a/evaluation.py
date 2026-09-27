"""Reproducible AI2A baseline report from manual labels and bounded evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Mapping

from footballai_v2.execution.ai1.artifacts import TrackletsArtifact
from footballai_v2.execution.ai2a.association import (
    CandidateGraph,
    CannotLink,
    TeamEvidence,
)
from footballai_v2.execution.ai2a.metrics import (
    UNSUPPORTED_METRICS,
    candidate_pair_recall,
    pairwise_association_metrics,
    team_classification_metrics,
)
from footballai_v2.execution.ai2a.model import IdentityGroundTruthArtifact, TrackletId


EVALUATION_SCHEMA = "footballai.identity-evaluation/v1"


def build_evaluation_report(
    ground_truth: IdentityGroundTruthArtifact,
    tracklets: TrackletsArtifact,
    team_evidence: Mapping[TrackletId, TeamEvidence],
    cannot_links: tuple[CannotLink, ...],
    graph: CandidateGraph,
    *,
    team_prototype_tracklets: Mapping[str, TrackletId],
    annotation_effort_minutes: float,
) -> dict:
    candidate_pairs = {frozenset((item.left, item.right)) for item in graph.edges}
    predicted_pairs = {
        frozenset((item.left, item.right)) for item in graph.edges if item.baseline_match
    }
    pairwise = pairwise_association_metrics(ground_truth.annotations, predicted_pairs)
    candidates = candidate_pair_recall(ground_truth.annotations, candidate_pairs)
    teams = team_classification_metrics(ground_truth.annotations, team_evidence)
    identity_count = len({
        item.ground_truth_player_id
        for item in ground_truth.annotations
        if item.ground_truth_player_id is not None
    })
    unknown = sum(item.ground_truth_player_id is None for item in ground_truth.annotations)
    uncertain = sum(item.status.value == "uncertain" for item in ground_truth.annotations)
    measured = {
        "pairwise_association": pairwise,
        "candidate_pair_recall": candidates,
        "team_classification": teams,
    }
    return {
        "schema": EVALUATION_SCHEMA,
        "source": asdict(ground_truth.source),
        "scope": {
            "purpose": "identity_evaluation_foundation",
            "persistent_identity_resolver": False,
            "clip_is_long_enough_for_long_term_identity_claims": False,
        },
        "annotation": {
            "effort_minutes": float(annotation_effort_minutes),
            "tracklets": len(tracklets.tracklets),
            "annotated_tracklets": len(ground_truth.annotations),
            "ground_truth_identities": identity_count,
            "unknown_tracklets": unknown,
            "uncertain_tracklets": uncertain,
        },
        "team_baseline": {
            "method": "central_upper_body_hsv_nearest_manual_anchor/v1",
            "prototype_tracklets": {key: value.value for key, value in sorted(team_prototype_tracklets.items())},
            "evidence": {
                key.value: {"team": value.team.value, "confidence": value.confidence, "method": value.method}
                for key, value in sorted(team_evidence.items())
            },
        },
        "temporal_constraints": {
            "rule": "shared sampled frame with non-overlapping pixel boxes",
            "cannot_link_count": len(cannot_links),
            "cannot_links": [
                {
                    **asdict(item),
                    "left": item.left.value,
                    "right": item.right.value,
                }
                for item in cannot_links
            ],
        },
        "candidate_graph": {
            "schema": graph.schema,
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "baseline_positive_edges": len(predicted_pairs),
            "edge_evidence_channels": ["temporal_compatibility", "team_compatibility", "jersey_color_similarity"],
            "pairs": [
                {
                    **asdict(item),
                    "left": item.left.value,
                    "right": item.right.value,
                }
                for item in graph.edges
            ],
        },
        "evaluation_metrics": {
            "measured": measured,
            "not_measured": {
                "identity_observation_coverage": "No resolver output exists in AI2A.",
                **{name: "Annotation semantics do not support this tracking metric." for name in sorted(UNSUPPORTED_METRICS)},
            },
        },
    }


def evaluation_to_json_bytes(report: dict) -> bytes:
    if report.get("schema") != EVALUATION_SCHEMA:
        raise ValueError(f"schema must be {EVALUATION_SCHEMA}")
    return (json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
