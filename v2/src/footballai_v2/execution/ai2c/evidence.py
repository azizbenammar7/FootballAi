"""Load already-computed AI2A evidence (team guard, cannot-links) by checksum.

AI2C does not recompute AI2A evidence. It reads the committed
``footballai.identity-evaluation/v1`` report and refuses one whose tracklets
checksum differs from the resolver input.
"""

from __future__ import annotations

import json

from footballai_v2.execution.ai1.artifacts import sha256_bytes
from footballai_v2.execution.ai2a.association import CannotLink, TeamEvidence
from footballai_v2.execution.ai2a.evaluation import EVALUATION_SCHEMA
from footballai_v2.execution.ai2a.model import TeamLabel, TrackletId
from footballai_v2.execution.ai2c.artifact import EvidenceInput


def ai2a_evidence_from_evaluation_bytes(
    content: bytes, *, tracklets_sha256: str | None
) -> tuple[dict[TrackletId, TeamEvidence], tuple[CannotLink, ...], tuple[EvidenceInput, ...]]:
    try:
        value = json.loads(content)
        if value.get("schema") != EVALUATION_SCHEMA:
            raise ValueError(f"schema must be {EVALUATION_SCHEMA}")
        if tracklets_sha256 is None or value["source"]["tracklets_sha256"] != tracklets_sha256:
            raise ValueError("AI2A evidence is pinned to a different tracklets artifact")
        teams = {
            TrackletId(key): TeamEvidence(TeamLabel(item["team"]), float(item["confidence"]), item["method"])
            for key, item in value["team_baseline"]["evidence"].items()
        }
        links = tuple(
            CannotLink(
                left=TrackletId(item["left"]),
                right=TrackletId(item["right"]),
                reason=item["reason"],
                frame_index=item["frame_index"],
                timestamp_seconds=item["timestamp_seconds"],
            )
            for item in value["temporal_constraints"]["cannot_links"]
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid AI2A evaluation evidence") from exc
    digest = sha256_bytes(content)
    inputs = (
        EvidenceInput("team_evidence", EVALUATION_SCHEMA, digest),
        EvidenceInput("cannot_links", EVALUATION_SCHEMA, digest),
    )
    return teams, links, inputs
