"""Append-only, human-reviewed identity corrections with audit history.

A corrections log is pinned to one AI1 tracklets checksum. Entries are never
edited or deleted: a mistaken decision is withdrawn by appending a ``revoke``
entry, so the full review history stays inspectable. Only effective (not
revoked) decisions reach the resolver.

Corrections express reviewer assertions about tracklets; they never bypass
physical constraints. A ``must_link`` between temporally overlapping tracklets
is rejected by the resolver rather than silently applied.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from footballai_v2.execution.ai2a.model import TrackletId


CORRECTIONS_SCHEMA = "footballai.identity-corrections/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVIEWER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class CorrectionAction(str, Enum):
    MUST_LINK = "must_link"
    CANNOT_LINK = "cannot_link"
    EXCLUDE = "exclude"
    REVOKE = "revoke"


@dataclass(frozen=True, slots=True)
class CorrectionEntry:
    sequence: int
    action: CorrectionAction
    tracklet_ids: tuple[TrackletId, ...]
    reviewer: str
    reviewed_at: str
    reason: str
    revokes_sequence: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 1:
            raise ValueError("correction sequence must be a positive integer")
        if not isinstance(self.action, CorrectionAction):
            raise ValueError("correction action must be typed")
        if not isinstance(self.tracklet_ids, tuple) or not all(
            isinstance(item, TrackletId) for item in self.tracklet_ids
        ):
            raise ValueError("correction tracklet_ids must be a tuple of typed tracklet IDs")
        if len(set(self.tracklet_ids)) != len(self.tracklet_ids):
            raise ValueError("correction tracklet_ids must be distinct")
        if not isinstance(self.reviewer, str) or not _REVIEWER_RE.fullmatch(self.reviewer):
            raise ValueError("reviewer must be a pseudonymous reviewer handle")
        if not isinstance(self.reviewed_at, str) or not _TIMESTAMP_RE.fullmatch(self.reviewed_at):
            raise ValueError("reviewed_at must be an ISO-8601 UTC timestamp like 2026-11-06T10:00:00Z")
        datetime.strptime(self.reviewed_at, "%Y-%m-%dT%H:%M:%SZ")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("every correction requires a reason")
        if self.action in (CorrectionAction.MUST_LINK, CorrectionAction.CANNOT_LINK):
            if len(self.tracklet_ids) != 2:
                raise ValueError(f"{self.action.value} requires exactly two tracklets")
        elif self.action is CorrectionAction.EXCLUDE:
            if not self.tracklet_ids:
                raise ValueError("exclude requires at least one tracklet")
        elif self.tracklet_ids:
            raise ValueError("revoke must not name tracklets")
        if self.action is CorrectionAction.REVOKE:
            if (
                not isinstance(self.revokes_sequence, int)
                or isinstance(self.revokes_sequence, bool)
                or not 1 <= self.revokes_sequence < self.sequence
            ):
                raise ValueError("revoke must reference an earlier sequence")
        elif self.revokes_sequence is not None:
            raise ValueError("only revoke entries may reference another sequence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "action": self.action.value,
            "tracklet_ids": [item.value for item in self.tracklet_ids],
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "reason": self.reason,
            "revokes_sequence": self.revokes_sequence,
        }


@dataclass(frozen=True, slots=True)
class EffectiveCorrections:
    must_links: tuple[tuple[TrackletId, TrackletId, int], ...]
    cannot_links: tuple[tuple[TrackletId, TrackletId, int], ...]
    excluded: tuple[tuple[TrackletId, int], ...]


@dataclass(frozen=True, slots=True)
class IdentityCorrections:
    schema: str
    tracklets_sha256: str
    entries: tuple[CorrectionEntry, ...]

    def __post_init__(self) -> None:
        if self.schema != CORRECTIONS_SCHEMA:
            raise ValueError(f"schema must be {CORRECTIONS_SCHEMA}")
        if not isinstance(self.tracklets_sha256, str) or not _SHA256_RE.fullmatch(self.tracklets_sha256):
            raise ValueError("tracklets_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.entries, tuple):
            raise ValueError("entries must be a tuple")
        for expected, entry in enumerate(self.entries, start=1):
            if entry.sequence != expected:
                raise ValueError("correction sequences must be contiguous from 1 (append-only log)")
        by_sequence = {item.sequence: item for item in self.entries}
        revoked: set[int] = set()
        for entry in self.entries:
            if entry.action is not CorrectionAction.REVOKE:
                continue
            target = by_sequence[entry.revokes_sequence]
            if target.action is CorrectionAction.REVOKE:
                raise ValueError("a revoke entry cannot itself be revoked")
            if target.sequence in revoked:
                raise ValueError(f"sequence {target.sequence} is already revoked")
            revoked.add(target.sequence)
        self.effective()

    def effective(self) -> EffectiveCorrections:
        revoked = {
            item.revokes_sequence for item in self.entries if item.action is CorrectionAction.REVOKE
        }
        live = [
            item
            for item in self.entries
            if item.action is not CorrectionAction.REVOKE and item.sequence not in revoked
        ]
        must = tuple(
            (*sorted(item.tracklet_ids), item.sequence)
            for item in live
            if item.action is CorrectionAction.MUST_LINK
        )
        cannot = tuple(
            (*sorted(item.tracklet_ids), item.sequence)
            for item in live
            if item.action is CorrectionAction.CANNOT_LINK
        )
        excluded: dict[TrackletId, int] = {}
        for item in live:
            if item.action is CorrectionAction.EXCLUDE:
                for identifier in item.tracklet_ids:
                    excluded.setdefault(identifier, item.sequence)
        must_pairs = {frozenset(item[:2]) for item in must}
        conflicting = must_pairs & {frozenset(item[:2]) for item in cannot}
        if conflicting:
            pair = sorted(next(iter(conflicting)))
            raise ValueError(
                f"effective corrections both link and separate {pair[0].value} and {pair[1].value}; "
                "revoke one decision first"
            )
        for left, right, _ in must:
            if left in excluded or right in excluded:
                raise ValueError("an excluded tracklet cannot also be must-linked; revoke one decision first")
        return EffectiveCorrections(must, cannot, tuple(sorted(excluded.items())))


def corrections_to_json_bytes(corrections: IdentityCorrections) -> bytes:
    value = {
        "schema": corrections.schema,
        "tracklets_sha256": corrections.tracklets_sha256,
        "entries": [item.to_dict() for item in corrections.entries],
    }
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def corrections_from_json_bytes(content: bytes) -> IdentityCorrections:
    try:
        value = json.loads(content)
        entries = tuple(
            CorrectionEntry(
                sequence=item["sequence"],
                action=CorrectionAction(item["action"]),
                tracklet_ids=tuple(TrackletId(identifier) for identifier in item.get("tracklet_ids", [])),
                reviewer=item["reviewer"],
                reviewed_at=item["reviewed_at"],
                reason=item["reason"],
                revokes_sequence=item.get("revokes_sequence"),
            )
            for item in value["entries"]
        )
        return IdentityCorrections(value["schema"], value["tracklets_sha256"], entries)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid identity corrections artifact") from exc
