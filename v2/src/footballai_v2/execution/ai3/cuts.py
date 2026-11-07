"""Suspected camera cuts from AI1 tracklets (invalidation-only evidence).

Same rule as the AI2C resolver's shot-cut barrier: a sampled frame where at
least ``min_births`` tracklets start at once (after the first frame) suggests
a cut or replay. It has not been evaluated as a shot-boundary detector, so AI3
uses it only to *remove* calibration (frames between keyframes separated by a
suspected cut become ``INVALIDATED``); it never creates or extends one.
"""

from __future__ import annotations

from collections import Counter

from footballai_v2.execution.ai1.artifacts import TrackletsArtifact


BIRTH_BURST_RULE = "tracklet_birth_burst/v1"


def suspected_cuts_from_tracklets(tracklets: TrackletsArtifact, *, min_births: int = 8) -> tuple[int, ...]:
    if not isinstance(min_births, int) or isinstance(min_births, bool) or min_births < 2:
        raise ValueError("min_births must be an integer of at least two")
    if not tracklets.tracklets:
        return ()
    births = Counter(item.start_frame for item in tracklets.tracklets)
    first = min(births)
    return tuple(sorted(frame for frame, count in births.items() if count >= min_births and frame > first))


def birth_burst_source(min_births: int) -> str:
    return f"{BIRTH_BURST_RULE}:min_births={min_births}"


def analysed_frames(tracklets: TrackletsArtifact) -> tuple[tuple[int, float], ...]:
    """Every (frame_index, timestamp) that carries at least one tracklet observation."""
    frames: dict[int, float] = {}
    for tracklet in tracklets.tracklets:
        for item in tracklet.observations:
            existing = frames.setdefault(item.frame_index, item.timestamp_seconds)
            if existing != item.timestamp_seconds:
                raise ValueError(f"frame {item.frame_index} has two timestamps in the tracklets artifact")
    return tuple(sorted(frames.items()))
