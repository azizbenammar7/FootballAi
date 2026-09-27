"""Ground-truth-free, deterministic tracklet fragmentation diagnostics."""

from __future__ import annotations

import math
import statistics
from typing import Iterable

from footballai_v2.execution.ai1.artifacts import DetectionsArtifact, TrackletsArtifact


DIAGNOSTICS_SCHEMA = "footballai.analysis-diagnostics/v2"
SHORT_THRESHOLDS_SECONDS = (1, 5, 10)
INSUFFICIENT_OBSERVATIONS = 50


def _round(value: float) -> float:
    return round(float(value), 6)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: Iterable[float]) -> dict[str, float]:
    items = [float(value) for value in values]
    if not items:
        return {key: 0.0 for key in ("min", "median", "mean", "p90", "p95", "max")}
    return {
        "min": _round(min(items)),
        "median": _round(statistics.median(items)),
        "mean": _round(statistics.fmean(items)),
        "p90": _round(_percentile(items, .90)),
        "p95": _round(_percentile(items, .95)),
        "max": _round(max(items)),
    }


def _percentage(count: int, total: int) -> float:
    return _round(count / total * 100) if total else 0.0


def _trend(tracklets: TrackletsArtifact, duration: float, bin_seconds: float = 300.0) -> list[dict]:
    if duration <= 0:
        return []
    bins = max(1, math.ceil(duration / bin_seconds))
    result = []
    for index in range(bins):
        start = index * bin_seconds
        end = min(duration, (index + 1) * bin_seconds)
        started = [item for item in tracklets.tracklets if start <= item.start_time_seconds < end]
        ended = [item for item in tracklets.tracklets if start <= item.end_time_seconds < end]
        overlapping = [item for item in tracklets.tracklets if item.start_time_seconds < end and item.end_time_seconds >= start]
        short = sum(item.duration_seconds < 10 for item in started)
        result.append({
            "start_seconds": _round(start), "end_seconds": _round(end),
            "starts": len(started), "ends": len(ended), "overlapping_tracklets": len(overlapping),
            "median_started_tracklet_duration_seconds": _round(statistics.median(
                [item.duration_seconds for item in started]
            )) if started else 0.0,
            "short_started_tracklet_percentage": _percentage(short, len(started)),
        })
    return result


def compute_fragmentation_diagnostics(
    detections: DetectionsArtifact,
    tracklets: TrackletsArtifact,
) -> dict:
    """Return observed proxies only; no labelled identity metric is inferred."""
    count = len(tracklets.tracklets)
    durations = [item.duration_seconds for item in tracklets.tracklets]
    observations = [item.observation_count for item in tracklets.tracklets]
    duration = detections.provenance.source_duration_seconds
    assigned = tracklets.observation_count
    insufficient = sum(value < INSUFFICIENT_OBSERVATIONS for value in observations)
    short = {
        f"{threshold}_seconds": {
            "criterion": f"duration_seconds < {threshold}",
            "count": sum(value < threshold for value in durations),
            "percentage": _percentage(sum(value < threshold for value in durations), count),
        }
        for threshold in SHORT_THRESHOLDS_SECONDS
    }
    metrics = {
        "total_detections": len(detections.detections),
        "total_tracklets": count,
        "tracklet_duration_seconds": _distribution(durations),
        "observations_per_tracklet": {
            "median": _round(statistics.median(observations)) if observations else 0.0,
            "mean": _round(statistics.fmean(observations)) if observations else 0.0,
        },
        "short_tracklets": short,
        "insufficient_tracklets": {
            "criterion": f"observation_count < {INSUFFICIENT_OBSERVATIONS}",
            "count": insufficient,
            "percentage": _percentage(insufficient, count),
        },
        "tracklets_per_video_minute": _round(count / (duration / 60)) if duration else 0.0,
        "track_start_rate_per_minute": _round(count / (duration / 60)) if duration else 0.0,
        "track_end_rate_per_minute": _round(count / (duration / 60)) if duration else 0.0,
        "detection_assignment_fraction": _round(assigned / len(detections.detections)) if detections.detections else 0.0,
        "active_simultaneous_tracklets": _active_distribution(tracklets),
        "fragmentation_trend": _trend(tracklets, duration),
    }
    severe = count > 0 and insufficient / count >= .8
    return {
        "schema": DIAGNOSTICS_SCHEMA,
        "observed_metrics": metrics,
        "interpretation": {
            "status": "descriptive_only",
            "summary": (
                "Most temporary tracks do not meet the existing 50-observation compatibility threshold."
                if severe else "Observed fragmentation proxies are reported without identity-quality claims."
            ),
        },
        "ground_truth_metrics": {
            "status": "not_measured",
            "reason": "No labelled persistent-identity ground truth is associated with this run.",
        },
    }


def _active_distribution(tracklets: TrackletsArtifact) -> dict[str, float]:
    times = sorted({
        item.timestamp_seconds
        for tracklet in tracklets.tracklets
        for item in tracklet.observations
    })
    active = [
        sum(
            any(observation.timestamp_seconds == timestamp for observation in tracklet.observations)
            for tracklet in tracklets.tracklets
        )
        for timestamp in times
    ]
    if not active:
        return {"median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "median": _round(statistics.median(active)),
        "mean": _round(statistics.fmean(active)),
        "p95": _round(_percentile([float(value) for value in active], .95)),
        "max": _round(max(active)),
    }
