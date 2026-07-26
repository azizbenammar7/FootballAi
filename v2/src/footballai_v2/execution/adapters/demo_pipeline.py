"""Fast deterministic, explicitly synthetic workflow adapter."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Callable

from footballai_v2.contracts.v1 import AnalysisRun


class DemoPipeline:
    profile_id = "demo_fast"
    warning = (
        "Synthetic workflow result. This run demonstrates the full upload, processing, progress and dashboard "
        "workflow. Its analytical values are generated for development and must not be treated as match measurements."
    )

    def build_artifacts(self, run: AnalysisRun, duration_seconds: float, input_path: Path | None = None, cancellation_requested: Callable[[], bool] | None = None) -> dict[str, dict]:
        seed = int(hashlib.sha256(f"{run.input.sha256}:{run.parameters.get('match_name', '')}".encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        track_count = 6 + seed % 5
        players: dict[str, dict] = {}
        advisories: dict[str, dict] = {}
        for index in range(1, track_count + 1):
            distance = round(450 + rng.random() * 900, 2)
            speed = round(1.4 + rng.random() * 2.4, 2)
            peak = round(speed + 2 + rng.random() * 3, 2)
            blocks = max(1, min(4, int(max(duration_seconds, 60) // 900) + 1))
            players[str(index)] = {
                "track_id": index, "total_distance_m": distance, "mean_speed_ms": speed,
                "peak_speed_ms": peak, "total_sprints": int(rng.random() * 5),
                "active_time_s": round(max(10, duration_seconds * (.55 + rng.random() * .4)), 2),
                "coverage_frac": round(.65 + rng.random() * .33, 3), "blocks_present": list(range(blocks)),
                "heatmap": [[round(rng.random(), 3) for _ in range(8)] for _ in range(5)],
                "speed_timeline": {str(block): round(max(.3, speed + rng.uniform(-.6, .6)), 2) for block in range(blocks)},
            }
            score = round(.18 + rng.random() * .65, 3)
            level = "LOW" if score < .4 else "MEDIUM" if score < .7 else "HIGH"
            advisories[str(index)] = {
                "track_id": index, "risk_score": score, "risk_flag": level,
                "reason": "Deterministic synthetic workload pattern for workflow demonstration only.",
                "fatigue_indicators": {"relative_load": score, "coverage": players[str(index)]["coverage_frac"]},
                "score_breakdown": {"synthetic_load": score}, "data_sufficiency": "demonstration",
                "confidence": "synthetic", "advisory_only": True,
            }
        metadata = {key: run.parameters.get(key) for key in ("match_name", "home_team", "away_team", "competition", "match_date", "venue", "notes")}
        summary = {
            "schema": "footballai.team-summary/v1", "run_id": run.run_id, "generator_profile": self.profile_id,
            "generated_at": run.created_at.isoformat().replace("+00:00", "Z"), "data_quality": "synthetic",
            "warnings": [self.warning], "provenance": {"input_sha256": run.input.sha256}, "match_metadata": metadata,
            "match_duration_s": duration_seconds, "total_tracks": track_count, "players": players,
        }
        track_summary = {**summary, "schema": "footballai.track-summary/v1"}
        details = {
            "schema": "footballai.track-detail/v1", "run_id": run.run_id, "generator_profile": self.profile_id,
            "generated_at": summary["generated_at"], "data_quality": "synthetic", "warnings": [self.warning],
            "provenance": summary["provenance"], "tracks": players,
        }
        workload = {
            "schema": "footballai.workload-advisory/v1", "run_id": run.run_id, "generator_profile": self.profile_id,
            "generated_at": summary["generated_at"], "data_quality": "synthetic", "warnings": [self.warning],
            "provenance": summary["provenance"], "disclaimer": "Advisory only; not diagnosis or clinical advice.",
            "tracks": advisories,
        }
        diagnostics = {
            "schema": "footballai.analysis-diagnostics/v1", "run_id": run.run_id, "generator_profile": self.profile_id,
            "generated_at": summary["generated_at"], "data_quality": "synthetic", "warnings": [self.warning],
            "provenance": summary["provenance"], "deterministic_seed": seed, "input_count": 1,
            "output_count": track_count, "message": "No ML model was executed.",
        }
        return {"team-summary": summary, "track-summary": track_summary, "track-detail": details, "workload-advisory": workload, "analysis-diagnostics": diagnostics}
