"""Optional genuine V1 algorithm-family adapter isolated inside a V2 run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from footballai_v2.contracts.v1 import AnalysisRun
from footballai_v2.execution.adapters.demo_pipeline import DemoPipeline
from footballai_v2.execution.adapters.v1_compat_runtime import (
    REPOSITORY_ROOT,
    V1CompatConfig,
    check_v1_compat_readiness,
    configured_model_path,
    sha256_file,
    validate_model_file,
)
from footballai_v2.execution.errors import CancellationObserved, ExecutionFailure


V1_WARNING = (
    "V1-compatible analysis. Track identities are unverified, positions are not homography-calibrated, "
    "and Workload and Fatigue Advisory outputs are heuristic and advisory only."
)


def _v1_missing() -> list[str]:
    """Backward-compatible helper retained for callers and characterization tests."""
    return list(check_v1_compat_readiness().missing_requirements)


def profile_catalog(*, include_test: bool = False) -> list[dict]:
    readiness = check_v1_compat_readiness()
    result = [
        {
            "profile_id": "demo_fast",
            "display_name": "Demo fast",
            "description": "Deterministic synthetic workflow for local development and CI.",
            "available": True,
            "readiness_status": "ready",
            "readiness_message": "Demo fast is ready.",
            "setup_command": None,
            "missing_requirements": [],
            "runtime_errors": [],
            "runtime": {"device": "not_required", "model": None},
            "warnings": [DemoPipeline.warning],
            "purpose": "Workflow and dashboard validation",
            "gpu": "not_required",
        },
        {
            "profile_id": "v1_compat",
            "display_name": "V1-compatible analysis",
            "description": "Local YOLOv8m and ByteTrack preserved-algorithm compatibility profile writing only to the V2 run namespace.",
            **readiness.public_dict(),
            "warnings": [V1_WARNING],
            "purpose": "Genuine local computer-vision compatibility path",
            "gpu": "optional",
        },
    ]
    if include_test:
        result.append({
            "profile_id": "test_fail", "display_name": "Test failure",
            "description": "Deterministic test-only failure profile.", "available": True,
            "readiness_status": "ready", "readiness_message": "Test profile is ready.",
            "setup_command": None, "missing_requirements": [], "runtime_errors": [],
            "runtime": {"device": "not_required", "model": None}, "warnings": [],
            "purpose": "Automated tests", "gpu": "not_required",
        })
    return result


class V1CompatPipeline:
    """Run V1-compatible stages in a private working tree under the V2 run."""

    profile_id = "v1_compat"
    warning = V1_WARNING

    def build_artifacts(
        self,
        run: AnalysisRun,
        duration_seconds: float,
        input_path: Path | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> dict[str, dict]:
        if input_path is None:
            raise ExecutionFailure("v1_input_missing", "The V1-compatible input is unavailable.", retryable=False)
        readiness = check_v1_compat_readiness()
        if not readiness.ready or readiness.config is None:
            raise ExecutionFailure(
                "v1_requirements_missing",
                f"The V1-compatible profile is unavailable ({readiness.status}). Run make v2-v1-compat-setup.",
                retryable=False,
            )
        config = self._execution_config(run, readiness.config)
        run_dir = input_path.parent.parent
        work = run_dir / "tmp" / "v1-compat"
        processed = work / "data" / "processed"
        pipeline_dir = work / "pipeline"
        settings_dir = work / "ultralytics-settings"
        cache_dir = work / "cache"
        processed.mkdir(parents=True, exist_ok=True)
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        settings_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tracker_path = pipeline_dir / "bytetrack_custom.yaml"
        shutil.copy2(REPOSITORY_ROOT / "pipeline" / "bytetrack_custom.yaml", tracker_path)
        log_path = run_dir / "logs" / "v1-compat.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        commands = [
            [
                sys.executable, "-m", "footballai_v2.execution.adapters.v1_compat_tracking",
                "--video", str(input_path), "--output-dir", str(processed),
                "--model", str(config.model_path), "--tracker", str(tracker_path),
                "--device", config.selected_device, "--target-fps", str(config.target_fps),
                "--image-size", str(config.image_size), "--confidence", str(config.confidence),
            ],
        ]
        timeout = float(os.getenv("FOOTBALLAI_V1_SUBPROCESS_TIMEOUT_SECONDS", "7200"))
        child_environment = os.environ.copy()
        child_environment.update({
            "FOOTBALLAI_V1_COMPAT_MODEL_PATH": str(config.model_path),
            "FOOTBALLAI_V1_COMPAT_DEVICE": config.selected_device,
            "YOLO_OFFLINE": "true",
            "ULTRALYTICS_SETTINGS_DIR": str(settings_dir),
            "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
            "XDG_CACHE_HOME": str(cache_dir),
            "PYTHONPATH": str(REPOSITORY_ROOT / "v2" / "src"),
        })
        try:
            with log_path.open("ab") as log:
                self._run(commands[0], work, log, timeout, cancellation_requested, child_environment, config.selected_device)
                tracking = self._tracking_summary(processed)
                if not tracking["empty_after_v1_filters"]:
                    self._run(
                        [sys.executable, str(REPOSITORY_ROOT / "pipeline" / "02_stats.py")],
                        work, log, timeout, cancellation_requested, child_environment, config.selected_device,
                    )
                    self._run(
                        [sys.executable, str(REPOSITORY_ROOT / "pipeline" / "03_fatigue.py")],
                        work, log, timeout, cancellation_requested, child_environment, config.selected_device,
                    )
        finally:
            self._truncate_log(log_path, int(os.getenv("FOOTBALLAI_MAX_RUN_LOG_BYTES", str(2 * 1024 * 1024))))

        tracking = self._tracking_summary(processed)
        if tracking["empty_after_v1_filters"]:
            summary = {"match_duration_s": duration_seconds, "total_tracks": 0, "players": {}}
            advisory: dict = {}
            result_message = "Real YOLOv8m and ByteTrack execution completed with no tracks surviving the preserved V1 filters."
        else:
            try:
                summary = json.loads((processed / "player_summary.json").read_text(encoding="utf-8"))
                advisory = json.loads((processed / "risk_scores.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ExecutionFailure("v1_output_invalid", "V1-compatible processing did not produce valid analytical outputs.") from exc
            result_message = "Preserved V1 statistics and fatigue stages executed in an isolated V2 workspace."

        generated_at = run.created_at.isoformat().replace("+00:00", "Z")
        provenance = {
            "input_sha256": run.input.sha256,
            "algorithm_family": "technical-test-v1.0",
            "adapter": "v1_compat",
            **config.public_dict(),
        }
        common = {
            "run_id": run.run_id, "generator_profile": self.profile_id,
            "generated_at": generated_at, "data_quality": "unverified",
            "warnings": [self.warning], "provenance": provenance,
        }
        track_summary = {**summary, **common, "schema": "footballai.track-summary/v1"}
        team_summary = {
            **summary, **common, "schema": "footballai.team-summary/v1",
            "match_metadata": {key: run.parameters.get(key) for key in ("match_name", "home_team", "away_team", "competition", "match_date", "venue")},
        }
        track_detail = {**common, "schema": "footballai.track-detail/v1", "tracks": summary.get("players", {})}
        workload = {
            **common, "schema": "footballai.workload-advisory/v1", "tracks": advisory,
            "disclaimer": "Advisory only; not diagnosis or clinical advice.",
        }
        diagnostics = {
            **common, "schema": "footballai.analysis-diagnostics/v1", "input_count": 1,
            "output_count": len(summary.get("players", {})), "message": result_message,
            "tracking": tracking,
        }
        return {
            "team-summary": team_summary, "track-summary": track_summary,
            "track-detail": track_detail, "workload-advisory": workload,
            "analysis-diagnostics": diagnostics,
        }

    @staticmethod
    def _execution_config(run: AnalysisRun, current: V1CompatConfig) -> V1CompatConfig:
        recorded = run.parameters.get("v1_compat")
        if not isinstance(recorded, dict):
            return current
        expected_checksum = recorded.get("model_sha256")
        model_path = configured_model_path()
        valid, _reason = validate_model_file(model_path)
        if not valid or sha256_file(model_path) != expected_checksum:
            raise ExecutionFailure(
                "v1_model_changed",
                "The configured YOLOv8m weights no longer match this run's recorded model checksum.",
                retryable=False,
            )
        return V1CompatConfig(
            target_fps=float(recorded["target_fps"]),
            image_size=int(recorded["image_size"]),
            confidence=float(recorded["confidence"]),
            requested_device=str(recorded["requested_device"]),
            selected_device=str(recorded["selected_device"]),
            model_path=model_path.resolve(),
            model_sha256=str(expected_checksum),
        )

    @staticmethod
    def _tracking_summary(processed: Path) -> dict:
        try:
            value = json.loads((processed / "tracking_summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutionFailure("v1_tracking_output_invalid", "V1-compatible tracking did not produce valid diagnostics.") from exc
        required = {"frames_processed", "detection_rows", "tracked_ids", "max_track_observations", "empty_after_v1_filters"}
        if set(value) != required:
            raise ExecutionFailure("v1_tracking_output_invalid", "V1-compatible tracking diagnostics are incomplete.")
        return value

    @staticmethod
    def _run(
        command: list[str], cwd: Path, log, timeout: float,
        cancellation_requested: Callable[[], bool] | None,
        environment: dict[str, str], selected_device: str,
    ) -> None:
        started = time.monotonic()
        process = subprocess.Popen(
            command, cwd=cwd, env=environment, shell=False,
            stdout=log, stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            if cancellation_requested and cancellation_requested():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise CancellationObserved()
            if time.monotonic() - started > timeout:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise ExecutionFailure("v1_subprocess_timeout", "V1-compatible processing exceeded its local time limit.")
            time.sleep(.25)
        if process.returncode != 0:
            if selected_device == "mps":
                raise ExecutionFailure(
                    "v1_mps_execution_failed",
                    "Apple MPS execution failed safely. Review the run log and explicitly choose cpu only if the slower runtime is acceptable.",
                )
            raise ExecutionFailure("v1_subprocess_failed", "V1-compatible processing stopped safely.")

    @staticmethod
    def _truncate_log(path: Path, limit: int) -> None:
        limit = max(1024, min(limit, 10 * 1024 * 1024))
        if path.exists() and path.stat().st_size > limit:
            with path.open("rb") as source:
                source.seek(-limit, os.SEEK_END)
                tail = source.read(limit)
            with path.open("wb") as target:
                target.write(tail)
