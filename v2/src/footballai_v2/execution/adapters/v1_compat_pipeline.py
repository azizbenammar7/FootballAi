"""Optional genuine V1 algorithm-family adapter isolated inside a V2 run."""

from __future__ import annotations

import importlib.util
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
from footballai_v2.execution.errors import CancellationObserved, ExecutionFailure


V1_WARNING = (
    "V1-compatible analysis. Track identities are unverified, positions are not homography-calibrated, "
    "and Workload and Fatigue Advisory outputs are heuristic and advisory only."
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def _v1_missing() -> list[str]:
    requirements = {
        "ultralytics": "ultralytics", "OpenCV": "cv2", "pandas": "pandas", "NumPy": "numpy",
        "SciPy": "scipy", "PyArrow": "pyarrow", "tqdm": "tqdm",
    }
    missing = [label for label, module in requirements.items() if importlib.util.find_spec(module) is None]
    if not (REPOSITORY_ROOT / "yolov8m.pt").is_file():
        missing.append("local yolov8m.pt weights")
    return missing


def profile_catalog(*, include_test: bool = False) -> list[dict]:
    missing = _v1_missing()
    result = [
        {"profile_id": "demo_fast", "display_name": "Demo fast", "description": "Deterministic synthetic workflow for local development and CI.", "available": True, "missing_requirements": [], "warnings": [DemoPipeline.warning], "purpose": "Workflow and dashboard validation", "gpu": "not_required"},
        {"profile_id": "v1_compat", "display_name": "V1-compatible analysis", "description": "Local YOLOv8 and ByteTrack algorithm-family adapter writing only to the V2 run namespace.", "available": not missing, "missing_requirements": missing, "warnings": [V1_WARNING], "purpose": "Genuine local computer-vision compatibility path", "gpu": "optional"},
    ]
    if include_test:
        result.append({"profile_id": "test_fail", "display_name": "Test failure", "description": "Deterministic test-only failure profile.", "available": True, "missing_requirements": [], "warnings": [], "purpose": "Automated tests", "gpu": "not_required"})
    return result


class V1CompatPipeline:
    """Run preserved V1 scripts in a private working tree under the V2 run."""

    profile_id = "v1_compat"
    warning = V1_WARNING

    def build_artifacts(self, run: AnalysisRun, duration_seconds: float, input_path: Path | None = None, cancellation_requested: Callable[[], bool] | None = None) -> dict[str, dict]:
        if input_path is None:
            raise ExecutionFailure("v1_input_missing", "The V1-compatible input is unavailable.", retryable=False)
        missing = _v1_missing()
        if missing:
            raise ExecutionFailure("v1_requirements_missing", "The V1-compatible profile is unavailable on this machine.", retryable=False)
        run_dir = input_path.parent.parent
        work = run_dir / "tmp" / "v1-compat"
        processed = work / "data" / "processed"; pipeline_dir = work / "pipeline"
        processed.mkdir(parents=True, exist_ok=True); pipeline_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / "pipeline" / "bytetrack_custom.yaml", pipeline_dir / "bytetrack_custom.yaml")
        os.link(REPOSITORY_ROOT / "yolov8m.pt", work / "yolov8m.pt")
        log_path = run_dir / "logs" / "v1-compat.log"
        commands = [
            [sys.executable, str(REPOSITORY_ROOT / "pipeline" / "01_track.py"), "--video", str(input_path), "--output_dir", str(processed)],
            [sys.executable, str(REPOSITORY_ROOT / "pipeline" / "02_stats.py")],
            [sys.executable, str(REPOSITORY_ROOT / "pipeline" / "03_fatigue.py")],
        ]
        timeout = float(os.getenv("FOOTBALLAI_V1_SUBPROCESS_TIMEOUT_SECONDS", "7200"))
        try:
            with log_path.open("ab") as log:
                for command in commands:
                    self._run(command, work, log, timeout, cancellation_requested)
        finally:
            self._truncate_log(log_path, int(os.getenv("FOOTBALLAI_MAX_RUN_LOG_BYTES", str(2 * 1024 * 1024))))
        try:
            summary = json.loads((processed / "player_summary.json").read_text(encoding="utf-8"))
            advisory = json.loads((processed / "risk_scores.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutionFailure("v1_output_invalid", "V1-compatible processing did not produce valid analytical outputs.") from exc
        generated_at = run.created_at.isoformat().replace("+00:00", "Z")
        provenance = {"input_sha256": run.input.sha256, "algorithm_family": "technical-test-v1.0", "adapter": "v1_compat"}
        common = {"run_id": run.run_id, "generator_profile": self.profile_id, "generated_at": generated_at, "data_quality": "unverified", "warnings": [self.warning], "provenance": provenance}
        track_summary = {**summary, **common, "schema": "footballai.track-summary/v1"}
        team_summary = {**summary, **common, "schema": "footballai.team-summary/v1", "match_metadata": {key: run.parameters.get(key) for key in ("match_name", "home_team", "away_team", "competition", "match_date", "venue")}}
        track_detail = {**common, "schema": "footballai.track-detail/v1", "tracks": summary.get("players", {})}
        workload = {**common, "schema": "footballai.workload-advisory/v1", "tracks": advisory, "disclaimer": "Advisory only; not diagnosis or clinical advice."}
        diagnostics = {**common, "schema": "footballai.analysis-diagnostics/v1", "input_count": 1, "output_count": len(summary.get("players", {})), "message": "Preserved V1 scripts executed in an isolated V2 workspace."}
        return {"team-summary": team_summary, "track-summary": track_summary, "track-detail": track_detail, "workload-advisory": workload, "analysis-diagnostics": diagnostics}

    @staticmethod
    def _run(command: list[str], cwd: Path, log, timeout: float, cancellation_requested: Callable[[], bool] | None) -> None:
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=cwd, shell=False, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if cancellation_requested and cancellation_requested():
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise CancellationObserved()
            if time.monotonic() - started > timeout:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise ExecutionFailure("v1_subprocess_timeout", "V1-compatible processing exceeded its local time limit.")
            time.sleep(.25)
        if process.returncode != 0:
            raise ExecutionFailure("v1_subprocess_failed", "V1-compatible processing stopped safely.")

    @staticmethod
    def _truncate_log(path: Path, limit: int) -> None:
        limit = max(1024, min(limit, 10 * 1024 * 1024))
        if path.exists() and path.stat().st_size > limit:
            with path.open("rb") as source:
                source.seek(-limit, os.SEEK_END); tail = source.read(limit)
            with path.open("wb") as target:
                target.write(tail)
