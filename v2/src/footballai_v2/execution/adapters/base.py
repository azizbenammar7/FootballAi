"""Pipeline adapter protocol."""

from pathlib import Path
from typing import Callable, Protocol

from footballai_v2.contracts.v1 import AnalysisRun


class PipelineAdapter(Protocol):
    profile_id: str
    def build_artifacts(self, run: AnalysisRun, duration_seconds: float, input_path: Path | None = None, cancellation_requested: Callable[[], bool] | None = None) -> dict[str, dict]: ...
