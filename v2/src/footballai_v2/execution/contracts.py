"""Provider-neutral execution job records."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from footballai_v2.contracts.v1 import parse_utc_datetime, utc_now, validate_run_id, validate_uuid_v4


@dataclass(frozen=True, slots=True)
class ExecutionJob:
    job_id: str
    run_id: str
    logical_analysis_id: str
    attempt_number: int
    pipeline_profile: str
    created_at: datetime
    claimed_at: datetime | None = None
    worker_id: str | None = None

    @classmethod
    def new(cls, run_id: str, logical_analysis_id: str, attempt_number: int, profile: str) -> "ExecutionJob":
        return cls(str(uuid.uuid4()), run_id, logical_analysis_id, attempt_number, profile, utc_now())

    def __post_init__(self) -> None:
        validate_uuid_v4(self.job_id, "job_id")
        validate_run_id(self.run_id)
        validate_uuid_v4(self.logical_analysis_id, "logical_analysis_id")
        if not isinstance(self.attempt_number, int) or self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if self.pipeline_profile not in {"demo_fast", "v1_compat", "test_fail"}:
            raise ValueError("unknown pipeline profile")
        if self.worker_id is not None and (not self.worker_id or len(self.worker_id) > 128):
            raise ValueError("invalid worker_id")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat().replace("+00:00", "Z")
        payload["claimed_at"] = self.claimed_at.isoformat().replace("+00:00", "Z") if self.claimed_at else None
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExecutionJob":
        expected = {"job_id", "run_id", "logical_analysis_id", "attempt_number", "pipeline_profile", "created_at", "claimed_at", "worker_id"}
        if set(value) != expected:
            raise ValueError("malformed queue record")
        return cls(
            job_id=value["job_id"], run_id=value["run_id"], logical_analysis_id=value["logical_analysis_id"],
            attempt_number=value["attempt_number"], pipeline_profile=value["pipeline_profile"],
            created_at=parse_utc_datetime(value["created_at"]),
            claimed_at=parse_utc_datetime(value["claimed_at"]) if value["claimed_at"] else None,
            worker_id=value["worker_id"],
        )
