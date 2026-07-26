"""Persistent cancellation requests shared by API and workers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from footballai_v2.contracts.v1 import utc_now, validate_run_id


class CancellationStore:
    NAME = "cancel-request.json"

    def __init__(self, run_root: str | Path) -> None:
        self.run_root = Path(run_root).expanduser().resolve()

    def request(self, run_id: str) -> None:
        validate_run_id(run_id)
        run_dir = (self.run_root / run_id).resolve()
        run_dir.relative_to(self.run_root)
        payload = json.dumps({"requested_at": utc_now().isoformat().replace("+00:00", "Z")}) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=".cancel-", dir=run_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, run_dir / self.NAME)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def requested(self, run_id: str) -> bool:
        validate_run_id(run_id)
        marker = self.run_root / run_id / self.NAME
        return marker.is_file() and not marker.is_symlink()
