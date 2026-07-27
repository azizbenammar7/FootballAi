"""Bounded local smoke client using the real ASGI API and filesystem queue."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from fastapi.testclient import TestClient

from footballai_v2.api import create_app
from footballai_v2.execution.coordinator import ExecutionSettings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    args = parser.parse_args()
    settings = ExecutionSettings.from_environment()
    app = create_app(settings.run_root, settings=settings)
    with TestClient(app) as client:
        profiles = client.get("/api/v1/pipeline-profiles").json()
        profile = next(item for item in profiles["profiles"] if item["profile_id"] == "v1_compat")
        assert profile["available"] and profile["readiness_status"] == "ready"
        assert "/Users/" not in json.dumps(profile)
        with args.fixture.open("rb") as video:
            created_response = client.post(
                "/api/v1/analyses",
                files={"video": (args.fixture.name, video, "video/mp4")},
                data={
                    "match_name": "Generated V1-compatible smoke",
                    "data_origin": "synthetic",
                    "pipeline_profile": "v1_compat",
                },
            )
        created_response.raise_for_status()
        run_id = created_response.json()["run_id"]
        deadline = time.monotonic() + 300
        while True:
            progress = client.get(f"/api/v1/runs/{run_id}/progress").json()
            if progress["status"] in {"succeeded", "partial", "failed", "cancelled"}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("V1-compatible smoke run did not become terminal within five minutes.")
            time.sleep(.25)
        detail = client.get(f"/api/v1/runs/{run_id}").json()
        artifacts = client.get(f"/api/v1/runs/{run_id}/artifacts").json()
    assert detail["status"] in {"succeeded", "partial"}, detail.get("failure")
    assert detail["provenance"]["parameters"]["pipeline_profile"] == "v1_compat"
    assert detail["provenance"]["models"][0]["name"] == "yolov8m.pt"
    terminal = {"succeeded", "partial", "skipped"}
    assert next(item for item in progress["stages"] if item["stage_name"] == "detection")["status"] in terminal
    assert next(item for item in progress["stages"] if item["stage_name"] == "tracking")["status"] in terminal
    assert all(
        not item["relative_path"].startswith("/") and "data/processed" not in item["relative_path"]
        for item in artifacts["artifacts"]
    )
    summary = {
        "fixture_duration_seconds": 2,
        "uploaded_size_bytes": args.fixture.stat().st_size,
        "profile": "v1_compat",
        "run_id": detail["run_id"],
        "result": detail["status"],
        "stage_statuses": {item["stage_name"]: item["status"] for item in progress["stages"]},
        "artifact_count": len(artifacts["artifacts"]),
        "v1_writes": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
