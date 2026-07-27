"""Prepare and report the optional local V1-compatible runtime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from footballai_v2.execution.adapters.v1_compat_runtime import (
    MODEL_NAME,
    REPOSITORY_ROOT,
    check_v1_compat_readiness,
    configured_model_path,
    sha256_file,
    validate_model_file,
)


def _prepare_model() -> int:
    from ultralytics import YOLO, __version__ as ultralytics_version

    target = configured_model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    source = "existing local model"
    if target.exists():
        valid, reason = validate_model_file(target)
        if not valid:
            print(f"ERROR: refusing to replace invalid configured model: {reason}")
            return 2
    else:
        legacy = REPOSITORY_ROOT / MODEL_NAME
        legacy_valid, _ = validate_model_file(legacy)
        if legacy_valid:
            print("Reusing the existing ignored local yolov8m.pt weight.")
            shutil.copyfile(legacy, target)
            source = "copied from existing ignored local model"
        else:
            print("Local optional model download: requesting only official yolov8m.pt through Ultralytics.")
            previous = Path.cwd()
            try:
                os.chdir(target.parent)
                YOLO(MODEL_NAME)
            finally:
                os.chdir(previous)
            source = "downloaded by the explicit local setup command"
    valid, reason = validate_model_file(target)
    if not valid:
        print(f"ERROR: model preparation failed: {reason}")
        return 2
    try:
        YOLO(str(target.resolve()))
    except Exception:
        print("ERROR: Ultralytics rejected the prepared yolov8m.pt file.")
        return 2
    checksum = sha256_file(target)
    metadata = {
        "model": MODEL_NAME,
        "sha256": checksum,
        "source": source,
        "prepared_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ultralytics_version": ultralytics_version,
    }
    (target.parent / f"{MODEL_NAME}.provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Model: {MODEL_NAME}")
    print(f"SHA-256: {checksum}")
    return 0


def _check(json_output: bool) -> int:
    readiness = check_v1_compat_readiness()
    if json_output:
        print(json.dumps(readiness.public_dict(), indent=2, sort_keys=True))
    else:
        print(readiness.message)
        print(f"Status: {readiness.status}")
        for name, value in readiness.runtime.items():
            print(f"{name.replace('_', ' ').title()}: {json.dumps(value, sort_keys=True) if isinstance(value, dict) else value}")
        if readiness.missing_requirements:
            print("Missing: " + ", ".join(readiness.missing_requirements))
        if readiness.runtime_errors:
            print("Runtime errors: " + "; ".join(readiness.runtime_errors))
    return 0 if readiness.ready else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--json", action="store_true")
    subparsers.add_parser("prepare-model")
    args = parser.parse_args()
    if args.command == "prepare-model":
        return _prepare_model()
    return _check(args.json)


if __name__ == "__main__":
    raise SystemExit(main())
