#!/usr/bin/env bash
set -euo pipefail

smoke_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$smoke_repo_root"
smoke_python="${FOOTBALLAI_V2_PYTHON:-.venv-test/bin/python}"
smoke_root="$(mktemp -d /tmp/footballai-v1-smoke.XXXXXX)"
smoke_run_root="$smoke_root/runs"
smoke_queue_root="$smoke_root/queue"
smoke_fixture="$smoke_root/generated-solid-color.mp4"
smoke_worker_log="$smoke_root/worker.log"
smoke_worker_pid=""

cleanup_v1_smoke() {
  [[ -n "$smoke_worker_pid" ]] && kill "$smoke_worker_pid" 2>/dev/null || true
  [[ -n "$smoke_worker_pid" ]] && wait "$smoke_worker_pid" 2>/dev/null || true
}
trap cleanup_v1_smoke EXIT INT TERM

export FOOTBALLAI_V2_RUN_ROOT="$smoke_run_root"
export FOOTBALLAI_V2_QUEUE_ROOT="$smoke_queue_root"
export FOOTBALLAI_V1_COMPAT_MODEL_PATH="${FOOTBALLAI_V1_COMPAT_MODEL_PATH:-$smoke_repo_root/.models/yolov8m.pt}"
export FOOTBALLAI_V1_COMPAT_DEVICE="${FOOTBALLAI_V1_COMPAT_DEVICE:-auto}"
export FOOTBALLAI_V1_COMPAT_TARGET_FPS="${FOOTBALLAI_V1_COMPAT_TARGET_FPS:-1}"
export FOOTBALLAI_V1_COMPAT_IMAGE_SIZE="${FOOTBALLAI_V1_COMPAT_IMAGE_SIZE:-320}"
export FOOTBALLAI_V1_COMPAT_CONFIDENCE="${FOOTBALLAI_V1_COMPAT_CONFIDENCE:-0.25}"
export FOOTBALLAI_V1_SUBPROCESS_TIMEOUT_SECONDS="${FOOTBALLAI_V1_SUBPROCESS_TIMEOUT_SECONDS:-240}"
export FOOTBALLAI_DEMO_STAGE_DELAY_SECONDS=0

PYTHONPATH=v2/src "$smoke_python" -m footballai_v2.cli.v1_compat_runtime check >/dev/null
git diff --quiet technical-test-v1.0 -- pipeline dashboard scripts data/processed

ffmpeg -v error -f lavfi -i "color=c=0x195f3b:s=320x192:r=12:d=2" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$smoke_fixture"

PYTHONPATH=v2/src "$smoke_python" -m footballai_v2.worker >"$smoke_worker_log" 2>&1 &
smoke_worker_pid=$!
PYTHONPATH=v2/src "$smoke_python" -m footballai_v2.cli.v1_compat_smoke --fixture "$smoke_fixture"

git diff --quiet technical-test-v1.0 -- pipeline dashboard scripts data/processed
