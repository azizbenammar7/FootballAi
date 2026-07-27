#!/usr/bin/env bash
set -euo pipefail

v2_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$v2_repo_root"

v2_python="${FOOTBALLAI_V2_PYTHON:-.venv-test/bin/python}"
v2_run_root="${FOOTBALLAI_V2_RUN_ROOT:-data/runs}"
v2_queue_root="${FOOTBALLAI_V2_QUEUE_ROOT:-data/job-queue}"
v2_model_path="${FOOTBALLAI_V1_COMPAT_MODEL_PATH:-$v2_repo_root/.models/yolov8m.pt}"
v2_device="${FOOTBALLAI_V1_COMPAT_DEVICE:-auto}"
if [[ ! -x "$v2_python" ]]; then
  echo "FootballAi V2 needs a local Python 3.13 environment."
  echo "Create it with: python3.13 -m venv .venv-test"
  echo "Then install: .venv-test/bin/python -m pip install -r requirements-test.txt"
  exit 2
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "FootballAi V2 needs Node.js and npm for the local dashboard."
  exit 2
fi

if [[ ! -d v2/dashboard/node_modules ]]; then
  (cd v2/dashboard && npm ci --no-audit --no-fund)
fi

find_available_port() {
  "$v2_python" -c 'import socket,sys
start=int(sys.argv[1])
for port in range(start,start+50):
    sock=socket.socket()
    try: sock.bind(("127.0.0.1",port))
    except OSError: sock.close(); continue
    sock.close(); print(port); break
else: raise SystemExit(2)' "$1"
}

v2_dashboard_port="$(find_available_port "${FOOTBALLAI_V2_DASHBOARD_PORT:-5173}")"
v2_api_port="$(find_available_port "${FOOTBALLAI_V2_API_PORT:-8000}")"

mkdir -p "$v2_run_root" "$v2_queue_root"
export FOOTBALLAI_V2_RUN_ROOT="$v2_run_root"
export FOOTBALLAI_V2_QUEUE_ROOT="$v2_queue_root"
export FOOTBALLAI_V1_COMPAT_MODEL_PATH="$v2_model_path"
export FOOTBALLAI_V1_COMPAT_DEVICE="$v2_device"
export FOOTBALLAI_V1_COMPAT_TARGET_FPS="${FOOTBALLAI_V1_COMPAT_TARGET_FPS:-5}"
export FOOTBALLAI_V1_COMPAT_IMAGE_SIZE="${FOOTBALLAI_V1_COMPAT_IMAGE_SIZE:-1280}"
export FOOTBALLAI_V1_COMPAT_CONFIDENCE="${FOOTBALLAI_V1_COMPAT_CONFIDENCE:-0.20}"
v2_manifest="$(find "$v2_run_root" -name manifest.json -print -quit 2>/dev/null || true)"
if [[ -z "$v2_manifest" ]]; then
  PYTHONPATH=v2/src "$v2_python" -m footballai_v2.cli.import_legacy_v1 \
    --source data/processed \
    --output-root "$v2_run_root"
fi

FOOTBALLAI_V2_RUN_ROOT="$v2_run_root" FOOTBALLAI_V2_QUEUE_ROOT="$v2_queue_root" \
PYTHONPATH=v2/src "$v2_python" -m footballai_v2.cli.seed_demo_workflow

FOOTBALLAI_V2_RUN_ROOT="$v2_run_root" \
FOOTBALLAI_V2_QUEUE_ROOT="$v2_queue_root" \
FOOTBALLAI_V2_CORS_ORIGINS="http://localhost:${v2_dashboard_port},http://127.0.0.1:${v2_dashboard_port}" \
PYTHONPATH=v2/src \
"$v2_python" -m uvicorn footballai_v2.api.main:app --host 127.0.0.1 --port "$v2_api_port" &
v2_api_pid=$!

FOOTBALLAI_V2_RUN_ROOT="$v2_run_root" FOOTBALLAI_V2_QUEUE_ROOT="$v2_queue_root" \
PYTHONPATH=v2/src "$v2_python" -m footballai_v2.worker &
v2_worker_pid=$!

(cd v2/dashboard && VITE_API_BASE="http://127.0.0.1:${v2_api_port}" npm run dev -- --host 127.0.0.1 --port "$v2_dashboard_port" --strictPort) &
v2_ui_pid=$!

cleanup_v2_demo() {
  kill "$v2_api_pid" "$v2_worker_pid" "$v2_ui_pid" 2>/dev/null || true
  wait "$v2_api_pid" "$v2_worker_pid" "$v2_ui_pid" 2>/dev/null || true
}
trap cleanup_v2_demo EXIT INT TERM

echo
echo "FootballAi V2 local demo"
echo "Dashboard: http://localhost:${v2_dashboard_port}"
echo "API:       http://localhost:${v2_api_port}"
echo "Worker:    running"
echo "Python:    $v2_python"
echo "V1 device: $v2_device"
echo "V1 model:  $(basename "$v2_model_path")"
echo "Run root:  ${v2_run_root}"
echo "Queue root: ${v2_queue_root}"
echo "Press Ctrl+C to stop all services."
echo

while kill -0 "$v2_api_pid" 2>/dev/null && kill -0 "$v2_worker_pid" 2>/dev/null && kill -0 "$v2_ui_pid" 2>/dev/null; do
  sleep 1
done
