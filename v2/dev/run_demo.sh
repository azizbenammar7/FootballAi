#!/usr/bin/env bash
set -euo pipefail

v2_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$v2_repo_root"

v2_python="${FOOTBALLAI_V2_PYTHON:-.venv-test/bin/python}"
v2_dashboard_port="${FOOTBALLAI_V2_DASHBOARD_PORT:-5173}"
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

mkdir -p data/runs
v2_manifest="$(find data/runs -name manifest.json -print -quit 2>/dev/null || true)"
if [[ -z "$v2_manifest" ]]; then
  PYTHONPATH=v2/src "$v2_python" -m footballai_v2.cli.import_legacy_v1 \
    --source data/processed \
    --output-root data/runs
fi

FOOTBALLAI_V2_RUN_ROOT=data/runs \
FOOTBALLAI_V2_CORS_ORIGINS="http://localhost:${v2_dashboard_port},http://127.0.0.1:${v2_dashboard_port}" \
PYTHONPATH=v2/src \
"$v2_python" -m uvicorn footballai_v2.api.main:app --host 127.0.0.1 --port 8000 &
v2_api_pid=$!

(cd v2/dashboard && VITE_API_BASE=http://127.0.0.1:8000 npm run dev -- --host 127.0.0.1 --port "$v2_dashboard_port" --strictPort) &
v2_ui_pid=$!

cleanup_v2_demo() {
  kill "$v2_api_pid" "$v2_ui_pid" 2>/dev/null || true
  wait "$v2_api_pid" "$v2_ui_pid" 2>/dev/null || true
}
trap cleanup_v2_demo EXIT INT TERM

echo
echo "FootballAi V2 local demo"
echo "Dashboard: http://localhost:${v2_dashboard_port}"
echo "API:       http://localhost:8000"
echo "Press Ctrl+C to stop both services."
echo

while kill -0 "$v2_api_pid" 2>/dev/null && kill -0 "$v2_ui_pid" 2>/dev/null; do
  sleep 1
done
