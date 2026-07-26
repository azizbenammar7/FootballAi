#!/usr/bin/env bash
set -euo pipefail

e2e_python="${FOOTBALLAI_V2_PYTHON:-../../.venv-test/bin/python}"

PYTHONPATH=../src "$e2e_python" -m footballai_v2.cli.import_legacy_v1 \
  --source ../../data/processed --output-root "${FOOTBALLAI_V2_RUN_ROOT}"

PYTHONPATH=../src "$e2e_python" -m footballai_v2.worker &
e2e_worker_pid=$!

cleanup_e2e() {
  kill "$e2e_worker_pid" 2>/dev/null || true
  wait "$e2e_worker_pid" 2>/dev/null || true
}
trap cleanup_e2e EXIT INT TERM

PYTHONPATH=../src "$e2e_python" -m uvicorn footballai_v2.api.main:app --host 127.0.0.1 --port "${FOOTBALLAI_V2_API_PORT:-8000}"
