#!/usr/bin/env bash
set -euo pipefail

v1_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$v1_repo_root"
v1_python="${FOOTBALLAI_V2_PYTHON:-.venv-test/bin/python}"
export FOOTBALLAI_V1_COMPAT_MODEL_PATH="${FOOTBALLAI_V1_COMPAT_MODEL_PATH:-$v1_repo_root/.models/yolov8m.pt}"

if ! PYTHONPATH=v2/src "$v1_python" -m footballai_v2.cli.v1_compat_runtime check; then
  echo "ERROR: v1_compat is not ready. Run: make v2-v1-compat-setup"
  exit 2
fi

exec ./v2/dev/run_demo.sh
