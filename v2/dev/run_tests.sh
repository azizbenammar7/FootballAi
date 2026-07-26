#!/usr/bin/env bash
set -euo pipefail

v2_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$v2_repo_root"
v2_python="${FOOTBALLAI_V2_PYTHON:-.venv-test/bin/python}"

PYTHONPYCACHEPREFIX=/tmp/footballai-v2-pycache PYTHONPATH=v2/src \
  "$v2_python" -m compileall -q v2/src v2/tests tests pipeline scripts dashboard
PYTHONPYCACHEPREFIX=/tmp/footballai-v2-pycache PYTHONPATH=v2/src \
  "$v2_python" -m pytest -q -ra
(cd v2/dashboard && npm ci --no-audit --no-fund && npm run check && npm run test:e2e)
