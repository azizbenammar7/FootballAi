#!/usr/bin/env bash
set -euo pipefail

v1_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$v1_repo_root"
v1_python="${FOOTBALLAI_V2_PYTHON:-.venv-test/bin/python}"

if [[ ! -x "$v1_python" ]]; then
  echo "ERROR: canonical local environment is missing: .venv-test"
  exit 2
fi

"$v1_python" -c 'import sys
if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit("ERROR: v1_compat requires Python 3.11 through 3.13")
print(f"Python: {sys.version.split()[0]} ({sys.executable})")'

echo "Installing the pinned optional V1-compatible runtime into the canonical local environment."
"$v1_python" -m pip install --disable-pip-version-check --no-deps -r v2/requirements-v1-compat.txt

if "$v1_python" -m pip show opencv-python >/dev/null 2>&1; then
  echo "ERROR: GUI opencv-python is installed. Remove it before using the headless V1-compatible runtime."
  exit 2
fi

"$v1_python" -c 'import cv2, lap, pyarrow, torch, tqdm, ultralytics
print(f"Ultralytics: {ultralytics.__version__}")
print(f"OpenCV headless: {cv2.__version__}")
print(f"PyArrow: {pyarrow.__version__}")
print(f"tqdm: {tqdm.__version__}")
print(f"PyTorch: {torch.__version__}")
print("LAP assignment solver: available")
print(f"Apple MPS available: {torch.backends.mps.is_available()}")'

command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
ffmpeg -version
ffprobe -version

PYTHONPATH=v2/src "$v1_python" -m footballai_v2.cli.v1_compat_runtime prepare-model
PYTHONPATH=v2/src "$v1_python" -m footballai_v2.cli.v1_compat_runtime check
echo "SUCCESS: v1_compat is installed and ready in the canonical local environment."
