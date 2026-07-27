#!/usr/bin/env bash
set -euo pipefail

tracked_models="$(git ls-files '*.pt' '*.pth' '*.onnx')"
if [[ -n "$tracked_models" ]]; then
  echo "ERROR: model weights must not be tracked:"
  echo "$tracked_models"
  exit 1
fi

tracked_uploads="$(git ls-files '*.mp4' '*.mov' '*.mkv' '*.webm' '*.avi')"
if [[ -n "$tracked_uploads" ]]; then
  echo "ERROR: uploaded or generated videos must not be tracked:"
  echo "$tracked_uploads"
  exit 1
fi

echo "No model weights or uploaded videos are tracked."
