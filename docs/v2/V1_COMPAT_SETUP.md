# V1-compatible local runtime

`v1_compat` is a preserved-algorithm compatibility profile. It is not the
future detector-neutral V2 production engine.

It executes the historical YOLOv8m + ByteTrack algorithm family for a newly
uploaded V2 input, then uses the preserved V1 statistics and fatigue stages
when enough tracking data exists. V1 source and committed outputs remain
unchanged. Inputs, intermediate files, caches, logs, and artifacts stay below
the current V2 run directory.

## Setup

Use Python 3.11–3.13 in the canonical local environment:

```bash
python3.13 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
make v2-v1-compat-setup
```

The setup installs the exact additional lock in
`v2/requirements-v1-compat.txt` with dependency auto-resolution disabled. This
keeps the repository's NumPy, pandas, and SciPy pins and supplies headless
OpenCV instead of Ultralytics' GUI OpenCV distribution. The runtime includes
Ultralytics, PyTorch/torchvision, `opencv-python-headless`, PyArrow for Parquet,
`tqdm`, the LAP assignment solver required lazily by ByteTrack, and their exact
additional runtime dependencies. Nothing is installed globally and
`demo_fast` remains independent of ML packages.

The command verifies imports, FFmpeg, ffprobe, Python, platform, model, and
device. Check again at any time:

```bash
make v2-v1-compat-readiness
```

Readiness is one of `missing_python_packages`, `missing_system_tools`,
`missing_model_weights`, `unsupported_python_version`,
`unsupported_platform`, `runtime_import_error`, or `ready`. Import and device
failures are sanitized. The public API never exposes an absolute model path.

## Model weights

The historical detector is exactly `yolov8m.pt`. The default ignored path is:

```text
.models/yolov8m.pt
```

Override it with `FOOTBALLAI_V1_COMPAT_MODEL_PATH`. A file must be non-empty,
have the expected PyTorch archive signature and name, load successfully in
Ultralytics during setup, and receive a printed SHA-256. If the default file is
absent, setup first reuses an existing ignored root weight; otherwise it
clearly requests only `yolov8m.pt` through the official Ultralytics loader.
Normal tests, API readiness, and worker execution never download a model. The
worker rejects missing, invalid, or checksum-changed weights.

The local historical weight validated for this milestone has SHA-256:

```text
5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5
```

Always trust the checksum printed by setup and recorded in run provenance for
the actual local file; do not substitute a same-named model silently.

## Device and bounded configuration

Defaults preserve V1-compatible settings:

```text
FOOTBALLAI_V1_COMPAT_TARGET_FPS=5
FOOTBALLAI_V1_COMPAT_IMAGE_SIZE=1280
FOOTBALLAI_V1_COMPAT_CONFIDENCE=0.20
FOOTBALLAI_V1_COMPAT_DEVICE=auto
```

Device values are `auto`, `mps`, `cpu`, and `cuda`. On Apple Silicon, `auto`
selects MPS when PyTorch reports it available, otherwise CPU. Explicit MPS or
CUDA fails readiness when unavailable. If MPS later fails during execution,
the run fails with an actionable error and is not silently restarted on CPU.
All effective settings, selected device, model name, and model checksum are
recorded in run provenance.

## Start and short smoke test

Run a genuine bounded environment test and then start the dashboard:

```bash
make v2-v1-compat-smoke
make v2-demo-v1-compat
```

The smoke command generates a two-second solid-color MP4 with FFmpeg, submits
it through the real FastAPI upload handler, queues it on the real filesystem
queue, and lets a separate worker execute real model inference. It does not
use `demo_fast` or copyrighted footage. Zero detections are an acceptable
succeeded-empty environment result. The smoke verifies terminal detection and
tracking, artifact isolation, model provenance, and no changes to V1 paths.

For a private longer test, first confirm readiness, free disk space, sustained
power, thermal conditions, input licensing, and the selected device. Start the
combined demo, upload the video, and select `v1_compat`. The 94-minute fixture
is deliberately not run by setup or CI and can take roughly the duration
documented by V1 for YOLOv8m at 1280 pixels.

## Scientific and operational limits

The profile preserves the V1 algorithm family; it does not scientifically
correct it. ByteTrack identities are unverified and can switch. Pixel movement
is not homography calibrated. Broadcast pan/zoom and sparse tracks make
absolute distance approximate. Fatigue/workload outputs are heuristic,
advisory only, and are not diagnosis or clinical advice. Empty or short clips
may have no tracks surviving preserved filters.

Troubleshooting:

- Run `make v2-v1-compat-readiness` and follow its exact status.
- Missing packages: rerun setup in `.venv-test`; do not use global Python.
- Missing model: rerun setup or set the explicit model environment variable.
- Missing FFmpeg tools: install both `ffmpeg` and `ffprobe` locally.
- MPS unavailable: confirm the terminal and demo use the same Python. Choose
  CPU explicitly only after accepting the much slower full-match runtime.
- Worker failure: inspect the private per-run `logs/v1-compat.log`; logs are
  bounded and are not exposed through the public API.

## Licence notice

This profile uses Ultralytics YOLOv8 code and model weights. Local use must
comply with the applicable Ultralytics code and model licence. Dependency and
model licensing must be reviewed before any public or commercial deployment;
this documentation does not claim legal approval. The licence review does not
block this local personal-project setup. The future V2 engine remains
detector-neutral.
