# V2 analysis execution

## Architecture

The React dashboard submits multipart metadata and one video to FastAPI.
FastAPI streams into a temporary file in 1 MiB chunks, calculates SHA-256,
enforces configured bounds, and asks `ffprobe` to validate the real container
and duration. Only then does it create a `footballai.analysis-run/v1`
namespace, atomically move the input to `input/source.<ext>`, and enqueue a
safe reference. The HTTP request never executes analysis.

The separate worker atomically claims one filesystem job, changes the attempt
from `queued` to `running`, persists weighted stage progress in `manifest.json`,
publishes content-addressed artifacts, and makes the attempt terminal. Its
operational log fields include job/run/logical IDs, attempt, worker, stage,
status, duration, counts, and safe error codes.

## Local queue

`data/job-queue/{queued,claimed,completed,failed,cancelled}` is a provider-neutral
development adapter. Enqueue uses exclusive per-run reservation plus atomic
hard-link publication; claim uses atomic rename. Records are bounded JSON with
IDs, attempt, profile, timestamps, and worker ID only. They contain no media,
secrets, or arbitrary paths. Stale claims are requeued only for non-terminal
attempts. This adapter is process-safe on one local filesystem; it is not a
distributed cloud queue.

## States and operations

Attempts follow `queued → running → succeeded|partial|failed|cancelled`.
Terminal manifests are immutable. Cancellation is a persistent run marker; a
worker checks it between safe steps and terminates V1-compatible child
processes gracefully. `partial` requires useful valid artifacts and terminal
stage evidence.

Only `failed` and `partial` can retry. A retry gets a new run directory and run
ID, increments the attempt number, preserves logical/input identity, and links
the previous attempt. Clone creates a new logical analysis from independently
copied source bytes. Cancelled and succeeded attempts cannot retry.

## Profiles

- `demo_fast`: deterministic results seeded by input checksum and match name.
  It has no ML dependency and always carries the synthetic-workflow warning.
- `v1_compat`: runs the preserved YOLOv8/ByteTrack, metrics, and advisory
  algorithm family inside `<run>/tmp/v1-compat`. V2 controls the explicit
  model, device, FPS, image size, and confidence; preserved V1 statistics and
  fatigue scripts run unchanged when tracking produces usable rows. All
  outputs and caches stay under the V2 run. Empty detections produce honest
  empty artifacts. The worker is offline and cannot auto-download weights or
  missing packages. This profile does not claim V2 identity resolution,
  calibration, or scientific validation.

Readiness distinguishes missing Python packages, missing system tools, missing
or invalid model weights, unsupported Python/platform, runtime import/device
errors, and `ready`. Public responses contain versions, selected device, model
name, and checksum, but never an absolute local model path. The dashboard only
enables the option for `ready` and otherwise shows `make v2-v1-compat-setup`.

## Resource configuration

`FOOTBALLAI_MAX_UPLOAD_BYTES`, `FOOTBALLAI_MAX_VIDEO_DURATION_SECONDS`,
`FOOTBALLAI_ALLOWED_VIDEO_EXTENSIONS`, `FFPROBE_TIMEOUT_SECONDS`,
`FOOTBALLAI_V2_RUN_ROOT`, `FOOTBALLAI_V2_QUEUE_ROOT`,
`FOOTBALLAI_WORKER_ID`, `FOOTBALLAI_WORKER_POLL_SECONDS`, and
`FOOTBALLAI_JOB_CLAIM_TIMEOUT_SECONDS` control local execution. Defaults are
bounded for development.

V1 compatibility additionally uses
`FOOTBALLAI_V1_COMPAT_MODEL_PATH`,
`FOOTBALLAI_V1_COMPAT_DEVICE` (`auto`, `mps`, `cpu`, or `cuda`),
`FOOTBALLAI_V1_COMPAT_TARGET_FPS`,
`FOOTBALLAI_V1_COMPAT_IMAGE_SIZE`, and
`FOOTBALLAI_V1_COMPAT_CONFIDENCE`. Effective values and model checksum are
recorded in immutable run provenance. On Apple Silicon, `auto` selects MPS
when available and otherwise selects CPU; an explicitly unavailable device
fails before execution. A runtime MPS failure is surfaced and never silently
replayed on CPU.

Artifacts use stable schemas `footballai.team-summary/v1`,
`footballai.track-summary/v1`, `footballai.track-detail/v1`,
`footballai.workload-advisory/v1`, and
`footballai.analysis-diagnostics/v1`. Workload and Fatigue Advisory is
heuristic and advisory only, never diagnosis or clinical advice.
