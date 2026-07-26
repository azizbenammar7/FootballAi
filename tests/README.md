# V1 characterization tests

These tests preserve the observable behavior and artifact contracts of the submitted technical-test implementation. They are a safety net for later V2 work, not a scientific validation of the football analytics or fatigue heuristics.

## Scope

- `unit/` characterizes kinematics, sprint counting, count-based heatmaps, fatigue components, score boundaries, clamping, and insufficient-data gating.
- `contract/` checks the small fixtures and the committed JSON artifacts without rewriting them.
- The committed `player_stats.parquet` receives an additional schema and non-negative-value check only when `pyarrow` is already installed. The default bounded test environment intentionally omits this heavier optional dependency, so that single check skips with an explicit reason.

Numeric-prefix modules are loaded with `importlib.util.spec_from_file_location`. Importing them does not call `main`, run inference, or write artifacts.

## Run locally

Use Python 3.13 in a repository-local virtual environment:

```bash
python3.13 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements-test.txt
PYTHONPYCACHEPREFIX=/tmp/footballai-pycache python -m compileall -q pipeline scripts dashboard tests
PYTHONPYCACHEPREFIX=/tmp/footballai-pycache python -m pytest -q
```

The suite requires no video, GPU, YOLO weights, Azure account, network access at test time, dashboard server, or full pipeline run. Dependency installation may require package-index access unless the pinned wheels are already cached.

To opt into the committed Parquet check, install a Python 3.13-compatible `pyarrow` in the local environment and rerun pytest. A skip in the default environment is deliberate and must not be reported as a successful Parquet validation.

## Intentionally excluded

- YOLO/ByteTrack inference and full-match processing
- regeneration or golden-file rewriting of committed real-match artifacts
- dashboard/UI integration
- identity stitching, calibration, fatigue-formula correction, and coverage redesign
- performance, medical validity, deployment, Azure, and network integration tests
