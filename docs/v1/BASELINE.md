# V1 preserved baseline

## Preservation record

- Preserved tag: `technical-test-v1.0`
- Original audited commit: `84b5457e0058b64fb2cbf31ea795c168debdcae5`
- Modernization branch: `test/v2-characterization-foundation`
- Baseline strategy: keep the submitted `pipeline/`, `dashboard/`, and `scripts/` layout and runtime behavior intact while adding an isolated test foundation.

The tag identifies the immutable V1 baseline. Publishing the tag and modernization branch is repository management only; no deployment or cloud resource operation is part of this baseline.

## Repository architecture

The repository is a compact file-based prototype with four stages:

1. `pipeline/01_track.py` uses YOLO and ByteTrack to produce raw track observations and video metadata.
2. `pipeline/02_stats.py` uses pandas and NumPy to produce per-block player statistics and a player summary with timelines and heatmaps.
3. `pipeline/03_fatigue.py` uses pandas, NumPy, and SciPy to produce heuristic risk records.
4. `dashboard/app.py` reads committed processed artifacts for Streamlit presentation.

Current output files are `data/processed/raw_tracks.parquet`, `meta.json`, `player_stats.parquet`, `player_summary.json`, and `risk_scores.json`. The tracking checkpoint is a transient CSV, not a resumable run record.

## Dependencies and Python constraint

The original `requirements.txt` is a broad runtime list with minimum versions and includes the ML, dashboard, and Parquet stack. V1 documentation supports Python 3.11–3.13 and notes that the stack is not compatible with the repository machine's Python 3.14 default.

This milestone adds `requirements-test.txt`, pinning only pytest, NumPy, pandas, and SciPy for the bounded suite. Python 3.13 is canonical. `pyarrow` remains optional for one committed-Parquet invariant because it is not needed for the unit or JSON contract tests and is intentionally excluded from the minimal test environment.

## Tests introduced

- Kinematics: stationary and linear motion, invalid and boundary gaps, speed capping, rolling-median jitter handling, minimal inputs, and non-negative distance.
- Sprints: threshold and duration boundaries, separated/interrupted efforts, irregular timestamps, and empty/minimal inputs.
- Heatmaps: dimensions, normalization, repeated/boundary/out-of-frame observations, empty and finite output, and count-based weighting.
- Fatigue and risk: component boundaries and clamping, slope/drop cases, the total-distance behavior of the current “HSR load,” total-score/flag boundaries, and insufficient-data output.
- Contracts: minimal JSON fixtures, committed metadata/player/risk JSON schemas, cross-file identifier and duration invariants, and optional committed Parquet checks.

## Deliberately excluded tests

No YOLO inference, video processing, full pipeline execution, artifact regeneration, dashboard server, network operation, Azure operation, identity correction, calibration correction, medical validation, or performance benchmark is included. The suite does not claim that a full match was successfully processed; the repository contains committed outputs but no run manifest sufficient to verify such an execution.

## Bounded commands

```bash
python3.13 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements-test.txt
PYTHONPYCACHEPREFIX=/tmp/footballai-pycache python -m compileall -q pipeline scripts dashboard tests
PYTHONPYCACHEPREFIX=/tmp/footballai-pycache python -m pytest -q
git diff --check
```

No full-match inference or artifact-producing pipeline script was executed while establishing this baseline.
