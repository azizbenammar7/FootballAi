# V2 Milestone 3 — Safe Legacy V1 Artifact Importer

## Purpose

The importer makes the preserved technical-test outputs reviewable through V2
without executing or modifying V1. It reads these supported files:

```text
data/processed/meta.json
data/processed/player_summary.json
data/processed/risk_scores.json
data/processed/player_stats.parquet
data/processed/raw_tracks.parquet
```

`meta.json` and `player_summary.json` are the minimum readable set. Missing
completion artifacts create a valid partial run. Malformed JSON, invalid
Parquet envelopes, symlink sources, missing minimum inputs, and duplicate
caller-provided run IDs fail safely.

## Import behavior

Each invocation creates a new UUID-v4 logical analysis and run directory under
the caller-configured root. Supported source bytes are copied, never moved,
and receive registered sizes and SHA-256 hashes. Source checksums are recorded
before copying and verified again afterward.

The V1 `risk_scores.json` source name remains historically unchanged. Its V2
copy is registered as `artifacts/workload_advisory.json`, category
`workload_advisory`, with the public label **Workload and Fatigue Advisory**.
This is a heuristic advisory, not medical diagnosis or clinical advice.

The run records `legacy_v1` origin and ingestion, video-validation, tracking,
metrics, workload-advisory, and artifact-publication stages. Stage messages
state that historical artifacts were copied and that pipeline work was not
re-executed.

## Mandatory warnings

Every imported manifest and its `quality_warnings.json` artifact disclose:

- track IDs are not verified player identities;
- halftime stitching is not identity-safe;
- pitch positions are not homography-calibrated;
- camera motion may affect movement estimates;
- active-time and coverage semantics are approximate;
- V1 HSR load uses total distance;
- V1 risk values are heuristic and advisory only;
- full-match execution provenance is incomplete.

## Local command

From the repository root:

```bash
PYTHONPATH=v2/src .venv-test/bin/python -m footballai_v2.cli.import_legacy_v1 \
  --source data/processed \
  --output-root data/runs
```

An optional `--run-id <uuid-v4>` supports deterministic local testing. The
command refuses to overwrite an existing namespace, returns non-zero on
expected errors, and suppresses raw tracebacks by default. `--verbose` adds a
bounded chained diagnostic without changing source files.

`data/runs/` is ignored local runtime state. The importer uses no network,
cloud, model, GPU, or V1 pipeline execution.
