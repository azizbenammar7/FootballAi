# FootballAi V2

V2 is the professional platform generation. It is developed alongside, and
does not import or write into, the preserved V1 implementation in
`../pipeline/`, `../dashboard/`, `../scripts/`, or `../data/processed/`.

## Milestone 2 boundary

This milestone introduces two provider-neutral foundations:

1. `footballai.analysis-run/v1`, a strict provenance and lifecycle contract;
2. a local storage adapter that gives each run a unique output namespace.

FastAPI, workers, PostgreSQL, and Azure Blob Storage are deliberately not
introduced here. Future adapters should consume this contract rather than
inventing transport- or provider-specific run records.

## Analysis-run contract

The Python implementation lives in
`src/footballai_v2/contracts/v1/analysis_run.py`. The reviewable wire schema is
`contracts/analysis-run/v1.schema.json`, with a terminal example under
`contracts/analysis-run/examples/`.

Every run records:

- an explicit contract version and canonical UUID run ID;
- lifecycle state and UTC timestamps;
- `real` or `synthetic` data origin;
- input URI, media type, and SHA-256 identity;
- repository revision and dirty-worktree flag;
- pipeline, parameters, and model versions;
- content-addressed output references and their schema versions;
- structured, sanitized failure details when a run fails.

Unknown fields and invalid lifecycle combinations are rejected. Contract
changes that remove, rename, reinterpret, or newly require a field must be
published as `footballai.analysis-run/v2`; compatible consumers may continue
to read v1 indefinitely.

## Output isolation

`LocalAnalysisRunStore` uses this layout:

```text
<configured-root>/
└── <analysis-run-id>/
    ├── manifest.json
    └── artifacts/
        └── ...
```

The root is always caller-configured; no V2 component defaults to the V1
`data/processed/` directory. Run IDs reserve directories exclusively,
artifact writes use exclusive creation, relative paths must stay below
`artifacts/`, and manifest updates are atomic. Run provenance cannot change
after creation, and a terminal manifest cannot be replaced.

The local adapter is a development/test implementation of a storage boundary.
An Azure adapter can later map the same relative keys to a run-prefixed blob
container without changing the public contract.

## Bounded verification

From the repository root:

```bash
PYTHONPATH=v2/src python -m pytest -q v2/tests
PYTHONPYCACHEPREFIX=/tmp/footballai-pycache python -m compileall -q v2/src v2/tests
```

These tests use temporary directories only. They do not process video, use a
GPU, access a network, or write V1 artifacts.

