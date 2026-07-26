# V2 Milestone 2 — Versioned Analysis Runs and Isolated Outputs

## Decision

V2 publishes its first analysis-run contract as
`footballai.analysis-run/v1`. The `v1` suffix identifies the contract version,
not the platform generation.

## Lifecycle and origin

Allowed transitions within one attempt are:

```text
queued -> running | failed | cancelled
running -> succeeded | partial | failed | cancelled
```

`succeeded`, `partial`, `failed`, and `cancelled` are terminal and immutable.
`partial` means processing ended before all mandatory work completed, but valid
and reviewable artifacts were produced. Continuing after a failed or partial
attempt requires a new attempt rather than a same-manifest state reversal.

Origins are `real`, `synthetic`, `evaluation`, and `legacy_v1`. The
`legacy_v1` value identifies imported historical outputs only; it does not
claim V2 accuracy, calibration, identity safety, or quality compliance.

## Logical analysis versus run attempt

```text
Logical analysis
├── Attempt 1 — failed
├── Attempt 2 — partial
└── Attempt 3 — succeeded
```

Every attempt has its own UUID-v4 `run_id`, immutable terminal manifest,
isolated directory and artifacts, visible provenance, and link to the previous
attempt. All attempts share one UUID-v4 `logical_analysis_id`. The initial
attempt has `attempt_number: 1` and `previous_attempt_run_id: null`. Each retry
increments the attempt number and links to the immediately preceding run ID.

Input checksum/identity, origin, and logical analysis ID remain constant across
the chain. Code revision and dirty state, pipeline version, configuration, and
model versions may change on a retry so a corrected implementation can be
tested without erasing the historical attempt. Those differences remain
explicit in the new manifest.

Retries are supported only from failed or partial attempts. Queued, running,
succeeded, and cancelled attempts cannot be retried by the store operation.

## Stage execution

The contract provides stable initial stage names:

```text
ingestion
video_validation
detection
tracking
identity_resolution
pitch_calibration
metrics
workload_advisory
artifact_publication
```

Not every run must execute every stage. Each record contains `stage_id`,
`stage_name`, `required`, `status`, `progress_percent`, `attempt_number`,
`started_at`, `finished_at`, `produced_artifact_ids`, `error`,
`performance_metrics`, and `message`.

Stage statuses are `queued`, `running`, `succeeded`, `partial`, `failed`,
`cancelled`, and `skipped`. Validation enforces bounded finite progress,
required timestamps, chronological finish times, 100% succeeded progress,
structured errors on failed stages, unique stage IDs and stable names, matching
attempt numbers, and artifact references that resolve within the manifest.

Structured errors include `error_code`, `safe_message`, `retryable`,
`occurred_at`, and optional `technical_details`. Raw tracebacks and secret-like
diagnostics are rejected by default. Performance metrics are an extensible,
finite JSON map; examples include duration, frame count, processing rate,
memory, and input/output counts.

Run/stage checks prevent active stages in terminal attempts. Success requires
registered artifacts, terminal stage records, and successful required stages.
Partial completion requires a safe reason, a useful artifact, and evidence of
incomplete mandatory work. Failure requires a structured run-level error.
Cancellation keeps a supplied safe reason. Optional future stages are not
required to succeed merely because they exist.

## Artifacts and Workload and Fatigue Advisory

Registered artifacts have unique IDs and names, categories, isolated relative
paths, media types, byte sizes, SHA-256 checksums, and optional schema versions.
`workload_advisory` is the stable serialized category and stage term. The
public label is **Workload and Fatigue Advisory**.

The advisory is not a medical diagnosis, validated injury prediction, or
clinical advice. Historical V1 names and schemas are preserved unchanged.

## Storage boundary

The root is always caller-configurable:

```text
Local development default later: data/runs
Configured worker:               <configured-root>/<run-id>
Future Azure Blob:               runs/<run-id>/...
```

Tests use temporary directories. Azure remains a future conceptual adapter;
there is no Azure SDK, account, resource, or deployment in Milestone 2.

The local adapter provides exclusive run namespaces and artifact creation,
atomic manifest replacement, traversal and symlink protection, content hash
verification, terminal immutability, and isolated retry directories.

## V1 preservation

The baseline remains tag `technical-test-v1.0`, commit
`84b5457e0058b64fb2cbf31ea795c168debdcae5`. Milestone 2 does not change V1
pipeline algorithms or committed artifacts and does not run inference.
