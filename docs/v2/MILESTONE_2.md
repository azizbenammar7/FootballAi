# V2 Milestone 2 — versioned analysis runs and isolated outputs

## Decision record

The V2 platform starts with a storage-neutral contract because API requests,
worker messages, database rows, and blob manifests will all need to agree on
what an analysis run means. The contract is versioned independently of the
platform: V2 currently publishes analysis-run contract **v1**.

## Guarantees delivered

- A run has one canonical UUID and one explicit `real` or `synthetic` origin.
- Input content, source revision, parameters, pipeline version, and model
  versions form reviewable provenance.
- Only valid queued, running, succeeded, failed, and cancelled records can be
  serialized.
- Successful outputs carry logical names, run-relative paths, media types,
  byte sizes, SHA-256 hashes, and optional output-schema versions.
- Local outputs with the same filename cannot collide across run IDs.
- A run cannot overwrite one of its own existing artifacts.
- Traversal paths and symlink escapes are rejected by the local adapter.
- Manifest replacement is atomic; provenance and terminal manifests are
  immutable.

## Deliberate scope limits

This milestone does not execute or refactor the V1 pipeline. It also does not
claim distributed transaction semantics. The local adapter has a single-host
scope; PostgreSQL will later own concurrency/state-transition coordination,
while Azure Blob Storage will own durable artifact writes.

## V1 preservation

The baseline remains tag `technical-test-v1.0`, commit
`84b5457e0058b64fb2cbf31ea795c168debdcae5`. No file under `pipeline/`,
`dashboard/`, `scripts/`, or `data/processed/` is changed by this milestone.

