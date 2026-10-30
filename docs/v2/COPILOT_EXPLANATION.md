# Evidence-grounded "Explain This Run" copilot (AI Phase 1)

This is the first AI-engineering increment on top of the FootballAI V2 platform:
one bounded, read-only LLM-powered feature that explains an existing analysis
run from **integrity-verified evidence**. It is intentionally *not* an
autonomous agent. See [ADR-003](ADR-003-grounded-explanation-copilot.md) for the
design decision.

## Where it sits in the CV workflow

FootballAI V2 ingests a video, runs it through a worker pipeline (AI1 YOLO
detection + ByteTrack tracking, or the deterministic `demo_fast` synthetic
profile), and publishes an immutable `footballai.analysis-run/v1` manifest with
content-addressed artifacts behind the `ObjectStorage` port. The copilot reads
that finished run — it never runs inference, never mutates a run, and adds no
pipeline stage.

```
AnalysisRepository ─┐
                    ├─ RunEvidenceService ─ ValidatedRunEvidence ─┐
ObjectStorage ──────┘   (strict artifact validation)              ├─ ExplanationService ─ ExplanationResponse
                                             LLMProvider ──────────┘   (grounding + citation validation)
```

## The evidence service

`RunEvidenceService.build(run_id)` returns a typed, bounded
`ValidatedRunEvidence` snapshot assembled only through the storage ports:

- run status, pipeline profile, pipeline version, data origin, timestamps;
- a `source_video` block limited to `sha256`, `media_type`, and (when available)
  probed `duration_seconds` — **never** the raw input URI or a SAS URL;
- model, code (repository/revision/dirty) provenance;
- registered artifact metadata with a per-artifact `integrity_state`;
- approved tracking-diagnostics fields;
- explicit `limitations` and `missing_evidence` lists.

### Strict artifact validation

Before the `analysis-diagnostics` artifact is used as evidence it must pass, in
order: JSON media-type registration → size ≤ `max_artifact_bytes` (1 MiB) →
SHA-256 integrity (`ObjectStorage.artifact_integrity`) → bounded read →
finite-only JSON parse (NaN/Infinity rejected) → object shape → approved schema
(`footballai.analysis-diagnostics/v2` real, or `/v1` synthetic) matching the
registered `schema_version` → an **allow-listed** field extraction. Any failure
yields a safe partial result with a reason in `missing_evidence`; it never
raises to the user and never fabricates a field. Unapproved fields (including
any injected instruction text) are dropped and never reach the model.

## Real vs. synthetic output

A real uploaded video does **not** imply real inference. A `demo_fast` run is
synthetic regardless of input origin, and that is surfaced as
`is_synthetic = true` with an explicit explanation and a leading limitation. The
`v1` diagnostics schema (`data_quality: synthetic`, "No ML model was executed")
and the `v2` descriptive fragmentation proxies are handled distinctly.

## Grounding policy

A versioned instruction block (`copilot.grounding-instructions/v1`) tells the
model to use only the supplied evidence, treat it as untrusted data (never
instructions), distinguish real from synthetic output, cite factual claims,
state limitations, and abstain when the information is not present. Questions
about passes, shots, possession, formations, tactics, player identities, verified
distances, or clinical/injury risk are classified as **unsupported** and the
answer abstains (`support_level: unavailable`).

## Citation contract and validation

Citations are **derived deterministically from the validated evidence**, never
emitted by the model, so a model cannot fabricate a source. Each citation
carries `run_id`, `field_path`, and — when evidence comes from an artifact —
`artifact_id`, `schema_version`, and `source_sha256`. `validate_citations` is a
fail-closed gate: a citation to another run, to a field that was not supplied, to
an unregistered or integrity-failed artifact, or with a mismatched schema/sha256
raises `ForgedCitationError`, and the endpoint returns `422` rather than an
answer.

## Provider abstraction: fake vs. real

`LLMProvider` has two implementations:

- **`DeterministicFakeProvider`** (default) — offline, deterministic, renders the
  explanation from the evidence snapshot. It always reports
  `model_mode: offline_demo_fake` and is **never** presented as a real model.
- **`AnthropicTextProvider`** (opt-in) — disabled unless both
  `FOOTBALLAI_COPILOT_PROVIDER=real` and
  `FOOTBALLAI_COPILOT_REAL_PROVIDER_ENABLED=1` are set and `ANTHROPIC_API_KEY` is
  present. It imports the SDK lazily (so the default install and CI never touch
  it), is bounded by `max_output_tokens` and `timeout_seconds`, performs no
  retries, and produces prose only (citations remain evidence-derived).

Budgets (`CopilotSettings`): `max_artifact_bytes` 1 MiB, `max_evidence_bytes`
32 KiB, `max_prompt_chars` 24 KiB, `max_output_tokens` 700, `timeout_seconds`
20, `provider_retries` 0.

## Feature gate (development only)

The application has no user authentication yet, so the endpoint is a
**development-only** capability. `CopilotSettings.from_environment()` enables it
only when `FOOTBALLAI_COPILOT_ENABLED=1` **and** `FOOTBALLAI_ENVIRONMENT` is
`local` or `test`; in `staging`/`production` it is forced off. When disabled the
route is **not registered at all** (a plain `404`), so it is invisible in hosted
environments. Run UUIDs are not treated as access control, and there is no shared
demo password. Production access control and quotas are a hard prerequisite
before any public exposure.

## API

`POST /api/v1/runs/{run_id}/explain` with an optional `{"question": "..."}` body
returns:

```json
{
  "run_id": "…",
  "answer": "…",
  "support_level": "supported | partial | unavailable",
  "model_mode": "offline_demo_fake | real_model",
  "citations": [{"run_id": "…", "field_path": "…", "artifact_id": "…", "schema_version": "…", "source_sha256": "…"}],
  "limitations": ["…"],
  "unavailable_reason": null
}
```

Invalid UUID → `422`; unknown run → `404`; non-completed run → `409`.

## Dashboard

The Run Detail page gains one "AI Analysis Explanation" panel with an *Explain
this run* action. It shows the answer, a support-level badge, a model-mode badge
(clearly labelling the offline deterministic demo), evidence citations, and
limitations, plus loading, error, and "disabled in this environment" states.

## Local setup

```bash
# backend (feature gate on, offline fake provider)
FOOTBALLAI_COPILOT_ENABLED=1 FOOTBALLAI_ENVIRONMENT=local \
  PYTHONPATH=v2/src .venv-test/bin/python -m uvicorn footballai_v2.api.main:app

# dashboard
cd v2/dashboard && npm run dev
```

## Testing

```bash
PYTHONPATH=v2/src .venv-test/bin/python -m pytest \
  v2/tests/test_copilot_explanation.py v2/tests/test_copilot_api.py -q
cd v2/dashboard && npm run test
```

The suite is fully deterministic and never makes a real LLM call: the default
provider is the offline fake, and the real provider is exercised only through
typed failure stubs.

## Why this is not yet an autonomous agent

Phase 1 is a single read-only query: no tool calling, no agent loop, no memory,
no conversation store, no external browsing, and no RAG. The model receives one
bounded evidence snapshot and returns one short grounded paragraph.

- **Phase 2** — a bounded, read-only *tool-using* copilot built on
  `RunEvidenceService` and the provider protocol.
- **Phase 3** — retrieval (RAG) only where it is genuinely justified.

Real authentication/authorization and quotas remain the prerequisite before any
of this is exposed publicly.
