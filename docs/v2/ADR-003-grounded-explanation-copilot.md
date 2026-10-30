# ADR-003: Evidence-grounded "Explain This Run" copilot (Phase 1)

**Status:** Accepted
**Date:** 2026-10-30
**Deciders:** FootballAI V2 maintainers

## Context

FootballAI V2 produces immutable analysis runs described by the
`footballai.analysis-run/v1` contract, with artifact bytes behind the
`ObjectStorage` port and lifecycle behind the `AnalysisRepository` port. There is
no LLM capability today, and the first AI-engineering increment is a single,
bounded feature: let a user open a completed run and receive an
**evidence-grounded** explanation of what the run actually is (status, profile,
whether the output is real or synthetic, provenance, artifacts, and the
registered AI1 tracking diagnostics).

Forces at play:

- **Correctness over fluency.** The explanation must never invent football facts
  (player names, persistent identities, passes, shots, possession, formations,
  tactics, verified distances, clinical risk). It must distinguish *real AI1
  inference* from *synthetic `demo_fast` output*, which is synthetic regardless
  of whether a real video was uploaded.
- **No authentication exists yet.** Run UUIDs are not access control. Any LLM
  endpoint must be a local/dev-only, default-disabled feature gate.
- **No new heavy dependencies.** No LangChain/LangGraph, no agent, no tool
  calling, no RAG, no MCP in this phase. A real paid model must be opt-in,
  server-side, bounded, and never billed in CI.
- **Reuse the existing boundaries.** Direct SQL, arbitrary filesystem reads, and
  reimplemented storage adapters are prohibited. Evidence must come through the
  existing ports with their integrity verification.

## Decision

Introduce a new, self-contained bounded context `footballai_v2.copilot` that
depends only on the `v1` contracts and the two storage ports. It contains:

1. **`RunEvidenceService`** — read-only. Given a `run_id`, it loads the run via
   `AnalysisRepository`, strictly validates the registered
   `analysis-diagnostics` artifact through `ObjectStorage`, and returns a typed,
   allow-listed `ValidatedRunEvidence` snapshot. No raw URIs, SAS URLs,
   credentials, or arbitrary manifest text are ever included.
2. **`LLMProvider` protocol** with two implementations: a default
   `DeterministicFakeProvider` (offline, deterministic, labelled as a demo) and an
   opt-in `AnthropicTextProvider` (disabled by default, server-side creds,
   bounded tokens/timeout, lazily imported so CI never touches the SDK).
3. **Grounding + citation validation** — a versioned instruction block, a bounded
   evidence payload, and deterministic fail-closed validation of every citation
   (must belong to the requested run, resolve to a supplied field path, and match
   an integrity-verified artifact).
4. **`ExplanationService`** — orchestrates evidence → provider → validation →
   typed Pydantic response (`answer`, `support_level`, `citations`,
   `limitations`, `unavailable_reason`, `model_mode`, `run_id`).
5. **One FastAPI route** `POST /api/v1/runs/{run_id}/explain`, registered only
   when the dev-only feature gate is enabled.

The React dashboard gains one `ExplanationPanel` on the existing Run Detail page.

## Options Considered

### Option A: New `copilot` bounded context over the existing ports (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost | No new prod dependency; fake provider is free |
| Scalability | Isolated; Phase 2 tool-calling can extend the same context |
| Team familiarity | Follows existing ports/contract/thin-route patterns |

**Pros:** Clear separation of the LLM concern from CV execution and transport;
testable against the ports with fakes; reuses integrity-verified reads; no
storage reimplementation; Phase-2 ready.
**Cons:** One more package to maintain.

### Option B: Put the logic directly in `api/app.py`

**Pros:** Fewer files.
**Cons:** Mixes LLM/grounding concerns with HTTP transport, breaks the existing
thin-route convention, and is hard to unit-test without a running app. Rejected.

### Option C: Add it as an execution stage in `execution/`

**Pros:** Reuses the worker.
**Cons:** Explanation is a read-only, post-hoc, on-demand query — not a pipeline
stage that mutates a run. Conflates execution with inspection. Rejected.

## Trade-off Analysis

The decisive trade-off is **isolation + testability vs. file count**. Because the
feature must fail closed on forged citations and must never leak synthetic/real
confusion, deterministic unit tests against the ports are worth far more than
saving a package. Option A is the only one that lets the evidence contract and
citation validator be tested in isolation from both FastAPI and real storage.

A second trade-off is **real-model optionality vs. dependency weight**. Rather
than adding `anthropic` as a production dependency, the real provider imports the
SDK lazily and reports `provider_unavailable` when the package or credentials are
absent. This keeps the default install, and CI, entirely LLM-free.

## Consequences

- **Easier:** Grounding policy, citation validation, and synthetic/real handling
  become unit-testable without a browser, a server, or a paid API.
- **Easier:** Phase 2 (bounded read-only tool-using agent) can build on
  `RunEvidenceService` and the provider protocol.
- **Harder:** Every new evidence field must be added to the allow-list *and* the
  citable-field set deliberately — by design, not by accident.
- **Revisit:** The dev-only feature gate must be replaced by real authentication,
  authorization, and quotas before any public/hosted exposure. This is an
  explicit Phase-1 non-goal and a hard prerequisite for production.

## Action Items

1. [x] Add `footballai_v2.copilot` with evidence, provider, grounding, service,
       models, config, and errors modules.
2. [x] Register `POST /api/v1/runs/{run_id}/explain` behind the dev-only gate.
3. [x] Add the React `ExplanationPanel` to the Run Detail page.
4. [x] Deterministic test suite covering real, synthetic, missing, corrupt,
       oversized, cross-run, forged-citation, and abstention cases.
5. [ ] (Phase 2 prerequisite) Real authentication/authorization + quotas.
