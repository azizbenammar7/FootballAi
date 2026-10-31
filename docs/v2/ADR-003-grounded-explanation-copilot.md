# ADR-003: Evidence-grounded "Explain This Run" copilot (Phase 1)

**Status:** Accepted, amended by Phase 1.1 (see [Amendment](#amendment-phase-11-server-rendered-claims))
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
   bounded tokens/timeout, lazily imported so the default install never needs it).
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

> Items 2–4 above describe the original Phase 1 design. The Amendment below
> supersedes how the provider, grounding, and service interact.

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
absent. This keeps the default install LLM-free. Since Phase 1.2, CI installs the
SDK to run contract tests against a local server; it sends nothing to a provider.

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
5. [x] Phase 1.1: server-rendered claim catalog, deterministic abstention,
       output-nature classification, bounded structured prompt, finite-number
       validation, and a fail-closed gate (see Amendment).
6. [ ] Validate the real provider against the live API. It has been exercised
       against the real SDK and a local server, not the live API. The evaluation
       harness is ready: see [COPILOT_LLM_EVALUATION.md](COPILOT_LLM_EVALUATION.md).
7. [ ] (Phase 2 prerequisite) Real authentication/authorization + quotas.

## Amendment: Phase 1.1 server-rendered claims

**Date:** 2026-10-30

### Context

Review of the Phase 1 design found that it did not ground real-model answers:

- Model free text was returned as `support_level: supported` with citations built
  from the evidence, not from what the text said. A reproduction against the
  original code returned `supported`, with 8 valid citations, for the answer
  "Player 7 ran 10.5 km and scored twice."
- The provider was called, and billed, before an unsupported question abstained.
- A word deny-list could not enumerate unsupported questions ("How far did player
  7 run?" was treated as a general question).
- Prompts were truncated by slicing serialized JSON, which produced invalid JSON
  and could remove `is_synthetic`.

A valid citation only proves that a field exists. It says nothing about the
sentence next to it.

### Decision

The model no longer writes anything the user reads as fact.

```
ValidatedRunEvidence
   │  question router (closed vocabulary) ── no route ──▶ abstain, no provider call
   ▼
evidence-claim catalog (server-authored text, per-claim citations)
   │  claims eligible for the question ───── none ─────▶ abstain, no provider call
   ▼
provider selects claim ids (fake: all eligible; real: relevance and order)
   │  unknown or ineligible id ──────────────────────▶ rejected (422)
   │  no claim on the question's topic ──────────────▶ unavailable
   ▼
server renders claim text; output-nature claim always first
   ▼
citations of the rendered claims only ─ validate_citations ─▶ ExplanationResponse
```

### Options considered

| Option | Stops invented facts? | Model usefulness | Complexity | Decision |
|---|---|---|---|---|
| **A. Claim selection, server rendering** | Yes, by construction: no model text is published | Limited to relevance and ordering | Low | **Chosen** |
| B. Free text plus keyword or number filtering | No: filters only catch patterns someone anticipated | Highest | Medium, with permanent filter upkeep | Rejected |
| C. Free text plus entailment checking against evidence | Probabilistic, and needs an evaluation set that does not exist | High | High | Deferred until an evaluation exists |
| D. Remove the LLM entirely | Yes | None | Lowest | Rejected: leaves no path to validate a real model |

### Trade-offs

**Safety over fluency.** Answers read as a list of verified statements rather
than natural prose. That cost is deliberate.

**The model's role is small, and stated plainly.** It decides which verified
facts answer a question, and in what order. It does not explain, interpret, or
summarize. Because the closed-vocabulary router handles scope, the value of the
real model over the deterministic fake is unproven; that is what the real-model
evaluation has to measure. If it adds nothing, option D is the honest outcome. A
richer role, such as clearly labelled unverified commentary, needs its own
decision and evaluation.

**Over-abstention over false confidence.** Any unknown word abstains, so some
reasonable questions are refused. The vocabulary is extended deliberately.

### Consequences

- **Easier:** The grounding guarantee is structural and testable without a model.
- **Harder:** Every new fact needs a claim template, citations, and vocabulary.
- **Not solved:** Claims are as accurate as the CV output behind them. Integrity
  checks prove bytes match the manifest, not that the content is true. Relevance
  of a model's selection is not validated.
- **Revisit:** The default model is now `claude-haiku-5-5`, chosen from documentation
  in Phase 1.2. It has not been validated with this structured-output request on the
  live API.
