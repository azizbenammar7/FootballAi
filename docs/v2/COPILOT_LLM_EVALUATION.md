# Copilot LLM evaluation (Phase 1.2)

Whether a real model selects evidence claims better than deterministic code, and
what it would cost to find out.

**Status on 2026-10-31.** The harness and the deterministic baseline are complete.
**No real model has been called.** Every score below is from deterministic code.
Anything about model quality is planned, not measured. No `ANTHROPIC_API_KEY` was
available during this work, so model availability is documented, not verified live.

Scope: this measures the `copilot.grounding-instructions/v3` selection step only.
It adds no agent, tool calling, RAG, MCP, or cloud service. See
[ADR-003](ADR-003-grounded-explanation-copilot.md) for why the model may only
select claims, and [COPILOT_EXPLANATION.md](COPILOT_EXPLANATION.md) for the feature.

## Model selection

**Recommended: `claude-haiku-5-5` at `effort: low`.** It is the new default in
`CopilotSettings`; the real provider stays disabled unless explicitly enabled.

Picking claim ids from a short list is a small classification task, which is what
this model is documented for. Nothing yet shows that a larger model selects better.

| Model | Lifecycle (docs) | Input / output, $ per MTok | Estimated cost of a 12-request sample | Worst case |
|---|---|---|---:|---:|
| `claude-haiku-5-5` | Active, not retired before 2027-10-07 | 0.10 / 0.50 | $0.0021 | $0.011 |
| `claude-sonnet-5-5` | Active | 2.00 / 10.00 | $0.043 | $0.21 |
| `claude-opus-4-8` (previous default) | Active, legacy | 5.00 / 25.00 | $0.11 | $0.54 |

Prices and lifecycle are from the [pricing](https://platform.claude.com/docs/en/about-claude/pricing)
and [deprecation](https://platform.claude.com/docs/en/about-claude/model-deprecations)
pages on 2026-10-31. Estimates are the offline heuristic described under
[Cost](#cost). Haiku 5.5 is documented as supporting structured outputs and effort;
`--preflight` confirms both for your account before any paid call.

The runner also accepts `claude-sonnet-5-5`. Run it only if Haiku's selections are
measurably poor, so that a larger model has something to prove.

## Provider corrections

The merged provider had only ever run against a local stub. Checking it against the
current docs and the real SDK found:

| Finding | Effect | Correction |
|---|---|---|
| A response with `content: null` raised an uncaught `TypeError` | HTTP 500 instead of a typed failure | Fails closed as `invalid_response` |
| Duplicate ids were silently de-duplicated | An invalid reply was accepted. The schema cannot express `uniqueItems` | Rejected by the typed contract |
| Timeouts were detected from a class name; other errors were untyped | A failure could not be diagnosed, or stopped on | Typed `ProviderFailureKind`, from SDK classes and status codes |
| No usage, request id, or latency was captured | Spend was invisible, including billed refusals and invalid replies | `ProviderTelemetry`, on success and on failure |
| Haiku 5.5 thinks by default, and thinking counts toward `max_tokens` | A 700 cap risked truncation | `effort: low`, cap raised to 1024 |
| Default model cost 50 times the recommendation | Unjustified for this task | Default is `claude-haiku-5-5` |
| `anthropic` was not declared anywhere | The real path could not be installed reproducibly | Pinned in `requirements-copilot-eval.txt`, outside every runtime image |

API clients see no change: `unavailable_reason` is still `provider_unavailable` or
`provider_timeout`. The detailed kind stays server-side.

SDK 2.x facts to keep in mind: it sends API version `2026-10-21`, on which sampling
parameters are a 400 (none are sent), and it sends the key as
`Authorization: Bearer`, not `X-Api-Key`, so a proxy reading the old header needs
changing. SDK 2.0 was released on 2026-10-21, so the pin is young.

Timeouts stay at 20 seconds. A schema's first use compiles a grammar, so a timeout
on the first requests would be a finding to investigate, not noise.

## Structured output and claim validation

The model returns only `{"claim_ids": [...]}`, constrained two ways:

1. **By the API:** `output_config.format` with a JSON schema whose `enum` holds the
   ids offered for this question.
2. **By the server:** `ClaimSelection` is strict about shape, type, extra fields,
   duplicates, and length. `render_from_selection` then rejects any id not eligible
   for this question on this run.

The second layer is not redundant. A refusal or truncated reply may not match the
schema, and `uniqueItems` cannot be expressed in it.

| Model reply | Result |
|---|---|
| Malformed JSON, prose, wrong types, missing or extra fields, duplicates, empty text | `provider_unavailable` |
| Unknown id, an id from another run's evidence, an id outside the question's topic | Rejected (`forged_citation`, HTTP 422) |
| Empty list, or only the label on a question about another topic | `no_relevant_claim` |
| Refusal or truncation | `provider_unavailable` |

The server still owns all wording, citations, the synthetic or real label, limitations,
and abstention. No model text reaches an answer.

Each distinct enum is a distinct schema with its own grammar compilation, cached for
24 hours. The runner reports first-use and repeat latency separately for this reason.

## Evaluation dataset

`evaluation/copilot/v1/` holds data only, apart from the code that scores it, and
reports record its SHA-256.

- `fixtures.json`: 9 run fixtures. Payloads are the approved test fixtures verbatim,
  and a test enforces that. They are hand-written test values, not match measurements.
- `cases.json`: 52 cases.
  - **42 selection cases** carry a question, fixture, expected outcome, required and
    permitted claims, support level, and citations. Any selector can run them.
  - **10 contract cases** replay scripted invalid replies. They are never sent to a
    model.

Labels state what a person would want, not what the router happens to do. An `either`
outcome marks a question where a permitted answer and abstaining are both acceptable.
`coverage_gap` marks a case where abstaining is today's policy although the evidence
could answer.

## Deterministic baseline (measured)

Reproduce with the first command under [Running](#running). A test fails if the
committed `baseline-report.json` stops matching the code and dataset.

| Metric | Deterministic fake (select everything) | Lexical overlap (reference rule) |
|---|---:|---:|
| Claim selection accuracy | **19/28, 67.9%** (CI 49–82%) | 20/28, 71.4% (CI 53–85%) |
| Abstention accuracy | **39/42, 92.9%** (CI 81–98%) | 36/42, 85.7% (CI 72–93%) |
| Unsupported claim rate | **53/139, 38.1%** | 19/77, 24.7% |
| Citation validity | **33/33, 100%** (CI 90–100%) | 29/29, 100% (CI 88–100%) |
| Support level accuracy | 39/42, 92.9% | 36/42, 85.7% |
| Output nature accuracy | 42/42, 100% | 42/42, 100% |
| Fabricated answers | 0 | 0 |
| Invalid output rate | 0/33 | 0/33 |
| In-process latency, p50 / p95 | 0.19 / 0.21 ms | 0.20 / 0.26 ms |

Intervals are Wilson 95%. The contract suite failed closed on **10/10** replies
(CI 72–100%).

"Unsupported" here means true and cited, but not permitted for that question. It
never means invented: fabrication is checked separately and is zero by construction.

| Category | Cases | Selection | Abstention |
|---|---:|---:|---:|
| verified_real | 7 | 5/7 | 7/7 |
| tracking_fragmentation | 7 | 2/6 | 6/7 |
| model_provenance | 3 | 0/2 | 2/3 |
| synthetic_demo_fast | 5 | 4/4 | 5/5 |
| missing_or_corrupt | 6 | 4/5 | 6/6 |
| unsupported_statistic | 2 | n/a | 2/2 |
| player_distance | 2 | n/a | 1/2 |
| possession_pass | 3 | n/a | 3/3 |
| ambiguous | 4 | 2/2 | 4/4 |
| malicious_artifact | 3 | 2/2 | 3/3 |

The two rules disagree on 7 cases: lexical overlap improves 4 and breaks 3. Neither
is a usable selector, which is the point of measuring them.

## What a model could change (measured, without a model)

The baseline bounds the upside of any selector:

| Selection cases | 42 |
|---|---:|
| Stopped before any selector, by the router or missing evidence | 9 |
| Reach a selector | 33 |
| Fake already right: a model can only tie or lose | 21 |
| **Fake wrong: a model could help** | **12** |

Of those 12:

- **9 are over-selection.** The fake adds true, cited claims from the right topic. The
  reader sees noise, not an error. Most come from coarse topics, for example "duration"
  routing to provenance as well as tracking, so finer deterministic routing may fix
  them with no model at all. That is unmeasured.
- **3 are false answers:** `dist-known-words`, `model-version`, and
  `track-short-no-claim`. The question passes the vocabulary, no claim answers it, and
  the fake reports `supported` with claims that do not answer. This is the clearest
  case for a selector, and also the most worrying baseline defect. Every word of "How
  long did the tracked run?" is known, so the router cannot stop it. The claims shown
  are true, but `supported` overstates them.

A model cannot help with `amb-median`: `median` is outside the closed vocabulary, so
no selector ever sees the question. Such gaps need vocabulary or new claims.

**These counts come from a dataset designed to include hard cases.** They show where
a model could matter, not how often real users would hit it. There is no production
question traffic to sample.

## Running

From the repository root. `--dataset` is always required.

**Dry run (default).** No SDK, key, or network:

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_eval --dataset evaluation/copilot/v1
```

**Preflight.** Needs a key and sends no `messages.create`, so it is not billed. The
Models API confirms the model is active with structured outputs and effort, and the
free token-counting endpoint gives exact input tokens:

```bash
pip install -r v2/requirements-copilot-eval.txt
```

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_eval --dataset evaluation/copilot/v1 --preflight --sample 12
```

**Paid run.** Only after explicit human approval. All three limits are mandatory:

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_eval --dataset evaluation/copilot/v1 --allow-paid-calls --model claude-haiku-5-5 --max-requests 12 --max-cost-usd 0.05 --output evaluation/copilot/v1/haiku-5-5-sample-12.json
```

A paid run repeats preflight first, runs the sampled cases through the real
`ExplanationService`, and scores the deterministic rules on exactly the same cases.
Exit codes: `2` refused, `3` preflight failed, `4` stopped early, with a report.

## Cost

An estimate is not a bill. The report's `list_price_estimate_usd` is reported tokens
at list price; only the Anthropic Console shows billed usage.

**Offline estimate, planned and unmeasured:** characters ÷ 3, plus an assumed 300
tokens for the system prompt the API adds for structured outputs (its size is not
documented) and an assumed 150 thinking tokens. `--preflight` replaces the input side
with exact counts. Thinking tokens remain unknown until a real run.

| Bounded first experiment | Value |
|---|---:|
| Model | `claude-haiku-5-5` |
| Requests | 12 (stratified, deterministic) |
| Expected input tokens per request | about 900 |
| Expected output tokens per request | about 177 |
| Estimated cost per request | $0.00018 |
| **Estimated total** | **$0.0021** |
| Worst case | $0.011 |
| Requested budget cap | $0.05 |

All 33 reachable cases are estimated at $0.0059, worst case $0.030. In this
experiment, cost is not the constraint: at these rates the question is whether a
model adds value, latency, and a failure mode, not whether it is affordable.

## Safety boundaries

Paid calls are refused unless all of these hold:

- `--allow-paid-calls`, an explicit `--model` with a configured price, `--max-requests`
  (at most 200), and `--max-cost-usd` (at most $5).
- `ANTHROPIC_API_KEY` is set, and `CI` or `GITHUB_ACTIONS` is not.
- Preflight passes.

During a run:

- A request is admitted only if its **worst case** still fits the budget.
- SDK retries are forced off, so one call is one request.
- A request with missing or malformed usage, such as a timeout, is charged at its
  worst case.
- Authentication, permission, billing, model, bad-request, and rate-limit failures stop
  the run at once. Three consecutive failures of any kind also stop it.
- Failures stay in the report and count against the model.

Data boundaries:

- Only versioned fixtures reach a model. No real run, video, or storage is read.
- Reports hold case ids, claim ids, token counts, latency, and request ids. They hold
  no key, prompt, or question.
- Contract cases and router-stopped cases are never sent.
- The API and worker never import `footballai_v2.copilot.evaluation`, and a test
  enforces that. Normal CI installs the SDK only to run contract tests against a local
  server on 127.0.0.1.

The existing dev-only gate is unchanged. Real authentication, authorization, and
quotas remain a prerequisite for any hosted use.

## Metrics

| Metric | Definition | Denominator |
|---|---|---|
| Claim selection accuracy | Answered, every required claim shown, nothing unpermitted | Cases expecting an answer |
| Abstention accuracy | Answered or abstained as expected, with the expected reason | All selection cases |
| Unsupported claim rate | Rendered claims the case does not permit | Rendered claims, excluding the label |
| Citation validity | Every citation resolves, and they are exactly those of the rendered claims | Answered cases |
| Fabricated answers | Answer is not exactly the rendered claims' server text | All selection cases |
| Invalid output rate | Schema-invalid or forged replies | Provider invocations |

A provider failure is never a correct abstention, even where abstaining was expected.

## Decision rule

Accepted by the project owner on 2026-10-31, before any model run, so the result
cannot choose its own bar. Changing it after a paid run needs a recorded reason.

Adopt a real selector only if, on the same cases, against the better deterministic rule:

1. Output nature stays 100%, with no new fabricated answer and no new false answer.
2. It corrects at least 2 of the 3 false answers.
3. It regresses no case the fake gets right, or each regression is reviewed and accepted.
4. Provider failures plus invalid outputs stay at or below 5%.
5. Repeat-schema p95 latency stays at or below 3 seconds.

Otherwise keep deterministic selection. ADR-003 already records that dropping the
model is an acceptable outcome.

A 12-case sample can show that the path works and find gross failures. It **cannot**
establish a rate: 12 of 12 still has a lower confidence bound near 76%. Treat it as
readiness evidence. The comparison needs the 33 reachable cases, and only discordant
cases carry evidence.

## Limitations

- **No model has been measured.** Haiku 5.5's suitability is a reasoned recommendation.
- **One author wrote the labels, the dataset, and the lexical rule.** They need review.
  A selector could be tuned to them, so a real tuning effort needs a held-out set.
- **Small sample.** With 28 scored cases, intervals span about 30 points.
- **Fixtures are synthetic test values**, and one question often recurs across fixtures.
- **Relevance labels are judgement calls.** Some permitted sets are deliberately strict.
- **Grounding is not accuracy.** This measures which claims are chosen, not whether the
  CV output behind them is right.
- **Local validation gaps.** gitleaks and the PostgreSQL and Azurite suites run in CI,
  not locally.
