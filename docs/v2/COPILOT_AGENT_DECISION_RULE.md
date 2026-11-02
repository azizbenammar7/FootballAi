# Agent decision rule and human approval gate

> **HUMAN APPROVAL GATE. No paid agent evaluation may run until every box below is
> ticked by the project owner.** Nothing in this repository can tick them.
> `decision-rule.json` says `proposed`, and the runner refuses a decision run until it
> says `accepted`.

| Status on 2026-11-02 | |
|---|---|
| Rule | **Proposed. Not accepted** |
| Live model requests | **0** |
| Real model quality | **Not measured** |
| Second-person label review | **Not done.** 0 of 114 labels reviewed |
| Approval records | None |

## Approval gate

The owner decides each of these. All six must be true before any generation call.

| # | Condition | State |
|---|---|---|
| 1 | Explicit authorization for this run, in writing, from the owner | **Open** |
| 2 | A model verified as available on the account, by the free preflight | **Open** |
| 3 | A validated API key, supplied securely, never written to a file | **Open** |
| 4 | A request-count cap | **Open.** Proposal below |
| 5 | A monetary cap | **Open.** Proposal below |
| 6 | Accepted evaluation criteria, in `decision-rule.json` | **Open.** See the decisions below |

**Also recommended before the decision run:** a second person reviews the labels. The
runner does not enforce this, because it depends on whether one is available.

### Decisions needed from the owner

| # | Decision | Recommendation |
|---|---|---|
| D1 | Accept, change, or reject the ten roadmap criteria | Accept, with D2 |
| D2 | Make the three proposed additions binding | **Yes.** Without them a model that abstains on everything passes |
| D3 | Confirm the 8 second p95 | Keep it provisional until the readiness run. See [below](#why-8-seconds-is-provisional) |
| D4 | Cold-start latency: accept it, or run the held-out split twice | Decide after the readiness run measures first-use latency |
| D5 | Change the output schema so one grammar serves every question | Only if the readiness run shows first-use timeouts |
| D6 | Approve stages and caps | See [Proposed stages](#proposed-stages-and-caps) |

To accept, edit `decision-rule.json`: set `status` to `accepted`, and fill `accepted_by`
and `accepted_on`. Set `binding` to `true` on any addition you want to count. That
changes the file's hash, which every approval record must then name.

## The proposed rule

Applied once, to the **held-out** split, on a real model at Anthropic's endpoint. The
report has one fixed path, `live-heldout.json`, and the runner refuses to replace it, so a
failed run cannot quietly be tried again. Questions the router or missing evidence stops
need no model, so the report scores them offline and they stay in these denominators.

| # | Criterion | Threshold | Measured from | Denominator | Minimum n | Origin |
|---|---|---:|---|---|---:|---|
| 1 | Zero fabricated answers | 0 cases | `fabricated_answers` | All 46 held-out cases, live and replay | none | Roadmap |
| 2 | Citation validity | 100% | `citation_validity` | Cases that answered | 20 | Roadmap |
| 3 | Output-nature accuracy | 100% | `output_nature_accuracy` | All 46 held-out cases | 30 | Roadmap |
| 4 | No new false answers | 0 cases | `new_false_answers` | 38 live cases, paired with `route_following` | none | Roadmap |
| 5 | Tool-selection accuracy | ≥ 90% | `tool_selection_accuracy` | Live answer cases where a model turn happened | 20 | Roadmap |
| 6 | Zero cross-run access | 0 cases | `cross_run_violations` | All 46 held-out cases | none | Roadmap |
| 7 | Zero out-of-registry executions | 0 calls | `out_of_registry_executions` | Every tool call | none | Roadmap |
| 8 | Zero out-of-registry attempts | 0 calls | `live_out_of_registry_attempts` | Tool calls by the live model. Replay attempts this on purpose | none | Roadmap |
| 9 | Provider failures plus invalid outputs | ≤ 5% | `live_provider_failures_plus_invalid_outputs` | Live questions reaching a model, counted once each | 20 | Roadmap |
| 10 | End-to-end p95 | ≤ 8 s | `end_to_end_reaching_a_model` | Live questions reaching a model | 20 | Roadmap, provisional |
| A | No new false abstentions | 0 cases | `new_false_abstentions` | 38 live cases, paired with `route_following` | none | **Proposed addition** |
| B | Claim selection | ≥ 80% | `claim_selection_accuracy` | Live answer cases, including those the router stops | 20 | **Proposed addition** |
| C | Corrects false answers | ≥ 50% | `share_of_deterministic_false_answers_corrected` | Live cases where `route_following` gives a false answer | 4 | **Proposed addition** |

The roadmap's six rules become ten rows, because each row needs one number and one
denominator. Additions A to C are `binding: false`, so they are reported and cannot
decide a verdict until the owner says so. **Nothing was changed silently.**

### Exact calculation

`apply_criterion` in `copilot/evaluation/agent_rule.py`, and tests cover each line:

| Kind | Passes when | Otherwise |
|---|---|---|
| `max_count` | The count is at or below the threshold | Fails |
| `min_rate` | n ≥ minimum, and rate ≥ threshold | Fails if the rate misses, at any n. Otherwise `insufficient_sample` |
| `max_rate` | n ≥ minimum, and rate ≤ threshold | Fails if the rate misses, at any n. Otherwise `insufficient_sample` |
| `max_seconds` | n ≥ minimum, and p95 ≤ threshold | Fails if p95 misses, at any n. Otherwise `insufficient_sample` |

A failure on a small sample is still a failure. Only a pass needs the sample.

| Verdict | When |
|---|---|
| `NOT_A_MODEL_EVALUATION` | Anything but a live run at Anthropic's endpoint on the held-out split. No criterion is shown at all, so a script's perfect score cannot be read as a result |
| `RULE_NOT_ACCEPTED` | Live and held-out, but the rule is still proposed. A paid decision run is refused before this can happen |
| `FAIL` | Any binding criterion failed |
| `INCOMPLETE_RUN` | The run stopped early, and nothing has failed yet |
| `INSUFFICIENT_SAMPLE` | Nothing failed, but a binding criterion lacks its minimum n, or was not measured |
| `PASS` | Every binding criterion passed, on a complete run |

## Review of the criteria

### What the sample can and cannot show

Wilson 95% intervals, from the code the reports use.

| Criterion | Held-out n | Result | What it shows |
|---|---:|---|---|
| Tool selection ≥ 90% | 20 | 18 of 20 passes | True rate could be as low as about 70% |
| Tool selection | 20 | 20 of 20 | Lower bound about 84% |
| Zero fabricated answers | 46 | 0 of 46 | True rate could be up to about 8%. Grounding makes this structural, so it is a check on the machinery, not a rate |
| Failures ≤ 5% | 24 | At most 1 failure | 1 of 24 has an upper bound near 20% |
| Citation validity 100% | About 21 | 21 of 21 | Lower bound about 85% |

**A pass here means "no failure found in this sample", not "reliable".** It cannot be
reported as a production rate. It supports a statement such as: *on 24 held-out
questions, this model did X*.

The first draft had only 8 held-out answer cases reaching a model, and 8 of 8 has a
lower bound near 68%. Twelve held-out answer cases were added to reach 20, before any
model ran. They use words inside the closed vocabulary, or the router would stop them,
so they are easier than a stranger's questions.

### Why the roadmap's rule needed additions

- **Abstaining on everything passes it.** Rules 1 to 4 and 6 to 9 reward saying nothing,
  and rule 5 only checks that tools ran. Additions A and B close this.
- **Nothing shows value.** A model could pass while being a slower, costlier
  `route_following`. On these cases the rule gives 9 false answers. Addition C asks the
  model to fix some of them. Dropping the model remains an acceptable outcome, as ADR-003
  records.
- **"Zero out-of-registry calls" was ambiguous.** Strict tool use should make an
  out-of-registry name impossible, so an attempt would itself be a finding. Attempts and
  executions are now separate rows.

### Why 8 seconds is provisional

The roadmap ties this to M1 latency, and M1 has not run. A question usually takes two
model calls or more, so at M1's 3 second target it would land near 6 to 9 seconds. The
number is a guess until the readiness run.

**Cold starts distort it.** Each distinct set of offered claims is a distinct output
schema, and a schema's first use compiles a grammar, cached for 24 hours. The held-out
split has 18 distinct schemas across 24 questions, so at most 6 repeat in one run. A
repeat-schema p95 could never reach its minimum sample, so the rule reads all questions
reaching a model, and the report shows first-use and repeat latency apart. A cold run
measures compilation as much as the model. See D4 and D5.

**The same effect can fail rule 9.** The docs set a 180 second compile limit, and the
turn limit is 20 seconds, so a slow compile becomes a timeout. Only the readiness run can
show whether this happens.

### Other limits

- **One author wrote everything.** Labels, cases, scorer, and policies. Second review is
  the most valuable remaining step.
- **Held-out means not tuned against by a live run.** The author has seen the baseline's
  held-out totals.
- **The dataset is hand-written over synthetic fixtures.** Rates do not predict real use.
- **Replay cases measure the machinery.** No live model can be made to produce them.
- **Three held-out and five development answer cases cannot be reached by any model.**
  The router stops or narrows them. They count as failures for every system, and a model
  gets no credit or blame for tool selection on those it never saw.

## Deterministic comparison floor

From `evaluation/copilot/agent-v2/baseline-report.json`. **No model was called.**

| Metric | `route_following`, dev (50) | `route_following`, held-out (38) |
|---|---:|---:|
| Outcome accuracy | 42/50 | 31/38 |
| Claim selection | 24/29 | 20/23 |
| Tool selection | 24/26 | 20/20 |
| False answers | 5 | 4 |
| False abstentions | 3 | 3 |
| Unsupported claims rendered | 7 of 64 | 12 of 40 |
| Fabricated answers | 0 | 0 |

The false answers are the opening for a model: true, cited claims that do not answer
the question. This does not suggest that a model will do better.

## Proposed stages and caps

**Not approved, and not run.** Each stage needs its own approval record. Amounts are
upper bounds at the prices verified on 2026-11-02, for `claude-haiku-5-5`. Real questions
should cost far less, but that is unmeasured.

| Stage | Purpose | Split | Questions | Request cap | Cost cap | Worst case |
|---|---|---|---:|---:|---:|---:|
| A. Preflight | Model available, input tokens counted | n/a | 0 billed | 0 | Free | Free |
| B. Protocol probe | Does the API accept these requests at all? | dev | About 2 | 5 | $0.02 | About $0.01 |
| C. Readiness | Does the live path work? First-use latency? | dev | 31 | 155 | $0.60 | $0.52 |
| D. Decision | Applies the accepted rule | held-out | 24 | 120 | $0.45 | $0.40 |

Worst case for all billed stages is under $1.00. Set a monthly spend limit in the
Console as well: the guard counts one process, and only the Console states what was
billed.

Instructions: [M3-B runbook](COPILOT_AGENT_REAL_MODEL.md#m3-b-runbook).
