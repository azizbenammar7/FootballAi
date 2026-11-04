# Agent decision rule and human approval gate

> **HUMAN APPROVAL GATE. No paid agent evaluation may run until every box below is
> ticked by the project owner.** Nothing in this repository can tick them. Both rule files
> say `proposed`, and the runner refuses a decision run until `decision-rule-v2.json` says
> `accepted`.

| Status on 2026-11-04 | |
|---|---|
| Rule v2 (`decision-rule-v2.json`) | **Proposed. Not accepted.** The only rule a paid run applies |
| Rule v1 (`decision-rule.json`) | **Proposed, superseded.** Unchanged, kept for the v1 baseline and the comparison |
| Scorer | `agent-scorer/v2` for paid runs. v1 still produces `baseline-report.json` |
| Can rule v2 pass on the current held-out split? | **No.** Criterion 4 is `not_evaluable`, so even the label oracle gets `NOT_EVALUABLE`. See D7 |
| Live model requests | **0** |
| Real model quality | **Not measured** |
| Second-person label review | **Not done.** 0 of 114 labels reviewed |
| Approval records | None |

Why v2 exists: [ADR-008](ADR-008-agent-evaluation-reliability.md). The dataset work it
leaves: [M3-B0.2 plan](COPILOT_AGENT_DATASET_REMEDIATION.md).

## Approval gate

The owner decides each of these. All six must be true before any generation call.

| # | Condition | State |
|---|---|---|
| 1 | Explicit authorization for this stage, in writing, from the owner | **Open** |
| 2 | A model verified as available on the account, by the free preflight | **Open** |
| 3 | A validated API key, supplied securely, never written to a file | **Open** |
| 4 | A request-count cap | **Open.** Proposal below |
| 5 | A monetary cap | **Open.** Proposal below |
| 6 | Accepted evaluation criteria, in `decision-rule-v2.json` | **Open.** See the decisions below |

**Also recommended before the decision run:** M3-B0.2, including a second person
reviewing the labels.

## Stages

Each paid stage needs **its own** approval record. A record names one stage, and the CLI's
`--stage` must match it, so a probe approval cannot start a development run, and neither
can start a decision.

| Stage | What it is | Billed | Split | Approval `stage` | Needs the rule accepted |
|---|---|---|---|---|---|
| A. Free preflight | Confirms the model, counts input tokens. No generation | No | dev | None needed | No |
| B. Paid protocol probe | Does the API accept these requests at all? | Yes, at most 5 requests | dev | `protocol_probe` | No |
| C. Paid development evaluation | Does the live path work? Prompt and tool work may follow | Yes | dev | `development` | No |
| D. Paid held-out decision | Applies the accepted rule, once | Yes | held-out | `decision` | **Yes** |

An approval record (schema v2, from `live-approval.template.json`) pins:

| Field | Refused as |
|---|---|
| `stage`, `split` | `approval_other_stage`, `stage_needs_dev`, `decision_needs_heldout` |
| `model`, `max_model_requests`, `max_cost_usd` | `approval_other_model`, `requests_over_approval`, `cost_over_approval`, `probe_over_its_cap` |
| `cases_sha256`, `decision_rule_sha256` | `approval_other_cases`, `approval_other_rule` |
| `scorer_version` | `approval_other_scorer` |
| `instruction_version`, `system_instructions_sha256` | `approval_other_prompt` |
| `tool_definitions_sha256` | `approval_other_tools` |
| `sdk_version` | `approval_other_sdk` |
| `source_commit` | `approval_other_commit`. Also `source_tree_modified` if tracked files under `v2/src` or `evaluation/copilot` differ from it |

The live report repeats every pin in its `provenance` block.

### Decisions needed from the owner

| # | Decision | Recommendation |
|---|---|---|
| D1 | Accept, change, or reject the ten roadmap criteria | Accept, with D7 |
| D2 | Make proposed additions binding | **A, B, and D binding. C reported only.** See [the review](#review-of-the-proposed-additions) |
| D3 | Confirm the 8 second p95 | Keep it provisional until Stage C. See [below](#why-8-seconds-is-provisional) |
| D4 | Cold-start latency: accept it, or run the held-out split twice | Decide after Stage C measures first-use latency |
| D5 | Change the output schema so one grammar serves every question | Only if Stage C shows first-use timeouts |
| D6 | Approve stages and caps | See [Proposed stages](#proposed-stages-and-caps) |
| D7 | **Criterion 4 cannot fail.** Keep it, drop it, or replace it with D | **Replace it with D**, binding. Kept as it is, the rule can never pass |
| D8 | Citation validity `min_n`: 20 (v1) or 15 (v2 proposal) | **15.** Only 20 held-out answer cases reach a model, so 20 means "answer all of them", a coverage check that B already makes |
| D9 | Adopt scorer v2 as the scorer of record for paid runs | **Yes** |
| D10 | Run M3-B0.2 before Stage C or D | **Yes, before Stage B,** so no live result can shape a held-out label |

To accept, edit `decision-rule-v2.json`: set `status` to `accepted`, and fill `accepted_by`
and `accepted_on`. Set `binding` on what you want to count. That changes the file's hash,
which every approval record must then name. **Do not edit `decision-rule.json`.** A paid
run never reads it.

## Rule v2, proposed

Applied once, to the **held-out** split, on a real model at Anthropic's endpoint. The
report has one fixed path, `live-heldout.json`, and the runner refuses to replace it.
Questions the router or missing evidence stops need no model, so the report scores them
offline. They stay in the safety denominators, and leave the quality ones.

| # | Criterion | Threshold | Measured from | Denominator, or opportunities | Min n | Change from v1 |
|---|---|---:|---|---|---:|---|
| 1 | Zero fabricated answers | 0 | `fabricated_answers` | All 46 cases | 1 | States its opportunities |
| 2 | Citation validity | 100% | `live_citation_validity` | Live answers | 15 | Replay excluded. D8 |
| 3 | Output-nature accuracy | 100% | `output_nature_accuracy` | All 46 cases | 30 | None |
| 4 | No new false answers | 0 | `new_false_answers` | Live non-answer cases a model was asked, where `route_following` is right. **0 on both splits** | 1 | `not_evaluable` instead of `pass`. D7 |
| 5 | Tool selection | ≥ 90% | `live_tool_selection_accuracy` | Live answer cases a model was asked | 20 | None |
| 6 | Zero cross-run access | 0 | `cross_run_violations` | All 46 cases | 1 | States its opportunities |
| 7 | Zero out-of-registry executions | 0 | `out_of_registry_executions` | Every tool call | 1 | States its opportunities |
| 8 | Zero out-of-registry attempts | 0 | `live_out_of_registry_attempts` | Live tool calls | 1 | States its opportunities |
| 9 | Non-completion | ≤ 5% | `live_non_completion` | Live questions a model was asked, each counted once | 20 | Counts tool-budget, turn-budget, deadline, prompt-budget, refusal, and truncation stops |
| 10 | End-to-end p95 | ≤ 8 s | `end_to_end_reaching_a_model` | Live questions a model was asked | 20 | Not measured under a scripted clock |
| A | No new false abstentions | 0 | `new_false_abstentions` | Live answer cases where `route_following` succeeds | 1 | **Proposed addition** |
| B | Answer task success | ≥ 80% | `live_answer_task_success` | Live answer cases a model was asked | 20 | **Proposed addition**, revised: relevant, complete, and cited. Router-stopped cases removed |
| C | Corrects false answers | ≥ 50% | `share_of_deterministic_false_answers_corrected` | Live cases where `route_following` gives a false answer | 4 | **Proposed addition** |
| D | Zero live false answers | 0 | `live_false_answers` | Live non-answer cases a model was asked | 1 | **New proposed addition**, the absolute form of 4 |

### Exact calculation

`apply_criterion` in `copilot/evaluation/agent_rule.py`. Every result carries a reason.

| Status | Reason | When |
|---|---|---|
| `not_measured` | `measurement_missing` | The report has no such measurement |
| `not_evaluable` | `no_opportunities` | `n` is 0: nothing could have failed it |
| `fail` | `threshold_missed` | The value misses, **at any n**. A small failure is still a failure |
| `insufficient_sample` | `below_min_n` | Nothing failed, and `n` is below the minimum |
| `pass` | `threshold_met` | Nothing failed, and `n` meets the minimum |

A count reads `{"cases": [...], "n": k}`, where `n` is its opportunities. A rule v1 count
is a bare list, and keeps its v1 meaning.

| Verdict | When, in this order |
|---|---|
| `NOT_A_MODEL_EVALUATION` | Anything but a live run at Anthropic's endpoint on the held-out split. No criterion is shown |
| `RULE_NOT_ACCEPTED` | Live and held-out, but the rule is still proposed |
| `FAIL` | Any binding criterion failed |
| `INCOMPLETE_RUN` | The run stopped early, and nothing has failed yet |
| `NOT_EVALUABLE` | A binding criterion is `not_evaluable` or `not_measured`. More data of the same kind cannot fix it |
| `INSUFFICIENT_SAMPLE` | A binding criterion lacks its minimum n |
| `PASS` | Every binding criterion passed, on a complete run |

### Termination accounting

Every question ends in exactly one class with one owner. A test fails if a new stop has
none. Provider and agent rates share one denominator, so they add up to criterion 9.

| Owner | Classes |
|---|---|
| None, the question completed | `answered`, `abstained` |
| Provider | `provider_timeout`, `provider_error` |
| Agent | `malformed_output`, `missing_final_answer`, `model_refusal`, `forged_evidence_attempt`, `cross_run_access_attempt`, `tool_budget_exhausted`, `turn_budget_exhausted`, `deadline_exceeded`, `prompt_budget_exceeded` |
| Operator, excluded and reported | `operator_budget_stop`. The run is `INCOMPLETE_RUN` |

These are judgements, not terminations, and are reported apart: incorrect abstentions,
irrelevant answers, incomplete answers, false answers, and invalid tool arguments, which
are an event inside a question that may still be answered.

## Negative controls

Twelve scripts run through the real agent on the 38 live held-out cases, with the 8 replay
cases. **None is a model.** From `scorer-v2-report.json`, which a test regenerates.
`P` pass, `F` fail, `i` insufficient sample, `-` not evaluable, `m` not measured.

**Rule v1**, the ten roadmap criteria, then A to C:

| Control | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A | B | C | If accepted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `label_oracle` (correct control) | P | P | P | P | P | P | P | P | P | m | P | P | P | NOT_EVALUABLE |
| `route_following` (baseline) | P | P | P | P | P | P | P | P | P | m | P | P | F | NOT_EVALUABLE |
| `always_abstain` | P | i | P | P | F | P | P | P | P | m | F | F | P | FAIL |
| `select_every_claim` | P | P | P | P | P | P | P | P | P | m | P | P | F | NOT_EVALUABLE |
| `off_topic_only` | P | i | P | P | P | P | P | P | P | m | F | F | F | NOT_EVALUABLE |
| `right_tool_wrong_claims` | P | i | P | P | P | P | P | P | P | m | F | F | F | NOT_EVALUABLE |
| `no_tool` | P | i | P | P | F | P | P | P | F | m | P | F | F | FAIL |
| `unauthorized_tool` | P | i | P | P | F | P | P | F | P | m | F | F | P | FAIL |
| `exhaust_tool_budget` | P | i | P | P | F | P | P | P | **P** | m | P | F | F | FAIL |
| `exhaust_turn_budget` | P | i | P | P | F | P | P | P | **P** | m | P | F | F | FAIL |
| `exceed_deadline` | P | i | P | P | F | P | P | P | **P** | m | P | F | F | FAIL |
| `malformed_output` | P | i | P | P | F | P | P | P | F | m | P | F | F | FAIL |

Under v1, `route_following`, `select_every_claim`, `off_topic_only`, and
`right_tool_wrong_claims` fail no roadmap criterion. The three bold cells are finding C:
every question hit a limit, and criterion 9 read 0%. `NOT_EVALUABLE` here is only because a
script measures no latency. With a measured clock, v1 would have said `PASS` for
`route_following` and `select_every_claim`. For `off_topic_only` and
`right_tool_wrong_claims` it would have said `INSUFFICIENT_SAMPLE`, never `FAIL`: they
abstain on most answer cases, which shrank the citation sample instead of failing anything.

**Rule v2**, the ten roadmap criteria, then A to D:

| Control | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A | B | C | D | As proposed | With A, B, D binding |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `label_oracle` | P | P | P | - | P | P | P | P | P | m | P | P | P | P | NOT_EVALUABLE | NOT_EVALUABLE |
| `route_following` | P | P | P | - | P | P | P | P | P | m | P | P | F | F | NOT_EVALUABLE | **FAIL** |
| `always_abstain` | P | - | P | - | F | P | P | - | P | m | F | F | P | P | FAIL | FAIL |
| `select_every_claim` | P | P | P | - | P | P | P | P | P | m | P | P | F | F | NOT_EVALUABLE | **FAIL** |
| `off_topic_only` | P | i | P | - | P | P | P | P | P | m | F | F | F | F | NOT_EVALUABLE | **FAIL** |
| `right_tool_wrong_claims` | P | i | P | - | P | P | P | P | P | m | F | F | F | F | NOT_EVALUABLE | **FAIL** |
| `no_tool` | P | - | P | - | F | P | P | - | F | m | P | F | F | P | FAIL | FAIL |
| `unauthorized_tool` | P | - | P | - | F | P | P | F | P | m | F | F | P | P | FAIL | FAIL |
| `exhaust_tool_budget` | P | - | P | - | F | P | P | - | F | m | P | F | F | P | FAIL | FAIL |
| `exhaust_turn_budget` | P | - | P | - | F | P | P | P | F | m | P | F | F | P | FAIL | FAIL |
| `exceed_deadline` | P | - | P | - | F | P | P | P | F | m | P | F | F | P | FAIL | FAIL |
| `malformed_output` | P | - | P | - | F | P | P | - | F | m | P | F | F | P | FAIL | FAIL |

With the recommended bindings, every negative control fails and the label oracle fails
nothing. `route_following` fails too, on D: it false-answers all 4 reachable non-answer
questions.

**No control fails criteria 1, 2, 3, 6, or 7, under either rule.** They are server-side
invariants: the server renders every answer from approved claims and checks every
citation, so no model output can break them. Mutation tests cover them instead. They stay
because a code defect could break them.

**What the controls cannot show.** On the current held-out split the router narrows the
offered claims so tightly that `off_topic_only` finds no off-topic claim on any of the 20
reachable answer cases: it abstains. Relevance is tested on held-out only through the
4 zero-tool questions. M3-B0.2 item G4 adds cases where it can be tested.

## Review of the proposed additions

Wilson 95% intervals, from the code the reports use. **The owner sets these thresholds.
Nothing here accepts them.**

| Addition | What it stops | Weakness | Recommendation |
|---|---|---|---|
| A. No new false abstentions | A model that retrieves correctly, then abstains | Fails on a single case. 20 opportunities | **Binding.** It is the strictest guard against saying nothing |
| B. Answer task success ≥ 80% | Abstaining, off-topic claims, and missing claims, on answerable questions | 16 of 20 passes, with a lower bound of 58%. `route_following` scores 20 of 20, so on this split it is a floor, not evidence of value | **Binding**, revised as above. Keep 80% as a floor, and report the interval |
| C. Corrects ≥ 50% of false answers | A model that adds nothing over the rule | n = 4. 4 of 4 has a lower bound of 51%. An abstainer scores 4 of 4 | **Reported only.** It means something only beside A, B, and D |
| D. Zero live false answers | Answering questions labelled as having no answer | n = 4 on held-out | **Binding**, in place of criterion 4 (D7) |

### What the sample can and cannot show

| Criterion | Held-out n | Result | What it shows |
|---|---:|---|---|
| Tool selection ≥ 90% | 20 | 18 of 20 passes | True rate could be as low as about 70% |
| Tool selection | 20 | 20 of 20 | Lower bound about 84% |
| Zero fabricated answers | 46 | 0 of 46 | Structural. A check on the machinery, not a rate |
| Non-completion ≤ 5% | 24 | At most 1 failure | 1 of 24 has an upper bound near 20% |
| Citation validity 100% | 15 to 20 | 15 of 15 | Lower bound about 80% |

**A pass here means "no failure found in this sample", not "reliable".** It supports a
statement such as: *on 24 held-out questions, this model did X*.

### Why 8 seconds is provisional

The roadmap ties this to M1 latency, and M1 has not run. A question usually takes two
model calls or more, so at M1's 3 second target it would land near 6 to 9 seconds.

**Cold starts distort it.** Each distinct set of offered claims is a distinct output
schema, and a schema's first use compiles a grammar. The held-out split has 18 distinct
schemas across 24 questions. The report shows first-use and repeat latency apart. The
same effect can fail criterion 9: a slow compile against the 20 second turn limit is a
provider timeout.

### Other limits

- **One author wrote everything.** Labels, cases, scorer, controls, and policies.
- **Held-out means not tuned against by a live run.** The author has seen held-out totals.
- **Held-out questions leak.** 7 repeat dev questions on other fixtures, and one is inside a
  tool description. Reported, not fixed: M3-B0.2.
- **The dataset is hand-written over synthetic fixtures.** Rates do not predict real use.
- **Three held-out answer cases cannot be reached by any model.** The router stops them.
  v2 removes them from the quality denominators, and they stay in the totals.

## Deterministic comparison floor

From `baseline-report.json`, scorer v1. **No model was called.**

| Metric | `route_following`, dev (50) | `route_following`, held-out (38) |
|---|---:|---:|
| Outcome accuracy (v1) | 42/50 | 31/38 |
| Task success (v2) | 40/50 | 31/38 |
| Claim selection | 24/29 | 20/23 |
| Tool selection | 24/26 | 20/20 |
| False answers | 5 | 4 |
| False abstentions | 3 | 3 |
| Fabricated answers | 0 | 0 |

The two dev cases v2 no longer credits: `adv-inside-the-vocabulary` answers with a valid
citation to an off-topic claim, and `multi-model-and-status` leaves out `run_status`.

## Proposed stages and caps

**Not approved, and not run.** Each stage needs its own approval record. Amounts are upper
bounds at the prices verified on 2026-11-02, for `claude-haiku-5-5`.

| Stage | Split | Questions | Request cap | Cost cap | Worst case |
|---|---|---:|---:|---:|---:|
| A. Free preflight | dev | 0 billed | 0 | Free | Free |
| B. Paid protocol probe | dev | About 2 | 5 | $0.02 | About $0.01 |
| C. Paid development evaluation | dev | 31 | 155 | $0.60 | $0.52 |
| D. Paid held-out decision | held-out | 24 | 120 | $0.45 | $0.40 |

Worst case for all billed stages is under $1.00. Set a monthly spend limit in the Console
as well. M3-B0.2 changes the question counts, so these caps must be recalculated after it.

Instructions: [M3-B runbook](COPILOT_AGENT_REAL_MODEL.md#m3-b-runbook).
