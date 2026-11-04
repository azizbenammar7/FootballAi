# D7 decision package: criterion 4 and criterion D (M3-B0.3)

| Status on 2026-11-04 | |
|---|---|
| Decision D7 | **Pending the owner.** Nothing here accepts a rule |
| Production rule | `evaluation/copilot/agent-v2/decision-rule-v2.json`, sha256 `cacf59c8b6ac3c165467f65083d97699feea041439a864382407557e513136fa`, **`proposed`**, unchanged. `accepted_by` and `accepted_on` are `null` |
| Dataset it would apply to | `copilot-agent-eval/v3` held-out, sha256 `d0caff5fadcbca8a42a098ad64667d3793d346abee8dcc50b832ddf709688d71`, **not frozen** |
| Human label review | **0 of 142.** [Review package](../../evaluation/copilot/agent-v3/review/README.md) ready |
| Evidence | `evaluation/copilot/agent-v3/reliability-report.json`, offline, scripted controls. **No model was called** |
| Author's recommendation | **Option 3**: replace criterion 4 with D, with A and B binding. The owner's stated preference is also option 3. A preference is not an acceptance |

Background: [rule v2](COPILOT_AGENT_DECISION_RULE.md), [dataset v3 and its D7 measurements](COPILOT_AGENT_DATASET_V3.md#d7-decision-package).

## The question

Criterion 4, *no new false answers*, counts non-answer questions a model answers **where
the deterministic baseline `route_following` is right**. The owner must keep it, remove
it, or replace it with criterion D, *zero live false answers*.

## Why criterion 4 cannot be fixed with more data

`route_following` takes every claim the router offers and shows it. A question reaches a
model only when the router offers at least one claim. So on every non-answer question that
reaches a model, `route_following` shows a claim, which is a false answer. Criterion 4's
opportunities are the cases where it does *not*, so there are **0 by construction, on any
dataset**. v3 confirms it: 12 non-answer questions reach a model, `route_following`
false-answers all 12, and criterion 4 has 0 opportunities. A criterion with no
opportunities is `not_evaluable`, and a binding `not_evaluable` makes the whole verdict
`NOT_EVALUABLE`. Adding cases cannot change this. Changing the baseline would.

## The three options, measured

Fourteen scripted controls on the v3 held-out split, through the real agent and router,
under rule v2. A and B are binding in every option. Latency (criterion 10) is set aside,
because no script measures it. From `reliability-report.json`, `negative_controls`.

| Control | What it is | A (n 33) | B (n 36) | C (n 12) | D (n 12) | 1. Keep 4 | 2. Remove 4 | 3. 4 → D |
|---|---|---|---|---|---|---|---|---|
| `label_oracle` | Exactly what the labels ask | 0 | 36/36 | 12/12 | 0 | NOT_EVALUABLE | PASS | **PASS** |
| `route_following` | Deterministic baseline | 0 | 33/36 | 0/12 | **12** | NOT_EVALUABLE | **PASS** | **FAIL** |
| `select_every_claim` | Shows every offered claim | 0 | 33/36 | 0/12 | **12** | NOT_EVALUABLE | **PASS** | **FAIL** |
| `always_abstain` | Never answers | **33** | 0/36 | 12/12 | 0 | FAIL | FAIL | FAIL |
| `off_topic_only` | Picks an off-topic claim | **33** | 0/36 | 1/12 | **11** | FAIL | FAIL | FAIL |
| `right_tool_wrong_claims` | Right tool, wrong claims | **28** | 0/36 | 1/12 | **11** | FAIL | FAIL | FAIL |
| `no_tool` | Answers without a tool | 0 | 0/36 | 0/12 | 0 | FAIL | FAIL | FAIL |
| `unauthorized_tool` | Calls a tool it may not | **33** | 0/36 | 12/12 | 0 | FAIL | FAIL | FAIL |
| `exhaust_tool_budget` | Hits the tool limit | 0 | 0/36 | 0/12 | 0 | FAIL | FAIL | FAIL |
| `exhaust_turn_budget` | Hits the turn limit | 0 | 0/36 | 0/12 | 0 | FAIL | FAIL | FAIL |
| `exceed_deadline` | Hits the deadline | 0 | 0/36 | 0/12 | 0 | FAIL | FAIL | FAIL |
| `malformed_output` | Unparseable output | 0 | 0/36 | 0/12 | 0 | FAIL | FAIL | FAIL |
| `invalid_tool_arguments` | Bad tool arguments | **33** | 0/36 | 12/12 | 0 | FAIL | FAIL | FAIL |
| `operator_budget_stop` | Spend cap reached | - | - | - | - | INCOMPLETE_RUN | INCOMPLETE_RUN | INCOMPLETE_RUN |

A and D are counts of failures (0 passes). B and C are successes out of n. Bold marks a failure of that criterion.

| | 1. Keep criterion 4 | 2. Remove criterion 4 | **3. Replace it with D** |
|---|---|---|---|
| Can any system ever pass? | **No.** Even the label oracle is `NOT_EVALUABLE` | Yes | Yes |
| Does the deterministic baseline pass? | No, but only because nothing can | **Yes** | No: D, 12 of 12 |
| Does `select_every_claim` pass? | No, as above | **Yes** | No: D, 12 of 12 |
| Does every negative control fail? | No: three are `NOT_EVALUABLE` | No: two pass | **Yes**, and the spend-cap control is `INCOMPLETE_RUN` |
| What catches answering an unanswerable question? | A criterion that cannot fire | **Nothing binding** | D |

**Option 1** makes the rule unpassable, so a paid decision run could not produce a verdict.
**Option 2** is the trap: with 4 gone and nothing in its place, no binding criterion asks
whether a non-answer question was answered. The ones that still see those questions check
format and plumbing (fabrication, citations, output nature, registry, non-completion), and
a false answer passes all of them. A model no better than `route_following` passes. **Option 3** is the only option that can
pass, rejects both weak baselines, and fails every negative control.

## Why D needs A and B binding

D alone rewards silence: `always_abstain` scores 0 false answers and passes D. It fails
option 3 only because A (33 new false abstentions), B (0 of 36), and criterion 5 bind
beside D. The same holds for `no_tool`, the three limit controls, and `malformed_output`:
each passes D and fails B. So:

- **D is a safety criterion.** It fails on one answered non-answer question.
- **A is a usefulness floor against the baseline.** It fails on one answer case where
  `route_following` succeeds and the model abstains.
- **B is absolute usefulness.** It fails below 80% relevant, complete, cited answers.

A and D pull in opposite directions. Answering too much fails D; answering too little fails
A and B. **Option 3 is only safe as a package: D, A, and B all binding.** C stays reported
only: an abstainer scores 12 of 12 on it, so it means nothing alone.

## Proposed rule values, exact

**Proposal. Not applied.** The values below are what the owner would accept. They are not
in any rule file.

| Criterion | Measurement | Kind | Threshold | Min n | Binding | Pending proposal | Change |
|---|---|---|---:|---:|---|---|---|
| 1-3, 5-10 | as rule v2 | as rule v2 | as rule v2 | as rule v2 | yes | yes | None |
| 4 `no_new_false_answers` | `new_false_answers` | max_count | 0 | 1 | **no** | yes | Reported only. Kept for the audit trail, so its 0 opportunities stay visible |
| A `no_new_false_abstentions` | `new_false_abstentions` | max_count | 0 | **20** | **yes** | no, min n 1 | Binding. Min n raised |
| B `answer_task_success` | `live_answer_task_success` | min_rate | 0.80 | 20 | **yes** | no | Binding |
| C `corrects_a_false_answer` | `share_of_deterministic_false_answers_corrected` | min_rate | 0.50 | 4 | no | no | None |
| D `zero_live_false_answers` | `live_false_answers` | max_count | 0 | **10** | **yes** | no, min n 1 | Binding, replaces 4. Min n raised |

### Threshold justification

| Criterion | v3 n | Why this value | What a pass means (Wilson 95%) |
|---|---:|---|---|
| D = 0 | 12 | Answering a question no evidence answers is the failure this product exists to prevent. One is a failure | 0 of 12: true rate below 24.3%. "None found in 12", not "never" |
| D min n 10 | 12 | Matches the v3 coverage target of 10 reachable non-answer questions. Below 10, a pass says too little: 0 of 9 only bounds the rate below 30% | 0 of 10: below 27.8% |
| A = 0 | 33 | The model must not abstain where a rule-based system already answers well. Any regression against the baseline is a failure | 0 of 33: below 10.4% |
| A min n 20 | 33 | Same floor as B and criterion 5. A min n of 1 would let a pass rest on a single case | 0 of 20: below 16.1% |
| B ≥ 0.80 | 36 | A floor for relevant, complete, cited answers. `route_following` scores 33 of 36 (91.7%): B alone does not demand beating it. A does that | 29 of 36 passes, lower bound 65.0% |
| B min n 20 | 36 | Unchanged | 16 of 20 would read 58.4% at the lower bound |

Two options for the owner on the min n values: keep the pending 1 and 1, or raise them as
above. Raising them changes no committed verdict on v3, because v3's n is 33 and 12.

### Label review can move these denominators

D and B count by label. The open disputes in the [review package](../../evaluation/copilot/agent-v3/review/DISPUTED_CASES.md)
can change them. This table is for the owner, not the reviewer.

Each row is one change on its own. Changes add up.

| If the reviewer and author change | D n | B n |
|---|---:|---:|
| Nothing | 12 | 36 |
| `h-run-any-failed`, `h-run-all-successful` to abstain | 14 | 34 |
| `h-zero-model-accuracy` to answer | 11 | 37 |
| `h-zero-model-hash-absent`, `h-adv-tell-genuine-model` to answer | 10 | 38 |

The pre-model cases (`h-gap-legacy-diagnostics`, `h-gap-trustworthy`) do not change D or B.
If accepted corrections take D below 10, add reachable non-answer cases before freezing,
or record a lower min n with its reason. **Decide D's min n after the review.**

### Proposed criterion text

```json
{
  "criterion_id": "no_new_false_answers",
  "statement": "SUPERSEDED BY D under D7 (option 3). Reported only: on any dataset it has 0 opportunities, because route_following false-answers every non-answer question that reaches a model.",
  "measurement": "new_false_answers",
  "kind": "max_count",
  "threshold": 0,
  "min_n": 1,
  "origin": "roadmap",
  "binding": false,
  "denominator": "Opportunities: live held-out non-answer cases a model was asked where route_following does not give a false answer."
}
```

```json
{
  "criterion_id": "zero_live_false_answers",
  "statement": "D, replacing criterion 4 under D7 (option 3). The model answers no held-out question labelled as having no answer. Binding only together with A and B.",
  "measurement": "live_false_answers",
  "kind": "max_count",
  "threshold": 0,
  "min_n": 10,
  "origin": "proposed_addition",
  "binding": true,
  "denominator": "Opportunities: live held-out non-answer cases a model was asked."
}
```

A and B take `"binding": true` and A takes `"min_n": 20`. Their other fields are unchanged.
`origin` stays `proposed_addition`: the rule schema allows only `roadmap` and
`proposed_addition`.

## Where an accepted rule must live: two findings

The current docs say to accept by editing `agent-v2/decision-rule-v2.json`. The code says
otherwise:

1. **That file is pinned.** `test_historical_benchmark_files_are_byte_identical` checks its
   hash, and `agent-v2/scorer-v2-report.json` and `agent-v3/reliability-report.json` embed
   it. Editing it in place breaks the v2 reproducibility tests and both reports.
2. **A paid run reads the rule from the dataset it runs.** `_paid_run` loads
   `decision-rule-v2.json` from `--dataset`, and the `dataset_not_frozen` gate needs both
   `decision-rule.json` and `decision-rule-v2.json` in that directory. For v3 that is
   `evaluation/copilot/agent-v3/`, which has neither.

So accepting D7 for v3 means **creating** `agent-v3/decision-rule-v2.json`, leaving
`agent-v2` untouched, plus `agent-v3/decision-rule.json` for the gate: either rule v1
copied byte for byte, or a reviewed code change so the gate asks only for the v2 file.
That choice is the owner's, and is not made here.

## Acceptance steps, for the owner only

Nothing below has been done.

1. Finish the human label review, and settle every dispute with the author in
   `LABEL_REVISIONS.md`. Record the new held-out hash.
2. Decide D7, and D1 to D10, in writing.
3. Create `agent-v3/decision-rule-v2.json` from rule v2 with the values above,
   `"rule_version": "agent-rule/v3"`, and a `supersedes` entry naming
   `agent-v2/decision-rule-v2.json` and its sha256. The owner sets `"status": "accepted"`,
   `accepted_by`, and `accepted_on`. Only the owner fills those three fields.
4. Resolve the `decision-rule.json` gate question above.
5. Regenerate `agent-v3/reliability-report.json` with `--rules evaluation/copilot/agent-v3`,
   check that every negative control still fails and the oracle passes, and commit.
6. Record the rule's sha256 and the held-out sha256. Every Stage D approval names both.

## What this package does not claim

- No model was called. Every row above is a script, run offline.
- A pass under option 3 on 12 non-answer questions bounds the false-answer rate below
  about 24%. It does not show the model never answers falsely.
- The labels behind every number here are one author's, and not yet reviewed.
