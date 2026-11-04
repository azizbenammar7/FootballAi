# ADR-008: A scorer and decision rule that can fail

**Status:** Proposed for M3-B0.1, pending owner review of the pull request
**Date:** 2026-11-04
**Deciders:** Aziz Ben Ammar

## Context

The M3-B0 review of the agent benchmark (ADR-007, PR #57) found that the proposed
decision rule could pass behaviour it should reject. M3-B0.1 reproduced each finding
offline, with scripts run through the real agent. No model was called.

- **A.** `route_following`, the deterministic baseline, passes every measurable roadmap
  criterion on the held-out split. So does a script that picks off-topic claims whenever
  it can. The router narrows what is offered, so on held-out answer cases there is no
  off-topic claim to pick.
- **B.** `new_false_answers` cannot fail. All 4 held-out non-answer questions that reach a
  model (`h-zero-model-reliable`, `h-zero-model-revision`, `h-zero-model-verified`,
  `h-zero-tracks-complete`) are already false answers for `route_following`. A count with
  no opportunities read as `pass`.
- **C.** Criterion 9 left out tool-budget, turn-budget, and deadline stops. A script that
  hits the tool budget on all 24 questions read 0% failures. It also counted a model's
  refusal or truncation as a provider failure.
- **D.** `outcome_correct` meant "terminated as expected". An answer to an answerable
  question counted as correct whatever it said. On dev, `adv-inside-the-vocabulary`
  (off-topic, validly cited) and `multi-model-and-status` (missing `run_status`) both
  counted as correct.
- **E.** The citation sample included a scripted replay (`h-arg-recovers`), so a perfect
  live run reached n = 21 against a minimum of 20. Claim selection's denominator held 3
  router-stopped cases no model can reach. A missing measurement read
  `INSUFFICIENT_SAMPLE`, as if more data could fix it.
- **F.** 7 held-out questions repeat dev questions word for word on other fixtures, and 4
  of them reach a model. `h-gap-legacy` sits inside the `get_evidence_limitations`
  description.
- **G.** No held-out adversarial, cross-run, unsupported, or missing-evidence question
  reaches a model. The router stops all 10.
- **H.** An approval named the dataset and rule hashes only. The scorer, prompt, tools,
  SDK, and commit could change under it.

## Decision

1. **Scorer v2 beside scorer v1.** `agent_scoring.py` reads the same runs and labels.
   v1 is untouched, so `baseline-report.json` still reproduces byte for byte.
2. **Nine separate judgements.** Factual validity, citation validity, relevance,
   completeness, abstention correctness, tool selection, task success, provider
   reliability, and agent execution reliability. A valid citation to an irrelevant fact
   is `relevant: false`, and never a task success.
3. **One class and one owner per termination.** Every `AgentStopReason` maps to one
   `TerminationClass` and one `Owner`: none, provider, agent, or operator. A test fails if
   a new stop has no class. Provider and agent rates share one denominator, so they add up
   and never overlap. Spend-cap stops are reported and excluded, and make the run
   incomplete.
4. **Counts state their opportunities.** A count is `{"cases": [...], "n": k}`. With
   `n = 0` the criterion is `not_evaluable`, with reason `no_opportunities`. Every
   criterion result carries a reason code.
5. **A new verdict, `NOT_EVALUABLE`.** Order: `FAIL`, `INCOMPLETE_RUN`, `NOT_EVALUABLE`,
   `INSUFFICIENT_SAMPLE`, `PASS`. A missing binding measurement is `NOT_EVALUABLE`.
6. **Rule v2 in its own file, proposed.** `decision-rule-v2.json` names its scorer and the
   rule it supersedes. `decision-rule.json` (v1) is unchanged and is never applied to a
   paid run. Nobody has accepted either.
7. **Approvals pin the code.** Approval schema v2 adds scorer version, instruction
   version, system-prompt and tool-definition hashes, SDK version, and source commit. The
   runner refuses if tracked files under `v2/src` or `evaluation/copilot` differ from that
   commit. v1 approval records no longer parse. None existed.
8. **One approval per paid stage.** Stages are `protocol_probe` (dev, at most 5 requests),
   `development` (dev), and `decision` (held-out, rule accepted). The CLI needs `--stage`,
   and it must match the record.
9. **Twelve controls test the rule.** A label oracle, the baseline, and ten negative
   controls run on held-out under both rules. The report is committed and a test
   regenerates it.

## Options considered

| Option | For | Against |
|---|---|---|
| Fix scorer v1 in place | One scorer | Rewrites historical numbers. The baseline would no longer reproduce |
| **Scorer v2 beside v1** (chosen) | History stays. Old and new can be compared on the same runs | Two scorers to keep until v1 is retired |
| Edit `decision-rule.json` in place | One rule file | Changes the hash every note refers to, and hides what v1 said |
| **New `decision-rule-v2.json`** (chosen) | v1 stays for comparison. The supersession is explicit | Two proposed files. Docs say only v2 can gate a paid run |
| Make criterion 4 non-binding, so v2 can pass today | A usable rule now | Picks the bar after seeing what the split can do. That is the owner's call |

## What this does not prove

- **No model was measured.** Every number comes from scripts and a rule.
- **Seven criteria fail for no control.** Fabrication, citation validity, output nature,
  cross-run access, and out-of-registry executions are server-side invariants. A model
  cannot break them, and mutation tests cover them instead. `no_new_false_answers` has no
  opportunities, and p95 is not measured by a script.
- **The dataset is unchanged.** Findings B, F, and G are only reported here. M3-B0.2
  remediates them, with a second person reviewing the labels.

## Consequences

- **Easier:** a reader can see why each criterion passed or did not, from its reason code.
- **Easier:** a paid report names everything that produced it.
- **Harder:** rule v2 cannot pass on the current held-out split. Even the label oracle gets
  `NOT_EVALUABLE`. This is deliberate: criterion 4 has nothing that could fail it.
- **Harder:** any commit to the scoring code needs a fresh approval record.
- **Accepted:** an adapter defect (`unknown`) still counts as a provider error. From
  outside, it cannot be told apart from one.
- **Accepted:** the turn budget cannot be reached under the default limits. Each turn
  before the last must call a tool, so four turns use the four tool calls, and the fifth
  stops on the tool budget first. Fault injection tightens the limits to reach it.

## Revisit when

- M3-B0.2 adds held-out cases where `route_following` abstains correctly and a model is
  asked. Criterion 4 can then be evaluated.
- The owner accepts rule v2, or rejects criterion 4.
- Scorer v1 has no remaining reader.

## Action items

1. [x] Scorer v2, termination classes, opportunity counts, `NOT_EVALUABLE`.
2. [x] Rule v2, proposed. Approval schema v2 and stage separation.
3. [x] Twelve controls, old-versus-new comparison, dataset audit, committed report.
4. [ ] Owner decides D7 to D10 in [the decision rule](COPILOT_AGENT_DECISION_RULE.md).
5. [ ] M3-B0.2: [dataset remediation](COPILOT_AGENT_DATASET_REMEDIATION.md). Dataset v3
   built offline ([results](COPILOT_AGENT_DATASET_V3.md)). Second-person review and D7 remain.
