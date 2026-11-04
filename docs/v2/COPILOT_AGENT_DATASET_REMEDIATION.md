# M3-B0.2 plan: agent dataset remediation

**Plan only. Nothing here has been done.** M3-B0.1 measured these problems and changed no
question or label. `copilot-agent-eval/v2` stays frozen, because its hashes are recorded in
notes, reports, and ADR-008.

| | |
|---|---|
| Owner | Aziz Ben Ammar |
| Second label reviewer | **Not yet named.** Needed before any held-out label is final |
| Paid calls | None. M3-B0.2 needs no model |
| Output | A new dataset version, `copilot-agent-eval/v3`, in `evaluation/copilot/agent-v3/` |

## What to fix

Each problem has evidence in `evaluation/copilot/agent-v2/scorer-v2-report.json`, section
`dataset_audit`, and a test in `v2/tests/test_copilot_agent_scoring_v2.py`.

| # | Problem | Evidence | Fix in v3 |
|---|---|---|---|
| F1 | Held-out questions repeat dev questions on other fixtures | 7 cases: `h-ans-active`, `h-ans-duration`, `h-ans-status-synthetic`, `h-gap-diagnostics`, `h-gap-legacy`, `h-gap-reliability`, `h-multi-summary-partial`. 4 reach a model | New wording for each. `load_benchmark` compares question text across splits, ignoring fixture, after case and closing punctuation are normalised |
| F2 | A held-out question is inside a tool description | `h-gap-legacy` ("What evidence is missing?") in `get_evidence_limitations` | New wording. Do **not** edit the tool description to suit a benchmark: it is production text. The loader refuses any held-out question found in the tool definitions or system instructions |
| G1 | No held-out adversarial question reaches a model | `h-adv-reveal-key` and `h-adv-roleplay` are router-stopped | Add at least 4 held-out injections that use closed-vocabulary words, like dev's `adv-inside-the-vocabulary` |
| G2 | No held-out cross-run question reaches a model | `h-run-previous` is router-stopped | Add at least 2, such as a status question naming another run id inside supported words |
| G3 | Artifact injection is dev-only | Only dev uses `malicious_artifact` | Add a second malicious fixture to `copilot-eval/v3` fixtures, held-out only. A new fixture set means a new pinned hash |
| G4 | Off-topic selection is barely testable on held-out answer cases | The router narrows offers, so `off_topic_only` finds an off-topic claim on 0 of 20 reachable held-out answer cases | Add answer cases whose offered claims include plausible wrong ones: broad questions such as summaries, with narrow `permitted_claims` |
| B1 | Criterion 4 cannot fail | `route_following` false-answers all 9 reachable non-answer questions, on both splits | **Not a dataset problem.** It selects everything it retrieves, and a reachable question always retrieves something. More cases cannot fix it. The owner decides D7 |
| S1 | Small denominators | 20 held-out answer cases reach a model. 20 of 20 has a Wilson lower bound of 84%, and 16 of 20 one of 58% | At least 30 reachable held-out answer cases, and at least 10 reachable non-answer cases, so addition D has a real denominator |
| L1 | One author wrote every label | 0 of 114 reviewed | Second-person review, below |

## Label review protocol

1. **The reviewer works blind.** They see the question, the fixture's claims
   (`--show-fixture`), and the label. They do not see the baseline report or any score.
2. **Each label gets one of** `agreed` or `disputed`, with a sentence if disputed. That
   goes into `label.review`, which the schema already has.
3. **Agreement is reported** as raw agreement and Cohen's kappa on `outcome`, and as raw
   agreement on `required_claims` and `permitted_claims`. No threshold is set in advance.
   A low number is a finding to report, not a reason to stop.
4. **Disputes are settled** by discussion and recorded in the case's `rationale`. Neither
   author can settle a dispute alone.
5. **Held-out labels are final before any model sees dev.** Freeze the v3 held-out hash
   before Stage B, so no live result can shape a held-out label.

## Order of work

1. Write the v3 cases from v2: new wording for F1 and F2, and the G1–G4 and S1 additions.
2. Extend `load_benchmark` for F1 and F2. Write a test for each, and check that it refuses
   the v2 dataset.
3. Run the replay and reliability reports on v3. Check that every category has live
   held-out cases that reach a model, and that `off_topic_only` finds an off-topic claim on
   some held-out answer cases.
4. Second-person review, then fixes.
5. Freeze: record the v3 hashes in the decision rule and the approval template.

## Done when

- [ ] `agent-v3` loads, and its committed replay and reliability reports reproduce.
- [ ] The dataset audit shows 0 overlaps, 0 leaks, and at least one reachable held-out
      case in every live category except `unsupported`, which the router stops by design.
- [ ] Every label is `agreed`, or `disputed` with a recorded resolution.
- [ ] The owner has decided D7. Until then the rule cannot pass on any split.

## Not in M3-B0.2

- Any change to the router, tools, or prompt to suit a benchmark.
- Any paid call.
- Accepting the decision rule. Only the owner can do that, after this work.
