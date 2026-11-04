# Agent benchmark, `copilot-agent-eval/v3`

M3-B0.2. Labelled questions for evaluating the read-only agent with any model or none.
It replays the immutable `v2` fixtures, pinned by hash, and is scored by
`agent-scorer/v2`. **No language model has run it.** Every number in
`reliability-report.json` comes from scripts run through the real agent.

> **INDEPENDENT HUMAN REVIEW PENDING.** One author wrote every label. A check by another
> AI agent is not a review. 0 of 142 labels have been reviewed by a second person.

| File | Holds |
|---|---|
| `cases-dev.json` | 68 development cases. v2 development, unchanged. May be read and tuned against |
| `cases-heldout.json` | 74 held-out cases, for the decision rule once the owner freezes it |
| `LABELLING.md` | The labelling rules, the reviewer's instructions, and how disputes are settled |
| `LABEL_REVISIONS.md` | Every label change, with its reason |
| `PRE_REVIEW_TRIAGE.md` | An AI agent's blind challenge of 38 labels: 17 disputes, none applied. Input for the human reviewer, **not** a review |
| `review/` | The human review package (M3-B0.3): a worksheet for all 142 labels, a dossier for the 24 disputed and close-call cases, and every fixture's claims in plain English. [Start here](review/README.md). 0 of 142 reviewed |
| `reliability-report.json` | Coverage, leakage audit, fourteen controls under rule v2 and the three D7 options, and the routing probes. A test regenerates it byte for byte |

There is **no decision rule in this directory** on purpose. v3 is not frozen: the owner
has not decided D7, and no second person has reviewed it. Until a rule is placed here,
the benchmark CLI refuses every replay or paid run on v3 with `dataset_not_frozen`.
Only `--show-fixture` and `--review-sheet` work.

## What changed from v2

| | v2 | v3 |
|---|---:|---:|
| Held-out cases | 46 | 74 |
| Held-out answer cases a model can reach | 20 | **36** |
| Held-out non-answer cases a model can reach | 4 | **12** |
| Held-out injections a model can reach | 0 | **7** |
| Held-out cross-run questions a model can reach | 0 | **2** |
| Held-out answer cases where an off-topic selection can be caught | 0 | **3** |
| Held-out questions leaking from development or the prompt | 10 | **0** |
| Criterion 4 opportunities | 0 | **0**, structurally. See D7 |
| Criterion D opportunities | 4 | **12** |

"Reach a model" is measured, not asserted: `route_following` is run through the real
router and evidence checks, and a case counts when a model would be asked.

## Commands

All offline. No SDK call, key, or network.

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_agent_reliability --dataset evaluation/copilot/agent-v3 --rules evaluation/copilot/agent-v2
```

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_agent_benchmark --dataset evaluation/copilot/agent-v3 --review-sheet heldout
```

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_agent_benchmark --dataset evaluation/copilot/agent-v3 --show-fixture malicious_artifact
```

Methodology, results, and limits: [COPILOT_AGENT_DATASET_V3.md](../../../docs/v2/COPILOT_AGENT_DATASET_V3.md).
