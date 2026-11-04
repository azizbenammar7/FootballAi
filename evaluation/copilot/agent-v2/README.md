# Agent benchmark, `copilot-agent-eval/v2`

Labelled questions for evaluating the read-only agent, with any model or none. It replays
the immutable `v2` fixtures, pinned by hash.

**No language model has run this.** Everything in `baseline-report.json` comes from
scripts and a rule. Its verdict is `NOT_A_MODEL_EVALUATION`, and it always will be.

| File | Holds |
|---|---|
| `cases-dev.json` | 68 development cases |
| `cases-heldout.json` | 46 held-out cases, for the decision rule |
| `LABELLING.md` | The rules, and how a second person reviews them |
| `decision-rule.json` | Rule v1. **Proposed, superseded**, unchanged. Never applied to a paid run |
| `decision-rule-v2.json` | Rule v2. **Proposed, not accepted.** The rule a paid run applies |
| `live-approval.template.json` | What a person fills in to approve one paid stage. Schema v2 |
| `baseline-report.json` | Scorer v1 replay report. A test compares it with the code, byte for byte |
| `scorer-v2-report.json` | Scorer v2 beside v1, twelve controls under both rules, and a dataset audit |

Reproduce the report. This needs no SDK, key, or network:

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_agent_benchmark --dataset evaluation/copilot/agent-v2
```

Reproduce the scorer v2 report, also offline:

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_agent_reliability --dataset evaluation/copilot/agent-v2
```

**Known dataset problems**, reported and not fixed here: 7 held-out questions repeat dev
questions on other fixtures, one is inside a tool description, and no held-out
adversarial or cross-run question reaches a model. See the
[M3-B0.2 plan](../../../docs/v2/COPILOT_AGENT_DATASET_REMEDIATION.md).

Read next: [the guide](../../../docs/v2/COPILOT_AGENT_REAL_MODEL.md), and
[the rule and approval gate](../../../docs/v2/COPILOT_AGENT_DECISION_RULE.md).
