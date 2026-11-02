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
| `decision-rule.json` | **Proposed, not accepted** |
| `live-approval.template.json` | What a person fills in to approve one paid run |
| `baseline-report.json` | Reproducible replay report. A test compares it with the code |

Reproduce the report. This needs no SDK, key, or network:

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_agent_benchmark --dataset evaluation/copilot/agent-v2
```

Read next: [the guide](../../../docs/v2/COPILOT_AGENT_REAL_MODEL.md), and
[the rule and approval gate](../../../docs/v2/COPILOT_AGENT_DECISION_RULE.md).
