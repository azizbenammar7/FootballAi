# Copilot claim-selection evaluation

Versioned, hand-labelled evaluation data for the FootballAI copilot. Files here are
test data, not measurements of any match. Changing a case or fixture means a new
dataset version, and published versions are immutable.

| Version | Routing | Status | Contents |
|---|---|---|---|
| `v1/` | `copilot.routing/v1`, frozen | Immutable. The accepted decision rule refers to it | 42 selection and 10 scripted cases, 9 fixtures |
| `v2/` | `copilot.routing/v2`, current | M2-A regression suite | 88 selection and 18 scripted cases, 15 fixtures |
| `agent-v1/` | `copilot.routing/v2`, current | M2-B agent suite, offline | 69 scripted tool-calling cases. Replays the `v2` fixtures, pinned by hash |
| `agent-v2/` | `copilot.routing/v2`, current | M3-A benchmark for a real model. **No model has run it** | 114 labelled cases, 68 development and 46 held-out. Replays the `v2` fixtures, pinned by hash |
| `agent-v3/` | `copilot.routing/v2`, current | M3-B0.2. Leakage fixed, coverage measured. **Not frozen, not reviewed, no model has run it** | 142 labelled cases, 68 development and 74 held-out. Replays the `v2` fixtures, pinned by hash |

Each version holds `cases.json`, `fixtures.json`, and a deterministic
`baseline-report.json`. No model was called for any report here.
`v2/v1-comparison.json` compares only identically labelled cases.

**Do not compare headline scores between versions.** v2 was written by the same
person, alongside the router it tests, so its scores are regression evidence and not
a benchmark.

`agent-v1/` scores scripts and rules through the real agent, so it is regression
evidence about safety machinery and says nothing about model quality.

See [`../../docs/v2/COPILOT_LLM_EVALUATION.md`](../../docs/v2/COPILOT_LLM_EVALUATION.md)
and [`../../docs/v2/COPILOT_AGENT.md`](../../docs/v2/COPILOT_AGENT.md) for commands,
results, and limitations.
