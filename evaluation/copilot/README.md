# Copilot claim-selection evaluation

Versioned, hand-labelled evaluation data for the FootballAI copilot. Files here are
test data, not measurements of any match. Changing a case or fixture means a new
dataset version, and published versions are immutable.

| Version | Routing | Status | Contents |
|---|---|---|---|
| `v1/` | `copilot.routing/v1`, frozen | Immutable. The accepted decision rule refers to it | 42 selection and 10 scripted cases, 9 fixtures |
| `v2/` | `copilot.routing/v2`, current | M2-A regression suite | 88 selection and 18 scripted cases, 15 fixtures |

Each version holds `cases.json`, `fixtures.json`, and a deterministic
`baseline-report.json`. No model was called for any report here.
`v2/v1-comparison.json` compares only identically labelled cases.

**Do not compare headline scores between versions.** v2 was written by the same
person, alongside the router it tests, so its scores are regression evidence and not
a benchmark.

See [`../../docs/v2/COPILOT_LLM_EVALUATION.md`](../../docs/v2/COPILOT_LLM_EVALUATION.md)
for commands, results, and limitations.
