# Copilot claim-selection evaluation

Versioned, hand-labelled evaluation data for the "Explain This Run" copilot. Files
here are test data, not measurements of any match. Changing a case or fixture means
a new dataset version.

- `v1/cases.json`: 42 selection cases and 10 scripted invalid-output cases.
- `v1/fixtures.json`: run fixtures, copied from the approved copilot test fixtures.
- `v1/baseline-report.json`: deterministic baseline. No model was called.

See [`../../docs/v2/COPILOT_LLM_EVALUATION.md`](../../docs/v2/COPILOT_LLM_EVALUATION.md)
for commands, results, and limitations.
