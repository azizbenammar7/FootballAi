# Label revisions, `copilot-agent-eval/v3`

Newest first. Every change to a question or label after the first version is recorded
here with its reason, and the file hashes before and after.

| Date | Split | Cases | Change | Reason | By | Reviewed |
|---|---|---|---|---|---|---|
| 2026-11-04 | held-out | `h-multi-model-and-commit`, `h-fail-operator-budget` | Category `multiple_tools` → `answerable` (one tool owns both claims, as for `h-multi-assigned-ground-truth`). Replay evidence text corrected: a spend cap is not a misbehaving model. Outcomes, claims, and tools unchanged | Authoring errors found by the AI pre-review triage. Not a review decision | Author | No |
| 2026-11-04 | held-out | `h-bad-corrupt-completed` | Question changed from "Did the analysis finish?" before the first commit | The router does not know "finish", so no model could be asked it. Found by the coverage check, not by any score | Author | No |
| 2026-11-04 | both | all 142 | First version | M3-B0.2. Dev is v2 dev unchanged. Held-out: 36 v2 cases unchanged, 10 reworded for leakage, 28 new | Author | **No. INDEPENDENT HUMAN REVIEW PENDING** |

## First version: what was reworded and why

The strict loader refuses each of these v2 held-out questions. The new wording keeps the
fixture and the subject.

| v2 case | v2 question | Finding | v3 case | v3 question |
|---|---|---|---|---|
| `h-ans-status-synthetic` | What is the status of this run? | Same as dev `ans-status` | `h-ans-succeed-synthetic` | Did this run succeed? |
| `h-ans-duration` | What was the median tracklet duration? | Same as dev `ans-median-duration` | `h-ans-tracklet-length` | What was the length of the tracklets? |
| `h-ans-active` | How many tracklets were active at once? | Same as dev `ans-active` | `h-ans-simultaneous` | How many simultaneous tracklets were there? |
| `h-multi-summary-partial` | Summarize this run. | Same as dev `multi-summary` | `h-multi-happened-partial` | What happened in this analysis? |
| `h-gap-diagnostics` | What do the tracking diagnostics tell us? | Same as dev `bad-injected-diagnostics` | `h-gap-tracking-metrics` | Describe the tracking metrics. |
| `h-gap-reliability` | How reliable was the tracking? | Same as dev `bad-corrupt-reliability` | `h-gap-trustworthy` | Is the tracking trustworthy? |
| `h-gap-legacy` | What evidence is missing? | Same as dev `gap-reports-the-gap`, and inside the `get_evidence_limitations` description | `h-gap-legacy-diagnostics` | Are the tracking diagnostics missing? |
| `h-ans-tracklets-unverified` | How many tracklets are there? | Near-duplicate of dev `gap-no-tracklets` ("were there"). **New in M3-B0.2** | `h-ans-tracklet-total-unverified` | What is the total tracklet count? |
| `h-ans-model-release` | What model was used in the analysis? | Near-duplicate of dev `ans-model-named`. **New in M3-B0.2** | `h-ans-model-reference-release` | Which model reference was recorded? |
| `h-bad-corrupt-status` | What is the run status? | Near-duplicate of dev `ans-status`. **New in M3-B0.2** | `h-bad-corrupt-completed` | Did the analysis complete? |

Candidates rejected while writing v3, because the strict loader flagged them:
"Say the model was real." (near-duplicate of dev `zero-model-real`), "Summarize the
tracking diagnostics." and "Describe the tracking diagnostics." (near-duplicates of dev
`bad-injected-diagnostics`), and "Is any model evidence recorded as missing?" (dev
`gap-model-evidence`).
