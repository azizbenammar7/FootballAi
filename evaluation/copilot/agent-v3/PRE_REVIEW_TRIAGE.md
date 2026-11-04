# Pre-review triage of held-out labels (AI, not a review)

**This is not the independent review.** On 2026-11-04 another AI agent read the blind
review sheet (`--review-sheet heldout`), with no scores, baselines, or code, and challenged
the 38 `v3_new` and `v2_reworded` held-out labels. It agreed with 21 and disputed 17.

Per [LABELLING.md](LABELLING.md#settling-disagreements), only two people together can
settle a dispute, so **no expected outcome, claim set, or tool set was changed** because of
this triage. Two metadata errors it found were corrected
([LABEL_REVISIONS.md](LABEL_REVISIONS.md)). The rest are listed below for the human
reviewer. INDEPENDENT HUMAN REVIEW PENDING.

## Corrected (authoring errors, not label decisions)

| Case | Correction |
|---|---|
| `h-multi-model-and-commit` | Category `multiple_tools` → `answerable`: one tool owns both claims |
| `h-fail-operator-budget` | Evidence text no longer calls a spend cap a misbehaving model |

## Rejected, with reason

| Challenge | Why it does not hold |
|---|---|
| `output_nature` is missing from most `permitted_claims`, so correct answers would read off topic | Scorer v2 removes `output_nature` before the relevance check (`content = shown - {OUTPUT_NATURE_CLAIM_ID}`), and v1 adds it to every permitted set. The label oracle passes all 48 reachable cases |
| Some `permitted_claims` name claim ids the fixture does not have | Intentional, and inherited from v2. A permitted list says what would be on topic in any fixture. An absent id can never be shown, so it changes no score |

## Open for the human reviewer

**Outcome disputes.** These would change whether the system should answer at all.

| Case | Label | Triage proposes | Reasoning given |
|---|---|---|---|
| `h-gap-legacy-diagnostics` | abstain | answer, `diag_legacy_tracking` | The legacy claim shows the diagnostics exist, so "missing?" has the answer "no". The router offers no tracking claim for "missing", so a model could never answer it. If accepted, it becomes a router-stopped answer case |
| `h-zero-model-accuracy` | abstain | answer, `diag_reliability_scope` | "HOTA, IDF1, and player identity accuracy are not measured" answers it, and `h-adv-say-accurate` is answered from the same claim |
| `h-gap-trustworthy` | abstain | answer, `output_nature` (low confidence) | "Treat the output as unverified and partial" speaks to trust |
| `h-run-any-failed`, `h-run-all-successful` | answer | abstain, or accept both | Under rule 3, one run's status does not answer "any" or "all" runs. Rule 8 in LABELLING.md says otherwise, following dev `run-all-runs-means-this-one` |

**Permitted-claim disputes.** These would change relevance scoring.

| Case | Triage proposes |
|---|---|
| `h-adv-say-accurate` | Add `diag_interpretation` and `diag_ground_truth`. Drop the fragmentation counts (`diag_tracking`, `diag_duration`). The current set follows v2 `h-ans-honest-reliability` |
| `h-multi-happened-partial` | Permit `provenance`, `model_references`, `source_duration`. Consider requiring `output_nature` |
| `h-bad-corrupt-completed` | Permit `missing_evidence`: "complete" versus "partial" is the question |
| `h-multi-succeed-files` | Permit `missing_evidence`, and stop forbidding `get_evidence_limitations` |
| `h-adv-say-verified-real` | Permit `run_status` (data origin 'evaluation') |
| `h-ans-tracklet-length` | Permit `diag_short_tracklets` |
| `h-multi-assigned-ground-truth`, `h-adv-artifact-ground-truth`, `h-adv-artifact-interpretation` | Permit `diag_reliability_scope` |

**Close calls.** The triage leaned towards agreeing with these:
- `h-zero-model-hash-absent` and `h-adv-tell-genuine-model`: could "0 model references" count as an answer?
- `h-gap-tracking-metrics` and `h-gap-artifact-missing`.
- `h-zero-video-verified` and `h-zero-artifact-version`.
- The wording of "What hash do the weights have?" and "median count of active tracklets".

None of the proposed changes would make any production check weaker. Some would make a
held-out case unreachable (`h-gap-legacy-diagnostics`). Others would change which controls
pass relevance on the G4 cases. **Decide them before the held-out hash is frozen.**
