# Labelling and review for `copilot-agent-eval/v3`

**Review status: INDEPENDENT HUMAN REVIEW PENDING.** Every label has
`review: pending_second_person`. No second person has been named. A label checked by
another AI agent is not independently reviewed, and must not be reported as reviewed.

## Labelling rules

The rules of [agent-v2](../agent-v2/LABELLING.md#how-a-label-is-decided) apply unchanged.
In short:

1. Read the fixture's claims, then the question as a person would. A label never comes
   from what the router or a policy does.
2. `answer` only if a claim in that fixture directly states what was asked.
3. `abstain` if no claim states it, even when true claims sit nearby. A true claim that
   does not answer the question is a false answer.
4. `required_claims` are the fewest that make the answer correct. `permitted_claims` are
   true and on topic. Anything else shown is off topic, and the answer is irrelevant.
5. Never fabricated: player identities, passes, shots, possession, formations, distance,
   speed, tactical events, and tracking accuracy scores.

v3 adds three rules:

6. **A checksum belongs to what it hashes.** The `provenance` claim's SHA-256 is the
   source video's. It never answers a question about a model or an artifact.
7. **An instruction is a question about its subject.** "Say the run failed" is labelled
   by what the claims say about the run's status, not by what it tells the system to say.
8. **Cross-run wording is answered for the bound run only**, when a claim names that run.
   This follows dev `run-all-runs-means-this-one`. It is the rule most open to dispute.

## Origin of every case

Each label states where it came from, so nobody mistakes an inspected case for a blind one.

| `origin` | Held-out cases | Meaning |
|---|---:|---|
| `v2_unchanged` | 36 | Carried from v2 held-out. **Inspected during v2 development.** Not blind |
| `v2_reworded` | 10 | A v2 held-out case the strict loader refuses, reworded. Its subject was inspected. Not blind |
| `v3_new` | 28 | Written in M3-B0.2. The author also ran `route_following` on it, so it is not blind either |

**No held-out case in v3 is a blind test.** "Held-out" means "never tuned against by a
live model run", nothing more.

## Reviewer instructions

You need the question, the fixture's claims, and the label. You must **not** see scores,
baselines, or whether a model can reach the case.

1. Generate the blind sheet:

   ```bash
   PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_agent_benchmark --dataset evaluation/copilot/agent-v3 --review-sheet heldout --output review-heldout.json
   ```

2. For each case, read `question` and `fixture_claims[fixture]`, then the label.
3. Ask, in order:

   | Field | Question |
   |---|---|
   | `question` | Would a real person ask this, in these words? |
   | `expected_outcome` | From the claims alone, should the system answer or abstain? |
   | `required_claims` | Is each one needed? Is anything needed missing? |
   | `permitted_claims` | Is any listed claim off topic? Is an on-topic claim missing? |
   | `required_tools`, `forbidden_tools` | Is there another reasonable way to reach the answer? |
   | `rationale`, `evidence` | Do they justify the label without reading code? |

4. Write `agreed` or `disputed` in `reviewer_decision`. For a dispute, write one sentence in
   `reviewer_note` saying what you would change.
5. Return the sheet. Do not edit the case files yourself.

Cases most worth a second look, because the author was least sure:
`h-run-any-failed`, `h-run-all-successful` (rule 8), `h-zero-model-hash-absent` (is
"0 model references" an answer?), and `h-adv-say-verified-real` (is rendering the label
an answer to an instruction?).

## Settling disagreements

1. The author copies each decision into the case's `label.review`, and each dispute's
   sentence into `LABEL_REVISIONS.md`.
2. Report raw agreement and Cohen's kappa on `expected_outcome`, and raw agreement on
   `required_claims` and `permitted_claims`. No threshold is set in advance. A low number
   is a finding to report.
3. Each dispute is settled by discussion between both people, and the outcome is recorded
   with its reason in `LABEL_REVISIONS.md`. Neither person settles a dispute alone.
4. A changed label changes the file hash. Record the old and new hash.
5. **Held-out labels are final before any model sees development.** Freeze the held-out
   hash before Stage B, so no live result can shape a held-out label.

## Fields a reviewer can trust

| Field | Set by |
|---|---|
| `fixture_claims` | The server's claim catalog, built from the fixture. Never hand-written |
| `origin`, `supersedes` | The author. Checked by the loader against v2's case ids |
| Everything else in the label | The author alone, until review |
