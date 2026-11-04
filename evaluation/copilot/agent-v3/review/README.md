# Label review package, `copilot-agent-eval/v3`

> **HUMAN REVIEW STATUS: 0 of 142 labels reviewed by a person.** Nothing in this folder is
> a review until a named person fills it in. An AI agent's check is pre-review triage only,
> and must never be written into the decision columns.

This folder is for the **second reviewer**: a person who did not write the labels. You do
not need to be a developer. You need a spreadsheet program and about three to five hours.

| File | What it is | You |
|---|---|---|
| `worksheet.csv` | Every label, all 142, one row each, with empty decision columns | **Fill this in.** It is the record of the review |
| `DISPUTED_CASES.md` | The 24 cases the AI triage raised, with their evidence and options | Read it for priority 1 to 3 rows |
| `FIXTURE_CLAIMS.md` | Every test run's evidence, in plain English, and which tool owns each sentence | Look things up here |
| `build_review_package.py` | Builds the three files from the dataset | Ignore it |

## What you are checking

FootballAI answers questions about a video-analysis run. It can only answer with fixed
sentences the server wrote from that run's evidence, called **claims**. It must refuse
(**abstain**) when no claim answers the question. Each test case says what the correct
behaviour is. That is the **label**. You are checking the labels, not the software.

For each row, ask:

1. **Question.** Would a real person ask this, in these words?
2. **Outcome.** Read the `fixture_claims` column. Does any sentence *directly* answer the
   question? If yes, the label should be `answer`. If no, it should be `abstain`, even when
   nearby sentences are true. A true sentence that answers a different question is a wrong answer.
3. **Required claims.** Are these the fewest sentences a correct answer needs? Is one missing?
4. **Permitted claims.** Is every listed sentence on topic? Is an on-topic sentence missing?
   Anything shown that is not permitted makes the answer off topic.
5. **Tools.** The five tools in `FIXTURE_CLAIMS.md` each own some claims. Do the required
   and forbidden tools fit the claims above?
6. **Rationale.** Does the author's reason justify the label without reading code?

The full rules are in [LABELLING.md](../LABELLING.md). Rules 6 to 8 are new in v3 and the
most open to dispute: a checksum belongs to what it hashes; an instruction ("say X") is a
question about its subject; and a question about "all runs" is answered for the one run
the system can see.

Rows whose `mode` is `replay_only` are scripted failures, such as a timeout. No model is
asked. Only check that the expected termination (for example `budget_exhausted`) fits the
case's description. Rows with `expected_outcome` `fail_closed` are of this kind.

## How to record a decision

Fill these columns. Leave every other column alone.

| Column | Write |
|---|---|
| `HUMAN_DECISION` | Exactly one of `ACCEPT`, `CORRECT`, `REJECT`, `NEEDS DISCUSSION` |
| `HUMAN_RATIONALE` | One sentence. Required for everything except `ACCEPT` |
| `HUMAN_CORRECTION` | For `CORRECT` or `REJECT`: what the label should say instead |
| `REVIEWER` | Your name. A person's name, never a tool's |
| `REVIEW_DATE` | `YYYY-MM-DD` |

| Decision | Means |
|---|---|
| `ACCEPT` | The label is right as written |
| `CORRECT` | The outcome is right, but a claim, tool, wording, or reason needs to change |
| `REJECT` | The outcome is wrong: it should answer where it abstains, or the reverse |
| `NEEDS DISCUSSION` | You cannot decide alone, or the rules do not settle it |

You decide nothing alone that changes a label. Every `CORRECT`, `REJECT`, and
`NEEDS DISCUSSION` is settled later by you and the author together, and recorded in
[LABEL_REVISIONS.md](../LABEL_REVISIONS.md). Do not edit the case files.

## Order of work

Sort or filter the sheet by `priority`. Rows are already in this order.

| Priority | Rows | What | Time |
|---:|---:|---|---|
| 1 | 5 | The AI triage disputes whether the system should answer at all | 10 min each |
| 2 | 9 | The AI triage disputes the claim or tool sets | 5 min each |
| 3 | 13 | Close calls, two author corrections to confirm, and three labels linked to a dispute | 3 min each |
| 4 | 14 | New or reworded held-out labels the AI triage agreed with | 2 min each |
| 5 | 34 | Held-out labels carried from v2, never triaged | 1 to 2 min each |
| 6 | 67 | Development labels, never triaged | 1 min each |

For priority 1 to 3, `DISPUTED_CASES.md` has a section per case. **Decide from the
question and the claims first.** Then read *Why it was challenged*, and change your mind
only if the argument convinces you. The AI triage is one opinion, not a reference answer.

The `ai_triage` column says what the triage thought. Hide it during your first pass if you
can: a reviewer who reads it first is no longer independent.

You will not see scores, baselines, or whether the software can reach a case. That is on
purpose: a label must come from the question and the evidence alone.

## When you are done

Return `worksheet.csv` to the author. Do not run anything. The author then:

1. Checks the file: `PYTHONPATH=v2/src python -m pytest v2/tests/test_copilot_agent_review_package.py`.
   It refuses unknown decisions, missing names or dates, missing reasons, and any reviewer
   name that looks like an AI tool.
2. Reports raw agreement and Cohen's kappa on the outcome, and agreement on claims.
3. Settles each non-`ACCEPT` row with you, in `LABEL_REVISIONS.md`, and only then edits the
   case files. A changed label changes the held-out hash, which the decision rule and every
   paid-run approval must then name.

Partial reviews are fine. The count of reviewed labels is the number of rows with a
decision, never more.

## Rebuilding

Only after a label changes:

```bash
PYTHONPATH=v2/src python evaluation/copilot/agent-v3/review/build_review_package.py
```

It refuses to overwrite a worksheet or dossier that holds any human text, so move filled ones aside
first, rebuild, and copy the decisions back by `case_id`.
