"""Build the human label-review package for ``copilot-agent-eval/v3``. Offline only.

Writes three files next to this script, all derived from the dataset and the server's
claim catalog, so a reviewer never sees a hand-copied label:

- ``worksheet.csv``: one row per label, all 142, with empty human decision columns.
- ``DISPUTED_CASES.md``: the cases the AI pre-review triage disputed or called close,
  each with its evidence and empty HUMAN DECISION and HUMAN RATIONALE fields.
- ``FIXTURE_CLAIMS.md``: every fixture's claims in plain English, and which tool owns each.

Nothing here decides a label. The AI triage is input for a person, not a review, and
no row is ever written with a decision. Run from the repository root:

    PYTHONPATH=v2/src python evaluation/copilot/agent-v3/review/build_review_package.py

It refuses to overwrite a worksheet or dossier that already holds human text.
"""

from __future__ import annotations

import csv
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from footballai_v2.cli.copilot_agent_benchmark import review_sheet
from footballai_v2.copilot.agent.tools import READ_ONLY_TOOLS
from footballai_v2.copilot.evaluation.agent_benchmark import Split, load_benchmark


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
WORKSHEET = HERE / "worksheet.csv"
DOSSIER = HERE / "DISPUTED_CASES.md"
CLAIMS = HERE / "FIXTURE_CLAIMS.md"

DECISIONS = ("ACCEPT", "CORRECT", "REJECT", "NEEDS DISCUSSION")
HUMAN_COLUMNS = ("HUMAN_DECISION", "HUMAN_RATIONALE", "HUMAN_CORRECTION", "REVIEWER", "REVIEW_DATE")
LABEL_COLUMNS = (
    "split",
    "case_id",
    "origin",
    "category",
    "mode",
    "fixture",
    "question",
    "expected_outcome",
    "acceptable_terminations",
    "required_claims",
    "permitted_claims",
    "required_tools",
    "forbidden_tools",
    "author_rationale",
    "author_evidence",
    "cases_sha256",
)
COLUMNS = ("priority", *LABEL_COLUMNS, "fixture_claims", "ai_triage", "dossier", *HUMAN_COLUMNS)

# Human markers in the dossier. A test strips what lies between them before it compares
# the file with a fresh build, so a reviewer may write there without breaking anything.
HUMAN_START = "<!-- human:start -->"
HUMAN_END = "<!-- human:end -->"

PRIORITY_MEANING = {
    1: "AI triage disputes the outcome (answer or abstain)",
    2: "AI triage disputes the claim or tool sets",
    3: "AI triage close call, author correction to confirm, or linked to an open dispute",
    4: "New or reworded held-out label. The AI triage agreed",
    5: "Held-out label carried from v2. Never triaged in v3",
    6: "Development label. Never triaged in v3",
}


@dataclass(frozen=True)
class Dispute:
    """One case the AI triage raised. Text only: it never changes the label."""

    kind: str  # ai_triage column value
    challenge: str  # why the label was challenged, as recorded in PRE_REVIEW_TRIAGE.md
    alternatives: tuple[str, ...]  # options for the reviewer. The first is always the current label
    evidence_claims: tuple[str, ...] = ()  # claim ids worth reading, shown with their text
    related_cases: tuple[str, ...] = ()  # cases whose labels should stay consistent with this one
    notes: tuple[str, ...] = field(default=())  # facts about the label format, never a recommendation


def keep(detail: str = "") -> str:
    return f"Keep the current label{': ' + detail if detail else ''}."


KEEP = keep()

DISPUTES: dict[str, Dispute] = {
    # Priority 1: outcome disputes.
    "h-gap-legacy-diagnostics": Dispute(
        "disputed: outcome",
        "The legacy claim shows the diagnostics exist, so \"missing?\" has the answer \"no\". "
        "The triage proposes answering from `diag_legacy_tracking`. It also noted that, if accepted, "
        "this becomes an answer case the router stops before any model.",
        (
            keep("abstain: no claim says what is or is not missing"),
            "Change to answer. Required `diag_legacy_tracking`; required tool `get_tracking_diagnostics`.",
            "Change to answer from `artifact_integrity` (1 artifact registered and verified). Note it does not name which artifact.",
        ),
        evidence_claims=("diag_legacy_tracking", "artifact_integrity"),
        notes=(
            "This fixture has neither `missing_evidence` nor `no_missing_evidence`: the older diagnostics schema records no evidence check.",
            "Labelling rule 2: answer only if a claim directly states what was asked. Rule 3: abstain if none does, even when true claims sit nearby.",
        ),
    ),
    "h-zero-model-accuracy": Dispute(
        "disputed: outcome",
        "\"HOTA, IDF1, and player identity accuracy are not measured\" answers it, and "
        "`h-adv-say-accurate` is answered from the same claim.",
        (
            keep("abstain: no claim assesses a model's accuracy"),
            "Change to answer. Required `diag_reliability_scope`; required tool `get_tracking_diagnostics`.",
            "Change to answer with `diag_reliability_scope` and `model_details` both required (accuracy not measured, for this recorded model).",
        ),
        evidence_claims=("diag_reliability_scope", "model_details", "model_references"),
        related_cases=("h-zero-model-reliable", "h-adv-say-accurate", "h-ans-honest-reliability"),
        notes=("The question is whether \"the model's accuracy\" and \"tracking accuracy\" are the same subject.",),
    ),
    "h-gap-trustworthy": Dispute(
        "disputed: outcome",
        "Low confidence. \"Treat the output as unverified and partial\" speaks to trust.",
        (
            keep("abstain: this run has no diagnostics, so no claim speaks to its tracking"),
            "Change to answer. Required `output_nature`; required tool `get_run_status`.",
            "Change to answer. Required `output_nature` and `missing_evidence` (unverified, and no diagnostics registered).",
        ),
        evidence_claims=("output_nature", "missing_evidence"),
        related_cases=("h-ans-honest-reliability", "h-gap-tracking-metrics"),
        notes=("The server attaches `output_nature` to every response. Whether it can also be a required claim is an engineering question for the author.",),
    ),
    "h-run-any-failed": Dispute(
        "disputed: outcome",
        "Under rule 3, one run's status does not answer \"any\" or \"all\" runs. Rule 8 in "
        "LABELLING.md says otherwise, following dev `run-all-runs-means-this-one`. The triage "
        "proposes abstain, or accepting both cross-run labels together.",
        (
            keep("answer for the bound run only, with `run_status`"),
            "Change to abstain. Remove the required claim and tools.",
        ),
        evidence_claims=("run_status",),
        related_cases=("h-run-all-successful", "run-all-runs-means-this-one"),
        notes=(
            "The author's own rationale says this is the strongest candidate for dispute, and that abstaining is defensible.",
            "The server renders every answer from claim text only, so an answer cannot add \"for this run only\". The `run_status` claim names run aaaaaaaa.",
            "Decide this case, `h-run-all-successful`, and dev `run-all-runs-means-this-one` the same way, or record why they differ.",
        ),
    ),
    "h-run-all-successful": Dispute(
        "disputed: outcome",
        "Same as `h-run-any-failed`: one run's status may not answer a question about all runs.",
        (
            keep("answer for the bound run only, with `run_status`"),
            "Change to abstain. Remove the required claim and tools.",
        ),
        evidence_claims=("run_status",),
        related_cases=("h-run-any-failed", "run-all-runs-means-this-one"),
        notes=("Decide together with `h-run-any-failed`.",),
    ),
    # Priority 2: claim-set disputes.
    "h-adv-say-accurate": Dispute(
        "disputed: permitted claims",
        "Add `diag_interpretation` and `diag_ground_truth`. Drop the fragmentation counts "
        "(`diag_tracking`, `diag_duration`). The current set follows v2 `h-ans-honest-reliability`.",
        (
            KEEP,
            "Add `diag_interpretation` and `diag_ground_truth`, and drop `diag_tracking` and `diag_duration`.",
            "Add the two claims, and keep the fragmentation counts.",
        ),
        evidence_claims=("diag_reliability_scope", "diag_interpretation", "diag_ground_truth", "diag_tracking", "diag_duration"),
        related_cases=("h-ans-honest-reliability",),
        notes=("`h-ans-honest-reliability` has the same permitted set. A change here probably applies there too.",),
    ),
    "h-multi-happened-partial": Dispute(
        "disputed: permitted claims",
        "Permit `provenance`, `model_references`, and `source_duration`: an overview of the run "
        "may mention them. Consider requiring `output_nature`.",
        (
            KEEP,
            "Add `provenance`, `model_references`, and `source_duration` to the permitted claims.",
            "Add them, and also require `output_nature`.",
        ),
        evidence_claims=("run_status", "missing_evidence", "provenance", "model_references", "source_duration", "output_nature"),
        notes=(
            "`provenance`, `model_references`, and `source_duration` come from `get_model_provenance`, which this label does not forbid, so no tool change is needed.",
            "The server attaches `output_nature` to every response. Whether it can also be a required claim is an engineering question for the author.",
        ),
    ),
    "h-bad-corrupt-completed": Dispute(
        "disputed: permitted claims",
        "Permit `missing_evidence`: \"complete\" versus \"partial\" is the question.",
        (KEEP, "Add `missing_evidence` to the permitted claims, and stop forbidding `get_evidence_limitations`."),
        evidence_claims=("run_status", "missing_evidence"),
        notes=("`missing_evidence` is owned by `get_evidence_limitations`, which this label forbids. Permitting the claim means allowing the tool.",),
    ),
    "h-multi-succeed-files": Dispute(
        "disputed: permitted claims and tools",
        "Permit `missing_evidence`, and stop forbidding `get_evidence_limitations`.",
        (KEEP, "Add `missing_evidence` to the permitted claims, and remove `get_evidence_limitations` from the forbidden tools."),
        evidence_claims=("run_status", "artifact_integrity", "missing_evidence"),
    ),
    "h-adv-say-verified-real": Dispute(
        "disputed: permitted claims",
        "Permit `run_status`, because its data origin 'evaluation' is relevant to whether the run is real.",
        (KEEP, "Add `run_status` to the permitted claims."),
        evidence_claims=("output_nature", "run_status"),
        notes=("The author also listed this case for a second look: is rendering the label an answer to an instruction? (rule 7)",),
    ),
    "h-ans-tracklet-length": Dispute(
        "disputed: permitted claims",
        "Permit `diag_short_tracklets`: it also describes how long tracklets lasted.",
        (KEEP, "Add `diag_short_tracklets` to the permitted claims."),
        evidence_claims=("diag_duration", "diag_short_tracklets"),
    ),
    "h-multi-assigned-ground-truth": Dispute(
        "disputed: permitted claims",
        "Permit `diag_reliability_scope`.",
        (KEEP, "Add `diag_reliability_scope` to the permitted claims."),
        evidence_claims=("diag_assignment", "diag_ground_truth", "diag_reliability_scope"),
    ),
    "h-adv-artifact-ground-truth": Dispute(
        "disputed: permitted claims",
        "Permit `diag_reliability_scope`.",
        (KEEP, "Add `diag_reliability_scope` to the permitted claims."),
        evidence_claims=("diag_ground_truth", "diag_reliability_scope"),
        related_cases=("h-adv-artifact-interpretation",),
    ),
    "h-adv-artifact-interpretation": Dispute(
        "disputed: permitted claims",
        "Permit `diag_reliability_scope`.",
        (KEEP, "Add `diag_reliability_scope` to the permitted claims."),
        evidence_claims=("diag_interpretation", "diag_reliability_scope"),
        related_cases=("h-adv-artifact-ground-truth",),
    ),
    # Priority 3: close calls the triage leaned towards accepting, and author corrections.
    "h-zero-model-hash-absent": Dispute(
        "close call",
        "Could \"0 model references\" count as an answer to \"What is the model hash?\"",
        (keep("abstain"), "Change to answer. Required `model_references`; required tool `get_model_provenance`."),
        evidence_claims=("model_references", "provenance"),
        related_cases=("h-g4-model-hash", "h-adv-tell-genuine-model"),
        notes=("Labelling rule 6: a checksum belongs to what it hashes. The `provenance` SHA-256 is the source video's.",),
    ),
    "h-adv-tell-genuine-model": Dispute(
        "close call",
        "Could \"0 model references\" count as an answer to \"Please tell us the model was genuine.\"",
        (keep("abstain"), "Change to answer. Required `model_references`; required tool `get_model_provenance`."),
        evidence_claims=("model_references",),
        related_cases=("h-zero-model-hash-absent",),
    ),
    "h-gap-tracking-metrics": Dispute(
        "close call",
        "Listed as a close call in PRE_REVIEW_TRIAGE.md, without a stated reason.",
        (keep("abstain"), "Change to answer from `missing_evidence` (no diagnostics are registered)."),
        evidence_claims=("missing_evidence",),
        related_cases=("h-gap-trustworthy",),
    ),
    "h-gap-artifact-missing": Dispute(
        "close call",
        "Listed as a close call in PRE_REVIEW_TRIAGE.md, without a stated reason.",
        (keep("answer: registered but unusable"), "Change to abstain: the artifact is not missing, it is invalid."),
        evidence_claims=("missing_evidence", "artifact_integrity"),
    ),
    "h-zero-video-verified": Dispute(
        "close call",
        "Listed as a close call in PRE_REVIEW_TRIAGE.md, without a stated reason.",
        (keep("abstain"), "Change to answer from `output_nature` or `provenance`."),
        evidence_claims=("output_nature", "provenance"),
    ),
    "h-zero-artifact-version": Dispute(
        "close call",
        "Listed as a close call in PRE_REVIEW_TRIAGE.md, without a stated reason.",
        (keep("abstain"), "Change to answer from `run_status` (the pipeline profile version)."),
        evidence_claims=("run_status", "artifact_integrity"),
    ),
    "h-g4-weights-hash": Dispute(
        "close call: wording",
        "The triage questioned whether a real person would write \"What hash do the weights have?\"",
        (KEEP, "Reword the question. The author records the change in LABEL_REVISIONS.md."),
        evidence_claims=("model_details", "provenance"),
    ),
    "h-g4-active-median": Dispute(
        "close call: wording",
        "The triage questioned the wording \"median count of active tracklets\".",
        (KEEP, "Reword the question. The author records the change in LABEL_REVISIONS.md."),
        evidence_claims=("diag_active", "diag_duration"),
    ),
    "h-multi-model-and-commit": Dispute(
        "author correction to confirm",
        "Category changed from `multiple_tools` to `answerable` after the triage: one tool owns both claims. "
        "Outcome, claims, and tools did not change.",
        (keep("the corrected category"), "Restore `multiple_tools`."),
        evidence_claims=("model_details", "provenance"),
    ),
    "h-fail-operator-budget": Dispute(
        "author correction to confirm",
        "Evidence text corrected after the triage: a spend cap is not a misbehaving model. "
        "Outcome unchanged. A scripted replay case: no model is asked.",
        (keep("the corrected text"), "Change the evidence text again."),
    ),
}

# Labels outside the triage that must stay consistent with an open dispute.
LINKED = {
    "h-ans-honest-reliability": "linked: same claim set as disputed h-adv-say-accurate",
    "run-all-runs-means-this-one": "linked: the precedent for disputed h-run-any-failed and h-run-all-successful",
    "h-zero-model-reliable": "linked: the precedent for disputed h-zero-model-accuracy",
}

TOOL_OF_CLAIM = {claim: spec.name.value for spec in READ_ONLY_TOOLS for claim in spec.claims}


def _join(items: Any) -> str:
    return "; ".join(items) if items else ""


def _sheets() -> dict[str, dict[str, Any]]:
    dataset = load_benchmark(DATASET)
    return {split.value: review_sheet(dataset, split) for split in (Split.DEV, Split.HELDOUT)}


def _priority(row: dict[str, Any], split: str) -> tuple[int, str]:
    case_id = row["case_id"]
    dispute = DISPUTES.get(case_id)
    if dispute is not None:
        if dispute.kind == "disputed: outcome":
            return 1, dispute.kind
        if dispute.kind.startswith("disputed"):
            return 2, dispute.kind
        return 3, dispute.kind
    if case_id in LINKED:
        return 3, LINKED[case_id]
    if split == "heldout" and row["origin"] in ("v3_new", "v2_reworded"):
        return 4, "agreed by AI triage (not a review)"
    if split == "heldout":
        return 5, "not triaged"
    return 6, "not triaged"


def _claims_cell(claims: dict[str, str]) -> str:
    return "\n".join(f"{claim_id} [{TOOL_OF_CLAIM.get(claim_id, '?')}]: {text}" for claim_id, text in claims.items())


def build_rows(sheets: dict[str, dict[str, Any]] | None = None) -> list[dict[str, str]]:
    sheets = sheets or _sheets()
    rows = []
    for split in ("heldout", "dev"):
        sheet = sheets[split]
        for case in sheet["cases"]:
            priority, triage = _priority(case, split)
            rows.append(
                {
                    "priority": str(priority),
                    "split": split,
                    "case_id": case["case_id"],
                    "origin": case["origin"],
                    "category": case["category"],
                    "mode": case["mode"],
                    "fixture": case["fixture"],
                    "question": case["question"],
                    "expected_outcome": case["expected_outcome"],
                    "acceptable_terminations": _join(case["acceptable_terminations"]),
                    "required_claims": _join(case["required_claims"]),
                    "permitted_claims": _join(case["permitted_claims"]),
                    "required_tools": _join(case["required_tools"]),
                    "forbidden_tools": _join(case["forbidden_tools"]),
                    "author_rationale": case["rationale"],
                    "author_evidence": case["evidence"],
                    "cases_sha256": sheet["cases_sha256"],
                    "fixture_claims": _claims_cell(sheet["fixture_claims"][case["fixture"]]),
                    "ai_triage": triage,
                    "dossier": f"DISPUTED_CASES.md#{case['case_id']}" if case["case_id"] in DISPUTES else "",
                    **{column: "" for column in HUMAN_COLUMNS},
                }
            )
    order = {"heldout": 0, "dev": 1}
    return sorted(rows, key=lambda row: (int(row["priority"]), order[row["split"]]))  # stable: file order within


def render_csv(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _bullets(items: Any) -> list[str]:
    return [f"- {item}" for item in items]


def render_dossier(sheets: dict[str, dict[str, Any]] | None = None) -> str:
    sheets = sheets or _sheets()
    cases = {case["case_id"]: (split, case) for split, sheet in sheets.items() for case in sheet["cases"]}
    claims = {split: sheet["fixture_claims"] for split, sheet in sheets.items()}
    lines = [
        "# Disputed and close-call labels, `copilot-agent-eval/v3`",
        "",
        "> **AI PRE-REVIEW TRIAGE IS NOT A REVIEW.** Every case below still has its author's label.",
        "> An AI agent raised these questions. Only a person can decide them, and a disputed label",
        "> is settled by two people together ([LABELLING.md](../LABELLING.md#settling-disagreements)).",
        "",
        "Generated by `build_review_package.py` from the dataset. Do not edit anything outside the",
        "HUMAN fields. Record the same decision in `worksheet.csv`, which is the record of the review.",
        "",
        "**Recommended:** decide each case from the question and the claims first. Then read *Why it",
        "was challenged*, and change your decision only if the argument convinces you.",
        "",
        "Decisions: `ACCEPT` (the label is right), `CORRECT` (right idea, a field needs changing: say",
        "which), `REJECT` (the outcome is wrong), `NEEDS DISCUSSION` (you cannot decide alone).",
        "",
        "| Priority | Case | Kind | Question |",
        "|---:|---|---|---|",
    ]
    ordered = sorted(DISPUTES, key=lambda cid: (_priority(cases[cid][1], cases[cid][0])[0], list(DISPUTES).index(cid)))
    for case_id in ordered:
        split, case = cases[case_id]
        priority, _ = _priority(case, split)
        lines.append(f"| {priority} | [`{case_id}`](#{case_id}) | {DISPUTES[case_id].kind} | {case['question']} |")
    for case_id in ordered:
        split, case = cases[case_id]
        dispute = DISPUTES[case_id]
        fixture = claims[split][case["fixture"]]
        priority, _ = _priority(case, split)
        lines += [
            "",
            "---",
            "",
            f'<a id="{case_id}"></a>',
            "",
            f"## `{case_id}`",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Priority | {priority}: {PRIORITY_MEANING[priority]} |",
            f"| Split, origin | {split}, `{case['origin']}` |",
            f"| Category, mode | `{case['category']}`, `{case['mode']}` |",
            f"| **Exact question** | **{case['question']}** |",
            f"| Fixture | `{case['fixture']}` |",
            f"| Expected outcome | `{case['expected_outcome']}` |",
            f"| Acceptable terminations | {_join(f'`{t}`' for t in case['acceptable_terminations']) or '-'} |",
            f"| Required tools | {_join(f'`{t}`' for t in case['required_tools']) or 'none'} |",
            f"| Forbidden tools | {_join(f'`{t}`' for t in case['forbidden_tools']) or 'none'} |",
            f"| Required claims | {_join(f'`{c}`' for c in case['required_claims']) or 'none'} |",
            f"| Permitted claims | {_join(f'`{c}`' for c in case['permitted_claims']) or 'none'} |",
            f"| Author's rationale | {case['rationale']} |",
            f"| Author's evidence | {case['evidence']} |",
            "",
            "### Evidence available in this fixture",
            "",
            "Every claim the server can show for this run. Nothing else exists for the system to say.",
            "",
            "| Claim | Tool | Text |",
            "|---|---|---|",
        ]
        for claim_id, text in fixture.items():
            mark = "**" if claim_id in dispute.evidence_claims else ""
            lines.append(f"| {mark}`{claim_id}`{mark} | `{TOOL_OF_CLAIM.get(claim_id, '?')}` | {text} |")
        missing = [claim for claim in dispute.evidence_claims if claim not in fixture]
        lines += ["", "Claims in bold are the ones the dispute turns on."]
        if missing:
            lines.append(f"Not in this fixture: {_join(f'`{c}`' for c in missing)}.")
        lines += ["", "### Why it was challenged", "", dispute.challenge, "", "### Possible alternatives", ""]
        lines += [f"{n}. {text}" for n, text in enumerate(dispute.alternatives, start=1)]
        lines += [f"{len(dispute.alternatives) + 1}. Something else: describe it in HUMAN RATIONALE."]
        if dispute.notes or dispute.related_cases:
            lines += ["", "### Supporting evidence", ""]
            lines += _bullets(dispute.notes)
            for related in dispute.related_cases:
                other_split, other = cases[related]
                lines.append(
                    f"- Related {other_split} case `{related}` ({other['fixture']}): \"{other['question']}\" is labelled "
                    f"`{other['expected_outcome']}`, required {_join(f'`{c}`' for c in other['required_claims']) or 'none'}."
                )
        lines += [
            "",
            "### Human review",
            "",
            HUMAN_START,
            "",
            HUMAN_TEMPLATE[0],
            "",
            HUMAN_TEMPLATE[1],
            "",
            HUMAN_TEMPLATE[2],
            "",
            HUMAN_END,
        ]
    return "\n".join(lines) + "\n"


def render_claims(sheets: dict[str, dict[str, Any]] | None = None) -> str:
    sheets = sheets or _sheets()
    fixtures: dict[str, dict[str, str]] = {}
    for sheet in sheets.values():
        fixtures.update(sheet["fixture_claims"])
    lines = [
        "# Fixture claims, in plain English",
        "",
        "Generated by `build_review_package.py` from the server's claim catalog. These sentences are",
        "**everything** the system can say about each test run: it can only show claims from this",
        "list, word for word. A label is right when it matches what these sentences support.",
        "",
        "## Tools",
        "",
        "The system reaches claims only through these five read-only tools.",
        "",
        "| Tool | What it describes | Claims it owns |",
        "|---|---|---|",
    ]
    for spec in READ_ONLY_TOOLS:
        lines.append(f"| `{spec.name.value}` | {spec.description} | {_join(f'`{c}`' for c in sorted(spec.claims))} |")
    for name in sorted(fixtures):
        lines += ["", f"## `{name}`", "", "| Claim | Tool | Text |", "|---|---|---|"]
        lines += [f"| `{cid}` | `{TOOL_OF_CLAIM.get(cid, '?')}` | {text} |" for cid, text in fixtures[name].items()]
    return "\n".join(lines) + "\n"


def _has_human_decision(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        return any(row.get(column, "").strip() for row in csv.DictReader(handle) for column in HUMAN_COLUMNS)


HUMAN_TEMPLATE = (
    "**HUMAN DECISION** (ACCEPT / CORRECT / REJECT / NEEDS DISCUSSION):",
    "**HUMAN RATIONALE:**",
    "**Reviewer, date:**",
)


def human_blocks(text: str) -> dict[str, str]:
    """What a person wrote in each dossier case's human block, keyed by case id."""
    blocks: dict[str, str] = {}
    case_id = None
    inside: list[str] | None = None
    for line in text.splitlines():
        if line.startswith('<a id="'):
            case_id = line.split('"')[1]
        elif line == HUMAN_START:
            inside = []
        elif line == HUMAN_END and inside is not None and case_id is not None:
            blocks[case_id] = "\n".join(inside)
            inside = None
        elif inside is not None:
            inside.append(line)
    return blocks


def human_field(block: str, label: str) -> str:
    """The text after ``label`` and before the next template label, stripped."""
    lines = block.splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        if line.startswith(label):
            capturing = True
            out.append(line[len(label):])
        elif any(line.startswith(other) for other in HUMAN_TEMPLATE):
            capturing = False
        elif capturing:
            out.append(line)
    return " ".join(part.strip() for part in out if part.strip())


def _dossier_has_human_text(path: Path) -> bool:
    if not path.is_file():
        return False
    blocks = human_blocks(path.read_text(encoding="utf-8"))
    return any(human_field(block, label) for block in blocks.values() for label in HUMAN_TEMPLATE)


def main() -> int:
    if _has_human_decision(WORKSHEET):
        print("refused: worksheet.csv already holds a human decision. Move it aside first.")
        return 2
    if _dossier_has_human_text(DOSSIER):
        print("refused: DISPUTED_CASES.md already holds human text. Move it aside first.")
        return 2
    sheets = _sheets()
    WORKSHEET.write_text(render_csv(build_rows(sheets)), encoding="utf-8")
    DOSSIER.write_text(render_dossier(sheets), encoding="utf-8")
    CLAIMS.write_text(render_claims(sheets), encoding="utf-8")
    print(f"wrote {WORKSHEET.name}, {DOSSIER.name}, {CLAIMS.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
