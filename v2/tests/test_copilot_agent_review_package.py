"""The human label-review package for dataset v3 stays true to the dataset, and honest.

The worksheet is where a person records label decisions. These tests keep its label
columns identical to the case files, so a reviewer never approves a stale copy, and keep
anyone, including an AI agent, from recording a decision the protocol does not allow.
They never require a decision to exist: 0 of 142 reviewed is a valid state.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest


REVIEW = Path(__file__).resolve().parents[2] / "evaluation" / "copilot" / "agent-v3" / "review"


def _builder():
    spec = importlib.util.spec_from_file_location("build_review_package", REVIEW / "build_review_package.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _builder()


@pytest.fixture(scope="module")
def sheets(builder):
    return builder._sheets()


@pytest.fixture(scope="module")
def committed_rows():
    # utf-8-sig and newline="": a sheet saved by a spreadsheet keeps its meaning.
    with (REVIEW / "worksheet.csv").open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _strip_human(text: str, builder) -> str:
    pattern = re.escape(builder.HUMAN_START) + r".*?" + re.escape(builder.HUMAN_END)
    return re.sub(pattern, "", text, flags=re.DOTALL)


def test_the_worksheet_covers_every_label_exactly_once(sheets, committed_rows):
    expected = {case["case_id"] for sheet in sheets.values() for case in sheet["cases"]}
    ids = [row["case_id"] for row in committed_rows]
    assert len(ids) == len(set(ids)) == len(expected) == 142
    assert set(ids) == expected


def test_the_worksheet_label_columns_match_the_case_files(builder, sheets, committed_rows):
    fresh = {row["case_id"]: row for row in builder.build_rows(sheets)}
    fixed = (*builder.LABEL_COLUMNS, "priority", "fixture_claims", "ai_triage", "dossier")
    for row in committed_rows:
        assert {column: row[column] for column in fixed} == {column: fresh[row["case_id"]][column] for column in fixed}, (
            f"{row['case_id']}: rebuild the package with build_review_package.py"
        )


def test_the_worksheet_columns_are_the_documented_ones(builder, committed_rows):
    assert tuple(committed_rows[0].keys()) == builder.COLUMNS


def test_a_recorded_decision_is_one_a_person_may_make(builder, committed_rows):
    for row in committed_rows:
        assert builder.review_row_problems(row) == [], row["case_id"]


def test_the_dossier_matches_the_case_files_outside_the_human_fields(builder, sheets):
    committed = (REVIEW / "DISPUTED_CASES.md").read_text(encoding="utf-8")
    assert _strip_human(committed, builder) == _strip_human(builder.render_dossier(sheets), builder)


def test_every_disputed_case_has_empty_or_human_fields_and_is_in_the_dataset(builder, sheets):
    ids = {case["case_id"] for sheet in sheets.values() for case in sheet["cases"]}
    assert set(builder.DISPUTES) <= ids
    assert set(builder.LINKED) <= ids
    committed = (REVIEW / "DISPUTED_CASES.md").read_text(encoding="utf-8")
    assert committed.count(builder.HUMAN_START) == committed.count(builder.HUMAN_END) == len(builder.DISPUTES)
    for dispute in builder.DISPUTES.values():
        assert dispute.alternatives[0].startswith("Keep the current label"), "the current label is always option 1"


def test_the_claim_catalog_is_exactly_what_the_server_builds(builder, sheets):
    assert (REVIEW / "FIXTURE_CLAIMS.md").read_text(encoding="utf-8") == builder.render_claims(sheets)


@pytest.fixture
def scratch_package(builder, sheets, tmp_path, monkeypatch):
    """The builder pointed at a temporary copy, so no test can touch the committed files."""
    paths = {name: tmp_path / name for name in ("worksheet.csv", "DISPUTED_CASES.md", "FIXTURE_CLAIMS.md")}
    paths["worksheet.csv"].write_text(builder.render_csv(builder.build_rows(sheets)), encoding="utf-8")
    paths["DISPUTED_CASES.md"].write_text(builder.render_dossier(sheets), encoding="utf-8")
    for attribute, name in (("WORKSHEET", "worksheet.csv"), ("DOSSIER", "DISPUTED_CASES.md"), ("CLAIMS", "FIXTURE_CLAIMS.md")):
        monkeypatch.setattr(builder, attribute, paths[name])
    return paths


def test_the_builder_never_overwrites_a_worksheet_decision(builder, sheets, scratch_package):
    rows = builder.build_rows(sheets)
    rows[0]["HUMAN_DECISION"] = "ACCEPT"
    scratch_package["worksheet.csv"].write_text(builder.render_csv(rows), encoding="utf-8")
    assert builder.main() == 2
    assert "ACCEPT" in scratch_package["worksheet.csv"].read_text(encoding="utf-8")
    assert not scratch_package["FIXTURE_CLAIMS.md"].exists()


def test_the_builder_never_overwrites_a_dossier_decision(builder, scratch_package):
    dossier = scratch_package["DISPUTED_CASES.md"]
    filled = dossier.read_text(encoding="utf-8").replace(
        builder.HUMAN_TEMPLATE[1], builder.HUMAN_TEMPLATE[1] + " The claim does not answer it.", 1
    )
    dossier.write_text(filled, encoding="utf-8")
    assert builder.main() == 2
    assert dossier.read_text(encoding="utf-8") == filled


def test_a_dossier_decision_matches_the_worksheet(builder, committed_rows):
    """Two records of one decision must never disagree. The worksheet is the record."""
    decisions = {row["case_id"]: row["HUMAN_DECISION"].strip() for row in committed_rows}
    blocks = builder.human_blocks((REVIEW / "DISPUTED_CASES.md").read_text(encoding="utf-8"))
    assert set(blocks) == set(builder.DISPUTES)
    for case_id, block in blocks.items():
        written = builder.human_field(block, builder.HUMAN_TEMPLATE[0])
        if written:
            assert written in builder.DECISIONS, f"{case_id}: {written!r}"
            assert written == decisions[case_id], f"{case_id}: dossier says {written}, worksheet says {decisions[case_id]!r}"


# --- M3-B0.3 hardening: every label field is in front of the reviewer ---------------------

# The case files name two fields differently from the worksheet columns.
RENAMED = {"outcome": "expected_outcome", "terminations": "acceptable_terminations"}
OUTPUT_NATURES = {"synthetic", "verified_real", "unverified_or_partial"}
# The 24 dossiers as committed when the package was first built. A dossier may only leave
# this list when its case is settled in LABEL_REVISIONS.md, never as a side effect.
DOSSIER_CASES = {
    "h-gap-legacy-diagnostics", "h-zero-model-accuracy", "h-gap-trustworthy", "h-run-any-failed",
    "h-run-all-successful", "h-adv-say-accurate", "h-multi-happened-partial", "h-bad-corrupt-completed",
    "h-multi-succeed-files", "h-adv-say-verified-real", "h-ans-tracklet-length", "h-multi-assigned-ground-truth",
    "h-adv-artifact-ground-truth", "h-adv-artifact-interpretation", "h-zero-model-hash-absent",
    "h-adv-tell-genuine-model", "h-gap-tracking-metrics", "h-gap-artifact-missing", "h-zero-video-verified",
    "h-zero-artifact-version", "h-g4-weights-hash", "h-g4-active-median", "h-multi-model-and-commit",
    "h-fail-operator-budget",
}


@pytest.fixture(scope="module")
def raw_cases():
    """The canonical case files read as plain JSON: no loader and no builder code in between."""
    return {
        split: json.loads((REVIEW.parent / f"cases-{split}.json").read_text(encoding="utf-8"))["cases"]
        for split in ("dev", "heldout")
    }


def test_the_splits_have_the_documented_sizes(raw_cases, committed_rows):
    assert {split: len(cases) for split, cases in raw_cases.items()} == {"dev": 68, "heldout": 74}
    assert Counter(row["split"] for row in committed_rows) == {"dev": 68, "heldout": 74}
    for split, cases in raw_cases.items():
        assert {case["case_id"] for case in cases} == {row["case_id"] for row in committed_rows if row["split"] == split}


def test_every_expected_label_field_has_a_worksheet_column(builder, raw_cases):
    """A reviewer cannot accept a label field the worksheet never shows them."""
    fields = {key for cases in raw_cases.values() for case in cases for key in case["expected"]}
    assert {"output_nature", "forbidden_text"} <= fields
    assert {RENAMED.get(name, name) for name in fields} <= set(builder.LABEL_COLUMNS)


def test_output_nature_and_forbidden_text_match_the_case_files_exactly(builder, raw_cases, committed_rows):
    rows = {row["case_id"]: row for row in committed_rows}
    for split, cases in raw_cases.items():
        for case in cases:
            row, expected = rows[case["case_id"]], case["expected"]
            assert row["output_nature"] == expected["output_nature"] and row["output_nature"] in OUTPUT_NATURES
            # An empty list is written as "[]", never as an empty cell, so "no forbidden text"
            # cannot be confused with "not shown".
            assert row["forbidden_text"] == json.dumps(expected["forbidden_text"], ensure_ascii=False)
            assert builder.decode_text_list(row["forbidden_text"]) == expected["forbidden_text"]


@pytest.mark.parametrize(
    "texts",
    [[], ["a, b"], ['say "verified"'], ["line\nbreak"], ["x; y", "p|q"], ["é ü ✓"], [" padded "], ["[not json]"]],
)
def test_forbidden_text_survives_the_csv_round_trip(builder, sheets, texts):
    edited = copy.deepcopy(sheets)
    edited["heldout"]["cases"][0]["forbidden_text"] = list(texts)
    rows = builder.build_rows(edited)
    case_id = edited["heldout"]["cases"][0]["case_id"]
    back = {row["case_id"]: row for row in csv.DictReader(io.StringIO(builder.render_csv(rows)))}
    assert builder.decode_text_list(back[case_id]["forbidden_text"]) == texts


@pytest.mark.parametrize("field, value", [("output_nature", None), ("output_nature", ""), ("forbidden_text", None)])
def test_a_missing_label_value_stops_the_build(builder, sheets, field, value):
    """Never write a blank cell that a reviewer would read as a real, empty label."""
    edited = copy.deepcopy(sheets)
    edited["dev"]["cases"][0][field] = value
    with pytest.raises(ValueError):
        builder.build_rows(edited)


@pytest.mark.parametrize("cell", ["", "player 7", '"player 7"', "[1]", '[""]', '["a", "a"]', "{}", "null"])
def test_a_malformed_text_list_is_refused(builder, cell):
    with pytest.raises(ValueError):
        builder.decode_text_list(cell)


def test_a_fresh_build_never_holds_a_human_value(builder, sheets):
    assert {"HUMAN_OUTPUT_NATURE_CORRECTION", "HUMAN_FORBIDDEN_TEXT_CORRECTION"} <= set(builder.HUMAN_COLUMNS)
    assert {"HUMAN_OUTPUT_NATURE_RATIONALE", "HUMAN_FORBIDDEN_TEXT_RATIONALE"} <= set(builder.HUMAN_COLUMNS)
    for row in builder.build_rows(sheets):
        assert all(row[column] == "" for column in builder.HUMAN_COLUMNS), row["case_id"]


def test_the_committed_worksheet_is_a_fresh_build_plus_human_cells(builder, sheets, committed_rows):
    """Deterministic: the same dataset gives the same bytes. Every cell is compared; human
    cells, row order, and a spreadsheet's line endings or byte-order mark are allowed to differ."""
    fresh = {row["case_id"]: row for row in builder.build_rows(sheets)}
    for row in committed_rows:
        fresh[row["case_id"]].update({column: row[column] for column in builder.HUMAN_COLUMNS})
        assert row == fresh[row["case_id"]], f"{row['case_id']}: rebuild the package with build_review_package.py"
    if not any(row[column] for row in committed_rows for column in builder.HUMAN_COLUMNS):
        assert builder.render_csv(builder.build_rows(sheets)) == (REVIEW / "worksheet.csv").read_text(encoding="utf-8")
    assert builder.render_csv(builder.build_rows(sheets)) == builder.render_csv(builder.build_rows(builder._sheets()))
    assert builder.render_dossier(sheets) == builder.render_dossier(builder._sheets())


def test_every_dossier_is_kept_and_shows_both_fields(builder, sheets):
    assert set(builder.DISPUTES) == DOSSIER_CASES
    kinds = Counter(
        "outcome" if d.kind == "disputed: outcome" else "claims" if d.kind.startswith("disputed")
        else "correction" if d.kind.startswith("author") else "close"
        for d in builder.DISPUTES.values()
    )
    assert kinds == {"outcome": 5, "claims": 9, "close": 8, "correction": 2}
    text = builder.render_dossier(sheets)
    cases = {case["case_id"]: case for sheet in sheets.values() for case in sheet["cases"]}
    for case_id in DOSSIER_CASES:
        section = text.split(f'<a id="{case_id}"></a>', 1)[1].split("\n---\n", 1)[0]
        case = cases[case_id]
        assert f"| Expected output nature | `{case['output_nature']}` |" in section
        listed = json.dumps(case["forbidden_text"], ensure_ascii=False)
        assert f"| Forbidden text (JSON list) | `{listed}` |" in section


def test_the_dossier_and_triage_counts_reconcile(builder, committed_rows):
    """The package records 38 triaged held-out labels. The triage's own headline was 21 agreed, 17 disputed."""
    triaged = [r for r in committed_rows if r["split"] == "heldout" and r["origin"] in ("v3_new", "v2_reworded")]
    assert len(triaged) == 38
    raised = {r["case_id"] for r in triaged if r["case_id"] in builder.DISPUTES}
    assert raised == DOSSIER_CASES
    assert sum(r["priority"] == "4" for r in triaged) == 38 - len(DOSSIER_CASES) == 14


# --- Field-specific corrections --------------------------------------------------------------


@pytest.fixture
def blank_row(builder, sheets):
    row = next(r for r in builder.build_rows(sheets) if r["forbidden_text"] != "[]")
    return dict(row)


def _with(row, **cells):
    return {**row, **cells}


def _other_nature(row):
    return sorted(OUTPUT_NATURES - {row["output_nature"]})[0]


def test_a_blank_row_has_no_field_problem(builder, blank_row):
    assert builder.field_correction_problems(blank_row) == []


@pytest.mark.parametrize("decision", ["CORRECT", "REJECT", "NEEDS DISCUSSION"])
def test_a_complete_field_correction_is_accepted(builder, blank_row, decision):
    row = _with(
        blank_row,
        HUMAN_DECISION=decision,
        HUMAN_OUTPUT_NATURE_CORRECTION=_other_nature(blank_row),
        HUMAN_OUTPUT_NATURE_RATIONALE="The fixture's diagnostics are not integrity-verified.",
        HUMAN_FORBIDDEN_TEXT_CORRECTION="[]",
        HUMAN_FORBIDDEN_TEXT_RATIONALE="None of these phrases can appear in a claim.",
    )
    assert builder.field_correction_problems(row) == []


@pytest.mark.parametrize("field", ["OUTPUT_NATURE", "FORBIDDEN_TEXT"])
@pytest.mark.parametrize(
    "decision, rationale, problem",
    [
        ("CORRECT", "", "rationale"),
        ("ACCEPT", "Because.", "decision"),
        ("", "Because.", "decision"),
    ],
)
def test_a_field_correction_needs_a_reason_and_a_non_accept_decision(builder, blank_row, field, decision, rationale, problem):
    correction = _other_nature(blank_row) if field == "OUTPUT_NATURE" else '["player 99"]'
    row = _with(
        blank_row,
        HUMAN_DECISION=decision,
        **{f"HUMAN_{field}_CORRECTION": correction, f"HUMAN_{field}_RATIONALE": rationale},
    )
    problems = builder.field_correction_problems(row)
    assert len(problems) == 1 and problem in problems[0], problems


@pytest.mark.parametrize("value", ["real", "VERIFIED_REAL", "verified real", "null"])
def test_an_output_nature_correction_names_a_real_value(builder, blank_row, value):
    row = _with(blank_row, HUMAN_DECISION="CORRECT", HUMAN_OUTPUT_NATURE_CORRECTION=value, HUMAN_OUTPUT_NATURE_RATIONALE="x")
    assert any("output_nature" in problem for problem in builder.field_correction_problems(row))


def test_a_correction_must_change_the_field(builder, blank_row):
    same = _with(
        blank_row,
        HUMAN_DECISION="CORRECT",
        HUMAN_OUTPUT_NATURE_CORRECTION=blank_row["output_nature"],
        HUMAN_OUTPUT_NATURE_RATIONALE="x",
        HUMAN_FORBIDDEN_TEXT_CORRECTION=json.dumps(json.loads(blank_row["forbidden_text"]), indent=2),
        HUMAN_FORBIDDEN_TEXT_RATIONALE="x",
    )
    problems = builder.field_correction_problems(same)
    assert len(problems) == 2 and all("same as the current label" in p for p in problems), problems


@pytest.mark.parametrize("cell", ["player 7", '[""]', '["a", "a"]', "[1]"])
def test_a_forbidden_text_correction_is_a_valid_json_list(builder, blank_row, cell):
    row = _with(blank_row, HUMAN_DECISION="CORRECT", HUMAN_FORBIDDEN_TEXT_CORRECTION=cell, HUMAN_FORBIDDEN_TEXT_RATIONALE="x")
    assert any("forbidden_text" in problem for problem in builder.field_correction_problems(row))


def test_every_committed_field_correction_is_well_formed(builder, committed_rows):
    for row in committed_rows:
        assert builder.field_correction_problems(row) == [], row["case_id"]


# M3-B0.4 regressions: exercise the checks the worksheet validation command runs.
@pytest.mark.parametrize("phrase", ["", " ", "   ", "\t", "\n"])
def test_f1_whitespace_only_forbidden_phrase_is_rejected(builder, blank_row, phrase):
    row = _with(blank_row, HUMAN_DECISION="CORRECT", HUMAN_FORBIDDEN_TEXT_CORRECTION=json.dumps([phrase]),
                HUMAN_FORBIDDEN_TEXT_RATIONALE="This phrase should be banned.", HUMAN_RATIONALE="Change the phrase list.",
                REVIEWER="Jane Doe", REVIEW_DATE="2026-11-05")
    with pytest.raises(AssertionError):
        test_a_recorded_decision_is_one_a_person_may_make(builder, [row])


@pytest.mark.parametrize("cell", ['["a", "a"]', "[1]", "not json", '[""]'])
def test_f1_other_invalid_forbidden_corrections_stay_rejected(builder, blank_row, cell):
    row = _with(blank_row, HUMAN_DECISION="CORRECT", HUMAN_FORBIDDEN_TEXT_CORRECTION=cell,
                HUMAN_FORBIDDEN_TEXT_RATIONALE="This phrase should be banned.", HUMAN_RATIONALE="Change the phrase list.",
                REVIEWER="Jane Doe", REVIEW_DATE="2026-11-05")
    with pytest.raises(AssertionError):
        test_a_recorded_decision_is_one_a_person_may_make(builder, [row])


@pytest.mark.parametrize("phrase", ["player 7", "10.5 km"])
def test_f1_spaces_inside_real_phrases_stay_valid(builder, phrase):
    assert builder.decode_text_list(json.dumps([phrase])) == [phrase]


@pytest.mark.parametrize("column,value", [
    ("HUMAN_CORRECTION", "Change the outcome"),
    ("HUMAN_OUTPUT_NATURE_CORRECTION", "verified_real"),
    ("HUMAN_FORBIDDEN_TEXT_CORRECTION", "[]"),
    ("HUMAN_CORRECTION", " "),
    ("HUMAN_OUTPUT_NATURE_CORRECTION", "\t"),
    ("HUMAN_FORBIDDEN_TEXT_CORRECTION", "   "),
    ("HUMAN_OUTPUT_NATURE_RATIONALE", "The provenance is different"),
    ("HUMAN_FORBIDDEN_TEXT_RATIONALE", "The phrase is too broad"),
])
def test_f2_accept_cannot_hold_a_correction_or_its_reason(builder, blank_row, column, value):
    row = _with(blank_row, HUMAN_DECISION="ACCEPT", REVIEWER="Jane Doe", REVIEW_DATE="2026-11-05", **{column: value})
    with pytest.raises(AssertionError):
        test_a_recorded_decision_is_one_a_person_may_make(builder, [row])


@pytest.mark.parametrize("name", ["Codex", "Codex review agent", "OpenAI Codex", "Claude", "ChatGPT"])
def test_f3_machine_names_are_rejected_by_review_workflow(builder, blank_row, name):
    row = _with(blank_row, HUMAN_DECISION="ACCEPT", REVIEWER=name, REVIEW_DATE="2026-11-05")
    with pytest.raises(AssertionError):
        test_a_recorded_decision_is_one_a_person_may_make(builder, [row])


@pytest.mark.parametrize("name", ["Aisha Ben Ali", "Ai Nakamura", "Jai Patel", "Claude Dupont", "Gemini Smith"])
def test_f3_real_names_with_ai_letters_are_allowed(builder, blank_row, name):
    row = _with(blank_row, HUMAN_DECISION="ACCEPT", REVIEWER=name, REVIEW_DATE="2026-11-05")
    test_a_recorded_decision_is_one_a_person_may_make(builder, [row])


@pytest.mark.parametrize("date", ["tomorrow", "2026/11/05", "2026-2-5", "2026-02-30", "2025-13-01", "2025-02-29", "", "   "])
def test_f4_invalid_dates_are_rejected_by_review_workflow(builder, blank_row, date):
    row = _with(blank_row, HUMAN_DECISION="ACCEPT", REVIEWER="Jane Doe", REVIEW_DATE=date)
    with pytest.raises(AssertionError):
        test_a_recorded_decision_is_one_a_person_may_make(builder, [row])


@pytest.mark.parametrize("date", ["2026-11-05", "2024-02-29"])
def test_f4_real_iso_calendar_dates_are_allowed(builder, blank_row, date):
    row = _with(blank_row, HUMAN_DECISION="ACCEPT", REVIEWER="Jane Doe", REVIEW_DATE=date)
    test_a_recorded_decision_is_one_a_person_may_make(builder, [row])


@pytest.mark.parametrize("decision", ["CORRECT", "REJECT", "NEEDS DISCUSSION"])
def test_non_accept_review_with_a_reason_and_correction_remains_valid(builder, blank_row, decision):
    row = _with(blank_row, HUMAN_DECISION=decision, HUMAN_RATIONALE="The label needs discussion.",
                HUMAN_CORRECTION="Change the label as described.", REVIEWER="Jane Doe", REVIEW_DATE="2026-11-05")
    test_a_recorded_decision_is_one_a_person_may_make(builder, [row])


def test_accept_with_no_correction_remains_valid(builder, blank_row):
    row = _with(blank_row, HUMAN_DECISION="ACCEPT", REVIEWER="Jane Doe", REVIEW_DATE="2026-11-05")
    test_a_recorded_decision_is_one_a_person_may_make(builder, [row])


# --- Overwrite protection ---------------------------------------------------------------------


def _write_rows(path, header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


@pytest.mark.parametrize("column", [
    "HUMAN_DECISION", "HUMAN_RATIONALE", "HUMAN_CORRECTION", "HUMAN_OUTPUT_NATURE_CORRECTION",
    "HUMAN_OUTPUT_NATURE_RATIONALE", "HUMAN_FORBIDDEN_TEXT_CORRECTION", "HUMAN_FORBIDDEN_TEXT_RATIONALE",
    "REVIEWER", "REVIEW_DATE",
])
def test_the_builder_never_overwrites_any_human_cell(builder, sheets, scratch_package, column):
    rows = builder.build_rows(sheets)
    rows[-1][column] = "x"
    before = builder.render_csv(rows)
    scratch_package["worksheet.csv"].write_text(before, encoding="utf-8")
    assert builder.main() == 2
    assert scratch_package["worksheet.csv"].read_text(encoding="utf-8") == before
    assert not scratch_package["FIXTURE_CLAIMS.md"].exists()


def test_the_builder_never_overwrites_a_column_it_does_not_own(builder, sheets, scratch_package):
    """A reviewer's own notes column, or a cell past the header, is human text too."""
    rows = builder.build_rows(sheets)
    header = [*builder.COLUMNS, "MY_NOTES"]
    _write_rows(scratch_package["worksheet.csv"], header, [[*r.values(), ""] for r in rows[:-1]] + [[*rows[-1].values(), "check"]])
    assert builder.main() == 2
    _write_rows(scratch_package["worksheet.csv"], builder.COLUMNS, [[*r.values()] for r in rows] + [[*rows[0].values(), "stray"]])
    assert builder.main() == 2


def test_the_builder_protects_a_decision_in_the_old_worksheet_layout(builder, sheets, scratch_package):
    """A worksheet made before the two fields existed still holds a person's work."""
    old = [c for c in builder.COLUMNS if c not in ("output_nature", "forbidden_text") and "_OUTPUT_NATURE_" not in c and "_FORBIDDEN_TEXT_" not in c]
    rows = builder.build_rows(sheets)
    rows[0]["HUMAN_DECISION"] = "ACCEPT"
    _write_rows(scratch_package["worksheet.csv"], old, [[r[c] for c in old] for r in rows])
    assert builder.main() == 2
    rows[0]["HUMAN_DECISION"] = ""
    _write_rows(scratch_package["worksheet.csv"], old, [[r[c] for c in old] for r in rows])
    assert builder.main() == 0, "a blank sheet in the old layout is safe to rebuild"
    assert tuple(next(csv.reader(io.StringIO(scratch_package["worksheet.csv"].read_text(encoding="utf-8"))))) == builder.COLUMNS


def test_a_blank_sheet_saved_by_a_spreadsheet_is_still_blank(builder, sheets, scratch_package):
    """A byte-order mark and CRLF line endings are not human text."""
    text = builder.render_csv(builder.build_rows(sheets)).replace("\n", "\r\n")
    scratch_package["worksheet.csv"].write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    assert builder.main() == 0


def test_a_damaged_label_cell_is_not_blamed_on_the_correction(builder, blank_row):
    row = _with(blank_row, forbidden_text="", HUMAN_DECISION="CORRECT",
                HUMAN_FORBIDDEN_TEXT_CORRECTION="[]", HUMAN_FORBIDDEN_TEXT_RATIONALE="x")
    assert builder.field_correction_problems(row) == ["forbidden_text: the label cell itself is damaged. Rebuild the worksheet"]
    old_layout = {k: v for k, v in row.items() if k != "forbidden_text"}
    assert "damaged" in builder.field_correction_problems(old_layout)[0]
