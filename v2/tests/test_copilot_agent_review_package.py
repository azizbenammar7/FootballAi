"""The human label-review package for dataset v3 stays true to the dataset, and honest.

The worksheet is where a person records label decisions. These tests keep its label
columns identical to the case files, so a reviewer never approves a stale copy, and keep
anyone, including an AI agent, from recording a decision the protocol does not allow.
They never require a decision to exist: 0 of 142 reviewed is a valid state.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
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
    with (REVIEW / "worksheet.csv").open(newline="", encoding="utf-8") as handle:
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
        decision = row["HUMAN_DECISION"].strip()
        if not decision:
            continue
        assert decision in builder.DECISIONS, f"{row['case_id']}: {decision!r}"
        assert row["REVIEWER"].strip() and row["REVIEW_DATE"].strip(), f"{row['case_id']}: who and when"
        if decision != "ACCEPT":
            assert row["HUMAN_RATIONALE"].strip(), f"{row['case_id']}: a change needs a reason"


def test_no_ai_agent_is_recorded_as_a_reviewer(committed_rows):
    """An AI check is pre-review triage. It must never appear as the second person."""
    machine = re.compile(
        r"claude|anthropic|chatgpt|\bgpt|openai|gemini|copilot|\bllm\b|\bbot\b|\bai (agent|triage|model|review)",
        re.IGNORECASE,
    )
    for row in committed_rows:
        assert not machine.search(row["REVIEWER"]), f"{row['case_id']}: {row['REVIEWER']!r}"


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
