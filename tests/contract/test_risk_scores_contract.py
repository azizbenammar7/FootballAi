"""Schema and invariant checks for V1 risk-score output."""

from __future__ import annotations

import json

from conftest import ROOT, assert_finite_json


ACCEPTED_FLAGS = {"LOW", "MEDIUM", "HIGH", "INSUFFICIENT"}
COMPONENTS = {"speed_decay", "sprint_drop", "distance_drop", "hsr_load"}


def assert_risk_scores_contract(risk_scores):
    assert isinstance(risk_scores, dict)
    for key, record in risk_scores.items():
        assert isinstance(key, str)
        assert isinstance(record["track_id"], int)
        assert str(record["track_id"]) == key
        assert record["risk_flag"] in ACCEPTED_FLAGS
        assert 0 <= record["coverage_frac"] <= 1
        assert isinstance(record["fatigue_indicators"], dict)
        assert isinstance(record["score_breakdown"], dict)

        if record["risk_flag"] == "INSUFFICIENT":
            assert record["risk_score"] is None
            assert isinstance(record["reason"], str) and record["reason"]
            assert record["score_breakdown"] == {}
        else:
            assert isinstance(record["risk_score"], int)
            assert 0 <= record["risk_score"] <= 100
            assert set(record["score_breakdown"]) == COMPONENTS
            assert all(0 <= value <= 25 for value in record["score_breakdown"].values())
    assert_finite_json(risk_scores)


def test_minimal_risk_scores_fixture_contract(minimal_risk_scores):
    assert_risk_scores_contract(minimal_risk_scores)


def test_committed_risk_scores_contract():
    scores = json.loads((ROOT / "data/processed/risk_scores.json").read_text())
    assert_risk_scores_contract(scores)
