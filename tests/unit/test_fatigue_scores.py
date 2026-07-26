"""Characterize V1 fatigue component, gating, and risk-boundary behavior."""

from __future__ import annotations

import json

import pandas as pd
import pytest


def test_all_component_scores_have_zero_cases(fatigue_module):
    assert fatigue_module.speed_decay_score({"0": 1.0, "1": 1.0, "2": 1.0}) == (0.0, 0.0)
    assert fatigue_module.sprint_drop_score(0, 0) == (0.0, 0.0)
    assert fatigue_module.dist_drop_score(0.0, 0.0) == (0.0, 0.0)
    assert fatigue_module.hsr_load_score(100.0, 0.0) == 0.0


def test_speed_decay_normal_low_input_and_exact_boundary(fatigue_module):
    low_score, low_slope = fatigue_module.speed_decay_score({"0": 1.0, "1": 0.95, "2": 0.9})
    max_score, boundary_slope = fatigue_module.speed_decay_score({"0": 1.0, "1": 0.9, "2": 0.8})

    assert low_slope == pytest.approx(-0.05)
    assert low_score == pytest.approx(12.5)
    assert boundary_slope == pytest.approx(-0.1)
    assert max_score == pytest.approx(25.0)


def test_speed_decay_positive_slope_scores_zero(fatigue_module):
    score, slope = fatigue_module.speed_decay_score({"0": 1.0, "1": 1.1, "2": 1.2})
    assert slope > 0
    assert score == 0.0


@pytest.mark.parametrize("timeline", [{}, {"0": 1.0}, {"0": 1.0, "1": 0.5}])
def test_speed_decay_requires_at_least_three_blocks(fatigue_module, timeline):
    assert fatigue_module.speed_decay_score(timeline) == (0.0, 0.0)


def test_component_scores_clamp_at_25_points(fatigue_module):
    assert fatigue_module.speed_decay_score({"0": 1.0, "1": 0.8, "2": 0.6})[0] == 25.0
    assert fatigue_module.sprint_drop_score(1, -1)[0] == 25.0
    assert fatigue_module.dist_drop_score(100.0, -100.0)[0] == 25.0
    assert fatigue_module.hsr_load_score(200.0, 100.0) == 25.0


def test_zero_first_half_sprints_scores_zero(fatigue_module):
    assert fatigue_module.sprint_drop_score(0, 5) == (0.0, 0.0)


@pytest.mark.parametrize(
    "h1, h2, expected_score, expected_drop",
    [(10, 10, 0.0, 0.0), (10, 8, 5.0, 0.2), (10, 0, 25.0, 1.0), (10, 12, 0.0, 0.0)],
)
def test_first_half_vs_second_half_sprint_reduction(fatigue_module, h1, h2, expected_score, expected_drop):
    score, drop = fatigue_module.sprint_drop_score(h1, h2)
    assert score == pytest.approx(expected_score)
    assert drop == pytest.approx(expected_drop)


@pytest.mark.parametrize(
    "h1, h2, expected_score, expected_drop",
    [(100, 100, 0.0, 0.0), (100, 70, 12.5, 0.3), (100, 40, 25.0, 0.6), (100, 120, 0.0, 0.0)],
)
def test_first_half_vs_second_half_distance_reduction(fatigue_module, h1, h2, expected_score, expected_drop):
    score, drop = fatigue_module.dist_drop_score(h1, h2)
    assert score == pytest.approx(expected_score)
    assert drop == pytest.approx(expected_drop)


@pytest.mark.parametrize("total, expected", [(50, 12.5), (100, 25.0), (150, 25.0)])
def test_v1_hsr_load_uses_total_distance_below_at_and_above_reference(fatigue_module, total, expected):
    assert fatigue_module.hsr_load_score(total, 100.0) == pytest.approx(expected)


def run_main_with_player(fatigue_module, monkeypatch, tmp_path, player_overrides=None):
    player = {
        "track_id": 1,
        "total_distance_m": 100.0,
        "present_h1": True,
        "present_h2": True,
        "coverage_frac": 1.0,
        "speed_timeline": {"0": 1.0, "1": 1.0, "2": 1.0},
    }
    player.update(player_overrides or {})
    summary = {"players": {"1": player}}
    (tmp_path / "player_summary.json").write_text(json.dumps(summary))
    stats = pd.DataFrame(
        [
            {"track_id": 1, "half": 0, "sprint_count": 1, "distance_m": 50.0},
            {"track_id": 1, "half": 1, "sprint_count": 1, "distance_m": 50.0},
        ]
    )
    monkeypatch.setattr(fatigue_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(fatigue_module.pd, "read_parquet", lambda _path: stats)
    fatigue_module.main()
    return json.loads((tmp_path / "risk_scores.json").read_text())["1"]


def set_component_total(fatigue_module, monkeypatch, total):
    remaining = float(total)
    values = []
    for _ in range(4):
        value = min(25.0, remaining)
        values.append(value)
        remaining -= value
    monkeypatch.setattr(fatigue_module, "speed_decay_score", lambda _timeline: (values[0], 0.0))
    monkeypatch.setattr(fatigue_module, "sprint_drop_score", lambda _h1, _h2: (values[1], 0.0))
    monkeypatch.setattr(fatigue_module, "dist_drop_score", lambda _h1, _h2: (values[2], 0.0))
    monkeypatch.setattr(fatigue_module, "hsr_load_score", lambda _total, _p75: values[3])


@pytest.mark.parametrize(
    "total, expected_flag",
    [(0, "LOW"), (39, "LOW"), (40, "MEDIUM"), (69, "MEDIUM"), (70, "HIGH"), (100, "HIGH")],
)
def test_risk_flag_boundaries(fatigue_module, monkeypatch, tmp_path, total, expected_flag):
    set_component_total(fatigue_module, monkeypatch, total)
    record = run_main_with_player(fatigue_module, monkeypatch, tmp_path)
    assert record["risk_score"] == total
    assert record["risk_flag"] == expected_flag


@pytest.mark.parametrize("component_value, expected", [(30.0, 100), (-10.0, 0)])
def test_total_score_is_clamped_to_zero_through_100(fatigue_module, monkeypatch, tmp_path, component_value, expected):
    monkeypatch.setattr(fatigue_module, "speed_decay_score", lambda _timeline: (component_value, 0.0))
    monkeypatch.setattr(fatigue_module, "sprint_drop_score", lambda _h1, _h2: (component_value, 0.0))
    monkeypatch.setattr(fatigue_module, "dist_drop_score", lambda _h1, _h2: (component_value, 0.0))
    monkeypatch.setattr(fatigue_module, "hsr_load_score", lambda _total, _p75: component_value)
    record = run_main_with_player(fatigue_module, monkeypatch, tmp_path)
    assert record["risk_score"] == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"present_h1": False},
        {"present_h2": False},
        {"coverage_frac": 0.049},
    ],
)
def test_insufficient_gating_preserves_current_schema(fatigue_module, monkeypatch, tmp_path, overrides):
    record = run_main_with_player(fatigue_module, monkeypatch, tmp_path, overrides)
    assert record["risk_score"] is None
    assert record["risk_flag"] == "INSUFFICIENT"
    assert record["reason"] == "partial track (ID switch / sub) — not present across both halves"
    assert record["score_breakdown"] == {}
