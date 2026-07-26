"""Schema and invariant checks for V1 player summaries."""

from __future__ import annotations

import json

import pytest

from conftest import ROOT, assert_finite_json


def assert_player_summary_contract(summary):
    assert isinstance(summary, dict)
    required_root = {
        "match_duration_s", "effective_fps", "total_players", "n_blocks",
        "m_per_px", "sprint_speed_threshold_ms", "players",
    }
    assert required_root <= summary.keys()
    assert isinstance(summary["players"], dict)
    assert summary["total_players"] == len(summary["players"])
    assert summary["match_duration_s"] > 0
    assert summary["effective_fps"] > 0
    assert summary["n_blocks"] > 0

    for key, player in summary["players"].items():
        assert isinstance(key, str)
        assert isinstance(player["track_id"], int)
        assert str(player["track_id"]) == key
        assert player["total_distance_m"] >= 0
        assert player["mean_speed_ms"] >= 0
        assert player["peak_speed_ms"] >= 0
        assert isinstance(player["total_sprints"], int) and player["total_sprints"] >= 0
        assert player["active_time_s"] >= 0
        assert isinstance(player["blocks_present"], list)
        assert isinstance(player["speed_timeline"], dict)
        assert all(isinstance(row, list) and len(row) == 10 for row in player["heatmap"])
        assert len(player["heatmap"]) == 10
        assert sum(map(sum, player["heatmap"])) == pytest.approx(1.0)
        assert all(speed >= 0 for speed in player["speed_timeline"].values())
    assert_finite_json(summary)


def test_minimal_player_summary_fixture_contract(minimal_player_summary):
    assert_player_summary_contract(minimal_player_summary)


def test_committed_player_summary_contract():
    summary = json.loads((ROOT / "data/processed/player_summary.json").read_text())
    assert_player_summary_contract(summary)
