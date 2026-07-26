"""Bounded cross-file checks over committed V1 artifacts."""

from __future__ import annotations

import json

import pytest

from conftest import ROOT, assert_finite_json


PROCESSED = ROOT / "data" / "processed"


def load_json(name):
    return json.loads((PROCESSED / name).read_text())


def test_meta_required_fields_and_internal_consistency():
    meta = load_json("meta.json")
    required = {"src_fps", "total_frames", "width", "height", "duration_s", "stride", "effective_fps"}
    assert required <= meta.keys()
    assert meta["src_fps"] > 0
    assert isinstance(meta["total_frames"], int) and meta["total_frames"] > 0
    assert isinstance(meta["width"], int) and meta["width"] > 0
    assert isinstance(meta["height"], int) and meta["height"] > 0
    assert meta["duration_s"] > 0
    assert isinstance(meta["stride"], int) and meta["stride"] > 0
    assert meta["effective_fps"] == pytest.approx(meta["src_fps"] / meta["stride"])
    assert meta["duration_s"] == pytest.approx(meta["total_frames"] / meta["src_fps"])
    assert_finite_json(meta)


def test_cross_file_identifier_and_duration_invariants():
    meta = load_json("meta.json")
    summary = load_json("player_summary.json")
    risk = load_json("risk_scores.json")
    players = summary["players"]

    assert set(risk) <= set(players)
    for identifier, record in risk.items():
        assert isinstance(identifier, str)
        assert isinstance(players[identifier]["track_id"], int)
        assert isinstance(record["track_id"], int)
        assert players[identifier]["track_id"] == record["track_id"]
    assert summary["match_duration_s"] <= meta["duration_s"]
    assert meta["duration_s"] - summary["match_duration_s"] <= meta["stride"] / meta["src_fps"] + 1.0
    assert all(player["total_distance_m"] >= 0 for player in players.values())
    assert all(player["active_time_s"] >= 0 for player in players.values())
    max_block = max((max(player["blocks_present"], default=-1) for player in players.values()), default=-1)
    assert max_block < summary["n_blocks"]
    assert max_block * 15 * 60 <= meta["duration_s"]
    assert all(
        record["risk_score"] is None or 0 <= record["risk_score"] <= 100
        for record in risk.values()
    )


def test_committed_player_stats_parquet_schema_and_non_negative_aggregates():
    parquet = pytest.importorskip(
        "pyarrow.parquet",
        reason="optional committed-Parquet validation requires pyarrow; bounded JSON contracts still run",
    )
    table = parquet.read_table(PROCESSED / "player_stats.parquet")
    required = {
        "track_id", "block", "half", "distance_m", "mean_speed_ms",
        "peak_speed_ms", "sprint_count", "active_frames", "active_time_s",
    }
    assert required <= set(table.column_names)
    for column in ("distance_m", "mean_speed_ms", "peak_speed_ms", "sprint_count", "active_frames", "active_time_s"):
        values = table[column].to_pylist()
        assert all(value is not None and value >= 0 for value in values)
