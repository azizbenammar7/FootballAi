"""Characterize V1 observation-count heatmap behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def heatmap(module, xs, ys, x_max=10.0, y_max=10.0):
    return np.asarray(module.build_heatmap(pd.Series(xs, dtype=float), pd.Series(ys, dtype=float), x_max, y_max))


def test_heatmap_has_configured_dimensions(stats_module):
    result = heatmap(stats_module, [1], [1])
    assert result.shape == (stats_module.HEATMAP_BINS, stats_module.HEATMAP_BINS)


def test_nonempty_heatmap_normalizes_total_occupancy(stats_module):
    result = heatmap(stats_module, [1, 2, 3], [1, 2, 3])
    assert result.sum() == pytest.approx(1.0)


def test_repeated_positions_accumulate_in_one_cell(stats_module):
    result = heatmap(stats_module, [1, 1, 1], [1, 1, 1])
    assert np.count_nonzero(result) == 1
    assert result.sum() == pytest.approx(1.0)


def test_boundary_positions_land_in_first_and_last_bins(stats_module):
    result = heatmap(stats_module, [0, 10], [0, 10])
    assert result[0, 0] == pytest.approx(0.5)
    assert result[-1, -1] == pytest.approx(0.5)


def test_outside_positions_are_clipped_to_frame_bounds(stats_module):
    result = heatmap(stats_module, [-5, 15], [-3, 14])
    assert result[0, 0] == pytest.approx(0.5)
    assert result[-1, -1] == pytest.approx(0.5)


def test_empty_input_returns_zero_finite_grid(stats_module):
    result = heatmap(stats_module, [], [])
    assert result.shape == (10, 10)
    assert result.sum() == 0.0
    assert np.isfinite(result).all()


def test_heatmap_contains_no_nan_or_infinity(stats_module):
    result = heatmap(stats_module, [-100, 0, 5, 100], [-100, 0, 5, 100])
    assert np.isfinite(result).all()


def test_v1_heatmap_is_count_based_not_time_weighted(stats_module):
    result = heatmap(stats_module, [1, 1, 9], [1, 1, 9])
    assert result[1, 1] == pytest.approx(2 / 3)
    assert result[9, 9] == pytest.approx(1 / 3)
