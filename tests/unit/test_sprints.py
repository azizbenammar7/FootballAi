"""Characterize sprint interval counting at V1 boundaries."""

from __future__ import annotations

import numpy as np
import pytest


def count(module, times, speeds):
    return module.count_sprints(np.asarray(times, dtype=float), np.asarray(speeds, dtype=float))


def test_speed_below_threshold_is_not_a_sprint(stats_module):
    assert count(stats_module, [0, 1, 2], [5.49, 5.49, 5.49]) == 0


def test_speed_exactly_at_threshold_qualifies(stats_module):
    threshold = stats_module.SPRINT_SPEED_MS
    assert count(stats_module, [0, 1], [threshold, threshold]) == 1


def test_continuous_interval_shorter_than_minimum_is_not_counted(stats_module):
    threshold = stats_module.SPRINT_SPEED_MS
    assert count(stats_module, [0, 0.9], [threshold, threshold]) == 0


def test_continuous_interval_exactly_at_minimum_is_counted(stats_module):
    threshold = stats_module.SPRINT_SPEED_MS
    duration = stats_module.MIN_SPRINT_DURATION_S
    assert count(stats_module, [0, duration], [threshold, threshold]) == 1


def test_continuous_interval_longer_than_minimum_is_counted(stats_module):
    threshold = stats_module.SPRINT_SPEED_MS
    assert count(stats_module, [0, 1.5], [threshold, threshold]) == 1


def test_two_separated_sprint_intervals_are_counted(stats_module):
    threshold = stats_module.SPRINT_SPEED_MS
    assert count(stats_module, [0, 1, 1.1, 2, 3], [threshold, threshold, 0, threshold, threshold]) == 2


def test_interruption_below_threshold_ends_current_interval(stats_module):
    threshold = stats_module.SPRINT_SPEED_MS
    assert count(stats_module, [0, 0.5, 0.6, 1, 2], [threshold, threshold, 0, threshold, threshold]) == 1


def test_irregular_spacing_uses_below_threshold_sample_as_interval_end(stats_module):
    threshold = stats_module.SPRINT_SPEED_MS
    assert count(stats_module, [0, 0.1, 10], [threshold, threshold, 0]) == 1


def test_empty_input_returns_zero(stats_module):
    assert count(stats_module, [], []) == 0


@pytest.mark.parametrize("speed", [0.0, 5.5])
def test_single_observation_is_insufficient_for_a_sprint(stats_module, speed):
    assert count(stats_module, [0], [speed]) == 0
