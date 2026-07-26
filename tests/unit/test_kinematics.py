"""Characterize V1 kinematics without changing its approximations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def observations(times, xs, ys=None):
    if ys is None:
        ys = np.zeros(len(xs))
    return pd.DataFrame({"time_sec": times, "cx": xs, "cy": ys})


def test_stationary_player_has_zero_displacement_distance_and_speed(stats_module):
    group = observations([0.0, 0.2, 0.4, 0.6, 0.8], [50.0] * 5, [75.0] * 5)
    speed, distance = stats_module.compute_kinematics(group)

    assert speed.tolist() == [0.0] * 5
    assert distance.tolist() == [0.0] * 5


def test_constant_linear_motion_preserves_current_centered_smoothing(stats_module):
    group = observations([0, 1, 2, 3, 4], [0, 10, 20, 30, 40])
    speed, distance = stats_module.compute_kinematics(group)
    expected_step = 5 * stats_module.M_PER_PX

    assert distance.to_numpy() == pytest.approx([0, expected_step, expected_step, expected_step, expected_step])
    assert speed.to_numpy() == pytest.approx([0, expected_step, expected_step, expected_step, expected_step])


def test_zero_time_delta_is_ignored_without_division_failure(stats_module):
    group = observations([0.0, 0.2, 0.4, 0.4, 0.6, 0.8], [0, 0, 0, 100, 100, 100])
    speed, distance = stats_module.compute_kinematics(group)

    assert speed.iloc[3] == 0.0
    assert distance.iloc[3] == 0.0
    assert np.isfinite(speed).all()
    assert np.isfinite(distance).all()


def test_negative_out_of_order_time_delta_is_ignored(stats_module):
    group = observations([0.0, 0.2, 0.4, 0.3, 0.5, 0.7], [0, 0, 0, 100, 100, 100])
    speed, distance = stats_module.compute_kinematics(group)

    assert speed.iloc[3] == 0.0
    assert distance.iloc[3] == 0.0


def test_gap_longer_than_max_gap_discards_reappearance_jump(stats_module):
    gap = stats_module.MAX_GAP_S + 0.1
    group = observations([0.0, 0.2, 0.4, 0.4 + gap, 0.6 + gap, 0.8 + gap], [0, 0, 0, 100, 100, 100])
    speed, distance = stats_module.compute_kinematics(group)

    assert speed.iloc[3] == 0.0
    assert distance.iloc[3] == 0.0


def test_gap_at_threshold_preserves_current_movement(stats_module):
    group = observations([0.0, 0.2, 0.4, 3.4, 3.6, 3.8], [0, 0, 0, 100, 100, 100])
    speed, distance = stats_module.compute_kinematics(group)
    expected_distance = 100 * stats_module.M_PER_PX

    assert speed.iloc[3] == pytest.approx(expected_distance / stats_module.MAX_GAP_S)
    assert distance.iloc[3] == pytest.approx(expected_distance)


def test_speed_cap_reconstructs_distance_from_capped_speed(stats_module):
    group = observations([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0, 0, 1000, 1000, 1000])
    speed, distance = stats_module.compute_kinematics(group)

    assert speed.iloc[3] == stats_module.MAX_SPEED_MS
    assert distance.iloc[3] == pytest.approx(stats_module.MAX_SPEED_MS * 0.2)


def test_single_position_spike_is_removed_by_rolling_median(stats_module):
    group = observations([0.0, 0.2, 0.4, 0.6, 0.8], [0, 0, 100, 0, 0])
    speed, distance = stats_module.compute_kinematics(group)

    assert speed.tolist() == [0.0] * 5
    assert distance.tolist() == [0.0] * 5


@pytest.mark.parametrize(
    "group, expected_length",
    [
        (observations([], []), 0),
        (observations([0.0], [10.0]), 1),
        (observations([0.0, 0.2], [0.0, 10.0]), 2),
    ],
)
def test_missing_or_minimal_observations_are_supported(stats_module, group, expected_length):
    speed, distance = stats_module.compute_kinematics(group)

    assert len(speed) == expected_length
    assert len(distance) == expected_length
    assert (speed == 0).all()
    assert (distance == 0).all()


def test_kinematics_never_produces_negative_distance(stats_module):
    group = observations([0.0, 0.2, 0.4, 0.6, 4.0, 4.2], [100, 50, 75, 10, 500, -20], [10, 20, 5, 30, -40, 80])
    _, distance = stats_module.compute_kinematics(group)

    assert (distance >= 0).all()
