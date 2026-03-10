"""Tests for profile-level feedforward controller wrappers."""

from __future__ import annotations

import numpy as np

from simulation import ExtendedPlantParams
from simulation.feedforward_controller import FeedforwardController
from simulation.inverse_dynamics import AnalyticInverseFeedforward


def test_compute_action_profile_matches_stepwise_inverse() -> None:
    params = ExtendedPlantParams()
    controller = FeedforwardController(params)
    inverse = AnalyticInverseFeedforward(params)

    target = np.array([0.5, 0.2, -0.1, -1.0], dtype=np.float64)
    speed = np.array([5.0, 6.0, 6.5, 6.2], dtype=np.float64)
    grade = np.array([0.0, 0.01, 0.01, -0.01], dtype=np.float64)

    profile = controller.compute_action_profile(target, speed, grade)

    expected_raw = np.array(
        [inverse.compute_action(target_accel=float(a), speed=float(v), grade_rad=float(g)).raw_action for a, v, g in zip(target, speed, grade)],
        dtype=np.float64,
    )
    expected_action = np.clip(expected_raw, -1.0, 1.0)

    np.testing.assert_allclose(profile.raw_action, expected_raw)
    np.testing.assert_allclose(profile.action, expected_action)
    assert profile.was_clipped.dtype == bool
    assert len(profile.mode) == target.size


def test_compute_action_profile_uses_default_grade() -> None:
    params = ExtendedPlantParams()
    params.body.grade_rad = 0.03
    controller = FeedforwardController(params)

    target = np.array([0.4, 0.4, 0.4], dtype=np.float64)
    speed = np.array([8.0, 8.5, 9.0], dtype=np.float64)

    profile = controller.compute_action_profile(target, speed)

    assert np.allclose(profile.grade_rad, 0.03)
    assert profile.action.shape == target.shape


def test_rollout_action_profile_runs_with_physical_outputs() -> None:
    params = ExtendedPlantParams()
    controller = FeedforwardController(params)

    n = 40
    target = np.concatenate([
        np.full(15, 0.6),
        np.full(10, 0.0),
        np.full(15, -0.5),
    ]).astype(np.float64)
    grade = np.zeros(n, dtype=np.float64)

    rollout = controller.rollout_action_profile(
        target_accel_profile=target,
        initial_speed=10.0,
        dt=0.1,
        grade_profile=grade,
        substeps=3,
    )

    assert rollout.action.shape == target.shape
    assert rollout.raw_action.shape == target.shape
    assert rollout.speed.shape == target.shape
    assert rollout.acceleration.shape == target.shape
    assert np.isfinite(rollout.speed).all()
    assert np.isfinite(rollout.acceleration).all()

    # Basic directional consistency: early segment accelerates more than late braking segment.
    assert np.mean(rollout.acceleration[:10]) > np.mean(rollout.acceleration[-10:])
