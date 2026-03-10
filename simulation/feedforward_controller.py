"""Profile-level wrappers around the analytic inverse feedforward model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from simulation.dynamics import ExtendedPlant, ExtendedPlantParams
from simulation.inverse_dynamics import AnalyticInverseFeedforward


@dataclass(slots=True)
class FeedforwardProfileResult:
    """Batch feedforward output for a target-acceleration profile."""

    target_accel: np.ndarray
    speed: np.ndarray
    grade_rad: np.ndarray
    raw_action: np.ndarray
    action: np.ndarray
    was_clipped: np.ndarray
    mode: list[str]


@dataclass(slots=True)
class FeedforwardClosedLoopResult:
    """Closed-loop rollout result using feedforward actions on the plant."""

    time: np.ndarray
    target_accel: np.ndarray
    raw_action: np.ndarray
    action: np.ndarray
    was_clipped: np.ndarray
    mode: list[str]
    speed: np.ndarray
    acceleration: np.ndarray


class FeedforwardController:
    """Runtime wrapper for mapping acceleration profiles to action profiles."""

    def __init__(self, params: ExtendedPlantParams):
        self.params = params
        self.inverse = AnalyticInverseFeedforward(params)

    def compute_action_profile(
        self,
        target_accel_profile: Sequence[float],
        speed_profile: Sequence[float],
        grade_profile: Sequence[float] | None = None,
    ) -> FeedforwardProfileResult:
        """Map `(target_accel_profile, speed_profile)` to feedforward actions."""
        target_arr = _to_1d_float_array(target_accel_profile, name="target_accel_profile")
        speed_arr = _to_1d_float_array(speed_profile, name="speed_profile")
        if target_arr.size != speed_arr.size:
            raise ValueError("target_accel_profile and speed_profile must have the same length")

        if grade_profile is None:
            grade_arr = np.full_like(target_arr, self.params.body.grade_rad, dtype=np.float64)
        else:
            grade_arr = _to_1d_float_array(grade_profile, name="grade_profile")
            if grade_arr.size != target_arr.size:
                raise ValueError("grade_profile must have the same length as target_accel_profile")

        raw_action = np.zeros_like(target_arr)
        action = np.zeros_like(target_arr)
        was_clipped = np.zeros(target_arr.size, dtype=bool)
        mode: list[str] = []

        for k in range(target_arr.size):
            ff = self.inverse.compute_action(
                target_accel=float(target_arr[k]),
                speed=float(speed_arr[k]),
                grade_rad=float(grade_arr[k]),
            )
            raw_action[k] = ff.raw_action
            action[k] = ff.action
            was_clipped[k] = ff.was_clipped
            mode.append(ff.mode)

        return FeedforwardProfileResult(
            target_accel=target_arr,
            speed=speed_arr,
            grade_rad=grade_arr,
            raw_action=raw_action,
            action=action,
            was_clipped=was_clipped,
            mode=mode,
        )

    def rollout_action_profile(
        self,
        target_accel_profile: Sequence[float],
        initial_speed: float,
        dt: float,
        grade_profile: Sequence[float] | None = None,
        substeps: int = 1,
    ) -> FeedforwardClosedLoopResult:
        """Run feedforward online and collect the resulting action profile.

        At each time step `k`:
        1. Use current plant speed and `target_accel[k]` to compute action.
        2. Apply action to `ExtendedPlant.step(...)`.
        3. Store action and realized state.
        """
        target_arr = _to_1d_float_array(target_accel_profile, name="target_accel_profile")

        if grade_profile is None:
            grade_arr = np.full_like(target_arr, self.params.body.grade_rad, dtype=np.float64)
        else:
            grade_arr = _to_1d_float_array(grade_profile, name="grade_profile")
            if grade_arr.size != target_arr.size:
                raise ValueError("grade_profile must have the same length as target_accel_profile")

        n = target_arr.size
        time = np.arange(n, dtype=np.float64) * float(dt)
        raw_action = np.zeros(n, dtype=np.float64)
        action = np.zeros(n, dtype=np.float64)
        was_clipped = np.zeros(n, dtype=bool)
        mode: list[str] = []
        speed = np.zeros(n, dtype=np.float64)
        acceleration = np.zeros(n, dtype=np.float64)

        plant = ExtendedPlant(self.params)
        plant.reset(speed=float(initial_speed))

        for k in range(n):
            ff = self.inverse.compute_action(
                target_accel=float(target_arr[k]),
                speed=float(plant.speed),
                grade_rad=float(grade_arr[k]),
            )
            state = plant.step(
                action=ff.action,
                dt=float(dt),
                substeps=int(substeps),
                grade_rad=float(grade_arr[k]),
            )
            raw_action[k] = ff.raw_action
            action[k] = ff.action
            was_clipped[k] = ff.was_clipped
            mode.append(ff.mode)
            speed[k] = state.speed
            acceleration[k] = state.acceleration

        return FeedforwardClosedLoopResult(
            time=time,
            target_accel=target_arr,
            raw_action=raw_action,
            action=action,
            was_clipped=was_clipped,
            mode=mode,
            speed=speed,
            acceleration=acceleration,
        )


def _to_1d_float_array(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


__all__ = [
    "FeedforwardProfileResult",
    "FeedforwardClosedLoopResult",
    "FeedforwardController",
]
