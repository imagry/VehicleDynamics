#!/usr/bin/env python3
"""Run analytic feedforward on a trip and save raw/clipped action traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fitting.fitter import FittedVehicleParams
from simulation import (
    BodyParams,
    BrakeParams,
    ExtendedPlant,
    ExtendedPlantParams,
    FeedforwardController,
    MotorParams,
    WheelParams,
)


def fitted_to_extended_params(fitted: FittedVehicleParams) -> ExtendedPlantParams:
    """Convert fitted parameter object to plant parameters."""
    motor = MotorParams(
        R=fitted.motor_R,
        K_e=fitted.motor_K,
        K_t=fitted.motor_K,
        b=fitted.motor_b,
        J=fitted.motor_J,
        V_max=fitted.motor_V_max,
        T_max=fitted.motor_T_max,
        P_max=fitted.motor_P_max,
        gamma_throttle=fitted.motor_gamma_throttle,
        throttle_tau=fitted.motor_throttle_tau,
        min_current_A=fitted.motor_min_current_A,
        gear_ratio=fitted.gear_ratio,
        eta_gb=fitted.eta_gb,
    )
    brake = BrakeParams(
        T_br_max=fitted.brake_T_max,
        p_br=fitted.brake_p,
        tau_br=fitted.brake_tau,
        mu=fitted.mu,
    )
    body = BodyParams(
        mass=fitted.mass,
        drag_area=fitted.drag_area,
        rolling_coeff=fitted.rolling_coeff,
        grade_rad=0.0,
    )
    wheel = WheelParams(
        radius=fitted.wheel_radius,
        inertia=fitted.wheel_inertia,
    )
    return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel)


def _select_trip(dataset: dict, trip_id: str | None) -> tuple[str, dict]:
    if trip_id is not None:
        if trip_id not in dataset:
            raise KeyError(f"Trip '{trip_id}' not found in dataset")
        return trip_id, dataset[trip_id]

    first_trip = next((k for k in dataset.keys() if k != "metadata"), None)
    if first_trip is None:
        raise ValueError("No trip entries found in dataset")
    return first_trip, dataset[first_trip]


def _extract_grade(trip: dict, angle_is_deg: bool) -> np.ndarray:
    if "angle" not in trip:
        return np.zeros_like(np.asarray(trip["speed"], dtype=np.float64))
    grade = np.asarray(trip["angle"], dtype=np.float64)
    if angle_is_deg:
        return np.deg2rad(grade)
    return grade


def run_feedforward_on_trip(
    params: ExtendedPlantParams,
    trip: dict,
    dt: float,
    angle_is_deg: bool,
    substeps: int,
) -> dict[str, np.ndarray]:
    speed = np.asarray(trip["speed"], dtype=np.float64)
    target_accel = np.asarray(trip.get("acceleration", np.gradient(speed, dt)), dtype=np.float64)
    grade = _extract_grade(trip, angle_is_deg=angle_is_deg)

    controller = FeedforwardController(params)
    profile = controller.compute_action_profile(
        target_accel_profile=target_accel,
        speed_profile=speed,
        grade_profile=grade,
    )

    # Optional forward rollout for quick sanity checks of realized states.
    plant = ExtendedPlant(params)
    plant.reset(speed=float(speed[0]))
    speed_sim = np.zeros_like(speed)
    accel_sim = np.zeros_like(speed)
    speed_sim[0] = speed[0]
    accel_sim[0] = plant.acceleration

    for k in range(1, speed.size):
        state = plant.step(
            action=float(profile.action[k - 1]),
            dt=float(dt),
            substeps=int(substeps),
            grade_rad=float(grade[k - 1]),
        )
        speed_sim[k] = state.speed
        accel_sim[k] = state.acceleration

    return {
        "time": np.arange(speed.size, dtype=np.float64) * dt,
        "speed": speed,
        "target_accel": target_accel,
        "grade": grade,
        "raw_action": profile.raw_action,
        "action": profile.action,
        "was_clipped": profile.was_clipped.astype(np.int8),
        "mode_drive": np.asarray([1 if m == "drive" else 0 for m in profile.mode], dtype=np.int8),
        "throttle_ff": np.clip(profile.action, 0.0, 1.0),
        "brake_ff": np.clip(-profile.action, 0.0, 1.0),
        "speed_sim": speed_sim,
        "accel_sim": accel_sim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute analytic feedforward traces for a trip")
    parser.add_argument("--trip-data", type=Path, required=True, help="Path to parsed trip .pt file")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file path")
    parser.add_argument("--params", type=Path, default=None, help="Optional fitted params JSON path")
    parser.add_argument("--trip-id", type=str, default=None, help="Trip key in dataset (default: first)")
    parser.add_argument("--dt", type=float, default=None, help="Override dt (default: dataset metadata dt or 0.1)")
    parser.add_argument("--angle-is-deg", action="store_true", help="Interpret trip angle as degrees")
    parser.add_argument("--substeps", type=int, default=1, help="Plant substeps for optional rollout")
    args = parser.parse_args()

    dataset = torch.load(args.trip_data, map_location="cpu", weights_only=False)
    trip_id, trip = _select_trip(dataset, args.trip_id)

    metadata = dataset.get("metadata", {}) if isinstance(dataset, dict) else {}
    dt = float(args.dt if args.dt is not None else metadata.get("dt", 0.1))

    if args.params is None:
        params = ExtendedPlantParams()
    else:
        fitted = FittedVehicleParams.load(args.params)
        params = fitted_to_extended_params(fitted)

    traces = run_feedforward_on_trip(
        params=params,
        trip=trip,
        dt=dt,
        angle_is_deg=args.angle_is_deg,
        substeps=max(int(args.substeps), 1),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **traces)

    clip_ratio = float(np.mean(traces["was_clipped"])) * 100.0
    print(f"Trip: {trip_id}")
    print(f"Saved feedforward traces to: {args.output}")
    print(f"Samples: {traces['time'].size}, clipped ratio: {clip_ratio:.2f}%")


if __name__ == "__main__":
    main()
