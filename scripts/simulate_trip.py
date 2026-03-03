#!/usr/bin/env python3
"""Run vehicle simulation with fitted parameters.

This script loads fitted vehicle parameters and runs a simulation using
the ExtendedPlant model. It can simulate from trip data or generate
synthetic trajectories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fitting.fitter import FittedVehicleParams
from simulation.dynamics import ExtendedPlant, ExtendedPlantParams, MotorParams, BrakeParams, BodyParams, WheelParams, CreepParams


def load_fitted_params(path: Path) -> FittedVehicleParams:
    """Load fitted vehicle parameters from JSON file."""
    return FittedVehicleParams.load(path)


def fitted_to_extended_params(fitted: FittedVehicleParams) -> ExtendedPlantParams:
    """Convert FittedVehicleParams to ExtendedPlantParams."""
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
        gear_ratio=fitted.gear_ratio,
        eta_gb=fitted.eta_gb,
    )
    
    brake = BrakeParams(
        T_br_max=fitted.brake_T_max,
        p_br=fitted.brake_p,
        tau_br=fitted.brake_tau,
        kappa_c=fitted.brake_kappa,
        mu=fitted.mu,
    )
    
    body = BodyParams(
        mass=fitted.mass,
        drag_area=fitted.drag_area,
        rolling_coeff=fitted.rolling_coeff,
        grade_rad=0.0,  # Can be overridden per step
    )
    
    wheel = WheelParams(
        radius=fitted.wheel_radius,
        inertia=fitted.wheel_inertia,
    )
    
    creep = CreepParams(
        a_max=fitted.creep_a_max,
        v_cutoff=fitted.creep_v_cutoff,
        v_hold=fitted.creep_v_hold,
    )
    
    return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel, creep=creep)


def simulate_from_trip_data(
    fitted_params: FittedVehicleParams,
    trip_data_path: Path,
    output_path: Path,
    dt: float = 0.1,
) -> None:
    """Simulate using trip data inputs."""
    import torch
    
    # Load trip data
    data = torch.load(trip_data_path, map_location="cpu", weights_only=False)
    
    # Get first trip
    trip_id = next(k for k in data.keys() if k != "metadata")
    trip = data[trip_id]
    
    # Convert fitted params to ExtendedPlant params
    plant_params = fitted_to_extended_params(fitted_params)
    plant = ExtendedPlant(plant_params)
    
    # Get inputs
    throttle = np.asarray(trip["throttle"], dtype=np.float64) / 100.0  # Convert 0-100 to 0-1
    brake = np.asarray(trip["brake"], dtype=np.float64) / 100.0
    grade = np.asarray(trip.get("angle", np.zeros_like(throttle)), dtype=np.float64)
    
    # Initialize
    initial_speed = float(trip["speed"][0])
    plant.reset(speed=initial_speed, position=0.0)
    
    # Simulate
    n = len(throttle)
    results = {
        "time": np.zeros(n),
        "speed_sim": np.zeros(n),
        "speed_measured": np.asarray(trip["speed"], dtype=np.float64),
        "acceleration_sim": np.zeros(n),
        "acceleration_measured": np.asarray(trip.get("acceleration", np.zeros(n)), dtype=np.float64),
        "throttle": throttle,
        "brake": brake,
        "grade": grade,
    }
    
    results["speed_sim"][0] = initial_speed
    
    for t in range(n - 1):
        # Action: positive = throttle, negative = brake
        action = throttle[t] if throttle[t] > 0 else -brake[t]
        
        # Step simulation
        state = plant.step(action, dt=dt, grade_rad=float(grade[t]))
        
        # Store results
        results["time"][t + 1] = results["time"][t] + dt
        results["speed_sim"][t + 1] = state.speed
        results["acceleration_sim"][t + 1] = state.acceleration
    
    # Save results
    np.savez(output_path, **results)
    print(f"Simulation results saved to {output_path}")
    
    # Print summary
    speed_error = np.abs(results["speed_sim"] - results["speed_measured"])
    print(f"Mean speed error: {np.mean(speed_error):.4f} m/s")
    print(f"Max speed error: {np.max(speed_error):.4f} m/s")
    print(f"RMSE: {np.sqrt(np.mean(speed_error**2)):.4f} m/s")


def simulate_synthetic(
    fitted_params: FittedVehicleParams,
    output_path: Path,
    duration: float = 60.0,
    dt: float = 0.1,
) -> None:
    """Simulate with synthetic throttle/brake inputs."""
    # Convert fitted params to ExtendedPlant params
    plant_params = fitted_to_extended_params(fitted_params)
    plant = ExtendedPlant(plant_params)
    
    # Initialize
    plant.reset(speed=0.0, position=0.0)
    
    # Generate simple synthetic trajectory
    n = int(duration / dt)
    throttle = np.zeros(n)
    brake = np.zeros(n)
    
    # Simple pattern: accelerate, cruise, brake
    for t in range(n):
        if t < n // 3:
            throttle[t] = 0.5  # Accelerate
        elif t < 2 * n // 3:
            throttle[t] = 0.3  # Cruise
        else:
            brake[t] = 0.5  # Brake
    
    # Simulate
    results = {
        "time": np.zeros(n),
        "speed": np.zeros(n),
        "acceleration": np.zeros(n),
        "throttle": throttle,
        "brake": brake,
    }
    
    for t in range(n - 1):
        action = throttle[t] if throttle[t] > 0 else -brake[t]
        state = plant.step(action, dt=dt)
        
        results["time"][t + 1] = results["time"][t] + dt
        results["speed"][t + 1] = state.speed
        results["acceleration"][t + 1] = state.acceleration
    
    # Save results
    np.savez(output_path, **results)
    print(f"Simulation results saved to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run vehicle simulation with fitted parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--params", type=Path, required=True,
        help="Path to fitted_params.json file",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for simulation results (.npz file)",
    )
    parser.add_argument(
        "--trip-data", type=Path, default=None,
        help="Optional: Path to trip data .pt file to simulate from",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Simulation duration in seconds (for synthetic simulation, default: 60.0)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1,
        help="Time step in seconds (default: 0.1)",
    )
    
    args = parser.parse_args()
    
    if not args.params.exists():
        print(f"Error: Parameters file not found: {args.params}")
        return 1
    
    # Load fitted parameters
    fitted_params = load_fitted_params(args.params)
    print(f"Loaded parameters from {args.params}")
    
    # Run simulation
    if args.trip_data is not None:
        if not args.trip_data.exists():
            print(f"Error: Trip data file not found: {args.trip_data}")
            return 1
        simulate_from_trip_data(fitted_params, args.trip_data, args.output, dt=args.dt)
    else:
        simulate_synthetic(fitted_params, args.output, duration=args.duration, dt=args.dt)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
