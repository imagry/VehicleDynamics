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

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fitting.fitter import FittedVehicleParams
from simulation.dynamics import ExtendedPlant, ExtendedPlantParams, MotorParams, BrakeParams, BodyParams, WheelParams


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
        grade_rad=0.0,  # Can be overridden per step
    )
    
    wheel = WheelParams(
        radius=fitted.wheel_radius,
        inertia=fitted.wheel_inertia,
    )
    
    return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel)


def plot_simulation_results(results: dict, plot_path: Path | None = None) -> None:
    """Plot comprehensive simulation results with all states.
    
    Creates a figure with multiple subplots in a single column, all sharing the x-axis (time).
    """
    time = results["time"]
    n_plots = 14  # Total number of subplots
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    fig.suptitle("Simulation Results - All States", fontsize=14, fontweight='bold')
    
    plot_idx = 0
    
    # 1. Actuations
    ax = axes[plot_idx]
    ax.plot(time, results["throttle"] * 100, "g-", label="Throttle", linewidth=2)
    ax.plot(time, results["brake"] * 100, "r-", label="Brake", linewidth=2)
    ax.set_ylabel("Actuation (%)")
    ax.set_ylim([0, 100])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title("Actuations")
    plot_idx += 1
    
    # 2. Speed
    ax = axes[plot_idx]
    if "speed_measured" in results:
        ax.plot(time, results["speed_measured"], "k--", label="Measured", linewidth=1.5, alpha=0.7)
    speed_data = results.get("speed_sim", results.get("speed"))
    ax.plot(time, speed_data, "b-", label="Simulated", linewidth=2)
    ax.set_ylabel("Speed (m/s)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title("Vehicle Speed")
    plot_idx += 1
    
    # 3. Acceleration
    ax = axes[plot_idx]
    if "acceleration_measured" in results:
        ax.plot(time, results["acceleration_measured"], "k--", label="Measured", linewidth=1.5, alpha=0.7)
    accel_data = results.get("acceleration_sim", results.get("acceleration"))
    ax.plot(time, accel_data, "g-", label="Simulated", linewidth=2)
    ax.set_ylabel("Acceleration (m/s²)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title("Acceleration")
    plot_idx += 1
    
    # 4. Road Grade
    ax = axes[plot_idx]
    grade_deg = np.degrees(results["grade"])
    ax.plot(time, grade_deg, "brown", linewidth=2)
    ax.set_ylabel("Grade (deg)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Road Grade")
    plot_idx += 1
    
    # 5. Motor Angular Speed
    ax = axes[plot_idx]
    ax.plot(time, results["motor_omega"], "purple", linewidth=2)
    ax.set_ylabel("Motor ω (rad/s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Motor Angular Speed")
    plot_idx += 1
    
    # 6. Motor Current
    ax = axes[plot_idx]
    ax.plot(time, results["motor_current"], "orange", linewidth=2)
    if "i_limit" in results:
        ax.plot(time, results["i_limit"], "r--", label="Dynamic Current Limit", linewidth=1.5, alpha=0.7)
    if "i_limit_static" in results:
        ax.axhline(y=results["i_limit_static"], color="tab:purple", linestyle=":", label="Static Current Limit", alpha=0.8)
        ax.axhline(y=-results["i_limit_static"], color="tab:purple", linestyle=":", alpha=0.4)
    if "i_limit" in results or "i_limit_static" in results:
        ax.legend(loc="best")
    ax.set_ylabel("Current (A)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Motor Current")
    plot_idx += 1
    
    # 7. Motor Voltage
    ax = axes[plot_idx]
    ax.plot(time, results["V_cmd"], "blue", linewidth=2, label="Commanded")
    if "back_emf_voltage" in results:
        ax.plot(time, results["back_emf_voltage"], "green", linewidth=2, label="Back-EMF", alpha=0.7)
        ax.legend(loc="best")
    if "V_max" in results:
        ax.axhline(y=results["V_max"], color="r", linestyle="--", label="V_max", alpha=0.7)
        ax.axhline(y=-results["V_max"], color="r", linestyle="--", alpha=0.7)
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Motor Voltage")
    plot_idx += 1
    
    # 8. Drive / Motor Torque
    ax = axes[plot_idx]
    ax.plot(time, results["drive_torque"], "cyan", linewidth=2, label="Drive Torque (Wheel)")
    if "motor_shaft_torque" in results:
        ax.plot(time, results["motor_shaft_torque"], "tab:blue", linewidth=1.5, linestyle="--", label="Motor Shaft Torque")
    if "T_max" in results and results["T_max"] is not None:
        # Motor torque limit (shaft-side) and equivalent wheel-side limit.
        t_max_motor = float(results["T_max"])
        ax.axhline(y=t_max_motor, color="r", linestyle="--", label="T_max (Motor)", alpha=0.7)
        ax.axhline(y=-t_max_motor, color="r", linestyle="--", alpha=0.7)
        if "gear_ratio" in results and "eta_gb" in results:
            t_max_wheel = t_max_motor * float(results["gear_ratio"]) * float(results["eta_gb"])
            ax.axhline(y=t_max_wheel, color="tab:purple", linestyle=":", label="T_max (Wheel)", alpha=0.7)
            ax.axhline(y=-t_max_wheel, color="tab:purple", linestyle=":", alpha=0.7)
        ax.legend(loc="best")
    ax.set_ylabel("Torque (Nm)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Drive Torque (Wheel + Motor Shaft)")
    plot_idx += 1
    
    # 9. Motor Power
    ax = axes[plot_idx]
    power = results["V_cmd"] * results["motor_current"]
    ax.plot(time, power, "magenta", linewidth=2)
    if "P_max" in results and results["P_max"] is not None and results["P_max"] > 0:
        ax.axhline(y=results["P_max"], color="r", linestyle="--", label="P_max", alpha=0.7)
        ax.axhline(y=-results["P_max"], color="r", linestyle="--", alpha=0.7)
        ax.legend(loc="best")
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Motor Power")
    plot_idx += 1
    
    # 10. Brake Torque
    ax = axes[plot_idx]
    ax.plot(time, results["brake_torque"], "red", linewidth=2)
    if "brake_T_max" in results:
        ax.axhline(y=results["brake_T_max"], color="r", linestyle="--", label="T_br_max", alpha=0.7)
        ax.legend(loc="best")
    ax.set_ylabel("Torque (Nm)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Brake Torque")
    plot_idx += 1
    
    # 11. Forces
    ax = axes[plot_idx]
    ax.plot(time, results["tire_force"], "b-", label="Tire", linewidth=2)
    ax.plot(time, results["drag_force"], "g-", label="Drag", linewidth=2)
    ax.plot(time, results["rolling_force"], "orange", label="Rolling", linewidth=2)
    ax.plot(time, results["grade_force"], "brown", label="Grade", linewidth=2)
    ax.plot(time, results["net_force"], "k-", label="Net", linewidth=2.5)
    ax.set_ylabel("Force (N)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Forces")
    plot_idx += 1
    
    # 12. Wheel Angular Speed and Slip
    ax = axes[plot_idx]
    ax_twin = ax.twinx()
    ax.plot(time, results["wheel_omega"], "blue", linewidth=2, label="Wheel ω")
    ax.set_ylabel("Wheel Angular Speed (rad/s)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax_twin.plot(time, results["slip_ratio"], "red", linewidth=2, label="Slip Ratio", linestyle="--")
    ax_twin.set_ylabel("Slip Ratio", color="red")
    ax_twin.tick_params(axis="y", labelcolor="red")
    ax.grid(True, alpha=0.3)
    ax.set_title("Wheel Angular Speed and Slip Ratio")
    plot_idx += 1
    
    # 13. Position
    ax = axes[plot_idx]
    ax.plot(time, results["position"], "green", linewidth=2)
    ax.set_ylabel("Position (m)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Vehicle Position")
    plot_idx += 1
    
    # 14. Status Flags
    ax = axes[plot_idx]
    ax.plot(time, results["held_by_brakes"].astype(float), "r-", label="Held by Brakes", linewidth=2)
    ax.plot(time, results["coupling_enabled"].astype(float), "b-", label="Coupling Enabled", linewidth=2)
    ax.set_ylabel("Status (0/1)")
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel("Time (s)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title("Status Flags")
    
    plt.tight_layout()
    
    if plot_path is not None:
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    else:
        plt.show()


def simulate_from_trip_data(
    fitted_params: FittedVehicleParams,
    trip_data_path: Path,
    output_path: Path,
    dt: float = 0.1,
    plot: bool = False,
    plot_path: Path | None = None,
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
        # Motor states
        "motor_omega": np.zeros(n),
        "motor_current": np.zeros(n),
        "V_cmd": np.zeros(n),
        "back_emf_voltage": np.zeros(n),
        "i_limit": np.zeros(n),
        "drive_torque": np.zeros(n),
        "motor_shaft_torque": np.zeros(n),
        # Brake
        "brake_torque": np.zeros(n),
        # Forces
        "tire_force": np.zeros(n),
        "drag_force": np.zeros(n),
        "rolling_force": np.zeros(n),
        "grade_force": np.zeros(n),
        "net_force": np.zeros(n),
        # Wheel
        "wheel_speed": np.zeros(n),  # Linear wheel speed (m/s)
        "wheel_omega": np.zeros(n),  # Wheel angular speed (rad/s)
        "slip_ratio": np.zeros(n),
        # Position
        "position": np.zeros(n),
        # Status flags
        "held_by_brakes": np.zeros(n, dtype=bool),
        "coupling_enabled": np.zeros(n, dtype=bool),
        # Limits (for plotting)
        "V_max": plant_params.motor.V_max,
        "i_limit_static": (
            (plant_params.motor.T_max / max(plant_params.motor.K_t, 1e-9))
            if plant_params.motor.T_max is not None
            else (plant_params.motor.V_max / max(plant_params.motor.R, 1e-9))
        ),
        "T_max": plant_params.motor.T_max,
        "P_max": plant_params.motor.P_max,
        "gear_ratio": plant_params.motor.gear_ratio,
        "eta_gb": plant_params.motor.eta_gb,
        "brake_T_max": plant_params.brake.T_br_max,
    }
    
    # Store initial state
    state = plant.state
    results["speed_sim"][0] = initial_speed
    results["motor_omega"][0] = state.motor_omega
    results["motor_current"][0] = state.motor_current
    results["V_cmd"][0] = state.V_cmd
    results["back_emf_voltage"][0] = state.back_emf_voltage
    results["i_limit"][0] = state.i_limit
    results["drive_torque"][0] = state.drive_torque
    results["motor_shaft_torque"][0] = state.drive_torque / max(plant_params.motor.eta_gb * plant_params.motor.gear_ratio, 1e-9)
    results["brake_torque"][0] = state.brake_torque
    results["tire_force"][0] = state.tire_force
    results["drag_force"][0] = state.drag_force
    results["rolling_force"][0] = state.rolling_force
    results["grade_force"][0] = state.grade_force
    results["net_force"][0] = state.net_force
    results["wheel_speed"][0] = state.wheel_speed
    # Compute wheel angular speed from motor omega and gear ratio
    results["wheel_omega"][0] = state.motor_omega / plant_params.motor.gear_ratio
    results["slip_ratio"][0] = state.slip_ratio
    results["position"][0] = state.position
    results["held_by_brakes"][0] = state.held_by_brakes
    results["coupling_enabled"][0] = state.coupling_enabled
    
    for t in range(n - 1):
        # Action: positive = throttle, negative = brake
        action = throttle[t] if throttle[t] > 0 else -brake[t]
        
        # Step simulation
        state = plant.step(action, dt=dt, grade_rad=float(grade[t]))
        
        # Store results
        results["time"][t + 1] = results["time"][t] + dt
        results["speed_sim"][t + 1] = state.speed
        results["acceleration_sim"][t + 1] = state.acceleration
        results["motor_omega"][t + 1] = state.motor_omega
        results["motor_current"][t + 1] = state.motor_current
        results["V_cmd"][t + 1] = state.V_cmd
        results["back_emf_voltage"][t + 1] = state.back_emf_voltage
        results["i_limit"][t + 1] = state.i_limit
        results["drive_torque"][t + 1] = state.drive_torque
        results["motor_shaft_torque"][t + 1] = state.drive_torque / max(plant_params.motor.eta_gb * plant_params.motor.gear_ratio, 1e-9)
        results["brake_torque"][t + 1] = state.brake_torque
        results["tire_force"][t + 1] = state.tire_force
        results["drag_force"][t + 1] = state.drag_force
        results["rolling_force"][t + 1] = state.rolling_force
        results["grade_force"][t + 1] = state.grade_force
        results["net_force"][t + 1] = state.net_force
        results["wheel_speed"][t + 1] = state.wheel_speed
        # Compute wheel angular speed from motor omega and gear ratio
        results["wheel_omega"][t + 1] = state.motor_omega / plant_params.motor.gear_ratio
        results["slip_ratio"][t + 1] = state.slip_ratio
        results["position"][t + 1] = state.position
        results["held_by_brakes"][t + 1] = state.held_by_brakes
        results["coupling_enabled"][t + 1] = state.coupling_enabled
    
    # Save results
    np.savez(output_path, **results)
    print(f"Simulation results saved to {output_path}")
    
    # Print summary
    speed_error = np.abs(results["speed_sim"] - results["speed_measured"])
    print(f"Mean speed error: {np.mean(speed_error):.4f} m/s")
    print(f"Max speed error: {np.max(speed_error):.4f} m/s")
    print(f"RMSE: {np.sqrt(np.mean(speed_error**2)):.4f} m/s")
    
    # Plot if requested
    if plot:
        plot_simulation_results(results, plot_path)


def simulate_synthetic(
    fitted_params: FittedVehicleParams,
    output_path: Path,
    duration: float = 60.0,
    dt: float = 0.1,
    plot: bool = False,
    plot_path: Path | None = None,
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
    grade = np.zeros(n)  # Flat road for synthetic
    
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
        "grade": grade,
        # Motor states
        "motor_omega": np.zeros(n),
        "motor_current": np.zeros(n),
        "V_cmd": np.zeros(n),
        "back_emf_voltage": np.zeros(n),
        "i_limit": np.zeros(n),
        "drive_torque": np.zeros(n),
        "motor_shaft_torque": np.zeros(n),
        # Brake
        "brake_torque": np.zeros(n),
        # Forces
        "tire_force": np.zeros(n),
        "drag_force": np.zeros(n),
        "rolling_force": np.zeros(n),
        "grade_force": np.zeros(n),
        "net_force": np.zeros(n),
        # Wheel
        "wheel_speed": np.zeros(n),  # Linear wheel speed (m/s)
        "wheel_omega": np.zeros(n),  # Wheel angular speed (rad/s)
        "slip_ratio": np.zeros(n),
        # Position
        "position": np.zeros(n),
        # Status flags
        "held_by_brakes": np.zeros(n, dtype=bool),
        "coupling_enabled": np.zeros(n, dtype=bool),
        # Limits (for plotting)
        "V_max": plant_params.motor.V_max,
        "i_limit_static": (
            (plant_params.motor.T_max / max(plant_params.motor.K_t, 1e-9))
            if plant_params.motor.T_max is not None
            else (plant_params.motor.V_max / max(plant_params.motor.R, 1e-9))
        ),
        "T_max": plant_params.motor.T_max,
        "P_max": plant_params.motor.P_max,
        "gear_ratio": plant_params.motor.gear_ratio,
        "eta_gb": plant_params.motor.eta_gb,
        "brake_T_max": plant_params.brake.T_br_max,
    }
    
    # Store initial state
    state = plant.state
    results["motor_omega"][0] = state.motor_omega
    results["motor_current"][0] = state.motor_current
    results["V_cmd"][0] = state.V_cmd
    results["back_emf_voltage"][0] = state.back_emf_voltage
    results["i_limit"][0] = state.i_limit
    results["drive_torque"][0] = state.drive_torque
    results["motor_shaft_torque"][0] = state.drive_torque / max(plant_params.motor.eta_gb * plant_params.motor.gear_ratio, 1e-9)
    results["brake_torque"][0] = state.brake_torque
    results["tire_force"][0] = state.tire_force
    results["drag_force"][0] = state.drag_force
    results["rolling_force"][0] = state.rolling_force
    results["grade_force"][0] = state.grade_force
    results["net_force"][0] = state.net_force
    results["wheel_speed"][0] = state.wheel_speed
    # Compute wheel angular speed from motor omega and gear ratio
    results["wheel_omega"][0] = state.motor_omega / plant_params.motor.gear_ratio
    results["slip_ratio"][0] = state.slip_ratio
    results["position"][0] = state.position
    results["held_by_brakes"][0] = state.held_by_brakes
    results["coupling_enabled"][0] = state.coupling_enabled
    
    for t in range(n - 1):
        action = throttle[t] if throttle[t] > 0 else -brake[t]
        state = plant.step(action, dt=dt, grade_rad=float(grade[t]))
        
        results["time"][t + 1] = results["time"][t] + dt
        results["speed"][t + 1] = state.speed
        results["acceleration"][t + 1] = state.acceleration
        results["motor_omega"][t + 1] = state.motor_omega
        results["motor_current"][t + 1] = state.motor_current
        results["V_cmd"][t + 1] = state.V_cmd
        results["back_emf_voltage"][t + 1] = state.back_emf_voltage
        results["i_limit"][t + 1] = state.i_limit
        results["drive_torque"][t + 1] = state.drive_torque
        results["motor_shaft_torque"][t + 1] = state.drive_torque / max(plant_params.motor.eta_gb * plant_params.motor.gear_ratio, 1e-9)
        results["brake_torque"][t + 1] = state.brake_torque
        results["tire_force"][t + 1] = state.tire_force
        results["drag_force"][t + 1] = state.drag_force
        results["rolling_force"][t + 1] = state.rolling_force
        results["grade_force"][t + 1] = state.grade_force
        results["net_force"][t + 1] = state.net_force
        results["wheel_speed"][t + 1] = state.wheel_speed
        # Compute wheel angular speed from motor omega and gear ratio
        results["wheel_omega"][t + 1] = state.motor_omega / plant_params.motor.gear_ratio
        results["slip_ratio"][t + 1] = state.slip_ratio
        results["position"][t + 1] = state.position
        results["held_by_brakes"][t + 1] = state.held_by_brakes
        results["coupling_enabled"][t + 1] = state.coupling_enabled
    
    # Save results
    np.savez(output_path, **results)
    print(f"Simulation results saved to {output_path}")
    
    # Plot if requested
    if plot:
        plot_simulation_results(results, plot_path)


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
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate comprehensive plots of all simulation states",
    )
    parser.add_argument(
        "--plot-output", type=Path, default=None,
        help="Output path for plot figure (default: <output_path>.png)",
    )
    
    args = parser.parse_args()
    
    if not args.params.exists():
        print(f"Error: Parameters file not found: {args.params}")
        return 1
    
    # Load fitted parameters
    fitted_params = load_fitted_params(args.params)
    print(f"Loaded parameters from {args.params}")
    
    # Determine plot path
    plot_path = args.plot_output
    if args.plot and plot_path is None:
        plot_path = args.output.with_suffix(".png")
    
    # Run simulation
    if args.trip_data is not None:
        if not args.trip_data.exists():
            print(f"Error: Trip data file not found: {args.trip_data}")
            return 1
        simulate_from_trip_data(
            fitted_params, args.trip_data, args.output, 
            dt=args.dt, plot=args.plot, plot_path=plot_path
        )
    else:
        simulate_synthetic(
            fitted_params, args.output, 
            duration=args.duration, dt=args.dt,
            plot=args.plot, plot_path=plot_path
        )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
