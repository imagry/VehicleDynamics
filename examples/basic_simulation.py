#!/usr/bin/env python3
"""Basic simulation example.

This example demonstrates how to create a vehicle model and run a simple simulation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.dynamics import (
    ExtendedPlant,
    ExtendedPlantParams,
    MotorParams,
    BrakeParams,
    BodyParams,
    WheelParams,
)


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
    throttle_pct = np.array(results["throttle"]) * 100
    brake_pct = np.array(results["brake"]) * 100
    ax.plot(time, throttle_pct, "g-", label="Throttle", linewidth=2)
    ax.plot(time, brake_pct, "r-", label="Brake", linewidth=2)
    ax.set_ylabel("Actuation (%)")
    ax.set_ylim([0, 100])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title("Actuations")
    plot_idx += 1
    
    # 2. Speed
    ax = axes[plot_idx]
    ax.plot(time, results["speed"], "b-", label="Speed", linewidth=2)
    ax.set_ylabel("Speed (m/s)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title("Vehicle Speed")
    plot_idx += 1
    
    # 3. Acceleration
    ax = axes[plot_idx]
    ax.plot(time, results["acceleration"], "g-", label="Acceleration", linewidth=2)
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


def main() -> None:
    """Run a basic simulation example."""
    parser = argparse.ArgumentParser(
        description="Run a basic vehicle simulation example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate comprehensive plots of all simulation states",
    )
    parser.add_argument(
        "--plot-output", type=Path, default=None,
        help="Output path for plot figure (default: simulation_results.png)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Simulation duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1,
        help="Time step in seconds (default: 0.1)",
    )
    
    args = parser.parse_args()
    
    # Create vehicle parameters
    motor = MotorParams(
        R=0.2,  # armature resistance (Ω)
        K_e=0.2,  # back-EMF constant (V/(rad/s))
        K_t=0.2,  # torque constant (Nm/A)
        b=1e-3,  # viscous friction
        J=1e-3,  # rotor inertia
        V_max=400.0,  # max voltage
        min_current_A=5.0,
        gear_ratio=10.0,
        eta_gb=0.9,
    )
    
    brake = BrakeParams(
        T_br_max=15000.0,
        p_br=1.2,
        tau_br=0.08,
        kappa_c=0.08,
        mu=0.9,
    )
    
    body = BodyParams(
        mass=1800.0,  # kg
        drag_area=0.65,  # m²
        rolling_coeff=0.011,
    )
    
    wheel = WheelParams(
        radius=0.346,  # m
        inertia=1.5,  # kg·m²
    )
    
    params = ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel)
    
    # Create plant
    plant = ExtendedPlant(params)
    
    # Initialize at rest
    plant.reset(speed=0.0, position=0.0)
    
    # Simulate
    dt = args.dt
    duration = args.duration
    n_steps = int(duration / dt)
    
    # Initialize results dictionary
    results = {
        "time": np.zeros(n_steps),
        "speed": np.zeros(n_steps),
        "acceleration": np.zeros(n_steps),
        "throttle": np.zeros(n_steps),
        "brake": np.zeros(n_steps),
        "grade": np.zeros(n_steps),
        # Motor states
        "motor_omega": np.zeros(n_steps),
        "motor_current": np.zeros(n_steps),
        "V_cmd": np.zeros(n_steps),
        "back_emf_voltage": np.zeros(n_steps),
        "i_limit": np.zeros(n_steps),
        "drive_torque": np.zeros(n_steps),
        "motor_shaft_torque": np.zeros(n_steps),
        # Brake
        "brake_torque": np.zeros(n_steps),
        # Forces
        "tire_force": np.zeros(n_steps),
        "drag_force": np.zeros(n_steps),
        "rolling_force": np.zeros(n_steps),
        "grade_force": np.zeros(n_steps),
        "net_force": np.zeros(n_steps),
        # Wheel
        "wheel_speed": np.zeros(n_steps),
        "wheel_omega": np.zeros(n_steps),
        "slip_ratio": np.zeros(n_steps),
        # Position
        "position": np.zeros(n_steps),
        # Status flags
        "held_by_brakes": np.zeros(n_steps, dtype=bool),
        "coupling_enabled": np.zeros(n_steps, dtype=bool),
        # Limits (for plotting)
        "V_max": params.motor.V_max,
        "i_limit_static": (
            (params.motor.T_max / max(params.motor.K_t, 1e-9))
            if params.motor.T_max is not None
            else (params.motor.V_max / max(params.motor.R, 1e-9))
        ),
        "T_max": params.motor.T_max,
        "P_max": params.motor.P_max,
        "gear_ratio": params.motor.gear_ratio,
        "eta_gb": params.motor.eta_gb,
        "brake_T_max": params.brake.T_br_max,
    }
    
    # Store initial state
    state = plant.state
    # Determine initial action
    if 0 < n_steps // 2:
        initial_throttle = 0.5
        initial_brake = 0.0
    else:
        initial_throttle = 0.0
        initial_brake = 0.5
    results["throttle"][0] = initial_throttle
    results["brake"][0] = initial_brake
    results["grade"][0] = 0.0
    results["motor_omega"][0] = state.motor_omega
    results["motor_current"][0] = state.motor_current
    results["V_cmd"][0] = state.V_cmd
    results["back_emf_voltage"][0] = state.back_emf_voltage
    results["i_limit"][0] = state.i_limit
    results["drive_torque"][0] = state.drive_torque
    results["motor_shaft_torque"][0] = state.drive_torque / max(params.motor.eta_gb * params.motor.gear_ratio, 1e-9)
    results["brake_torque"][0] = state.brake_torque
    results["tire_force"][0] = state.tire_force
    results["drag_force"][0] = state.drag_force
    results["rolling_force"][0] = state.rolling_force
    results["grade_force"][0] = state.grade_force
    results["net_force"][0] = state.net_force
    results["wheel_speed"][0] = state.wheel_speed
    results["wheel_omega"][0] = state.motor_omega / params.motor.gear_ratio
    results["slip_ratio"][0] = state.slip_ratio
    results["position"][0] = state.position
    results["held_by_brakes"][0] = state.held_by_brakes
    results["coupling_enabled"][0] = state.coupling_enabled
    
    for t in range(n_steps - 1):
        # Simple control: accelerate for first half, then brake
        if t < n_steps // 2:
            action = 0.5  # 50% throttle
            throttle_cmd = 0.5
            brake_cmd = 0.0
        else:
            action = -0.5  # 50% brake
            throttle_cmd = 0.0
            brake_cmd = 0.5
        
        state = plant.step(action, dt=dt)
        
        # Store results
        results["time"][t + 1] = results["time"][t] + dt
        results["speed"][t + 1] = state.speed
        results["acceleration"][t + 1] = state.acceleration
        results["throttle"][t + 1] = throttle_cmd
        results["brake"][t + 1] = brake_cmd
        results["grade"][t + 1] = 0.0  # Flat road in this example
        results["motor_omega"][t + 1] = state.motor_omega
        results["motor_current"][t + 1] = state.motor_current
        results["V_cmd"][t + 1] = state.V_cmd
        results["back_emf_voltage"][t + 1] = state.back_emf_voltage
        results["i_limit"][t + 1] = state.i_limit
        results["drive_torque"][t + 1] = state.drive_torque
        results["motor_shaft_torque"][t + 1] = state.drive_torque / max(params.motor.eta_gb * params.motor.gear_ratio, 1e-9)
        results["brake_torque"][t + 1] = state.brake_torque
        results["tire_force"][t + 1] = state.tire_force
        results["drag_force"][t + 1] = state.drag_force
        results["rolling_force"][t + 1] = state.rolling_force
        results["grade_force"][t + 1] = state.grade_force
        results["net_force"][t + 1] = state.net_force
        results["wheel_speed"][t + 1] = state.wheel_speed
        results["wheel_omega"][t + 1] = state.motor_omega / params.motor.gear_ratio
        results["slip_ratio"][t + 1] = state.slip_ratio
        results["position"][t + 1] = state.position
        results["held_by_brakes"][t + 1] = state.held_by_brakes
        results["coupling_enabled"][t + 1] = state.coupling_enabled
    
    # Print results
    print("Simulation Results:")
    print(f"Final speed: {results['speed'][-1]:.2f} m/s ({results['speed'][-1]*3.6:.2f} km/h)")
    print(f"Final position: {results['position'][-1]:.2f} m")
    print(f"Max speed: {max(results['speed']):.2f} m/s ({max(results['speed'])*3.6:.2f} km/h)")
    print(f"Max acceleration: {max(results['acceleration']):.2f} m/s²")
    
    # Plot if requested
    if args.plot:
        plot_path = args.plot_output
        if plot_path is None:
            plot_path = Path("simulation_results.png")
        plot_simulation_results(results, plot_path)


if __name__ == "__main__":
    main()
