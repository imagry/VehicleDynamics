#!/usr/bin/env python3
"""Example of creating a custom vehicle configuration.

This example shows how to create a vehicle with custom parameters and run simulations.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def create_electric_sedan() -> ExtendedPlantParams:
    """Create parameters for a typical electric sedan."""
    motor = MotorParams(
        R=0.15,  # Lower resistance for better efficiency
        K_e=0.25,
        K_t=0.25,
        b=5e-4,  # Lower friction
        J=2e-3,
        V_max=400.0,
        T_max=300.0,  # Max torque limit
        P_max=150000.0,  # 150 kW max power
        min_current_A=6.0,
        gear_ratio=9.5,
        eta_gb=0.95,  # High efficiency gearbox
    )
    
    brake = BrakeParams(
        T_br_max=18000.0,  # Strong brakes
        p_br=1.3,
        tau_br=0.06,  # Fast brake response
        mu=0.95,  # High friction tires
    )
    
    body = BodyParams(
        mass=2000.0,  # Heavier sedan
        drag_area=0.6,  # Lower drag (better aerodynamics)
        rolling_coeff=0.009,  # Low rolling resistance tires
    )
    
    wheel = WheelParams(
        radius=0.35,  # Larger wheels
        inertia=2.0,
    )
    
    return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel)


def create_lightweight_ev() -> ExtendedPlantParams:
    """Create parameters for a lightweight electric vehicle."""
    motor = MotorParams(
        R=0.25,
        K_e=0.15,
        K_t=0.15,
        b=8e-4,
        J=1e-3,
        V_max=350.0,
        min_current_A=4.0,
        gear_ratio=12.0,
        eta_gb=0.92,
    )
    
    brake = BrakeParams(
        T_br_max=12000.0,
        p_br=1.2,
        tau_br=0.08,
        mu=0.85,
    )
    
    body = BodyParams(
        mass=1200.0,  # Lightweight
        drag_area=0.5,
        rolling_coeff=0.008,
    )
    
    wheel = WheelParams(
        radius=0.30,
        inertia=1.0,
    )
    
    return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel)


def simulate_acceleration_test(plant: ExtendedPlant, dt: float = 0.1) -> dict:
    """Run a 0-100 km/h acceleration test."""
    plant.reset(speed=0.0)
    
    speeds = []
    times = []
    t = 0.0
    
    target_speed = 100.0 / 3.6  # 100 km/h in m/s
    
    while plant.speed < target_speed and t < 30.0:  # Max 30 seconds
        state = plant.step(1.0, dt=dt)  # Full throttle
        speeds.append(state.speed)
        times.append(t)
        t += dt
    
    return {
        "times": np.array(times),
        "speeds": np.array(speeds),
        "final_time": t,
    }


def main() -> None:
    """Compare two vehicle configurations."""
    print("Electric Sedan Configuration:")
    sedan_params = create_electric_sedan()
    sedan_plant = ExtendedPlant(sedan_params)
    sedan_results = simulate_acceleration_test(sedan_plant)
    print(f"  0-100 km/h: {sedan_results['final_time']:.2f} seconds")
    
    print("\nLightweight EV Configuration:")
    lightweight_params = create_lightweight_ev()
    lightweight_plant = ExtendedPlant(lightweight_params)
    lightweight_results = simulate_acceleration_test(lightweight_plant)
    print(f"  0-100 km/h: {lightweight_results['final_time']:.2f} seconds")


if __name__ == "__main__":
    main()
