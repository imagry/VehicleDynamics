#!/usr/bin/env python3
"""Basic simulation example.

This example demonstrates how to create a vehicle model and run a simple simulation.
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
    CreepParams,
)


def main() -> None:
    """Run a basic simulation example."""
    # Create vehicle parameters
    motor = MotorParams(
        R=0.2,  # armature resistance (Ω)
        K_e=0.2,  # back-EMF constant (V/(rad/s))
        K_t=0.2,  # torque constant (Nm/A)
        b=1e-3,  # viscous friction
        J=1e-3,  # rotor inertia
        V_max=400.0,  # max voltage
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
    
    creep = CreepParams(
        a_max=0.5,
        v_cutoff=1.5,
        v_hold=0.08,
    )
    
    params = ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel, creep=creep)
    
    # Create plant
    plant = ExtendedPlant(params)
    
    # Initialize at rest
    plant.reset(speed=0.0, position=0.0)
    
    # Simulate for 10 seconds
    dt = 0.1
    duration = 10.0
    n_steps = int(duration / dt)
    
    speeds = []
    positions = []
    accelerations = []
    
    for t in range(n_steps):
        # Simple control: accelerate for 5s, then brake
        if t < n_steps // 2:
            action = 0.5  # 50% throttle
        else:
            action = -0.5  # 50% brake
        
        state = plant.step(action, dt=dt)
        
        speeds.append(state.speed)
        positions.append(state.position)
        accelerations.append(state.acceleration)
    
    # Print results
    print("Simulation Results:")
    print(f"Final speed: {speeds[-1]:.2f} m/s ({speeds[-1]*3.6:.2f} km/h)")
    print(f"Final position: {positions[-1]:.2f} m")
    print(f"Max speed: {max(speeds):.2f} m/s ({max(speeds)*3.6:.2f} km/h)")
    print(f"Max acceleration: {max(accelerations):.2f} m/s²")


if __name__ == "__main__":
    main()
