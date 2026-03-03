"""Integration tests for end-to-end simulation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from fitting.fitter import FittedVehicleParams, VehicleParamFitter, FitterConfig
from simulation.dynamics import ExtendedPlant, ExtendedPlantParams, MotorParams, BrakeParams, BodyParams, WheelParams, CreepParams


class TestEndToEndSimulation:
    """End-to-end simulation tests."""
    
    def test_fit_and_simulate(self) -> None:
        """Test fitting parameters and then simulating with them."""
        # Create synthetic trip data
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test_data.pt"
            
            # Create simple synthetic trip
            n = 100
            dt = 0.1
            time = np.arange(n) * dt
            speed = 10.0 + 2.0 * np.sin(2 * np.pi * time / 10.0)  # Oscillating speed
            throttle = np.clip((speed - 8.0) / 5.0, 0, 1) * 100  # Simple throttle mapping
            brake = np.clip((8.0 - speed) / 5.0, 0, 1) * 100  # Simple brake mapping
            acceleration = np.gradient(speed, dt)
            angle = np.zeros(n)
            
            trip_data = {
                "test_trip": {
                    "speed": speed,
                    "acceleration": acceleration,
                    "throttle": throttle,
                    "brake": brake,
                    "angle": angle,
                    "time": time,
                },
                "metadata": {
                    "dt": dt,
                    "num_valid_trips": 1,
                },
            }
            
            torch.save(trip_data, data_path)
            
            # Fit parameters
            config = FitterConfig(
                segments_per_batch=1,
                num_epochs=1,
                max_iter=5,  # Quick test
                validation_fraction=0.0,
            )
            
            fitter = VehicleParamFitter(config)
            
            # Fit with very loose bounds for quick convergence
            fitted = fitter.fit(data_path, verbose=False)
            
            # Verify fitted params are reasonable
            assert fitted.mass > 0
            assert fitted.motor_V_max > 0
            assert fitted.motor_R > 0
            
            # Simulate with fitted params
            motor = MotorParams(
                R=fitted.motor_R,
                K_e=fitted.motor_K,
                K_t=fitted.motor_K,
                b=fitted.motor_b,
                J=fitted.motor_J,
                V_max=fitted.motor_V_max,
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
            
            plant_params = ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel, creep=creep)
            plant = ExtendedPlant(plant_params)
            
            # Run simulation
            plant.reset(speed=speed[0])
            simulated_speeds = [speed[0]]
            
            for t in range(n - 1):
                action = throttle[t] / 100.0 if throttle[t] > 0 else -brake[t] / 100.0
                state = plant.step(action, dt=dt)
                simulated_speeds.append(state.speed)
            
            # Verify simulation ran
            assert len(simulated_speeds) == n
            assert all(s >= 0 for s in simulated_speeds)  # No negative speeds
