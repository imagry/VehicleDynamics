#!/usr/bin/env python3
"""Analytic inverse feedforward example.

This example maps target acceleration to signed actuation using closed-form
inversion and runs the command through the forward plant.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation import AnalyticInverseFeedforward, ExtendedPlant, ExtendedPlantParams


def main() -> None:
    params = ExtendedPlantParams()
    inverse = AnalyticInverseFeedforward(params)
    plant = ExtendedPlant(params)

    dt = 0.1
    target_profile = ([0.8] * 30) + ([0.0] * 20) + ([-1.2] * 30)

    plant.reset(speed=8.0)
    print("t[s]  a_target  raw_u   u_clip  speed   accel   mode")

    for k, target_accel in enumerate(target_profile):
        ff = inverse.compute_action(
            target_accel=target_accel,
            speed=plant.speed,
        )
        state = plant.step(action=ff.action, dt=dt)

        if k % 10 == 0:
            print(
                f"{k * dt:4.1f}  {target_accel:8.3f}  {ff.raw_action:6.3f}  "
                f"{ff.action:6.3f}  {state.speed:6.3f}  {state.acceleration:6.3f}  {ff.mode}"
            )


if __name__ == "__main__":
    main()
