"""Vehicle dynamics simulation module.

This module provides the ExtendedPlant model for simulating vehicle longitudinal dynamics,
including DC motor, brakes, wheel dynamics, and all related parameter classes.
"""

from simulation.dynamics import (
    GRAVITY,
    DEFAULT_AIR_DENSITY,
    ExtendedPlant,
    ExtendedPlantParams,
    ExtendedPlantState,
    MotorParams,
    BrakeParams,
    BodyParams,
    WheelParams,
    CreepParams,
)

__all__ = [
    "GRAVITY",
    "DEFAULT_AIR_DENSITY",
    "ExtendedPlant",
    "ExtendedPlantParams",
    "ExtendedPlantState",
    "MotorParams",
    "BrakeParams",
    "BodyParams",
    "WheelParams",
    "CreepParams",
]
