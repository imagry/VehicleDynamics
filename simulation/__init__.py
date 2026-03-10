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
)
from simulation.inverse_dynamics import (
    AnalyticInverseFeedforward,
    InverseFeedforwardResult,
    compute_feedforward_action,
)
from simulation.feedforward_controller import (
    FeedforwardController,
    FeedforwardProfileResult,
    FeedforwardClosedLoopResult,
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
    "AnalyticInverseFeedforward",
    "InverseFeedforwardResult",
    "compute_feedforward_action",
    "FeedforwardController",
    "FeedforwardProfileResult",
    "FeedforwardClosedLoopResult",
]
