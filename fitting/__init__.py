"""Vehicle parameter fitting module.

This module provides tools for fitting vehicle dynamics parameters from
real trip data, enabling accurate simulation of specific vehicles.
"""

from fitting.fitter import (
    FittedVehicleParams,
    VehicleParamFitter,
    FitterConfig,
)

__all__ = [
    "FittedVehicleParams",
    "VehicleParamFitter",
    "FitterConfig",
]
