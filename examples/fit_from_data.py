#!/usr/bin/env python3
"""Example of fitting vehicle parameters from trip data.

This example shows how to fit vehicle dynamics parameters from real trip data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fitting.fitter import VehicleParamFitter, FitterConfig


def main() -> None:
    """Fit parameters from trip data."""
    # Example: Fit parameters from a trip data file
    # Replace with your actual data path
    data_path = Path("data/processed/NiroEV/NIROEV_HA_02/all_trips_data.pt")
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable to point to your trip data file.")
        return
    
    # Create fitting configuration
    config = FitterConfig(
        segments_per_batch=10,
        num_epochs=1,
        max_iter=30,
        validation_fraction=0.1,
    )
    
    # Create fitter
    fitter = VehicleParamFitter(config)
    
    # Fit parameters
    print(f"Fitting parameters from {data_path}...")
    fitted = fitter.fit(data_path, verbose=True)
    
    # Save results
    output_path = data_path.parent / "fitted_params.json"
    fitted.save(output_path)
    
    print(f"\nFitted parameters saved to {output_path}")
    print(f"Fit loss: {fitted.fit_loss:.6f}")
    print(f"R²: {fitted.r_squared:.4f}")
    print(f"Number of samples: {fitted.num_samples}")


if __name__ == "__main__":
    main()
