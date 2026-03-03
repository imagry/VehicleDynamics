# Getting Started

This guide will help you get started with the vehicle simulation repository.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Install from Source

```bash
# Clone or navigate to the repository
cd simulation_repo

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Simulation

Run a simple simulation with default vehicle parameters:

```python
from simulation.dynamics import ExtendedPlant, ExtendedPlantParams

# Use default parameters
params = ExtendedPlantParams()
plant = ExtendedPlant(params)

# Initialize and simulate
plant.reset(speed=0.0)
for _ in range(100):
    state = plant.step(0.5, dt=0.1)  # 50% throttle
    print(f"Speed: {state.speed:.2f} m/s")
```

See `examples/basic_simulation.py` for a complete example.

### 2. Fit Parameters from Data

Fit vehicle parameters from trip data:

```bash
python scripts/fit_params.py \
    --data data/processed/vehicle/all_trips_data.pt \
    --output fitted_params.json
```

Or use the GUI:

```bash
python examples/gui_usage.py
```

### 3. Run Simulation with Fitted Parameters

```bash
python scripts/simulate_trip.py \
    --params fitted_params.json \
    --trip-data data/processed/vehicle/all_trips_data.pt \
    --output simulation_results.npz
```

## Next Steps

- Read the [Dynamics Model Documentation](dynamics_model.md) to understand the physics
- Check the [Fitting Guide](fitting_guide.md) for parameter fitting workflows
- Explore the [API Reference](api_reference.md) for detailed API documentation
- Try the examples in the `examples/` directory
