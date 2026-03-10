# Vehicle Dynamics Simulation

A modular Python library for simulating vehicle longitudinal dynamics, fitting parameters from real trip data, and running physics-based simulations.

## Features

- **Physics-Based Simulation**: Detailed DC motor + nonlinear brake + wheel dynamics model
- **Parameter Fitting**: Fit 18+ vehicle parameters from real trip data using trajectory optimization
- **Interactive GUI**: User-friendly interface for parameter fitting and visualization
- **Data Processing**: Tools for fetching and parsing trip data from S3 or local files
- **Data GUIs**: Separate GUIs for trip fetching and trip parsing workflows
- **Modular Design**: Clean separation between simulation, fitting, and data handling

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# Or install as package:
pip install -e .
```

### Basic Usage

```python
from simulation.dynamics import ExtendedPlant, ExtendedPlantParams

# Create vehicle model
params = ExtendedPlantParams()
plant = ExtendedPlant(params)

# Simulate
plant.reset(speed=0.0)
for _ in range(100):
    state = plant.step(0.5, dt=0.1)  # 50% throttle
    print(f"Speed: {state.speed:.2f} m/s")
```

### Fit Parameters from Data

```bash
# Using command line
python scripts/fit_params.py \
    --data data/processed/vehicle/all_trips_data.pt \
    --output fitted_params.json

# Or using GUI
python examples/gui_usage.py
```

### Data GUIs

```bash
# Fetch trips from S3 interactively
python scripts/fetch_trips_gui.py

# Parse raw trips into .pt datasets interactively
python scripts/parse_trips_gui.py
```

### Run Simulation

```bash
python scripts/simulate_trip.py \
    --params fitted_params.json \
    --trip-data data/processed/vehicle/all_trips_data.pt \
    --output results.npz
```

### Generate Feedforward Traces

```bash
python scripts/feedforward_trip.py \
    --trip-data data/processed/vehicle/all_trips_data.pt \
    --params fitted_params.json \
    --output feedforward_traces.npz
```

## Documentation

- [Getting Started](docs/getting_started.md) - Installation and quick start
- [Dynamics Model](docs/dynamics_model.md) - Physics equations and model details
- [Inverse Model](docs/inverse_model.md) - Analytic feedforward inverse (target accel to action)
- [Fitting Guide](docs/fitting_guide.md) - Parameter fitting workflow
- [API Reference](docs/api_reference.md) - Complete API documentation

## Examples

See the `examples/` directory for:
- `basic_simulation.py` - Simple simulation example
- `fit_from_data.py` - Parameter fitting example
- `gui_usage.py` - GUI usage example
- `custom_vehicle.py` - Custom vehicle configurations
- `inverse_feedforward.py` - Analytic inverse feedforward example

## Project Structure

```
simulation_repo/
├── simulation/          # Core dynamics simulation
├── fitting/             # Parameter fitting
├── data/                # Data fetching and parsing
├── utils/               # Utility functions
├── scripts/             # Command-line tools
├── tests/               # Test suite
├── examples/            # Usage examples
└── docs/                # Documentation
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_dynamics.py
pytest tests/test_fitting.py
pytest tests/test_data_parsing.py
pytest tests/test_simulation.py
```

## Requirements

- Python 3.10+
- numpy, scipy, torch
- pandas, boto3 (for data handling)
- matplotlib (for visualization)
- tkinter (for GUI, included with Python)

See `requirements.txt` for complete list.

