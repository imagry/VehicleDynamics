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

#### Visualization Options

Both the simulation script and examples support comprehensive plotting of all simulation states:

**Simulation Script (`scripts/simulate_trip.py`):**

```bash
# Generate plots with all internal states
python scripts/simulate_trip.py \
    --params fitted_params.json \
    --trip-data data/processed/vehicle/all_trips_data.pt \
    --output simulation_results.npz \
    --plot \
    --plot-output simulation_plot.png

# For synthetic simulation (without trip data)
python scripts/simulate_trip.py \
    --params fitted_params.json \
    --output simulation_results.npz \
    --plot \
    --duration 60.0 \
    --dt 0.1
```

**Basic Simulation Example (`examples/basic_simulation.py`):**

```bash
# Run with plotting enabled
python examples/basic_simulation.py --plot

# Customize duration and output path
python examples/basic_simulation.py \
    --plot \
    --plot-output my_results.png \
    --duration 20.0 \
    --dt 0.05
```

**Plot Contents:**

The plotting functionality generates a comprehensive figure with 15 subplots (all sharing the x-axis for time):

1. **Actuations** - Throttle and brake commands (0-100%)
2. **Vehicle Speed** - Simulated speed (and measured if available)
3. **Acceleration** - Simulated acceleration (and measured if available)
4. **Road Grade** - Road grade in degrees
5. **Motor Angular Speed** - Motor shaft angular speed (rad/s)
6. **Motor Current** - With current limit overlay
7. **Motor Voltage** - Commanded voltage, back-EMF voltage, and V_max limits
8. **Motor Drive Torque** - With T_max limits
9. **Motor Power** - V × I, with P_max limits
10. **Brake Torque** - With brake T_max limit
11. **Creep Torque** - Creep torque at motor shaft
12. **Forces** - Tire, drag, rolling, grade, and net forces
13. **Wheel Angular Speed and Slip Ratio** - Dual y-axis plot
14. **Vehicle Position** - Position over time
15. **Status Flags** - Held by brakes and coupling enabled (0/1)

All plots are saved as PNG files with 150 DPI resolution and can be used for detailed analysis of vehicle dynamics behavior.

## Next Steps

- Read the [Dynamics Model Documentation](dynamics_model.md) to understand the physics
- Check the [Fitting Guide](fitting_guide.md) for parameter fitting workflows
- Explore the [API Reference](api_reference.md) for detailed API documentation
- Try the examples in the `examples/` directory
