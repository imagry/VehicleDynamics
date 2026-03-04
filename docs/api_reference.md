# API Reference

## Core Simulation

### `simulation.dynamics`

#### `ExtendedPlant`

Main vehicle dynamics simulator.

**Methods:**
- `reset(speed: float = 0.0, position: float = 0.0) -> ExtendedPlantState`
- `step(action: float, dt: float, substeps: int = 1, grade_rad: float | None = None) -> ExtendedPlantState`

**Parameters:**
- `action`: Control input in [-1, 1], positive = throttle, negative = brake
- `dt`: Time step (seconds)
- `substeps`: Number of internal substeps for numerical stability
- `grade_rad`: Road grade in radians (None = use default from body params)

#### `ExtendedPlantParams`

Vehicle parameter container.

**Fields:**
- `motor`: `MotorParams` - Motor parameters
- `brake`: `BrakeParams` - Brake parameters
- `body`: `BodyParams` - Body parameters
- `wheel`: `WheelParams` - Wheel parameters
- `creep`: `CreepParams` - Creep torque parameters

## Parameter Fitting

### `fitting.fitter`

#### `VehicleParamFitter`

Fits vehicle parameters from trip data.

**Methods:**
- `fit(data_path: Path, verbose: bool = True, log_path: Path | None = None) -> FittedVehicleParams`
- `load_trip_data(data_path: Path) -> Dict[str, Dict[str, np.ndarray]]`

#### `FitterConfig`

Configuration for parameter fitting.

**Key Parameters:**
- `segments_per_batch`: Number of segments per optimization batch
- `num_epochs`: Number of training epochs
- `max_iter`: Maximum optimizer iterations per batch
- `validation_fraction`: Fraction of data to use for validation

## Data Handling

### `data.fetch`

#### `TripFetcher`

Downloads trip data from S3.

**Methods:**
- `run() -> None`

### `data.parsing`

#### `TripDatasetParser`

Parses raw trip folders into synchronized datasets.

**Methods:**
- `parse() -> dict[str, dict[str, np.ndarray]]`
- `save() -> Path`

## Utilities

### `utils.randomization`

#### `CenteredRandomizationConfig`

Creates parameter randomization ranges centered on fitted parameters.

**Methods:**
- `from_fitted_params(fitted: FittedVehicleParams, spread_pct: float = 0.1) -> CenteredRandomizationConfig`
- `to_extended_randomization_dict() -> Dict`

## Command-Line Scripts

### `scripts/simulate_trip.py`

Run vehicle simulation with fitted parameters.

**Usage:**
```bash
python scripts/simulate_trip.py \
    --params <fitted_params.json> \
    --output <output.npz> \
    [--trip-data <trip_data.pt>] \
    [--duration <seconds>] \
    [--dt <seconds>] \
    [--plot] \
    [--plot-output <path.png>]
```

**Arguments:**
- `--params` (required): Path to fitted parameters JSON file
- `--output` (required): Output path for simulation results (.npz file)
- `--trip-data` (optional): Path to trip data .pt file to simulate from. If not provided, runs synthetic simulation.
- `--duration` (optional): Simulation duration in seconds (default: 60.0, only for synthetic simulation)
- `--dt` (optional): Time step in seconds (default: 0.1)
- `--plot` (optional): Generate comprehensive plots of all simulation states
- `--plot-output` (optional): Output path for plot figure (default: `<output_path>.png`)

**Output:**
- `.npz` file containing all simulation states (time, speed, acceleration, motor states, forces, torques, etc.)
- If `--plot` is enabled, a PNG file with 15 subplots showing all internal states

**Plot Contents:**
The plot includes 15 subplots in a single column, all sharing the x-axis (time):
1. Actuations (throttle, brake)
2. Vehicle Speed
3. Acceleration
4. Road Grade
5. Motor Angular Speed
6. Motor Current (with limits)
7. Motor Voltage (commanded, back-EMF, limits)
8. Motor Drive Torque (with limits)
9. Motor Power (with limits)
10. Brake Torque (with limits)
11. Creep Torque
12. Forces (tire, drag, rolling, grade, net)
13. Wheel Angular Speed and Slip Ratio
14. Vehicle Position
15. Status Flags (held by brakes, coupling enabled)

## Examples

### `examples/basic_simulation.py`

Basic simulation example demonstrating vehicle dynamics.

**Usage:**
```bash
python examples/basic_simulation.py \
    [--plot] \
    [--plot-output <path.png>] \
    [--duration <seconds>] \
    [--dt <seconds>]
```

**Arguments:**
- `--plot` (optional): Generate comprehensive plots of all simulation states
- `--plot-output` (optional): Output path for plot figure (default: `simulation_results.png`)
- `--duration` (optional): Simulation duration in seconds (default: 10.0)
- `--dt` (optional): Time step in seconds (default: 0.1)

**Description:**
This example creates a vehicle with default parameters and runs a simple simulation:
- First half: 50% throttle (acceleration)
- Second half: 50% brake (deceleration)

The example demonstrates how to:
- Create vehicle parameters
- Initialize and run a simulation
- Collect and visualize all internal states

**Plot Output:**
Same comprehensive 15-subplot visualization as `scripts/simulate_trip.py` (see above).
