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

### `simulation.inverse_dynamics`

#### `AnalyticInverseFeedforward`

Closed-form feedforward inverse that maps target acceleration to signed action.

**Methods:**
- `compute_action(target_accel: float, speed: float, grade_rad: float | None = None) -> InverseFeedforwardResult`
- `__call__(target_accel: float, speed: float, grade_rad: float | None = None) -> InverseFeedforwardResult`

**Notes:**
- Uses algebraic inversion only (no search/bisection).
- Does not enforce internal feasibility clipping from physical limits.
- Applies only final command clipping to `[-1, 1]`.
- `P_max` is not used in inverse current-span scaling.

#### `InverseFeedforwardResult`

Result payload for inverse feedforward queries.

**Key fields:**
- `raw_action`: Unconstrained analytic command.
- `action`: Final clipped command in `[-1, 1]`.
- `was_clipped`: Whether final clipping changed the command.
- `mode`: Inversion branch (`drive` or `brake`).

#### `compute_feedforward_action`

One-shot convenience wrapper:
- `compute_feedforward_action(target_accel: float, speed: float, params: ExtendedPlantParams, grade_rad: float | None = None) -> InverseFeedforwardResult`

### `simulation.feedforward_controller`

#### `FeedforwardController`

Profile-level runtime wrapper around the analytic inverse model.

**Methods:**
- `compute_action_profile(target_accel_profile, speed_profile, grade_profile=None) -> FeedforwardProfileResult`
- `rollout_action_profile(target_accel_profile, initial_speed, dt, grade_profile=None, substeps=1) -> FeedforwardClosedLoopResult`

#### `FeedforwardProfileResult`

Batch mapping output with:
- `raw_action`, `action`, `was_clipped`, `mode`
- aligned `target_accel`, `speed`, `grade_rad`

#### `FeedforwardClosedLoopResult`

Closed-loop rollout output with:
- feedforward traces (`raw_action`, `action`, `was_clipped`, `mode`)
- realized states (`speed`, `acceleration`)

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
- `flip_grade_sign_from_data`: Use `-angle` from dataset during fitting
- `mask_loss_for_abs_grade_gt_2deg`: Mask loss where `|grade| > 2 deg`

#### `FittedVehicleParams`

**Load behavior:**
- `from_dict` / `load` accepts multiple JSON payload styles:
    - direct fitted-parameter dictionaries
    - checkpoint dictionaries with nested `params`
    - fitter config dictionaries with `*_init` fields
    - GUI settings dictionaries with `parameters.<name>.init`

### `fitting.feedforward_gui`

#### `FeedforwardComparisonGUI`

Tkinter GUI for feedforward action comparison on parsed trips.

**Core capabilities:**
- Open-loop and closed-loop comparison modes
- GT acceleration Butterworth LPF cutoff control
- Separate throttle/brake feedforward gain scaling
- Full-state diagnostics including powers, currents, voltages, and longitudinal forces
- Defaults initialized from `fitting/gui_settings.json` (`parameters.<name>.init`)

**Related helpers:**
- `run_open_loop_ff_comparison(...)`
- `run_closed_loop_ff_comparison(...)`
- `compute_open_loop_metrics(...)`
- `compute_closed_loop_metrics(...)`

## Data Handling

### `data.fetch`

#### `TripFetcher`

Downloads trip data from S3.

**Methods:**
- `run() -> None`

### `data.fetch_gui`

#### `FetchTripsGUI`

Tkinter GUI for configuring and running trip downloads.

**Entry Point:**
- `main() -> None`

### `data.parsing`

#### `TripDatasetParser`

Parses raw trip folders into synchronized datasets.

**Methods:**
- `parse() -> dict[str, dict[str, np.ndarray]]`
- `save() -> Path`

### `data.parsing_gui`

#### `TripParsingGUI`

Tkinter GUI for configuring and running trip parsing.

**Entry Point:**
- `main() -> None`

## Utilities

### `utils.randomization`

#### `CenteredRandomizationConfig`

Creates parameter randomization ranges centered on fitted parameters.

**Methods:**
- `from_fitted_params(fitted: FittedVehicleParams, spread_pct: float = 0.1) -> CenteredRandomizationConfig`
- `to_extended_randomization_dict() -> Dict`

## Command-Line Scripts

### `scripts/fetch_trips_gui.py`

Launch the trip fetching GUI.

**Usage:**
```bash
python scripts/fetch_trips_gui.py
```

### `scripts/parse_trips_gui.py`

Launch the trip parsing GUI.

**Usage:**
```bash
python scripts/parse_trips_gui.py
```

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
- If `--plot` is enabled, a PNG file with 14 subplots showing all internal states

**Plot Contents:**
The plot includes 14 subplots in a single column, all sharing the x-axis (time):
1. Actuations (throttle, brake)
2. Vehicle Speed
3. Acceleration
4. Road Grade
5. Motor Angular Speed
6. Motor Current (with limits)
7. Motor Voltage (commanded, back-EMF, limits)
8. Drive Torque at Wheel (with motor/wheel limit overlays)
9. Motor Power (with limits)
10. Brake Torque (with limits)
11. Forces (tire, drag, rolling, grade, net)
12. Wheel Angular Speed and Slip Ratio
13. Vehicle Position
14. Status Flags (held by brakes, coupling enabled)

### `scripts/feedforward_trip.py`

Generate feedforward traces (`raw_action` and clipped `action`) from a parsed trip file.

**Usage:**
```bash
python scripts/feedforward_trip.py \
    --trip-data <trip_data.pt> \
    --output <feedforward_traces.npz> \
    [--params <fitted_params.json>] \
    [--trip-id <trip_key>] \
    [--dt <seconds>] \
    [--angle-is-deg] \
    [--substeps <n>]
```

### `scripts/feedforward_compare_gui.py`

Launch the feedforward comparison GUI.

**Usage:**
```bash
python scripts/feedforward_compare_gui.py
```

**Console entry point:**
```bash
feedforward-compare-gui
```

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
Same comprehensive 14-subplot visualization as `scripts/simulate_trip.py` (see above).

### `examples/feedforward_gui_usage.py`

Launches the feedforward comparison GUI from the examples directory.

**Usage:**
```bash
python examples/feedforward_gui_usage.py
```
