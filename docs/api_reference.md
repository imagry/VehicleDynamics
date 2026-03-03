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
