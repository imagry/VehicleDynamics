# Parameter Fitting Guide

## Overview

The parameter fitting process estimates vehicle dynamics parameters by minimizing the error between simulated and measured vehicle trajectories.

## Data Preparation

### Required Data Format

Trip data should be in `.pt` format (PyTorch) with the following structure:

```python
{
    "trip_id_1": {
        "speed": np.ndarray,      # m/s
        "acceleration": np.ndarray,  # m/s²
        "throttle": np.ndarray,    # 0-100
        "brake": np.ndarray,       # 0-100
        "angle": np.ndarray,       # radians (road grade)
        "time": np.ndarray,        # seconds (optional)
    },
    "trip_id_2": { ... },
    "metadata": { ... },
}
```

### Data Quality Requirements

- Minimum segment length: 50 samples (configurable)
- Valid speed range: 0-30 m/s (configurable)
- Valid acceleration range: ±5 m/s² (configurable)
- Finite values only (no NaN or Inf)

## Fitting Workflow

### 1. Prepare Data

```bash
# Fetch trip data from S3
python scripts/fetch_trips.py \
    --car NiroEV \
    --start 2024-01-01 \
    --end 2024-01-31 \
    --dest data/raw/trips

# Parse raw trips into .pt format
python scripts/parse_trips.py \
    --root data/raw/trips \
    --car NiroEV \
    --out-dir data/processed/NiroEV
```

### 2. Fit Parameters

**Command Line:**
```bash
python scripts/fit_params.py \
    --data data/processed/NiroEV/all_trips_data.pt \
    --output fitted_params.json \
    --epochs 3 \
    --batch-size 10 \
    --max-iter 50
```

**GUI:**
```bash
python examples/gui_usage.py
```

## GUI Features Reference

The GUI provides a comprehensive interface for configuring and executing parameter fitting. This section documents all available options, features, and visualizations.

### GUI Layout

The GUI is divided into two main panels:
- **Left Panel**: Configuration options, parameter inputs, and control buttons
- **Right Panel**: Real-time simulation preview plots

### Fitting Configuration Section

#### Dataset Selection
- **Dataset**: Dropdown menu that automatically scans `data/processed/` for `.pt` files
  - Selects the trip data file to use for fitting
  - Shows relative paths from the project root

#### Fitting Name
- **Fitting Name**: Text entry for naming the fitting run
  - Default: Auto-generated timestamp (e.g., `fit_20240101_120000`)
  - Results are saved to `results/<fitting_name>/`
  - Used for organizing multiple fitting experiments

#### Example Segment Selection
- **Example Segment**: Controls which validation segment is displayed in the preview plots
  - **Longest**: Shows the longest available segment (default)
  - **Random**: Randomly selects a segment each time
  - **Index**: Selects segment by index (0-based)
  - When "Index" is selected, an entry field appears for specifying the segment number

#### Motor Model Selection
- **Motor Model**: Radio buttons to choose the motor model type
  - **DC Motor**: Uses DC motor model with parameters (R, K, b, J, etc.)
  - **Polynomial Map**: Uses polynomial torque map with 10 coefficients
  - Changing the model updates the parameter fields automatically
  - **Fit DC from map**: Checkbox (only visible for polynomial model)
    - If enabled, after polynomial optimization, fits DC parameters from the resulting torque map
    - Useful for converting polynomial fits to DC model parameters

#### Barrier Functions
- **Use Barrier Functions**: Checkbox to enable barrier function constraints
  - When enabled, adds penalty terms to keep parameters away from bounds
  - **Barrier μ**: Barrier function strength parameter (default: 0.01)
    - Smaller values = stronger barrier (parameters stay further from bounds)
    - Larger values = weaker barrier (parameters can approach bounds more closely)
  - Useful for preventing parameters from hitting optimization boundaries

### Training & Optimization Section

#### Basic Optimization Settings

**Max Iterations**
- Maximum number of optimizer iterations per batch/epoch
- Default: 50
- Higher values allow more refinement but take longer
- Typical range: 20-100

**Min/Max Segment Length**
- **Min Segment Length**: Minimum number of timesteps per segment (default: 50)
- **Max Segment Length**: Maximum number of timesteps per segment (default: 100)
- Segments shorter than min are discarded
- Segments longer than max are split (unless "Use whole trips" is enabled)
- Units: timesteps (not seconds)

**Use whole trips (no segmenting)**
- When enabled, keeps entire trips as single segments
- Disables max segment length splitting
- Useful for fitting to complete trip trajectories

#### Segment Filtering Options

**Filter zero-speed segments**
- When enabled, limits the fraction of segments with near-zero speed
- Default: Enabled
- Filters segments where mean speed < 0.1 m/s
- Maximum zero-speed fraction: 5% (configurable in config)
- Helps focus optimization on dynamic segments

**Disable segment filtering (raw trips)**
- When enabled, bypasses all segment quality filters
- Only filters for finite values (no NaN/Inf)
- Useful for debugging or when you want to use all available data
- Warning: May include low-quality segments that hurt fitting

#### Batching Configuration

**Segments per Batch**
- Number of segments to include in each optimization batch
- Default: 16
- Larger batches = more stable gradients but slower per iteration
- Smaller batches = faster iterations but noisier gradients
- Set to 0 to use all segments in one batch

**Epochs**
- Number of training epochs (full passes through the dataset)
- Default: 1
- Each epoch processes all batches
- Segments are shuffled between epochs (if enabled)

**Random fixed-length batches**
- When enabled, uses random sampling of fixed-length segments
- **Batch segment length**: Target length for randomly sampled segments (default: 100 timesteps)
- **Random batches/epoch**: Number of random batches to generate per epoch (default: 10)
- **Random batch max iter**: Maximum iterations per random batch (default: 5)
- Useful for:
  - Training on diverse segment lengths
  - Reducing memory usage
  - Improving generalization

**Debug batch progress**
- When enabled, prints detailed progress during batch optimization
- **Progress step (%)**: Print progress every N% of batch completion (default: 10%)
- Useful for monitoring long-running optimizations

#### Validation Settings

**Fixed-length validation**
- When enabled, uses fixed-length segments for validation (matching random batch length)
- When disabled, uses all validation segments regardless of length
- **Validation fraction**: Fraction of data to use for validation (default: 0.1 = 10%)
- **Validation split seed**: Random seed for train/validation split (empty = use default)
  - Set a specific seed for reproducible train/val splits
  - Useful for comparing different fitting runs

#### Loss Function Configuration

**Speed Loss Weight**
- Weight for velocity (speed) error in loss function
- Default: 1.0
- Primary loss component - should typically be 1.0

**Accel Loss Weight**
- Weight for instantaneous acceleration error
- Default: 0.0 (disabled)
- Can be enabled (e.g., 0.1-0.5) to improve acceleration matching
- Higher values emphasize acceleration accuracy over speed

**Brake Loss Boost**
- Extra weight multiplier for samples with active brake
- Default: 0.0 (no boost)
- When > 0, brake-active samples contribute more to loss
- Useful for improving brake parameter fitting
- Typical range: 0.0-2.0

**Full-stop loss cap (fraction)**
- Caps the loss contribution from full-stop segments
- Default: 0.0 (no cap)
- When > 0, limits how much full-stop segments can contribute to total loss
- Prevents full-stop segments from dominating the optimization
- Typical range: 0.0-0.3

**Mask loss for negative GT speeds**
- When enabled, ignores loss where ground truth speed is negative
- Default: Disabled
- Useful for filtering out data artifacts or reverse motion

**Mask loss for |grade| > 2 deg**
- When enabled, ignores loss samples where absolute grade is above 2 degrees
- Default: Disabled
- Useful for de-emphasizing steep-grade segments during fitting

#### ExtendedPlant (RL Dynamics) Settings

**Use ExtendedPlant (RL)**
- When enabled, uses the full ExtendedPlant dynamics model
- Default: Enabled
- Provides more accurate simulation with:
  - Throttle delay and gamma shaping
  - Brake lag dynamics
  - Minimum motor current floor at zero throttle
  - Current/power/voltage limits
- When disabled, uses simplified dynamics (faster but less accurate)

**Optimize without grade**
- When enabled, fitting rollout forces grade to 0 regardless of dataset angle
- Useful for identifying non-grade parameters when grade quality is uncertain

**Flip grade sign from data**
- When enabled, fitter uses `-angle` from dataset files
- Useful when source data uses opposite grade sign convention

**Plant Substeps**
- Number of internal simulation substeps per timestep
- Default: 2
- Higher values = more accurate integration but slower
- Typical range: 1-4
- Should match the value used in RL training (if applicable)

**Apply LPF to fitting data (accel: 2Hz, speed: 5Hz)**
- When enabled, applies low-pass Butterworth filter to input data
- Speed filtered at 5 Hz cutoff
- Acceleration filtered at 2 Hz cutoff
- Reduces high-frequency noise in ground truth data
- Useful when data has significant measurement noise
- Default: Disabled

#### Actuator Settings

**Actuator Smoothing α**
- Exponential smoothing factor for actuator commands
- Default: 0.2
- Range: 0.0-1.0
- 0.0 = no smoothing (raw commands)
- 1.0 = maximum smoothing (very slow response)
- Simulates actuator response lag

**Actuator Deadband (%)**
- Deadband threshold for actuator commands
- Default: 1.0%
- Commands below this threshold are treated as zero
- Simulates actuator deadzone

**Max |Accel| (m/s²)**
- Maximum absolute acceleration for data filtering
- Default: 6.0 m/s²
- Segments with accelerations exceeding this are filtered out
- Prevents unrealistic data from affecting fitting

#### Optimization Method Settings

**Scale params (0-1)**
- When enabled, normalizes all parameters to [0, 1] range for optimization
- Default: Enabled
- Improves optimizer convergence by equalizing parameter scales
- Automatically converts back to physical units

**Optimization Mode**
- **joint**: Optimizes all parameters simultaneously (default)
- **sequential**: Optimizes parameters in phases (throttle then brake, or vice versa)
  - When sequential is selected, "Phase Order" becomes active
  - Allows focusing optimization on specific actuation phases
  - Useful when one phase has more data than another

**Phase Order** (only active in sequential mode)
- **throttle -> brake**: Optimize throttle parameters first, then brake parameters
- **brake -> throttle**: Optimize brake parameters first, then throttle parameters
- In sequential mode, parameters from the inactive phase are frozen

**Optimizer**
- Optimization algorithm to use
- Options:
  - **L-BFGS-B**: Limited-memory BFGS with bounds (default, recommended)
  - **TNC**: Truncated Newton with bounds
  - **SLSQP**: Sequential Least Squares Programming
  - **Powell**: Powell's method (derivative-free)
- L-BFGS-B is typically fastest and most reliable

**Overfit longest training trip first**
- When enabled, performs a warmup phase on the longest training trip
- **Overfit epochs**: Number of epochs to overfit the longest trip (default: 1)
- Useful for:
  - Initializing parameters on a representative long segment
  - Improving convergence on complex trajectories
  - When enabled with random batches, also uses random fixed-length segments from the longest trip

### Parameter Configuration Section

The parameter section displays all vehicle parameters organized by category. For each parameter, you can set:

- **Initial**: Starting value for optimization
- **Min**: Lower bound (parameter cannot go below this)
- **Max**: Upper bound (parameter cannot go above this)

**Parameter Groups (DC Motor Model):**
- **Body**: mass, drag_area, rolling_coeff
- **Motor**: motor_V_max, motor_R, motor_K, motor_b, motor_J, motor_gamma_throttle, motor_throttle_tau, motor_min_current_A
- **Motor Limits**: motor_T_max, motor_P_max
- **Drivetrain**: gear_ratio, eta_gb
- **Brake**: brake_T_max, brake_tau, brake_p, mu
- **Wheel**: wheel_radius, wheel_inertia

**Parameter Groups (Polynomial Motor Model):**
- **Body**: mass, drag_area, rolling_coeff
- **Motor**: motor_V_max, motor_gamma_throttle, motor_throttle_tau, motor_min_current_A
- **Polynomial Coefficients**: 10 coefficients (c_00, c_10, c_01, c_20, c_11, c_02, c_30, c_21, c_12, c_03)
- **Drivetrain**: gear_ratio, eta_gb
- **Brake**: brake_T_max, brake_tau, brake_p, mu
- **Wheel**: wheel_radius, wheel_inertia

**Note**: If min == max for a parameter, that parameter is fixed (not optimized) at that value.

### Control Buttons

**Update Simulation**
- Updates the preview plots with current parameter values
- Does not start fitting - just shows what the simulation looks like
- Useful for manually tuning parameters and seeing immediate feedback

**Validation RMSE**
- Opens the validation analysis window (see Validation Metrics section below)
- Computes RMSE and MAE metrics on validation segments
- Shows detailed analysis and visualization options

**Save as Default**
- Saves current GUI settings to a default configuration file
- Includes all optimization options, parameter bounds, and settings
- Settings are saved to `~/.vehicle_fitting_settings.json`

**Load Default**
- Loads previously saved default settings
- Restores all GUI fields to saved values

**Start Fitting**
- Begins the parameter fitting process
- Disables the button and shows progress
- Results are saved to `results/<fitting_name>/`

**Abort**
- Stops the current fitting process
- Only enabled during fitting
- Saves current best parameters before aborting

**Advance Phase** (only in sequential mode)
- Manually advances to the next optimization phase
- Only enabled in sequential optimization mode
- Allows you to control when to move from throttle to brake phase (or vice versa)

### Simulation Preview Plots

The right panel shows real-time simulation previews with 8 subplots:

#### Left Column (Throttle & Brake Dynamics)

1. **Braking Dynamics (from 20 m/s)**
   - Shows speed vs time for different brake levels (10%, 20%, ..., 100%)
   - Simulates braking from 20 m/s initial speed
   - Useful for validating brake parameters

2. **Throttle Dynamics - Speed (from 0 m/s)**
   - Shows speed vs time for different throttle levels (10%, 20%, ..., 100%)
   - Simulates acceleration from 0 m/s
   - Useful for validating throttle/motor parameters

3. **Throttle Dynamics - Power**
   - Shows motor power vs time for different throttle levels
   - Only available when using ExtendedPlant
   - Shows power consumption and limits

4. **Throttle Dynamics - Current**
   - Shows motor current vs time for different throttle levels
   - Only available when using ExtendedPlant
   - Shows current draw and limits

#### Right Column (Validation Segment)

5. **Validation Segment Comparison**
   - Overlays simulated speed (red) vs ground truth speed (black dashed)
   - Shows how well current parameters match real data
   - Updates when "Update Simulation" is clicked or during fitting

6. **Validation Segment Actuations**
   - Shows throttle (green) and brake (blue) commands over time
   - Helps understand what the vehicle was doing during the segment

7. **Validation Motor Power**
   - Shows motor power during the validation segment
   - Only available when using ExtendedPlant
   - Displays power consumption and limits

8. **Validation Motor Current**
   - Shows motor current during the validation segment
   - Only available when using ExtendedPlant
   - Displays current draw and limits

All plots have zoom/pan functionality via the matplotlib toolbar at the top.

### Validation Metrics and Analysis

Clicking "Validation RMSE" opens a comprehensive validation analysis window with multiple features:

#### Validation Analysis Menu

The analysis menu allows you to select and visualize validation segments by:

- **Segment Length**: Pre-computed analysis for 5s, 10s, 15s, 25s, and 50s segments
- **Overall Metrics**: Shows RMSE and MAE for speed and acceleration
- **Segment Selection Buttons**:
  - **Best/Worst/Median speed RMSE**: Select segments with best, worst, or median speed error
  - **Best/Worst/Median accel RMSE**: Select segments with best, worst, or median acceleration error

#### Validation Segment Window

When you select a segment, a detailed window opens showing:

1. **Speed Plot**
   - Ground truth speed (blue, filtered at 5 Hz)
   - Simulated speed (red dashed)
   - Title shows RMSE and MAE for speed

2. **Acceleration Plot**
   - Ground truth acceleration (green, filtered at 2 Hz)
   - Simulated acceleration (magenta dashed)
   - Title shows RMSE and MAE for acceleration

3. **Actuation Plot**
   - Throttle command (orange)
   - Brake command (red)
   - Y-axis: 0-100%

4. **Road Grade Plot**
   - Road grade in degrees (brown)
   - Converted from radians for readability

All plots share the same time axis for easy comparison.

#### Validation Summary Table

Clicking "View Summary Table" opens a comprehensive metrics visualization:

**Row 1: Speed Metrics by Speed Range**
- **Speed RMSE by Speed Range**: Bar chart showing RMSE for 0-2, 2-5, 5-10, 10-20 m/s ranges
- **Speed MAE by Speed Range**: Mean Absolute Error by speed range
- **Speed STD by Speed Range**: Standard deviation of speed errors by range
- Each metric shown for different segment lengths (5s, 10s, 15s, 25s, 50s)

**Row 2: Acceleration Metrics by Speed Range**
- **Accel RMSE by Speed Range**: Acceleration RMSE by speed range
- **Accel MAE by Speed Range**: Acceleration MAE by speed range
- **Accel STD by Speed Range**: Acceleration error standard deviation by range

**Row 3: Overall Metrics**
- Bar chart spanning all columns showing:
  - v_RMSE, v_MAE, v_STD (speed metrics)
  - a_RMSE, a_MAE, a_STD (acceleration metrics)
- Grouped by segment length for comparison

All bar charts include value labels on the bars and use color coding by segment length.

### Settings Persistence

The GUI automatically saves and loads settings:

- **Auto-save**: Settings are saved when you click "Save as Default"
- **Auto-load**: Settings are loaded when the GUI starts (if a saved file exists)
- **Saved Settings Include**:
  - All optimization options
  - Parameter initial values and bounds
  - Motor model selection
  - All checkboxes and dropdowns
  - Example segment selection

Settings file location:
- `fitting/gui_settings.json`

### Tips for Using the GUI

1. **Start with Defaults**: Use the default parameter bounds initially, then narrow them based on results
2. **Use Preview**: Click "Update Simulation" frequently to see how parameter changes affect behavior
3. **Check Validation**: Use "Validation RMSE" to assess fit quality before and after optimization
4. **Sequential Mode**: Use sequential optimization when you have imbalanced throttle/brake data
5. **Random Batches**: Enable random fixed-length batches for better generalization
6. **LPF Filtering**: Enable LPF if your data has significant noise
7. **Overfit Warmup**: Use overfit longest trip for complex trajectories or poor initial guesses
8. **Parameter Scaling**: Keep enabled unless you have specific reasons to disable it
9. **Save Settings**: Save your working configuration as default for future runs

### 3. Validate Results

- Check fit loss (lower is better)
- Check R² (closer to 1.0 is better)
- Visualize simulated vs measured speeds
- Run simulation with fitted params and compare

## Interpreting Results

- **fit_loss**: Mean squared error of velocity (m²/s²)
- **r_squared**: Coefficient of determination (0-1)
- **num_samples**: Number of data points used

Good fits typically have:
- R² > 0.95
- Low fit_loss relative to speed variance
- Physically reasonable parameter values

## Troubleshooting

### Poor Fit Quality

- Check data quality (noise, outliers)
- Increase number of epochs or iterations
- Adjust parameter bounds
- Try different optimization methods

### Unphysical Parameters

- Check parameter bounds
- Verify data units are correct
- Check for data preprocessing issues

### Convergence Issues

- Try warmup initialization
- Adjust learning rate/optimizer settings
- Reduce batch size
- Check for numerical issues (NaN/Inf)
