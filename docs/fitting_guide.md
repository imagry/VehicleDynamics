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
