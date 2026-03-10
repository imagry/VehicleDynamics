"""Vehicle parameter fitting from trip data using trajectory simulation.

Fits vehicle dynamics parameters by simulating full trip trajectories and
minimizing velocity error (not instantaneous acceleration error).

This is more robust because:
1. Errors compound over time, forcing physically consistent parameters
2. Velocity integration smooths measurement noise
3. Tests the model's ability to predict real vehicle behavior
"""

from __future__ import annotations

import json
import time
import logging
import sys
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from simulation.dynamics import (
    ExtendedPlant,
    ExtendedPlantParams,
    MotorParams,
    BrakeParams,
    BodyParams,
    WheelParams,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

LOGGER = logging.getLogger(__name__)


def _load_torch_file_compat(file_path: Path):
    """Load torch file with numpy version compatibility handling.
    
    Handles the case where files were saved with newer numpy versions
    that use numpy._core, but current environment has older numpy.
    """
    import sys
    import torch
    
    # Try normal loading first
    try:
        return torch.load(file_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            # Handle numpy version mismatch by temporarily aliasing numpy._core to numpy.core
            LOGGER.warning(
                "Encountered numpy version mismatch (numpy._core). "
                "Attempting compatibility workaround..."
            )
            
            # Temporarily add numpy._core as an alias to numpy.core in sys.modules
            # This allows pickle to find the modules it needs
            import numpy.core
            import numpy.core.multiarray
            import numpy.core._multiarray_umath
            
            # Create aliases for numpy._core modules
            sys.modules["numpy._core"] = numpy.core
            sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
            sys.modules["numpy._core._multiarray_umath"] = numpy.core._multiarray_umath
            
            try:
                # Now try loading again
                result = torch.load(file_path, map_location="cpu", weights_only=False)
                return result
            finally:
                # Clean up the temporary aliases (only if we added them)
                # Note: We don't remove them completely as they might be needed for subsequent loads
                # The aliases are harmless and will be overwritten if numpy._core is actually installed
                pass
        else:
            raise


class AbortFitting(Exception):
    """Raised when fitting is aborted by user."""

# Physical constants (matching utils/dynamics.py)
GRAVITY: float = 9.80665  # m/s²
AIR_DENSITY: float = 1.225  # kg/m³


@dataclass(slots=True)
class FittedVehicleParams:
    """Fitted vehicle parameters matching ExtendedPlant model.
    
    ALL parameters are fitted by default.
    """
    
    # === BODY PARAMETERS ===
    mass: float = 1900.0  # kg - vehicle mass
    drag_area: float = 0.65  # m² - CdA (drag coefficient * frontal area)
    rolling_coeff: float = 0.011  # dimensionless - rolling resistance coefficient
    
    # === MOTOR PARAMETERS ===
    motor_V_max: float = 400.0  # V - maximum motor voltage
    motor_R: float = 0.2  # Ω - armature resistance
    motor_K: float = 0.2  # Nm/A = V·s/rad - K_t = K_e (torque/back-EMF constant)
    motor_b: float = 1e-3  # Nm·s/rad - viscous friction
    motor_J: float = 1e-3  # kg·m² - rotor inertia
    motor_gamma_throttle: float = 1.0  # throttle nonlinearity exponent
    motor_throttle_tau: float = 0.1  # s - throttle time constant
    motor_min_current_A: float = 0.0  # A - minimum commanded current at zero throttle
    motor_T_max: float | None = None  # Nm - max motor torque
    motor_P_max: float | None = None  # W - max motor power
    
    # === DRIVETRAIN PARAMETERS ===
    gear_ratio: float = 10.0  # N - gear reduction ratio
    eta_gb: float = 0.92  # gearbox efficiency
    
    # === BRAKE PARAMETERS ===
    brake_T_max: float = 15000.0  # Nm - maximum brake torque at wheel
    brake_tau: float = 0.08  # s - brake time constant
    brake_p: float = 1.2  # brake exponent
    mu: float = 0.9  # tire friction coefficient
    
    # === WHEEL PARAMETERS ===
    wheel_radius: float = 0.346  # m - wheel radius
    wheel_inertia: float = 1.5  # kg·m² - wheel + rotating assembly

    # === FITTING METADATA ===
    fit_loss: float = 0.0  # Final loss value (velocity MSE)
    num_samples: int = 0  # Number of data points used
    r_squared: float = 0.0  # Coefficient of determination for velocity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "FittedVehicleParams":
        """Create from dictionary with backward compatibility."""
        # Get valid field names from the dataclass
        valid_fields = {f.name for f in fields(cls)}
        
        # Handle very old files with motor_force_coeff
        if "motor_force_coeff" in d:
            d.pop("motor_force_coeff")
        
        # Default values for backward compatibility
        defaults = {
            "motor_V_max": 400.0,
            "motor_R": 0.2,
            "motor_K": 0.2,
            "motor_b": 1e-3,
            "motor_J": 1e-3,
            "motor_gamma_throttle": 1.0,
            "motor_throttle_tau": 0.1,
            "motor_min_current_A": 0.0,
            "motor_T_max": None,
            "motor_P_max": None,
            "gear_ratio": 10.0,
            "eta_gb": 0.92,
            "brake_T_max": 15000.0,
            "brake_tau": 0.08,
            "brake_p": 1.2,
            "mu": 0.9,
            "wheel_radius": 0.346,
            "wheel_inertia": 1.5,
        }

        # Drop legacy creep fields if present.
        d.pop("creep_a_max", None)
        d.pop("creep_v_cutoff", None)
        d.pop("creep_v_hold", None)

        # Handle motor_I_max -> motor_T_max conversion (before filtering)
        if "motor_I_max" in d and "motor_T_max" not in d and "motor_K" in d:
            motor_k = float(d.get("motor_K", defaults["motor_K"]))
            motor_i_max = d.pop("motor_I_max")
            d["motor_T_max"] = None if motor_i_max is None else float(motor_i_max) * motor_k
        
        # Remove deprecated/unknown fields (like 'epoch', 'batch', etc.) after handling conversions
        d = {k: v for k, v in d.items() if k in valid_fields}

        for key, val in defaults.items():
            if key not in d:
                d[key] = val
        return cls(**d)
    
    def save(self, path: Path) -> None:
        """Save fitted parameters to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved fitted params to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "FittedVehicleParams":
        """Load fitted parameters from JSON file."""
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)
    
    def to_extended_plant_params(self) -> Dict:
        """Convert to ExtendedPlantParams-compatible dictionary."""
        return {
            "mass": self.mass,
            "drag_area": self.drag_area,
            "rolling_coeff": self.rolling_coeff,
            "motor": {
                "V_max": self.motor_V_max,
                "R": self.motor_R,
                "K_t": self.motor_K,
                "K_e": self.motor_K,  # K_t = K_e for DC motors
                "b": self.motor_b,
                "J": self.motor_J,
                "gamma_throttle": self.motor_gamma_throttle,
                "throttle_tau": self.motor_throttle_tau,
                "min_current_A": self.motor_min_current_A,
                "T_max": self.motor_T_max,
                "P_max": self.motor_P_max,
            },
            "gear_ratio": self.gear_ratio,
            "eta_gb": self.eta_gb,
            "brake": {
                "T_max": self.brake_T_max,
                "tau": self.brake_tau,
                "p": self.brake_p,
                "mu": self.mu,
            },
            "wheel": {
                "radius": self.wheel_radius,
                "inertia": self.wheel_inertia,
            },
        }


@dataclass(slots=True)
class FitterConfig:
    """Configuration for vehicle parameter fitting.
    
    ALL parameters are fitted by default. Use same min/max bounds to fix a parameter.
    """
    
    # === SIMULATION SETTINGS ===
    dt: float = 0.1  # s - simulation timestep (will be estimated from data)
    
    # === INITIAL GUESSES (all parameters) ===
    mass_init: float = 1800.0  # kg
    drag_area_init: float = 0.65  # m² (Cd~0.29 * A~2.2m²)
    rolling_coeff_init: float = 0.011  # typical EV
    motor_V_max_init: float = 400.0  # V
    motor_R_init: float = 0.2  # Ω
    motor_K_init: float = 0.2  # Nm/A
    motor_b_init: float = 1e-3  # Nm·s/rad - viscous friction
    motor_J_init: float = 1e-3  # kg·m² - rotor inertia
    motor_gamma_throttle_init: float = 1.0  # throttle nonlinearity exponent
    motor_throttle_tau_init: float = 0.1  # s - throttle time constant
    motor_min_current_A_init: float = 0.0  # A - minimum commanded current at zero throttle
    motor_T_max_init: float | None = None  # Nm - max motor torque (optional)
    motor_P_max_init: float | None = None  # W - max motor power (optional)
    gear_ratio_init: float = 10.0  # gear ratio
    eta_gb_init: float = 0.92  # gearbox efficiency
    brake_T_max_init: float = 15000.0  # Nm
    brake_tau_init: float = 0.08  # s - brake time constant
    brake_p_init: float = 1.2  # brake exponent
    mu_init: float = 0.9  # tire friction coefficient
    wheel_radius_init: float = 0.346  # m
    wheel_inertia_init: float = 1.5  # kg·m²

    # === PARAMETER BOUNDS (all fitted by default) ===
    mass_bounds: Tuple[float, float] = (1800.0, 2300.0)  # kg
    drag_area_bounds: Tuple[float, float] = (0.4, 1.2)   # CdA (m²)
    rolling_coeff_bounds: Tuple[float, float] = (0.006, 0.020)  # C_rr
    motor_V_max_bounds: Tuple[float, float] = (320.0, 400.0)  # V
    motor_R_bounds: Tuple[float, float] = (0.02, 0.5)  # Ω
    motor_K_bounds: Tuple[float, float] = (0.05, 0.5)  # Nm/A
    motor_b_bounds: Tuple[float, float] = (1e-6, 1e-1)  # Nm·s/rad
    motor_J_bounds: Tuple[float, float] = (1e-2, 1e-1)  # kg·m²
    motor_gamma_throttle_bounds: Tuple[float, float] = (0.5, 2.0)  # throttle nonlinearity exponent
    motor_throttle_tau_bounds: Tuple[float, float] = (0.02, 0.5)  # s - throttle time constant
    motor_min_current_A_bounds: Tuple[float, float] = (0.0, 80.0)  # A
    motor_T_max_bounds: Tuple[float, float] = (0.0, 1000.0)  # Nm - optional torque limit
    motor_P_max_bounds: Tuple[float, float] = (0.0, 250000.0)  # W - optional power limit
    gear_ratio_bounds: Tuple[float, float] = (4.3, 11.0)  # N
    eta_gb_bounds: Tuple[float, float] = (0.85, 0.99)  # efficiency
    brake_T_max_bounds: Tuple[float, float] = (10000.0, 20000.0)  # Nm
    brake_tau_bounds: Tuple[float, float] = (0.01, 0.5)  # s
    brake_p_bounds: Tuple[float, float] = (0.5, 3.0)  # exponent
    mu_bounds: Tuple[float, float] = (0.5, 1.2)  # friction coefficient
    wheel_radius_bounds: Tuple[float, float] = (0.315, 0.34)  # m
    wheel_inertia_bounds: Tuple[float, float] = (1.0, 2.0)  # kg·m²

    # === LOSS SETTINGS ===
    speed_loss_weight: float = 1.0  # weight for velocity MSE
    accel_loss_weight: float = 0.0  # weight for instantaneous acceleration MSE
    brake_loss_boost: float = 0.0  # extra weight for samples with active brake
    use_uniform_speed_accel_bin_loss: bool = False  # reweight samples so speed-accel bins contribute more uniformly
    speed_accel_speed_bins: int = 20  # number of bins on speed axis
    speed_accel_accel_bins: int = 20  # number of bins on acceleration axis
    speed_accel_speed_range: Tuple[float, float] = (0.0, 25.0)  # fixed speed bounds for bucketization (m/s)
    speed_accel_accel_range: Tuple[float, float] = (-4.0, 4.0)  # fixed acceleration bounds for bucketization (m/s^2)
    speed_accel_bin_weight_cap: float = 10.0  # clamp for per-sample bucket weights (>0)
    optimize_without_grade: bool = False  # if True, force road grade to 0 during optimization simulation
    mask_negative_gt_speed: bool = False  # ignore loss where GT speed is negative
    full_stop_loss_cap_fraction: float = 0.0  # cap full-stop segment loss as fraction of total (0=off)
    
    # === PLANT PARITY SETTINGS ===
    use_extended_plant: bool = True  # use ExtendedPlant dynamics for simulation
    extended_plant_substeps: int = 1  # substeps per dt (match env config)
    extended_plant_zero_brake_lag: bool = False  # force brake tau ~0 for parity checks
    
    # === DATA PREPROCESSING ===
    apply_lpf_to_fitting_data: bool = False  # apply low-pass filter to acceleration (2Hz) and speed (5Hz) data during fitting

    # === SEGMENT FILTERING ===
    min_speed: float = 0.5  # m/s - minimum speed to include
    max_speed: float = 20.0  # m/s - maximum speed to include (filter unrealistic values)
    max_accel: float = 6.0  # m/s² - filter extreme accelerations
    min_segment_length: int = 50  # minimum timesteps per segment
    max_segment_length: int = 100  # maximum timesteps per segment (memory)
    use_whole_trips: bool = False  # if True, do not split valid regions by max_segment_length
    downsampling_factor: int = 1  # downsample data by taking every Nth sample (1 = no downsampling)
    max_zero_speed_fraction: float = 0.05  # maximum fraction of segments with zero/near-zero speed (0.05 = 5%)
    zero_speed_eps: float = 0.1  # epsilon for zero speed threshold (m/s) - segments with mean speed < eps are considered zero-speed
    filter_zero_speed_segments: bool = True  # enable zero-speed segment filtering
    disable_segment_filtering: bool = False  # if True, skip segment validity/suspicious filtering
    
    # === SUSPICIOUS SEGMENT FILTERING ===
    min_speed_std: float = 0.02  # minimum std of speed within segment (m/s)
    min_accel_std: float = 0.05  # minimum std of accel within segment (m/s²)
    flat_speed_eps: float = 0.02  # delta threshold for flat speed detection (m/s)
    flat_act_eps: float = 0.5  # delta threshold for flat actuation detection (% command)
    max_flat_speed_fraction: float = 0.95  # max fraction of near-constant speed diffs
    max_flat_act_fraction: float = 0.97  # max fraction of near-constant throttle/brake diffs
    max_zero_command_fraction: float = 0.95  # max fraction of timesteps with throttle & brake ~0
    saturation_threshold: float = 99.0  # percent considered saturated
    max_saturation_fraction: float = 0.5  # max fraction of time at saturation
    brake_deadband_pct: float = 5.0  # brake deadband (%) to ignore small brake noise
    actuator_deadband_pct: float = 1.0  # deadband for throttle/brake (%)
    actuator_smoothing_alpha: float = 0.2  # exp smoothing alpha for throttle/brake (0 = off)
    
    # === OPTIMIZATION SETTINGS ===
    max_iter: int = 1  # iterations per optimization call (reduced for faster convergence)
    tolerance: float = 1e-6  # relaxed tolerance for faster convergence
    use_param_scaling: bool = True  # optimize in 0-1 scaled parameter space
    optimizer_method: str = "L-BFGS-B"  # optimization method (supports bounds)

    # === PHASED OPTIMIZATION ===
    optimization_mode: str = "joint"  # "joint" or "sequential"
    phase_order: List[str] = field(default_factory=lambda: ["throttle", "brake"])
    pause_between_phases: bool = False  # require manual advance between phases
    
    # === TRAJECTORY BATCHING ===
    segments_per_batch: int = 16  # number of trip segments per batch (smaller = faster)
    num_epochs: int = 1  # number of passes over all segments
    shuffle_segments: bool = True
    validation_fraction: float = 0.1  # fraction of segments for validation
    validation_split_seed: Optional[int] = None  # seed for train/val split
    use_random_segment_batches: bool = False  # sample random fixed-length segments per batch
    random_segment_length: int = 100  # fixed length for random batch sampling (timesteps)
    random_batches_per_epoch: int = 10  # number of random batches per epoch when enabled
    debug_batch_progress: bool = False  # print progress within a batch
    debug_batch_progress_step: float = 0.1  # fraction of segment length between progress prints
    random_batch_max_iter: int = 5  # max L-BFGS-B iterations per random batch
    use_fixed_length_validation: bool = False  # use fixed-length segments for validation

    # === OVERFIT LONGEST TRAINING TRIP ===
    use_overfit_longest_trip: bool = False  # run a warmup phase on the longest training trip
    overfit_longest_trip_epochs: int = 1  # epochs for the overfit warmup phase

    # === BACKWARD-COMPATIBILITY ALIASES ===
    wheel_radius: Optional[float] = None  # alias for wheel_radius_init
    batch_size: Optional[int] = None  # alias for segments_per_batch

    def __post_init__(self) -> None:
        if self.wheel_radius is None:
            self.wheel_radius = self.wheel_radius_init
        else:
            self.wheel_radius_init = float(self.wheel_radius)

        if self.batch_size is None:
            self.batch_size = self.segments_per_batch
        else:
            self.segments_per_batch = int(self.batch_size)
    
    # === WARMUP ===
    use_warmup: bool = False  # enable warmup to find better initial guess
    warmup_samples: int = 10  # number of random parameter sets to try
    warmup_seed: int = 42  # random seed for warmup sampling
    
    # === BARRIER FUNCTIONS ===
    use_barrier: bool = False  # enable interior-point barrier method to avoid active constraints
    barrier_mu: float = 0.01  # barrier parameter μ (smaller = stronger barrier, keeps params away from boundaries)
    
    # === GPU ACCELERATION ===
    use_gpu: bool = True  # use GPU for parallel simulation and loss computation (if available)
    
    # === MOTOR MODEL TYPE ===
    motor_model_type: str = "dc"  # "dc" or "polynomial"
    fit_dc_from_map: bool = False  # if True, fit DC motor params to match polynomial map after optimization
    
    # === POLYNOMIAL MOTOR MAP COEFFICIENTS (order 3 with cross-terms) ===
    # τ_m = c_00 + c_10*V + c_01*ω + c_20*V² + c_11*V*ω + c_02*ω² + c_30*V³ + c_21*V²*ω + c_12*V*ω² + c_03*ω³
    # where V = V_cmd/V_max (normalized 0-1), ω = omega_m (rad/s)
    poly_c_00_init: float = 0.0  # constant term
    poly_c_10_init: float = 200.0  # V term
    poly_c_01_init: float = -0.1  # ω term
    poly_c_20_init: float = 0.0  # V² term
    poly_c_11_init: float = -0.5  # V*ω term
    poly_c_02_init: float = 0.0  # ω² term
    poly_c_30_init: float = 0.0  # V³ term
    poly_c_21_init: float = 0.0  # V²*ω term
    poly_c_12_init: float = 0.0  # V*ω² term
    poly_c_03_init: float = 0.0  # ω³ term
    
    poly_c_00_bounds: Tuple[float, float] = (-50.0, 50.0)
    poly_c_10_bounds: Tuple[float, float] = (0.0, 1000.0)  # V term should be positive
    poly_c_01_bounds: Tuple[float, float] = (-10.0, 0.0)  # ω term typically negative (back-EMF)
    poly_c_20_bounds: Tuple[float, float] = (-100.0, 100.0)
    poly_c_11_bounds: Tuple[float, float] = (-5.0, 0.0)  # V*ω typically negative
    poly_c_02_bounds: Tuple[float, float] = (-0.01, 0.01)
    poly_c_30_bounds: Tuple[float, float] = (-200.0, 200.0)
    poly_c_21_bounds: Tuple[float, float] = (-10.0, 10.0)
    poly_c_12_bounds: Tuple[float, float] = (-0.1, 0.1)
    poly_c_03_bounds: Tuple[float, float] = (-0.001, 0.001)


@dataclass
class TripSegment:
    """A single trip segment for trajectory simulation."""
    trip_id: str
    speed: np.ndarray         # measured speeds (m/s) - target
    acceleration: np.ndarray  # measured accelerations (m/s²) - target
    throttle: np.ndarray      # throttle input (0-100)
    brake: np.ndarray         # brake input (0-100)
    grade: np.ndarray         # road grade (rad)
    dt: float                 # timestep (s)
    sample_weights: Optional[np.ndarray] = None  # optional per-step loss weights
    
    @property
    def length(self) -> int:
        return len(self.speed)
    
    @property
    def initial_speed(self) -> float:
        return float(self.speed[0])


class VehicleParamFitter:
    """Fits vehicle dynamics parameters by simulating full trajectories.
    
    Uses quasi-steady-state DC motor model:
    
        Motor current (quasi-steady): i = (V_cmd - K_e * ω_m) / R
        Motor torque: τ_m = K_t * i
        Wheel torque: τ_w = η_gb * N * τ_m
        Drive force: F_drive = τ_w / r
        
    The key difference from single-step fitting is that we:
    1. Start with the measured initial speed v[0]
    2. Simulate forward: v[t+1] = v[t] + a(params, v[t], inputs[t]) * dt
    3. Minimize MSE between simulated v and measured v over full trajectory
    
    This forces parameters to be physically consistent over time.
    """
    
    @property
    def PARAM_NAMES(self) -> List[str]:
        """Get parameter names based on motor model type."""
        if self.config.motor_model_type == "polynomial":
            return [
                "mass", "drag_area", "rolling_coeff",
                "motor_V_max", "motor_gamma_throttle", "motor_throttle_tau",
                "poly_c_00", "poly_c_10", "poly_c_01", "poly_c_20", "poly_c_11", "poly_c_02",
                "poly_c_30", "poly_c_21", "poly_c_12", "poly_c_03",
                "gear_ratio", "eta_gb",
                "brake_T_max", "brake_tau", "brake_p", "mu",
                "wheel_radius", "wheel_inertia",
                "motor_min_current_A",
            ]
        else:  # DC model
            return [
                "mass", "drag_area", "rolling_coeff",
                "motor_V_max", "motor_R", "motor_K", "motor_b", "motor_J", "motor_gamma_throttle", "motor_throttle_tau",
                "motor_T_max", "motor_P_max",
                "gear_ratio", "eta_gb",
                "brake_T_max", "brake_tau", "brake_p", "mu",
                "wheel_radius", "wheel_inertia",
                "motor_min_current_A",
            ]
    
    def __init__(self, config: Optional[FitterConfig] = None):
        self.config = config or FitterConfig()
        if self.config.motor_model_type == "dc":
            if not self.config.use_extended_plant:
                LOGGER.info("Forcing use_extended_plant=True for DC motor fitting")
            self.config.use_extended_plant = True
        elif self.config.use_extended_plant:
            LOGGER.warning("ExtendedPlant is only supported for DC motor model; disabling for polynomial")
            self.config.use_extended_plant = False
        self._segments: List[TripSegment] = []
        self._current_loss: float = 0.0
        self._bounds: Optional[List[Tuple[float, float]]] = None  # Store bounds for barrier computation
        self._trips: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._raw_metadata: Dict[str, object] = {}
        self._dt: Optional[float] = None
        self._last_speed_accel_distribution: Optional[Dict[str, np.ndarray]] = None
        self._phase_advance_event = threading.Event()
        self.current_phase: Optional[str] = None
        self._extreme_actuator_seen: set[tuple[str, str]] = set()
        self._split_seed: Optional[int] = None
        self._abort_event = threading.Event()
        self._debug_batch_active = False
        self._debug_batch_remaining_calls = 0
        self._debug_batch_label = ""
        self._debug_batch_progress_step = 0.1
        
        # GPU setup
        self._device = None
        if self.config.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                    LOGGER.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    LOGGER.warning("GPU requested but not available, falling back to CPU")
            except ImportError:
                LOGGER.warning("PyTorch not available, falling back to CPU")

    def request_phase_advance(self) -> None:
        """Request early termination of the current optimization phase."""
        self._phase_advance_event.set()

    def request_abort(self) -> None:
        """Request abort of the fitting process."""
        self._abort_event.set()

    def _get_phase_param_names(self, phase: str) -> List[str]:
        brake_params = {"brake_T_max", "brake_tau", "brake_p", "mu"}
        if phase == "brake":
            return [p for p in self.PARAM_NAMES if p in brake_params]
        if phase == "throttle":
            return [p for p in self.PARAM_NAMES if p not in brake_params]
        return list(self.PARAM_NAMES)

    def _freeze_non_phase_params(
        self,
        params: np.ndarray,
        bounds: List[Tuple[float, float]],
        phase: str,
    ) -> List[Tuple[float, float]]:
        keep = set(self._get_phase_param_names(phase))
        new_bounds: List[Tuple[float, float]] = []
        for name, val, b in zip(self.PARAM_NAMES, params, bounds):
            if name in keep:
                new_bounds.append(b)
            else:
                new_bounds.append((float(val), float(val)))
        return new_bounds

    def _filter_segments_for_phase(
        self,
        segments: List[TripSegment],
        phase: str,
    ) -> List[TripSegment]:
        """Filter segments to contiguous actuation-only segments for the phase."""
        if phase not in {"throttle", "brake"}:
            return segments

        throttle_thresh = float(self.config.actuator_deadband_pct)
        brake_thresh = float(self.config.brake_deadband_pct)
        filtered: List[TripSegment] = []

        def _append_run(run_start: int, run_end: int, seg: TripSegment) -> None:
            run_len = run_end - run_start
            if run_len < self.config.min_segment_length:
                return
            max_len = self.config.max_segment_length
            start = run_start
            idx = 0
            while start < run_end:
                end = min(start + max_len, run_end)
                if end - start < self.config.min_segment_length:
                    break
                speed = seg.speed[start:end].copy()
                if speed.size:
                    speed[0] = seg.speed[start]
                sub = TripSegment(
                    trip_id=f"{seg.trip_id}_{phase}_{run_start}_{idx}",
                    speed=speed,
                    acceleration=seg.acceleration[start:end],
                    throttle=seg.throttle[start:end],
                    brake=seg.brake[start:end],
                    grade=seg.grade[start:end],
                    dt=seg.dt,
                    sample_weights=(seg.sample_weights[start:end].copy() if seg.sample_weights is not None else None),
                )
                filtered.append(sub)
                idx += 1
                start = end

        for seg in segments:
            throttle_active = seg.throttle > throttle_thresh
            brake_active = seg.brake > brake_thresh
            if phase == "throttle":
                active_mask = throttle_active & (~brake_active)
            else:
                active_mask = brake_active

            if not np.any(active_mask):
                continue

            in_run = False
            run_start = 0
            for i, active in enumerate(active_mask):
                if active and not in_run:
                    in_run = True
                    run_start = i
                elif not active and in_run:
                    _append_run(run_start, i, seg)
                    in_run = False
            if in_run:
                _append_run(run_start, len(active_mask), seg)

        return filtered

    def _sanitize_actuator(self, signal: np.ndarray, name: str) -> np.ndarray:
        """Sanitize actuator signals to 0-100% range.

        - Converts NaN/Inf to 0
        - If signal is in [0,1], scales to [0,100]
        - Fixes known offset encoding when values are slightly above 100
        - Clips out-of-range values
        """
        cleaned = np.asarray(signal, dtype=np.float64)
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)

        max_val = float(np.max(cleaned)) if cleaned.size else 0.0
        if 100.0 < max_val < 200.0:
            # Known encoding issue: values above 100 may be offset by ~108.47
            offset_mask = cleaned > 100.0
            if np.any(offset_mask):
                cleaned = cleaned.copy()
                cleaned[offset_mask] -= 108.47458
                LOGGER.debug("Applied actuator offset correction for %s (max=%.2f)", name, max_val)
                max_val = float(np.max(cleaned)) if cleaned.size else 0.0

        if max_val <= 1.5:
            cleaned = cleaned * 100.0
            max_val = float(np.max(cleaned)) if cleaned.size else 0.0

        if max_val > 1000.0:
            LOGGER.warning("Actuator %s has extreme values (max=%.2f); clipping to [0,100]", name, max_val)

        return np.clip(cleaned, 0.0, 100.0)

    def _save_actuator_extreme_example(
        self,
        data_path: Path,
        trip_id: str,
        name: str,
        raw: np.ndarray,
        time: Optional[np.ndarray],
    ) -> None:
        """Save actuator examples with extreme values for investigation."""
        key = (trip_id, name)
        if key in self._extreme_actuator_seen:
            return
        self._extreme_actuator_seen.add(key)

        safe_trip_id = str(trip_id).replace("/", "_").replace("\\", "_")
        out_dir = data_path.parent / "actuator_extremes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_base = out_dir / f"{safe_trip_id}_{name}_extreme"

        raw_clean = np.asarray(raw, dtype=np.float64)
        stats = {
            "trip_id": trip_id,
            "actuator": name,
            "max": float(np.nanmax(raw_clean)) if raw_clean.size else 0.0,
            "min": float(np.nanmin(raw_clean)) if raw_clean.size else 0.0,
            "mean": float(np.nanmean(raw_clean)) if raw_clean.size else 0.0,
            "std": float(np.nanstd(raw_clean)) if raw_clean.size else 0.0,
            "num_nan": int(np.isnan(raw_clean).sum()),
            "num_inf": int(np.isinf(raw_clean).sum()),
            "num_samples": int(raw_clean.size),
        }

        np.savez_compressed(
            f"{out_base}.npz",
            raw=raw_clean,
            time=np.asarray(time, dtype=np.float64) if time is not None else np.array([]),
        )
        with open(f"{out_base}.json", "w") as f:
            json.dump(stats, f, indent=2)

        if MATPLOTLIB_AVAILABLE:
            t = np.asarray(time, dtype=np.float64) if time is not None and len(time) else np.arange(raw_clean.size)
            fig = plt.figure(figsize=(10, 4), dpi=120)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(t, raw_clean, color="tab:red", linewidth=1)
            ax.set_title(f"{trip_id} - {name} (raw)")
            ax.set_xlabel("time" if time is not None and len(time) else "index")
            ax.set_ylabel("value")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{out_base}.png")
            plt.close(fig)

    def _apply_actuator_deadband(self, signal: np.ndarray) -> np.ndarray:
        """Apply deadband to actuator signal (values below threshold set to 0)."""
        deadband = self.config.actuator_deadband_pct
        if deadband <= 0:
            return signal
        cleaned = signal.copy()
        cleaned[cleaned < deadband] = 0.0
        return cleaned

    def _smooth_actuator_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to actuator signal."""
        alpha = self.config.actuator_smoothing_alpha
        if alpha <= 0.0 or signal.size == 0:
            return signal
        smoothed = signal.copy()
        for i in range(1, len(smoothed)):
            smoothed[i] = alpha * smoothed[i] + (1.0 - alpha) * smoothed[i - 1]
        return smoothed

    def _params_to_unit(self, params: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Map parameters to [0,1] based on bounds."""
        lows = np.array([b[0] for b in bounds], dtype=np.float64)
        highs = np.array([b[1] for b in bounds], dtype=np.float64)
        scale = highs - lows
        unit = np.zeros_like(params, dtype=np.float64)
        mask = scale > 0
        unit[mask] = (params[mask] - lows[mask]) / scale[mask]
        return unit

    def _params_from_unit(self, unit: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Map parameters from [0,1] back to original bounds."""
        lows = np.array([b[0] for b in bounds], dtype=np.float64)
        highs = np.array([b[1] for b in bounds], dtype=np.float64)
        scale = highs - lows
        params = lows + unit * scale
        return params
    
    def _barrier_penalty(self, params: np.ndarray, bounds: List[Tuple[float, float]]) -> float:
        """Compute logarithmic barrier penalty to keep parameters away from boundaries.
        
        Barrier function: -μ * Σ (log(x_i - l_i) + log(u_i - x_i))
        
        Args:
            params: Parameter vector
            bounds: List of (min, max) tuples for each parameter
            
        Returns:
            Barrier penalty value (positive, added to loss)
        """
        if not self.config.use_barrier:
            return 0.0
        
        mu = self.config.barrier_mu
        eps = 1e-10  # Small epsilon to prevent log(0) and handle numerical issues
        
        penalty_sum = 0.0
        for i, (x, (l, u)) in enumerate(zip(params, bounds)):
            # Skip barrier for fixed parameters (min == max)
            if l == u:
                continue
            
            # Ensure we're within bounds (with small margin for numerical stability)
            if x <= l + eps or x >= u - eps:
                # Return large penalty if too close to boundary
                return 1e10
            
            # Compute barrier terms: log(x - l) + log(u - x)
            # Both terms are negative (since x is between l and u)
            # We negate the sum and multiply by mu to get positive penalty
            penalty_sum += np.log(x - l) + np.log(u - x)
        
        # Return -mu * sum (negative because we want to add this to loss)
        # Since penalty_sum is negative, -mu * penalty_sum is positive
        return -mu * penalty_sum

    def _is_suspicious_segment(
        self,
        speed: np.ndarray,
        accel: np.ndarray,
        throttle: np.ndarray,
        brake: np.ndarray,
    ) -> Tuple[bool, str]:
        """Check for suspicious/low-quality segment patterns that harm training."""
        cfg = self.config

        if speed.size < cfg.min_segment_length:
            return True, "too_short"

        if not (np.isfinite(speed).all() and np.isfinite(accel).all() and np.isfinite(throttle).all() and np.isfinite(brake).all()):
            return True, "non_finite"

        if np.std(speed) < cfg.min_speed_std:
            return True, "low_speed_variance"

        if np.std(accel) < cfg.min_accel_std:
            return True, "low_accel_variance"

        # Flat/constant speed for most of the segment
        speed_diffs = np.abs(np.diff(speed))
        if speed_diffs.size > 0:
            flat_speed_frac = float(np.mean(speed_diffs < cfg.flat_speed_eps))
            if flat_speed_frac > cfg.max_flat_speed_fraction:
                return True, "flat_speed"

        # Flat/constant actuation (throttle or brake) for most of the segment
        th_diffs = np.abs(np.diff(throttle))
        br_diffs = np.abs(np.diff(brake))
        if th_diffs.size > 0 and br_diffs.size > 0:
            flat_th = float(np.mean(th_diffs < cfg.flat_act_eps))
            flat_br = float(np.mean(br_diffs < cfg.flat_act_eps))
            if max(flat_th, flat_br) > cfg.max_flat_act_fraction:
                return True, "flat_actuation"

        # Mostly coasting with zero commands
        zero_cmd_frac = float(np.mean((throttle < 1.0) & (brake < 1.0)))
        if zero_cmd_frac > cfg.max_zero_command_fraction:
            return True, "mostly_zero_commands"

        # Too much saturation
        sat_th = float(np.mean(throttle >= cfg.saturation_threshold))
        sat_br = float(np.mean(brake >= cfg.saturation_threshold))
        if max(sat_th, sat_br) > cfg.max_saturation_fraction:
            return True, "saturation"

        return False, "ok"
    
    def load_trip_data(self, data_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
        """Load trip data from .pt file.
        
        Args:
            data_path: Path to all_trips_data.pt file
            
        Returns:
            Dictionary with trip_id -> {speed, acceleration, throttle, brake, angle, time}
        """
        raw = _load_torch_file_compat(data_path)
        trips = {}

        metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
        self._raw_metadata = metadata if isinstance(metadata, dict) else {}

        metadata_dt: Optional[float] = None
        dt_raw = self._raw_metadata.get("dt")
        if dt_raw is not None:
            try:
                dt_candidate = float(dt_raw)
                if np.isfinite(dt_candidate) and dt_candidate > 0.0:
                    metadata_dt = dt_candidate
            except (TypeError, ValueError):
                metadata_dt = None
        
        for key, value in raw.items():
            if key == "metadata":
                continue
            if not isinstance(value, dict):
                continue
            
            # Extract required fields
            try:
                raw_throttle = np.asarray(value["throttle"], dtype=np.float64)
                raw_brake = np.asarray(value["brake"], dtype=np.float64)
                raw_time = np.asarray(value["time"], dtype=np.float64) if "time" in value else None

                max_throttle = float(np.nanmax(raw_throttle)) if raw_throttle.size else 0.0
                max_brake = float(np.nanmax(raw_brake)) if raw_brake.size else 0.0
                if max_throttle > 1000.0 or not np.isfinite(raw_throttle).all():
                    self._save_actuator_extreme_example(data_path, key, "throttle", raw_throttle, raw_time)
                if max_brake > 1000.0 or not np.isfinite(raw_brake).all():
                    self._save_actuator_extreme_example(data_path, key, "brake", raw_brake, raw_time)

                throttle = self._sanitize_actuator(raw_throttle, "throttle")
                brake = self._sanitize_actuator(raw_brake, "brake")

                # Brake dominance: when braking, suppress throttle to avoid overlap
                brake_active = brake > self.config.brake_deadband_pct
                if np.any(brake_active):
                    throttle = throttle.copy()
                    throttle[brake_active] = 0.0

                throttle = self._apply_actuator_deadband(throttle)
                brake = self._apply_actuator_deadband(brake)
                throttle = self._smooth_actuator_signal(throttle)
                brake = self._smooth_actuator_signal(brake)
                throttle = np.clip(throttle, 0.0, 100.0)
                brake = np.clip(brake, 0.0, 100.0)

                # Re-apply exclusivity after smoothing to prevent overlap
                brake_active = brake > self.config.brake_deadband_pct
                throttle_active = throttle > self.config.actuator_deadband_pct
                if np.any(brake_active):
                    throttle = throttle.copy()
                    throttle[brake_active] = 0.0
                if np.any(throttle_active):
                    brake = brake.copy()
                    brake[throttle_active] = 0.0

                trip_data = {
                    "speed": np.asarray(value["speed"], dtype=np.float64),
                    "acceleration": np.asarray(value["acceleration"], dtype=np.float64),
                    "throttle": throttle,
                    "brake": brake,
                    "angle": np.asarray(value.get("angle", np.zeros_like(value["speed"])), dtype=np.float64),
                }
                # Try to get time directly; if missing, synthesize from metadata dt.
                if raw_time is not None and raw_time.size > 1:
                    trip_data["time"] = raw_time
                elif metadata_dt is not None:
                    trip_data["time"] = np.arange(len(trip_data["speed"]), dtype=np.float64) * metadata_dt
                trips[key] = trip_data
            except KeyError as e:
                LOGGER.warning(f"Trip {key} missing field {e}, skipping")
                continue
        
        LOGGER.info(f"Loaded {len(trips)} trips from {data_path}")
        return trips
    
    def _estimate_dt(self, trips: Dict[str, Dict[str, np.ndarray]]) -> float:
        """Estimate timestep from trip data."""
        metadata_dt: Optional[float] = None
        if isinstance(self._raw_metadata, dict):
            dt_raw = self._raw_metadata.get("dt")
            if dt_raw is not None:
                try:
                    dt_candidate = float(dt_raw)
                    if np.isfinite(dt_candidate) and dt_candidate > 0.0:
                        metadata_dt = dt_candidate
                except (TypeError, ValueError):
                    metadata_dt = None

        dts = []
        for trip_id, data in trips.items():
            if "time" in data:
                t = data["time"]
                if len(t) > 1:
                    diffs = np.diff(t)
                    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                    if diffs.size > 0:
                        dt_trip = float(np.median(diffs))
                        dts.append(dt_trip)

        if metadata_dt is not None and dts:
            dt_from_time = float(np.median(dts))
            rel_err = abs(dt_from_time - metadata_dt) / max(metadata_dt, 1e-12)
            if rel_err > 0.2:
                LOGGER.warning(
                    "Time-derived dt %.6f differs from metadata dt %.6f; using metadata dt",
                    dt_from_time,
                    metadata_dt,
                )
            return metadata_dt

        if metadata_dt is not None:
            return metadata_dt
        
        if dts:
            return float(np.median(dts))
        return self.config.dt  # fallback to config default
    
    def _apply_lpf(self, data: np.ndarray, dt: float, cutoff: float) -> np.ndarray:
        """Apply low-pass filter to data.
        
        Args:
            data: Input data array
            dt: Timestep in seconds
            cutoff: Cutoff frequency in Hz
            
        Returns:
            Filtered data array (or original if filtering fails)
        """
        if len(data) <= 3:
            return data
        
        nyquist = 0.5 / dt
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1.0:
            return data
        
        try:
            from scipy import signal
            b, a = signal.butter(2, normal_cutoff, btype="low")
            return signal.filtfilt(b, a, data)
        except Exception:
            # If filtering fails, return original data
            return data
    
    def _create_segments(
        self, trips: Dict[str, Dict[str, np.ndarray]], dt: float
    ) -> List[TripSegment]:
        """Create trip segments for trajectory simulation.
        
        Splits trips into segments, filtering for quality data.
        """
        cfg = self.config
        segments = []
        
        for trip_id, data in trips.items():
            v = data["speed"]
            a = data["acceleration"]
            th = data["throttle"]
            br = data["brake"]
            grade = data["angle"]
            
            # Apply LPF to speed and acceleration if enabled
            if cfg.apply_lpf_to_fitting_data:
                v = self._apply_lpf(v, dt, cutoff=5.0)  # 5 Hz cutoff for speed
                a = self._apply_lpf(a, dt, cutoff=2.0)  # 2 Hz cutoff for acceleration
            
            n = len(v)
            if n < cfg.min_segment_length:
                continue

            if cfg.disable_segment_filtering:
                finite = (
                    np.isfinite(v) &
                    np.isfinite(a) &
                    np.isfinite(th) &
                    np.isfinite(br) &
                    np.isfinite(grade)
                )
                segment_start = None
                for i in range(n):
                    if finite[i] and segment_start is None:
                        segment_start = i
                    elif (not finite[i] or i == n - 1) and segment_start is not None:
                        end = i if not finite[i] else i + 1
                        length = end - segment_start
                        if length >= cfg.min_segment_length:
                            if cfg.use_whole_trips:
                                segments.append(TripSegment(
                                    trip_id=f"{trip_id}_{segment_start}",
                                    speed=v[segment_start:end].copy(),
                                    acceleration=a[segment_start:end].copy(),
                                    throttle=th[segment_start:end].copy(),
                                    brake=br[segment_start:end].copy(),
                                    grade=grade[segment_start:end].copy(),
                                    dt=dt,
                                ))
                            else:
                                for seg_start in range(segment_start, end, cfg.max_segment_length):
                                    seg_end = min(seg_start + cfg.max_segment_length, end)
                                    seg_len = seg_end - seg_start
                                    if seg_len < cfg.min_segment_length:
                                        continue
                                    segments.append(TripSegment(
                                        trip_id=f"{trip_id}_{seg_start}",
                                        speed=v[seg_start:seg_end].copy(),
                                        acceleration=a[seg_start:seg_end].copy(),
                                        throttle=th[seg_start:seg_end].copy(),
                                        brake=br[seg_start:seg_end].copy(),
                                        grade=grade[seg_start:seg_end].copy(),
                                        dt=dt,
                                    ))
                        segment_start = None
                continue
            
            # Find contiguous valid regions
            valid = (
                (v >= cfg.min_speed) &
                (v <= cfg.max_speed) &
                (np.abs(a) <= cfg.max_accel) &
                np.isfinite(v) &
                np.isfinite(a) &
                np.isfinite(th) &
                np.isfinite(br) &
                np.isfinite(grade)
            )
            
            # Split into contiguous segments
            segment_start = None
            for i in range(n):
                if valid[i] and segment_start is None:
                    segment_start = i
                elif (not valid[i] or i == n - 1) and segment_start is not None:
                    end = i if not valid[i] else i + 1
                    length = end - segment_start
                    
                    if length >= cfg.min_segment_length:
                        if cfg.use_whole_trips:
                            seg_speed = v[segment_start:end].copy()
                            seg_accel = a[segment_start:end].copy()
                            seg_th = th[segment_start:end].copy()
                            seg_br = br[segment_start:end].copy()
                            seg_grade = grade[segment_start:end].copy()

                            suspicious, reason = self._is_suspicious_segment(
                                seg_speed, seg_accel, seg_th, seg_br
                            )
                            if suspicious:
                                LOGGER.debug(
                                    "Skipping segment %s_%d (%s)",
                                    trip_id,
                                    segment_start,
                                    reason,
                                )
                            else:
                                segments.append(TripSegment(
                                    trip_id=f"{trip_id}_{segment_start}",
                                    speed=seg_speed,
                                    acceleration=seg_accel,
                                    throttle=seg_th,
                                    brake=seg_br,
                                    grade=seg_grade,
                                    dt=dt,
                                ))
                        else:
                            # Possibly split long segments
                            for seg_start in range(segment_start, end, cfg.max_segment_length):
                                seg_end = min(seg_start + cfg.max_segment_length, end)
                                seg_len = seg_end - seg_start
                                
                                if seg_len >= cfg.min_segment_length:
                                    seg_speed = v[seg_start:seg_end].copy()
                                    seg_accel = a[seg_start:seg_end].copy()
                                    seg_th = th[seg_start:seg_end].copy()
                                    seg_br = br[seg_start:seg_end].copy()
                                    seg_grade = grade[seg_start:seg_end].copy()

                                    suspicious, reason = self._is_suspicious_segment(
                                        seg_speed, seg_accel, seg_th, seg_br
                                    )
                                    if suspicious:
                                        LOGGER.debug(
                                            "Skipping segment %s_%d (%s)",
                                            trip_id,
                                            seg_start,
                                            reason,
                                        )
                                        continue

                                    segments.append(TripSegment(
                                        trip_id=f"{trip_id}_{seg_start}",
                                        speed=seg_speed,
                                        acceleration=seg_accel,
                                        throttle=seg_th,
                                        brake=seg_br,
                                        grade=seg_grade,
                                        dt=dt,
                                    ))
                    
                    segment_start = None

        if not segments:
            LOGGER.warning("No segments passed filters; falling back to raw trips")
            raw_lengths: List[int] = []
            for trip_id, data in trips.items():
                v = data["speed"]
                a = data["acceleration"]
                th = data["throttle"]
                br = data["brake"]
                grade = data["angle"]
                raw_lengths.append(len(v))
                if len(v) < cfg.min_segment_length:
                    continue
                segments.append(TripSegment(
                    trip_id=f"{trip_id}_raw",
                    speed=np.asarray(v, dtype=np.float64),
                    acceleration=np.asarray(a, dtype=np.float64),
                    throttle=np.asarray(th, dtype=np.float64),
                    brake=np.asarray(br, dtype=np.float64),
                    grade=np.asarray(grade, dtype=np.float64),
                    dt=dt,
                ))

            if not segments and raw_lengths:
                max_len = max(raw_lengths)
                relaxed_min = max(3, min(cfg.min_segment_length, max_len))
                if max_len >= relaxed_min:
                    LOGGER.warning(
                        "All raw trips shorter than min_segment_length=%d (max=%d). "
                        "Relaxing fallback minimum to %d samples.",
                        cfg.min_segment_length,
                        max_len,
                        relaxed_min,
                    )
                    for trip_id, data in trips.items():
                        v = np.asarray(data["speed"], dtype=np.float64)
                        if len(v) < relaxed_min:
                            continue
                        segments.append(TripSegment(
                            trip_id=f"{trip_id}_raw_relaxed",
                            speed=v,
                            acceleration=np.asarray(data["acceleration"], dtype=np.float64),
                            throttle=np.asarray(data["throttle"], dtype=np.float64),
                            brake=np.asarray(data["brake"], dtype=np.float64),
                            grade=np.asarray(data["angle"], dtype=np.float64),
                            dt=dt,
                        ))

        LOGGER.info(f"Created {len(segments)} segments from {len(trips)} trips")
        return segments

    def _build_equal_width_edges(self, values: np.ndarray, n_bins: int) -> np.ndarray:
        """Build robust equal-width bin edges for 1D values."""
        if values.size == 0:
            return np.linspace(-1.0, 1.0, n_bins + 1, dtype=np.float64)

        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return np.linspace(-1.0, 1.0, n_bins + 1, dtype=np.float64)

        if abs(vmax - vmin) < 1e-12:
            pad = max(abs(vmin) * 0.05, 1e-3)
            vmin -= pad
            vmax += pad

        return np.linspace(vmin, vmax, n_bins + 1, dtype=np.float64)

    def _bin_pair_indices(
        self,
        speed: np.ndarray,
        accel: np.ndarray,
        speed_edges: np.ndarray,
        accel_edges: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map speed/accel samples to 2D bin indices."""
        speed_idx = np.digitize(speed, speed_edges, right=False) - 1
        accel_idx = np.digitize(accel, accel_edges, right=False) - 1
        speed_idx = np.clip(speed_idx, 0, len(speed_edges) - 2)
        accel_idx = np.clip(accel_idx, 0, len(accel_edges) - 2)
        return speed_idx.astype(np.int64), accel_idx.astype(np.int64)

    def compute_speed_accel_distribution(
        self,
        segments: List[TripSegment],
        speed_bins: Optional[int] = None,
        accel_bins: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute a 2D histogram over (speed, acceleration) for the provided segments."""
        n_speed_bins = int(speed_bins if speed_bins is not None else self.config.speed_accel_speed_bins)
        n_accel_bins = int(accel_bins if accel_bins is not None else self.config.speed_accel_accel_bins)
        n_speed_bins = max(n_speed_bins, 2)
        n_accel_bins = max(n_accel_bins, 2)

        speed_min, speed_max = self.config.speed_accel_speed_range
        accel_min, accel_max = self.config.speed_accel_accel_range
        speed_edges = np.linspace(float(speed_min), float(speed_max), n_speed_bins + 1, dtype=np.float64)
        accel_edges = np.linspace(float(accel_min), float(accel_max), n_accel_bins + 1, dtype=np.float64)

        if not segments:
            counts = np.zeros((n_speed_bins, n_accel_bins), dtype=np.int64)
            return {
                "speed_edges": speed_edges,
                "accel_edges": accel_edges,
                "counts": counts,
                "total_samples": np.array([0], dtype=np.int64),
                "nonzero_bins": np.array([0], dtype=np.int64),
            }
        all_speed = np.concatenate([seg.speed for seg in segments]).astype(np.float64)
        all_accel = np.concatenate([seg.acceleration for seg in segments]).astype(np.float64)

        counts = np.zeros((n_speed_bins, n_accel_bins), dtype=np.int64)
        for seg in segments:
            s_idx, a_idx = self._bin_pair_indices(seg.speed, seg.acceleration, speed_edges, accel_edges)
            np.add.at(counts, (s_idx, a_idx), 1)

        return {
            "speed_edges": speed_edges,
            "accel_edges": accel_edges,
            "counts": counts,
            "total_samples": np.array([all_speed.size], dtype=np.int64),
            "nonzero_bins": np.array([int(np.count_nonzero(counts))], dtype=np.int64),
        }

    def _compute_uniform_weight_map(self, counts: np.ndarray) -> np.ndarray:
        """Compute per-bin sample weights so non-empty bins contribute more uniformly."""
        counts_f = counts.astype(np.float64)
        total_samples = float(np.sum(counts_f))
        nonzero = counts_f > 0.0
        nonzero_bins = int(np.count_nonzero(nonzero))

        weight_map = np.zeros_like(counts_f, dtype=np.float64)
        if total_samples <= 0.0 or nonzero_bins == 0:
            return weight_map

        normalizer = total_samples / float(nonzero_bins)
        weight_map[nonzero] = normalizer / counts_f[nonzero]

        cap = float(self.config.speed_accel_bin_weight_cap)
        if cap > 0.0:
            weight_map[nonzero] = np.minimum(weight_map[nonzero], cap)

        weighted_sum = float(np.sum(weight_map * counts_f))
        if weighted_sum > 0.0:
            mean_weight = weighted_sum / total_samples
            weight_map[nonzero] /= mean_weight

        return weight_map

    def apply_uniform_speed_accel_bucket_weights(
        self,
        segments: List[TripSegment],
    ) -> Dict[str, np.ndarray]:
        """Assign per-step weights based on (speed, acceleration) bucket occupancy."""
        distribution = self.compute_speed_accel_distribution(segments)
        speed_edges = distribution["speed_edges"]
        accel_edges = distribution["accel_edges"]
        counts = distribution["counts"]
        weight_map = self._compute_uniform_weight_map(counts)

        for seg in segments:
            s_idx, a_idx = self._bin_pair_indices(seg.speed, seg.acceleration, speed_edges, accel_edges)
            seg.sample_weights = weight_map[s_idx, a_idx].astype(np.float64)

        distribution["weight_map"] = weight_map
        return distribution

    def get_longest_validation_display_segment(self) -> Optional[TripSegment]:
        """Return the longest validation segment without max-length splitting.

        This uses the validation trip IDs and rebuilds contiguous valid regions
        without applying max_segment_length, so the GUI can display a full
        validation trip segment.
        """
        if not hasattr(self, "val_segments") or not self.val_segments:
            return None
        if self._trips is None:
            return None

        dt = self._dt if self._dt is not None else self.config.dt
        val_trip_ids = {seg.trip_id.rsplit("_", 1)[0] for seg in self.val_segments}

        cfg = self.config
        display_segments: List[TripSegment] = []

        for trip_id, data in self._trips.items():
            if trip_id not in val_trip_ids:
                continue

            v = data["speed"]
            a = data["acceleration"]
            th = data["throttle"]
            br = data["brake"]
            grade = data["angle"]

            n = len(v)
            if n < cfg.min_segment_length:
                continue

            if cfg.disable_segment_filtering:
                finite = (
                    np.isfinite(v) &
                    np.isfinite(a) &
                    np.isfinite(th) &
                    np.isfinite(br) &
                    np.isfinite(grade)
                )
                segment_start = None
                for i in range(n):
                    if finite[i] and segment_start is None:
                        segment_start = i
                    elif (not finite[i] or i == n - 1) and segment_start is not None:
                        end = i if not finite[i] else i + 1
                        length = end - segment_start
                        if length >= cfg.min_segment_length:
                            display_segments.append(TripSegment(
                                trip_id=f"{trip_id}_full",
                                speed=v[segment_start:end].copy(),
                                acceleration=a[segment_start:end].copy(),
                                throttle=th[segment_start:end].copy(),
                                brake=br[segment_start:end].copy(),
                                grade=grade[segment_start:end].copy(),
                                dt=dt,
                            ))
                        segment_start = None
                continue

            valid = (
                (v >= cfg.min_speed) &
                (v <= cfg.max_speed) &
                (np.abs(a) <= cfg.max_accel) &
                np.isfinite(v) &
                np.isfinite(a) &
                np.isfinite(th) &
                np.isfinite(br) &
                np.isfinite(grade)
            )

            segment_start = None
            for i in range(n):
                if valid[i] and segment_start is None:
                    segment_start = i
                elif (not valid[i] or i == n - 1) and segment_start is not None:
                    end = i if not valid[i] else i + 1
                    length = end - segment_start
                    if length >= cfg.min_segment_length:
                        seg_speed = v[segment_start:end].copy()
                        seg_accel = a[segment_start:end].copy()
                        seg_th = th[segment_start:end].copy()
                        seg_br = br[segment_start:end].copy()
                        seg_grade = grade[segment_start:end].copy()

                        suspicious, reason = self._is_suspicious_segment(
                            seg_speed, seg_accel, seg_th, seg_br
                        )
                        if suspicious:
                            LOGGER.debug(
                                "Skipping display segment %s (%s)",
                                trip_id,
                                reason,
                            )
                        else:
                            display_segments.append(TripSegment(
                                trip_id=f"{trip_id}_full",
                                speed=seg_speed,
                                acceleration=seg_accel,
                                throttle=seg_th,
                                brake=seg_br,
                                grade=seg_grade,
                                dt=dt,
                            ))
                    segment_start = None

        if not display_segments:
            return None
        return max(display_segments, key=lambda s: s.length)

    def get_longest_training_display_segment(self) -> Optional[TripSegment]:
        """Return the longest training segment without max-length splitting.

        This uses the training trip IDs and rebuilds contiguous valid regions
        without applying max_segment_length.
        """
        if not hasattr(self, "train_segments") or not self.train_segments:
            return None
        if self._trips is None:
            return None

        dt = self._dt if self._dt is not None else self.config.dt
        train_trip_ids = {seg.trip_id.rsplit("_", 1)[0] for seg in self.train_segments}

        cfg = self.config
        display_segments: List[TripSegment] = []

        for trip_id, data in self._trips.items():
            if trip_id not in train_trip_ids:
                continue

            v = data["speed"]
            a = data["acceleration"]
            th = data["throttle"]
            br = data["brake"]
            grade = data["angle"]

            n = len(v)
            if n < cfg.min_segment_length:
                continue

            if cfg.disable_segment_filtering:
                finite = (
                    np.isfinite(v) &
                    np.isfinite(a) &
                    np.isfinite(th) &
                    np.isfinite(br) &
                    np.isfinite(grade)
                )
                segment_start = None
                for i in range(n):
                    if finite[i] and segment_start is None:
                        segment_start = i
                    elif (not finite[i] or i == n - 1) and segment_start is not None:
                        end = i if not finite[i] else i + 1
                        length = end - segment_start
                        if length >= cfg.min_segment_length:
                            display_segments.append(TripSegment(
                                trip_id=f"{trip_id}_{segment_start}",
                                speed=np.asarray(v[segment_start:end], dtype=np.float64),
                                acceleration=np.asarray(a[segment_start:end], dtype=np.float64),
                                throttle=np.asarray(th[segment_start:end], dtype=np.float64),
                                brake=np.asarray(br[segment_start:end], dtype=np.float64),
                                grade=np.asarray(grade[segment_start:end], dtype=np.float64),
                                dt=dt,
                            ))
                        segment_start = None
                continue

            valid = self._get_valid_mask(v, a, th, br, grade)
            segment_start = None
            for i in range(n):
                if valid[i] and segment_start is None:
                    segment_start = i
                elif (not valid[i] or i == n - 1) and segment_start is not None:
                    end = i if not valid[i] else i + 1
                    length = end - segment_start
                    if length >= cfg.min_segment_length:
                        seg_speed = v[segment_start:end].copy()
                        seg_accel = a[segment_start:end].copy()
                        seg_th = th[segment_start:end].copy()
                        seg_br = br[segment_start:end].copy()
                        seg_grade = grade[segment_start:end].copy()

                        suspicious, _reason = self._is_suspicious_segment(
                            seg_speed, seg_accel, seg_th, seg_br
                        )
                        if not suspicious:
                            display_segments.append(TripSegment(
                                trip_id=f"{trip_id}_{segment_start}",
                                speed=seg_speed,
                                acceleration=seg_accel,
                                throttle=seg_th,
                                brake=seg_br,
                                grade=seg_grade,
                                dt=dt,
                            ))
                    segment_start = None

        if not display_segments:
            return None

        return max(display_segments, key=lambda s: s.length)
    
    def _downsample_segments(
        self,
        segments: List[TripSegment],
        factor: int,
    ) -> List[TripSegment]:
        """Downsample segments by taking every Nth sample.
        
        Args:
            segments: List of trip segments
            factor: Downsampling factor (1 = no downsampling, 2 = every 2nd sample, etc.)
            
        Returns:
            List of downsampled segments (dt is updated accordingly)
        """
        if factor <= 1:
            return segments
        
        downsampled = []
        for segment in segments:
            # Take every Nth sample
            indices = np.arange(0, segment.length, factor)
            
            if len(indices) < self.config.min_segment_length:
                # Skip segments that become too short after downsampling
                continue
            
            downsampled.append(TripSegment(
                trip_id=segment.trip_id,
                speed=segment.speed[indices].copy(),
                acceleration=segment.acceleration[indices].copy(),
                throttle=segment.throttle[indices].copy(),
                brake=segment.brake[indices].copy(),
                grade=segment.grade[indices].copy(),
                dt=segment.dt * factor,  # Update dt to reflect new sampling rate
                sample_weights=(segment.sample_weights[indices].copy() if segment.sample_weights is not None else None),
            ))
        
        LOGGER.info(f"Downsampled {len(segments)} segments to {len(downsampled)} segments (factor={factor})")
        return downsampled
    
    def _filter_zero_speed_segments(
        self,
        segments: List[TripSegment],
        max_fraction: float,
        eps: float,
    ) -> List[TripSegment]:
        """Filter segments to limit the fraction of zero-speed segments.
        
        A segment is considered "zero-speed" if its median speed is below eps.
        This prevents the dataset from being dominated by stationary/idle segments.
        Using median instead of mean makes the classification more robust to outliers.
        
        Args:
            segments: List of trip segments
            max_fraction: Maximum fraction of segments that can be zero-speed (0.05 = 5%)
            eps: Epsilon threshold for zero speed (m/s)
            
        Returns:
            Filtered list of segments
        """
        if max_fraction >= 1.0:
            return segments  # No filtering needed
        
        # Classify segments as zero-speed or not
        zero_speed_segments = []
        non_zero_segments = []
        
        for segment in segments:
            median_speed = np.median(segment.speed)
            if median_speed < eps:
                zero_speed_segments.append(segment)
            else:
                non_zero_segments.append(segment)
        
        n_total = len(segments)
        n_zero = len(zero_speed_segments)
        current_fraction = n_zero / n_total if n_total > 0 else 0.0
        
        if current_fraction <= max_fraction:
            # Already within limit, no filtering needed
            return segments
        
        # Need to filter: keep all non-zero segments, limit zero-speed segments
        max_zero_allowed = int(n_total * max_fraction)
        
        # If we have too many zero-speed segments, randomly sample to keep only max_zero_allowed
        if n_zero > max_zero_allowed:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_zero, size=max_zero_allowed, replace=False)
            zero_speed_segments = [zero_speed_segments[i] for i in indices]
        
        filtered = non_zero_segments + zero_speed_segments
        LOGGER.info(
            f"Filtered zero-speed segments: {n_zero}/{n_total} ({current_fraction:.1%}) -> "
            f"{len(zero_speed_segments)}/{len(filtered)} ({len(zero_speed_segments)/len(filtered):.1%})"
        )
        return filtered

    def _sample_fixed_length_batch(
        self,
        segments: List[TripSegment],
        batch_size: int,
        length: int,
        rng: np.random.Generator,
    ) -> List[TripSegment]:
        """Sample a batch of fixed-length segments from existing segments."""
        if batch_size <= 0:
            return []

        target_len = max(int(length), 1)
        eligible = [s for s in segments if s.length >= target_len]
        if not eligible:
            LOGGER.warning(
                "No segments long enough for random sampling (len=%d); falling back to original batch.",
                target_len,
            )
            return [segments[i] for i in rng.integers(0, len(segments), size=batch_size)]

        batch: List[TripSegment] = []
        for _ in range(batch_size):
            seg = eligible[int(rng.integers(0, len(eligible)))]
            start = int(rng.integers(0, seg.length - target_len + 1))
            end = start + target_len
            batch.append(TripSegment(
                trip_id=f"{seg.trip_id}_rand_{start}",
                speed=seg.speed[start:end].copy(),
                acceleration=seg.acceleration[start:end].copy(),
                throttle=seg.throttle[start:end].copy(),
                brake=seg.brake[start:end].copy(),
                grade=seg.grade[start:end].copy(),
                dt=seg.dt,
                sample_weights=(seg.sample_weights[start:end].copy() if seg.sample_weights is not None else None),
            ))
        return batch
    
    def _compute_polynomial_motor_torque(
        self,
        V_norm: float,
        omega_m: float,
        coeffs: np.ndarray,
    ) -> float:
        """Compute motor torque from polynomial map.
        
        Polynomial: τ_m = c_00 + c_10*V + c_01*ω + c_20*V² + c_11*V*ω + c_02*ω² 
                   + c_30*V³ + c_21*V²*ω + c_12*V*ω² + c_03*ω³
        
        Args:
            V_norm: Normalized voltage (V_cmd / V_max), range [0, 1]
            omega_m: Motor angular speed (rad/s)
            coeffs: Array of 10 coefficients [c_00, c_10, c_01, c_20, c_11, c_02, c_30, c_21, c_12, c_03]
            
        Returns:
            Motor torque (Nm), clamped to non-negative
        """
        # Check for invalid coefficients (fast check)
        if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
            return 0.0
        
        c_00, c_10, c_01, c_20, c_11, c_02, c_30, c_21, c_12, c_03 = coeffs
        
        # Clamp inputs to reasonable ranges to prevent overflow
        V = np.clip(V_norm, 0.0, 1.0)
        w = omega_m
        
        # Compute polynomial terms efficiently
        # Constant
        tau = c_00
        
        # Linear terms
        tau += c_10 * V
        tau += c_01 * w
        
        # Quadratic terms
        V2 = V * V
        w2 = w * w
        tau += c_20 * V2
        tau += c_11 * V * w
        tau += c_02 * w2
        
        # Cubic terms
        tau += c_30 * V2 * V
        tau += c_21 * V2 * w
        tau += c_12 * V * w2
        tau += c_03 * w2 * w
        
        # Check for invalid result
        if np.isnan(tau) or np.isinf(tau):
            return 0.0
        
        # Clamp to non-negative (no regeneration)
        return max(tau, 0.0)
    
    def _fit_dc_from_polynomial_map(
        self,
        poly_params: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Fit DC motor parameters to match polynomial motor map.
        
        Samples polynomial map over V and ω ranges and fits DC model:
        τ_dc = K * (V - K*ω)/R - b*ω
        
        Args:
            poly_params: Parameter array from polynomial model fitting
            verbose: Print progress
            
        Returns:
            Dictionary with fitted DC parameters: R, K, b
        """
        # Extract polynomial parameters
        V_max = poly_params[3]
        poly_coeffs = poly_params[4:14]  # 10 coefficients
        
        # Sample polynomial map
        V_samples = np.linspace(0.1, 1.0, 20)  # Normalized voltage
        omega_max = 1000.0  # rad/s - reasonable max motor speed
        omega_samples = np.linspace(0.0, omega_max, 30)
        
        V_grid, omega_grid = np.meshgrid(V_samples, omega_samples)
        V_flat = V_grid.flatten()
        omega_flat = omega_grid.flatten()
        
        # Compute polynomial torques
        tau_poly = np.array([
            self._compute_polynomial_motor_torque(V_n, w, poly_coeffs)
            for V_n, w in zip(V_flat, omega_flat)
        ])
        
        # Filter out zero torques (not useful for fitting)
        valid = tau_poly > 1e-3
        V_valid = V_flat[valid]
        omega_valid = omega_flat[valid]
        tau_valid = tau_poly[valid]
        
        if len(tau_valid) < 10:
            LOGGER.warning("Not enough valid samples for DC fitting")
            return {"R": 0.2, "K": 0.2, "b": 1e-3}
        
        # Convert normalized V to actual voltage
        V_actual = V_valid * V_max
        
        # Fit DC model: τ = K * (V - K*ω)/R - b*ω
        # Rearranging: τ = (K*V)/R - (K²*ω)/R - b*ω
        # Let a = K/R, b_eff = (K²/R + b)
        # τ = a*V - b_eff*ω
        
        # Linear regression: τ = a*V - b_eff*ω
        A = np.column_stack([V_actual, -omega_valid])
        coeffs, residuals, rank, s = np.linalg.lstsq(A, tau_valid, rcond=None)
        
        a = coeffs[0]  # K/R
        b_eff = coeffs[1]  # K²/R + b
        
        # Solve for R, K, b
        # We need additional constraint. Use typical K value to estimate
        # Or use least squares with constraint
        
        # Simple approach: assume K is in reasonable range, solve for R and b
        # Try multiple K values and pick best fit
        K_candidates = np.linspace(0.05, 0.5, 20)
        best_error = float('inf')
        best_R, best_K, best_b = 0.2, 0.2, 1e-3
        
        for K in K_candidates:
            R = K / a if a > 0 else 0.2
            if R < 0.01 or R > 1.0:
                continue
            
            b = b_eff - (K * K / R)
            if b < 0 or b > 0.1:
                continue
            
            # Evaluate fit
            tau_dc = K * (V_actual - K * omega_valid) / R - b * omega_valid
            tau_dc = np.maximum(tau_dc, 0.0)  # Clamp to non-negative
            
            error = np.mean((tau_dc - tau_valid) ** 2)
            if error < best_error:
                best_error = error
                best_R, best_K, best_b = R, K, b
        
        if verbose:
            print(f"\nFitted DC parameters from polynomial map:")
            print(f"  R: {best_R:.4f} Ω")
            print(f"  K: {best_K:.4f} Nm/A")
            print(f"  b: {best_b:.6f} Nm·s/rad")
            print(f"  Fit RMSE: {np.sqrt(best_error):.4f} Nm")
        
        return {
            "R": float(best_R),
            "K": float(best_K),
            "b": float(best_b),
            "fit_rmse": float(np.sqrt(best_error)),
        }
    
    def _compute_acceleration(
        self,
        params: np.ndarray,
        speed: float,
        throttle: float,
        brake: float,
        grade: float,
    ) -> float:
        """Compute acceleration for a single timestep.
        
        Supports both DC motor model and polynomial motor map.
        
        NOTE: This simplified acceleration model uses the same zero-throttle current
        floor concept as the ExtendedPlant via `motor_min_current_A`.
        """
        if self.config.motor_model_type == "polynomial":
            # Polynomial model: 25 parameters
            (mass, drag_area, rolling_coeff,
             V_max, gamma_throttle, throttle_tau,
             poly_c_00, poly_c_10, poly_c_01, poly_c_20, poly_c_11, poly_c_02,
             poly_c_30, poly_c_21, poly_c_12, poly_c_03,
             gear_ratio, eta,
             brake_T_max, brake_tau, brake_p, mu,
             r_w, wheel_inertia,
             *_) = params
            
            # Motor speed from wheel speed
            omega_m = gear_ratio * speed / max(r_w, 1e-3)
            
            # Commanded voltage and normalized voltage
            throttle_frac = max(throttle, 0.0) / 100.0
            V_cmd = (throttle_frac ** gamma_throttle) * V_max
            V_norm = V_cmd / max(V_max, 1e-6)
            
            # Polynomial coefficients
            poly_coeffs = np.array([
                poly_c_00, poly_c_10, poly_c_01, poly_c_20, poly_c_11, poly_c_02,
                poly_c_30, poly_c_21, poly_c_12, poly_c_03
            ])
            
            # Compute motor torque from polynomial map
            motor_torque = self._compute_polynomial_motor_torque(V_norm, omega_m, poly_coeffs)
            
            # Wheel torque through gearbox
            wheel_torque = eta * gear_ratio * motor_torque
            
            # For polynomial model, use default motor inertia (not fitted)
            J = 1e-3  # Default motor rotor inertia (kg·m²)
        else:
            # DC motor model: 21 parameters
            (mass, drag_area, rolling_coeff,
             V_max, R, K, b, J, gamma_throttle, throttle_tau, T_max, P_max,
             gear_ratio, eta,
             brake_T_max, brake_tau, brake_p, mu,
             r_w, wheel_inertia,
             *tail) = params

            motor_min_current_A = float(tail[0]) if len(tail) > 0 else self.config.motor_min_current_A_init
            
            # Motor speed from wheel speed
            omega_m = gear_ratio * speed / max(r_w, 1e-3)
            
            # Current control (quasi-steady, no regen)
            I_max = (T_max / max(K, 1e-6)) if T_max > 0.0 else (V_max / max(R, 1e-4))
            back_emf = K * omega_m
            throttle_frac = max(throttle, 0.0) / 100.0
            i_floor = max(motor_min_current_A, 0.0)
            i_span = max(I_max - i_floor, 0.0)
            target_current = i_floor + (throttle_frac ** gamma_throttle) * i_span
            v_required = target_current * R + back_emf
            v_applied = min(v_required, V_max)
            motor_current = max((v_applied - back_emf) / max(R, 1e-4), 0.0)
            if P_max > 0.0 and v_applied > 1e-6:
                motor_current = min(motor_current, P_max / v_applied)
            
            # Motor torque with viscous friction loss
            motor_torque = K * motor_current - b * omega_m
            motor_torque = max(motor_torque, 0.0)
            
            # Wheel torque through gearbox
            wheel_torque = eta * gear_ratio * motor_torque
        
        # Drive force
        F_drive = wheel_torque / max(r_w, 1e-3)
        
        # Brake force with nonlinear power-law response
        brake_cmd = max(brake, 0.0) / 100.0
        brake_p_eff = max(brake_p, 0.1)
        brake_frac = brake_cmd ** brake_p_eff
        F_brake = brake_T_max * brake_frac / max(r_w, 1e-3)
        
        # Aerodynamic drag
        F_drag = 0.5 * AIR_DENSITY * drag_area * speed * abs(speed)

        if self.config.optimize_without_grade:
            grade = 0.0
        
        # Rolling resistance
        cos_grade = np.cos(grade)
        F_roll = rolling_coeff * mass * GRAVITY * cos_grade
        
        # Grade resistance
        sin_grade = np.sin(grade)
        F_grade = mass * GRAVITY * sin_grade
        
        # Net force and acceleration
        # Effective mass includes rotational inertia of wheels and motor
        effective_mass = mass + (4 * wheel_inertia + J * gear_ratio**2) / max(r_w**2, 1e-6)
        F_net = F_drive - F_brake - F_drag - F_roll - F_grade
        a = F_net / max(effective_mass, 1.0)
        
        return a
    
    def _simulate_segment(
        self,
        params: np.ndarray,
        segment: TripSegment,
        debug_progress: bool = False,
        debug_label: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a full trip segment and return predicted velocities and accelerations.
        
        Args:
            params: Parameter vector
            segment: Trip segment with inputs and initial conditions
            
        Returns:
            Tuple of (v_sim, a_sim)
        """
        if self.config.use_extended_plant and self.config.motor_model_type == "dc":
            return self._simulate_segment_extended(
                params,
                segment,
                debug_progress=debug_progress,
                debug_label=debug_label,
            )

        n = segment.length
        v_sim = np.zeros(n)
        v_sim[0] = segment.initial_speed
        a_sim = np.zeros(n)

        throttle_tau = self.config.motor_throttle_tau_init
        if "motor_throttle_tau" in self.PARAM_NAMES and len(params) > self.PARAM_NAMES.index("motor_throttle_tau"):
            throttle_tau = float(params[self.PARAM_NAMES.index("motor_throttle_tau")])
        throttle_tau = max(throttle_tau, 1e-4)
        throttle_state = float(segment.throttle[0]) if n > 0 else 0.0
        
        next_print = self._debug_batch_progress_step if debug_progress else None
        for t in range(n - 1):
            throttle_raw = float(segment.throttle[t])
            alpha_th = segment.dt / throttle_tau
            throttle_state += alpha_th * (throttle_raw - throttle_state)
            a = self._compute_acceleration(
                params,
                v_sim[t],
                throttle_state,
                segment.brake[t],
                0.0 if self.config.optimize_without_grade else segment.grade[t],
            )
            a_sim[t] = a
            # Euler integration with speed clamp (no negative speeds)
            v_sim[t + 1] = max(v_sim[t] + a * segment.dt, 0.0)

            if debug_progress and next_print is not None:
                frac = (t + 1) / max(n - 1, 1)
                if frac >= next_print:
                    print(
                        f"[{debug_label}] segment {segment.trip_id}: "
                        f"{frac * 100:.0f}% ({t + 1}/{n - 1})"
                    )
                    next_print += self._debug_batch_progress_step
        
        return v_sim, a_sim

    def _build_extended_plant_params(self, params: np.ndarray) -> ExtendedPlantParams:
        """Create ExtendedPlantParams from current DC fitted parameter array."""
        values = np.asarray(params, dtype=np.float64).flatten()
        expected_names = [
            "mass", "drag_area", "rolling_coeff",
            "V_max", "R", "K", "b", "J", "gamma_throttle", "throttle_tau", "T_max", "P_max",
            "gear_ratio", "eta",
            "brake_T_max", "brake_tau", "brake_p", "mu",
            "r_w", "wheel_inertia",
            "motor_min_current_A",
        ]
        if values.size != len(expected_names):
            raise ValueError(
                f"Unsupported DC parameter length for ExtendedPlant build: {values.size}. "
                f"Expected {len(expected_names)} parameters."
            )
        param_map = {name: float(value) for name, value in zip(expected_names, values)}

        mass = param_map["mass"]
        drag_area = param_map["drag_area"]
        rolling_coeff = param_map["rolling_coeff"]
        V_max = param_map["V_max"]
        R = param_map["R"]
        K = param_map["K"]
        b = param_map["b"]
        J = param_map["J"]
        gamma_throttle = param_map["gamma_throttle"]
        throttle_tau = param_map["throttle_tau"]
        T_max = param_map["T_max"]
        P_max = param_map["P_max"]
        gear_ratio = param_map["gear_ratio"]
        eta = param_map["eta"]
        brake_T_max = param_map["brake_T_max"]
        brake_tau = param_map["brake_tau"]
        brake_p = param_map["brake_p"]
        mu = param_map["mu"]
        r_w = param_map["r_w"]
        wheel_inertia = param_map["wheel_inertia"]
        motor_min_current_A = param_map["motor_min_current_A"]

        t_max_val = T_max if T_max > 0.0 else None
        p_max_val = P_max if P_max > 0.0 else None

        motor = MotorParams(
            R=R,
            K_e=K,
            K_t=K,
            b=b,
            J=J,
            V_max=V_max,
            T_max=t_max_val,
            P_max=p_max_val,
            gamma_throttle=gamma_throttle,
            throttle_tau=throttle_tau,
            min_current_A=max(motor_min_current_A, 0.0),
            gear_ratio=gear_ratio,
            eta_gb=eta,
        )

        brake_tau_val = 1e-4 if self.config.extended_plant_zero_brake_lag else brake_tau
        brake = BrakeParams(
            T_br_max=brake_T_max,
            p_br=brake_p,
            tau_br=brake_tau_val,
            mu=mu,
        )

        body = BodyParams(
            mass=mass,
            drag_area=drag_area,
            rolling_coeff=rolling_coeff,
            grade_rad=0.0,
            air_density=AIR_DENSITY,
        )

        wheel = WheelParams(
            radius=r_w,
            inertia=wheel_inertia,
            v_eps=0.1,
        )

        return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel)

    def _simulate_segment_extended(
        self,
        params: np.ndarray,
        segment: TripSegment,
        debug_progress: bool = False,
        debug_label: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a full trip segment using ExtendedPlant dynamics."""
        ext_params = self._build_extended_plant_params(params)
        plant = ExtendedPlant(ext_params)

        n = segment.length
        v_sim = np.zeros(n)
        a_sim = np.zeros(n)
        plant.reset(speed=segment.initial_speed, position=0.0)
        v_sim[0] = segment.initial_speed

        substeps = max(int(self.config.extended_plant_substeps), 1)
        next_print = self._debug_batch_progress_step if debug_progress else None
        for t in range(n - 1):
            throttle = float(segment.throttle[t]) / 100.0
            brake = float(segment.brake[t]) / 100.0

            brake_active = brake * 100.0 > self.config.brake_deadband_pct
            if brake_active:
                action = -brake
            else:
                action = throttle

            action = float(np.clip(action, -1.0, 1.0))
            grade = 0.0 if self.config.optimize_without_grade else float(segment.grade[t])
            state = plant.step(action, segment.dt, substeps=substeps, grade_rad=grade)
            v_sim[t + 1] = state.speed
            a_sim[t] = state.acceleration

            if debug_progress and next_print is not None:
                frac = (t + 1) / max(n - 1, 1)
                if frac >= next_print:
                    print(
                        f"[{debug_label}] segment {segment.trip_id}: "
                        f"{frac * 100:.0f}% ({t + 1}/{n - 1})"
                    )
                    next_print += self._debug_batch_progress_step

        return v_sim, a_sim
    
    def _compute_acceleration_torch(
        self,
        params: torch.Tensor,
        speed: torch.Tensor,
        throttle: torch.Tensor,
        brake: torch.Tensor,
        grade: torch.Tensor,
    ) -> torch.Tensor:
        """Compute acceleration vectorized on GPU using PyTorch.
        
        Args:
            params: Parameter tensor [batch_size, n_params] or [n_params]
            speed: Speed tensor [batch_size] or scalar
            throttle: Throttle tensor [batch_size] or scalar
            brake: Brake tensor [batch_size] or scalar
            grade: Grade tensor [batch_size] or scalar
            
        Returns:
            Acceleration tensor [batch_size] or scalar
        """
        if self.config.motor_model_type == "polynomial":
            # Polynomial model: 25 parameters
            mass = params[..., 0]
            drag_area = params[..., 1]
            rolling_coeff = params[..., 2]
            V_max = params[..., 3]
            gamma_throttle = params[..., 4]
            throttle_tau = params[..., 5]
            poly_coeffs = params[..., 6:16]  # 10 coefficients
            gear_ratio = params[..., 16]
            eta = params[..., 17]
            brake_T_max = params[..., 18]
            brake_tau = params[..., 19]
            brake_p = params[..., 20]
            mu = params[..., 21]
            r_w = params[..., 22]
            wheel_inertia = params[..., 23]
            
            # Motor speed from wheel speed
            omega_m = gear_ratio * speed / torch.clamp(r_w, min=1e-3)
            
            # Commanded voltage and normalized voltage
            throttle_frac = torch.clamp(throttle, min=0.0) / 100.0
            V_cmd = torch.pow(throttle_frac, gamma_throttle) * V_max
            V_norm = V_cmd / torch.clamp(V_max, min=1e-6)
            
            # Compute polynomial motor torque
            V = torch.clamp(V_norm, 0.0, 1.0)
            w = torch.clamp(omega_m, -2000.0, 2000.0)
            
            # Polynomial: τ = c_00 + c_10*V + c_01*ω + c_20*V² + c_11*V*ω + c_02*ω² 
            #            + c_30*V³ + c_21*V²*ω + c_12*V*ω² + c_03*ω³
            c_00, c_10, c_01, c_20, c_11, c_02, c_30, c_21, c_12, c_03 = torch.unbind(poly_coeffs, dim=-1)
            
            tau = c_00
            tau = tau + c_10 * V
            tau = tau + c_01 * w
            tau = tau + c_20 * V * V
            tau = tau + c_11 * V * w
            tau = tau + c_02 * w * w
            tau = tau + c_30 * V * V * V
            tau = tau + c_21 * V * V * w
            tau = tau + c_12 * V * w * w
            tau = tau + c_03 * w * w * w
            
            motor_torque = torch.clamp(tau, min=0.0)
            
            # Wheel torque through gearbox
            wheel_torque = eta * gear_ratio * motor_torque
            
            # Default motor inertia for polynomial model
            J = torch.full_like(mass, 1e-3)
        else:
            # DC motor model: 21 parameters
            mass = params[..., 0]
            drag_area = params[..., 1]
            rolling_coeff = params[..., 2]
            V_max = params[..., 3]
            R = params[..., 4]
            K = params[..., 5]
            b = params[..., 6]
            J = params[..., 7]
            gamma_throttle = params[..., 8]
            throttle_tau = params[..., 9]
            T_max_param = params[..., 10]
            P_max_param = params[..., 11]
            gear_ratio = params[..., 12]
            eta = params[..., 13]
            brake_T_max = params[..., 14]
            brake_tau = params[..., 15]
            brake_p = params[..., 16]
            mu = params[..., 17]
            r_w = params[..., 18]
            wheel_inertia = params[..., 19]
            
            # Motor speed from wheel speed
            omega_m = gear_ratio * speed / torch.clamp(r_w, min=1e-3)
            
            # Current control (quasi-steady, no regen)
            I_max = torch.where(
                T_max_param > 0.0,
                T_max_param / torch.clamp(K, min=1e-6),
                V_max / torch.clamp(R, min=1e-4),
            )
            back_emf = K * omega_m
            throttle_frac = torch.clamp(throttle, min=0.0) / 100.0
            target_current = torch.pow(throttle_frac, gamma_throttle) * I_max
            v_required = target_current * R + back_emf
            v_applied = torch.minimum(v_required, V_max)
            motor_current = torch.clamp((v_applied - back_emf) / torch.clamp(R, min=1e-4), min=0.0)
            p_limit = torch.clamp(P_max_param, min=0.0)
            motor_current = torch.where(
                p_limit > 0.0,
                torch.minimum(motor_current, p_limit / torch.clamp(v_applied, min=1e-6)),
                motor_current,
            )
            
            # Motor torque with viscous friction loss
            motor_torque = K * motor_current - b * omega_m
            motor_torque = torch.clamp(motor_torque, min=0.0)
            
            # Wheel torque through gearbox
            wheel_torque = eta * gear_ratio * motor_torque
        
        # Drive force
        F_drive = wheel_torque / torch.clamp(r_w, min=1e-3)
        
        # Brake force with nonlinear power-law response
        brake_cmd = torch.clamp(brake, min=0.0) / 100.0
        brake_p_eff = torch.clamp(brake_p, min=0.1)
        brake_frac = torch.pow(brake_cmd, brake_p_eff)
        F_brake = brake_T_max * brake_frac / torch.clamp(r_w, min=1e-3)
        
        # Aerodynamic drag
        F_drag = 0.5 * AIR_DENSITY * drag_area * speed * torch.abs(speed)

        if self.config.optimize_without_grade:
            grade = torch.zeros_like(grade)
        
        # Rolling resistance
        cos_grade = torch.cos(grade)
        F_roll = rolling_coeff * mass * GRAVITY * cos_grade
        
        # Grade resistance
        sin_grade = torch.sin(grade)
        F_grade = mass * GRAVITY * sin_grade
        
        # Net force and acceleration
        effective_mass = mass + (4 * wheel_inertia + J * gear_ratio**2) / torch.clamp(r_w**2, min=1e-6)
        F_net = F_drive - F_brake - F_drag - F_roll - F_grade
        a = F_net / torch.clamp(effective_mass, min=1.0)
        
        return a
    
    def _simulate_segments_batch_torch(
        self,
        params: torch.Tensor,
        segments: List[TripSegment],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate multiple segments in parallel on GPU.
        
        Args:
            params: Parameter tensor [n_params]
            segments: List of trip segments
            
        Returns:
            v_sim: Simulated velocities tensor [n_segments, max_length]
            a_sim: Simulated accelerations tensor [n_segments, max_length]
            valid_mask: Valid mask tensor [n_segments, max_length] (1 where valid, 0 where padded)
            sample_weights: Per-sample weights tensor [n_segments, max_length]
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        n_segments = len(segments)
        max_length = max(s.length for s in segments)
        
        # Prepare input tensors
        throttles_list = []
        brakes_list = []
        grades_list = []
        weight_list = []
        initial_speeds_list = []
        valid_mask_list = []
        
        for seg in segments:
            length = seg.length
            throttles_list.append(torch.tensor(seg.throttle, device=self._device, dtype=torch.float32))
            brakes_list.append(torch.tensor(seg.brake, device=self._device, dtype=torch.float32))
            if self.config.optimize_without_grade:
                grades_list.append(torch.zeros(length, device=self._device, dtype=torch.float32))
            else:
                grades_list.append(torch.tensor(seg.grade, device=self._device, dtype=torch.float32))
            if seg.sample_weights is not None:
                weight_list.append(torch.tensor(seg.sample_weights, device=self._device, dtype=torch.float32))
            else:
                weight_list.append(torch.ones(length, device=self._device, dtype=torch.float32))
            initial_speeds_list.append(seg.initial_speed)
            
            # Create valid mask
            mask = torch.zeros(max_length, device=self._device, dtype=torch.float32)
            mask[:length] = 1.0
            valid_mask_list.append(mask)
        
        # Pad and stack
        throttles = torch.stack([
            torch.nn.functional.pad(t, (0, max_length - t.shape[0]), value=0.0)
            for t in throttles_list
        ])
        brakes = torch.stack([
            torch.nn.functional.pad(b, (0, max_length - b.shape[0]), value=0.0)
            for b in brakes_list
        ])
        grades = torch.stack([
            torch.nn.functional.pad(g, (0, max_length - g.shape[0]), value=0.0)
            for g in grades_list
        ])
        sample_weights = torch.stack([
            torch.nn.functional.pad(w, (0, max_length - w.shape[0]), value=1.0)
            for w in weight_list
        ])
        valid_mask = torch.stack(valid_mask_list)
        
        # Expand params to batch dimension
        params_expanded = params.unsqueeze(0).expand(n_segments, -1)
        
        # Initialize velocity and acceleration tensors
        v_sim = torch.zeros(n_segments, max_length, device=self._device, dtype=torch.float32)
        v_sim[:, 0] = torch.tensor(initial_speeds_list, device=self._device, dtype=torch.float32)
        a_sim = torch.zeros(n_segments, max_length, device=self._device, dtype=torch.float32)

        throttle_tau = float(self.config.motor_throttle_tau_init)
        if "motor_throttle_tau" in self.PARAM_NAMES:
            throttle_tau = float(params[self.PARAM_NAMES.index("motor_throttle_tau")].item())
        throttle_tau = max(throttle_tau, 1e-4)
        throttle_state = throttles[:, 0].clone()
        
        # Forward simulation for all segments in parallel
        for t in range(max_length - 1):
            # Get current speeds
            v_curr = v_sim[:, t]

            # First-order lag on throttle command
            dt_tensor = torch.tensor(
                [segments[i].dt if t < segments[i].length - 1 else 0.0 for i in range(n_segments)],
                device=self._device,
                dtype=torch.float32,
            )
            alpha_th = dt_tensor / throttle_tau
            throttle_state = throttle_state + alpha_th * (throttles[:, t] - throttle_state)
            
            # Compute accelerations for all segments at once
            a = self._compute_acceleration_torch(
                params_expanded,
                v_curr,
                throttle_state,
                brakes[:, t],
                grades[:, t],
            )
            a_sim[:, t] = a
            
            # Euler integration with speed clamp
            v_sim[:, t + 1] = torch.clamp(v_curr + a * dt_tensor, min=0.0)
        
        return v_sim, a_sim, valid_mask, sample_weights
    
    def _trajectory_loss(
        self,
        params: np.ndarray,
        segments: List[TripSegment],
    ) -> float:
        """Compute velocity and acceleration MSE over all trajectory segments.
        
        Args:
            params: Parameter vector
            segments: List of trip segments to simulate
            
        Returns:
            Weighted mean squared error (with barrier penalty if enabled)
        """
        cfg = self.config
        
        # Use GPU if available
        if (
            self._device is not None
            and len(segments) > 0
            and TORCH_AVAILABLE
            and not self.config.use_extended_plant
        ):
            
            # Convert params to GPU tensor
            params_torch = torch.tensor(params, device=self._device, dtype=torch.float32)
            
            # Simulate all segments in parallel on GPU
            v_sim_batch, a_sim_batch, valid_mask, sample_weights = self._simulate_segments_batch_torch(params_torch, segments)
            
            # Compute GT tensors
            speeds_gt_list = []
            accels_gt_list = []
            brakes_gt_list = []
            throttles_gt_list = []
            for seg in segments:
                # Speed GT
                speed_padded = torch.zeros(v_sim_batch.shape[1], device=self._device, dtype=torch.float32)
                speed_padded[:seg.length] = torch.tensor(seg.speed, device=self._device, dtype=torch.float32)
                speeds_gt_list.append(speed_padded)
                
                # Acceleration GT
                accel_padded = torch.zeros(a_sim_batch.shape[1], device=self._device, dtype=torch.float32)
                accel_padded[:seg.length] = torch.tensor(seg.acceleration, device=self._device, dtype=torch.float32)
                accels_gt_list.append(accel_padded)

                brake_padded = torch.zeros(v_sim_batch.shape[1], device=self._device, dtype=torch.float32)
                brake_padded[:seg.length] = torch.tensor(seg.brake, device=self._device, dtype=torch.float32)
                brakes_gt_list.append(brake_padded)

                throttle_padded = torch.zeros(v_sim_batch.shape[1], device=self._device, dtype=torch.float32)
                throttle_padded[:seg.length] = torch.tensor(seg.throttle, device=self._device, dtype=torch.float32)
                throttles_gt_list.append(throttle_padded)
                
            speeds_gt = torch.stack(speeds_gt_list)
            accels_gt = torch.stack(accels_gt_list)
            brakes_gt = torch.stack(brakes_gt_list)
            throttles_gt = torch.stack(throttles_gt_list)
            
            # Compute squared errors (only where valid)
            speed_errors = (v_sim_batch - speeds_gt) ** 2
            accel_errors = (a_sim_batch - accels_gt) ** 2
            
            # Weighting
            weighted_errors = (
                cfg.speed_loss_weight * speed_errors +
                cfg.accel_loss_weight * accel_errors
            ) * valid_mask * sample_weights

            if cfg.mask_negative_gt_speed:
                gt_mask = (speeds_gt >= 0.0).float()
                weighted_errors = weighted_errors * gt_mask
                valid_mask = valid_mask * gt_mask

            if cfg.brake_loss_boost > 0.0:
                brake_active = (brakes_gt > cfg.brake_deadband_pct).float()
                weight = 1.0 + cfg.brake_loss_boost * brake_active
                weighted_errors = weighted_errors * weight

            # Sum over all segments and timesteps
            total_loss = torch.sum(weighted_errors).item()
            total_samples = torch.sum(valid_mask * sample_weights).item()

            if cfg.full_stop_loss_cap_fraction > 0.0 and total_loss > 0.0:
                eps = cfg.zero_speed_eps
                valid_bool = valid_mask > 0.0
                stop_mask = (torch.abs(speeds_gt) <= eps) | (~valid_bool)
                full_stop_segments = torch.all(stop_mask, dim=1)
                if torch.any(full_stop_segments):
                    full_stop_mask = full_stop_segments[:, None]
                    full_stop_loss = torch.sum(weighted_errors * full_stop_mask).item()
                    cap = cfg.full_stop_loss_cap_fraction * total_loss
                    if full_stop_loss > cap:
                        total_loss = total_loss - (full_stop_loss - cap)
            
            mse = total_loss / total_samples if total_samples > 0 else 0.0
        else:
            # CPU fallback
            total_loss = 0.0
            total_samples = 0.0
            full_stop_loss = 0.0
            
            debug_progress = False
            debug_label = ""
            if self._debug_batch_active and self._debug_batch_remaining_calls > 0:
                debug_progress = True
                debug_label = self._debug_batch_label
                self._debug_batch_remaining_calls -= 1

            for segment in segments:
                v_sim, a_sim = self._simulate_segment(
                    params,
                    segment,
                    debug_progress=debug_progress,
                    debug_label=debug_label,
                )
                
                se_speed = (v_sim - segment.speed) ** 2
                se_accel = (a_sim - segment.acceleration) ** 2
                
                weighted_se = cfg.speed_loss_weight * se_speed + cfg.accel_loss_weight * se_accel
                sample_weights = segment.sample_weights if segment.sample_weights is not None else np.ones(segment.length, dtype=np.float64)
                if cfg.mask_negative_gt_speed:
                    gt_mask = segment.speed >= 0.0
                    weighted_se = weighted_se * gt_mask
                    total_samples += float(np.sum(sample_weights * gt_mask))
                else:
                    total_samples += float(np.sum(sample_weights))
                weighted_se = weighted_se * sample_weights
                if cfg.brake_loss_boost > 0.0:
                    brake_active = segment.brake > cfg.brake_deadband_pct
                    weight = 1.0 + cfg.brake_loss_boost * brake_active
                    weighted_se = weighted_se * weight
                total_loss += np.sum(weighted_se)
                if cfg.full_stop_loss_cap_fraction > 0.0:
                    if np.all(np.abs(segment.speed) <= cfg.zero_speed_eps):
                        full_stop_loss += np.sum(weighted_se)
            if cfg.full_stop_loss_cap_fraction > 0.0 and total_loss > 0.0:
                cap = cfg.full_stop_loss_cap_fraction * total_loss
                if full_stop_loss > cap:
                    total_loss = total_loss - (full_stop_loss - cap)

            mse = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Add barrier penalty if enabled
        if self.config.use_barrier and self._bounds is not None:
            barrier_penalty = self._barrier_penalty(params, self._bounds)
            mse += barrier_penalty
        
        # Sanitize loss for the optimizer
        if not np.isfinite(mse):
            # If we have NaNs or Inf, return a very large value to force optimizer away
            # Using 1e12 which is larger than typical MSE but small enough for L-BFGS-B
            mse = 1e12
            
        self._current_loss = mse
        return mse
    
    def _trajectory_loss_with_numerical_gradient(
        self,
        params: np.ndarray,
        segments: List[TripSegment],
        eps: float = 1e-6,
    ) -> Tuple[float, np.ndarray]:
        """Compute loss and numerical gradient.
        
        Uses forward finite differences for faster computation (2x faster than central).
        """
        t0 = time.time() if self._debug_batch_active else 0.0
        loss = self._trajectory_loss(params, segments)
        
        grad = np.zeros_like(params)
        # Use forward differences instead of central (2x faster, slightly less accurate)
        # This reduces from 2*N+1 to N+1 function evaluations
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            
            loss_plus = self._trajectory_loss(params_plus, segments)
            
            grad[i] = (loss_plus - loss) / eps

        if self.config.debug_batch_progress:
            debug_params = [
                "motor_gamma_throttle",
                "motor_min_current_A",
            ]
            name_to_idx = {name: idx for idx, name in enumerate(self.PARAM_NAMES)}
            for name in debug_params:
                idx = name_to_idx.get(name)
                if idx is None:
                    continue
                print(
                    f"  [grad-debug] {name}: value={params[idx]:.6g} grad={grad[idx]:.6g}"
                )

        if self._debug_batch_active:
            print(
                f"  [grad] params={len(params)} eval_time={time.time() - t0:.2f}s"
            )
        
        return loss, grad
    
    def _save_checkpoint(
        self,
        log_path: Path,
        params: np.ndarray,
        loss: float,
        epoch: int,
        batch: int,
    ) -> None:
        """Save current best parameters to log file."""
        checkpoint = {
            "epoch": epoch,
            "batch": batch,
            "loss": float(loss),
            "rmse": float(np.sqrt(loss)),
            "params": {name: float(val) for name, val in zip(self.PARAM_NAMES, params)},
        }
        with open(log_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
    
    def _sample_random_params(
        self,
        bounds: List[Tuple[float, float]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample random parameters uniformly within bounds.
        
        Args:
            bounds: List of (min, max) tuples for each parameter
            rng: Random number generator
            
        Returns:
            Random parameter vector
        """
        params = np.array([
            rng.uniform(low=bound[0], high=bound[1]) for bound in bounds
        ])
        return params
    
    def _run_warmup(
        self,
        bounds: List[Tuple[float, float]],
        val_segments: List[TripSegment],
        num_samples: int,
        seed: int,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Run warmup to find best initial parameters.
        
        Randomly samples parameter sets and evaluates on validation set.
        
        Args:
            bounds: Parameter bounds
            val_segments: Validation segments for evaluation
            num_samples: Number of random samples to try
            seed: Random seed
            verbose: Print progress
            
        Returns:
            Tuple of (best_params, best_loss)
        """
        if verbose:
            print(f"\nWarmup: Evaluating {num_samples} random parameter sets on validation set...")
        
        rng = np.random.default_rng(seed)
        best_loss = float('inf')
        best_params = None
        
        warmup_iter = range(num_samples)
        if verbose and tqdm is not None:
            warmup_iter = tqdm(warmup_iter, desc="  Warmup", position=0)
        
        for i in warmup_iter:
            # Sample random parameters
            params = self._sample_random_params(bounds, rng)
            
            # Evaluate on validation set
            val_loss = self._trajectory_loss(params, val_segments)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params.copy()
            
            if verbose and tqdm is not None:
                warmup_iter.set_postfix({
                    "best_RMSE": f"{np.sqrt(best_loss):.3f}",
                    "current_RMSE": f"{np.sqrt(val_loss):.3f}"
                })
        
        if verbose:
            print(f"Warmup complete: best validation RMSE = {np.sqrt(best_loss):.4f} m/s")
            print("Best warmup parameters:")
            for name, val in zip(self.PARAM_NAMES, best_params):
                print(f"  {name}: {val:.4f}")
        
        return best_params, best_loss
    
    def _plot_speed_histograms(
        self,
        train_segments: List[TripSegment],
        val_segments: List[TripSegment],
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot histograms of speed values for train and validation sets."""
        if not MATPLOTLIB_AVAILABLE:
            LOGGER.warning("matplotlib not available, skipping histogram plot")
            return
        
        # Use non-GUI backend when called from background thread
        import matplotlib
        current_backend = matplotlib.get_backend()
        if current_backend.lower() in ['tkagg', 'qt5agg', 'qt4agg', 'gtk3agg', 'gtk4agg']:
            matplotlib.use('Agg')
        
        train_speeds = np.concatenate([s.speed for s in train_segments])
        val_speeds = np.concatenate([s.speed for s in val_segments])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training set histogram
        ax1.hist(train_speeds, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Speed (m/s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Training Set Speed Distribution\n({len(train_segments)} segments, {len(train_speeds):,} samples)')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(train_speeds.mean(), color='red', linestyle='--', label=f'Mean: {train_speeds.mean():.2f} m/s')
        ax1.legend()
        
        # Validation set histogram
        ax2.hist(val_speeds, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Speed (m/s)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Validation Set Speed Distribution\n({len(val_segments)} segments, {len(val_speeds):,} samples)')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(val_speeds.mean(), color='red', linestyle='--', label=f'Mean: {val_speeds.mean():.2f} m/s')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            LOGGER.info(f"Saved speed histograms to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
        
        # Restore original backend if changed
        if current_backend.lower() in ['tkagg', 'qt5agg', 'qt4agg', 'gtk3agg', 'gtk4agg']:
            matplotlib.use(current_backend)
        
        # Restore original backend if changed
        if current_backend.lower() in ['tkagg', 'qt5agg', 'qt4agg', 'gtk3agg', 'gtk4agg']:
            matplotlib.use(current_backend)
    
    def _plot_validation_trips(
        self,
        params: np.ndarray,
        val_segments: List[TripSegment],
        save_path: Path,
        max_trips: int = 5,
    ) -> None:
        """Plot GT vs simulated speed for validation trips."""
        if not MATPLOTLIB_AVAILABLE:
            LOGGER.warning("matplotlib not available, skipping validation plot")
            return
        
        # Use non-GUI backend when called from background thread
        import matplotlib
        current_backend = matplotlib.get_backend()
        if current_backend.lower() in ['tkagg', 'qt5agg', 'qt4agg', 'gtk3agg', 'gtk4agg']:
            matplotlib.use('Agg')
        
        n_trips = min(max_trips, len(val_segments))
        selected_segments = val_segments[:n_trips]
        
        fig, axes = plt.subplots(n_trips, 1, figsize=(12, 3 * n_trips), sharex=True)
        if n_trips == 1:
            axes = [axes]
        
        for idx, segment in enumerate(selected_segments):
            ax = axes[idx]
            v_sim, _ = self._simulate_segment(params, segment)
            time = np.arange(segment.length) * segment.dt
            
            ax.plot(time, segment.speed, 'b-', label='GT Speed', alpha=0.8, linewidth=2)
            ax.plot(time, v_sim, 'r--', label='Simulated', alpha=0.8, linewidth=2)
            ax.set_ylabel('Velocity (m/s)')
            ax.set_title(f'Validation Trip: {segment.trip_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add RMSE text
            rmse = np.sqrt(np.mean((v_sim - segment.speed) ** 2))
            ax.text(0.02, 0.98, f'RMSE: {rmse:.3f} m/s', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        LOGGER.info(f"Saved validation trip comparison to {save_path}")
        plt.close(fig)
        
        # Restore original backend if changed
        if current_backend.lower() in ['tkagg', 'qt5agg', 'qt4agg', 'gtk3agg', 'gtk4agg']:
            matplotlib.use(current_backend)
    
    def fit(
        self,
        data_path: Path,
        verbose: bool = True,
        log_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[np.ndarray, float], None]] = None,
    ) -> FittedVehicleParams:
        """Fit vehicle parameters by minimizing trajectory velocity error.

        Args:
            data_path: Path to trip data (.pt file)
            verbose: Print progress
            log_path: Path to save best-so-far parameters (updated on each improvement)
            progress_callback: Optional callback called with (best_params, best_loss) when new best is found

        Returns:
            Fitted vehicle parameters
        """
        cfg = self.config
        
        # Default log path next to data
        if log_path is None:
            log_path = data_path.parent / "fitting_checkpoint.json"
        
        # Load and process data
        if verbose:
            print(f"Loading data from {data_path}...")
        
        trips = self.load_trip_data(data_path)
        self._trips = trips
        
        # Estimate dt from data
        dt = self._estimate_dt(trips)
        self._dt = dt
        if verbose:
            print(f"Estimated dt: {dt:.4f} s")
        
        # Create segments
        all_segments = self._create_segments(trips, dt)
        
        if len(all_segments) == 0:
            raise ValueError("No valid segments found in data")
        
        # Downsample if requested
        if cfg.downsampling_factor > 1:
            if verbose:
                print(f"Downsampling data by factor {cfg.downsampling_factor}...")
            all_segments = self._downsample_segments(all_segments, cfg.downsampling_factor)
            # Update dt to reflect downsampling
            dt = dt * cfg.downsampling_factor
            if verbose:
                print(f"Updated dt after downsampling: {dt:.4f} s")
        
        # Filter zero-speed segments to limit their fraction
        if (
            cfg.filter_zero_speed_segments
            and not cfg.disable_segment_filtering
            and cfg.max_zero_speed_fraction < 1.0
        ):
            if verbose:
                print(f"Filtering zero-speed segments (max fraction: {cfg.max_zero_speed_fraction:.1%}, eps: {cfg.zero_speed_eps:.3f} m/s)...")
            all_segments = self._filter_zero_speed_segments(
                all_segments,
                max_fraction=cfg.max_zero_speed_fraction,
                eps=cfg.zero_speed_eps,
            )

        if cfg.use_uniform_speed_accel_bin_loss:
            if verbose:
                print(
                    "Applying uniform (speed, accel) bucket weighting "
                    f"[{cfg.speed_accel_speed_bins}x{cfg.speed_accel_accel_bins}]..."
                )
            distribution = self.apply_uniform_speed_accel_bucket_weights(all_segments)
            self._last_speed_accel_distribution = distribution
            if verbose:
                nonzero = int(distribution["nonzero_bins"][0])
                total = int(distribution["total_samples"][0])
                print(f"Weighted distribution bins (non-empty): {nonzero}, samples: {total:,}")
        else:
            for seg in all_segments:
                seg.sample_weights = None
            self._last_speed_accel_distribution = self.compute_speed_accel_distribution(all_segments)
        
        # Split into train/validation
        if self._split_seed is not None:
            rng_seed = int(self._split_seed)
        elif cfg.validation_split_seed is not None:
            rng_seed = int(cfg.validation_split_seed)
        else:
            rng_seed = 42
        rng = np.random.default_rng(rng_seed)
        n_total = len(all_segments)
        if n_total <= 1:
            n_val = 0
        else:
            n_val = max(1, int(n_total * cfg.validation_fraction))
            n_val = min(n_val, n_total - 1)
        n_train = n_total - n_val
        
        perm = rng.permutation(n_total)
        train_segments = [all_segments[i] for i in perm[:n_train]]
        val_segments = [all_segments[i] for i in perm[n_train:]]
        self._segments = all_segments
        self.train_segments = train_segments
        self.val_segments = val_segments  # Store for GUI access

        fixed_val_segments = None
        if cfg.use_random_segment_batches and cfg.use_fixed_length_validation:
            target_len = max(cfg.random_segment_length, cfg.min_segment_length)
            rng_val = np.random.default_rng(rng_seed)
            fixed_val_segments = self._sample_fixed_length_batch(
                val_segments,
                max(len(val_segments), 1),
                target_len,
                rng_val,
            )
        
        train_samples = sum(s.length for s in train_segments)
        val_samples = sum(s.length for s in (fixed_val_segments or val_segments))
        
        if verbose:
            print(f"Created {n_total} segments: {n_train} train, {n_val} validation")
            print(f"Train samples: {train_samples:,}, Validation samples: {val_samples:,}")
            speeds = np.concatenate([s.speed for s in all_segments])
            print(f"Speed range: [{speeds.min():.2f}, {speeds.max():.2f}] m/s")
        
        # Plot speed histograms for train/val split
        if verbose:
            hist_path = data_path.parent / "speed_histograms.png"
            self._plot_speed_histograms(train_segments, fixed_val_segments or val_segments, save_path=hist_path)
        
        # Initial parameters and bounds (variable count based on motor model)
        if cfg.motor_model_type == "polynomial":
            # Polynomial model: 25 parameters
            x0 = np.array([
                cfg.mass_init,
                cfg.drag_area_init,
                cfg.rolling_coeff_init,
                cfg.motor_V_max_init,
                cfg.motor_gamma_throttle_init,
                cfg.motor_throttle_tau_init,
                cfg.poly_c_00_init,
                cfg.poly_c_10_init,
                cfg.poly_c_01_init,
                cfg.poly_c_20_init,
                cfg.poly_c_11_init,
                cfg.poly_c_02_init,
                cfg.poly_c_30_init,
                cfg.poly_c_21_init,
                cfg.poly_c_12_init,
                cfg.poly_c_03_init,
                cfg.gear_ratio_init,
                cfg.eta_gb_init,
                cfg.brake_T_max_init,
                cfg.brake_tau_init,
                cfg.brake_p_init,
                cfg.mu_init,
                cfg.wheel_radius_init,
                cfg.wheel_inertia_init,
                cfg.motor_min_current_A_init,
            ])
            
            bounds = [
                cfg.mass_bounds,
                cfg.drag_area_bounds,
                cfg.rolling_coeff_bounds,
                cfg.motor_V_max_bounds,
                cfg.motor_gamma_throttle_bounds,
                cfg.motor_throttle_tau_bounds,
                cfg.poly_c_00_bounds,
                cfg.poly_c_10_bounds,
                cfg.poly_c_01_bounds,
                cfg.poly_c_20_bounds,
                cfg.poly_c_11_bounds,
                cfg.poly_c_02_bounds,
                cfg.poly_c_30_bounds,
                cfg.poly_c_21_bounds,
                cfg.poly_c_12_bounds,
                cfg.poly_c_03_bounds,
                cfg.gear_ratio_bounds,
                cfg.eta_gb_bounds,
                cfg.brake_T_max_bounds,
                cfg.brake_tau_bounds,
                cfg.brake_p_bounds,
                cfg.mu_bounds,
                cfg.wheel_radius_bounds,
                cfg.wheel_inertia_bounds,
                cfg.motor_min_current_A_bounds,
            ]
        else:
            # DC motor model: 21 parameters
            x0 = np.array([
                cfg.mass_init,
                cfg.drag_area_init,
                cfg.rolling_coeff_init,
                cfg.motor_V_max_init,
                cfg.motor_R_init,
                cfg.motor_K_init,
                cfg.motor_b_init,
                cfg.motor_J_init,
                cfg.motor_gamma_throttle_init,
                cfg.motor_throttle_tau_init,
                cfg.motor_T_max_init if cfg.motor_T_max_init is not None else (cfg.motor_K_init * (cfg.motor_V_max_init / max(cfg.motor_R_init, 1e-4))),
                cfg.motor_P_max_init if cfg.motor_P_max_init is not None else cfg.motor_P_max_bounds[1],
                cfg.gear_ratio_init,
                cfg.eta_gb_init,
                cfg.brake_T_max_init,
                cfg.brake_tau_init,
                cfg.brake_p_init,
                cfg.mu_init,
                cfg.wheel_radius_init,
                cfg.wheel_inertia_init,
                cfg.motor_min_current_A_init,
            ])
            
            bounds = [
                cfg.mass_bounds,
                cfg.drag_area_bounds,
                cfg.rolling_coeff_bounds,
                cfg.motor_V_max_bounds,
                cfg.motor_R_bounds,
                cfg.motor_K_bounds,
                cfg.motor_b_bounds,
                cfg.motor_J_bounds,
                cfg.motor_gamma_throttle_bounds,
                cfg.motor_throttle_tau_bounds,
                cfg.motor_T_max_bounds,
                cfg.motor_P_max_bounds,
                cfg.gear_ratio_bounds,
                cfg.eta_gb_bounds,
                cfg.brake_T_max_bounds,
                cfg.brake_tau_bounds,
                cfg.brake_p_bounds,
                cfg.mu_bounds,
                cfg.wheel_radius_bounds,
                cfg.wheel_inertia_bounds,
                cfg.motor_min_current_A_bounds,
            ]
        
        # Store bounds for barrier function computation
        self._bounds = bounds
        
        # Warmup: find better initial guess by random sampling
        if cfg.use_warmup:
            warmup_params, warmup_loss = self._run_warmup(
                bounds=bounds,
                val_segments=val_segments,
                num_samples=cfg.warmup_samples,
                seed=cfg.warmup_seed,
                verbose=verbose,
            )
            # Use warmup result as initial guess
            x0 = warmup_params
            if verbose:
                print(f"Using warmup parameters as initial guess (val_RMSE={np.sqrt(warmup_loss):.4f} m/s)")
        
        # Batch setup (training segments only)
        if cfg.segments_per_batch <= 0:
            batch_size = n_train
        else:
            batch_size = min(cfg.segments_per_batch, n_train)
        num_batches = (n_train + batch_size - 1) // batch_size
        if cfg.use_random_segment_batches and cfg.random_batches_per_epoch > 0:
            num_batches = int(cfg.random_batches_per_epoch)
        
        if verbose:
            print(f"\nOptimization: {cfg.num_epochs} epochs, {num_batches} batches/epoch")
            print(f"Parameters: {self.PARAM_NAMES}")
            print(f"Checkpoint log: {log_path}")
            print("-" * 70)
        
        best_val_loss = float('inf')
        global_best_val_loss = float('inf')
        best_params = x0.copy()

        phases = ["joint"] if cfg.optimization_mode == "joint" else list(cfg.phase_order)
        if cfg.optimization_mode == "sequential" and not phases:
            phases = ["throttle", "brake"]

        optimizer_method = str(cfg.optimizer_method).strip() or "L-BFGS-B"
        jac_methods = {"L-BFGS-B", "TNC", "SLSQP"}
        use_jac = optimizer_method in jac_methods

        def _build_scaled_objective(phase_bounds: List[Tuple[float, float]], x0_phase: np.ndarray):
            bounds_opt = [(0.0, 0.0) if b[0] == b[1] else (0.0, 1.0) for b in phase_bounds]
            x0_opt = self._params_to_unit(x0_phase, phase_bounds)

            def _objective(x_unit: np.ndarray, segs: List[TripSegment]):
                t0 = time.time() if verbose and cfg.debug_batch_progress else 0.0
                params = self._params_from_unit(x_unit, phase_bounds)
                if use_jac:
                    loss, grad = self._trajectory_loss_with_numerical_gradient(params, segs)
                else:
                    loss = self._trajectory_loss(params, segs)
                    grad = None
                if verbose and cfg.debug_batch_progress:
                    print(
                        f"  [objective] loss={loss:.6f} eval_time={time.time() - t0:.2f}s"
                    )
                lows = np.array([b[0] for b in phase_bounds], dtype=np.float64)
                highs = np.array([b[1] for b in phase_bounds], dtype=np.float64)
                scale = highs - lows
                if use_jac and grad is not None:
                    grad_unit = grad * scale
                    return loss, grad_unit
                return loss

            def _to_params(x_opt: np.ndarray) -> np.ndarray:
                return self._params_from_unit(x_opt, phase_bounds)

            return bounds_opt, x0_opt, _objective, _to_params

        def _build_objective(phase_bounds: List[Tuple[float, float]], x0_phase: np.ndarray):
            def _objective(x: np.ndarray, segs: List[TripSegment]):
                t0 = time.time() if verbose and cfg.debug_batch_progress else 0.0
                if use_jac:
                    loss, grad = self._trajectory_loss_with_numerical_gradient(x, segs)
                else:
                    loss = self._trajectory_loss(x, segs)
                    grad = None
                if verbose and cfg.debug_batch_progress:
                    print(
                        f"  [objective] loss={loss:.6f} eval_time={time.time() - t0:.2f}s"
                    )
                if use_jac and grad is not None:
                    return loss, grad
                return loss

            return phase_bounds, x0_phase.copy(), _objective, lambda x: x

        # Optional overfit warmup on the longest training trip
        if cfg.use_overfit_longest_trip and cfg.overfit_longest_trip_epochs > 0:
            overfit_segment = self.get_longest_training_display_segment()
            if overfit_segment is None:
                LOGGER.warning("Overfit warmup requested but no training trip segment found")
            else:
                self.current_phase = "overfit_longest"
                if verbose:
                    print(
                        f"Overfit warmup on longest training trip: {overfit_segment.trip_id} "
                        f"(len={overfit_segment.length}) for {cfg.overfit_longest_trip_epochs} epoch(s)"
                    )

                if cfg.use_param_scaling:
                    bounds_opt, x0_opt, _objective, _to_params = _build_scaled_objective(bounds, best_params.copy())
                else:
                    bounds_opt, x0_opt, _objective, _to_params = _build_objective(bounds, best_params.copy())

                # When using random fixed-length batches, also overfit using random
                # fixed-length segments from the longest trip, with the same
                # target length setting.
                use_random_overfit_batches = cfg.use_random_segment_batches and cfg.random_segment_length > 0
                if use_random_overfit_batches:
                    batch_size_overfit = cfg.segments_per_batch if cfg.segments_per_batch > 0 else 1
                    target_len_overfit = max(cfg.random_segment_length, cfg.min_segment_length)

                for _epoch in range(1, cfg.overfit_longest_trip_epochs + 1):
                    if use_random_overfit_batches:
                        batch_segments = self._sample_fixed_length_batch(
                            [overfit_segment],
                            batch_size_overfit,
                            target_len_overfit,
                            rng,
                        )
                        segs_for_epoch = batch_segments
                    else:
                        segs_for_epoch = [overfit_segment]

                    if verbose:
                        seg_len = segs_for_epoch[0].length if segs_for_epoch else 0
                        if use_random_overfit_batches:
                            # Calculate how many segments of this length can be extracted from the longest trip
                            max_segments_from_trip = max(1, overfit_segment.length - target_len_overfit + 1)
                            print(
                                f"  [overfit] epoch {_epoch}/{cfg.overfit_longest_trip_epochs} "
                                f"batch_size={len(segs_for_epoch)} len={seg_len} "
                                f"(sampling from {max_segments_from_trip} possible segments)"
                            )
                        else:
                            print(
                                f"  [overfit] epoch {_epoch}/{cfg.overfit_longest_trip_epochs} "
                                f"using full segment (len={overfit_segment.length})"
                            )

                    result = minimize(
                        _objective,
                        x0_opt,
                        args=(segs_for_epoch,),
                        method=optimizer_method,
                        jac=use_jac,
                        bounds=bounds_opt,
                        options={"maxiter": cfg.max_iter, "ftol": cfg.tolerance, "disp": False},
                    )

                    params_opt = _to_params(result.x)
                    val_loss = self._trajectory_loss(params_opt, fixed_val_segments or val_segments)

                    if val_loss < global_best_val_loss:
                        global_best_val_loss = val_loss
                        best_val_loss = val_loss
                        best_params = params_opt.copy()

                        # Treat overfit warmup like a special "epoch 0" for checkpointing
                        self._save_checkpoint(log_path, best_params, best_val_loss, epoch=0, batch=_epoch - 1)
                        if verbose:
                            print(
                                f"  [overfit] new best val_RMSE={np.sqrt(best_val_loss):.4f} m/s "
                                f"(epoch {_epoch})"
                            )

                        # Notify GUI so it can refresh parameter table and simulation plots
                        if progress_callback is not None:
                            progress_callback(best_params, best_val_loss)

                    x0_opt = result.x

        for phase_index, phase in enumerate(phases):
            if self._abort_event.is_set():
                raise AbortFitting("Fitting aborted by user")
            self.current_phase = phase
            self._phase_advance_event.clear()
            phase_best_val_loss = float('inf')
            if cfg.optimization_mode == "sequential":
                best_val_loss = phase_best_val_loss

            phase_train_segments = self._filter_segments_for_phase(train_segments, phase)
            phase_val_segments = self._filter_segments_for_phase(fixed_val_segments or val_segments, phase)

            if not phase_train_segments:
                LOGGER.warning("No %s-active segments for training; using full training set", phase)
                phase_train_segments = train_segments
            if not phase_val_segments:
                LOGGER.warning("No %s-active segments for validation; using full validation set", phase)
                phase_val_segments = val_segments

            if phase == "brake" and phase_train_segments:
                brake_samples = np.concatenate([s.brake for s in phase_train_segments])
                if brake_samples.size:
                    max_brake = float(np.max(brake_samples))
                    p95_brake = float(np.percentile(brake_samples, 95.0))
                    frac_high = float(np.mean(brake_samples > self.config.saturation_threshold))
                    if max_brake < self.config.saturation_threshold:
                        LOGGER.warning(
                            "Brake phase has low excitation (max=%.2f, p95=%.2f). T_max may not move.",
                            max_brake,
                            p95_brake,
                        )
                    LOGGER.info(
                        "Brake phase stats: max=%.2f, p95=%.2f, frac>sat=%.3f",
                        max_brake,
                        p95_brake,
                        frac_high,
                    )

            phase_n_train = len(phase_train_segments)
            if cfg.segments_per_batch <= 0:
                phase_batch_size = phase_n_train
            else:
                phase_batch_size = min(cfg.segments_per_batch, phase_n_train)
            phase_num_batches = (phase_n_train + phase_batch_size - 1) // phase_batch_size
            if cfg.use_random_segment_batches and cfg.random_batches_per_epoch > 0:
                phase_num_batches = int(cfg.random_batches_per_epoch)

            phase_bounds = bounds
            if cfg.optimization_mode == "sequential" and phase in ("throttle", "brake"):
                phase_bounds = self._freeze_non_phase_params(best_params, bounds, phase)

            x0_phase = best_params.copy()
            if cfg.use_param_scaling:
                bounds_opt, x0_opt, _objective, _to_params = _build_scaled_objective(phase_bounds, x0_phase)
            else:
                bounds_opt, x0_opt, _objective, _to_params = _build_objective(phase_bounds, x0_phase)

            epoch_iter = range(1, cfg.num_epochs + 1)
            if verbose and tqdm is not None:
                epoch_iter = tqdm(epoch_iter, desc="Epochs", position=0)

            for epoch in epoch_iter:
                if self._abort_event.is_set():
                    raise AbortFitting("Fitting aborted by user")
                # Shuffle training segments for this epoch
                if cfg.shuffle_segments:
                    train_indices = rng.permutation(phase_n_train)
                else:
                    train_indices = np.arange(phase_n_train)

                epoch_train_losses = []

                # Batch iteration
                batch_iter = range(phase_num_batches)
                if verbose and tqdm is not None:
                    batch_iter = tqdm(batch_iter, desc=f"  Epoch {epoch}", leave=False, position=1)

                for b in batch_iter:
                    if self._abort_event.is_set():
                        raise AbortFitting("Fitting aborted by user")
                    batch_start_t = time.time() if verbose else 0.0
                    i_start = b * phase_batch_size
                    i_end = min((b + 1) * phase_batch_size, phase_n_train)
                    if cfg.use_random_segment_batches:
                        target_len = max(cfg.random_segment_length, cfg.min_segment_length)
                        batch_segments = self._sample_fixed_length_batch(
                            phase_train_segments,
                            phase_batch_size,
                            target_len,
                            rng,
                        )
                    else:
                        batch_indices = train_indices[i_start:i_end]
                        batch_segments = [phase_train_segments[i] for i in batch_indices]

                    if verbose and cfg.debug_batch_progress:
                        full_stop_segments = sum(
                            int(np.all(np.abs(seg.speed) <= cfg.zero_speed_eps))
                            for seg in batch_segments
                        )
                        full_stop_samples = sum(
                            int(np.sum(np.abs(seg.speed) <= cfg.zero_speed_eps))
                            for seg in batch_segments
                        )
                        batch_samples = sum(seg.length for seg in batch_segments)
                        full_stop_frac = (full_stop_samples / batch_samples) if batch_samples > 0 else 0.0
                        print(
                            f"  [batch-debug] full_stop_segments={full_stop_segments}/{len(batch_segments)} "
                            f"full_stop_samples={full_stop_samples}/{batch_samples} "
                            f"full_stop_frac={full_stop_frac:.3f}"
                        )
                        self._debug_batch_active = True
                        self._debug_batch_remaining_calls = 1
                        self._debug_batch_label = f"{phase} epoch {epoch} batch {b + 1}/{phase_num_batches}"
                        self._debug_batch_progress_step = float(cfg.debug_batch_progress_step)
                    else:
                        self._debug_batch_active = False

                    # Optimize on this batch
                    # Using forward differences (2x faster than central) via explicit gradient
                    options_max_iter = cfg.max_iter
                    if cfg.use_random_segment_batches:
                        options_max_iter = max(1, int(cfg.random_batch_max_iter))

                    iter_start_t = time.time() if verbose else 0.0
                    def _iter_callback(_xk):
                        if verbose and cfg.use_random_segment_batches:
                            elapsed = time.time() - iter_start_t
                            print(f"  [batch {b + 1}] iter callback elapsed={elapsed:.2f}s")

                    result = minimize(
                        _objective,
                        x0_opt,
                        args=(batch_segments,),
                        method=optimizer_method,
                        jac=use_jac,
                        bounds=bounds_opt,
                        options={"maxiter": options_max_iter, "ftol": cfg.tolerance, "disp": False},
                        callback=_iter_callback,
                    )

                    if verbose:
                        batch_time = time.time() - batch_start_t
                        batch_samples = sum(s.length for s in batch_segments)
                        print(
                            f"Batch {b + 1}/{phase_num_batches} | "
                            f"segments={len(batch_segments)} samples={batch_samples} "
                            f"time={batch_time:.2f}s"
                        )

                    train_loss = result.fun
                    epoch_train_losses.append(train_loss)

                    params_opt = _to_params(result.x)

                    # Evaluate on validation set
                    val_loss = self._trajectory_loss(params_opt, phase_val_segments)

                    if val_loss < phase_best_val_loss:
                        phase_best_val_loss = val_loss
                        if val_loss < global_best_val_loss:
                            global_best_val_loss = val_loss
                        best_params = params_opt.copy()
                        best_val_loss = phase_best_val_loss if cfg.optimization_mode == "sequential" else global_best_val_loss
                        # Save checkpoint
                        self._save_checkpoint(log_path, best_params, best_val_loss, epoch, b)
                        if verbose:
                            print(f"\n  New best: val_RMSE={np.sqrt(best_val_loss):.4f} m/s (saved)")
                            # Plot validation trips comparison only occasionally to avoid slowdown
                            # Plot every 10 improvements or at end of epoch
                            improvement_count = getattr(self, '_improvement_count', 0) + 1
                            self._improvement_count = improvement_count
                            if improvement_count % 10 == 0 or b == phase_num_batches - 1:
                                val_plot_path = data_path.parent / f"validation_trips_epoch{epoch}_batch{b}.png"
                                self._plot_validation_trips(best_params, phase_val_segments, val_plot_path, max_trips=5)

                    # Call progress callback if provided
                    if progress_callback is not None:
                        progress_callback(best_params, best_val_loss)

                    # Warm start next batch
                    x0_opt = result.x

                    if verbose and tqdm is not None:
                        batch_iter.set_postfix({
                            "train": f"{np.sqrt(train_loss):.3f}",
                            "val": f"{np.sqrt(val_loss):.3f}",
                            "best_val": f"{np.sqrt(best_val_loss):.3f}"
                        })

                    if cfg.optimization_mode == "sequential" and self._phase_advance_event.is_set():
                        break

                # Epoch summary
                epoch_mean_train = np.mean(epoch_train_losses) if epoch_train_losses else float("nan")
                epoch_val_loss = self._trajectory_loss(_to_params(x0_opt), phase_val_segments)
                if verbose:
                    if tqdm is not None:
                        epoch_iter.set_postfix({
                            "train": f"{np.sqrt(epoch_mean_train):.3f}",
                            "val": f"{np.sqrt(epoch_val_loss):.3f}",
                            "best_val": f"{np.sqrt(best_val_loss):.3f}"
                        })
                    else:
                        print(f"Epoch {epoch}: train={np.sqrt(epoch_mean_train):.4f}, val={np.sqrt(epoch_val_loss):.4f}, best_val={np.sqrt(best_val_loss):.4f}")

                if cfg.optimization_mode == "sequential" and self._phase_advance_event.is_set():
                    break

            if cfg.optimization_mode == "sequential" and self._phase_advance_event.is_set():
                self._phase_advance_event.clear()
                continue

            if (
                cfg.optimization_mode == "sequential"
                and cfg.pause_between_phases
                and phase_index < len(phases) - 1
            ):
                LOGGER.info("Phase '%s' complete. Waiting for advance to continue...", phase)
                self._phase_advance_event.wait()
                self._phase_advance_event.clear()
        
        # Final evaluation on ALL segments
        if verbose:
            print("\nFinal evaluation...")
        
        train_loss = self._trajectory_loss(best_params, train_segments)
        val_loss = self._trajectory_loss(best_params, fixed_val_segments or val_segments)
        all_loss = self._trajectory_loss(best_params, all_segments)
        
        # Compute R² for velocity on all segments
        all_v_measured = np.concatenate([s.speed for s in all_segments])
        all_v_simulated = np.concatenate([
            self._simulate_segment(best_params, s)[0] for s in all_segments
        ])
        
        ss_res = np.sum((all_v_measured - all_v_simulated) ** 2)
        ss_tot = np.sum((all_v_measured - np.mean(all_v_measured)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        if verbose:
            print("-" * 70)
            print(f"Train RMSE: {np.sqrt(train_loss):.4f} m/s")
            print(f"Val RMSE:   {np.sqrt(val_loss):.4f} m/s")
            print(f"All RMSE:   {np.sqrt(all_loss):.4f} m/s")
            print(f"Velocity R²: {r_squared:.4f}")
            print("\nFitted parameters:")
            for name, val in zip(self.PARAM_NAMES, best_params):
                print(f"  {name}: {val:.4f}")
        
        total_samples = train_samples + val_samples
        
        # Create result based on motor model type
        if cfg.motor_model_type == "polynomial":
            # For polynomial model, create a dictionary with all parameters
            param_dict = {name: float(val) for name, val in zip(self.PARAM_NAMES, best_params)}
            param_dict.update({
                "motor_model_type": "polynomial",
                "fit_loss": val_loss,
                "num_samples": total_samples,
                "r_squared": r_squared,
            })
            
            # If requested, fit DC motor parameters from polynomial map
            if cfg.fit_dc_from_map:
                dc_params = self._fit_dc_from_polynomial_map(best_params, verbose=verbose)
                param_dict["fitted_dc_params"] = dc_params
            
            # Create a simple result object (we'll save as dict)
            class PolynomialFittedParams:
                def __init__(self, data):
                    self.__dict__.update(data)
                def to_dict(self):
                    return self.__dict__.copy()
                def save(self, path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w") as f:
                        json.dump(self.to_dict(), f, indent=2)
                    LOGGER.info(f"Saved fitted params to {path}")
            
            fitted = PolynomialFittedParams(param_dict)
        else:
            # DC motor model: use existing FittedVehicleParams
            by_name = {name: float(val) for name, val in zip(self.PARAM_NAMES, best_params)}
            fitted = FittedVehicleParams(
                mass=by_name["mass"],
                drag_area=by_name["drag_area"],
                rolling_coeff=by_name["rolling_coeff"],
                motor_V_max=by_name["motor_V_max"],
                motor_R=by_name["motor_R"],
                motor_K=by_name["motor_K"],
                motor_b=by_name["motor_b"],
                motor_J=by_name["motor_J"],
                motor_gamma_throttle=by_name.get("motor_gamma_throttle", cfg.motor_gamma_throttle_init),
                motor_throttle_tau=by_name.get("motor_throttle_tau", cfg.motor_throttle_tau_init),
                motor_min_current_A=by_name.get("motor_min_current_A", cfg.motor_min_current_A_init),
                motor_T_max=(by_name.get("motor_T_max", 0.0) if by_name.get("motor_T_max", 0.0) > 0.0 else None),
                motor_P_max=(by_name.get("motor_P_max", 0.0) if by_name.get("motor_P_max", 0.0) > 0.0 else None),
                gear_ratio=by_name["gear_ratio"],
                eta_gb=by_name["eta_gb"],
                brake_T_max=by_name["brake_T_max"],
                brake_tau=by_name["brake_tau"],
                brake_p=by_name["brake_p"],
                mu=by_name["mu"],
                wheel_radius=by_name["wheel_radius"],
                wheel_inertia=by_name["wheel_inertia"],
                fit_loss=val_loss,
                num_samples=total_samples,
                r_squared=r_squared,
            )
        
        return fitted

    def fit_with_validation(
        self,
        data_path: Path,
        val_fraction: float = 0.1,
        seed: int = 42,
        verbose: bool = True,
    ) -> Tuple[FittedVehicleParams, float]:
        """Fit with a specific train/validation split.

        Returns:
            (fitted_params, validation_loss)
        """
        prev_fraction = self.config.validation_fraction
        prev_seed = self._split_seed
        self.config.validation_fraction = float(val_fraction)
        self._split_seed = int(seed)
        try:
            fitted = self.fit(data_path, verbose=verbose)
        finally:
            self.config.validation_fraction = prev_fraction
            self._split_seed = prev_seed

        val_loss = float(getattr(fitted, "fit_loss", 0.0))
        return fitted, val_loss

    def _expand_params_for_prediction(self, params: np.ndarray) -> np.ndarray:
        """Expand partial parameter vectors to full DC parameter set."""
        cfg = self.config
        params = np.asarray(params, dtype=np.float64).flatten()
        expected_size = 21
        if params.size == expected_size:
            return params.astype(np.float64, copy=False)

        defaults = {
            "mass": cfg.mass_init,
            "drag_area": cfg.drag_area_init,
            "rolling_coeff": cfg.rolling_coeff_init,
            "motor_V_max": cfg.motor_V_max_init,
            "motor_R": cfg.motor_R_init,
            "motor_K": cfg.motor_K_init,
            "motor_b": cfg.motor_b_init,
            "motor_J": cfg.motor_J_init,
            "motor_gamma_throttle": cfg.motor_gamma_throttle_init,
            "motor_throttle_tau": cfg.motor_throttle_tau_init,
            "motor_T_max": cfg.motor_T_max_init if cfg.motor_T_max_init is not None else (cfg.motor_K_init * (cfg.motor_V_max_init / max(cfg.motor_R_init, 1e-4))),
            "motor_P_max": cfg.motor_P_max_init if cfg.motor_P_max_init is not None else cfg.motor_P_max_bounds[1],
            "gear_ratio": cfg.gear_ratio_init,
            "eta_gb": cfg.eta_gb_init,
            "brake_T_max": cfg.brake_T_max_init,
            "brake_tau": cfg.brake_tau_init,
            "brake_p": cfg.brake_p_init,
            "mu": cfg.mu_init,
            "wheel_radius": cfg.wheel_radius_init,
            "wheel_inertia": cfg.wheel_inertia_init,
            "motor_min_current_A": cfg.motor_min_current_A_init,
        }

        if params.size == 8:
            (
                defaults["mass"],
                defaults["drag_area"],
                defaults["rolling_coeff"],
                defaults["motor_V_max"],
                defaults["motor_R"],
                defaults["motor_K"],
                defaults["gear_ratio"],
                defaults["brake_T_max"],
            ) = params
            return np.array([
                defaults["mass"],
                defaults["drag_area"],
                defaults["rolling_coeff"],
                defaults["motor_V_max"],
                defaults["motor_R"],
                defaults["motor_K"],
                defaults["motor_b"],
                defaults["motor_J"],
                defaults["motor_gamma_throttle"],
                defaults["motor_throttle_tau"],
                defaults["motor_T_max"],
                defaults["motor_P_max"],
                defaults["gear_ratio"],
                defaults["eta_gb"],
                defaults["brake_T_max"],
                defaults["brake_tau"],
                defaults["brake_p"],
                defaults["mu"],
                defaults["wheel_radius"],
                defaults["wheel_inertia"],
                defaults["motor_min_current_A"],
            ], dtype=np.float64)

        raise ValueError(
            f"Unsupported parameter length for prediction: {params.size}. "
            f"Expected {expected_size} parameters."
        )

    def _predict_acceleration(
        self,
        params: np.ndarray,
        speed: np.ndarray,
        throttle: np.ndarray,
        brake: np.ndarray,
        grade: np.ndarray,
    ) -> np.ndarray:
        """Predict acceleration for vector inputs (DC model)."""
        params_full = self._expand_params_for_prediction(params)
        speed_arr, throttle_arr, brake_arr, grade_arr = np.broadcast_arrays(
            np.asarray(speed, dtype=np.float64),
            np.asarray(throttle, dtype=np.float64),
            np.asarray(brake, dtype=np.float64),
            np.asarray(grade, dtype=np.float64),
        )

        out = np.zeros_like(speed_arr, dtype=np.float64)
        for idx in np.ndindex(speed_arr.shape):
            out[idx] = self._compute_acceleration(
                params_full,
                float(speed_arr[idx]),
                float(throttle_arr[idx]),
                float(brake_arr[idx]),
                float(grade_arr[idx]),
            )
        return out

    def predict_acceleration(
        self,
        params: FittedVehicleParams,
        speed: np.ndarray,
        throttle: np.ndarray,
        brake: np.ndarray,
        grade: np.ndarray,
    ) -> np.ndarray:
        """Predict acceleration from fitted parameters."""
        param_array = np.array([
            params.mass,
            params.drag_area,
            params.rolling_coeff,
            params.motor_V_max,
            params.motor_R,
            params.motor_K,
            params.motor_b,
            params.motor_J,
            params.motor_gamma_throttle,
            params.motor_throttle_tau,
            params.motor_T_max if params.motor_T_max is not None else (params.motor_K * (params.motor_V_max / max(params.motor_R, 1e-4))),
            params.motor_P_max if params.motor_P_max is not None else 0.0,
            params.gear_ratio,
            params.eta_gb,
            params.brake_T_max,
            params.brake_tau,
            params.brake_p,
            params.mu,
            params.wheel_radius,
            params.wheel_inertia,
            params.motor_min_current_A,
        ], dtype=np.float64)

        return self._predict_acceleration(param_array, speed, throttle, brake, grade)
    
    def evaluate_fit(
        self,
        params: FittedVehicleParams,
        data_path: Optional[Path] = None,
        segments: Optional[List[TripSegment]] = None,
    ) -> Dict:
        """Evaluate fit quality on data.
        
        Args:
            params: Fitted parameters to evaluate
            data_path: Path to data (if segments not provided)
            segments: Pre-loaded segments (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if segments is None:
            if data_path is None:
                if self._segments:
                    segments = self._segments
                else:
                    raise ValueError("Must provide data_path or segments")
            else:
                trips = self.load_trip_data(data_path)
                dt = self._estimate_dt(trips)
                segments = self._create_segments(trips, dt)
        
        # Convert params to array
        param_array = np.array([
            params.mass,
            params.drag_area,
            params.rolling_coeff,
            params.motor_V_max,
            params.motor_R,
            params.motor_K,
            params.motor_b,
            params.motor_J,
            params.motor_gamma_throttle,
            params.motor_throttle_tau,
            params.motor_T_max if params.motor_T_max is not None else (params.motor_K * (params.motor_V_max / max(params.motor_R, 1e-4))),
            params.motor_P_max if params.motor_P_max is not None else 0.0,
            params.gear_ratio,
            params.eta_gb,
            params.brake_T_max,
            params.brake_tau,
            params.brake_p,
            params.mu,
            params.wheel_radius,
            params.wheel_inertia,
            params.motor_min_current_A,
        ])
        
        # Simulate all segments
        all_v_measured = []
        all_v_simulated = []
        segment_errors = []
        
        for segment in segments:
            v_sim, _ = self._simulate_segment(param_array, segment)
            all_v_measured.append(segment.speed)
            all_v_simulated.append(v_sim)
            
            # Per-segment metrics
            seg_mse = np.mean((v_sim - segment.speed) ** 2)
            seg_rmse = np.sqrt(seg_mse)
            segment_errors.append({
                "trip_id": segment.trip_id,
                "length": segment.length,
                "mse": seg_mse,
                "rmse": seg_rmse,
                "max_error": np.max(np.abs(v_sim - segment.speed)),
            })
        
        all_v_measured = np.concatenate(all_v_measured)
        all_v_simulated = np.concatenate(all_v_simulated)
        
        # Global metrics
        mse = np.mean((all_v_measured - all_v_simulated) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_v_measured - all_v_simulated))
        
        ss_res = np.sum((all_v_measured - all_v_simulated) ** 2)
        ss_tot = np.sum((all_v_measured - np.mean(all_v_measured)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
            "num_segments": len(segments),
            "total_samples": len(all_v_measured),
            "segment_errors": segment_errors,
        }
    
    def plot_segment_comparison(
        self,
        params: FittedVehicleParams,
        segment_idx: int = 0,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot measured vs simulated velocity for a segment.
        
        Args:
            params: Fitted parameters
            segment_idx: Index of segment to plot
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            LOGGER.warning("matplotlib not available for plotting")
            return
        
        if not self._segments:
            raise ValueError("No segments loaded. Call fit() first.")
        
        segment = self._segments[segment_idx]
        
        param_array = np.array([
            params.mass, params.drag_area, params.rolling_coeff,
            params.motor_V_max, params.motor_R,
            params.motor_K, params.motor_b, params.motor_J,
            params.motor_gamma_throttle,
            params.motor_throttle_tau,
            params.motor_T_max if params.motor_T_max is not None else (params.motor_K * (params.motor_V_max / max(params.motor_R, 1e-4))),
            params.motor_P_max if params.motor_P_max is not None else 0.0,
            params.gear_ratio, params.eta_gb,
            params.brake_T_max, params.brake_tau, params.brake_p,
            params.mu,
            params.wheel_radius, params.wheel_inertia,
            params.motor_min_current_A,
        ])
        
        v_sim, _ = self._simulate_segment(param_array, segment)
        time = np.arange(segment.length) * segment.dt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Velocity comparison
        ax = axes[0]
        ax.plot(time, segment.speed, 'b-', label='Measured', alpha=0.8)
        ax.plot(time, v_sim, 'r--', label='Simulated', alpha=0.8)
        ax.set_ylabel('Velocity (m/s)')
        ax.legend()
        ax.set_title(f'Segment: {segment.trip_id}')
        ax.grid(True, alpha=0.3)
        
        # Velocity error
        ax = axes[1]
        error = v_sim - segment.speed
        ax.plot(time, error, 'g-', alpha=0.8)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('Velocity Error (m/s)')
        ax.grid(True, alpha=0.3)
        
        # Inputs
        ax = axes[2]
        ax.plot(time, segment.throttle, 'b-', label='Throttle', alpha=0.7)
        ax.plot(time, segment.brake, 'r-', label='Brake', alpha=0.7)
        ax.plot(time, np.degrees(segment.grade), 'g-', label='Grade (°)', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Input')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            LOGGER.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
