"""Parameter randomization and sampling utilities.

This module provides utilities for:
- Creating parameter randomization ranges from fitted parameters
- Sampling extended plant parameters with rejection sampling
- Configuration for parameter randomization
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from fitting.fitter import FittedVehicleParams
from simulation.dynamics import (
    ExtendedPlantParams,
    MotorParams,
    BrakeParams,
    BodyParams,
    WheelParams,
    CreepParams,
)
from utils.capabilities import compute_vehicle_capabilities

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExtendedPlantRandomization and sampling
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExtendedPlantRandomization:
    """Configuration for randomizing extended plant parameters.
    
    Ranges are based on realistic vehicle/motor specifications.
    Use log-uniform sampling for params spanning orders of magnitude (R, K_t, b, J_m).
    Use uniform sampling for other parameters.
    """

    # Basic vehicle parameters
    mass_range: Tuple[float, float] = (1500.0, 6000.0)  # kg - vehicle mass (per doc)
    drag_area_range: Tuple[float, float] = (0.2, 0.8)   # CdA (m²) - drag coefficient * area
    rolling_coeff_range: Tuple[float, float] = (0.007, 0.015)  # C_rr - rolling resistance
    actuator_tau_range: Tuple[float, float] = (0.05, 0.30)  # seconds - actuator time constant
    grade_deg_range: Tuple[float, float] = (-5.7, 5.7)  # degrees - ±10% grade

    # Motor electrical parameters (log-uniform recommended for R, K_t)
    motor_Vmax_range: Tuple[float, float] = (200.0, 800.0)  # V - motor max voltage
    motor_R_range: Tuple[float, float] = (0.02, 0.6)  # Ω - armature resistance (log-uniform)
    motor_K_range: Tuple[float, float] = (0.05, 0.4)  # Nm/A and V·s/rad - K_t = K_e (log-uniform)
    motor_b_range: Tuple[float, float] = (1e-6, 5e-3)  # Nm·s/rad - viscous friction (log-uniform)
    motor_J_range: Tuple[float, float] = (1e-4, 1e-2)  # kg·m² - rotor inertia (log-uniform)
    motor_Tmax_range: Tuple[float, float] | None = None  # Nm - optional max motor torque
    motor_Pmax_range: Tuple[float, float] | None = None  # W - optional max motor power

    # Gearbox
    gear_ratio_range: Tuple[float, float] = (4.0, 20.0)  # N - gear reduction ratio

    # Brake parameters
    brake_tau_range: Tuple[float, float] = (0.04, 0.12)  # seconds - brake time constant
    brake_accel_range: Tuple[float, float] = (8.0, 11.0)  # m/s² - max braking deceleration magnitude
    brake_p_range: Tuple[float, float] = (1.0, 1.8)  # brake exponent (per doc)
    brake_kappa_range: Tuple[float, float] = (0.02, 0.25)  # brake slip constant
    mu_range: Tuple[float, float] = (0.7, 1.0)  # tire friction coefficient

    # Wheel parameters
    wheel_radius_range: Tuple[float, float] = (0.26, 0.38)  # m - typical passenger wheel radii
    wheel_inertia_range: Tuple[float, float] = (0.5, 5.0)  # kg·m² - wheel + rotating assembly

    # Efficiency
    eta_gb_range: Tuple[float, float] = (0.85, 0.98)  # gearbox efficiency
    
    # Creep parameters (optional, None = use fixed default values)
    creep_a_max: float | None = None  # [m/s²] max creep acceleration (None = use default 0.5)
    creep_v_cutoff: float | None = None  # [m/s] creep fade speed (None = use default 1.5)
    creep_v_hold: float | None = None  # [m/s] standstill threshold (None = use default 0.08)
    
    # Feasibility thresholds (for rejection sampling)
    min_accel_from_rest: float = 2.5  # m/s² - minimum required acceleration at standstill
    min_brake_decel: float = 4.0  # m/s² - minimum braking deceleration (legacy, brake_accel_range replaces this)
    min_top_speed: float = 20.0  # m/s - minimum achievable top speed
    skip_feasibility_checks: bool = False  # If True, skip feasibility checks (fitted params mode)
    skip_sanity_checks: bool = False  # If True, skip sanity checks (viscous torque, stall current)

    @classmethod
    def from_config(cls, config: dict) -> 'ExtendedPlantRandomization':
        """Create ExtendedPlantRandomization from config dictionary."""
        if 'vehicle_randomization' not in config:
            return cls()  # Use defaults

        vr_config = config['vehicle_randomization']
        return cls(
            mass_range=tuple(vr_config.get('mass_range', (1500.0, 6000.0))),
            drag_area_range=tuple(vr_config.get('drag_area_range', (0.2, 0.8))),
            rolling_coeff_range=tuple(vr_config.get('rolling_coeff_range', (0.007, 0.015))),
            actuator_tau_range=tuple(vr_config.get('actuator_tau_range', (0.05, 0.30))),
            grade_deg_range=tuple(vr_config.get('grade_range_deg', (-5.7, 5.7))),
            motor_Vmax_range=tuple(vr_config.get('motor_Vmax_range', (200.0, 800.0))),
            motor_R_range=tuple(vr_config.get('motor_R_range', (0.02, 0.6))),
            motor_K_range=tuple(vr_config.get('motor_K_range', (0.05, 0.4))),
            # Support both old 'motor_Bm_range' and new 'motor_b_range' keys
            motor_b_range=tuple(vr_config.get('motor_b_range', vr_config.get('motor_Bm_range', (1e-6, 5e-3)))),
            motor_J_range=tuple(vr_config.get('motor_J_range', (1e-4, 1e-2))),
            motor_Tmax_range=vr_config.get('motor_Tmax_range', vr_config.get('motor_Imax_range')),
            motor_Pmax_range=vr_config.get('motor_Pmax_range'),
            gear_ratio_range=tuple(vr_config.get('gear_ratio_range', (4.0, 20.0))),
            brake_tau_range=tuple(vr_config.get('brake_tau_range', (0.04, 0.12))),
            brake_accel_range=tuple(vr_config.get('brake_accel_range', (8.0, 11.0))),
            brake_p_range=tuple(vr_config.get('brake_p_range', (1.0, 1.8))),
            brake_kappa_range=tuple(vr_config.get('brake_kappa_range', (0.02, 0.25))),
            mu_range=tuple(vr_config.get('mu_range', (0.7, 1.0))),
            wheel_radius_range=tuple(vr_config.get('wheel_radius_range', (0.26, 0.38))),
            wheel_inertia_range=tuple(vr_config.get('wheel_inertia_range', (0.5, 5.0))),
            eta_gb_range=tuple(vr_config.get('eta_gb_range', (0.85, 0.98))),
            # Creep parameters (optional, from top-level 'creep' key if present)
            creep_a_max=config.get('creep', {}).get('a_max'),
            creep_v_cutoff=config.get('creep', {}).get('v_cutoff'),
            creep_v_hold=config.get('creep', {}).get('v_hold'),
            # Feasibility thresholds
            min_accel_from_rest=vr_config.get('min_accel_from_rest', 2.5),
            min_brake_decel=vr_config.get('min_brake_decel', 4.0),
            min_top_speed=vr_config.get('min_top_speed', 20.0),
            skip_feasibility_checks=vr_config.get('skip_feasibility_checks', False),
            skip_sanity_checks=vr_config.get('skip_sanity_checks', False),
        )

    @classmethod
    def from_fitted_params(
        cls,
        fitted_params_path: str,
        spread_pct: float = 0.1,
    ) -> 'ExtendedPlantRandomization':
        """Create ExtendedPlantRandomization centered on fitted vehicle parameters.
        
        This factory method loads fitted parameters from a JSON file and creates
        a randomization config with ranges centered around the fitted values.
        
        Args:
            fitted_params_path: Path to fitted_params.json file
            spread_pct: Spread percentage around fitted means (default: 0.1 = ±10%)
            
        Returns:
            ExtendedPlantRandomization with ranges centered on fitted params
        """
        return create_extended_randomization_from_fitted(
            Path(fitted_params_path),
            spread_pct=spread_pct
        )


def sample_extended_params(rng: np.random.Generator, rand: ExtendedPlantRandomization) -> ExtendedPlantParams:
    """Sample plant parameters for the extended dynamics with rejection sampling.
    
    Uses log-uniform sampling for electrical/mechanical params that span orders of magnitude.
    Rejection sampling ensures:
    - Sufficient acceleration capability from standstill
    - Sufficient braking capability
    - Reasonable top speed
    """

    def _log_uniform(lo: float, hi: float) -> float:
        """Sample from log-uniform distribution."""
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))

    # Rejection sampling loop
    max_attempts = 200
    for attempt in range(max_attempts):
        # Sample vehicle parameters (uniform)
        mass = float(rng.uniform(*rand.mass_range))
        CdA = float(rng.uniform(*rand.drag_area_range))
        C_rr = float(rng.uniform(*rand.rolling_coeff_range))
        
        # Sample wheel parameters
        wheel_radius = float(rng.uniform(*rand.wheel_radius_range))
        wheel_inertia = _log_uniform(*rand.wheel_inertia_range)
        
        # Sample motor electrical parameters
        # V_max - uniform (user requirement)
        V_max = float(rng.uniform(*rand.motor_Vmax_range))
        # R - log-uniform (spans orders of magnitude)
        R = _log_uniform(*rand.motor_R_range)
        # K_t, K_e - log-uniform (SI: K_t = K_e)
        K_t = _log_uniform(*rand.motor_K_range)
        K_e = K_t  # SI units: K_e = K_t for DC motor
        # b - log-uniform (viscous friction)
        b = _log_uniform(*rand.motor_b_range)
        # J - log-uniform (rotor inertia)
        J = _log_uniform(*rand.motor_J_range)
        
        # Gearbox parameters (uniform)
        gear_ratio = float(rng.uniform(*rand.gear_ratio_range))
        eta_gb = float(rng.uniform(*rand.eta_gb_range))
        
        # Brake parameters - compute T_brake_max from desired braking acceleration
        # a_brake_max = T_brake_max / (r_w * mass)
        # => T_brake_max = a_brake_max * r_w * mass
        desired_brake_accel = float(rng.uniform(*rand.brake_accel_range))
        T_brake_max = desired_brake_accel * wheel_radius * mass
        
        # Optional current/power limits
        T_max = None if rand.motor_Tmax_range is None else float(rng.uniform(*rand.motor_Tmax_range))
        P_max = None if rand.motor_Pmax_range is None else float(rng.uniform(*rand.motor_Pmax_range))

        # Compute vehicle capabilities
        caps = compute_vehicle_capabilities(
            V_max=V_max, R=R, K_e=K_e, K_t=K_t, b=b,
            gear_ratio=gear_ratio, eta_gb=eta_gb, r_w=wheel_radius, mass=mass,
            CdA=CdA, C_rr=C_rr, T_brake_max=T_brake_max,
            T_max=T_max, P_max=P_max,
        )
        
        # Feasibility checks per new_params_randomization.md section 9
        # Skip if explicitly disabled (fitted params mode)
        if not rand.skip_feasibility_checks:
            # Check 1: Minimum acceleration from rest
            if caps['a_max_from_rest'] < rand.min_accel_from_rest:
                continue
            
            # Check 2: Minimum top speed (no-load or steady-state)
            if caps['v_no_load_max'] < rand.min_top_speed:
                continue
            if caps['v_ss_level'] < rand.min_top_speed * 0.8:  # Allow some margin
                continue
            
            # Check 3: Braking deceleration - verify it matches our constraint
            # Since we set T_brake_max = desired_brake_accel * r_w * mass,
            # caps['a_brake_max'] should equal desired_brake_accel
            # Add small tolerance check for numerical precision
            if abs(caps['a_brake_max'] - desired_brake_accel) > 0.1:
                continue
        
        # Sanity checks - skip if flag is set (fitted params mode)
        if not rand.skip_sanity_checks:
            # Sanity check: viscous torque should be small compared to EM torque
            omega_ref = 300.0  # rad/s reference
            I_ref = min(V_max / R, 500.0)  # capped reference current
            tau_visc = b * omega_ref
            tau_em = K_t * I_ref
            if tau_visc > 0.2 * tau_em:
                continue  # Viscous damping too high
            
            # Sanity check: reasonable stall current (cap at 2000A for realism)
            i_stall = V_max / R
            i_limit = i_stall
            if T_max is not None:
                i_limit = min(i_limit, T_max / max(K_t, 1e-9))
            if P_max is not None:
                i_limit = min(i_limit, P_max / max(V_max, 1e-6))
            if i_limit > 2000.0:
                continue
        
        # All checks passed - create parameter objects
        body = BodyParams(
            mass=mass,
            drag_area=CdA,
            rolling_coeff=C_rr,
            grade_rad=np.deg2rad(float(rng.uniform(*rand.grade_deg_range))),
        )
        motor = MotorParams(
            R=R,
            K_e=K_e,
            K_t=K_t,
            b=float(b),
            J=J,
            V_max=V_max,
            T_max=T_max,
            P_max=P_max,
            gear_ratio=gear_ratio,
            eta_gb=eta_gb,
        )
        brake = BrakeParams(
            T_br_max=T_brake_max,
            p_br=float(rng.uniform(*rand.brake_p_range)),
            tau_br=float(rng.uniform(*rand.brake_tau_range)),
            kappa_c=_log_uniform(*rand.brake_kappa_range),
            mu=float(rng.uniform(*rand.mu_range)),
        )
        wheel = WheelParams(
            radius=wheel_radius,
            inertia=wheel_inertia,
            v_eps=0.1,  # keep fixed
        )
        # Creep parameters: use from config if specified, otherwise use defaults
        creep = CreepParams(
            a_max=rand.creep_a_max if rand.creep_a_max is not None else 0.5,
            v_cutoff=rand.creep_v_cutoff if rand.creep_v_cutoff is not None else 1.5,
            v_hold=rand.creep_v_hold if rand.creep_v_hold is not None else 0.08,
        )
        return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel, creep=creep)

    # Fallback if rejection sampling fails
    raise RuntimeError(f"Could not find suitable parameters after {max_attempts} attempts. "
                       f"Consider adjusting parameter ranges in ExtendedPlantRandomization.")


# ---------------------------------------------------------------------------
# Centered randomization from fitted parameters
# ---------------------------------------------------------------------------

def _make_range(
    mean: float,
    spread_pct: float,
    bounds: Optional[Tuple[float, float]] = None,
    enforce_positivity: bool = False,
) -> Tuple[float, float]:
    """Create a range centered on mean with given spread percentage.
    
    Args:
        mean: Center value
        spread_pct: Spread as fraction (0.1 = ±10%)
        bounds: Optional (min, max) to clamp the range
        enforce_positivity: If True, only enforce positivity (low >= 0), ignore other bounds
        
    Returns:
        Tuple of (low, high)
    """
    low = mean * (1.0 - spread_pct)
    high = mean * (1.0 + spread_pct)
    
    if enforce_positivity:
        # Only enforce positivity constraint
        low = max(low, 0.0)
        # high can be anything positive
    elif bounds is not None:
        low = max(low, bounds[0])
        high = min(high, bounds[1])
    
    # Ensure low < high
    if low >= high:
        if enforce_positivity:
            high = max(low + 1e-6, mean * 1.01)  # Ensure positive and reasonable
        elif bounds is not None:
            low, high = bounds
        else:
            high = low + 1e-6
    
    return (low, high)


def _make_log_range(
    mean: float,
    spread_pct: float,
    bounds: Optional[Tuple[float, float]] = None,
    enforce_positivity: bool = False,
) -> Tuple[float, float]:
    """Create a range for log-uniform sampling centered on mean.
    
    For parameters sampled with log-uniform distribution, we want the
    geometric mean to be the fitted value.
    
    Args:
        mean: Center value (geometric mean)
        spread_pct: Spread as fraction in log space
        bounds: Optional (min, max) to clamp the range
        enforce_positivity: If True, only enforce positivity (low > 0), ignore other bounds
        
    Returns:
        Tuple of (low, high)
    """
    # Multiplicative factor: e.g., 10% spread -> 1.2x factor
    factor = 1.0 + spread_pct * 2
    
    low = mean / factor
    high = mean * factor
    
    if enforce_positivity:
        # Only enforce positivity constraint (must be > 0 for log-uniform)
        low = max(low, 1e-10)  # Small positive value
        # high can be anything positive
    elif bounds is not None:
        low = max(low, bounds[0])
        high = min(high, bounds[1])
    
    if low >= high:
        if enforce_positivity:
            high = max(low * 1.01, mean * 1.01)  # Ensure positive and reasonable
        elif bounds is not None:
            low, high = bounds
        else:
            high = low * 1.01
    
    return (low, high)


@dataclass
class CenteredRandomizationConfig:
    """Configuration for creating centered randomization from fitted params.
    
    This stores the parameters needed to build an ExtendedPlantRandomization
    instance centered on fitted vehicle parameters.
    
    All parameters match the actual ExtendedPlant model used in simulation.
    """
    
    # === BODY PARAMETERS (fitted) ===
    mass: float
    drag_area: float
    rolling_coeff: float
    
    # === MOTOR PARAMETERS (fitted) ===
    motor_V_max: float  # V - maximum motor voltage
    motor_R: float  # Ω - armature resistance
    motor_K: float  # Nm/A = V·s/rad - K_t = K_e
    motor_gamma_throttle: float  # throttle nonlinearity exponent
    motor_throttle_tau: float  # s - throttle time constant
    gear_ratio: float  # N - gear reduction ratio
    
    # === BRAKE PARAMETERS (fitted) ===
    brake_T_max: float  # Nm - maximum brake torque at wheel
    
    # === WHEEL PARAMETERS ===
    wheel_radius: float  # m - wheel radius
    
    # === FIXED PARAMETERS (from fitted or defaults) ===
    motor_b: float = 1e-3  # Nm·s/rad - viscous friction
    motor_J: float = 1e-3  # kg·m² - rotor inertia
    eta_gb: float = 0.9  # gearbox efficiency
    brake_tau: float = 0.08  # s - brake time constant
    brake_p: float = 1.2  # brake exponent
    brake_kappa: float = 0.08  # brake slip constant
    mu: float = 0.9  # tire friction coefficient
    wheel_inertia: float = 1.5  # kg·m² - wheel + rotating assembly
    
    # === SPREAD CONFIGURATION ===
    spread_pct: float = 0.1  # ±10% default
    
    # Use different spreads for different parameter types (optional)
    mass_spread_pct: Optional[float] = None
    drag_spread_pct: Optional[float] = None
    rolling_spread_pct: Optional[float] = None
    motor_spread_pct: Optional[float] = None
    brake_spread_pct: Optional[float] = None
    wheel_spread_pct: Optional[float] = None
    
    @classmethod
    def from_fitted_params(
        cls,
        fitted: FittedVehicleParams,
        spread_pct: float = 0.1,
        **overrides,
    ) -> "CenteredRandomizationConfig":
        """Create config from fitted vehicle parameters.
        
        Args:
            fitted: Fitted vehicle parameters
            spread_pct: Default spread percentage for all parameters
            **overrides: Override any field (e.g., mass_spread_pct=0.05)
            
        Returns:
            CenteredRandomizationConfig instance
        """
        return cls(
            # Body params
            mass=fitted.mass,
            drag_area=fitted.drag_area,
            rolling_coeff=fitted.rolling_coeff,
            # Motor params
            motor_V_max=fitted.motor_V_max,
            motor_R=fitted.motor_R,
            motor_K=fitted.motor_K,
            motor_gamma_throttle=fitted.motor_gamma_throttle,
            motor_throttle_tau=fitted.motor_throttle_tau,
            gear_ratio=fitted.gear_ratio,
            # Brake params
            brake_T_max=fitted.brake_T_max,
            # Wheel params
            wheel_radius=fitted.wheel_radius,
            # Fixed params from fitted
            motor_b=fitted.motor_b,
            motor_J=fitted.motor_J,
            eta_gb=fitted.eta_gb,
            brake_tau=fitted.brake_tau,
            brake_p=fitted.brake_p,
            brake_kappa=fitted.brake_kappa,
            mu=fitted.mu,
            wheel_inertia=fitted.wheel_inertia,
            # Spread
            spread_pct=spread_pct,
            **overrides,
        )
    
    def _get_spread(self, param_name: str) -> float:
        """Get spread for a specific parameter."""
        specific = getattr(self, f"{param_name}_spread_pct", None)
        return specific if specific is not None else self.spread_pct
    
    def to_extended_randomization_dict(self) -> Dict:
        """Convert to a dictionary suitable for ExtendedPlantRandomization.from_config().
        
        When used with fitted parameters, this creates ranges centered on fitted values
        with only positivity constraints (no hard bounds) and permissive feasibility
        thresholds, since the target speed profile generator handles feasibility.
        
        Returns:
            Dictionary with vehicle_randomization key for config loading
        """
        # Build ranges for each parameter, centered on fitted values
        # Use enforce_positivity=True to only enforce positivity, not hard bounds
        
        # === BODY PARAMETERS ===
        mass_range = _make_range(
            self.mass, 
            self._get_spread("mass"),
            enforce_positivity=True,  # Only enforce mass > 0
        )
        
        drag_area_range = _make_range(
            self.drag_area,
            self._get_spread("drag"),
            enforce_positivity=True,  # Only enforce CdA > 0
        )
        
        rolling_coeff_range = _make_range(
            self.rolling_coeff,
            self._get_spread("rolling"),
            enforce_positivity=True,  # Only enforce C_rr > 0
        )
        
        # === MOTOR PARAMETERS ===
        motor_spread = self._get_spread("motor")
        
        motor_Vmax_range = _make_range(
            self.motor_V_max,
            motor_spread,
            enforce_positivity=True,  # Only enforce V_max > 0
        )

        motor_gamma_throttle_range = _make_range(
            self.motor_gamma_throttle,
            motor_spread,
            enforce_positivity=True,
        )

        motor_throttle_tau_range = _make_range(
            self.motor_throttle_tau,
            motor_spread,
            enforce_positivity=True,
        )

        motor_R_range = _make_log_range(
            self.motor_R,
            motor_spread,
            enforce_positivity=True,  # Only enforce R > 0
        )
        
        motor_K_range = _make_log_range(
            self.motor_K,
            motor_spread,
            enforce_positivity=True,  # Only enforce K > 0
        )
        
        motor_b_range = _make_log_range(
            self.motor_b,
            motor_spread,
            enforce_positivity=True,  # Only enforce b > 0
        )
        
        motor_J_range = _make_log_range(
            self.motor_J,
            motor_spread,
            enforce_positivity=True,  # Only enforce J > 0
        )
        
        gear_ratio_range = _make_range(
            self.gear_ratio,
            motor_spread,
            enforce_positivity=True,  # Only enforce gear_ratio > 0
        )
        
        eta_gb_range = _make_range(
            self.eta_gb,
            motor_spread * 0.3,  # Efficiency shouldn't vary much
            enforce_positivity=True,  # Only enforce eta_gb > 0
        )
        
        # === BRAKE PARAMETERS ===
        brake_spread = self._get_spread("brake")
        
        brake_tau_range = _make_range(
            self.brake_tau,
            brake_spread,
            enforce_positivity=True,  # Only enforce tau > 0
        )
        
        brake_Tmax_range = _make_range(
            self.brake_T_max,
            brake_spread,
            enforce_positivity=True,  # Only enforce T_max > 0
        )
        
        # Compute brake acceleration range from torque range
        # a_brake = T_brake / (r_w * mass)
        brake_accel_low = brake_Tmax_range[0] / (self.wheel_radius * self.mass)
        brake_accel_high = brake_Tmax_range[1] / (self.wheel_radius * self.mass)
        brake_accel_range = (brake_accel_low, brake_accel_high)
        
        brake_p_range = _make_range(
            self.brake_p,
            brake_spread * 0.5,
            enforce_positivity=True,  # Only enforce p > 0
        )
        
        brake_kappa_range = _make_log_range(
            self.brake_kappa,
            brake_spread,
            enforce_positivity=True,  # Only enforce kappa > 0
        )
        
        mu_range = _make_range(
            self.mu,
            brake_spread * 0.5,
            enforce_positivity=True,  # Only enforce mu > 0
        )
        
        # === WHEEL PARAMETERS ===
        wheel_spread = self._get_spread("wheel") * 0.5  # Tighter for wheel
        
        wheel_radius_range = _make_range(
            self.wheel_radius,
            wheel_spread,
            enforce_positivity=True,  # Only enforce radius > 0
        )
        
        wheel_inertia_range = _make_log_range(
            self.wheel_inertia,
            self._get_spread("wheel"),
            enforce_positivity=True,  # Only enforce inertia > 0
        )
        
        return {
            "vehicle_randomization": {
                # Body
                "mass_range": list(mass_range),
                "drag_area_range": list(drag_area_range),
                "rolling_coeff_range": list(rolling_coeff_range),
                # Motor
                "motor_Vmax_range": list(motor_Vmax_range),
                "motor_R_range": list(motor_R_range),
                "motor_K_range": list(motor_K_range),
                "motor_b_range": list(motor_b_range),
                "motor_J_range": list(motor_J_range),
                "motor_gamma_throttle_range": list(motor_gamma_throttle_range),
                "motor_throttle_tau_range": list(motor_throttle_tau_range),
                "gear_ratio_range": list(gear_ratio_range),
                "eta_gb_range": list(eta_gb_range),
                # Brake
                "brake_tau_range": list(brake_tau_range),
                "brake_accel_range": list(brake_accel_range),
                "brake_p_range": list(brake_p_range),
                "brake_kappa_range": list(brake_kappa_range),
                "mu_range": list(mu_range),
                # Wheel
                "wheel_radius_range": list(wheel_radius_range),
                "wheel_inertia_range": list(wheel_inertia_range),
                # Grade is environmental, not vehicle-specific
                "grade_range_deg": [-5.7, 5.7],
                # Actuator tau (use default spread)
                "actuator_tau_range": [0.05, 0.30],
                # Feasibility thresholds - set very permissive since profile generator handles feasibility
                "min_accel_from_rest": 0.1,  # Very permissive (was 2.5)
                "min_brake_decel": 0.1,  # Very permissive (was 4.0)
                "min_top_speed": 0.1,  # Very permissive (was 20.0)
                # Skip feasibility and sanity checks when using fitted params (profile generator handles feasibility)
                "skip_feasibility_checks": True,
                "skip_sanity_checks": True,
            }
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            # Body params
            "mass": self.mass,
            "drag_area": self.drag_area,
            "rolling_coeff": self.rolling_coeff,
            # Motor params
            "motor_V_max": self.motor_V_max,
            "motor_R": self.motor_R,
            "motor_K": self.motor_K,
            "motor_gamma_throttle": self.motor_gamma_throttle,
            "motor_throttle_tau": self.motor_throttle_tau,
            "gear_ratio": self.gear_ratio,
            # Brake params
            "brake_T_max": self.brake_T_max,
            # Wheel params
            "wheel_radius": self.wheel_radius,
            # Fixed params
            "motor_b": self.motor_b,
            "motor_J": self.motor_J,
            "eta_gb": self.eta_gb,
            "brake_tau": self.brake_tau,
            "brake_p": self.brake_p,
            "brake_kappa": self.brake_kappa,
            "mu": self.mu,
            "wheel_inertia": self.wheel_inertia,
            # Spread
            "spread_pct": self.spread_pct,
            "mass_spread_pct": self.mass_spread_pct,
            "drag_spread_pct": self.drag_spread_pct,
            "rolling_spread_pct": self.rolling_spread_pct,
            "motor_spread_pct": self.motor_spread_pct,
            "brake_spread_pct": self.brake_spread_pct,
            "wheel_spread_pct": self.wheel_spread_pct,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "CenteredRandomizationConfig":
        """Create from dictionary."""
        return cls(**d)
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved centered randomization config to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "CenteredRandomizationConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_extended_randomization_from_fitted(
    fitted_params_path: Path,
    spread_pct: float = 0.1,
):
    """Create ExtendedPlantRandomization from fitted parameters file.
    
    This is a convenience function for use in simulation scripts.
    
    Args:
        fitted_params_path: Path to fitted_params.json
        spread_pct: Spread percentage around fitted means
        
    Returns:
        ExtendedPlantRandomization instance
    """
    fitted = FittedVehicleParams.load(fitted_params_path)
    config = CenteredRandomizationConfig.from_fitted_params(fitted, spread_pct)
    rand_dict = config.to_extended_randomization_dict()
    
    return ExtendedPlantRandomization.from_config(rand_dict)


__all__ = [
    "ExtendedPlantRandomization",
    "sample_extended_params",
    "CenteredRandomizationConfig",
    "create_extended_randomization_from_fitted",
]
