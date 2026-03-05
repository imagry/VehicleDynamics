"""Longitudinal dynamics helpers and extended plant models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


GRAVITY: float = 9.80665  # m/s^2
DEFAULT_AIR_DENSITY: float = 1.225  # kg/m^3


# ---------------------------------------------------------------------------
# Extended plant configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MotorParams:
    R: float = 0.2  # armature resistance (Ω)
    K_e: float = 0.2  # back-EMF constant (V/(rad/s))
    K_t: float = 0.2  # torque constant (Nm/A)
    b: float = 1e-3  # viscous friction (Nm·s/rad) - was B_m
    J: float = 1e-3  # rotor inertia (kg·m²) - CTMS model
    V_max: float = 400.0  # max motor voltage (V)
    T_max: float | None = None  # max motor torque (Nm) - optional
    P_max: float | None = None  # max motor power (W) - optional
    gamma_throttle: float = 1.0  # throttle-to-current nonlinearity exponent
    throttle_tau: float = 0.1  # throttle command time constant (s)
    gear_ratio: float = 10.0  # gear reduction ratio
    eta_gb: float = 0.9  # gearbox efficiency


@dataclass(slots=True)
class BrakeParams:
    T_br_max: float = 8000.0
    p_br: float = 1.2
    tau_br: float = 0.08
    kappa_c: float = 0.08
    mu: float = 0.9


@dataclass(slots=True)
class BodyParams:
    mass: float = 1400.0
    drag_area: float = 0.7
    rolling_coeff: float = 0.01
    grade_rad: float = 0.0
    air_density: float = DEFAULT_AIR_DENSITY


@dataclass(slots=True)
class WheelParams:
    radius: float = 0.30
    inertia: float = 1.2
    v_eps: float = 0.1


@dataclass(slots=True)
class CreepParams:
    """EV-style creep torque parameters.
    
    Creep provides low-speed forward motion at zero throttle, mimicking
    ICE idle behavior without introducing idle RPMs or discontinuities.
    """
    a_max: float = 0.5      # [m/s²] maximum creep acceleration
    v_cutoff: float = 1.5   # [m/s] speed where creep fully fades out
    v_hold: float = 0.08    # [m/s] standstill region threshold


@dataclass(slots=True)
class ExtendedPlantParams:
    motor: MotorParams = field(default_factory=MotorParams)
    brake: BrakeParams = field(default_factory=BrakeParams)
    body: BodyParams = field(default_factory=BodyParams)
    wheel: WheelParams = field(default_factory=WheelParams)
    creep: CreepParams = field(default_factory=CreepParams)


# ---------------------------------------------------------------------------
# Extended plant state and simulation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExtendedPlantState:
    speed: float
    position: float
    acceleration: float
    wheel_speed: float
    brake_torque: float
    slip_ratio: float
    action: float
    motor_current: float
    motor_omega: float  # motor shaft angular speed (rad/s) - CTMS model
    back_emf_voltage: float  # back-EMF voltage (K_e * omega_m)
    V_cmd: float  # commanded voltage (input motor voltage)
    i_limit: float  # voltage-limited current: (V_max - K_e * omega_m) / R (accounts for back EMF)
    # Forces and torques
    drive_torque: float
    tire_force: float
    drag_force: float
    rolling_force: float
    grade_force: float
    net_force: float
    held_by_brakes: bool  # True when vehicle is held at rest by brakes/static friction
    creep_torque: float  # Creep torque at motor shaft (Nm) - for diagnostics
    coupling_enabled: bool  # True when motor is coupled to wheel (False during braking)


class ExtendedPlant:
    """DC-motor throttle + nonlinear brake + wheel/vehicle longitudinal plant."""

    def __init__(self, params: ExtendedPlantParams):
        self.params = params
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, speed: float = 0.0, position: float = 0.0) -> ExtendedPlantState:
        self.speed = speed  # Allow negative speeds for testing reverse motion
        self.position = position
        self.acceleration = 0.0
        # Initialize wheel angular speed (rad/s) and previous state
        self.wheel_omega = self.speed / max(self.params.wheel.radius, 1e-3)  # rad/s
        #self.wheel_omega_prev = self.wheel_omega  # saved from previous substep
        self.brake_torque = 0.0
        self.motor_current = 0.0
        # Initialize motor angular speed (CTMS model) - synced to wheel initially
        self.motor_omega = self.wheel_omega * self.params.motor.gear_ratio  # motor shaft rad/s
        self.back_emf_voltage = self.params.motor.K_e * self.motor_omega  # back-EMF voltage
        self.V_cmd = 0.0  # commanded voltage (input motor voltage)
        self.i_limit = 0.0  # voltage-limited current: (V_max - K_e * omega_m) / R (accounts for back EMF)
        self.last_action = 0.0
        self.slip_ratio = 0.0
        # Initialize throttle state for first-order dynamics
        self.throttle_state = 0.0  # filtered throttle command
        # Initialize forces
        self.drive_torque = 0.0
        self.tire_force = 0.0
        self.drag_force = 0.0
        self.rolling_force = 0.0
        self.grade_force = 0.0
        self.net_force = 0.0
        self.held_by_brakes = False
        self.creep_torque = 0.0  # Initialize creep torque
        self._current_grade_rad = None  # Current grade override (None = use body.grade_rad)
        # Initialize previous motor states for coupling
        self.motor_current_prev = 0.0
        self.tau_m_prev = 0.0
        self.drive_torque_prev = 0.0
        # Initialize coupling state (motor coupled to wheel)
        self._coupling_enabled = True
        self._coupling_enabled_prev = True
        return self.state

    # ------------------------------------------------------------------
    @property
    def state(self) -> ExtendedPlantState:
        return ExtendedPlantState(
            speed=self.speed,
            position=self.position,
            acceleration=self.acceleration,
            wheel_speed=self.wheel_omega * self.params.wheel.radius,  # expose linear speed for API compatibility
            brake_torque=self.brake_torque,
            slip_ratio=self.slip_ratio,
            action=self.last_action,
            motor_current=self.motor_current,
            motor_omega=self.motor_omega,
            back_emf_voltage=self.back_emf_voltage,
            V_cmd=self.V_cmd,
            i_limit=self.i_limit,
            drive_torque=self.drive_torque,
            tire_force=self.tire_force,
            drag_force=self.drag_force,
            rolling_force=self.rolling_force,
            grade_force=self.grade_force,
            net_force=self.net_force,
            held_by_brakes=self.held_by_brakes,
            creep_torque=self.creep_torque,
            coupling_enabled=self._coupling_enabled,
        )

    # ------------------------------------------------------------------
    def step(self, action: float, dt: float, substeps: int = 1, grade_rad: float | None = None) -> ExtendedPlantState:
        """Advance the plant by ``dt`` seconds, optionally using sub-steps."""

        # Store the grade for this step (None means use default body.grade_rad)
        self._current_grade_rad = grade_rad

        dt = max(dt, 1e-6)
        sub_dt = dt / max(substeps, 1)
        clipped_action = float(np.clip(action, -1.0, 1.0))
        for _ in range(max(substeps, 1)):
            self._substep(clipped_action, sub_dt)
        self.last_action = clipped_action
        return self.state

    # ------------------------------------------------------------------
    def _substep(self, action: float, dt: float) -> None:
        """Single-DOF rigid coupling model: motor always coupled to wheel via gearbox.
        
        Based on coupling.md specification:
        - Motor always coupled to wheels (single rotational DOF)
        - Brake acts as torque at wheel, reflected to motor shaft
        - No regeneration: negative current clamped to zero
        - Uses combined inertia J_eq = J_m + (J_w + m*r_w²) / N²
        - omega_m is the single source of truth; speed derived from it
        """
        motor = self.params.motor
        brake_params = self.params.brake
        wheel = self.params.wheel
        body = self.params.body

        # ===== ACTION MAPPING (CURRENT CONTROL) =====
        u = action
        u_th = max(0, u)  # throttle: u > 0
        brake_cmd = max(-u, 0.0)

        # ===== THROTTLE DYNAMICS (FIRST-ORDER LAG + NONLINEARITY) =====
        # Apply first-order lag: tau * dθ/dt = θ_cmd - θ
        # Discretized: θ_new = θ_old + (θ_cmd - θ_old) * (dt / tau)
        tau_throttle = max(motor.throttle_tau, 1e-4)  # Prevent division by zero
        alpha = dt / (tau_throttle + dt)  # discrete time constant
        self.throttle_state = self.throttle_state + alpha * (u_th - self.throttle_state)
        
        # Apply throttle nonlinearity (gamma exponent)
        # This shapes the throttle response: γ < 1 gives more low-end response, γ > 1 reduces it
        u_th_shaped = self.throttle_state ** motor.gamma_throttle

        # ===== MOTOR PARAMETERS =====
        R = motor.R
        K_e = motor.K_e
        K_t = motor.K_t
        b = motor.b  # viscous friction
        J_m = motor.J  # motor rotor inertia
        N = motor.gear_ratio
        eta = motor.eta_gb
        r_w = wheel.radius
        J_w = wheel.inertia

        # Current and power limits
        I_max = (motor.T_max / max(K_t, 1e-9)) if motor.T_max is not None else (motor.V_max / max(R, 1e-9))
        P_max = motor.P_max
        
        # ===== COMPUTE BRAKE TORQUE EARLY (for creep subtraction) =====
        # Compute brake torque magnitude to subtract from creep torque
        # Brake dynamics (first-order lag)
        denom = 1.0 + brake_params.kappa_c * (1.0 - brake_cmd)
        denom = max(denom, 1e-6)
        T_br_cmd = brake_params.T_br_max * (brake_cmd ** brake_params.p_br) / denom
        self.brake_torque += dt / max(brake_params.tau_br, 1e-4) * (T_br_cmd - self.brake_torque)
        
        # Brake torque magnitude at wheel
        tau_brake_mag = max(self.brake_torque, 0.0)
        
        # Reflect brake torque to motor shaft (for subtracting from creep)
        # Use magnitude only - direction will be handled later in brake logic
        tau_brake_motor_mag = tau_brake_mag / max(eta * N, 1e-12)
        
        # ===== CREEP TORQUE COMPUTATION =====
        # Compute EV-style creep behavior: low-speed forward motion at zero throttle
        # Creep is parameterized by max acceleration and fades with speed
        creep = self.params.creep
        
        # Step 1: Convert creep acceleration to motor torque (dynamic computation)
        # This ensures creep adapts to vehicle mass, gear ratio, etc.
        F_creep_max = body.mass * creep.a_max  # [N] max creep force
        T_wheel_creep_max = F_creep_max * r_w  # [Nm] max creep torque at wheel
        T_motor_creep_max = T_wheel_creep_max / (N * eta)  # [Nm] max creep torque at motor shaft
        
        # Step 2: Speed-dependent fade using gentler power function
        # Creep fades smoothly from full at v=0 to zero at v=v_cutoff
        # Use a gentler fade curve that maintains more torque at higher speeds
        # Use current motor omega to compute vehicle speed
        omega_m_current = self.motor_omega
        v_current = (omega_m_current / N) * r_w  # current vehicle speed from motor
        v_abs = abs(v_current)
        x = v_abs / max(creep.v_cutoff, 1e-6)  # normalized speed
        if x < 1.0:
            # Use a very gentle power fade: w = 1 - x^5
            # Original cubic smoothstep: w = 1 - 3x^2 + 2x^3 (maintains ~50% torque at x=0.5)
            # Power fade: w = 1 - x^5 (maintains ~97% torque at x=0.5, ~33% at x=0.925)
            # This allows vehicle to reach very close to v_cutoff before equilibrium
            # The fade starts immediately but decays very slowly, avoiding steep curves
            w_fade = 1.0 - x**5  # very gentle power fade
        else:
            w_fade = 0.0
        
        # Step 3: Brake torque subtracts from creep torque
        # Creep torque is reduced by brake torque magnitude at motor shaft
        T_creep_motor_unclamped = T_motor_creep_max * w_fade
        T_creep_motor = max(0.0, T_creep_motor_unclamped - tau_brake_motor_mag)
        self.creep_torque = float(T_creep_motor)  # Store for diagnostics
        
        # Step 4: Convert creep torque to equivalent current (current control)
        I_creep = T_creep_motor / max(K_t, 1e-9)

        # Current command (creep always active, throttle always adds on top)
        target_current = I_creep + u_th_shaped * I_max

        # Compute voltage required to achieve target current
        back_emf = K_e * omega_m_current
        v_required = target_current * R + back_emf
        v_applied = min(v_required, motor.V_max) if target_current > 0 else 0.0
        self.V_cmd = max(v_applied, 0.0)

        # Combined inertia at motor shaft for single-DOF rigid coupling:
        # J_eff = J_m + (J_w + m * r_w^2) / N^2
        # 
        # The wheel and vehicle mass inertias are DIVIDED by N² (not multiplied)
        # because when the motor spins N times faster than the wheel, the 
        # reflected inertia is reduced by N² (from energy conservation).
        # 
        # Energy: 0.5 * J_w * ω_w² = 0.5 * J_w * (ω_m/N)² = 0.5 * (J_w/N²) * ω_m²
        # Similarly for vehicle mass: 0.5 * m * v² = 0.5 * (m*r_w²/N²) * ω_m²
        J_eq = J_m + (J_w + body.mass * r_w ** 2) / (N ** 2)

        allow_regen = False

        # ===== KINEMATICS FROM MOTOR STATE =====
        # omega_m is the single source of truth
        omega_m = self.motor_omega
        omega_w = omega_m / N  # wheel angular speed
        v_from_rot = omega_w * r_w  # vehicle speed derived from rotation

        # ===== COMPUTE TIRE FORCE =====
        # Tire force opposes motion (positive when opposing forward)
        # Use friction model based on current state
        mu_k = brake_params.mu
        N_normal = body.mass * GRAVITY
        F_fric_max = mu_k * N_normal

        # External resistive forces (for tire force calculation)
        F_drag = 0.5 * body.air_density * body.drag_area * v_from_rot * abs(v_from_rot)
        v_threshold = 0.1
        roll_factor = min(1.0, abs(v_from_rot) / v_threshold)
        F_roll = body.rolling_coeff * body.mass * GRAVITY * roll_factor
        grade_rad = self._current_grade_rad if self._current_grade_rad is not None else body.grade_rad
        F_grade = body.mass * GRAVITY * np.sin(grade_rad)

        # Tire force is what the ground applies to accelerate the vehicle
        # F_tire = m * a + F_resist (to achieve acceleration a)
        # In steady state or quasi-static, F_tire balances resistive forces
        # For now, compute based on what the motor can provide minus resistances
        # This will be updated iteratively

        # ===== BRAKE TORQUE DIRECTION LOGIC =====
        # Brake torque magnitude (already computed above for creep subtraction)
        # Now determine direction based on vehicle motion
        tau_brake_mag = max(self.brake_torque, 0.0)

        # Brake torque opposes current motion direction
        # Sign convention: positive torque in motor equation accelerates motor forward
        # So brake should apply negative torque if moving forward, positive if backward
        # At rest, brake should hold the vehicle
        v_hold = creep.v_hold  # Use creep parameter for velocity threshold
        v_eps = 0.05  # smooth transition zone
        
        if abs(v_from_rot) < v_hold and tau_brake_mag > 100.0:
            # Nearly stopped with brakes applied: clamp motor omega to zero
            # This prevents oscillation at zero crossing
            self.held_by_brakes = True
            # Use a smooth factor that ramps brake effect to zero near zero speed
            # to prevent discontinuous torque jumps
            speed_factor = min(1.0, abs(v_from_rot) / max(v_eps, 1e-6))
            if v_from_rot > 0:
                tau_brake_wheel = tau_brake_mag * speed_factor
            elif v_from_rot < 0:
                tau_brake_wheel = -tau_brake_mag * speed_factor
            else:
                tau_brake_wheel = 0.0
        elif v_from_rot > v_eps:
            # Moving forward: brake opposes forward motion
            tau_brake_wheel = tau_brake_mag
            self.held_by_brakes = False
        elif v_from_rot < -v_eps:
            # Moving backward: brake opposes backward motion
            tau_brake_wheel = -tau_brake_mag
            self.held_by_brakes = False
        else:
            # In transition zone: smooth interpolation
            speed_factor = v_from_rot / v_eps  # -1 to +1
            tau_brake_wheel = tau_brake_mag * speed_factor
            self.held_by_brakes = tau_brake_mag > 100.0

        # ===== COMPUTE TIRE CONTACT TORQUE =====
        # Tire contact torque at wheel (positive opposes forward motion)
        # T_tire = F_resist * r_w where F_resist opposes motion
        T_tire = (F_drag + F_roll + F_grade) * r_w  # Resistive torque at wheel

        # ===== REFLECT WHEEL TORQUES TO MOTOR SHAFT =====
        # Total wheel opposing torque (at rest, only resistive loads)
        tau_wheel_opp = tau_brake_wheel + T_tire
        # Reflected to motor shaft
        tau_reflected = tau_wheel_opp / max(eta * N, 1e-12)

        # ===== STEADY-STATE CURRENT (NO ELECTRICAL DYNAMICS) =====
        # Assume steady-state: di/dt = 0, so i = (V_cmd - K_e*omega_m) / R
        # This matches the logic in compute_max_accel_at_speed()
        
        omega_m_curr = self.motor_omega
        
        # Track initial omega sign for zero-crossing detection
        omega_initial_sign = np.sign(omega_m_curr) if abs(omega_m_curr) > 1e-6 else 0
        
        # Compute steady-state current from voltage equation
        # V_cmd = R*i + K_e*omega_m  =>  i = (V_cmd - K_e*omega_m) / R
        i_steady = (self.V_cmd - K_e * omega_m_curr) / max(R, 1e-9)
        
        # NO REGEN: clamp negative current to zero
        if not allow_regen:
            i_steady = max(i_steady, 0.0)
        
        # Voltage-limited current (accounts for back EMF)
        # This is the maximum current that can be drawn given voltage limit and back EMF
        i_limit_voltage = (motor.V_max - K_e * omega_m_curr) / max(R, 1e-9)
        # Ensure non-negative
        i_limit_voltage = max(i_limit_voltage, 0.0)
        self.i_limit = float(i_limit_voltage)  # Store for state logging (voltage-limited current)
        
        # Effective current limit (considering all constraints: voltage, torque, power)
        i_effective_limit = i_limit_voltage
        if I_max > 0:
            i_effective_limit = min(i_effective_limit, I_max)
        if P_max is not None and self.V_cmd > 1e-6:
            i_effective_limit = min(i_effective_limit, P_max / self.V_cmd)
        
        i_new = min(i_steady, i_effective_limit)
        
        # Mechanical dynamics: J_eq dω_m/dt = K_t*i - b*ω_m - τ_reflected
        tau_m_shaft = K_t * i_new
        domega_dt = (tau_m_shaft - b * omega_m_curr - tau_reflected) / max(J_eq, 1e-12)
        omega_m_new = omega_m_curr + dt * domega_dt
        
        # Prevent sign change when brakes are applied (would cause oscillation)
        # If braking and omega is about to cross zero, clamp to zero
        if brake_cmd > 0.1:
            if omega_initial_sign > 0 and omega_m_new < 0:
                omega_m_new = 0.0
            elif omega_initial_sign < 0 and omega_m_new > 0:
                omega_m_new = 0.0

        # ===== UPDATE MOTOR STATE =====
        self.motor_current = float(i_new)
        self.motor_omega = float(omega_m_new)
        self.back_emf_voltage = K_e * self.motor_omega

        # ===== COMPUTE DRIVE TORQUE AT WHEEL =====
        tau_m_shaft = K_t * self.motor_current
        tau_drive_wheel = eta * N * tau_m_shaft  # positive forward

        # Save for diagnostics
        self.drive_torque = float(tau_drive_wheel)
        self.tau_m_prev = tau_m_shaft
        self.motor_current_prev = self.motor_current
        self.drive_torque_prev = self.drive_torque

        # ===== COMPUTE WHEEL FORCE CHANNELS =====
        # Raw wheel-contact force channel from post-constraint motor torque and applied brake torque.
        # This is kept for internal mechanics visibility; reported tire_force/net_force are finalized
        # later from realized acceleration so that net_force == m * acceleration by construction.
        F_drive = tau_drive_wheel / r_w  # positive forward

        # tau_brake_wheel is signed (positive opposes forward, negative opposes backward)
        # F_brake is signed brake force at the tire contact patch.
        F_brake = tau_brake_wheel / r_w
        F_tire_raw = F_drive - F_brake

        # ===== DERIVE VEHICLE STATE FROM MOTOR (SINGLE SOURCE OF TRUTH) =====
        # Speed comes directly from omega_m
        v_old = self.speed
        v_new = (self.motor_omega / N) * r_w
        
        # ===== HELD BY BRAKES / ZERO SPEED CLAMPING =====
        # If brakes are applied and speed is very low, clamp to zero to prevent oscillation
        v_hold_clamp = max(creep.v_hold, 0.05)
        if brake_cmd > 0.4 and abs(v_new) < v_hold_clamp:
            # Clamp motor omega to zero (vehicle held)
            self.motor_omega = 0.0
            self.motor_current = 0.0
            v_new = 0.0
            self.held_by_brakes = True
            self.acceleration = 0.0
        else:
            self.held_by_brakes = False
            # Compute acceleration from actual speed change (single-DOF consistent)
            self.acceleration = (v_new - v_old) / max(dt, 1e-6)
        
        self.speed = float(v_new)

        # ===== FINALIZE REPORTED FORCE DIAGNOSTICS (REALIZED STATE) =====
        # Recompute opposing forces from realized vehicle speed for consistent reporting.
        # These reported channels are intentionally aligned so that:
        #   net_force == mass * acceleration
        #   tire_force == net_force + drag + rolling + grade
        # which makes tire/net traces directly interpretable against kinematics.
        v_diag = self.speed
        self.drag_force = 0.5 * body.air_density * body.drag_area * v_diag * abs(v_diag)
        roll_factor_diag = min(1.0, abs(v_diag) / v_threshold)
        self.rolling_force = body.rolling_coeff * body.mass * GRAVITY * roll_factor_diag
        self.grade_force = body.mass * GRAVITY * np.sin(grade_rad)
        self.net_force = body.mass * self.acceleration
        self.tire_force = self.net_force + self.drag_force + self.rolling_force + self.grade_force

        # Wheel omega follows motor
        self.wheel_omega = self.motor_omega / N

        # Update position
        self.position += self.speed * dt

        # Coupling is always enabled in single-DOF model
        self._coupling_enabled = True

        # Compute slip ratio (for diagnostics)
        # In single-DOF model, wheel and vehicle are rigidly coupled, so slip is minimal
        wheel_linear_speed = self.wheel_omega * wheel.radius
        v_ref = max(abs(self.speed), wheel.v_eps)
        self.slip_ratio = (wheel_linear_speed - self.speed) / v_ref

        # Save previous wheel state for next substep
        #self.wheel_omega_prev = self.wheel_omega


__all__ = [
    "GRAVITY",
    
    
    
    
    
    
    
    
    "ExtendedPlantParams",
    "ExtendedPlantRandomization",
    "ExtendedPlant",
    "ExtendedPlantState",
    "sample_extended_params",
    "compute_vehicle_capabilities",
    "compute_max_accel_at_speed",
]


