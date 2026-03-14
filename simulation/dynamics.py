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
    min_current_A: float = 0.0  # minimum commanded motor current at zero throttle (A)
    gear_ratio: float = 10.0  # gear reduction ratio
    eta_gb: float = 0.9  # gearbox efficiency


@dataclass(slots=True)
class BrakeParams:
    T_br_max: float = 8000.0
    p_br: float = 1.2
    tau_br: float = 0.08
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
class ExtendedPlantParams:
    motor: MotorParams = field(default_factory=MotorParams)
    brake: BrakeParams = field(default_factory=BrakeParams)
    body: BodyParams = field(default_factory=BodyParams)
    wheel: WheelParams = field(default_factory=WheelParams)


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
        # Single signed command convention:
        #   action > 0: throttle request
        #   action < 0: brake request
        #   action = 0: neutral / creep-only behavior via min_current_A
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

        # ===== EXTERNAL FORCES =====
        # Evaluate load forces from the current rotational state.
        # This keeps the hold/release decision tied to the same kinematics that
        # define back-EMF and motor torque in this substep.
        omega_m_curr = self.motor_omega
        v_forces = (omega_m_curr / N) * r_w
        # Quadratic aerodynamic drag (signed to oppose motion).
        F_drag = 0.5 * body.air_density * body.drag_area * v_forces * abs(v_forces)
        v_threshold = 0.1
        # Rolling resistance ramps in near zero speed to avoid a hard discontinuity.
        roll_factor = min(1.0, abs(v_forces) / v_threshold)
        F_roll = body.rolling_coeff * body.mass * GRAVITY * roll_factor
        # Positive grade_rad is uphill, so F_grade > 0 opposes forward motion.
        grade_rad = self._current_grade_rad if self._current_grade_rad is not None else body.grade_rad
        F_grade = body.mass * GRAVITY * np.sin(grade_rad)

        # ===== BRAKE ACTUATOR TORQUE =====
        # Brake map and first-order actuator lag:
        #   T_br_cmd = T_br_max * u_br^p_br
        #   dT_br/dt = (T_br_cmd - T_br) / tau_br
        T_br_cmd = brake_params.T_br_max * (brake_cmd ** brake_params.p_br)
        self.brake_torque += dt / max(brake_params.tau_br, 1e-4) * (T_br_cmd - self.brake_torque)
        T_brake_actuator = max(self.brake_torque, 0.0)

        # ===== VOLTAGE / CURRENT / MOTOR TORQUE =====
        # Current request is shaped throttle between floor and max current.
        # The floor allows creep-like behavior at zero throttle when configured.
        i_floor = max(motor.min_current_A, 0.0)
        i_span = max(I_max - i_floor, 0.0)
        target_current = i_floor + u_th_shaped * i_span

        # Electrical steady-state relation used for command voltage:
        #   V = R*i + K_e*omega_m
        back_emf = K_e * omega_m_curr
        v_required = target_current * R + back_emf
        v_applied = min(v_required, motor.V_max) if target_current > 0 else 0.0
        self.V_cmd = max(v_applied, 0.0)

        # Combined inertia reflected to motor shaft for the rigid single-DOF chain.
        # Reflected wheel/vehicle inertia scales as 1/N^2 because omega_m = N * omega_w.
        J_eq = J_m + (J_w + body.mass * r_w ** 2) / (N ** 2)

        allow_regen = False
        # Quasi-static electrical model: i = (V_cmd - K_e * omega_m) / R.
        i_steady = (self.V_cmd - K_e * omega_m_curr) / max(R, 1e-9)
        if not allow_regen:
            # Enforce no-regeneration behavior in this plant.
            i_steady = max(i_steady, 0.0)

        # Voltage-limited current accounts for back-EMF at the current shaft speed.
        i_limit_voltage = max((motor.V_max - K_e * omega_m_curr) / max(R, 1e-9), 0.0)
        self.i_limit = float(i_limit_voltage)

        # Compose final current limit from voltage, torque, and optional power limits.
        i_effective_limit = i_limit_voltage
        if I_max > 0:
            i_effective_limit = min(i_effective_limit, I_max)
        if P_max is not None and self.V_cmd > 1e-6:
            i_effective_limit = min(i_effective_limit, P_max / self.V_cmd)

        # Motor-to-wheel torque mapping through gearbox efficiency and ratio.
        self.motor_current = float(min(i_steady, i_effective_limit))
        tau_m_shaft = K_t * self.motor_current
        tau_drive_wheel = eta * N * tau_m_shaft
        self.drive_torque = float(tau_drive_wheel)

        # ===== STICK / HOLD TORQUE BALANCE =====
        # Hold window near standstill.
        v_stick = max(wheel.v_eps, 0.05)
        # Friction-limited transmissible brake torque at the contact patch.
        N_normal = body.mass * GRAVITY * np.cos(grade_rad)
        T_friction_limit = brake_params.mu * N_normal * r_w
        # Actual hold capacity is limited by both actuator torque and tire-road friction.
        T_hold_max = min(T_brake_actuator, T_friction_limit)

        # Positive T_load_wheel opposes forward-driving wheel torque.
        # Free net wheel torque (without brake hold term):
        #   T_net_free > 0 -> tends to move forward
        #   T_net_free < 0 -> tends to move backward
        T_load_wheel = (F_drag + F_roll + F_grade) * r_w
        T_net_free = tau_drive_wheel - T_load_wheel

        if abs(self.speed) < v_stick and abs(T_net_free) <= T_hold_max:
            # Stick condition: brakes can fully balance the free net wheel torque.
            # Result: kinematic state remains exactly at rest for this substep.
            self.held_by_brakes = True
            self.speed = 0.0
            self.acceleration = 0.0
            self.motor_omega = 0.0
            self.wheel_omega = 0.0
            self.back_emf_voltage = 0.0
            self.position += 0.0

            # Keep force channels coherent in the stuck state.
            self.drag_force = F_drag
            self.rolling_force = F_roll
            self.grade_force = F_grade
            self.net_force = 0.0
            self.tire_force = self.drag_force + self.rolling_force + self.grade_force
            self.slip_ratio = 0.0
            self._coupling_enabled = True

            self.tau_m_prev = tau_m_shaft
            self.motor_current_prev = self.motor_current
            self.drive_torque_prev = self.drive_torque
            return

        self.held_by_brakes = False

        # ===== BRAKE TORQUE DURING MOTION =====
        # Once released, brake torque opposes actual motion; if nearly zero speed,
        # use impending-motion direction from T_net_free.
        if abs(v_forces) > v_stick:
            motion_sign = np.sign(v_forces)
        else:
            motion_sign = np.sign(T_net_free) if abs(T_net_free) > 1e-9 else 0.0

        # Positive tau_brake_wheel always opposes forward-driving torque.
        tau_brake_wheel = motion_sign * T_hold_max

        # ===== NORMAL DYNAMICS AFTER RELEASE =====
        # Wheel-side opposing torque reflected to motor shaft.
        tau_wheel_opp = tau_brake_wheel + T_load_wheel
        tau_reflected = tau_wheel_opp / max(eta * N, 1e-12)

        # Single-DOF mechanical dynamics:
        #   J_eq * domega_m/dt = tau_m_shaft - b*omega_m - tau_reflected
        domega_dt = (tau_m_shaft - b * omega_m_curr - tau_reflected) / max(J_eq, 1e-12)
        omega_m_new = omega_m_curr + dt * domega_dt

        # Prevent sign changes under active brake command to avoid low-speed chatter.
        omega_initial_sign = np.sign(omega_m_curr) if abs(omega_m_curr) > 1e-6 else 0.0
        if brake_cmd > 0.1:
            if omega_initial_sign > 0.0 and omega_m_new < 0.0:
                omega_m_new = 0.0
            elif omega_initial_sign < 0.0 and omega_m_new > 0.0:
                omega_m_new = 0.0

        # Propagate rigid kinematics from motor state (single source of truth).
        self.motor_omega = float(omega_m_new)
        self.back_emf_voltage = K_e * self.motor_omega
        self.wheel_omega = self.motor_omega / N

        v_old = self.speed
        v_new = self.wheel_omega * r_w
        self.speed = float(v_new)
        self.acceleration = (v_new - v_old) / max(dt, 1e-6)
        self.position += self.speed * dt

        self.tau_m_prev = tau_m_shaft
        self.motor_current_prev = self.motor_current
        self.drive_torque_prev = self.drive_torque

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

        # Coupling is always enabled in single-DOF model
        self._coupling_enabled = True

        # Compute slip ratio (for diagnostics)
        # In single-DOF model, wheel and vehicle are rigidly coupled, so slip is minimal
        wheel_linear_speed = self.wheel_omega * wheel.radius
        # Guard denominator near zero to avoid numerical blow-up at standstill.
        v_ref = max(abs(self.speed), wheel.v_eps)
        self.slip_ratio = (wheel_linear_speed - self.speed) / v_ref

        # Save previous wheel state for next substep
        #self.wheel_omega_prev = self.wheel_omega


__all__ = [
    "GRAVITY",
    "DEFAULT_AIR_DENSITY",
    "MotorParams",
    "BrakeParams",
    "BodyParams",
    "WheelParams",
    "ExtendedPlantParams",
    "ExtendedPlant",
    "ExtendedPlantState",
]


