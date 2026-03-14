"""Analytic inverse feedforward mapping for the longitudinal plant.

This module maps target longitudinal acceleration to a signed plant action in
[-1, 1] by algebraically inverting the clean forward equations.

Design constraints:
- No numerical search / bisection / root finding.
- No internal feasibility enforcement from voltage/torque/power limits.
- Only final action clipping to [-1, 1].
- Delay and stability-hack states are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from simulation.dynamics import ExtendedPlantParams, GRAVITY


ROLLING_SPEED_THRESHOLD_MPS: float = 0.1


@dataclass(slots=True)
class InverseFeedforwardResult:
    """Result of inverting target acceleration into a signed actuation command."""

    action: float
    raw_action: float
    was_clipped: bool
    mode: str
    target_accel: float
    passive_accel: float
    speed: float
    grade_rad: float
    required_motor_current_A: float
    required_brake_torque_Nm: float
    required_wheel_force_N: float
    drag_force_N: float
    rolling_force_N: float
    grade_force_N: float


class AnalyticInverseFeedforward:
    """Closed-form, delay-free inverse feedforward controller.

    The inverse reuses the plant nonlinear mappings but does not enforce runtime
    feasibility constraints in the equations. It returns both an unconstrained
    `raw_action` and the final clipped `action` in [-1, 1].
    """

    def __init__(self, params: ExtendedPlantParams):
        self.params = params

    def __call__(
        self,
        target_accel: float,
        speed: float,
        grade_rad: float | None = None,
    ) -> InverseFeedforwardResult:
        return self.compute_action(target_accel=target_accel, speed=speed, grade_rad=grade_rad)

    def compute_action(
        self,
        target_accel: float,
        speed: float,
        grade_rad: float | None = None,
    ) -> InverseFeedforwardResult:
        """Invert target acceleration into signed action.

        Args:
            target_accel: Desired longitudinal acceleration in m/s^2.
            speed: Current vehicle speed in m/s.
            grade_rad: Road grade in radians. If None, uses body default.

        Returns:
            `InverseFeedforwardResult` with raw and clipped actions.
        """
        motor = self.params.motor
        brake = self.params.brake
        body = self.params.body
        wheel = self.params.wheel

        grade = body.grade_rad if grade_rad is None else float(grade_rad)
        v = float(speed)
        a_target = float(target_accel)

        r_w = max(wheel.radius, 1e-9)
        N = max(motor.gear_ratio, 1e-9)
        eta = max(motor.eta_gb, 1e-9)
        K_t = max(motor.K_t, 1e-9)

        omega_m = (N / r_w) * v
        domega_target = (N / r_w) * a_target

        F_drag, F_roll, F_grade = _resistive_forces(v=v, grade_rad=grade, params=self.params)
        T_tire = (F_drag + F_roll + F_grade) * r_w

        J_eq = _equivalent_inertia(self.params)
        i_floor = max(motor.min_current_A, 0.0)

        # Baseline acceleration at zero throttle and zero brake.
        domega_passive = (
            motor.K_t * i_floor
            - motor.b * omega_m
            - T_tire / (eta * N)
        ) / max(J_eq, 1e-12)
        a_passive = (r_w / N) * domega_passive

        required_wheel_force = body.mass * a_target + F_drag + F_roll + F_grade

        if a_target >= a_passive:
            mode = "drive"
            i_req = (
                J_eq * domega_target
                + motor.b * omega_m
                + T_tire / (eta * N)
            ) / K_t
            raw_action = _invert_throttle_from_current(i_req=i_req, params=self.params)
            required_brake_torque = 0.0
            required_motor_current = i_req
        else:
            mode = "brake"
            tau_brake_signed = (
                eta * N * (motor.K_t * i_floor - motor.b * omega_m - J_eq * domega_target)
                - T_tire
            )
            sign_motion = -1.0 if v < 0.0 else 1.0
            tau_brake_mag = sign_motion * tau_brake_signed
            required_brake_torque = max(tau_brake_mag, 0.0)

            p_br = max(brake.p_br, 1e-9)
            if brake.T_br_max <= 1e-12 or required_brake_torque <= 0.0:
                u_br = 0.0
            else:
                u_br = (required_brake_torque / brake.T_br_max) ** (1.0 / p_br)

            raw_action = -u_br
            required_motor_current = i_floor

        action = float(min(max(raw_action, -1.0), 1.0))
        was_clipped = abs(action - raw_action) > 1e-12

        return InverseFeedforwardResult(
            action=action,
            raw_action=float(raw_action),
            was_clipped=was_clipped,
            mode=mode,
            target_accel=a_target,
            passive_accel=float(a_passive),
            speed=v,
            grade_rad=grade,
            required_motor_current_A=float(required_motor_current),
            required_brake_torque_Nm=float(required_brake_torque),
            required_wheel_force_N=float(required_wheel_force),
            drag_force_N=float(F_drag),
            rolling_force_N=float(F_roll),
            grade_force_N=float(F_grade),
        )


def compute_feedforward_action(
    target_accel: float,
    speed: float,
    params: ExtendedPlantParams,
    grade_rad: float | None = None,
) -> InverseFeedforwardResult:
    """Convenience function wrapper for one-shot inverse feedforward queries."""
    return AnalyticInverseFeedforward(params).compute_action(
        target_accel=target_accel,
        speed=speed,
        grade_rad=grade_rad,
    )


def _equivalent_inertia(params: ExtendedPlantParams) -> float:
    motor = params.motor
    wheel = params.wheel
    body = params.body
    N = max(motor.gear_ratio, 1e-9)
    return motor.J + (wheel.inertia + body.mass * wheel.radius ** 2) / (N ** 2)


def _current_command_upper_bound(params: ExtendedPlantParams) -> float:
    motor = params.motor
    # Inverse mapping span must follow throttle->current map only.
    # Do not apply power limit here; forward plant handles P_max dynamically.
    i_upper = motor.V_max / max(motor.R, 1e-9)
    if motor.T_max is not None:
        i_upper = min(i_upper, motor.T_max / max(motor.K_t, 1e-9))
    return max(i_upper, 0.0)


def _invert_throttle_from_current(i_req: float, params: ExtendedPlantParams) -> float:
    motor = params.motor
    i_floor = max(motor.min_current_A, 0.0)
    i_upper = _current_command_upper_bound(params)
    i_span = max(i_upper - i_floor, 0.0)

    if i_span <= 1e-12:
        return 0.0

    gamma = max(motor.gamma_throttle, 1e-9)
    normalized = (i_req - i_floor) / i_span
    if normalized <= 0.0:
        return 0.0
    return normalized ** (1.0 / gamma)


def _resistive_forces(v: float, grade_rad: float, params: ExtendedPlantParams) -> tuple[float, float, float]:
    body = params.body
    F_drag = 0.5 * body.air_density * body.drag_area * v * abs(v)
    roll_factor = min(1.0, abs(v) / ROLLING_SPEED_THRESHOLD_MPS)
    F_roll = body.rolling_coeff * body.mass * GRAVITY * roll_factor
    F_grade = body.mass * GRAVITY * math.sin(grade_rad)
    return F_drag, F_roll, F_grade


__all__ = [
    "InverseFeedforwardResult",
    "AnalyticInverseFeedforward",
    "compute_feedforward_action",
]
