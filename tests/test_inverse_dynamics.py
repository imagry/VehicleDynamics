"""Unit tests for analytic inverse feedforward mapping."""

from __future__ import annotations

import math

from simulation.dynamics import (
    BodyParams,
    BrakeParams,
    ExtendedPlant,
    ExtendedPlantParams,
    MotorParams,
    WheelParams,
)
from simulation.inverse_dynamics import AnalyticInverseFeedforward


def _make_simple_params(
    *,
    p_max: float | None = None,
    throttle_tau: float = 1e-4,
    brake_tau: float = 1e-4,
) -> ExtendedPlantParams:
    motor = MotorParams(
        R=1.0,
        K_e=10.0,
        K_t=10.0,
        b=0.0,
        J=0.0,
        V_max=100.0,
        T_max=200.0,
        P_max=p_max,
        gamma_throttle=1.0,
        throttle_tau=throttle_tau,
        min_current_A=2.0,
        gear_ratio=1.0,
        eta_gb=1.0,
    )
    brake = BrakeParams(
        T_br_max=100.0,
        p_br=2.0,
        tau_br=brake_tau,
        mu=0.9,
    )
    body = BodyParams(
        mass=10.0,
        drag_area=0.0,
        rolling_coeff=0.0,
        grade_rad=0.0,
    )
    wheel = WheelParams(
        radius=1.0,
        inertia=0.0,
        v_eps=0.1,
    )
    return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel)


def test_drive_branch_closed_form_mapping() -> None:
    params = _make_simple_params()
    inv = AnalyticInverseFeedforward(params)

    result = inv.compute_action(target_accel=3.0, speed=0.0)

    # Closed-form expected values for the chosen parameters:
    # J_eq = 10, i_req = (J_eq * a_target) / K_t = (10 * 3) / 10 = 3 A
    # i_floor = 2 A, i_upper = min(V/R, T_max/K_t) = min(100, 20) = 20 A
    # u = (i_req - i_floor) / (i_upper - i_floor) = 1/18
    assert result.mode == "drive"
    assert math.isclose(result.required_motor_current_A, 3.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result.raw_action, 1.0 / 18.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result.action, 1.0 / 18.0, rel_tol=0.0, abs_tol=1e-9)
    assert result.was_clipped is False


def test_brake_branch_closed_form_mapping() -> None:
    params = _make_simple_params()
    inv = AnalyticInverseFeedforward(params)

    result = inv.compute_action(target_accel=0.0, speed=0.0)

    # With i_floor=2 A, K_t=10 Nm/A, required brake torque is 20 Nm.
    # u_br = sqrt(20 / 100) = sqrt(0.2)
    expected_u_br = math.sqrt(0.2)
    assert result.mode == "brake"
    assert math.isclose(result.required_brake_torque_Nm, 20.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result.raw_action, -expected_u_br, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result.action, -expected_u_br, rel_tol=0.0, abs_tol=1e-9)
    assert result.was_clipped is False


def test_clip_only_behavior_for_out_of_range_targets() -> None:
    params = _make_simple_params()
    inv = AnalyticInverseFeedforward(params)

    result_high = inv.compute_action(target_accel=100.0, speed=0.0)
    result_low = inv.compute_action(target_accel=-100.0, speed=0.0)

    assert result_high.raw_action > 1.0
    assert result_high.action == 1.0
    assert result_high.was_clipped is True

    assert result_low.raw_action < -1.0
    assert result_low.action == -1.0
    assert result_low.was_clipped is True


def test_round_trip_drive_and_brake_nominal_regime() -> None:
    params = _make_simple_params(brake_tau=0.02)
    inv = AnalyticInverseFeedforward(params)
    plant = ExtendedPlant(params)

    target_drive = 3.0
    drive_action = inv.compute_action(target_accel=target_drive, speed=0.0)
    plant.reset(speed=0.0)
    for _ in range(10):
        drive_state = plant.step(drive_action.action, dt=0.05, substeps=5)
    assert abs(drive_state.acceleration - target_drive) < 0.15

    # Keep deceleration moderate to stay away from zero-speed brake-hold behavior.
    target_brake = -0.2
    brake_action = inv.compute_action(target_accel=target_brake, speed=5.0)
    plant.reset(speed=5.0)
    for _ in range(10):
        brake_state = plant.step(brake_action.action, dt=0.05, substeps=5)
    assert abs(brake_state.acceleration - target_brake) < 0.2


def test_power_limit_parameter_changes_mapping_span() -> None:
    params_no_pmax = _make_simple_params(p_max=None)
    params_low_pmax = _make_simple_params(p_max=500.0)

    inv_no_pmax = AnalyticInverseFeedforward(params_no_pmax)
    inv_low_pmax = AnalyticInverseFeedforward(params_low_pmax)

    target_accel = 3.0
    raw_no_pmax = inv_no_pmax.compute_action(target_accel=target_accel, speed=0.0).raw_action
    raw_low_pmax = inv_low_pmax.compute_action(target_accel=target_accel, speed=0.0).raw_action

    # Lower P_max reduces mapping span, so same required current maps to a larger command.
    assert raw_low_pmax > raw_no_pmax
