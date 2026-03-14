"""Tests for feedforward comparison GUI logic helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

from fitting.feedforward_gui import (
    _default_vehicle_values,
    build_extended_params_from_values,
    build_gt_signed_action,
    compute_closed_loop_metrics,
    compute_open_loop_metrics,
    filter_gt_acceleration,
    run_closed_loop_ff_comparison,
    run_open_loop_ff_comparison,
)
from fitting.fitter import FitterConfig


def _vehicle_values() -> dict[str, float | None]:
    return {
        "mass": 1800.0,
        "drag_area": 0.65,
        "rolling_coeff": 0.011,
        "motor_V_max": 360.0,
        "motor_R": 0.2,
        "motor_K": 0.2,
        "motor_b": 1e-3,
        "motor_J": 1e-3,
        "motor_gamma_throttle": 1.0,
        "motor_throttle_tau": 0.1,
        "motor_min_current_A": 0.0,
        "motor_T_max": 250.0,
        "motor_P_max": 120000.0,
        "gear_ratio": 9.5,
        "eta_gb": 0.92,
        "brake_T_max": 14000.0,
        "brake_tau": 0.08,
        "brake_p": 1.2,
        "mu": 0.9,
        "wheel_radius": 0.33,
        "wheel_inertia": 1.5,
    }


def _trip(n: int = 20) -> dict[str, np.ndarray]:
    dt = 0.1
    t = np.arange(n, dtype=np.float64) * dt
    speed = 10.0 + 0.5 * np.sin(0.5 * t)
    accel = np.gradient(speed, dt)
    throttle = np.full(n, 25.0, dtype=np.float64)
    brake = np.zeros(n, dtype=np.float64)
    angle = 0.01 * np.sin(0.2 * t)
    return {
        "speed": speed,
        "acceleration": accel,
        "throttle": throttle,
        "brake": brake,
        "angle": angle,
        "time": t,
    }


def test_build_gt_signed_action_brake_dominant() -> None:
    throttle = np.array([50.0, 10.0, 0.0], dtype=np.float64)
    brake = np.array([0.0, 30.0, 40.0], dtype=np.float64)
    action = build_gt_signed_action(throttle, brake)
    np.testing.assert_allclose(action, np.array([0.5, -0.3, -0.4]))


def test_build_extended_params_from_values_optional_limits() -> None:
    values = _vehicle_values()
    values["motor_T_max"] = 0.0
    values["motor_P_max"] = None
    params = build_extended_params_from_values(values)
    assert params.motor.T_max is None
    assert params.motor.P_max is None


def test_default_vehicle_values_match_fitter_config(tmp_path: Path) -> None:
    defaults = _default_vehicle_values(settings_path=tmp_path / "missing_gui_settings.json")
    cfg = FitterConfig()

    assert defaults["mass"] == cfg.mass_init
    assert defaults["drag_area"] == cfg.drag_area_init
    assert defaults["rolling_coeff"] == cfg.rolling_coeff_init
    assert defaults["motor_V_max"] == cfg.motor_V_max_init
    assert defaults["motor_T_max"] == cfg.motor_T_max_init
    assert defaults["motor_P_max"] == cfg.motor_P_max_init
    assert defaults["wheel_radius"] == cfg.wheel_radius_init


def test_default_vehicle_values_from_saved_fitting_gui_settings(tmp_path: Path) -> None:
    settings_path = tmp_path / "gui_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "parameters": {
                    "mass": {"init": "1999.5", "min": "1800", "max": "2300"},
                    "wheel_radius": {"init": "0.334", "min": "0.315", "max": "0.34"},
                    "motor_T_max": {"init": "0", "min": "0", "max": "1000"},
                }
            }
        ),
        encoding="utf-8",
    )

    defaults = _default_vehicle_values(settings_path=settings_path)
    assert defaults["mass"] == 1999.5
    assert defaults["wheel_radius"] == 0.334
    assert defaults["motor_T_max"] is None


def test_run_open_loop_ff_comparison_shapes() -> None:
    params = build_extended_params_from_values(_vehicle_values())
    result = run_open_loop_ff_comparison(
        params,
        _trip(30),
        dt=0.1,
        use_trip_grade=True,
        accel_filter_cutoff_hz=2.0,
    )

    n = result["time"].size
    assert n == 30
    for key in (
        "speed",
        "gt_speed",
        "gt_accel_raw",
        "gt_accel_filtered",
        "target_accel",
        "gt_throttle_pct",
        "gt_brake_pct",
        "gt_action",
        "ff_raw_action",
        "ff_action",
        "ff_throttle_pct",
        "ff_brake_pct",
    ):
        assert result[key].size == n

    metrics = compute_open_loop_metrics(result)
    assert int(metrics["samples"]) == n
    assert metrics["clip_ratio_pct"] >= 0.0


def test_run_closed_loop_ff_comparison_shapes() -> None:
    params = build_extended_params_from_values(_vehicle_values())
    result = run_closed_loop_ff_comparison(
        params,
        _trip(25),
        dt=0.1,
        substeps=2,
        use_trip_grade=True,
        accel_filter_cutoff_hz=2.0,
    )

    n = result["time"].size
    assert n == 25
    for key in (
        "gt_speed",
        "gt_accel_raw",
        "gt_accel_filtered",
        "gt_acceleration",
        "gt_action",
        "ff_raw_action",
        "ff_action",
        "ff_throttle_pct",
        "ff_brake_pct",
        "sim_speed",
        "sim_acceleration",
    ):
        assert result[key].size == n

    metrics = compute_closed_loop_metrics(result)
    assert int(metrics["samples"]) == n
    assert metrics["speed_rmse"] >= 0.0


def test_filter_gt_acceleration_matches_fitter_butterworth() -> None:
    dt = 0.1
    t = np.arange(200, dtype=np.float64) * dt
    accel = np.sin(2.0 * np.pi * 0.5 * t) + 0.4 * np.sin(2.0 * np.pi * 6.0 * t)

    filtered_off = filter_gt_acceleration(accel, dt=dt, cutoff_hz=0.0)
    np.testing.assert_allclose(filtered_off, accel)

    cutoff = 2.0
    nyquist = 0.5 / dt
    b, a = signal.butter(2, cutoff / nyquist, btype="low")
    expected = signal.filtfilt(b, a, accel)
    filtered = filter_gt_acceleration(accel, dt=dt, cutoff_hz=cutoff)
    np.testing.assert_allclose(filtered, expected)
