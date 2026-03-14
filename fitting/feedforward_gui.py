"""GUI for comparing feedforward actions in open-loop and closed-loop modes."""

from __future__ import annotations

import json
import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from fitting.fitter import FittedVehicleParams, FitterConfig, VehicleParamFitter
from simulation import (
    BodyParams,
    BrakeParams,
    ExtendedPlant,
    ExtendedPlantParams,
    FeedforwardController,
    MotorParams,
    WheelParams,
)

# Prefer Tk backend for interactive GUI, but keep import-safe fallback for headless testing.
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

LOGGER = logging.getLogger(__name__)

FITTING_GUI_SETTINGS_FILE = Path(__file__).parent / "gui_settings.json"
# Keep startup deterministic: by default we use fitting-GUI defaults, not last session values.
LOAD_SAVED_VEHICLE_VALUES_ON_STARTUP_DEFAULT = False

VEHICLE_PARAM_GROUPS: dict[str, list[str]] = {
    "Body": ["mass", "drag_area", "rolling_coeff"],
    "Motor": [
        "motor_V_max",
        "motor_R",
        "motor_K",
        "motor_b",
        "motor_J",
        "motor_gamma_throttle",
        "motor_throttle_tau",
        "motor_min_current_A",
        "motor_T_max",
        "motor_P_max",
    ],
    "Drivetrain": ["gear_ratio", "eta_gb"],
    "Brake": ["brake_T_max", "brake_tau", "brake_p", "mu"],
    "Wheel": ["wheel_radius", "wheel_inertia"],
}

VEHICLE_PARAM_DISPLAY: dict[str, tuple[str, str, bool]] = {
    "mass": ("Mass", "kg", False),
    "drag_area": ("Drag Area", "m^2", False),
    "rolling_coeff": ("Rolling Coefficient", "", False),
    "motor_V_max": ("Motor V_max", "V", False),
    "motor_R": ("Motor R", "ohm", False),
    "motor_K": ("Motor K", "Nm/A", False),
    "motor_b": ("Motor b", "Nm*s/rad", False),
    "motor_J": ("Motor J", "kg*m^2", False),
    "motor_gamma_throttle": ("Throttle gamma", "", False),
    "motor_throttle_tau": ("Throttle tau", "s", False),
    "motor_min_current_A": ("Motor I_min", "A", False),
    "motor_T_max": ("Motor T_max", "Nm", True),
    "motor_P_max": ("Motor P_max", "W", True),
    "gear_ratio": ("Gear Ratio", "", False),
    "eta_gb": ("Gearbox Efficiency", "", False),
    "brake_T_max": ("Brake T_max", "Nm", False),
    "brake_tau": ("Brake tau", "s", False),
    "brake_p": ("Brake p", "", False),
    "mu": ("Friction mu", "", False),
    "wheel_radius": ("Wheel Radius", "m", False),
    "wheel_inertia": ("Wheel Inertia", "kg*m^2", False),
}


def _default_vehicle_values(settings_path: Optional[Path] = None) -> dict[str, float | None]:
    """Use fitting GUI saved defaults when available, otherwise FitterConfig init values."""
    cfg = FitterConfig()
    defaults: dict[str, float | None] = {
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
        "motor_min_current_A": cfg.motor_min_current_A_init,
        "motor_T_max": cfg.motor_T_max_init,
        "motor_P_max": cfg.motor_P_max_init,
        "gear_ratio": cfg.gear_ratio_init,
        "eta_gb": cfg.eta_gb_init,
        "brake_T_max": cfg.brake_T_max_init,
        "brake_tau": cfg.brake_tau_init,
        "brake_p": cfg.brake_p_init,
        "mu": cfg.mu_init,
        "wheel_radius": cfg.wheel_radius_init,
        "wheel_inertia": cfg.wheel_inertia_init,
    }

    source_path = settings_path if settings_path is not None else FITTING_GUI_SETTINGS_FILE
    if not source_path.exists():
        return defaults

    try:
        with open(source_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
    except Exception:
        LOGGER.exception("Failed reading fitting GUI settings from %s", source_path)
        return defaults

    parameters = settings.get("parameters")
    if not isinstance(parameters, dict):
        return defaults

    merged = defaults.copy()
    for key in defaults.keys():
        raw_entry = parameters.get(key)
        if not isinstance(raw_entry, dict):
            continue
        raw_init = raw_entry.get("init")
        if raw_init is None:
            continue

        if isinstance(raw_init, str):
            text = raw_init.strip()
            if text == "":
                parsed = None
            else:
                try:
                    parsed = float(text)
                except ValueError:
                    continue
        else:
            try:
                parsed = float(raw_init)
            except (TypeError, ValueError):
                continue

        if not np.isfinite(parsed):
            continue

        if key in {"motor_T_max", "motor_P_max"}:
            merged[key] = _coerce_optional_limit(parsed)
        else:
            merged[key] = parsed

    return merged


def _coerce_optional_limit(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if float(value) > 0.0 else None


def build_extended_params_from_values(values: dict[str, float | None]) -> ExtendedPlantParams:
    """Build plant parameters from GUI vehicle values."""
    motor = MotorParams(
        R=float(values["motor_R"]),
        K_e=float(values["motor_K"]),
        K_t=float(values["motor_K"]),
        b=float(values["motor_b"]),
        J=float(values["motor_J"]),
        V_max=float(values["motor_V_max"]),
        T_max=_coerce_optional_limit(values["motor_T_max"]),
        P_max=_coerce_optional_limit(values["motor_P_max"]),
        gamma_throttle=float(values["motor_gamma_throttle"]),
        throttle_tau=float(values["motor_throttle_tau"]),
        min_current_A=max(float(values["motor_min_current_A"]), 0.0),
        gear_ratio=float(values["gear_ratio"]),
        eta_gb=float(values["eta_gb"]),
    )
    brake = BrakeParams(
        T_br_max=float(values["brake_T_max"]),
        p_br=float(values["brake_p"]),
        tau_br=float(values["brake_tau"]),
        mu=float(values["mu"]),
    )
    body = BodyParams(
        mass=float(values["mass"]),
        drag_area=float(values["drag_area"]),
        rolling_coeff=float(values["rolling_coeff"]),
        grade_rad=0.0,
    )
    wheel = WheelParams(
        radius=float(values["wheel_radius"]),
        inertia=float(values["wheel_inertia"]),
    )
    return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel)


def build_gt_signed_action(throttle_pct: np.ndarray, brake_pct: np.ndarray) -> np.ndarray:
    """Convert GT throttle/brake percentages to signed action in [-1, 1]."""
    throttle = np.clip(np.asarray(throttle_pct, dtype=np.float64) / 100.0, 0.0, 1.0)
    brake = np.clip(np.asarray(brake_pct, dtype=np.float64) / 100.0, 0.0, 1.0)
    brake_active = brake > 0.0
    return np.where(brake_active, -brake, throttle)


def _extract_trip_arrays(trip: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    speed = np.asarray(trip["speed"], dtype=np.float64)
    accel = np.asarray(trip.get("acceleration", np.gradient(speed, dt)), dtype=np.float64)
    throttle = np.asarray(trip.get("throttle", np.zeros_like(speed)), dtype=np.float64)
    brake = np.asarray(trip.get("brake", np.zeros_like(speed)), dtype=np.float64)
    grade = np.asarray(trip.get("angle", np.zeros_like(speed)), dtype=np.float64)

    n = min(speed.size, accel.size, throttle.size, brake.size, grade.size)
    if n < 2:
        raise ValueError("Trip must have at least 2 synchronized samples")

    speed = speed[:n]
    accel = accel[:n]
    throttle = throttle[:n]
    brake = brake[:n]
    grade = grade[:n]

    time_raw = trip.get("time")
    if time_raw is not None:
        time = np.asarray(time_raw, dtype=np.float64)
        if time.size < n:
            time = np.arange(n, dtype=np.float64) * float(dt)
        else:
            time = time[:n]
    else:
        time = np.arange(n, dtype=np.float64) * float(dt)

    return {
        "time": time,
        "speed": speed,
        "acceleration": accel,
        "throttle": throttle,
        "brake": brake,
        "grade": grade,
    }


def filter_gt_acceleration(accel: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    """Apply the same Butterworth low-pass filter used by the fitter."""
    arr = np.asarray(accel, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()
    if arr.size <= 3:
        return arr.copy()

    try:
        cutoff = float(cutoff_hz)
        dt_val = float(dt)
    except (TypeError, ValueError):
        return arr.copy()

    if cutoff <= 0.0 or dt_val <= 0.0:
        return arr.copy()

    nyquist = 0.5 / dt_val
    if nyquist <= 0.0:
        return arr.copy()

    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0:
        return arr.copy()

    try:
        from scipy import signal

        b, a = signal.butter(2, normal_cutoff, btype="low")
        return signal.filtfilt(b, a, arr)
    except Exception:
        return arr.copy()


def _simulate_motor_state_profile(
    params: ExtendedPlantParams,
    action_profile: np.ndarray,
    initial_speed: float,
    dt: float,
    grade_profile: np.ndarray,
    substeps: int,
) -> dict[str, np.ndarray | float | None]:
    """Roll out plant state for a given action profile and collect full-state signals."""
    actions = np.asarray(action_profile, dtype=np.float64)
    grades = np.asarray(grade_profile, dtype=np.float64)
    n = actions.size
    if grades.size != n:
        raise ValueError("grade_profile must match action_profile length")

    motor_current = np.zeros(n, dtype=np.float64)
    motor_voltage = np.zeros(n, dtype=np.float64)
    back_emf_voltage = np.zeros(n, dtype=np.float64)
    motor_power = np.zeros(n, dtype=np.float64)
    motor_i_limit = np.zeros(n, dtype=np.float64)
    speed = np.zeros(n, dtype=np.float64)

    tire_force = np.zeros(n, dtype=np.float64)
    drag_force = np.zeros(n, dtype=np.float64)
    rolling_force = np.zeros(n, dtype=np.float64)
    grade_force = np.zeros(n, dtype=np.float64)
    net_force = np.zeros(n, dtype=np.float64)

    plant = ExtendedPlant(params)
    plant.reset(speed=float(initial_speed), position=0.0)

    for i in range(n):
        state = plant.step(
            action=float(actions[i]),
            dt=float(dt),
            substeps=max(int(substeps), 1),
            grade_rad=float(grades[i]),
        )
        motor_current[i] = state.motor_current
        motor_voltage[i] = state.V_cmd
        back_emf_voltage[i] = state.back_emf_voltage
        motor_power[i] = state.V_cmd * state.motor_current
        motor_i_limit[i] = state.i_limit
        speed[i] = state.speed

        tire_force[i] = state.tire_force
        drag_force[i] = state.drag_force
        rolling_force[i] = state.rolling_force
        grade_force[i] = state.grade_force
        net_force[i] = state.net_force

    k_t = max(float(params.motor.K_t), 1e-9)
    if params.motor.T_max is not None:
        i_max = float(params.motor.T_max) / k_t
    else:
        i_max = float(params.motor.V_max) / max(float(params.motor.R), 1e-9)

    p_max = None if params.motor.P_max is None else float(params.motor.P_max)

    tractive_power = tire_force * speed
    drag_power = drag_force * np.abs(speed)
    rolling_power = rolling_force * np.abs(speed)
    grade_power = grade_force * speed
    net_power = net_force * speed

    return {
        "speed": speed,
        "motor_current": motor_current,
        "motor_voltage": motor_voltage,
        "back_emf_voltage": back_emf_voltage,
        "motor_power": motor_power,
        "motor_i_limit": motor_i_limit,
        "motor_i_max": float(i_max),
        "motor_p_max": p_max,
        "tire_force": tire_force,
        "drag_force": drag_force,
        "rolling_force": rolling_force,
        "grade_force": grade_force,
        "net_force": net_force,
        "tractive_power": tractive_power,
        "drag_power": drag_power,
        "rolling_power": rolling_power,
        "grade_power": grade_power,
        "net_power": net_power,
    }


def _apply_action_gains(action: np.ndarray, throttle_gain: float, brake_gain: float) -> np.ndarray:
    """Apply separate gains to drive (>=0) and brake (<0) actions."""
    arr = np.asarray(action, dtype=np.float64)
    return np.where(arr >= 0.0, arr * float(throttle_gain), arr * float(brake_gain))


def run_open_loop_ff_comparison(
    params: ExtendedPlantParams,
    trip: dict[str, np.ndarray],
    dt: float,
    use_trip_grade: bool = True,
    accel_filter_cutoff_hz: float = 2.0,
    substeps: int = 1,
    throttle_gain: float = 1.0,
    brake_gain: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute feedforward actions on GT state traces without simulation."""
    arrays = _extract_trip_arrays(trip, dt)
    controller = FeedforwardController(params)

    grade = arrays["grade"] if use_trip_grade else np.zeros_like(arrays["grade"])
    accel_filtered = filter_gt_acceleration(
        arrays["acceleration"],
        dt=dt,
        cutoff_hz=accel_filter_cutoff_hz,
    )
    profile = controller.compute_action_profile(
        target_accel_profile=accel_filtered,
        speed_profile=arrays["speed"],
        grade_profile=grade,
    )

    ff_raw_action = _apply_action_gains(profile.raw_action, throttle_gain=throttle_gain, brake_gain=brake_gain)
    ff_action = np.clip(
        _apply_action_gains(profile.action, throttle_gain=throttle_gain, brake_gain=brake_gain),
        -1.0,
        1.0,
    )

    gt_action = build_gt_signed_action(arrays["throttle"], arrays["brake"])
    ff_throttle_pct = np.clip(ff_action, 0.0, 1.0) * 100.0
    ff_brake_pct = np.clip(-ff_action, 0.0, 1.0) * 100.0

    motor_state = _simulate_motor_state_profile(
        params=params,
        action_profile=ff_action,
        initial_speed=float(arrays["speed"][0]),
        dt=float(dt),
        grade_profile=grade,
        substeps=max(int(substeps), 1),
    )

    return {
        "time": arrays["time"],
        "speed": arrays["speed"],
        "gt_speed": arrays["speed"],
        "gt_accel_raw": arrays["acceleration"],
        "gt_accel_filtered": accel_filtered,
        "target_accel": accel_filtered,
        "grade": grade,
        "gt_throttle_pct": arrays["throttle"],
        "gt_brake_pct": arrays["brake"],
        "gt_action": gt_action,
        "ff_raw_action": ff_raw_action,
        "ff_action": ff_action,
        "ff_throttle_pct": ff_throttle_pct,
        "ff_brake_pct": ff_brake_pct,
        "ff_motor_current_A": motor_state["motor_current"],
        "ff_motor_voltage_V": motor_state["motor_voltage"],
        "ff_back_emf_voltage_V": motor_state["back_emf_voltage"],
        "ff_motor_power_W": motor_state["motor_power"],
        "ff_motor_i_limit_A": motor_state["motor_i_limit"],
        "ff_motor_i_max_A": np.full_like(arrays["time"], float(motor_state["motor_i_max"]), dtype=np.float64),
        "ff_tire_force_N": motor_state["tire_force"],
        "ff_drag_force_N": motor_state["drag_force"],
        "ff_rolling_force_N": motor_state["rolling_force"],
        "ff_grade_force_N": motor_state["grade_force"],
        "ff_net_force_N": motor_state["net_force"],
        "ff_tractive_power_W": motor_state["tractive_power"],
        "ff_drag_power_W": motor_state["drag_power"],
        "ff_rolling_power_W": motor_state["rolling_power"],
        "ff_grade_power_W": motor_state["grade_power"],
        "ff_net_power_W": motor_state["net_power"],
        "ff_motor_p_max_W": (
            np.full_like(arrays["time"], float(motor_state["motor_p_max"]), dtype=np.float64)
            if motor_state["motor_p_max"] is not None
            else np.full_like(arrays["time"], np.nan, dtype=np.float64)
        ),
    }


def run_closed_loop_ff_comparison(
    params: ExtendedPlantParams,
    trip: dict[str, np.ndarray],
    dt: float,
    substeps: int,
    use_trip_grade: bool = True,
    accel_filter_cutoff_hz: float = 2.0,
    throttle_gain: float = 1.0,
    brake_gain: float = 1.0,
) -> dict[str, np.ndarray]:
    """Roll out feedforward actions in the plant and compare against GT traces."""
    arrays = _extract_trip_arrays(trip, dt)
    controller = FeedforwardController(params)

    grade = arrays["grade"] if use_trip_grade else np.zeros_like(arrays["grade"])
    accel_filtered = filter_gt_acceleration(
        arrays["acceleration"],
        dt=dt,
        cutoff_hz=accel_filter_cutoff_hz,
    )
    n = accel_filtered.size
    ff_raw_action = np.zeros(n, dtype=np.float64)
    ff_action = np.zeros(n, dtype=np.float64)
    sim_speed = np.zeros(n, dtype=np.float64)
    sim_acceleration = np.zeros(n, dtype=np.float64)
    mode: list[str] = []

    motor_current = np.zeros(n, dtype=np.float64)
    motor_voltage = np.zeros(n, dtype=np.float64)
    back_emf_voltage = np.zeros(n, dtype=np.float64)
    motor_power = np.zeros(n, dtype=np.float64)
    motor_i_limit = np.zeros(n, dtype=np.float64)

    tire_force = np.zeros(n, dtype=np.float64)
    drag_force = np.zeros(n, dtype=np.float64)
    rolling_force = np.zeros(n, dtype=np.float64)
    grade_force = np.zeros(n, dtype=np.float64)
    net_force = np.zeros(n, dtype=np.float64)

    plant = ExtendedPlant(params)
    plant.reset(speed=float(arrays["speed"][0]), position=0.0)
    rollout_substeps = max(int(substeps), 1)

    for k in range(n):
        ff = controller.inverse.compute_action(
            target_accel=float(accel_filtered[k]),
            speed=float(plant.speed),
            grade_rad=float(grade[k]),
        )

        raw_adj = _apply_action_gains(
            np.asarray([ff.raw_action], dtype=np.float64),
            throttle_gain=throttle_gain,
            brake_gain=brake_gain,
        )[0]
        action_adj = _apply_action_gains(
            np.asarray([ff.action], dtype=np.float64),
            throttle_gain=throttle_gain,
            brake_gain=brake_gain,
        )[0]
        action_adj = float(np.clip(action_adj, -1.0, 1.0))

        state = plant.step(
            action=action_adj,
            dt=float(dt),
            substeps=rollout_substeps,
            grade_rad=float(grade[k]),
        )

        ff_raw_action[k] = float(raw_adj)
        ff_action[k] = action_adj
        sim_speed[k] = state.speed
        sim_acceleration[k] = state.acceleration
        motor_current[k] = state.motor_current
        motor_voltage[k] = state.V_cmd
        back_emf_voltage[k] = state.back_emf_voltage
        motor_power[k] = state.V_cmd * state.motor_current
        motor_i_limit[k] = state.i_limit
        tire_force[k] = state.tire_force
        drag_force[k] = state.drag_force
        rolling_force[k] = state.rolling_force
        grade_force[k] = state.grade_force
        net_force[k] = state.net_force
        mode.append(ff.mode)

    k_t = max(float(params.motor.K_t), 1e-9)
    if params.motor.T_max is not None:
        i_max = float(params.motor.T_max) / k_t
    else:
        i_max = float(params.motor.V_max) / max(float(params.motor.R), 1e-9)
    p_max = None if params.motor.P_max is None else float(params.motor.P_max)

    tractive_power = tire_force * sim_speed
    drag_power = drag_force * np.abs(sim_speed)
    rolling_power = rolling_force * np.abs(sim_speed)
    grade_power = grade_force * sim_speed
    net_power = net_force * sim_speed

    gt_action = build_gt_signed_action(arrays["throttle"], arrays["brake"])
    ff_throttle_pct = np.clip(ff_action, 0.0, 1.0) * 100.0
    ff_brake_pct = np.clip(-ff_action, 0.0, 1.0) * 100.0

    return {
        "time": arrays["time"],
        "gt_speed": arrays["speed"],
        "gt_accel_raw": arrays["acceleration"],
        "gt_accel_filtered": accel_filtered,
        "gt_acceleration": accel_filtered,
        "grade": grade,
        "gt_throttle_pct": arrays["throttle"],
        "gt_brake_pct": arrays["brake"],
        "gt_action": gt_action,
        "ff_raw_action": ff_raw_action,
        "ff_action": ff_action,
        "ff_throttle_pct": ff_throttle_pct,
        "ff_brake_pct": ff_brake_pct,
        "sim_speed": sim_speed,
        "sim_acceleration": sim_acceleration,
        "ff_mode": np.asarray(mode, dtype=object),
        "ff_motor_current_A": motor_current,
        "ff_motor_voltage_V": motor_voltage,
        "ff_back_emf_voltage_V": back_emf_voltage,
        "ff_motor_power_W": motor_power,
        "ff_motor_i_limit_A": motor_i_limit,
        "ff_motor_i_max_A": np.full_like(arrays["time"], float(i_max), dtype=np.float64),
        "ff_tire_force_N": tire_force,
        "ff_drag_force_N": drag_force,
        "ff_rolling_force_N": rolling_force,
        "ff_grade_force_N": grade_force,
        "ff_net_force_N": net_force,
        "ff_tractive_power_W": tractive_power,
        "ff_drag_power_W": drag_power,
        "ff_rolling_power_W": rolling_power,
        "ff_grade_power_W": grade_power,
        "ff_net_power_W": net_power,
        "ff_motor_p_max_W": (
            np.full_like(arrays["time"], float(p_max), dtype=np.float64)
            if p_max is not None
            else np.full_like(arrays["time"], np.nan, dtype=np.float64)
        ),
    }


def compute_open_loop_metrics(result: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute trip-level metrics for open-loop comparison."""
    throttle_err = result["ff_throttle_pct"] - result["gt_throttle_pct"]
    brake_err = result["ff_brake_pct"] - result["gt_brake_pct"]
    clipped = np.abs(result["ff_action"] - result["ff_raw_action"]) > 1e-9

    return {
        "samples": float(result["time"].size),
        "clip_ratio_pct": float(np.mean(clipped) * 100.0),
        "throttle_rmse_pct": float(np.sqrt(np.mean(throttle_err ** 2))),
        "throttle_mae_pct": float(np.mean(np.abs(throttle_err))),
        "brake_rmse_pct": float(np.sqrt(np.mean(brake_err ** 2))),
        "brake_mae_pct": float(np.mean(np.abs(brake_err))),
    }


def compute_closed_loop_metrics(result: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute trip-level metrics for closed-loop comparison."""
    speed_err = result["sim_speed"] - result["gt_speed"]
    accel_err = result["sim_acceleration"] - result["gt_acceleration"]
    throttle_err = result["ff_throttle_pct"] - result["gt_throttle_pct"]
    brake_err = result["ff_brake_pct"] - result["gt_brake_pct"]
    clipped = np.abs(result["ff_action"] - result["ff_raw_action"]) > 1e-9

    return {
        "samples": float(result["time"].size),
        "clip_ratio_pct": float(np.mean(clipped) * 100.0),
        "speed_rmse": float(np.sqrt(np.mean(speed_err ** 2))),
        "speed_mae": float(np.mean(np.abs(speed_err))),
        "accel_rmse": float(np.sqrt(np.mean(accel_err ** 2))),
        "accel_mae": float(np.mean(np.abs(accel_err))),
        "throttle_mae_pct": float(np.mean(np.abs(throttle_err))),
        "brake_mae_pct": float(np.mean(np.abs(brake_err))),
    }


class ParameterDialog(tk.Toplevel):
    """Popup editor for vehicle parameter values used by feedforward comparison."""

    def __init__(self, parent: tk.Misc, values: dict[str, float | None]):
        super().__init__(parent)
        self.title("Vehicle Parameters")
        self.geometry("720x720")
        self.resizable(True, True)
        self.result: Optional[dict[str, float | None]] = None

        self._default_values = _default_vehicle_values()
        self._vars: dict[str, tk.StringVar] = {}

        container = ttk.Frame(self, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(button_row, text="Load Fitted JSON", command=self._on_load_fitted).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_row, text="Reset Defaults", command=self._on_reset_defaults).pack(side=tk.LEFT, padx=4)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        form_frame = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=form_frame, anchor="nw")

        def _sync_scroll(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _resize_canvas(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        form_frame.bind("<Configure>", _sync_scroll)
        canvas.bind("<Configure>", _resize_canvas)

        self._build_form(form_frame, values)

        footer = ttk.Frame(self, padding=8)
        footer.pack(fill=tk.X)
        ttk.Button(footer, text="Apply", command=self._on_apply).pack(side=tk.RIGHT, padx=4)
        ttk.Button(footer, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=4)

        self.transient(parent)
        self.grab_set()
        self.focus_set()

    def _build_form(self, parent: ttk.Frame, values: dict[str, float | None]) -> None:
        row = 0
        for group_name, keys in VEHICLE_PARAM_GROUPS.items():
            ttk.Label(parent, text=group_name, font=("TkDefaultFont", 10, "bold")).grid(
                row=row,
                column=0,
                columnspan=3,
                sticky=tk.W,
                pady=(10, 4),
                padx=4,
            )
            row += 1

            for key in keys:
                display_name, unit, optional = VEHICLE_PARAM_DISPLAY[key]
                value = values.get(key)
                if value is None:
                    default_text = ""
                else:
                    default_text = f"{float(value):.8g}"

                ttk.Label(parent, text=f"{display_name}:").grid(row=row, column=0, sticky=tk.W, padx=4, pady=2)
                var = tk.StringVar(value=default_text)
                self._vars[key] = var
                ttk.Entry(parent, textvariable=var, width=20).grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)

                hint = unit if unit else "-"
                if optional:
                    hint = f"{hint} (empty = None)"
                ttk.Label(parent, text=hint).grid(row=row, column=2, sticky=tk.W, padx=4, pady=2)
                row += 1

    def _on_reset_defaults(self) -> None:
        for key, value in self._default_values.items():
            if value is None:
                self._vars[key].set("")
            else:
                self._vars[key].set(f"{float(value):.8g}")

    def _on_load_fitted(self) -> None:
        path = filedialog.askopenfilename(
            title="Select params JSON (fitted_params/config/checkpoint)",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return

        try:
            fitted = FittedVehicleParams.load(Path(path))
            for key in VEHICLE_PARAM_DISPLAY.keys():
                value = getattr(fitted, key)
                if value is None:
                    self._vars[key].set("")
                else:
                    self._vars[key].set(f"{float(value):.8g}")
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed loading fitted params:\n{exc}", parent=self)

    def _on_apply(self) -> None:
        parsed: dict[str, float | None] = {}
        for key, (_label, _unit, optional) in VEHICLE_PARAM_DISPLAY.items():
            raw = self._vars[key].get().strip()
            if optional and raw == "":
                parsed[key] = None
                continue
            try:
                value = float(raw)
            except ValueError:
                messagebox.showerror("Validation Error", f"Invalid value for '{key}'", parent=self)
                return

            if optional and value <= 0.0:
                parsed[key] = None
            else:
                parsed[key] = value

        self.result = parsed
        self.destroy()


class FeedforwardComparisonGUI:
    """GUI for open-loop and closed-loop feedforward trip comparisons."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Feedforward Comparison GUI")
        self.root.geometry("1450x900")

        self.settings_file = Path(__file__).parent / "feedforward_gui_settings.json"
        self.data_dir = Path(__file__).parent.parent / "data" / "processed"

        self.mode_var = tk.StringVar(value="open_loop")
        self.dataset_var = tk.StringVar(value="")
        self.trip_var = tk.StringVar(value="")
        self.use_trip_grade_var = tk.BooleanVar(value=True)
        self.substeps_var = tk.StringVar(value="1")
        self.accel_filter_cutoff_hz_var = tk.DoubleVar(value=2.0)
        self.accel_filter_label_var = tk.StringVar(value="2.0 Hz")
        self.throttle_gain_var = tk.DoubleVar(value=1.0)
        self.brake_gain_var = tk.DoubleVar(value=1.0)
        self.throttle_gain_label_var = tk.StringVar(value="1.00")
        self.brake_gain_label_var = tk.StringVar(value="1.00")
        self.status_var = tk.StringVar(value="Select dataset and trip, then run comparison")
        self.metrics_var = tk.StringVar(value="No run yet")

        self.vehicle_values: dict[str, float | None] = _default_vehicle_values()

        self._fitter: Optional[VehicleParamFitter] = None
        self._trips: dict[str, dict[str, np.ndarray]] = {}
        self._dt: float = 0.1

        self._create_widgets()
        self._populate_datasets()
        self._load_settings()
        self._on_accel_filter_slider(None)

    def _create_widgets(self) -> None:
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left = ttk.Frame(main_paned, padding=8)
        right = ttk.Frame(main_paned, padding=8)
        main_paned.add(left, weight=1)
        main_paned.add(right, weight=2)

        cfg = ttk.LabelFrame(left, text="Configuration", padding=8)
        cfg.pack(fill=tk.BOTH, expand=True)

        ttk.Label(cfg, text="Dataset:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.dataset_combo = ttk.Combobox(cfg, textvariable=self.dataset_var, state="readonly", width=48)
        self.dataset_combo.grid(row=0, column=1, columnspan=3, sticky=tk.EW, pady=4, padx=4)
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_selected_dataset())
        ttk.Button(cfg, text="Refresh", command=self._populate_datasets).grid(row=0, column=4, sticky=tk.E, padx=4)

        ttk.Label(cfg, text="Trip:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.trip_combo = ttk.Combobox(cfg, textvariable=self.trip_var, state="readonly", width=48)
        self.trip_combo.grid(row=1, column=1, columnspan=4, sticky=tk.EW, pady=4, padx=4)

        ttk.Label(cfg, text="Mode:").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(cfg, text="Open loop (no simulation)", value="open_loop", variable=self.mode_var).grid(
            row=2,
            column=1,
            columnspan=2,
            sticky=tk.W,
            padx=4,
        )
        ttk.Radiobutton(cfg, text="Closed loop simulation", value="closed_loop", variable=self.mode_var).grid(
            row=2,
            column=3,
            columnspan=2,
            sticky=tk.W,
            padx=4,
        )

        ttk.Checkbutton(cfg, text="Use trip grade", variable=self.use_trip_grade_var).grid(
            row=3,
            column=1,
            sticky=tk.W,
            padx=4,
            pady=4,
        )
        ttk.Label(cfg, text="Substeps (closed-loop):").grid(row=3, column=2, sticky=tk.E, pady=4)
        ttk.Entry(cfg, textvariable=self.substeps_var, width=8).grid(row=3, column=3, sticky=tk.W, padx=4)

        ttk.Label(cfg, text="GT accel LPF cutoff:").grid(row=4, column=0, sticky=tk.W, pady=4)
        accel_scale = tk.Scale(
            cfg,
            from_=0.0,
            to=10.0,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            variable=self.accel_filter_cutoff_hz_var,
            command=self._on_accel_filter_slider,
            length=220,
        )
        accel_scale.grid(row=4, column=1, columnspan=3, sticky=tk.W, padx=4)
        ttk.Label(cfg, textvariable=self.accel_filter_label_var).grid(row=4, column=4, sticky=tk.W, padx=4)

        ttk.Label(cfg, text="Throttle FF gain:").grid(row=5, column=0, sticky=tk.W, pady=4)
        throttle_gain_scale = tk.Scale(
            cfg,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            resolution=0.01,
            variable=self.throttle_gain_var,
            command=self._on_throttle_gain_slider,
            length=220,
        )
        throttle_gain_scale.grid(row=5, column=1, columnspan=3, sticky=tk.W, padx=4)
        ttk.Label(cfg, textvariable=self.throttle_gain_label_var).grid(row=5, column=4, sticky=tk.W, padx=4)

        ttk.Label(cfg, text="Brake FF gain:").grid(row=6, column=0, sticky=tk.W, pady=4)
        brake_gain_scale = tk.Scale(
            cfg,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            resolution=0.01,
            variable=self.brake_gain_var,
            command=self._on_brake_gain_slider,
            length=220,
        )
        brake_gain_scale.grid(row=6, column=1, columnspan=3, sticky=tk.W, padx=4)
        ttk.Label(cfg, textvariable=self.brake_gain_label_var).grid(row=6, column=4, sticky=tk.W, padx=4)

        ttk.Button(cfg, text="Vehicle Parameters...", command=self._open_parameter_dialog).grid(
            row=7,
            column=1,
            sticky=tk.W,
            padx=4,
            pady=(8, 4),
        )
        self.param_summary_label = ttk.Label(cfg, text="")
        self.param_summary_label.grid(row=7, column=2, columnspan=3, sticky=tk.W, padx=4, pady=(8, 4))
        self._refresh_param_summary()

        ttk.Separator(cfg, orient=tk.HORIZONTAL).grid(row=8, column=0, columnspan=5, sticky=tk.EW, pady=8)

        ttk.Button(cfg, text="Run Comparison", command=self._run_comparison).grid(
            row=9,
            column=0,
            columnspan=2,
            sticky=tk.W,
            padx=4,
            pady=4,
        )
        ttk.Button(cfg, text="Save Settings", command=self._save_settings).grid(
            row=9,
            column=2,
            sticky=tk.W,
            padx=4,
            pady=4,
        )

        ttk.Label(cfg, textvariable=self.status_var, wraplength=430, foreground="#444").grid(
            row=10,
            column=0,
            columnspan=5,
            sticky=tk.W,
            padx=4,
            pady=(8, 2),
        )

        metrics_frame = ttk.LabelFrame(cfg, text="Trip Metrics", padding=6)
        metrics_frame.grid(row=11, column=0, columnspan=5, sticky=tk.EW, padx=4, pady=(8, 4))
        ttk.Label(
            metrics_frame,
            textvariable=self.metrics_var,
            justify=tk.LEFT,
            anchor="w",
            font=("Consolas", 9),
        ).pack(fill=tk.X)

        for col in range(5):
            cfg.columnconfigure(col, weight=1 if col in {1, 2, 3} else 0)

        self.figure = Figure(figsize=(13, 10), dpi=100)
        # Left column: speed, accel, throttle, brake
        self.ax_speed = self.figure.add_subplot(4, 2, 1)
        self.ax_power = self.figure.add_subplot(4, 2, 2)
        self.ax_accel = self.figure.add_subplot(4, 2, 3, sharex=self.ax_speed)
        self.ax_current = self.figure.add_subplot(4, 2, 4, sharex=self.ax_power)
        self.ax_throttle = self.figure.add_subplot(4, 2, 5, sharex=self.ax_speed)
        self.ax_voltage = self.figure.add_subplot(4, 2, 6, sharex=self.ax_power)
        self.ax_brake = self.figure.add_subplot(4, 2, 7, sharex=self.ax_speed)
        self.ax_forces = self.figure.add_subplot(4, 2, 8, sharex=self.ax_power)

        plot_frame = ttk.Frame(right)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(right)
        toolbar_frame.pack(fill=tk.X)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)

    def _populate_datasets(self) -> None:
        datasets: list[str] = []
        if self.data_dir.exists():
            for pt_file in self.data_dir.rglob("*.pt"):
                try:
                    rel = pt_file.relative_to(self.data_dir.parent.parent)
                    datasets.append(str(rel))
                except ValueError:
                    continue

        datasets.sort()
        self.dataset_combo["values"] = datasets
        if datasets and not self.dataset_var.get():
            self.dataset_var.set(datasets[0])
            self._load_selected_dataset()
        elif not datasets:
            self.dataset_var.set("No datasets found")

    def _load_selected_dataset(self) -> None:
        dataset_rel = self.dataset_var.get().strip()
        if not dataset_rel or dataset_rel == "No datasets found":
            return

        dataset_path = Path(__file__).parent.parent / dataset_rel
        if not dataset_path.exists():
            messagebox.showerror("Dataset Error", f"Dataset not found:\n{dataset_path}")
            return

        try:
            config = FitterConfig(use_gpu=False)
            fitter = VehicleParamFitter(config)
            trips = fitter.load_trip_data(dataset_path)
            if not trips:
                raise ValueError("No trips found in selected dataset")

            self._fitter = fitter
            self._trips = trips
            self._dt = float(fitter._estimate_dt(trips))

            trip_ids = sorted(trips.keys())
            self.trip_combo["values"] = trip_ids
            if trip_ids:
                if self.trip_var.get() not in trip_ids:
                    self.trip_var.set(trip_ids[0])
            self.status_var.set(f"Loaded {len(trip_ids)} trips (dt ~ {self._dt:.4f}s)")
        except Exception as exc:
            LOGGER.exception("Failed loading dataset")
            messagebox.showerror("Dataset Error", f"Failed to load dataset:\n{exc}")

    def _open_parameter_dialog(self) -> None:
        dialog = ParameterDialog(self.root, self.vehicle_values)
        self.root.wait_window(dialog)
        if dialog.result is not None:
            self.vehicle_values = dialog.result
            self._refresh_param_summary()

    def _refresh_param_summary(self) -> None:
        mass = float(self.vehicle_values["mass"])
        vmax = float(self.vehicle_values["motor_V_max"])
        ratio = float(self.vehicle_values["gear_ratio"])
        self.param_summary_label.configure(
            text=f"mass={mass:.1f} kg, Vmax={vmax:.1f} V, gear={ratio:.3f}"
        )

    def _get_accel_filter_cutoff_hz(self) -> float:
        try:
            cutoff = max(float(self.accel_filter_cutoff_hz_var.get()), 0.0)
        except (ValueError, tk.TclError):
            cutoff = 2.0
        return cutoff

    def _on_accel_filter_slider(self, _value: object) -> None:
        cutoff = self._get_accel_filter_cutoff_hz()
        try:
            current = float(self.accel_filter_cutoff_hz_var.get())
        except (ValueError, tk.TclError):
            current = 2.0
        rounded = round(cutoff, 1)
        if not np.isclose(current, rounded):
            self.accel_filter_cutoff_hz_var.set(rounded)
        if rounded <= 0.0:
            self.accel_filter_label_var.set("off")
        else:
            self.accel_filter_label_var.set(f"{rounded:.1f} Hz")

    def _get_throttle_gain(self) -> float:
        try:
            value = float(self.throttle_gain_var.get())
        except (ValueError, tk.TclError):
            value = 1.0
        return min(max(value, 0.0), 1.0)

    def _get_brake_gain(self) -> float:
        try:
            value = float(self.brake_gain_var.get())
        except (ValueError, tk.TclError):
            value = 1.0
        return min(max(value, 0.0), 1.0)

    def _on_throttle_gain_slider(self, _value: object) -> None:
        gain = round(self._get_throttle_gain(), 2)
        try:
            current = float(self.throttle_gain_var.get())
        except (ValueError, tk.TclError):
            current = 1.0
        if not np.isclose(current, gain):
            self.throttle_gain_var.set(gain)
        self.throttle_gain_label_var.set(f"{gain:.2f}")

    def _on_brake_gain_slider(self, _value: object) -> None:
        gain = round(self._get_brake_gain(), 2)
        try:
            current = float(self.brake_gain_var.get())
        except (ValueError, tk.TclError):
            current = 1.0
        if not np.isclose(current, gain):
            self.brake_gain_var.set(gain)
        self.brake_gain_label_var.set(f"{gain:.2f}")

    def _run_comparison(self) -> None:
        if not self._trips:
            messagebox.showwarning("Run Warning", "Load a dataset first.")
            return

        trip_id = self.trip_var.get().strip()
        if trip_id not in self._trips:
            messagebox.showwarning("Run Warning", "Select a valid trip.")
            return

        try:
            substeps = max(int(self.substeps_var.get()), 1)
        except ValueError:
            messagebox.showerror("Validation Error", "Substeps must be a positive integer")
            return

        try:
            params = build_extended_params_from_values(self.vehicle_values)
            trip = self._trips[trip_id]
            use_grade = bool(self.use_trip_grade_var.get())
            accel_filter_cutoff_hz = self._get_accel_filter_cutoff_hz()
            throttle_gain = self._get_throttle_gain()
            brake_gain = self._get_brake_gain()
            filter_label = "off" if accel_filter_cutoff_hz <= 0.0 else f"{accel_filter_cutoff_hz:.1f} Hz"

            if self.mode_var.get() == "open_loop":
                result = run_open_loop_ff_comparison(
                    params=params,
                    trip=trip,
                    dt=self._dt,
                    use_trip_grade=use_grade,
                    accel_filter_cutoff_hz=accel_filter_cutoff_hz,
                    substeps=substeps,
                    throttle_gain=throttle_gain,
                    brake_gain=brake_gain,
                )
                self._plot_open_loop(result, trip_id=trip_id)
                metrics = compute_open_loop_metrics(result)
                self.metrics_var.set(self._format_metrics(metrics, mode="open_loop"))
                self.status_var.set(
                    f"Open-loop completed for {trip_id} | samples={int(metrics['samples'])} | clipped={metrics['clip_ratio_pct']:.2f}% | accel LPF={filter_label} | gains T={throttle_gain:.2f}, B={brake_gain:.2f}"
                )
            else:
                result = run_closed_loop_ff_comparison(
                    params=params,
                    trip=trip,
                    dt=self._dt,
                    substeps=substeps,
                    use_trip_grade=use_grade,
                    accel_filter_cutoff_hz=accel_filter_cutoff_hz,
                    throttle_gain=throttle_gain,
                    brake_gain=brake_gain,
                )
                self._plot_closed_loop(result, trip_id=trip_id)
                metrics = compute_closed_loop_metrics(result)
                self.metrics_var.set(self._format_metrics(metrics, mode="closed_loop"))
                self.status_var.set(
                    f"Closed-loop completed for {trip_id} | speed RMSE={metrics['speed_rmse']:.4f} m/s | accel RMSE={metrics['accel_rmse']:.4f} m/s^2 | accel LPF={filter_label} | gains T={throttle_gain:.2f}, B={brake_gain:.2f}"
                )

            self._save_settings(silent=True)
        except Exception as exc:
            LOGGER.exception("Comparison run failed")
            messagebox.showerror("Run Error", f"Comparison failed:\n{exc}")

    def _plot_open_loop(self, result: dict[str, np.ndarray], trip_id: str) -> None:
        time = result["time"]
        self.ax_speed.clear()
        self.ax_accel.clear()
        self.ax_throttle.clear()
        self.ax_brake.clear()
        self.ax_power.clear()
        self.ax_current.clear()
        self.ax_voltage.clear()
        self.ax_forces.clear()

        self.ax_speed.plot(time, result["gt_speed"], color="black", label="GT Speed")
        self.ax_speed.set_ylabel("Speed (m/s)")
        self.ax_speed.set_title(f"Open Loop | Trip {trip_id}")
        self.ax_speed.grid(True, alpha=0.3)
        self.ax_speed.legend(loc="best", fontsize=8)

        self.ax_accel.plot(time, result["gt_accel_filtered"], color="black", linestyle="--", label="GT Accel")
        self.ax_accel.plot(time, result["target_accel"], color="tab:blue", label="FF Target Accel")
        self.ax_accel.set_ylabel("Accel (m/s^2)")
        self.ax_accel.grid(True, alpha=0.3)
        self.ax_accel.legend(loc="best", fontsize=8)

        self.ax_throttle.plot(time, result["gt_throttle_pct"], color="green", label="GT Throttle (%)")
        self.ax_throttle.plot(
            time,
            result["ff_throttle_pct"],
            color="black",
            linestyle="--",
            label="FF Throttle Eqv. (%)",
        )
        self.ax_throttle.set_ylabel("Throttle (%)")
        self.ax_throttle.grid(True, alpha=0.3)
        self.ax_throttle.legend(loc="best", fontsize=8)

        self.ax_brake.plot(time, result["gt_brake_pct"], color="red", label="GT Brake (%)")
        self.ax_brake.plot(
            time,
            result["ff_brake_pct"],
            color="black",
            linestyle="--",
            label="FF Brake Eqv. (%)",
        )
        self.ax_brake.set_ylabel("Brake (%)")
        self.ax_brake.set_xlabel("Time (s)")
        self.ax_brake.grid(True, alpha=0.3)
        self.ax_brake.legend(loc="best", fontsize=8)

        self.ax_power.plot(time, result["ff_motor_power_W"], color="tab:blue", label="Electrical Power")
        self.ax_power.plot(time, result["ff_tractive_power_W"], color="tab:green", label="Tractive Power")
        self.ax_power.plot(time, result["ff_net_power_W"], color="tab:cyan", label="Net Power")
        self.ax_power.plot(time, result["ff_drag_power_W"], color="tab:red", linestyle="--", label="Drag Loss")
        self.ax_power.plot(time, result["ff_rolling_power_W"], color="tab:orange", linestyle="--", label="Rolling Loss")
        self.ax_power.plot(time, result["ff_grade_power_W"], color="tab:brown", linestyle=":", label="Grade Power")
        p_max = result["ff_motor_p_max_W"]
        if np.isfinite(p_max).any():
            self.ax_power.plot(time, p_max, color="tab:red", linestyle="--", label="P_max")
        self.ax_power.set_ylabel("Power (W)")
        self.ax_power.grid(True, alpha=0.3)
        self.ax_power.legend(loc="best", fontsize=8)

        self.ax_current.plot(time, result["ff_motor_current_A"], color="tab:green", label="FF Motor Current")
        self.ax_current.plot(time, result["ff_motor_i_limit_A"], color="tab:orange", linestyle=":", label="Dynamic I_limit")
        self.ax_current.plot(time, result["ff_motor_i_max_A"], color="tab:red", linestyle="--", label="Static I_max")
        self.ax_current.set_ylabel("Current (A)")
        self.ax_current.grid(True, alpha=0.3)
        self.ax_current.legend(loc="best", fontsize=8)

        self.ax_voltage.plot(time, result["ff_motor_voltage_V"], color="tab:blue", label="Command Voltage")
        self.ax_voltage.plot(time, result["ff_back_emf_voltage_V"], color="tab:purple", linestyle="--", label="Back-EMF")
        self.ax_voltage.set_ylabel("Voltage (V)")
        self.ax_voltage.set_xlabel("Time (s)")
        self.ax_voltage.grid(True, alpha=0.3)
        self.ax_voltage.legend(loc="best", fontsize=8)

        self.ax_forces.plot(time, result["ff_tire_force_N"], color="tab:green", label="Tire Force")
        self.ax_forces.plot(time, result["ff_net_force_N"], color="tab:cyan", label="Net Force")
        self.ax_forces.plot(time, result["ff_drag_force_N"], color="tab:red", linestyle="--", label="Drag")
        self.ax_forces.plot(time, result["ff_rolling_force_N"], color="tab:orange", linestyle="--", label="Rolling")
        self.ax_forces.plot(time, result["ff_grade_force_N"], color="tab:brown", linestyle=":", label="Grade")
        self.ax_forces.set_ylabel("Force (N)")
        self.ax_forces.set_xlabel("Time (s)")
        self.ax_forces.grid(True, alpha=0.3)
        self.ax_forces.legend(loc="best", fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_closed_loop(self, result: dict[str, np.ndarray], trip_id: str) -> None:
        time = result["time"]
        self.ax_speed.clear()
        self.ax_accel.clear()
        self.ax_throttle.clear()
        self.ax_brake.clear()
        self.ax_power.clear()
        self.ax_current.clear()
        self.ax_voltage.clear()
        self.ax_forces.clear()

        self.ax_speed.plot(time, result["gt_speed"], color="black", linestyle="--", label="GT Speed")
        self.ax_speed.plot(time, result["sim_speed"], color="tab:blue", label="Sim Speed (FF)")
        self.ax_speed.set_ylabel("Speed (m/s)")
        self.ax_speed.set_title(f"Closed Loop | Trip {trip_id}")
        self.ax_speed.grid(True, alpha=0.3)
        self.ax_speed.legend(loc="best", fontsize=8)

        self.ax_accel.plot(time, result["gt_accel_filtered"], color="black", linestyle="--", label="GT Accel")
        self.ax_accel.plot(time, result["sim_acceleration"], color="tab:purple", label="Sim Accel (FF)")
        self.ax_accel.set_ylabel("Accel (m/s^2)")
        self.ax_accel.grid(True, alpha=0.3)
        self.ax_accel.legend(loc="best", fontsize=8)

        self.ax_throttle.plot(time, result["gt_throttle_pct"], color="green", label="GT Throttle (%)")
        self.ax_throttle.plot(
            time,
            result["ff_throttle_pct"],
            color="black",
            linestyle="--",
            label="FF Throttle Eqv. (%)",
        )
        self.ax_throttle.set_ylabel("Throttle (%)")
        self.ax_throttle.grid(True, alpha=0.3)
        self.ax_throttle.legend(loc="best", fontsize=8)

        self.ax_brake.plot(time, result["gt_brake_pct"], color="red", label="GT Brake (%)")
        self.ax_brake.plot(
            time,
            result["ff_brake_pct"],
            color="black",
            linestyle="--",
            label="FF Brake Eqv. (%)",
        )
        self.ax_brake.set_ylabel("Brake (%)")
        self.ax_brake.set_xlabel("Time (s)")
        self.ax_brake.grid(True, alpha=0.3)
        self.ax_brake.legend(loc="best", fontsize=8)

        self.ax_power.plot(time, result["ff_motor_power_W"], color="tab:blue", label="Electrical Power")
        self.ax_power.plot(time, result["ff_tractive_power_W"], color="tab:green", label="Tractive Power")
        self.ax_power.plot(time, result["ff_net_power_W"], color="tab:cyan", label="Net Power")
        self.ax_power.plot(time, result["ff_drag_power_W"], color="tab:red", linestyle="--", label="Drag Loss")
        self.ax_power.plot(time, result["ff_rolling_power_W"], color="tab:orange", linestyle="--", label="Rolling Loss")
        self.ax_power.plot(time, result["ff_grade_power_W"], color="tab:brown", linestyle=":", label="Grade Power")
        p_max = result["ff_motor_p_max_W"]
        if np.isfinite(p_max).any():
            self.ax_power.plot(time, p_max, color="tab:red", linestyle="--", label="P_max")
        self.ax_power.set_ylabel("Power (W)")
        self.ax_power.grid(True, alpha=0.3)
        self.ax_power.legend(loc="best", fontsize=8)

        self.ax_current.plot(time, result["ff_motor_current_A"], color="tab:green", label="FF Motor Current")
        self.ax_current.plot(time, result["ff_motor_i_limit_A"], color="tab:orange", linestyle=":", label="Dynamic I_limit")
        self.ax_current.plot(time, result["ff_motor_i_max_A"], color="tab:red", linestyle="--", label="Static I_max")
        self.ax_current.set_ylabel("Current (A)")
        self.ax_current.grid(True, alpha=0.3)
        self.ax_current.legend(loc="best", fontsize=8)

        self.ax_voltage.plot(time, result["ff_motor_voltage_V"], color="tab:blue", label="Command Voltage")
        self.ax_voltage.plot(time, result["ff_back_emf_voltage_V"], color="tab:purple", linestyle="--", label="Back-EMF")
        self.ax_voltage.set_ylabel("Voltage (V)")
        self.ax_voltage.set_xlabel("Time (s)")
        self.ax_voltage.grid(True, alpha=0.3)
        self.ax_voltage.legend(loc="best", fontsize=8)

        self.ax_forces.plot(time, result["ff_tire_force_N"], color="tab:green", label="Tire Force")
        self.ax_forces.plot(time, result["ff_net_force_N"], color="tab:cyan", label="Net Force")
        self.ax_forces.plot(time, result["ff_drag_force_N"], color="tab:red", linestyle="--", label="Drag")
        self.ax_forces.plot(time, result["ff_rolling_force_N"], color="tab:orange", linestyle="--", label="Rolling")
        self.ax_forces.plot(time, result["ff_grade_force_N"], color="tab:brown", linestyle=":", label="Grade")
        self.ax_forces.set_ylabel("Force (N)")
        self.ax_forces.set_xlabel("Time (s)")
        self.ax_forces.grid(True, alpha=0.3)
        self.ax_forces.legend(loc="best", fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _save_settings(self, silent: bool = False) -> None:
        settings = {
            "dataset": self.dataset_var.get(),
            "trip": self.trip_var.get(),
            "mode": self.mode_var.get(),
            "use_trip_grade": bool(self.use_trip_grade_var.get()),
            "substeps": self.substeps_var.get(),
            "accel_filter_cutoff_hz": self._get_accel_filter_cutoff_hz(),
            "throttle_gain": self._get_throttle_gain(),
            "brake_gain": self._get_brake_gain(),
            "load_saved_vehicle_values_on_startup": False,
            "vehicle_values": self.vehicle_values,
        }
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            if not silent:
                messagebox.showinfo("Settings", f"Saved settings to:\n{self.settings_file}")
        except Exception as exc:
            if not silent:
                messagebox.showerror("Settings Error", f"Failed saving settings:\n{exc}")

    def _load_settings(self) -> None:
        if not self.settings_file.exists():
            return
        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                settings = json.load(f)

            dataset = settings.get("dataset", "")
            if dataset:
                self.dataset_var.set(dataset)
                self._load_selected_dataset()

            trip = settings.get("trip", "")
            if trip:
                self.trip_var.set(trip)

            self.mode_var.set(settings.get("mode", "open_loop"))
            self.use_trip_grade_var.set(bool(settings.get("use_trip_grade", True)))
            self.substeps_var.set(str(settings.get("substeps", "1")))
            try:
                self.throttle_gain_var.set(min(max(float(settings.get("throttle_gain", 1.0)), 0.0), 1.0))
            except (TypeError, ValueError):
                self.throttle_gain_var.set(1.0)
            try:
                self.brake_gain_var.set(min(max(float(settings.get("brake_gain", 1.0)), 0.0), 1.0))
            except (TypeError, ValueError):
                self.brake_gain_var.set(1.0)
            self._on_throttle_gain_slider(None)
            self._on_brake_gain_slider(None)
            if "accel_filter_cutoff_hz" in settings:
                try:
                    self.accel_filter_cutoff_hz_var.set(max(float(settings.get("accel_filter_cutoff_hz", 2.0)), 0.0))
                except (TypeError, ValueError):
                    self.accel_filter_cutoff_hz_var.set(2.0)
            else:
                # Legacy compatibility for old moving-average setting.
                try:
                    legacy_window = max(int(settings.get("accel_filter_window", 1)), 1)
                except (TypeError, ValueError):
                    legacy_window = 1
                self.accel_filter_cutoff_hz_var.set(0.0 if legacy_window <= 1 else 2.0)
            self._on_accel_filter_slider(None)

            # Avoid accidentally restoring stale per-session vehicle values (e.g. 400V)
            # when user expects fitting GUI defaults (e.g. 358V).
            load_saved_vehicle_values = bool(
                settings.get(
                    "load_saved_vehicle_values_on_startup",
                    LOAD_SAVED_VEHICLE_VALUES_ON_STARTUP_DEFAULT,
                )
            )
            if load_saved_vehicle_values:
                loaded_values = settings.get("vehicle_values")
                if isinstance(loaded_values, dict):
                    merged = _default_vehicle_values()
                    for key in merged.keys():
                        if key in loaded_values:
                            raw = loaded_values[key]
                            if raw is None:
                                merged[key] = None
                            else:
                                merged[key] = float(raw)
                    self.vehicle_values = merged

            self._refresh_param_summary()
        except Exception:
            LOGGER.exception("Failed to load feedforward GUI settings")

    def _format_metrics(self, metrics: dict[str, float], mode: str) -> str:
        if mode == "open_loop":
            return (
                f"samples: {int(metrics['samples'])}\n"
                f"clipped: {metrics['clip_ratio_pct']:.2f}%\n"
                f"throttle RMSE: {metrics['throttle_rmse_pct']:.3f}%\n"
                f"throttle MAE:  {metrics['throttle_mae_pct']:.3f}%\n"
                f"brake RMSE:    {metrics['brake_rmse_pct']:.3f}%\n"
                f"brake MAE:     {metrics['brake_mae_pct']:.3f}%"
            )
        return (
            f"samples: {int(metrics['samples'])}\n"
            f"clipped: {metrics['clip_ratio_pct']:.2f}%\n"
            f"speed RMSE: {metrics['speed_rmse']:.4f} m/s\n"
            f"speed MAE:  {metrics['speed_mae']:.4f} m/s\n"
            f"accel RMSE: {metrics['accel_rmse']:.4f} m/s^2\n"
            f"accel MAE:  {metrics['accel_mae']:.4f} m/s^2\n"
            f"throttle MAE: {metrics['throttle_mae_pct']:.3f}%\n"
            f"brake MAE:    {metrics['brake_mae_pct']:.3f}%"
        )


def main() -> None:
    root = tk.Tk()
    _app = FeedforwardComparisonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
