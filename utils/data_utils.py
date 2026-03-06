from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.dynamics import GRAVITY


@dataclass(slots=True)
class VehicleCapabilities:
    m: float
    r_w: float
    T_drive_max: float
    T_brake_max: float
    mu: float
    C_dA: float
    C_r: float
    rho: float = 1.225


@dataclass(slots=True)
class VehicleMotorCapabilities:
    r_w: float
    N_g: float
    eta: float
    K_e: float
    K_t: float
    R: float
    V_max: float
    mass: float
    C_dA: float
    C_r: float
    T_max: float | None = None
    rho: float = 1.225


def feasible_accel_bounds(
    v: float,
    grade: float,
    caps: VehicleCapabilities,
    safety_margin: float = 1.0,
) -> tuple[float, float]:
    f_drag = 0.5 * caps.rho * caps.C_dA * v * abs(v)
    f_roll = caps.C_r * caps.m * GRAVITY * np.cos(grade)
    f_grade = caps.m * GRAVITY * np.sin(grade)
    f_resist = f_drag + f_roll + f_grade

    f_mu_limit = safety_margin * caps.mu * caps.m * GRAVITY
    f_drive = min(safety_margin * caps.T_drive_max / max(caps.r_w, 1e-9), f_mu_limit)
    f_brake = min(safety_margin * caps.T_brake_max / max(caps.r_w, 1e-9), f_mu_limit)

    a_max = (f_drive - f_resist) / caps.m
    a_min = (-f_brake - f_resist) / caps.m
    return float(a_min), float(a_max)


def project_profile_to_feasible(
    speed: np.ndarray,
    grade: np.ndarray,
    caps: VehicleCapabilities,
    dt: float,
    safety_margin: float = 1.0,
    max_iters: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0:
        raise ValueError("dt must be positive")

    v_feasible = np.asarray(speed, dtype=float).copy()
    grade_feasible = np.asarray(grade, dtype=float).copy()
    if v_feasible.shape != grade_feasible.shape:
        raise ValueError("speed and grade must have the same shape")

    for _ in range(max_iters):
        changed = False
        for k in range(len(v_feasible) - 1):
            a_req = (v_feasible[k + 1] - v_feasible[k]) / dt
            a_min, a_max = feasible_accel_bounds(
                float(v_feasible[k]),
                float(grade_feasible[k]),
                caps,
                safety_margin=safety_margin,
            )
            a_clip = float(np.clip(a_req, a_min, a_max))
            next_speed = max(0.0, float(v_feasible[k] + a_clip * dt))
            if abs(next_speed - v_feasible[k + 1]) > 1e-12:
                v_feasible[k + 1] = next_speed
                changed = True
        if not changed:
            break

    return v_feasible, grade_feasible


def initial_target_feasible(
    speed: float,
    grade: float,
    caps: VehicleMotorCapabilities,
) -> tuple[bool, float, float]:
    omega_w = speed / max(caps.r_w, 1e-9)
    omega_m = caps.N_g * omega_w

    f_drag = 0.5 * caps.rho * caps.C_dA * speed * speed
    f_roll = caps.C_r * caps.mass * GRAVITY
    f_grade = caps.mass * GRAVITY * np.sin(grade)
    f_resist = f_drag + f_roll + f_grade

    t_req_wheel = f_resist * caps.r_w
    t_req_motor = t_req_wheel / max(caps.N_g * caps.eta, 1e-9)

    i_needed = t_req_motor / max(caps.K_t, 1e-9)
    v_needed = caps.K_e * omega_m + caps.R * i_needed

    current_ok = True
    if caps.T_max is not None:
        current_ok = t_req_motor <= 0.95 * caps.T_max

    feasible = (v_needed <= 0.95 * caps.V_max) and current_ok
    return bool(feasible), float(v_needed), float(i_needed)


def adjust_initial_target(
    speed: float,
    grade: float,
    caps: VehicleMotorCapabilities,
    v_step: float = 2.0,
    grade_step_deg: float = 0.5,
    max_iter_v: int = 20,
    max_iter_grade: int = 20,
) -> tuple[float, float, float, float]:
    v_adj = max(0.0, float(speed))
    grade_adj = float(grade)

    for _ in range(max_iter_v + 1):
        feasible, v_needed, i_needed = initial_target_feasible(v_adj, grade_adj, caps)
        if feasible:
            return v_adj, grade_adj, v_needed, i_needed
        if v_adj <= 0.0:
            break
        v_adj = max(0.0, v_adj - v_step)

    for _ in range(max_iter_grade):
        feasible, v_needed, i_needed = initial_target_feasible(v_adj, grade_adj, caps)
        if feasible:
            return v_adj, grade_adj, v_needed, i_needed
        if grade_adj <= 0.0:
            break
        grade_adj = max(0.0, grade_adj - np.deg2rad(grade_step_deg))

    feasible, v_needed, i_needed = initial_target_feasible(v_adj, grade_adj, caps)
    return v_adj, grade_adj, v_needed, i_needed


__all__ = [
    "VehicleCapabilities",
    "VehicleMotorCapabilities",
    "feasible_accel_bounds",
    "project_profile_to_feasible",
    "initial_target_feasible",
    "adjust_initial_target",
]
