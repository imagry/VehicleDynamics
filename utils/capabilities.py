"""Vehicle capability analysis functions.

This module provides functions to compute vehicle capabilities such as
maximum acceleration, top speed, and other performance metrics from
vehicle parameters.
"""

from __future__ import annotations

import numpy as np

from simulation.dynamics import GRAVITY


def compute_vehicle_capabilities(
    V_max: float, R: float, K_e: float, K_t: float, b: float, 
    gear_ratio: float, eta_gb: float, r_w: float, mass: float,
    CdA: float, C_rr: float, T_brake_max: float,
    T_max: float | None = None,
    P_max: float | None = None,
) -> dict:
    """Compute derived vehicle capabilities for feasibility checking.
    
    Returns:
        Dictionary with:
        - i_stall: Stall current (A)
        - tau_wheel_drive_max: Max drive torque at wheel (Nm)
        - F_drive_max0: Max drive force at standstill (N)
        - v_no_load_max: Max no-load speed (m/s) - back-EMF limit
        - v_ss_level: Steady-state max speed on level ground (m/s)
        - a_max_from_rest: Max acceleration from standstill (m/s²)
        - a_brake_max: Max braking deceleration magnitude (m/s²)
    """
    g = GRAVITY
    
    # Stall current (at zero speed, V_cmd = V_max)
    i_stall = V_max / max(R, 1e-9)
    i_limit = i_stall
    if T_max is not None:
        i_limit = min(i_limit, T_max / max(K_t, 1e-9))
    if P_max is not None:
        i_limit = min(i_limit, P_max / max(V_max, 1e-6))
    
    # Stall motor shaft torque
    tau_m_stall = K_t * i_limit
    
    # Stall wheel torque (accounting for gearbox)
    tau_wheel_drive_max = eta_gb * gear_ratio * tau_m_stall
    
    # Max drive force at standstill
    F_drive_max0 = tau_wheel_drive_max / max(r_w, 1e-6)
    
    # No-load max speed (back-EMF limit): V_max = K_e * omega_m
    omega_m_no_load = V_max / max(K_e, 1e-9)
    v_no_load_max = (omega_m_no_load / gear_ratio) * r_w
    
    # Resistive forces at standstill
    F_roll_0 = C_rr * mass * g
    
    # Max acceleration from rest (on level ground, no drag yet)
    a_max_from_rest = (F_drive_max0 - F_roll_0) / mass
    
    # Steady-state max speed under full throttle (on level ground)
    # Solve: F_drive_avail(v) = F_resist(v) numerically
    v_ss_level = _compute_steady_state_speed(
        V_max, R, K_e, K_t, b, gear_ratio, eta_gb, r_w, mass, CdA, C_rr,
        grade=0.0, T_max=T_max, P_max=P_max
    )
    
    # Max braking deceleration (at low speed on level ground)
    # Brake force + resistive forces
    F_brake_max = T_brake_max / max(r_w, 1e-6)
    # At low speed, resistive forces are minimal, so approx:
    a_brake_max = F_brake_max / mass
    
    return {
        'i_stall': i_stall,
        'tau_wheel_drive_max': tau_wheel_drive_max,
        'F_drive_max0': F_drive_max0,
        'v_no_load_max': v_no_load_max,
        'v_ss_level': v_ss_level,
        'a_max_from_rest': a_max_from_rest,
        'a_brake_max': a_brake_max,
    }


def _compute_steady_state_speed(
    V_max: float, R: float, K_e: float, K_t: float, b: float,
    gear_ratio: float, eta_gb: float, r_w: float, mass: float,
    CdA: float, C_rr: float, grade: float = 0.0, tol: float = 0.01,
    T_max: float | None = None, P_max: float | None = None,
) -> float:
    """Compute steady-state max speed under full throttle using bisection.
    
    Finds v where F_drive_avail(v) = F_resist(v, grade).
    """
    g = GRAVITY
    
    def F_drive_avail(v: float) -> float:
        """Drive force available at speed v (current control)."""
        omega_m = gear_ratio * v / max(r_w, 1e-6)
        i_limit = V_max / max(R, 1e-9) if T_max is None else (T_max / max(K_t, 1e-9))
        target_current = i_limit
        v_required = target_current * R + K_e * omega_m
        v_applied = min(v_required, V_max)
        i_actual = max((v_applied - K_e * omega_m) / max(R, 1e-9), 0.0)
        if P_max is not None and v_applied > 1e-6:
            i_actual = min(i_actual, P_max / v_applied)
        tau_m_net = max(0.0, K_t * i_actual - b * omega_m)
        tau_wheel = eta_gb * gear_ratio * tau_m_net
        return tau_wheel / max(r_w, 1e-6)
    
    def F_resist(v: float) -> float:
        """Resistive forces at speed v and grade."""
        F_aero = CdA * v * v
        F_roll = C_rr * mass * g * np.cos(grade)
        F_grade = mass * g * np.sin(grade)
        return F_aero + F_roll + F_grade
    
    # No-load max speed upper bound
    omega_m_no_load = V_max / max(K_e, 1e-9)
    v_hi = (omega_m_no_load / gear_ratio) * r_w
    v_lo = 0.0
    
    # Check if starting condition is feasible
    if F_drive_avail(0.0) < F_resist(0.0):
        return 0.0  # Can't even start
    
    # Bisection to find equilibrium
    for _ in range(60):
        v_mid = 0.5 * (v_lo + v_hi)
        if F_drive_avail(v_mid) >= F_resist(v_mid):
            v_lo = v_mid
        else:
            v_hi = v_mid
        if v_hi - v_lo < tol:
            break
    
    return v_lo


def compute_max_accel_at_speed(
    v: float, grade: float,
    V_max: float, R: float, K_e: float, K_t: float, b: float,
    gear_ratio: float, eta_gb: float, r_w: float, mass: float,
    CdA: float, C_rr: float,
    T_max: float | None = None, P_max: float | None = None,
) -> float:
    """Compute maximum feasible acceleration at given speed and grade.
    
    Uses the no-regen model: i >= 0 always.
    """
    g = GRAVITY
    
    # Motor angular speed
    omega_m = gear_ratio * v / max(r_w, 1e-6)
    
    # Max current at this speed (no-regen: clamp to 0)
    i_limit = V_max / max(R, 1e-9) if T_max is None else (T_max / max(K_t, 1e-9))
    target_current = i_limit
    v_required = target_current * R + K_e * omega_m
    v_applied = min(v_required, V_max)
    i_actual = max((v_applied - K_e * omega_m) / max(R, 1e-9), 0.0)
    if P_max is not None and v_applied > 1e-6:
        i_actual = min(i_actual, P_max / v_applied)

    # Motor shaft torque (accounting for viscous)
    tau_m_net = max(0.0, K_t * i_actual - b * omega_m)
    
    # Wheel drive torque and force
    tau_wheel = eta_gb * gear_ratio * tau_m_net
    F_drive = tau_wheel / max(r_w, 1e-6)
    
    # Resistive forces
    F_aero = CdA * v * abs(v)
    F_roll = C_rr * mass * g * np.cos(grade)
    F_grade = mass * g * np.sin(grade)
    F_resist = F_aero + F_roll + F_grade
    
    # Net force and acceleration
    F_net = F_drive - F_resist
    a_max = F_net / mass
    
    return a_max


__all__ = [
    "compute_vehicle_capabilities",
    "compute_max_accel_at_speed",
]
