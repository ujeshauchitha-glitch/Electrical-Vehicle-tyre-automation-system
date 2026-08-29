"""
Physics module for EV tyre state estimation.

Vehicle constants and tyre physics functions: rolling radius, resonance
frequency, rolling resistance, wet friction, toe drag, and road load.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Vehicle parameters
# ---------------------------------------------------------------------------

class Vehicle:
    """Physical constants for a representative mid-size EV."""

    mass          = 1800.0        # kg
    front_weight  = 0.48          # fraction on front axle
    g             = 9.81          # m/s^2
    r_belt        = 0.322         # belt radius, m
    tread_new     = 8.0           # mm
    tread_legal   = 1.6           # mm

    p_placard     = 240.0         # kPa gauge
    p_atm         = 101.325       # kPa

    k_z0          = 210_000.0     # N/m at placard pressure

    J_carcass     = 0.550         # kg.m^2
    J_per_mm      = 0.0332        # kg.m^2 per mm tread
    K_carcass     = 39_121.0      # N.m/rad at placard
    K_per_kPa     = 108.67        # N.m/rad per kPa

    C_rr0         = 0.0090        # baseline rolling resistance coeff
    p_exponent    = 0.45          # pressure exponent
    tread_rr_span = 0.20          # fractional span of tread effect
    T_coeff       = 0.0015        # per deg C
    T_ref         = 25.0          # reference temperature, deg C

    CdA           = 0.65          # drag area, m^2
    rho_air       = 1.20          # kg/m^3

    C_alpha       = 55_000.0      # cornering stiffness, N/rad

    mu_dry        = 1.00
    wet_floor     = 0.45
    wet_tau       = 2.5           # mm characteristic length

    @classmethod
    def Fz(cls, corner: str) -> float:
        """Static normal load on a single corner (N)."""
        share = cls.front_weight if corner in ("FL", "FR") else 1.0 - cls.front_weight
        return cls.mass * cls.g * share / 2.0


CORNERS = ("FL", "FR", "RL", "RR")


# ---------------------------------------------------------------------------
# Pressure helpers
# ---------------------------------------------------------------------------

def compensate_pressure(p_gauge_kPa: float, T_tyre_C: float, T_ref_C: float = Vehicle.T_ref) -> float:
    """Normalise running pressure to a cold reference temperature (reporting only)."""
    p_abs = p_gauge_kPa + Vehicle.p_atm
    p_ref_abs = p_abs * (T_ref_C + 273.15) / (T_tyre_C + 273.15)
    return p_ref_abs - Vehicle.p_atm


# ---------------------------------------------------------------------------
# Tyre physics
# ---------------------------------------------------------------------------

def effective_rolling_radius(tread_mm: float, p_kPa: float, Fz_N: float) -> float:
    """Effective rolling radius (m) under vertical load."""
    r_free = Vehicle.r_belt + tread_mm / 1000.0
    k_z = Vehicle.k_z0 * (p_kPa / Vehicle.p_placard)
    delta = Fz_N / k_z
    return r_free - delta / 3.0


def first_mode_frequency(tread_mm: float, p_kPa: float) -> float:
    """First structural resonance frequency (Hz)."""
    J = Vehicle.J_carcass + Vehicle.J_per_mm * tread_mm
    K = Vehicle.K_carcass + Vehicle.K_per_kPa * p_kPa
    return np.sqrt(K / J) / (2.0 * np.pi)


def rolling_resistance_coeff(tread_mm: float, p_kPa: float, T_C: float) -> float:
    """Rolling resistance coefficient (dimensionless)."""
    p_term = (Vehicle.p_placard / p_kPa) ** Vehicle.p_exponent
    s = Vehicle.tread_rr_span
    t_term = (1.0 - s) + s * (tread_mm / Vehicle.tread_new)
    T_term = 1.0 + Vehicle.T_coeff * (T_C - Vehicle.T_ref)
    return Vehicle.C_rr0 * p_term * t_term * T_term


def toe_drag_force(toe_deg: float) -> float:
    """Aero-equivalent drag force from toe (N)."""
    toe_rad = np.deg2rad(toe_deg)
    return 2.0 * Vehicle.C_alpha * toe_rad ** 2


def toe_drag_from_sq(toe_sq_deg2: float) -> float:
    """Toe drag from toe-squared (avoids sqrt in linearised estimator)."""
    return 2.0 * Vehicle.C_alpha * (np.pi / 180.0) ** 2 * toe_sq_deg2


def road_load(state, v_ms: float, accel_ms2: float = 0.0, grade_rad: float = 0.0) -> float:
    """Total road load force (N) at velocity v_ms."""
    C_rr_mean = np.mean([
        rolling_resistance_coeff(state.tread[c], state.pressure[c], state.temp[c])
        for c in CORNERS
    ])
    F_roll = C_rr_mean * Vehicle.mass * Vehicle.g * np.cos(grade_rad)
    F_aero = 0.5 * Vehicle.rho_air * Vehicle.CdA * v_ms ** 2
    F_grade = Vehicle.mass * Vehicle.g * np.sin(grade_rad)
    F_inertia = Vehicle.mass * accel_ms2
    F_toe = toe_drag_force(state.toe)
    return F_roll + F_aero + F_grade + F_inertia + F_toe


def slip_stiffness(tread_mm: float, p_kPa: float, Fz_N: float) -> float:
    """Longitudinal slip stiffness (N)."""
    p_term = (p_kPa / Vehicle.p_placard) ** 0.2
    wear_term = 1.0 + 0.04 * (Vehicle.tread_new - tread_mm)
    return 12.0 * Fz_N * p_term * wear_term


def wet_friction(tread_mm: float) -> float:
    """Effective wet friction coefficient as a function of tread depth."""
    k = Vehicle.wet_floor
    return Vehicle.mu_dry * (k + (1 - k) * (1 - np.exp(-tread_mm / Vehicle.wet_tau)))


def torque_limit(
    tread_est_mm: float,
    tread_sigma_mm: float,
    pressure_kPa: float,
    wet: bool = True,
    safety: float = 0.85,
) -> tuple[float, float, float]:
    """Compute torque ceiling from estimated tread state.

    Returns (torque_Nm, mu_effective, tread_lower_confidence_bound).
    """
    tread_lcb = max(0.5, tread_est_mm - 2.0 * tread_sigma_mm)
    mu = wet_friction(tread_lcb) if wet else Vehicle.mu_dry
    Fz_drive = Vehicle.Fz("RL") + Vehicle.Fz("RR")
    r_eff = effective_rolling_radius(tread_est_mm, pressure_kPa, Vehicle.Fz("RL"))
    F_max = mu * Fz_drive * safety
    return F_max * r_eff, mu, tread_lcb


def recoverable_energy(state) -> float:
    """Percentage excess road load vs ideal pressure and zero toe."""
    v = 22.0
    F_now = road_load(state, v)

    # Build an ideal-state version with the same tread and temperature
    from .estimator import TyreState  # avoid circular import at module level

    fixed = TyreState(
        tread=state.tread,
        pressure={c: Vehicle.p_placard for c in CORNERS},
        temp=state.temp,
        toe=0.0,
    )
    F_fixed = road_load(fixed, v)
    return 100.0 * (F_now - F_fixed) / F_fixed
