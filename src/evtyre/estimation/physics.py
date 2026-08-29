"""Phase 3 tyre physics models.

Ported from legacy/ev_tyre_fusion.py as reference (not copied). Every constant
is a REQUIRED config field with NO default, carrying an explicit "validated: no"
note. Follows the Phase 2 precedent of RoadLoadParams in features/road_load.py.

CONSTRAINT: Do NOT implement first_mode_frequency (it would prejudge the open
first-vs-second torsional mode question), wet_friction, torque_limit, or
recoverable_energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..schema.common import CORNERS


@dataclass(frozen=True)
class PhysicsConfig:
    """Physical constants for the tyre estimator.

    EVERY field is REQUIRED with NO default.  Every value is UNVALIDATED —
    it is a guess until checked against bench data.  This is deliberately not
    a sensible-defaults pattern: a default stiffness would silently set the
    rolling-radius model, which is the channel that makes tread observable.
    """

    k_z0: float
    """Vertical stiffness at placard pressure (N/m).  UNVALIDATED guess.
    Legacy value: 210_000."""

    cornering_stiffness: float
    """C_alpha (N/rad) for toe drag.  UNVALIDATED guess.
    Legacy value: 55_000."""

    tread_rr_span: float
    """Fraction of C_rr variation attributable to tread (dimensionless).
    UNVALIDATED guess.  Legacy value: 0.20.
    Worn tyres have LOWER rolling resistance: t_term < 1."""

    c_rr0: float
    """Reference rolling resistance coefficient (dimensionless).
    UNVALIDATED guess.  Legacy value: 0.0090."""

    p_exponent: float
    """Pressure exponent for C_rr (dimensionless).
    UNVALIDATED guess.  Legacy value: 0.45."""

    t_coeff: float
    """Temperature coefficient for C_rr (1/C).
    UNVALIDATED guess.  Legacy value: 0.0015."""

    deflection_factor: float
    """Fraction of vertical deflection that reduces effective rolling radius
    (dimensionless).  UNVALIDATED guess.  Legacy value: 1/3.

    Physics: only about one-third of total vertical tyre deflection translates
    into a reduction of the effective rolling radius, because the contact patch
    distributes deformation around the circumference.  This factor should be
    validated against bench data for each tyre model."""


def corner_weight(vehicle_config: VehicleConfig, corner: str) -> float:
    """Static vertical load on one corner (N).

    Uses the vehicle mass and front weight fraction.  This is the STATIC
    load only — dynamic load transfer is not modelled.
    """
    g = 9.80665  # measured constant, not a guess
    if corner in ("FL", "FR"):
        share = vehicle_config.front_weight_fraction
    else:
        share = 1.0 - vehicle_config.front_weight_fraction
    return vehicle_config.mass_kg * g * share / 2.0


def effective_rolling_radius(
    tread_mm: float,
    p_kpa: float,
    fz_n: float,
    r_belt: float,
    k_z0: float,
    p_placard_kpa: float,
    deflection_factor: float,
) -> float:
    """Effective rolling radius of a tyre.

    Ported from legacy/ev_tyre_fusion.py:
        r_free = r_belt + tread/1000
        k_z    = k_z0 * (p / p_placard)
        delta  = Fz / k_z
        r_eff  = r_free - deflection_factor * delta

    The deflection_factor is an empirical constant representing the fraction
    of vertical deflection that reduces effective rolling radius.  UNVALIDATED.
    REQUIRED — no default: it must come from PhysicsConfig so the constant has
    exactly one home. The legacy reference value is 1/3.
    k_z0 is the vertical stiffness at placard pressure (N/m).
    """
    r_free = r_belt + tread_mm / 1000.0
    k_z = k_z0 * (p_kpa / p_placard_kpa)
    delta = fz_n / max(k_z, 1.0)
    return r_free - deflection_factor * delta


def rolling_resistance_coeff(
    tread_mm: float,
    p_kpa: float,
    t_c: float,
    tyre_config: TyreConfig,
    physics: PhysicsConfig,
) -> float:
    """Rolling resistance coefficient (dimensionless).

    Ported from legacy/ev_tyre_fusion.py:
        C_rr = C_rr0 * p_term * t_term * T_term

    Worn tyres have LOWER rolling resistance (t_term < 1).
    """
    p_term = (tyre_config.placard_pressure_kpa / max(p_kpa, 1.0)) ** physics.p_exponent
    s = physics.tread_rr_span
    t_term = (1.0 - s) + s * (tread_mm / tyre_config.tread_new_mm)
    T_term = 1.0 + physics.t_coeff * (t_c - tyre_config.cold_reference_temperature_c)
    return physics.c_rr0 * p_term * t_term * T_term


def toe_drag_from_sq(toe_sq_deg2: float, physics: PhysicsConfig) -> float:
    """Toe drag force from toe-squared (N).

    F_toe = 2 * C_alpha * (pi/180)^2 * toe_sq

    The sign is NOT recoverable because drag is even in toe.
    This function returns the magnitude.
    """
    return 2.0 * physics.cornering_stiffness * (math.pi / 180.0) ** 2 * toe_sq_deg2
