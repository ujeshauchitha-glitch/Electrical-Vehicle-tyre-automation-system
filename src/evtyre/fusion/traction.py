"""Traction limit: a wet/dry torque ceiling from the estimated tread.

The argument for this output is genuinely good: every EV already cuts torque
AFTER a wheel slips. Knowing the tread lets it cut torque BEFORE. That is the
whole value proposition of the tread estimate.

It also cannot be delivered today, and this module says so rather than
producing a number.

WHY IT REFUSES
--------------
Legacy computes the ceiling from a conservative lower bound,
``tread_lcb = tread_est - 2*sigma``, then feeds that to the wet-friction curve.
With the current channel set, tread is WEAK: the axle-speed ratio constrains
the within-axle DIFFERENCE (variance reduction ~0.82 along FL-FR) while every
common-mode direction is unconstrained (reduction 0.0000). So on a
representative frame:

    drive-axle tread est   = 4.550 mm   <- exactly the prior, (7.5+1.6)/2
    drive-axle sigma       = 1.921 mm
    2-sigma lower bound    = 0.708 mm   <- below the 1.6 mm legal limit

That 0.708 mm is a restatement of how wide the prior is. It contains no tread
measurement. A traction limit derived from it would be a fabricated number
acted upon by a vehicle, which is a materially worse outcome than the same
class of bug in a report.

So: OK only when the drive-axle tread states are OBSERVED. That becomes
possible when an independent channel constrains the absolute tread level —
the resonance channel, currently blocked by G1 and the open first-vs-second
torsional mode question. The code path is complete and tested; it is the input
that is missing.
"""

from __future__ import annotations

import math

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..estimation.physics import corner_weight, effective_rolling_radius
from ..estimation.estimator import PhysicsConfig
from ..estimation.schema import TyreStateEstimate
from .config import FrictionConfig, driven_corners
from .contract import Decision, require_observed

_SIGMA_MULTIPLIER = 2.0
"""Lower-confidence bound width. 2 sigma ~ 97.7% one-sided under a Gaussian
posterior. A safety limit uses the pessimistic end of the estimate, never the
point estimate."""

_MIN_TREAD_MM = 0.0
"""Tread cannot be negative. Clamping the LCB at 0 rather than at some positive
floor is deliberate: a positive floor would quietly assert grip that the
estimate does not support."""


def wet_friction(tread_mm: float, friction: FrictionConfig) -> float:
    """Peak wet friction as a function of tread depth.

    mu_wet = mu_dry * (k + (1-k) * (1 - exp(-tread/tau)))

    UNVALIDATED curve shape with no wet-braking or wet-cornering data behind
    it. See FrictionConfig.
    """
    k = friction.wet_floor
    return friction.mu_dry * (k + (1.0 - k) * (1.0 - math.exp(-tread_mm / friction.wet_tau)))


def torque_ceiling(
    estimate: TyreStateEstimate,
    vehicle_config: VehicleConfig,
    tyre_config: TyreConfig,
    physics: PhysicsConfig,
    friction: FrictionConfig,
    *,
    wet: bool = True,
) -> tuple[Decision, ...]:
    """Maximum drive torque the tyres can carry, from the estimated tread.

    Returns the ceiling plus the intermediate quantities a reviewer needs to
    audit it: the lower-confidence tread bound actually used, and the friction
    coefficient at that bound.
    """
    corners = driven_corners(vehicle_config)
    tread_states = [f"tread_{c}" for c in corners]
    condition = "wet" if wet else "dry"

    blocked = require_observed(estimate, tread_states)
    if blocked is not None:
        reason = (
            f"Drive-axle tread is not OBSERVED, so no traction limit can be "
            f"derived: {blocked}. The axle-speed-ratio channel constrains only "
            f"the within-axle tread DIFFERENCE; the absolute level is "
            f"unconstrained, so a lower-confidence bound would be a restatement "
            f"of the prior width rather than a measurement. Requires an "
            f"independent absolute-tread channel (resonance, blocked by G1)."
        )
        return (
            Decision.unavailable(
                name=f"torque_ceiling_{condition}_nm",
                unit="N.m", reason=reason, basis=tread_states,
            ),
            Decision.unavailable(
                name="drive_tread_lower_bound_mm",
                unit="mm", reason=reason, basis=tread_states,
            ),
            Decision.unavailable(
                name=f"peak_friction_{condition}",
                unit="", reason=reason, basis=tread_states,
            ),
        )

    by_name = {s.name: s for s in estimate.states}
    tread_vals = [by_name[n].value for n in tread_states]
    tread_sigmas = [by_name[n].sigma for n in tread_states]

    tread_mean = sum(tread_vals) / len(tread_vals)
    sigma_mean = sum(tread_sigmas) / len(tread_sigmas)
    tread_lcb = max(_MIN_TREAD_MM, tread_mean - _SIGMA_MULTIPLIER * sigma_mean)

    mu = wet_friction(tread_lcb, friction) if wet else friction.mu_dry

    press_states = [f"press_{c}" for c in corners]
    press_blocked = require_observed(estimate, press_states)
    if press_blocked is None:
        press_mean = sum(by_name[n].value for n in press_states) / len(press_states)
    else:
        press_mean = tyre_config.placard_pressure_kpa

    fz_drive = sum(corner_weight(vehicle_config, c) for c in corners)
    r_eff = effective_rolling_radius(
        tread_mean, press_mean, corner_weight(vehicle_config, corners[0]),
        tyre_config.wheel_belt_radius_m, physics.k_z0,
        tyre_config.placard_pressure_kpa, physics.deflection_factor,
    )
    ceiling_nm = mu * fz_drive * friction.safety_factor * r_eff

    caveats = [
        "wet-friction curve is UNVALIDATED: assumed shape, no wet-braking or "
        "wet-cornering test data behind it",
        f"computed at the {_SIGMA_MULTIPLIER:.0f}-sigma lower tread bound "
        f"({tread_lcb:.2f} mm), not the point estimate ({tread_mean:.2f} mm)",
        f"drive layout {vehicle_config.drive_layout.value}: corners {', '.join(corners)}",
        "static corner load only; no dynamic load transfer is modelled, and "
        "load transfer is largest exactly when traction limits bind",
    ]
    if press_blocked is not None:
        caveats.append(
            f"drive-axle pressure not OBSERVED ({press_blocked}); rolling "
            f"radius computed at placard pressure instead"
        )

    return (
        Decision.ok(
            name=f"torque_ceiling_{condition}_nm", value=ceiling_nm, unit="N.m",
            basis=tread_states + press_states, caveats=caveats,
        ),
        Decision.ok(
            name="drive_tread_lower_bound_mm", value=tread_lcb, unit="mm",
            basis=tread_states,
            caveats=(f"{_SIGMA_MULTIPLIER:.0f}-sigma lower bound; the number "
                     f"the ceiling is actually computed from",),
        ),
        Decision.ok(
            name=f"peak_friction_{condition}", value=mu, unit="",
            basis=tread_states,
            caveats=("UNVALIDATED friction model",),
        ),
    )
