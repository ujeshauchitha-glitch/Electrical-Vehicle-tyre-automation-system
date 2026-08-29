"""Phase 4 decision-layer configuration.

Every physical constant here is a REQUIRED field with NO default, following the
Phase 2 RoadLoadParams and Phase 3 PhysicsConfig precedent. The wet-friction
constants matter most: they feed a traction limit, so a plausible-looking
default would quietly set a safety number.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.vehicle import DriveLayout, VehicleConfig


@dataclass(frozen=True)
class FrictionConfig:
    """Wet/dry friction model constants.

    UNVALIDATED. Legacy's curve shape with no wet-braking or wet-cornering test
    data behind it. CLAUDE.md lists this as one of the assumptions that must be
    validated before any number derived from it is trusted, and it is the only
    unvalidated model in this project that feeds a safety-relevant output.
    """

    mu_dry: float
    """Peak dry friction coefficient. UNVALIDATED. Legacy value: 1.00."""

    wet_floor: float
    """Wet friction as a fraction of dry at zero tread. UNVALIDATED.
    Legacy value: 0.45."""

    wet_tau: float
    """Tread depth (mm) scale over which wet friction recovers. UNVALIDATED.
    Legacy value: 2.5."""

    safety_factor: float
    """Fraction of the friction limit actually commanded. Legacy value: 0.85."""

    def __post_init__(self) -> None:
        if self.mu_dry <= 0:
            raise ValueError("mu_dry must be positive")
        if not 0.0 <= self.wet_floor <= 1.0:
            raise ValueError("wet_floor must be a fraction in [0, 1]")
        if self.wet_tau <= 0:
            raise ValueError("wet_tau must be positive")
        if not 0.0 < self.safety_factor <= 1.0:
            raise ValueError("safety_factor must be in (0, 1]")


@dataclass(frozen=True)
class MaintenanceConfig:
    """Thresholds for the maintenance view.

    These are policy, not physics — they decide when a driver is told something
    is wrong. Still required with no defaults, so the choice is explicit.
    """

    low_pressure_margin_kpa: float
    """How far below placard (cold-equivalent) counts as LOW."""

    cross_corner_spread_limit_kpa: float
    """Spread across corners above which corners are flagged as uneven."""

    def __post_init__(self) -> None:
        if self.low_pressure_margin_kpa <= 0:
            raise ValueError("low_pressure_margin_kpa must be positive")
        if self.cross_corner_spread_limit_kpa <= 0:
            raise ValueError("cross_corner_spread_limit_kpa must be positive")


def driven_corners(vehicle_config: VehicleConfig) -> tuple[str, ...]:
    """Which corners receive drive torque.

    Generalises legacy's hardcoded rear-drive assumption
    (``Fz("RL") + Fz("RR")``), which silently produced a wrong torque ceiling
    for any FWD or AWD vehicle. CLAUDE.md section 3 calls this out explicitly
    as something Phase 4 must move into configuration.
    """
    layout = vehicle_config.drive_layout
    if layout is DriveLayout.FWD:
        return ("FL", "FR")
    if layout is DriveLayout.RWD:
        return ("RL", "RR")
    if layout is DriveLayout.AWD:
        return ("FL", "FR", "RL", "RR")
    raise ValueError(f"Unhandled drive layout: {layout!r}")
