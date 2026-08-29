"""Vehicle-body configuration schema.

Split out from TyreConfig (tyre.py) because tyres get replaced independently of
the vehicle they're fitted to - a real deployment will pair one VehicleConfig
with a TyreConfig that can change over the vehicle's life.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DriveLayout(str, Enum):
    """Which axle(s) receive drive torque.

    legacy/ev_tyre_fusion.py's torque_limit() hardcodes a rear-drive axle
    (Fz("RL") + Fz("RR")). This field exists so a later phase can generalize
    that instead of repeating the same assumption - Phase 1 does not use it
    itself, it's defined now because config schemas are shared across all
    phases (see CLAUDE.md section 4).
    """

    FWD = "fwd"
    RWD = "rwd"
    AWD = "awd"


@dataclass(frozen=True)
class VehicleConfig:
    """Identity and body-level physical parameters for one vehicle.

    These are a configuration SHAPE, not validated physics constants. Later
    phases (state estimation, fusion) will read physical parameters from an
    instance of this class instead of legacy's hardcoded `Vehicle` class
    attributes. Populating real, bench/fleet-validated numbers for an actual
    vehicle is out of scope here - see CLAUDE.md sections 6 and 7.
    """

    vehicle_id: str
    mass_kg: float
    front_weight_fraction: float
    drive_layout: DriveLayout

    def __post_init__(self) -> None:
        if not self.vehicle_id:
            raise ValueError("vehicle_id must be non-empty")
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        if not 0.0 < self.front_weight_fraction < 1.0:
            raise ValueError("front_weight_fraction must be strictly between 0 and 1")
