"""Tyre-model configuration schema.

Kept separate from VehicleConfig (vehicle.py) since a tyre model can be swapped
independently of the vehicle it's fitted to.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TyreConfig:
    """Identity and physical parameters for one tyre model/fitment.

    Field names deliberately echo legacy/ev_tyre_fusion.py's `Vehicle` class
    attributes (`r_belt`, `tread_new`, `tread_legal`, `p_placard`) so the mapping
    is obvious, but no numeric defaults are baked in here - every value must be
    supplied explicitly and is a configuration input, not a validated constant
    (see CLAUDE.md sections 6 and 7).
    """

    tyre_model_id: str
    wheel_belt_radius_m: float
    tread_new_mm: float
    tread_legal_mm: float
    placard_pressure_kpa: float

    def __post_init__(self) -> None:
        if not self.tyre_model_id:
            raise ValueError("tyre_model_id must be non-empty")
        if self.wheel_belt_radius_m <= 0:
            raise ValueError("wheel_belt_radius_m must be positive")
        if self.tread_legal_mm < 0:
            raise ValueError("tread_legal_mm must be >= 0")
        if self.tread_new_mm <= self.tread_legal_mm:
            raise ValueError("tread_new_mm must be greater than tread_legal_mm")
        if self.placard_pressure_kpa <= 0:
            raise ValueError("placard_pressure_kpa must be positive")
