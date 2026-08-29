"""The raw (pre-normalization) input shape for Phase 1.

This dataclass only fixes the shape a real acquisition backend must hand off -
it says nothing about how those values were obtained. Building the actual live
CAN bus reader / DBC decoder that produces one of these from vehicle hardware is
explicitly out of scope for this phase (see CLAUDE.md section 9); a log-replay
or bench-data reader can already produce this shape today without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class RawTelemetrySample:
    """Already-decoded raw values, in their native (non-SI) units.

    A missing signal is represented as None, or - for a per-corner channel - by
    simply omitting that corner's key (validation.py treats an absent key the
    same as an explicit None). This class does not enforce corner-completeness
    itself; that happens during validation, not here.
    """

    timestamp_s: float

    wheel_speed_rpm: Mapping[str, Optional[float]] = field(default_factory=dict)
    tpms_pressure_psi: Mapping[str, Optional[float]] = field(default_factory=dict)
    tpms_temperature_f: Mapping[str, Optional[float]] = field(default_factory=dict)

    motor_torque_nm: Optional[float] = None
    motor_speed_rpm: Optional[float] = None
    accel_long_g: Optional[float] = None
    ambient_temp_c: Optional[float] = None
    vehicle_speed_kmh: Optional[float] = None
    odometer_km: Optional[float] = None
