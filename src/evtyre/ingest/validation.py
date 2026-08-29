"""Range validation: turns a raw normalized number into an explicit SensorReading.

The bounds here are generic physical plausibility checks (catching a decoder
bug, a stuck sensor, a unit-conversion mistake), not validated per-vehicle
limits - see CLAUDE.md section 7. An in-range value is not thereby "correct",
only "not obviously implausible".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from ..schema.common import CORNERS, SensorReading, SensorStatus


@dataclass(frozen=True)
class Range:
    low: float
    high: float


@dataclass(frozen=True)
class ValidationLimits:
    """Plausible physical ranges used to flag OUT_OF_RANGE readings."""

    wheel_speed_rad_s: Range = Range(0.0, 300.0)
    tpms_pressure_kpa: Range = Range(50.0, 450.0)
    tpms_temperature_c: Range = Range(-40.0, 150.0)
    motor_torque_nm: Range = Range(-1000.0, 1000.0)
    motor_speed_rad_s: Range = Range(0.0, 3000.0)
    accel_long_ms2: Range = Range(-20.0, 20.0)
    ambient_temp_c: Range = Range(-40.0, 85.0)
    vehicle_speed_ms: Range = Range(0.0, 90.0)
    odometer_km: Range = Range(0.0, 2_000_000.0)


DEFAULT_LIMITS = ValidationLimits()


def validate_scalar(value: Optional[float], limits: Range) -> SensorReading:
    """Classify one already-unit-converted value as OK / MISSING / OUT_OF_RANGE.

    Never returns a value other than the one passed in (or None) - this
    function flags implausible/absent data, it does not correct or replace it.
    """

    if value is None:
        return SensorReading.missing()
    if not math.isfinite(value):
        # NaN/inf means "no valid data arrived", not "an implausible number
        # arrived" - treat it as missing rather than out-of-range.
        return SensorReading.missing()
    if not (limits.low <= value <= limits.high):
        return SensorReading(value=value, status=SensorStatus.OUT_OF_RANGE)
    return SensorReading(value=value, status=SensorStatus.OK)


def validate_per_corner(
    values: Mapping[str, Optional[float]], limits: Range
) -> dict[str, SensorReading]:
    """Same as validate_scalar, applied per corner. A corner absent from
    `values` is treated identically to an explicit None for that corner."""

    return {corner: validate_scalar(values.get(corner), limits) for corner in CORNERS}
