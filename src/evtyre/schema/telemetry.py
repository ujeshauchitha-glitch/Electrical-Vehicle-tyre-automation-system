"""The Phase 1 -> Phase 2 data contract (see CLAUDE.md section 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from .common import CORNERS, SensorReading

# Where a TelemetryFrame came from. "simulated" exists so a synthetic frame can
# never be mistaken for real-vehicle data downstream - see CLAUDE.md section 8.
# "replay" is logged/recorded real (or bench) data being replayed, not live.
FrameSource = Literal["real", "replay", "simulated"]


@dataclass(frozen=True)
class TelemetryFrame:
    """One normalized, timestamped telemetry sample.

    All values are in SI (or SI-derived) units regardless of the raw units they
    arrived in - see ingest/units.py for the conversions applied before this
    object is built. Every channel is a SensorReading, never a bare float, so a
    missing or implausible value is always explicit rather than silently absent
    or silently substituted.
    """

    timestamp_s: float
    source: FrameSource

    wheel_speed_rad_s: Mapping[str, SensorReading]  # per corner, keyed by CORNERS
    tpms_pressure_kpa: Mapping[str, SensorReading]  # per corner, gauge pressure
    tpms_temperature_c: Mapping[str, SensorReading]  # per corner

    motor_torque_nm: SensorReading
    motor_speed_rad_s: SensorReading
    accel_long_ms2: SensorReading
    ambient_temp_c: SensorReading
    vehicle_speed_ms: SensorReading
    odometer_km: SensorReading

    def __post_init__(self) -> None:
        for name, per_corner in (
            ("wheel_speed_rad_s", self.wheel_speed_rad_s),
            ("tpms_pressure_kpa", self.tpms_pressure_kpa),
            ("tpms_temperature_c", self.tpms_temperature_c),
        ):
            missing_keys = set(CORNERS) - set(per_corner)
            if missing_keys:
                raise ValueError(
                    f"{name} is missing entries for corners {sorted(missing_keys)}; "
                    "every corner must have a SensorReading (status=MISSING is fine, "
                    "an absent key is not) so a truly absent sensor is still explicit"
                )
