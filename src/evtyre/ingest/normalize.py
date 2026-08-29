"""Raw -> TelemetryFrame: unit normalization plus range validation.

This is the core of Phase 1: it never fabricates a value. Every field it
cannot turn into a plausible SI reading becomes an explicit MISSING or
OUT_OF_RANGE SensorReading, never a guessed, interpolated, or default number
(see CLAUDE.md section 8, rule 2).
"""

from __future__ import annotations

from . import units
from ..schema.telemetry import FrameSource, TelemetryFrame
from .raw import RawTelemetrySample
from .validation import DEFAULT_LIMITS, ValidationLimits, validate_per_corner, validate_scalar


def build_telemetry_frame(
    raw: RawTelemetrySample,
    source: FrameSource,
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> TelemetryFrame:
    wheel_speed_rad_s = validate_per_corner(
        {corner: units.rpm_to_rad_s(rpm) for corner, rpm in raw.wheel_speed_rpm.items()},
        limits.wheel_speed_rad_s,
    )
    tpms_pressure_kpa = validate_per_corner(
        {corner: units.psi_to_kpa(psi) for corner, psi in raw.tpms_pressure_psi.items()},
        limits.tpms_pressure_kpa,
    )
    tpms_temperature_c = validate_per_corner(
        {
            corner: units.fahrenheit_to_celsius(fahrenheit)
            for corner, fahrenheit in raw.tpms_temperature_f.items()
        },
        limits.tpms_temperature_c,
    )

    return TelemetryFrame(
        timestamp_s=raw.timestamp_s,
        source=source,
        wheel_speed_rad_s=wheel_speed_rad_s,
        tpms_pressure_kpa=tpms_pressure_kpa,
        tpms_temperature_c=tpms_temperature_c,
        motor_torque_nm=validate_scalar(raw.motor_torque_nm, limits.motor_torque_nm),
        motor_speed_rad_s=validate_scalar(
            units.rpm_to_rad_s(raw.motor_speed_rpm), limits.motor_speed_rad_s
        ),
        accel_long_ms2=validate_scalar(units.g_to_ms2(raw.accel_long_g), limits.accel_long_ms2),
        ambient_temp_c=validate_scalar(raw.ambient_temp_c, limits.ambient_temp_c),
        vehicle_speed_ms=validate_scalar(
            units.kmh_to_ms(raw.vehicle_speed_kmh), limits.vehicle_speed_ms
        ),
        odometer_km=validate_scalar(raw.odometer_km, limits.odometer_km),
    )
