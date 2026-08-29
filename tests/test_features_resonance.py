"""Tests for the resonance feature extractor stub.

The resonance extractor MUST raise NotImplementedError with "G1" in the
message.  It must never return None, 0.0, or any default — a stub that
returns a value invites someone to fill it in without reopening G1.
"""

import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.features import resonance
from evtyre.schema.common import CORNERS, SensorReading, SensorStatus
from evtyre.schema.telemetry import TelemetryFrame


def _make_frame() -> TelemetryFrame:
    return TelemetryFrame(
        timestamp_s=0.0,
        source="simulated",
        wheel_speed_rad_s={c: SensorReading(value=50.0, status=SensorStatus.OK) for c in CORNERS},
        tpms_pressure_kpa={c: SensorReading(value=240.0, status=SensorStatus.OK) for c in CORNERS},
        tpms_temperature_c={c: SensorReading(value=30.0, status=SensorStatus.OK) for c in CORNERS},
        motor_torque_nm=SensorReading(100.0, SensorStatus.OK),
        motor_speed_rad_s=SensorReading(120.0, SensorStatus.OK),
        accel_long_ms2=SensorReading(0.0, SensorStatus.OK),
        ambient_temp_c=SensorReading(20.0, SensorStatus.OK),
        vehicle_speed_ms=SensorReading(22.0, SensorStatus.OK),
        odometer_km=SensorReading(1000.0, SensorStatus.OK),
    )


class ResonanceStubTests(unittest.TestCase):
    def test_raises_not_implemented_error(self):
        """The stub must raise, not return a value."""
        with self.assertRaises(NotImplementedError) as ctx:
            resonance.extract(
                _make_frame(),
                VehicleConfig("v", 1800.0, 0.48, DriveLayout.RWD),
                TyreConfig("t", 0.322, 8.0, 1.6, 240.0, 25.0),
            )
        self.assertIn("G1", str(ctx.exception))

    def test_raises_with_g1_in_message(self):
        """The word G1 must appear in the error message."""
        with self.assertRaises(NotImplementedError) as ctx:
            resonance.extract(
                _make_frame(),
                VehicleConfig("v", 1800.0, 0.48, DriveLayout.RWD),
                TyreConfig("t", 0.322, 8.0, 1.6, 240.0, 25.0),
            )
        self.assertIn("G1", str(ctx.exception))

    def test_does_not_return_any_value(self):
        """The stub must never silently return a value."""
        try:
            result = resonance.extract(
                _make_frame(),
                VehicleConfig("v", 1800.0, 0.48, DriveLayout.RWD),
                TyreConfig("t", 0.322, 8.0, 1.6, 240.0, 25.0),
            )
            # If we get here, the stub returned something instead of raising
            self.fail(
                f"Resonance stub returned {result!r} instead of raising "
                f"NotImplementedError.  A stub that returns a value "
                f"invites silent fabrication of the resonance channel."
            )
        except NotImplementedError:
            pass  # Expected

    def test_error_message_mentions_windowed_sample(self):
        """The error should point to the G1 design document."""
        with self.assertRaises(NotImplementedError) as ctx:
            resonance.extract(
                _make_frame(),
                VehicleConfig("v", 1800.0, 0.48, DriveLayout.RWD),
                TyreConfig("t", 0.322, 8.0, 1.6, 240.0, 25.0),
            )
        self.assertIn("windowed sample", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
