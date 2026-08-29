import unittest

from evtyre.schema.common import CORNERS, SensorReading, SensorStatus
from evtyre.schema.telemetry import TelemetryFrame


def _ok_per_corner(value: float) -> dict:
    return {c: SensorReading(value=value, status=SensorStatus.OK) for c in CORNERS}


def _make_frame(**overrides) -> TelemetryFrame:
    defaults = dict(
        timestamp_s=0.0,
        source="simulated",
        wheel_speed_rad_s=_ok_per_corner(50.0),
        tpms_pressure_kpa=_ok_per_corner(240.0),
        tpms_temperature_c=_ok_per_corner(25.0),
        motor_torque_nm=SensorReading(100.0, SensorStatus.OK),
        motor_speed_rad_s=SensorReading(120.0, SensorStatus.OK),
        accel_long_ms2=SensorReading(0.0, SensorStatus.OK),
        ambient_temp_c=SensorReading(20.0, SensorStatus.OK),
        vehicle_speed_ms=SensorReading(22.0, SensorStatus.OK),
        odometer_km=SensorReading(1000.0, SensorStatus.OK),
    )
    defaults.update(overrides)
    return TelemetryFrame(**defaults)


class SensorReadingTests(unittest.TestCase):
    def test_missing_factory_has_no_value(self):
        reading = SensorReading.missing()
        self.assertIsNone(reading.value)
        self.assertEqual(reading.status, SensorStatus.MISSING)
        self.assertFalse(reading.is_usable)

    def test_missing_status_requires_none_value(self):
        with self.assertRaises(ValueError):
            SensorReading(value=1.0, status=SensorStatus.MISSING)

    def test_ok_status_rejects_none_value(self):
        # guards against exactly the kind of bug this schema exists to prevent:
        # a status of OK/OUT_OF_RANGE with no real value behind it would look
        # like a present-but-empty reading instead of an honest MISSING one
        with self.assertRaises(ValueError):
            SensorReading(value=None, status=SensorStatus.OK)

    def test_out_of_range_status_rejects_none_value(self):
        with self.assertRaises(ValueError):
            SensorReading(value=None, status=SensorStatus.OUT_OF_RANGE)

    def test_ok_reading_is_usable(self):
        reading = SensorReading(value=240.0, status=SensorStatus.OK)
        self.assertTrue(reading.is_usable)

    def test_out_of_range_reading_is_not_usable_but_keeps_value(self):
        reading = SensorReading(value=999.0, status=SensorStatus.OUT_OF_RANGE)
        self.assertFalse(reading.is_usable)
        self.assertEqual(reading.value, 999.0)


class TelemetryFrameTests(unittest.TestCase):
    def test_well_formed_frame_constructs(self):
        frame = _make_frame()
        self.assertEqual(frame.source, "simulated")
        self.assertEqual(set(frame.wheel_speed_rad_s), set(CORNERS))

    def test_missing_corner_key_is_rejected(self):
        incomplete = {c: SensorReading(value=50.0, status=SensorStatus.OK) for c in CORNERS if c != "RR"}
        with self.assertRaises(ValueError):
            _make_frame(wheel_speed_rad_s=incomplete)

    def test_a_corner_may_explicitly_be_missing(self):
        readings = _ok_per_corner(240.0)
        readings["FL"] = SensorReading.missing()
        frame = _make_frame(tpms_pressure_kpa=readings)
        self.assertEqual(frame.tpms_pressure_kpa["FL"].status, SensorStatus.MISSING)
        self.assertTrue(frame.tpms_pressure_kpa["FR"].is_usable)


if __name__ == "__main__":
    unittest.main()
