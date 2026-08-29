import math
import unittest

from evtyre.ingest import ReplayTelemetrySource, RawTelemetrySample, build_telemetry_frame
from evtyre.ingest.units import fahrenheit_to_celsius, g_to_ms2, kmh_to_ms, psi_to_kpa, rpm_to_rad_s
from evtyre.ingest.validation import Range, validate_scalar
from evtyre.schema.common import CORNERS, SensorStatus


class UnitConversionTests(unittest.TestCase):
    def test_rpm_to_rad_s(self):
        self.assertAlmostEqual(rpm_to_rad_s(60.0), 2.0 * math.pi, places=9)

    def test_psi_to_kpa(self):
        self.assertAlmostEqual(psi_to_kpa(1.0), 6.894757293168361, places=9)

    def test_fahrenheit_to_celsius(self):
        self.assertAlmostEqual(fahrenheit_to_celsius(32.0), 0.0, places=9)
        self.assertAlmostEqual(fahrenheit_to_celsius(212.0), 100.0, places=9)

    def test_g_to_ms2(self):
        self.assertAlmostEqual(g_to_ms2(1.0), 9.80665, places=9)

    def test_kmh_to_ms(self):
        self.assertAlmostEqual(kmh_to_ms(36.0), 10.0, places=9)

    def test_none_passes_through_as_none(self):
        self.assertIsNone(rpm_to_rad_s(None))
        self.assertIsNone(psi_to_kpa(None))
        self.assertIsNone(fahrenheit_to_celsius(None))
        self.assertIsNone(g_to_ms2(None))
        self.assertIsNone(kmh_to_ms(None))


class ValidateScalarTests(unittest.TestCase):
    def setUp(self):
        self.limits = Range(0.0, 100.0)

    def test_none_is_missing(self):
        reading = validate_scalar(None, self.limits)
        self.assertEqual(reading.status, SensorStatus.MISSING)
        self.assertIsNone(reading.value)

    def test_nan_is_missing_not_out_of_range(self):
        reading = validate_scalar(float("nan"), self.limits)
        self.assertEqual(reading.status, SensorStatus.MISSING)

    def test_in_range_is_ok(self):
        reading = validate_scalar(50.0, self.limits)
        self.assertEqual(reading.status, SensorStatus.OK)
        self.assertEqual(reading.value, 50.0)

    def test_below_range_is_out_of_range_but_keeps_value(self):
        reading = validate_scalar(-5.0, self.limits)
        self.assertEqual(reading.status, SensorStatus.OUT_OF_RANGE)
        self.assertEqual(reading.value, -5.0)

    def test_above_range_is_out_of_range(self):
        reading = validate_scalar(1000.0, self.limits)
        self.assertEqual(reading.status, SensorStatus.OUT_OF_RANGE)

    def test_boundary_values_are_ok(self):
        self.assertEqual(validate_scalar(0.0, self.limits).status, SensorStatus.OK)
        self.assertEqual(validate_scalar(100.0, self.limits).status, SensorStatus.OK)


class BuildTelemetryFrameTests(unittest.TestCase):
    def test_full_sample_produces_all_ok_readings(self):
        raw = RawTelemetrySample(
            timestamp_s=123.0,
            wheel_speed_rpm={c: 300.0 for c in CORNERS},
            tpms_pressure_psi={c: 34.8 for c in CORNERS},  # ~240 kPa
            tpms_temperature_f={c: 77.0 for c in CORNERS},  # 25 C
            motor_torque_nm=150.0,
            motor_speed_rpm=1200.0,
            accel_long_g=0.1,
            ambient_temp_c=20.0,
            vehicle_speed_kmh=79.2,  # ~22 m/s
            odometer_km=15000.0,
        )
        frame = build_telemetry_frame(raw, source="simulated")

        for corner in CORNERS:
            self.assertTrue(frame.wheel_speed_rad_s[corner].is_usable)
            self.assertTrue(frame.tpms_pressure_kpa[corner].is_usable)
            self.assertTrue(frame.tpms_temperature_c[corner].is_usable)
        self.assertTrue(frame.motor_torque_nm.is_usable)
        self.assertAlmostEqual(frame.vehicle_speed_ms.value, 22.0, places=2)
        self.assertEqual(frame.source, "simulated")

    def test_missing_channel_stays_missing_not_fabricated(self):
        raw = RawTelemetrySample(
            timestamp_s=0.0,
            wheel_speed_rpm={"FL": 300.0, "FR": 300.0, "RL": 300.0},  # RR omitted
            tpms_pressure_psi={c: 34.8 for c in CORNERS},
            tpms_temperature_f={c: 77.0 for c in CORNERS},
            motor_torque_nm=None,  # explicitly missing
        )
        frame = build_telemetry_frame(raw, source="replay")

        self.assertEqual(frame.wheel_speed_rad_s["RR"].status, SensorStatus.MISSING)
        self.assertIsNone(frame.wheel_speed_rad_s["RR"].value)
        self.assertEqual(frame.motor_torque_nm.status, SensorStatus.MISSING)
        # channels that were present must still be usable - one missing field
        # must not contaminate the rest of the frame
        self.assertTrue(frame.wheel_speed_rad_s["FL"].is_usable)

    def test_all_defaulted_raw_sample_produces_an_entirely_missing_frame(self):
        # RawTelemetrySample's own defaults (None / empty dict) are the last
        # line of defence against fabrication: if one of them were ever
        # accidentally changed to a real-looking number (e.g. 0.0), this is
        # the test that would catch it. No field is supplied here at all.
        raw = RawTelemetrySample(timestamp_s=42.0)
        frame = build_telemetry_frame(raw, source="simulated")

        for corner in CORNERS:
            self.assertEqual(frame.wheel_speed_rad_s[corner].status, SensorStatus.MISSING)
            self.assertIsNone(frame.wheel_speed_rad_s[corner].value)
            self.assertEqual(frame.tpms_pressure_kpa[corner].status, SensorStatus.MISSING)
            self.assertEqual(frame.tpms_temperature_c[corner].status, SensorStatus.MISSING)

        for scalar_reading in (
            frame.motor_torque_nm,
            frame.motor_speed_rad_s,
            frame.accel_long_ms2,
            frame.ambient_temp_c,
            frame.vehicle_speed_ms,
            frame.odometer_km,
        ):
            self.assertEqual(scalar_reading.status, SensorStatus.MISSING)
            self.assertIsNone(scalar_reading.value)

        self.assertEqual(frame.timestamp_s, 42.0)

    def test_implausible_value_is_flagged_out_of_range_not_corrected(self):
        raw = RawTelemetrySample(
            timestamp_s=0.0,
            wheel_speed_rpm={c: 300.0 for c in CORNERS},
            tpms_pressure_psi={c: 34.8 for c in CORNERS},
            tpms_temperature_f={c: 77.0 for c in CORNERS},
            odometer_km=-10.0,  # physically impossible
        )
        frame = build_telemetry_frame(raw, source="simulated")
        self.assertEqual(frame.odometer_km.status, SensorStatus.OUT_OF_RANGE)
        self.assertEqual(frame.odometer_km.value, -10.0)


class ReplayTelemetrySourceTests(unittest.TestCase):
    def test_replays_samples_in_order_then_returns_none(self):
        samples = [
            RawTelemetrySample(timestamp_s=0.0, motor_torque_nm=10.0),
            RawTelemetrySample(timestamp_s=1.0, motor_torque_nm=20.0),
        ]
        source = ReplayTelemetrySource(samples)

        first = source.read_frame()
        second = source.read_frame()
        third = source.read_frame()

        self.assertEqual(first.timestamp_s, 0.0)
        self.assertEqual(first.source, "replay")
        self.assertEqual(second.timestamp_s, 1.0)
        self.assertIsNone(third)  # exhausted - must not fabricate a frame

    def test_source_label_can_be_overridden_to_real(self):
        # e.g. replaying a logged real-vehicle capture rather than simulated data
        source = ReplayTelemetrySource([RawTelemetrySample(timestamp_s=0.0)], label="real")
        frame = source.read_frame()
        self.assertEqual(frame.source, "real")


if __name__ == "__main__":
    unittest.main()
