"""Adversarial tests: all-sensors-missing frame.

For every feature module, a frame with all sensors MISSING must produce
features that are ALL UNAVAILABLE with non-empty reasons, and never a number.
No feature may return 0.0, NaN, or a default in place of a missing input.
"""

import math
import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.features.contract import FeatureStatus
from evtyre.features import pressure_thermal, kinematics, resonance
from evtyre.schema.common import CORNERS, SensorReading, SensorStatus
from evtyre.schema.telemetry import TelemetryFrame


def _make_vehicle() -> VehicleConfig:
    return VehicleConfig("test", 1800.0, 0.48, DriveLayout.RWD)


def _make_tyre() -> TyreConfig:
    return TyreConfig("test", 0.322, 8.0, 1.6, 240.0, 25.0)


def _all_missing_frame() -> TelemetryFrame:
    """Frame where every sensor is MISSING."""
    missing_corner = {c: SensorReading.missing() for c in CORNERS}
    return TelemetryFrame(
        timestamp_s=42.0,
        source="simulated",
        wheel_speed_rad_s=missing_corner,
        tpms_pressure_kpa=missing_corner,
        tpms_temperature_c=missing_corner,
        motor_torque_nm=SensorReading.missing(),
        motor_speed_rad_s=SensorReading.missing(),
        accel_long_ms2=SensorReading.missing(),
        ambient_temp_c=SensorReading.missing(),
        vehicle_speed_ms=SensorReading.missing(),
        odometer_km=SensorReading.missing(),
    )


class AllMissingPressureThermalTests(unittest.TestCase):
    def test_all_features_are_unavailable(self):
        frame = _all_missing_frame()
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            with self.subTest(name=f.name):
                self.assertEqual(
                    f.status, FeatureStatus.UNAVAILABLE,
                    f"Feature {f.name} should be UNAVAILABLE but has "
                    f"status {f.status} and value {f.value}"
                )

    def test_no_feature_returns_a_number(self):
        frame = _all_missing_frame()
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            with self.subTest(name=f.name):
                self.assertIsNone(
                    f.value,
                    f"Feature {f.name} has value {f.value!r} — "
                    f"a missing-input feature must not return a number"
                )

    def test_all_unavailable_reasons_are_nonempty(self):
        frame = _all_missing_frame()
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            with self.subTest(name=f.name):
                self.assertTrue(
                    f.unavailable_reason,
                    f"Feature {f.name} is UNAVAILABLE but has no reason"
                )

    def test_no_zero_or_nan_values(self):
        """A missing input must never produce 0.0 or NaN."""
        frame = _all_missing_frame()
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            with self.subTest(name=f.name):
                if f.value is not None:
                    self.assertNotEqual(f.value, 0.0,
                        f"Feature {f.name} has value 0.0 — likely fabricated")
                    self.assertFalse(math.isnan(f.value),
                        f"Feature {f.name} has NaN value")


class AllMissingKinematicsTests(unittest.TestCase):
    def test_all_features_are_unavailable(self):
        frame = _all_missing_frame()
        features = kinematics.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            with self.subTest(name=f.name):
                self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)

    def test_no_feature_returns_a_number(self):
        frame = _all_missing_frame()
        features = kinematics.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            with self.subTest(name=f.name):
                self.assertIsNone(f.value)

    def test_all_unavailable_reasons_are_nonempty(self):
        frame = _all_missing_frame()
        features = kinematics.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            with self.subTest(name=f.name):
                self.assertTrue(f.unavailable_reason)


class ResonanceAlwaysRaisesTests(unittest.TestCase):
    def test_raises_not_implemented(self):
        frame = _all_missing_frame()
        with self.assertRaises(NotImplementedError):
            resonance.extract(frame, _make_vehicle(), _make_tyre())


if __name__ == "__main__":
    unittest.main()
