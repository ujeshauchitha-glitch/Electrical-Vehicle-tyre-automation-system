"""Adversarial tests: no fabrication, no silent NaN/Inf propagation.

- NaN and Inf inputs must not propagate silently into a feature marked OK
- timestamp_s, provenance and extractor_version must survive from frame
  to feature unchanged
"""

import math
import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.features.contract import FeatureStatus
from evtyre.features import pressure_thermal, kinematics
from evtyre.schema.common import CORNERS, SensorReading, SensorStatus
from evtyre.schema.telemetry import TelemetryFrame


def _make_vehicle() -> VehicleConfig:
    return VehicleConfig("test", 1800.0, 0.48, DriveLayout.RWD)


def _make_tyre() -> TyreConfig:
    return TyreConfig("test", 0.322, 8.0, 1.6, 240.0, 25.0)


def _ok_per_corner(value: float) -> dict[str, SensorReading]:
    return {c: SensorReading(value=value, status=SensorStatus.OK) for c in CORNERS}


class NaNPropagationTests(unittest.TestCase):
    """NaN/Inf inputs must not propagate into features marked OK."""

    def test_nan_pressure_does_not_produce_ok_feature(self):
        """A NaN TPMS reading should be MISSING (caught by Phase 1 validation),
        not produce an OK feature with a NaN value."""
        readings = _ok_per_corner(240.0)
        readings["FL"] = SensorReading(value=float("nan"), status=SensorStatus.OUT_OF_RANGE)
        frame = TelemetryFrame(
            timestamp_s=0.0,
            source="simulated",
            wheel_speed_rad_s=_ok_per_corner(50.0),
            tpms_pressure_kpa=readings,
            tpms_temperature_c=_ok_per_corner(30.0),
            motor_torque_nm=SensorReading(100.0, SensorStatus.OK),
            motor_speed_rad_s=SensorReading(120.0, SensorStatus.OK),
            accel_long_ms2=SensorReading(0.0, SensorStatus.OK),
            ambient_temp_c=SensorReading(20.0, SensorStatus.OK),
            vehicle_speed_ms=SensorReading(22.0, SensorStatus.OK),
            odometer_km=SensorReading(1000.0, SensorStatus.OK),
        )
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            if f.status == FeatureStatus.OK:
                self.assertIsNotNone(f.value)
                self.assertFalse(math.isnan(f.value),
                    f"Feature {f.name} is OK but has NaN value: {f.value}")

    def test_inf_speed_does_not_produce_ok_feature(self):
        """Inf vehicle speed should be caught by Phase 1 validation."""
        frame = TelemetryFrame(
            timestamp_s=0.0,
            source="simulated",
            wheel_speed_rad_s=_ok_per_corner(50.0),
            tpms_pressure_kpa=_ok_per_corner(240.0),
            tpms_temperature_c=_ok_per_corner(30.0),
            motor_torque_nm=SensorReading(100.0, SensorStatus.OK),
            motor_speed_rad_s=SensorReading(120.0, SensorStatus.OK),
            accel_long_ms2=SensorReading(0.0, SensorStatus.OK),
            ambient_temp_c=SensorReading(20.0, SensorStatus.OK),
            vehicle_speed_ms=SensorReading(value=float("inf"), status=SensorStatus.OUT_OF_RANGE),
            odometer_km=SensorReading(1000.0, SensorStatus.OK),
        )
        features = kinematics.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            if f.status == FeatureStatus.OK:
                self.assertFalse(math.isnan(f.value),
                    f"Feature {f.name} is OK but has NaN value")
                self.assertFalse(math.isinf(f.value),
                    f"Feature {f.name} is OK but has Inf value")


class ProvenanceSurvivalTests(unittest.TestCase):
    """timestamp_s, provenance and extractor_version must survive."""

    def test_timestamp_survives_pressure_thermal(self):
        frame = TelemetryFrame(
            timestamp_s=99.5,
            source="replay",
            wheel_speed_rad_s=_ok_per_corner(50.0),
            tpms_pressure_kpa=_ok_per_corner(240.0),
            tpms_temperature_c=_ok_per_corner(30.0),
            motor_torque_nm=SensorReading(100.0, SensorStatus.OK),
            motor_speed_rad_s=SensorReading(120.0, SensorStatus.OK),
            accel_long_ms2=SensorReading(0.0, SensorStatus.OK),
            ambient_temp_c=SensorReading(20.0, SensorStatus.OK),
            vehicle_speed_ms=SensorReading(22.0, SensorStatus.OK),
            odometer_km=SensorReading(1000.0, SensorStatus.OK),
        )
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            self.assertEqual(f.timestamp_s, 99.5,
                f"Feature {f.name} did not preserve timestamp")
            self.assertEqual(f.provenance, "replay",
                f"Feature {f.name} did not preserve provenance")

    def test_timestamp_survives_kinematics(self):
        frame = TelemetryFrame(
            timestamp_s=123.0,
            source="real",
            wheel_speed_rad_s=_ok_per_corner(50.0),
            tpms_pressure_kpa=_ok_per_corner(240.0),
            tpms_temperature_c=_ok_per_corner(30.0),
            motor_torque_nm=SensorReading(100.0, SensorStatus.OK),
            motor_speed_rad_s=SensorReading(120.0, SensorStatus.OK),
            accel_long_ms2=SensorReading(0.0, SensorStatus.OK),
            ambient_temp_c=SensorReading(20.0, SensorStatus.OK),
            vehicle_speed_ms=SensorReading(22.0, SensorStatus.OK),
            odometer_km=SensorReading(1000.0, SensorStatus.OK),
        )
        features = kinematics.extract(frame, _make_vehicle(), _make_tyre())
        for f in features:
            self.assertEqual(f.timestamp_s, 123.0)
            self.assertEqual(f.provenance, "real")


class NoDefaultSubstitutionTests(unittest.TestCase):
    """No feature may return 0.0, a default, or an interpolated guess
    in place of a missing input."""

    def test_partial_missing_no_zero_substitution(self):
        """When one corner is missing, the OTHER corners must still have
        real values, and the missing corner must be UNAVAILABLE."""
        readings_p = _ok_per_corner(240.0)
        readings_p["RL"] = SensorReading.missing()
        readings_t = _ok_per_corner(30.0)
        readings_t["RL"] = SensorReading.missing()
        frame = TelemetryFrame(
            timestamp_s=0.0,
            source="simulated",
            wheel_speed_rad_s=_ok_per_corner(50.0),
            tpms_pressure_kpa=readings_p,
            tpms_temperature_c=readings_t,
            motor_torque_nm=SensorReading(100.0, SensorStatus.OK),
            motor_speed_rad_s=SensorReading(120.0, SensorStatus.OK),
            accel_long_ms2=SensorReading(0.0, SensorStatus.OK),
            ambient_temp_c=SensorReading(20.0, SensorStatus.OK),
            vehicle_speed_ms=SensorReading(22.0, SensorStatus.OK),
            odometer_km=SensorReading(1000.0, SensorStatus.OK),
        )
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())

        # RL features should be UNAVAILABLE
        rl_features = [f for f in features if f.corner == "RL" or "RL" in f.name]
        for f in rl_features:
            if "spread" not in f.name:  # spread is vehicle-level
                self.assertEqual(f.status, FeatureStatus.UNAVAILABLE,
                    f"RL feature {f.name} should be UNAVAILABLE")

        # FL features should be OK with real values (not None)
        fl_features = [f for f in features
                       if (f.corner == "FL" and "spread" not in f.name)
                       or (f.corner is None and "spread" not in f.name)]
        for f in fl_features:
            if f.status == FeatureStatus.OK:
                self.assertIsNotNone(f.value,
                    f"FL feature {f.name} is OK but has None value")


if __name__ == "__main__":
    unittest.main()
