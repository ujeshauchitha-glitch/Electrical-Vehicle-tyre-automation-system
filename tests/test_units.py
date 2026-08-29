"""Adversarial tests: unit correctness, boundary values, negative temperature.

- Assert declared units against computed magnitudes
- Boundary values: zero speed, negative temperature, pressure at zero
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
    return TyreConfig("test", 0.322, 8.0, 1.6, 240.0)


def _ok_per_corner(value: float) -> dict[str, SensorReading]:
    return {c: SensorReading(value=value, status=SensorStatus.OK) for c in CORNERS}


def _make_frame(**overrides) -> TelemetryFrame:
    defaults = dict(
        timestamp_s=0.0,
        source="simulated",
        wheel_speed_rad_s=_ok_per_corner(50.0),
        tpms_pressure_kpa=_ok_per_corner(240.0),
        tpms_temperature_c=_ok_per_corner(30.0),
        motor_torque_nm=SensorReading(100.0, SensorStatus.OK),
        motor_speed_rad_s=SensorReading(120.0, SensorStatus.OK),
        accel_long_ms2=SensorReading(0.5, SensorStatus.OK),
        ambient_temp_c=SensorReading(20.0, SensorStatus.OK),
        vehicle_speed_ms=SensorReading(22.0, SensorStatus.OK),
        odometer_km=SensorReading(1000.0, SensorStatus.OK),
    )
    defaults.update(overrides)
    return TelemetryFrame(**defaults)


def _by_name(features, name):
    return [f for f in features if f.name == name]


class PressureUnitTests(unittest.TestCase):
    """Declared units must match computed magnitudes."""

    def test_running_pressure_in_pa_not_kpa(self):
        """running_pressure_pa should be in Pa, not kPa."""
        frame = _make_frame(
            tpms_pressure_kpa=_ok_per_corner(240.0),
        )
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        f = _by_name(features, "running_pressure_pa_FL")[0]
        self.assertEqual(f.unit, "Pa")
        # 240 kPa gauge → 241325 Pa absolute
        # If someone accidentally returned 240 (kPa), this would fail
        self.assertGreater(f.value, 100_000,
            "running_pressure_pa value looks like kPa, not Pa")

    def test_cold_equivalent_in_pa(self):
        frame = _make_frame()
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        f = _by_name(features, "cold_equivalent_pressure_pa_FL")[0]
        self.assertEqual(f.unit, "Pa")
        self.assertGreater(f.value, 100_000)

    def test_pressure_deviation_in_pa(self):
        """pressure_deviation should be in Pa, not kPa."""
        frame = _make_frame(
            tpms_pressure_kpa=_ok_per_corner(260.0),
        )
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        f = _by_name(features, "pressure_deviation_from_placard_FL")[0]
        self.assertEqual(f.unit, "Pa")
        # (260 - 240) kPa = 20 kPa = 20000 Pa
        self.assertAlmostEqual(f.value, 20_000.0)
        # If someone returned 20 (kPa), this would fail

    def test_temperature_in_celsius(self):
        frame = _make_frame()
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        f = _by_name(features, "tyre_temperature_c_FL")[0]
        self.assertEqual(f.unit, "°C")
        self.assertAlmostEqual(f.value, 30.0)


class BoundaryValueTests(unittest.TestCase):
    """Boundary values: zero speed, negative temperature, pressure at zero."""

    def test_zero_vehicle_speed_kinematics(self):
        frame = _make_frame(
            vehicle_speed_ms=SensorReading(value=0.0, status=SensorStatus.OK),
        )
        features = kinematics.extract(frame, _make_vehicle(), _make_tyre())
        # Slip ratio should be UNAVAILABLE at zero speed
        for corner in CORNERS:
            f = _by_name(features, f"slip_ratio_{corner}")[0]
            self.assertEqual(f.status, FeatureStatus.UNAVAILABLE,
                f"slip_ratio should be UNAVAILABLE at zero speed")

    def test_negative_temperature_handled(self):
        """Negative ambient temperature should not break extraction."""
        frame = _make_frame(
            ambient_temp_c=SensorReading(value=-10.0, status=SensorStatus.OK),
            tpms_temperature_c=_ok_per_corner(-5.0),
        )
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        # Temperature rise: -5 - (-10) = 5 °C
        f = _by_name(features, "temperature_rise_above_ambient_FL")[0]
        self.assertEqual(f.status, FeatureStatus.OK)
        self.assertAlmostEqual(f.value, 5.0)

    def test_very_low_pressure_not_crash(self):
        """Low pressure should still produce features."""
        frame = _make_frame(
            tpms_pressure_kpa=_ok_per_corner(60.0),  # very low but in range
        )
        features = pressure_thermal.extract(frame, _make_vehicle(), _make_tyre())
        f = _by_name(features, "running_pressure_pa_FL")[0]
        self.assertEqual(f.status, FeatureStatus.OK)
        self.assertGreater(f.value, 0)


class KinematicsUnitTests(unittest.TestCase):
    """Unit correctness for kinematic features."""

    def test_slip_ratio_is_dimensionless(self):
        frame = _make_frame()
        features = kinematics.extract(frame, _make_vehicle(), _make_tyre())
        f = _by_name(features, "slip_ratio_FL")[0]
        self.assertEqual(f.unit, "")

    def test_rolling_radius_ratio_is_dimensionless(self):
        frame = _make_frame()
        features = kinematics.extract(frame, _make_vehicle(), _make_tyre())
        f = _by_name(features, "effective_rolling_radius_ratio_FL")[0]
        self.assertEqual(f.unit, "")

    def test_axle_speed_ratio_is_dimensionless(self):
        frame = _make_frame()
        features = kinematics.extract(frame, _make_vehicle(), _make_tyre())
        f = _by_name(features, "axle_speed_ratio_front")[0]
        self.assertEqual(f.unit, "")


if __name__ == "__main__":
    unittest.main()
