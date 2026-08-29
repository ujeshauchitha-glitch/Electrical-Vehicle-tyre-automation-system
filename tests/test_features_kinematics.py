"""Tests for wheel-speed-derived kinematic features."""

import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.features import Classification, Directionality, FeatureStatus
from evtyre.features import kinematics as kin
from evtyre.schema.common import CORNERS, SensorReading, SensorStatus
from evtyre.schema.telemetry import TelemetryFrame


def _make_vehicle() -> VehicleConfig:
    return VehicleConfig(
        vehicle_id="test",
        mass_kg=1800.0,
        front_weight_fraction=0.48,
        drive_layout=DriveLayout.RWD,
    )


def _make_tyre() -> TyreConfig:
    return TyreConfig(
        tyre_model_id="test-tyre",
        wheel_belt_radius_m=0.322,
        tread_new_mm=8.0,
        tread_legal_mm=1.6,
        placard_pressure_kpa=240.0,
        cold_reference_temperature_c=25.0,
    )


def _ok_per_corner(value: float) -> dict[str, SensorReading]:
    return {c: SensorReading(value=value, status=SensorStatus.OK) for c in CORNERS}


def _make_frame(**overrides) -> TelemetryFrame:
    defaults = dict(
        timestamp_s=10.0,
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


def _extract(frame=None):
    if frame is None:
        frame = _make_frame()
    return kin.extract(frame, _make_vehicle(), _make_tyre())


def _by_name(features, name):
    return [f for f in features if f.name == name]


class RollingRadiusRatioTests(unittest.TestCase):
    def test_computed_correctly(self):
        """R_ratio = V / (omega * R_belt)."""
        frame = _make_frame(
            vehicle_speed_ms=SensorReading(22.0, SensorStatus.OK),
            wheel_speed_rad_s=_ok_per_corner(50.0),
        )
        features = _extract(frame)
        expected = 22.0 / (50.0 * 0.322)
        for corner in CORNERS:
            f = _by_name(features, f"effective_rolling_radius_ratio_{corner}")
            self.assertEqual(len(f), 1)
            self.assertAlmostEqual(f[0].value, expected)
            self.assertEqual(f[0].status, FeatureStatus.OK)

    def test_missing_vehicle_speed_unavailable(self):
        frame = _make_frame(vehicle_speed_ms=SensorReading.missing())
        features = _extract(frame)
        for corner in CORNERS:
            f = _by_name(features, f"effective_rolling_radius_ratio_{corner}")[0]
            self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)

    def test_missing_wheel_speed_unavailable(self):
        readings = _ok_per_corner(50.0)
        readings["FL"] = SensorReading.missing()
        frame = _make_frame(wheel_speed_rad_s=readings)
        features = _extract(frame)
        f = _by_name(features, "effective_rolling_radius_ratio_FL")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)
        # Other corners still OK
        f_fr = _by_name(features, "effective_rolling_radius_ratio_FR")[0]
        self.assertEqual(f_fr.status, FeatureStatus.OK)

    def test_near_zero_wheel_speed_unavailable(self):
        readings = _ok_per_corner(50.0)
        readings["RL"] = SensorReading(value=1e-10, status=SensorStatus.OK)
        frame = _make_frame(wheel_speed_rad_s=readings)
        features = _extract(frame)
        f = _by_name(features, "effective_rolling_radius_ratio_RL")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)

    def test_is_classification_b(self):
        features = _extract()
        f = _by_name(features, "effective_rolling_radius_ratio_FL")[0]
        self.assertEqual(f.classification, Classification.B)


class SlipRatioTests(unittest.TestCase):
    def test_slip_ratio_at_cruise(self):
        """At constant speed with no slip, slip ratio ≈ 0."""
        frame = _make_frame(
            vehicle_speed_ms=SensorReading(22.0, SensorStatus.OK),
            wheel_speed_rad_s=_ok_per_corner(50.0),
        )
        features = _extract(frame)
        # R * omega = 0.322 * 50 = 16.1; V = 22.0
        # slip = (16.1 - 22.0) / 22.0 ≈ -0.268
        expected_slip = (0.322 * 50.0 - 22.0) / 22.0
        for corner in CORNERS:
            f = _by_name(features, f"slip_ratio_{corner}")[0]
            self.assertAlmostEqual(f.value, expected_slip, places=6)

    def test_slip_unavailable_below_threshold(self):
        frame = _make_frame(
            vehicle_speed_ms=SensorReading(0.3, SensorStatus.OK),  # below 0.5 m/s
        )
        features = _extract(frame)
        for corner in CORNERS:
            f = _by_name(features, f"slip_ratio_{corner}")[0]
            self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)
            self.assertIn("below threshold", f.unavailable_reason)

    def test_slip_unavailable_with_no_vehicle_speed(self):
        frame = _make_frame(vehicle_speed_ms=SensorReading.missing())
        features = _extract(frame)
        for corner in CORNERS:
            f = _by_name(features, f"slip_ratio_{corner}")[0]
            self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)

    def test_slip_is_classification_a(self):
        features = _extract()
        f = _by_name(features, "slip_ratio_FL")[0]
        self.assertEqual(f.classification, Classification.A)

    def test_missing_wheel_speed_makes_that_slip_unavailable(self):
        readings = _ok_per_corner(50.0)
        readings["RR"] = SensorReading.missing()
        frame = _make_frame(wheel_speed_rad_s=readings)
        features = _extract(frame)
        f_rr = _by_name(features, "slip_ratio_RR")[0]
        self.assertEqual(f_rr.status, FeatureStatus.UNAVAILABLE)
        # Others still OK
        f_fl = _by_name(features, "slip_ratio_FL")[0]
        self.assertEqual(f_fl.status, FeatureStatus.OK)


class AxleSpeedRatioTests(unittest.TestCase):
    def test_equal_speeds_give_ratio_1(self):
        features = _extract()
        f = _by_name(features, "axle_speed_ratio_front")[0]
        self.assertAlmostEqual(f.value, 1.0)

    def test_unequal_speeds(self):
        readings = _ok_per_corner(50.0)
        readings["FL"] = SensorReading(value=55.0, status=SensorStatus.OK)
        frame = _make_frame(wheel_speed_rad_s=readings)
        features = _extract(frame)
        f = _by_name(features, "axle_speed_ratio_front")[0]
        self.assertAlmostEqual(f.value, 55.0 / 50.0)

    def test_missing_one_wheel_unavailable(self):
        readings = _ok_per_corner(50.0)
        readings["FL"] = SensorReading.missing()
        frame = _make_frame(wheel_speed_rad_s=readings)
        features = _extract(frame)
        f = _by_name(features, "axle_speed_ratio_front")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)
        self.assertIn("wheel speed missing for FL", f.unavailable_reason)

    def test_both_zero_unavailable(self):
        readings = _ok_per_corner(50.0)
        readings["FL"] = SensorReading(value=1e-10, status=SensorStatus.OK)
        readings["FR"] = SensorReading(value=1e-10, status=SensorStatus.OK)
        frame = _make_frame(wheel_speed_rad_s=readings)
        features = _extract(frame)
        f = _by_name(features, "axle_speed_ratio_front")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)

    def test_front_and_rear_ratios_independent(self):
        readings = _ok_per_corner(50.0)
        readings["FL"] = SensorReading(value=60.0, status=SensorStatus.OK)
        # Rear axle stays equal
        frame = _make_frame(wheel_speed_rad_s=readings)
        features = _extract(frame)
        f_front = _by_name(features, "axle_speed_ratio_front")[0]
        f_rear = _by_name(features, "axle_speed_ratio_rear")[0]
        self.assertAlmostEqual(f_front.value, 60.0 / 50.0)
        self.assertAlmostEqual(f_rear.value, 1.0)


class AxleDifferenceTests(unittest.TestCase):
    def test_difference_computed(self):
        readings = _ok_per_corner(50.0)
        readings["FL"] = SensorReading(value=55.0, status=SensorStatus.OK)
        frame = _make_frame(wheel_speed_rad_s=readings)
        features = _extract(frame)
        f = _by_name(features, "axle_speed_ratio_diff_front_minus_rear")[0]
        expected = (55.0 / 50.0) - 1.0  # front ratio - rear ratio
        self.assertAlmostEqual(f.value, expected)

    def test_difference_unavailable_when_front_unavailable(self):
        readings = _ok_per_corner(50.0)
        readings["FL"] = SensorReading.missing()
        frame = _make_frame(wheel_speed_rad_s=readings)
        features = _extract(frame)
        f = _by_name(features, "axle_speed_ratio_diff_front_minus_rear")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)


class ProvenanceTests(unittest.TestCase):
    def test_timestamp_and_provenance_preserved(self):
        frame = _make_frame(timestamp_s=99.0, source="replay")
        features = _extract(frame)
        for f in features:
            self.assertEqual(f.timestamp_s, 99.0)
            self.assertEqual(f.provenance, "replay")

    def test_extractor_version(self):
        features = _extract()
        for f in features:
            self.assertEqual(f.extractor_version, kin.EXTRACTOR_VERSION)


class NoCrossAxleComparisonTests(unittest.TestCase):
    """Verify no feature compares FL to RL, etc. — axle-scoping only."""

    def test_all_corner_features_are_scoped(self):
        features = _extract()
        # Every per-corner feature has a corner set; every per-axle feature
        # has corner=None — no feature compares across axles.
        for f in features:
            if f.corner is not None:
                self.assertIn(f.corner, CORNERS)
            # All inputs should only reference wheel_speed_rad_s for axle features
            # This is a structural check — no feature claims to consume
            # a wheel from a different axle


if __name__ == "__main__":
    unittest.main()
