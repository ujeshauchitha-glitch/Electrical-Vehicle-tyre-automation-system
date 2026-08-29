"""Tests for road load feature extraction."""

import math
import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.features import Classification, Directionality, FeatureStatus
from evtyre.features.road_load import (
    EXTRACTOR_VERSION,
    RoadLoadParams,
    extract,
)
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


def _make_params() -> RoadLoadParams:
    return RoadLoadParams(
        drag_coefficient=0.25,
        frontal_area_m2=2.3,
        driveline_efficiency=0.95,
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


def _extract(frame=None, params=None, grade_rad=0.0):
    if frame is None:
        frame = _make_frame()
    if params is None:
        params = _make_params()
    return extract(frame, _make_vehicle(), _make_tyre(), road_load_params=params)


def _by_name(features, name):
    return [f for f in features if f.name == name]


class RoadLoadParamsTests(unittest.TestCase):
    def test_valid_params(self):
        p = RoadLoadParams(drag_coefficient=0.25, frontal_area_m2=2.3,
                           driveline_efficiency=0.95)
        self.assertEqual(p.drag_coefficient, 0.25)

    def test_rejects_zero_drag(self):
        with self.assertRaises(ValueError):
            RoadLoadParams(drag_coefficient=0.0, frontal_area_m2=2.3,
                           driveline_efficiency=0.95)

    def test_rejects_zero_area(self):
        with self.assertRaises(ValueError):
            RoadLoadParams(drag_coefficient=0.25, frontal_area_m2=0.0,
                           driveline_efficiency=0.95)

    def test_rejects_efficiency_zero(self):
        with self.assertRaises(ValueError):
            RoadLoadParams(drag_coefficient=0.25, frontal_area_m2=2.3,
                           driveline_efficiency=0.0)

    def test_rejects_efficiency_above_one(self):
        with self.assertRaises(ValueError):
            RoadLoadParams(drag_coefficient=0.25, frontal_area_m2=2.3,
                           driveline_efficiency=1.1)


class RollingResistanceTests(unittest.TestCase):
    def test_rolling_resistance_positive(self):
        features = _extract()
        f = _by_name(features, "rolling_resistance_force_n")[0]
        self.assertEqual(f.status, FeatureStatus.OK)
        self.assertGreater(f.value, 0)  # rolling resistance is always positive

    def test_rolling_resistance_unavailable_without_pressure(self):
        readings = {c: SensorReading.missing() for c in CORNERS}
        frame = _make_frame(tpms_pressure_kpa=readings)
        features = _extract(frame)
        f = _by_name(features, "rolling_resistance_force_n")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)

    def test_classification_c(self):
        features = _extract()
        f = _by_name(features, "rolling_resistance_force_n")[0]
        self.assertEqual(f.classification, Classification.C)


class AeroDragTests(unittest.TestCase):
    def test_aero_drag_proportional_to_v_squared(self):
        """F_aero = 0.5 * rho * CdA * v^2."""
        params = _make_params()
        cda = params.drag_coefficient * params.frontal_area_m2
        features = _extract()
        f = _by_name(features, "aerodynamic_drag_force_n")[0]
        expected = 0.5 * 1.225 * cda * 22.0 ** 2
        self.assertAlmostEqual(f.value, expected, places=2)

    def test_aero_drag_unavailable_without_speed(self):
        frame = _make_frame(vehicle_speed_ms=SensorReading.missing())
        features = _extract(frame)
        f = _by_name(features, "aerodynamic_drag_force_n")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)


class GradeResistanceTests(unittest.TestCase):
    def test_grade_is_always_unavailable(self):
        features = _extract()
        f = _by_name(features, "grade_resistance_force_n")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)
        self.assertIsNone(f.value)


class InertialForceTests(unittest.TestCase):
    def test_inertial_force(self):
        features = _extract()
        f = _by_name(features, "inertial_force_n")[0]
        self.assertAlmostEqual(f.value, 1800.0 * 0.5)

    def test_unavailable_without_accel(self):
        frame = _make_frame(accel_long_ms2=SensorReading.missing())
        features = _extract(frame)
        f = _by_name(features, "inertial_force_n")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)


class TotalRoadLoadTests(unittest.TestCase):
    def test_total_unavailable_when_grade_unavailable(self):
        features = _extract()
        total = _by_name(features, "total_road_load_force_n")[0]
        self.assertEqual(total.status, FeatureStatus.UNAVAILABLE)
        self.assertIsNone(total.value)


class RoadLoadCoefficientTests(unittest.TestCase):
    def test_coefficient_computed(self):
        features = _extract()
        f = _by_name(features, "road_load_coefficient")[0]
        self.assertEqual(f.status, FeatureStatus.OK)
        self.assertGreater(f.value, 0)

    def test_coefficient_classification_c(self):
        features = _extract()
        f = _by_name(features, "road_load_coefficient")[0]
        self.assertEqual(f.classification, Classification.C)

    def test_coefficient_natural_not_magnitude_only(self):
        features = _extract()
        f = _by_name(features, "road_load_coefficient")[0]
        self.assertEqual(f.directionality, Directionality.NATURAL)

    def test_coefficient_is_dimensionless(self):
        features = _extract()
        f = _by_name(features, "road_load_coefficient")[0]
        self.assertEqual(f.unit, "")

    def test_coefficient_ok_even_without_speed(self):
        frame = _make_frame(vehicle_speed_ms=SensorReading.missing())
        features = _extract(frame)
        f = _by_name(features, "road_load_coefficient")[0]
        self.assertEqual(f.status, FeatureStatus.OK)





class NoToeFeatureTests(unittest.TestCase):
    """Worker 4 must NOT emit any toe feature."""

    def test_no_toe_features(self):
        features = _extract()
        for f in features:
            self.assertNotIn("toe", f.name.lower(),
                           f"Worker 4 must not emit toe features, found: {f.name}")


class ProvenanceTests(unittest.TestCase):
    def test_timestamp_and_provenance(self):
        frame = _make_frame(timestamp_s=77.0, source="replay")
        features = _extract(frame)
        for f in features:
            self.assertEqual(f.timestamp_s, 77.0)
            self.assertEqual(f.provenance, "replay")

    def test_extractor_version(self):
        features = _extract()
        for f in features:
            self.assertEqual(f.extractor_version, EXTRACTOR_VERSION)


class WornTyresLowerRRTests(unittest.TestCase):
    """Verify the project rule: worn tyres have LOWER rolling resistance."""

    def test_note_in_docstring(self):
        """The module docstring must state this explicitly."""
        import evtyre.features.road_load as rl_mod
        docstring = rl_mod.__doc__ or ""
        self.assertIn("LOWER rolling resistance", docstring)


if __name__ == "__main__":
    unittest.main()
