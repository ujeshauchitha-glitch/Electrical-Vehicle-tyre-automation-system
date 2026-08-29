"""Tests for pressure and temperature feature extraction."""

import math
import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.features import Classification, Directionality, FeatureStatus
from evtyre.features import pressure_thermal as pt
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
    )


def _ok_per_corner(value: float, key: str) -> dict[str, SensorReading]:
    return {c: SensorReading(value=value, status=SensorStatus.OK) for c in CORNERS}


def _make_frame(**overrides) -> TelemetryFrame:
    defaults = dict(
        timestamp_s=10.0,
        source="simulated",
        wheel_speed_rad_s=_ok_per_corner(50.0, "ws"),
        tpms_pressure_kpa=_ok_per_corner(240.0, "p"),
        tpms_temperature_c=_ok_per_corner(30.0, "t"),
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
    return pt.extract(frame, _make_vehicle(), _make_tyre())


def _by_name(features, name):
    return [f for f in features if f.name == name]


class RunningPressureTests(unittest.TestCase):
    def test_ok_pressure_converted_to_absolute_pa(self):
        features = _extract()
        for corner in CORNERS:
            f = _by_name(features, f"running_pressure_pa_{corner}")
            self.assertEqual(len(f), 1)
            f = f[0]
            self.assertEqual(f.status, FeatureStatus.OK)
            self.assertEqual(f.unit, "Pa")
            # 240 kPa gauge + 101325 Pa atmospheric = 341325 Pa
            self.assertAlmostEqual(f.value, 240_000.0 + pt.ATMOSPHERIC_PRESSURE_PA)

    def test_missing_tpms_pressure_is_unavailable(self):
        readings = _ok_per_corner(240.0, "p")
        readings["FL"] = SensorReading.missing()
        frame = _make_frame(tpms_pressure_kpa=readings)
        features = _extract(frame)
        f = _by_name(features, "running_pressure_pa_FL")
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].status, FeatureStatus.UNAVAILABLE)
        self.assertIsNone(f[0].value)
        self.assertIn("TPMS pressure missing", f[0].unavailable_reason)
        # Other corners still OK
        f_fr = _by_name(features, "running_pressure_pa_FR")
        self.assertEqual(f_fr[0].status, FeatureStatus.OK)

    def test_running_pressure_inputs(self):
        features = _extract()
        f = _by_name(features, "running_pressure_pa_FL")[0]
        self.assertEqual(f.inputs, ("tpms_pressure_kpa",))

    def test_running_pressure_is_classification_a(self):
        features = _extract()
        f = _by_name(features, "running_pressure_pa_FL")[0]
        self.assertEqual(f.classification, Classification.A)


class ColdEquivalentPressureTests(unittest.TestCase):
    def test_cold_equivalent_uses_gas_law(self):
        """Gay-Lussac: P_cold = P * T_ref / T_running."""
        # At 30 °C = 303.15 K, P_cold should be P_running * 293.15 / 303.15
        frame = _make_frame(
            tpms_pressure_kpa=_ok_per_corner(240.0, "p"),
            tpms_temperature_c=_ok_per_corner(30.0, "t"),
        )
        features = _extract(frame)
        for corner in CORNERS:
            f = _by_name(features, f"cold_equivalent_pressure_pa_{corner}")[0]
            p_running_pa = (240_000.0 + pt.ATMOSPHERIC_PRESSURE_PA)
            expected = p_running_pa * 293.15 / 303.15
            self.assertAlmostEqual(f.value, expected, places=2)
            self.assertEqual(f.status, FeatureStatus.OK)

    def test_cold_equivalent_is_classification_b(self):
        features = _extract()
        f = _by_name(features, "cold_equivalent_pressure_pa_FL")[0]
        self.assertEqual(f.classification, Classification.B)

    def test_cold_equivalent_unavailable_without_temp(self):
        temps = _ok_per_corner(30.0, "t")
        temps["RL"] = SensorReading.missing()
        frame = _make_frame(tpms_temperature_c=temps)
        features = _extract(frame)
        f = _by_name(features, "cold_equivalent_pressure_pa_RL")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)
        self.assertIn("TPMS temperature missing", f.unavailable_reason)

    def test_cold_equivalent_unavailable_without_pressure(self):
        pressures = _ok_per_corner(240.0, "p")
        pressures["RR"] = SensorReading.missing()
        frame = _make_frame(tpms_pressure_kpa=pressures)
        features = _extract(frame)
        f = _by_name(features, "cold_equivalent_pressure_pa_RR")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)

    def test_cold_equivalent_inputs_include_both_channels(self):
        features = _extract()
        f = _by_name(features, "cold_equivalent_pressure_pa_FL")[0]
        self.assertIn("tpms_pressure_kpa", f.inputs)
        self.assertIn("tpms_temperature_c", f.inputs)


class PressureDeviationTests(unittest.TestCase):
    def test_deviation_from_placard(self):
        # Placard is 240 kPa; actual is 240 kPa → deviation = 0
        features = _extract()
        f = _by_name(features, "pressure_deviation_from_placard_FL")[0]
        self.assertAlmostEqual(f.value, 0.0)

    def test_high_pressure_positive_deviation(self):
        readings = _ok_per_corner(260.0, "p")
        frame = _make_frame(tpms_pressure_kpa=readings)
        features = _extract(frame)
        f = _by_name(features, "pressure_deviation_from_placard_FL")[0]
        self.assertAlmostEqual(f.value, 20_000.0)  # (260-240) * 1000 Pa

    def test_low_pressure_negative_deviation(self):
        readings = _ok_per_corner(220.0, "p")
        frame = _make_frame(tpms_pressure_kpa=readings)
        features = _extract(frame)
        f = _by_name(features, "pressure_deviation_from_placard_FL")[0]
        self.assertAlmostEqual(f.value, -20_000.0)


class TemperatureTests(unittest.TestCase):
    def test_temperature_ok(self):
        features = _extract()
        f = _by_name(features, "tyre_temperature_c_FL")[0]
        self.assertEqual(f.value, 30.0)
        self.assertEqual(f.unit, "°C")

    def test_missing_temperature(self):
        temps = _ok_per_corner(30.0, "t")
        temps["FR"] = SensorReading.missing()
        frame = _make_frame(tpms_temperature_c=temps)
        features = _extract(frame)
        f = _by_name(features, "tyre_temperature_c_FR")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)


class TemperatureRiseTests(unittest.TestCase):
    def test_temperature_rise(self):
        features = _extract()
        f = _by_name(features, "temperature_rise_above_ambient_FL")[0]
        # 30 °C - 20 °C ambient = 10 °C rise
        self.assertAlmostEqual(f.value, 10.0)

    def test_unavailable_without_ambient(self):
        frame = _make_frame(ambient_temp_c=SensorReading.missing())
        features = _extract(frame)
        f = _by_name(features, "temperature_rise_above_ambient_FL")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)
        self.assertIn("ambient temperature missing", f.unavailable_reason)


class CrossCornerSpreadTests(unittest.TestCase):
    def test_spread_zero_when_all_equal(self):
        features = _extract()
        f = _by_name(features, "cross_corner_pressure_spread_kpa")[0]
        self.assertAlmostEqual(f.value, 0.0)
        self.assertIsNone(f.corner)  # vehicle-level

    def test_spread_nonzero(self):
        readings = {c: SensorReading(value=240.0, status=SensorStatus.OK) for c in CORNERS}
        readings["FL"] = SensorReading(value=250.0, status=SensorStatus.OK)
        frame = _make_frame(tpms_pressure_kpa=readings)
        features = _extract(frame)
        f = _by_name(features, "cross_corner_pressure_spread_kpa")[0]
        self.assertAlmostEqual(f.value, 10.0)

    def test_spread_unavailable_with_fewer_than_two(self):
        readings = {c: SensorReading.missing() for c in CORNERS}
        readings["FL"] = SensorReading(value=240.0, status=SensorStatus.OK)
        frame = _make_frame(tpms_pressure_kpa=readings)
        features = _extract(frame)
        f = _by_name(features, "cross_corner_pressure_spread_kpa")[0]
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)

    def test_spread_is_magnitude_only(self):
        features = _extract()
        f = _by_name(features, "cross_corner_pressure_spread_kpa")[0]
        self.assertEqual(f.directionality, Directionality.MAGNITUDE_ONLY)


class ProvenanceTests(unittest.TestCase):
    def test_timestamp_and_provenance_copied(self):
        frame = _make_frame(timestamp_s=42.0, source="replay")
        features = _extract(frame)
        for f in features:
            self.assertEqual(f.timestamp_s, 42.0)
            self.assertEqual(f.provenance, "replay")

    def test_extractor_version_set(self):
        features = _extract()
        for f in features:
            self.assertEqual(f.extractor_version, pt.EXTRACTOR_VERSION)


if __name__ == "__main__":
    unittest.main()
