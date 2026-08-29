"""Tests for the pipeline integration."""

import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.features import pressure_thermal, kinematics, resonance
from evtyre.features.contract import FeatureStatus
from evtyre.pipeline import Pipeline
from evtyre.schema.common import CORNERS, SensorReading, SensorStatus
from evtyre.schema.telemetry import TelemetryFrame


def _make_vehicle() -> VehicleConfig:
    return VehicleConfig("test", 1800.0, 0.48, DriveLayout.RWD)


def _make_tyre() -> TyreConfig:
    return TyreConfig("test", 0.322, 8.0, 1.6, 240.0, 25.0)


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


class PipelineRegistryTests(unittest.TestCase):
    def test_register_and_extract(self):
        pipe = Pipeline(_make_vehicle(), _make_tyre())
        pipe.register("pressure_thermal", pressure_thermal.extract)
        features = pipe.extract_features(_make_frame())
        # Should have features from pressure_thermal
        names = [f.name for f in features]
        self.assertTrue(any("running_pressure" in n for n in names))

    def test_multiple_extractors(self):
        pipe = Pipeline(_make_vehicle(), _make_tyre())
        pipe.register("pressure_thermal", pressure_thermal.extract)
        pipe.register("kinematics", kinematics.extract)
        features = pipe.extract_features(_make_frame())
        names = [f.name for f in features]
        self.assertTrue(any("running_pressure" in n for n in names))
        self.assertTrue(any("slip_ratio" in n for n in names))


class ResonanceErrorToleranceTests(unittest.TestCase):
    def test_resonance_error_does_not_kill_pipeline(self):
        """Worker 5's resonance extractor raises — pipeline must catch it."""
        pipe = Pipeline(_make_vehicle(), _make_tyre())
        pipe.register("pressure_thermal", pressure_thermal.extract)
        pipe.register("resonance", resonance.extract)  # will raise

        features = pipe.extract_features(_make_frame())
        # Should still have pressure_thermal features
        names = [f.name for f in features]
        self.assertTrue(any("running_pressure" in n for n in names))
        # Should have an UNAVAILABLE feature for the resonance error
        error_features = [f for f in features if f.status == FeatureStatus.UNAVAILABLE
                         and "resonance" in f.name.lower()]
        self.assertTrue(len(error_features) > 0,
            "Pipeline should have recorded resonance as UNAVAILABLE")
        # The error reason should mention G1
        for f in error_features:
            self.assertIn("G1", f.unavailable_reason or "")

    def test_extractor_exception_recorded_as_unavailable(self):
        """Any unexpected exception is recorded, not raised."""
        def broken_extractor(frame, vc, tc):
            raise ValueError("something broke")

        pipe = Pipeline(_make_vehicle(), _make_tyre())
        pipe.register("broken", broken_extractor)
        pipe.register("pressure_thermal", pressure_thermal.extract)

        features = pipe.extract_features(_make_frame())
        # broken_extractor's error should be recorded
        error_features = [f for f in features
                         if f.name == "broken_error"]
        self.assertEqual(len(error_features), 1)
        self.assertEqual(error_features[0].status, FeatureStatus.UNAVAILABLE)
        self.assertIn("something broke", error_features[0].unavailable_reason)


class EndToEndTests(unittest.TestCase):
    def test_full_pipeline_run(self):
        pipe = Pipeline(_make_vehicle(), _make_tyre())
        pipe.register("pressure_thermal", pressure_thermal.extract)
        pipe.register("kinematics", kinematics.extract)
        pipe.register("resonance", resonance.extract)

        features, result = pipe.run(_make_frame())
        self.assertTrue(len(features) > 0)
        self.assertIsNotNone(result.states)
        self.assertGreater(result.n_states_observed > 0, 0)

    def test_pipeline_without_resonance(self):
        """Pipeline should work fine without registering resonance."""
        pipe = Pipeline(_make_vehicle(), _make_tyre())
        pipe.register("pressure_thermal", pressure_thermal.extract)
        pipe.register("kinematics", kinematics.extract)

        features, result = pipe.run(_make_frame())
        self.assertTrue(len(features) > 0)


if __name__ == "__main__":
    unittest.main()
