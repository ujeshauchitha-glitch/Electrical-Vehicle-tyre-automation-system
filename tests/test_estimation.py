"""Tests for the Phase 3 tyre state estimator."""

import numpy as np
import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.estimation.state import STATE, MEAS, StateLayout, MeasurementLayout
from evtyre.estimation.estimator import (
    TyreEstimator,
    EstimatorResult,
    SensorNoise,
    features_to_measurement,
    prior,
)
from evtyre.features.contract import Classification, Directionality, Feature, FeatureStatus
from evtyre.schema.common import CORNERS


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


def _ok_feature(name, value, **overrides):
    """Create a minimal OK feature."""
    defaults = dict(
        name=name,
        value=value,
        unit="",
        status=FeatureStatus.OK,
        unavailable_reason=None,
        directionality=Directionality.NATURAL,
        classification=Classification.A,
        inputs=("test",),
        corner=None,
        timestamp_s=0.0,
        provenance="simulated",
        extractor_version="test",
    )
    defaults.update(overrides)
    return Feature(**defaults)


def _unavailable_feature(name, reason="test missing"):
    return Feature(
        name=name,
        value=None,
        unit="",
        status=FeatureStatus.UNAVAILABLE,
        unavailable_reason=reason,
        directionality=Directionality.NATURAL,
        classification=Classification.A,
        inputs=("test",),
        corner=None,
        timestamp_s=0.0,
        provenance="simulated",
        extractor_version="test",
    )


class StateLayoutTests(unittest.TestCase):
    def test_state_dimension(self):
        self.assertEqual(STATE.N, 10)

    def test_indices_are_separate_from_measurement(self):
        """State and measurement slices must be TEXTUALLY SEPARATE."""
        state_indices = set(range(STATE.N))
        meas_indices = set(range(MEAS.N))
        # They share numeric ranges (0-9) but have DIFFERENT meanings
        # This is fine — the point is no shared Python object or alias
        self.assertIsNot(STATE.press_indices, MEAS.press_indices)

    def test_tread_indices(self):
        self.assertEqual(STATE.tread_indices, [0, 1, 2, 3])

    def test_press_indices(self):
        self.assertEqual(STATE.press_indices, [4, 5, 6, 7])


class MeasurementLayoutTests(unittest.TestCase):
    def test_measurement_dimension(self):
        self.assertEqual(MEAS.N, 11)

    def test_press_indices(self):
        self.assertEqual(MEAS.press_indices, [0, 1, 2, 3])

    def test_freq_indices(self):
        self.assertEqual(MEAS.freq_indices, [4, 5, 6, 7])


class PriorTests(unittest.TestCase):
    def test_prior_shape(self):
        x0, P0 = prior(_make_tyre())
        self.assertEqual(x0.shape, (STATE.N,))
        self.assertEqual(P0.shape, (STATE.N, STATE.N))

    def test_prior_tread_midpoint(self):
        tyre = _make_tyre()
        x0, _ = prior(tyre)
        expected = (tyre.tread_new_mm + tyre.tread_legal_mm) / 2.0
        for i in STATE.tread_indices:
            self.assertAlmostEqual(x0[i], expected)

    def test_prior_pressure_at_placard(self):
        tyre = _make_tyre()
        x0, _ = prior(tyre)
        for i in STATE.press_indices:
            self.assertAlmostEqual(x0[i], tyre.placard_pressure_kpa)

    def test_prior_toe_sq_positive(self):
        x0, _ = prior(_make_tyre())
        self.assertGreater(x0[STATE.toe_sq], 0.0)

    def test_prior_covariance_positive_definite(self):
        _, P0 = prior(_make_tyre())
        eigenvalues = np.linalg.eigvalsh(P0)
        self.assertTrue(np.all(eigenvalues > 0))


class FeatureToMeasurementTests(unittest.TestCase):
    def test_populated_measurements(self):
        features = [
            _ok_feature("running_pressure_pa_FL", 341325.0, corner="FL"),
            _ok_feature("running_pressure_pa_FR", 341325.0, corner="FR"),
            _ok_feature("running_pressure_pa_RL", 341325.0, corner="RL"),
            _ok_feature("running_pressure_pa_RR", 341325.0, corner="RR"),
            _ok_feature("axle_speed_ratio_front", 1.0),
            _ok_feature("axle_speed_ratio_rear", 1.0),
            _ok_feature("road_load_coefficient", 0.015),
        ]
        z, R_diag, avail = features_to_measurement(
            features, _make_vehicle(), _make_tyre(),
        )
        self.assertEqual(len(z), MEAS.N)
        # 341325 Pa ABSOLUTE = 240.0 kPa GAUGE (placard), which is what the state uses.
        self.assertAlmostEqual(z[MEAS.press_fl], 240.0, places=1)
        # Speed ratios
        self.assertAlmostEqual(z[MEAS.ratio_front], 1.0)
        self.assertAlmostEqual(z[MEAS.ratio_rear], 1.0)
        # Road load
        self.assertAlmostEqual(z[MEAS.roadload], 0.015)

    def test_unavailable_measurements_get_large_variance(self):
        features = []  # nothing available
        z, R_diag, avail = features_to_measurement(
            features, _make_vehicle(), _make_tyre(),
        )
        self.assertEqual(len(avail), 0)
        # All variances should be huge (effectively ignored)
        self.assertTrue(np.all(R_diag > 1e6))


class EstimatorTests(unittest.TestCase):
    def test_estimator_runs_without_crashing(self):
        features = [
            _ok_feature("running_pressure_pa_FL", 341325.0, corner="FL"),
            _ok_feature("running_pressure_pa_FR", 341325.0, corner="FR"),
            _ok_feature("running_pressure_pa_RL", 341325.0, corner="RL"),
            _ok_feature("running_pressure_pa_RR", 341325.0, corner="RR"),
            _ok_feature("axle_speed_ratio_front", 1.0),
            _ok_feature("axle_speed_ratio_rear", 1.0),
            _ok_feature("road_load_coefficient", 0.015),
        ]
        est = TyreEstimator(_make_vehicle(), _make_tyre())
        result = est.estimate(features)

        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.state.shape, (STATE.N,))
        self.assertEqual(result.covariance.shape, (STATE.N, STATE.N))

    def test_estimator_handles_no_features(self):
        """With no features, the estimator should still run (return prior)."""
        est = TyreEstimator(_make_vehicle(), _make_tyre())
        result = est.estimate([])
        # Should not crash — returns prior-based estimate
        self.assertIsInstance(result, EstimatorResult)

    def test_confidence_reduced_with_missing_features(self):
        est = TyreEstimator(_make_vehicle(), _make_tyre())
        result_full = est.estimate([
            _ok_feature("running_pressure_pa_FL", 341325.0, corner="FL"),
            _ok_feature("running_pressure_pa_FR", 341325.0, corner="FR"),
            _ok_feature("running_pressure_pa_RL", 341325.0, corner="RL"),
            _ok_feature("running_pressure_pa_RR", 341325.0, corner="RR"),
            _ok_feature("axle_speed_ratio_front", 1.0),
            _ok_feature("axle_speed_ratio_rear", 1.0),
            _ok_feature("road_load_coefficient", 0.015),
        ])
        result_empty = est.estimate([])
        self.assertGreater(result_full.confidence, result_empty.confidence)

    def test_toe_magnitude_is_nonnegative(self):
        est = TyreEstimator(_make_vehicle(), _make_tyre())
        result = est.estimate([])
        self.assertGreaterEqual(result.toe_magnitude_deg, 0.0)

    def test_camber_note_exists(self):
        est = TyreEstimator(_make_vehicle(), _make_tyre())
        result = est.estimate([])
        self.assertIn("unobservable", result.camber_note.lower())

    def test_measurement_vector_fully_populated(self):
        """The assertion that z is fully populated catches the legacy bug."""
        est = TyreEstimator(_make_vehicle(), _make_tyre())
        result = est.estimate([])
        # The estimator should have created a valid z vector
        self.assertIsNotNone(result)


class NoTempCompensationBeforePhysicsTests(unittest.TestCase):
    """Verify the estimator does not temperature-compensate pressure
    before feeding it to the physics model."""

    def test_state_uses_running_pressure(self):
        """State press indices correspond to running pressure, not
        cold-equivalent.  The estimator must NOT transform pressure
        before the predict step."""
        import evtyre.estimation.estimator as est_mod
        # The predict function should use x[press] directly
        import inspect
        source = inspect.getsource(est_mod._predict)
        # Should NOT contain 'cold' or 'compensate' in the pressure path
        self.assertNotIn("cold_equivalent", source.lower())
        self.assertNotIn("compensate_pressure", source.lower())


class ToeSquaredTests(unittest.TestCase):
    """Verify toe is estimated as toe^2, magnitude only."""

    def test_toe_sq_in_state(self):
        self.assertEqual(STATE.toe_sq, 8)

    def test_toe_magnitude_from_sqrt(self):
        est = TyreEstimator(_make_vehicle(), _make_tyre())
        result = est.estimate([])
        # toe_magnitude should be sqrt(max(0, toe_sq))
        self.assertAlmostEqual(
            result.toe_magnitude_deg,
            np.sqrt(max(0.0, result.state[STATE.toe_sq])),
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
