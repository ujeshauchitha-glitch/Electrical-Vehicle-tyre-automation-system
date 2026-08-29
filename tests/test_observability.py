"""Phase 3.1 observability gate tests.

This is the test class whose absence let E1/E2/E4 ship. Its whole job is to
make "the estimator returned the prior and called it an estimate" a test
failure rather than something a human has to notice in a demo printout.

The central invariant: a state whose value equals its prior must never be
labelled OBSERVED.
"""

import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.estimation.estimator import (
    JACOBIAN_ZERO_TOL,
    Observability,
    StateEstimate,
    TyreEstimator,
)
from evtyre.features.contract import (
    Classification,
    Directionality,
    Feature,
    FeatureStatus,
)

CORNER_NAMES = ("FL", "FR", "RL", "RR")
TREAD_STATES = tuple(f"tread_{c}" for c in CORNER_NAMES)
PRESS_STATES = tuple(f"press_{c}" for c in CORNER_NAMES)


def _vehicle() -> VehicleConfig:
    return VehicleConfig(
        vehicle_id="test",
        mass_kg=1800.0,
        front_weight_fraction=0.48,
        drive_layout=DriveLayout.RWD,
    )


def _tyre() -> TyreConfig:
    return TyreConfig(
        tyre_model_id="test-tyre",
        wheel_belt_radius_m=0.322,
        tread_new_mm=8.0,
        tread_legal_mm=1.6,
        placard_pressure_kpa=240.0,
        cold_reference_temperature_c=25.0,
    )


def _feature(name: str, value: float, corner: str | None = None) -> Feature:
    return Feature(
        name=name,
        value=value,
        unit="",
        status=FeatureStatus.OK,
        unavailable_reason=None,
        directionality=Directionality.NATURAL,
        classification=Classification.A,
        inputs=("test",),
        corner=corner,
        timestamp_s=0.0,
        provenance="simulated",
        extractor_version="test",
    )


def _features(pressure_kpa=238.0, ratio_front=1.0, ratio_rear=1.0):
    """A fully-populated feature set: 4 pressures, 2 axle ratios, 4 temps."""
    out = [
        _feature(f"running_pressure_pa_{c}", pressure_kpa * 1000.0 + 101_325.0, c)
        for c in CORNER_NAMES
    ]
    out.append(_feature("axle_speed_ratio_front", ratio_front))
    out.append(_feature("axle_speed_ratio_rear", ratio_rear))
    out.extend(_feature(f"tyre_temperature_c_{c}", 30.0, c) for c in CORNER_NAMES)
    return out


def _estimate(features):
    return TyreEstimator(_vehicle(), _tyre()).estimate(features)


def _by_name(result) -> dict[str, StateEstimate]:
    return {s.name: s for s in result.states}


class StateEstimateInvariantTests(unittest.TestCase):
    """The dataclass must enforce its own contract, like SensorReading/Feature."""

    def _make(self, observability, reason):
        return StateEstimate(
            name="tread_FL",
            value=4.8,
            sigma=2.5,
            observability=observability,
            prior_value=4.8,
            prior_sigma=2.5,
            variance_reduction=0.0,
            jacobian_column_norm=0.0,
            magnitude_only=False,
            reason=reason,
        )

    def test_unobservable_requires_a_reason(self):
        with self.assertRaises(ValueError):
            self._make(Observability.UNOBSERVABLE, None)

    def test_weak_requires_a_reason(self):
        with self.assertRaises(ValueError):
            self._make(Observability.WEAK, "")

    def test_observed_must_not_carry_a_reason(self):
        with self.assertRaises(ValueError):
            self._make(Observability.OBSERVED, "should not be here")

    def test_valid_combinations_construct(self):
        self._make(Observability.UNOBSERVABLE, "no sensitivity")
        self._make(Observability.OBSERVED, None)


class PressureObservabilityTests(unittest.TestCase):
    """Pressure has a near-direct sensor and must come out OBSERVED."""

    def test_pressure_is_observed(self):
        states = _by_name(_estimate(_features()))
        for name in PRESS_STATES:
            with self.subTest(state=name):
                self.assertIs(states[name].observability, Observability.OBSERVED)
                self.assertIsNone(states[name].reason)

    def test_pressure_moves_off_prior(self):
        states = _by_name(_estimate(_features(pressure_kpa=200.0)))
        for name in PRESS_STATES:
            with self.subTest(state=name):
                s = states[name]
                self.assertNotAlmostEqual(s.value, s.prior_value, places=3)

    def test_pressure_tracks_the_measurement(self):
        states = _by_name(_estimate(_features(pressure_kpa=200.0)))
        self.assertAlmostEqual(states["press_FL"].value, 200.0, delta=5.0)


class UnobservableStateTests(unittest.TestCase):
    """Toe and camber have no channel that depends on them.

    Absolute tread is covered separately in TreadObservabilityTests: the axle
    ratio does give tread a non-zero Jacobian, so it is not UNOBSERVABLE in the
    zero-sensitivity sense — it is unresolved in the common mode.
    """

    def test_toe_is_not_observed(self):
        s = _by_name(_estimate(_features()))["toe^2"]
        self.assertIsNot(s.observability, Observability.OBSERVED)
        self.assertTrue(s.reason)

    def test_toe_stays_at_prior(self):
        s = _by_name(_estimate(_features()))["toe^2"]
        self.assertAlmostEqual(s.value, s.prior_value, places=9)

    def test_toe_is_magnitude_only(self):
        self.assertTrue(_by_name(_estimate(_features()))["toe^2"].magnitude_only)

    def test_camber_is_unobservable(self):
        s = _by_name(_estimate(_features()))["camber"]
        self.assertIs(s.observability, Observability.UNOBSERVABLE)
        self.assertTrue(s.reason)
        self.assertLessEqual(s.jacobian_column_norm, JACOBIAN_ZERO_TOL)

    def test_camber_stays_at_prior(self):
        s = _by_name(_estimate(_features()))["camber"]
        self.assertAlmostEqual(s.value, s.prior_value, places=9)


class TreadObservabilityTests(unittest.TestCase):
    """Absolute tread must not be claimed without the resonance channel.

    The axle ratio r_R/r_L constrains the tread DIFFERENCE within an axle. The
    common mode (both corners moving together) is unconstrained, so an absolute
    tread number is not available from this channel set.
    """

    def test_absolute_tread_is_never_observed(self):
        states = _by_name(_estimate(_features()))
        for name in TREAD_STATES:
            with self.subTest(state=name):
                self.assertIsNot(states[name].observability, Observability.OBSERVED)

    def test_tread_does_not_move_without_ratio_evidence(self):
        """Ratios at 1.0 carry no tread-difference information, so tread must
        stay at its prior. This is the regression guard for the road-load
        channel: when it was admitted, pressure alone moved tread ~0.5 mm."""
        states = _by_name(_estimate(_features(ratio_front=1.0, ratio_rear=1.0)))
        for name in TREAD_STATES:
            with self.subTest(state=name):
                s = states[name]
                self.assertAlmostEqual(s.value, s.prior_value, places=6)

    def test_pressure_alone_does_not_move_tread(self):
        """Directly targets the confound: vary ONLY pressure, hold the tread
        evidence fixed, and assert the tread estimate is unchanged."""
        low = _by_name(_estimate(_features(pressure_kpa=180.0)))
        high = _by_name(_estimate(_features(pressure_kpa=240.0)))
        for name in TREAD_STATES:
            with self.subTest(state=name):
                self.assertAlmostEqual(low[name].value, high[name].value, places=6)

    def test_within_axle_tread_difference_responds_to_ratio(self):
        """The one tread claim that IS supported: a differential."""
        states = _by_name(_estimate(_features(ratio_front=0.998)))
        diff = states["tread_FL"].value - states["tread_FR"].value
        self.assertGreater(abs(diff), 1e-3)

    def test_tread_difference_reverses_with_ratio(self):
        below = _by_name(_estimate(_features(ratio_front=0.998)))
        above = _by_name(_estimate(_features(ratio_front=1.002)))
        d_below = below["tread_FL"].value - below["tread_FR"].value
        d_above = above["tread_FL"].value - above["tread_FR"].value
        self.assertLess(d_below * d_above, 0.0, "difference must change sign")

    def test_rear_ratio_does_not_move_front_tread(self):
        """Axle scoping: a rear-axle ratio must not leak into front tread."""
        base = _by_name(_estimate(_features()))
        rear = _by_name(_estimate(_features(ratio_rear=0.998)))
        for name in ("tread_FL", "tread_FR"):
            with self.subTest(state=name):
                self.assertAlmostEqual(base[name].value, rear[name].value, places=6)


class NoSilentPriorTests(unittest.TestCase):
    """The single invariant that would have caught E1, E2 and E4 together."""

    def test_no_state_sits_at_prior_while_labelled_observed(self):
        for features in (_features(), _features(ratio_front=0.998), []):
            result = _estimate(features)
            for s in result.states:
                with self.subTest(state=s.name, n_features=len(features)):
                    if abs(s.value - s.prior_value) < 1e-9:
                        self.assertIsNot(
                            s.observability,
                            Observability.OBSERVED,
                            f"{s.name} equals its prior but is labelled OBSERVED",
                        )

    def test_empty_features_leaves_every_state_unobserved(self):
        result = _estimate([])
        for s in result.states:
            with self.subTest(state=s.name):
                self.assertIsNot(s.observability, Observability.OBSERVED)
                self.assertTrue(s.reason)
                # Not exactly equal: the disabled-channel sentinel variance is
                # 1e12, not infinity, so Gauss-Newton leaves a ~1e-7 residue.
                # The claim under test is "did not meaningfully move".
                self.assertAlmostEqual(s.value, s.prior_value, delta=1e-4)

    def test_empty_features_reports_zero_observed(self):
        self.assertEqual(_estimate([]).n_states_observed, 0)

    def test_every_non_observed_state_explains_itself(self):
        for s in _estimate(_features()).states:
            with self.subTest(state=s.name):
                if s.observability is not Observability.OBSERVED:
                    self.assertTrue(s.reason and s.reason.strip())


class NoCamberSpecialCasingTests(unittest.TestCase):
    """Camber must be classified by the general rule, not by name.

    If camber is special-cased, tread and toe can silently keep being reported
    as estimates — which is exactly what happened before 3.1.
    """

    def test_estimator_does_not_branch_on_the_literal_camber(self):
        import inspect
        import evtyre.estimation.estimator as est

        source = inspect.getsource(est)
        offenders = [
            line.strip()
            for line in source.splitlines()
            if "camber" in line.lower()
            and line.strip().startswith(("if ", "elif "))
        ]
        self.assertEqual(
            offenders, [], f"camber must not be special-cased: {offenders}"
        )

    def test_camber_is_not_the_only_unobserved_state(self):
        """If camber were the only non-OBSERVED state, the contract would be
        doing nothing that the old hardcoded camber_note did not."""
        unobserved = [
            s.name
            for s in _estimate(_features()).states
            if s.observability is not Observability.OBSERVED
        ]
        self.assertGreater(len(unobserved), 1)
        self.assertIn("camber", unobserved)


class ReportedAggregateTests(unittest.TestCase):
    def test_mean_variance_reduction_ignores_unobserved_states(self):
        result = _estimate(_features())
        observed = [
            s.variance_reduction
            for s in result.states
            if s.observability is Observability.OBSERVED
        ]
        self.assertTrue(observed, "expected at least the pressure states")
        expected = sum(observed) / len(observed)
        self.assertAlmostEqual(result.mean_variance_reduction, expected, places=9)

    def test_n_states_observed_matches_the_labels(self):
        result = _estimate(_features())
        counted = sum(
            1 for s in result.states if s.observability is Observability.OBSERVED
        )
        self.assertEqual(result.n_states_observed, counted)

    def test_result_exposes_one_state_per_state_vector_entry(self):
        self.assertEqual(len(_estimate(_features()).states), 10)


class RoadLoadChannelDisabledTests(unittest.TestCase):
    """The road-load coefficient must not be admitted as a measurement.

    It is mean(C_rr) recomputed from TPMS pressure, so admitting it
    double-counts pressure and lets the estimator move tread to close a
    definitional gap. Guards against silent re-enabling.
    """

    def test_road_load_feature_does_not_change_the_estimate(self):
        without = _by_name(_estimate(_features()))
        with_rl = _by_name(
            _estimate(_features() + [_feature("road_load_coefficient", 0.0110)])
        )
        for name in without:
            with self.subTest(state=name):
                self.assertAlmostEqual(
                    without[name].value, with_rl[name].value, places=9
                )

    def test_road_load_feature_does_not_change_confidence(self):
        without = _estimate(_features())
        with_rl = _estimate(_features() + [_feature("road_load_coefficient", 0.0110)])
        self.assertEqual(without.n_states_observed, with_rl.n_states_observed)
        self.assertAlmostEqual(
            without.mean_variance_reduction, with_rl.mean_variance_reduction, places=9
        )


if __name__ == "__main__":
    unittest.main()
