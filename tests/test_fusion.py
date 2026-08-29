"""Phase 4 decision-layer tests.

The load-bearing test in this file is
TractionRefusalTests.test_torque_ceiling_is_withheld_while_tread_is_weak.
Legacy computes the wet torque ceiling from tread_est - 2*sigma; with tread
currently WEAK that evaluates to ~0.7 mm, below the legal limit, purely from
the width of the prior. Emitting a traction limit from it would be a fabricated
number acted on by a vehicle.
"""

import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.estimation.estimator import (
    Observability,
    PhysicsConfig,
    StateEstimate,
    TyreEstimator,
)
from evtyre.estimation.schema import TyreStateEstimate
from evtyre.features.contract import (
    Classification,
    Directionality,
    Feature,
    FeatureStatus,
)
from evtyre.fusion import (
    Decision,
    DecisionStatus,
    FrictionConfig,
    MaintenanceConfig,
    SnapshotFusionError,
    build_report,
    cold_equivalent_pressure_kpa,
    driven_corners,
    fuse_snapshots,
    maintenance_view,
    recoverable_energy,
    require_observed,
    torque_ceiling,
    wet_friction,
)
from evtyre.schema.common import CORNERS


def _vehicle(layout=DriveLayout.RWD) -> VehicleConfig:
    return VehicleConfig(
        vehicle_id="test", mass_kg=1850.0,
        front_weight_fraction=0.47, drive_layout=layout,
    )


def _tyre() -> TyreConfig:
    return TyreConfig(
        tyre_model_id="test-tyre", wheel_belt_radius_m=0.333,
        tread_new_mm=7.5, tread_legal_mm=1.6,
        placard_pressure_kpa=240.0, cold_reference_temperature_c=25.0,
    )


def _physics() -> PhysicsConfig:
    return PhysicsConfig(
        k_z0=210_000.0, cornering_stiffness=55_000.0, tread_rr_span=0.20,
        c_rr0=0.0090, p_exponent=0.45, t_coeff=0.0015, deflection_factor=1.0 / 3.0,
    )


def _friction() -> FrictionConfig:
    return FrictionConfig(
        mu_dry=1.00, wet_floor=0.45, wet_tau=2.5, safety_factor=0.85,
    )


def _maintenance() -> MaintenanceConfig:
    return MaintenanceConfig(
        low_pressure_margin_kpa=20.0, cross_corner_spread_limit_kpa=15.0,
    )


def _feature(name, value, corner=None):
    return Feature(
        name=name, value=value, unit="", status=FeatureStatus.OK,
        unavailable_reason=None, directionality=Directionality.NATURAL,
        classification=Classification.A, inputs=("test",), corner=corner,
        timestamp_s=0.0, provenance="simulated", extractor_version="test",
    )


def _real_estimate(pressure_kpa=238.0, ratio_front=0.9971, ratio_rear=1.0028):
    """A genuine Phase 3 estimate: pressure OBSERVED, tread WEAK."""
    feats = [
        _feature(f"running_pressure_pa_{c}", pressure_kpa * 1000.0 + 101_325.0, c)
        for c in CORNERS
    ]
    feats += [
        _feature("axle_speed_ratio_front", ratio_front),
        _feature("axle_speed_ratio_rear", ratio_rear),
    ]
    feats += [_feature(f"tyre_temperature_c_{c}", 30.0, c) for c in CORNERS]
    for f in feats:
        object.__setattr__(f, "timestamp_s", 10.0)
    est = TyreEstimator(_vehicle(), _tyre())
    schema = est.to_schema(est.estimate(feats), feats)
    # Phase 3's to_schema hardcodes odometer_km=None (it has no odometer
    # input). Phase 5 needs a distance axis, so supply one here to exercise
    # the Phase 4 -> 5 contract. Tracked as a real gap, not a test hack.
    return TyreStateEstimate(
        states=schema.states, covariance_diag=schema.covariance_diag,
        timestamp_s=schema.timestamp_s, odometer_km=15200.0,
        source=schema.source, model_version=schema.model_version,
        config_fingerprint=schema.config_fingerprint,
        n_measurements_available=schema.n_measurements_available,
        n_states_observed=schema.n_states_observed,
        mean_variance_reduction=schema.mean_variance_reduction,
        converged=schema.converged, singular_matrix=schema.singular_matrix,
        iteration_count=schema.iteration_count,
    )


def _with_observed_tread(estimate, tread_mm=5.0, sigma=0.25):
    """Force tread OBSERVED, to exercise the path that resonance would unlock."""
    states = []
    for s in estimate.states:
        if s.name.startswith("tread_"):
            states.append(StateEstimate(
                name=s.name, value=tread_mm, sigma=sigma,
                observability=Observability.OBSERVED,
                prior_value=s.prior_value, prior_sigma=s.prior_sigma,
                variance_reduction=0.99, reason=None,
                magnitude_only=False, jacobian_column_norm=1.0,
            ))
        else:
            states.append(s)
    return TyreStateEstimate(
        states=tuple(states),
        covariance_diag=tuple(s.sigma ** 2 for s in states),
        timestamp_s=estimate.timestamp_s, odometer_km=estimate.odometer_km,
        source=estimate.source, model_version=estimate.model_version,
        config_fingerprint=estimate.config_fingerprint,
        n_measurements_available=estimate.n_measurements_available,
        n_states_observed=sum(
            1 for s in states if s.observability is Observability.OBSERVED
        ),
        mean_variance_reduction=estimate.mean_variance_reduction,
        converged=estimate.converged, singular_matrix=estimate.singular_matrix,
        iteration_count=estimate.iteration_count,
    )


class DecisionContractTests(unittest.TestCase):
    def test_ok_requires_a_value(self):
        with self.assertRaises(ValueError):
            Decision(name="x", value=None, unit="", status=DecisionStatus.OK,
                     unavailable_reason=None, basis=())

    def test_ok_must_not_carry_a_reason(self):
        with self.assertRaises(ValueError):
            Decision(name="x", value=1.0, unit="", status=DecisionStatus.OK,
                     unavailable_reason="why", basis=())

    def test_unavailable_must_not_carry_a_value(self):
        with self.assertRaises(ValueError):
            Decision(name="x", value=1.0, unit="",
                     status=DecisionStatus.UNAVAILABLE,
                     unavailable_reason="why", basis=())

    def test_unavailable_requires_a_reason(self):
        with self.assertRaises(ValueError):
            Decision(name="x", value=None, unit="",
                     status=DecisionStatus.UNAVAILABLE,
                     unavailable_reason="  ", basis=())


class RequireObservedTests(unittest.TestCase):
    def test_passes_for_observed_state(self):
        self.assertIsNone(require_observed(_real_estimate(), ["press_FL"]))

    def test_blocks_on_weak_state_and_names_it(self):
        reason = require_observed(_real_estimate(), ["tread_FL"])
        self.assertIsNotNone(reason)
        self.assertIn("tread_FL", reason)
        self.assertIn("weak", reason.lower())

    def test_blocks_on_unknown_state(self):
        reason = require_observed(_real_estimate(), ["nonexistent"])
        self.assertIn("nonexistent", reason)


class TractionRefusalTests(unittest.TestCase):
    """The safety-critical refusal."""

    def test_torque_ceiling_is_withheld_while_tread_is_weak(self):
        decisions = torque_ceiling(
            _real_estimate(), _vehicle(), _tyre(), _physics(), _friction(),
            wet=True,
        )
        for d in decisions:
            with self.subTest(decision=d.name):
                self.assertIs(d.status, DecisionStatus.UNAVAILABLE)
                self.assertIsNone(d.value)

    def test_refusal_explains_why(self):
        d = torque_ceiling(
            _real_estimate(), _vehicle(), _tyre(), _physics(), _friction(),
        )[0]
        reason = d.unavailable_reason.lower()
        self.assertIn("tread", reason)
        self.assertIn("difference", reason)

    def test_no_lower_bound_is_published_either(self):
        """The LCB must not leak out as a standalone number."""
        decisions = torque_ceiling(
            _real_estimate(), _vehicle(), _tyre(), _physics(), _friction(),
        )
        lcb = [d for d in decisions if d.name == "drive_tread_lower_bound_mm"][0]
        self.assertIs(lcb.status, DecisionStatus.UNAVAILABLE)

    def test_becomes_available_once_tread_is_observed(self):
        est = _with_observed_tread(_real_estimate())
        decisions = torque_ceiling(
            est, _vehicle(), _tyre(), _physics(), _friction(), wet=True,
        )
        ceiling = decisions[0]
        self.assertIs(ceiling.status, DecisionStatus.OK)
        self.assertGreater(ceiling.value, 0.0)

    def test_ceiling_uses_the_lower_bound_not_the_point_estimate(self):
        est = _with_observed_tread(_real_estimate(), tread_mm=5.0, sigma=0.5)
        d = {x.name: x for x in torque_ceiling(
            est, _vehicle(), _tyre(), _physics(), _friction(), wet=True)}
        self.assertAlmostEqual(d["drive_tread_lower_bound_mm"].value, 4.0, places=6)

    def test_wet_ceiling_is_below_dry(self):
        est = _with_observed_tread(_real_estimate(), tread_mm=3.0, sigma=0.2)
        wet = torque_ceiling(est, _vehicle(), _tyre(), _physics(),
                             _friction(), wet=True)[0]
        dry = torque_ceiling(est, _vehicle(), _tyre(), _physics(),
                             _friction(), wet=False)[0]
        self.assertLess(wet.value, dry.value)

    def test_worn_tread_lowers_the_wet_ceiling(self):
        worn = _with_observed_tread(_real_estimate(), tread_mm=2.0, sigma=0.2)
        fresh = _with_observed_tread(_real_estimate(), tread_mm=7.0, sigma=0.2)
        w = torque_ceiling(worn, _vehicle(), _tyre(), _physics(), _friction())[0]
        f = torque_ceiling(fresh, _vehicle(), _tyre(), _physics(), _friction())[0]
        self.assertLess(w.value, f.value)

    def test_ceiling_carries_the_unvalidated_caveat(self):
        est = _with_observed_tread(_real_estimate())
        d = torque_ceiling(est, _vehicle(), _tyre(), _physics(), _friction())[0]
        self.assertTrue(any("UNVALIDATED" in c for c in d.caveats))


class DriveLayoutTests(unittest.TestCase):
    """Generalises legacy's hardcoded rear-drive assumption."""

    def test_rwd_uses_rear_corners(self):
        self.assertEqual(driven_corners(_vehicle(DriveLayout.RWD)), ("RL", "RR"))

    def test_fwd_uses_front_corners(self):
        self.assertEqual(driven_corners(_vehicle(DriveLayout.FWD)), ("FL", "FR"))

    def test_awd_uses_all_four(self):
        self.assertEqual(len(driven_corners(_vehicle(DriveLayout.AWD))), 4)

    def test_fwd_ceiling_differs_from_rwd(self):
        """Front/rear weight split differs, so the ceiling must too."""
        est = _with_observed_tread(_real_estimate())
        rwd = torque_ceiling(est, _vehicle(DriveLayout.RWD), _tyre(),
                             _physics(), _friction())[0]
        fwd = torque_ceiling(est, _vehicle(DriveLayout.FWD), _tyre(),
                             _physics(), _friction())[0]
        self.assertNotAlmostEqual(rwd.value, fwd.value, places=3)

    def test_awd_ceiling_exceeds_single_axle(self):
        est = _with_observed_tread(_real_estimate())
        awd = torque_ceiling(est, _vehicle(DriveLayout.AWD), _tyre(),
                             _physics(), _friction())[0]
        rwd = torque_ceiling(est, _vehicle(DriveLayout.RWD), _tyre(),
                             _physics(), _friction())[0]
        self.assertGreater(awd.value, rwd.value)


class WetFrictionTests(unittest.TestCase):
    def test_more_tread_gives_more_wet_grip(self):
        self.assertGreater(wet_friction(6.0, _friction()),
                           wet_friction(2.0, _friction()))

    def test_zero_tread_falls_to_the_floor(self):
        f = _friction()
        self.assertAlmostEqual(wet_friction(0.0, f), f.mu_dry * f.wet_floor,
                               places=9)

    def test_never_exceeds_dry(self):
        f = _friction()
        for tread in (0.0, 2.0, 5.0, 8.0, 20.0):
            self.assertLessEqual(wet_friction(tread, f), f.mu_dry + 1e-12)


class MaintenanceViewTests(unittest.TestCase):
    """The one Phase 4 output that is fully deliverable today."""

    def _temps(self, value=30.0):
        return {c: value for c in CORNERS}

    def test_cold_pressure_is_available(self):
        decisions = maintenance_view(
            _real_estimate(), _tyre(), _maintenance(), self._temps())
        cold = [d for d in decisions if d.name == "cold_equivalent_pressure_kpa"]
        self.assertEqual(len(cold), 4)
        for d in cold:
            self.assertIs(d.status, DecisionStatus.OK)

    def test_hot_tyre_reads_lower_when_normalised(self):
        """A hot tyre reads high; normalising must bring it DOWN."""
        hot = maintenance_view(
            _real_estimate(), _tyre(), _maintenance(), self._temps(50.0))
        cold_d = [d for d in hot if d.name == "cold_equivalent_pressure_kpa"][0]
        running = next(
            s.value for s in _real_estimate().states if s.name == "press_FL")
        self.assertLess(cold_d.value, running)

    def test_missing_temperature_blocks_that_corner(self):
        temps = self._temps()
        temps["FL"] = None
        decisions = maintenance_view(
            _real_estimate(), _tyre(), _maintenance(), temps)
        fl = [d for d in decisions
              if d.name == "cold_equivalent_pressure_kpa" and d.corner == "FL"][0]
        self.assertIs(fl.status, DecisionStatus.UNAVAILABLE)
        self.assertIn("temperature", fl.unavailable_reason.lower())

    def test_other_corners_survive_one_missing_temperature(self):
        temps = self._temps()
        temps["FL"] = None
        decisions = maintenance_view(
            _real_estimate(), _tyre(), _maintenance(), temps)
        fr = [d for d in decisions
              if d.name == "cold_equivalent_pressure_kpa" and d.corner == "FR"][0]
        self.assertIs(fr.status, DecisionStatus.OK)

    def test_low_pressure_is_flagged(self):
        decisions = maintenance_view(
            _real_estimate(pressure_kpa=180.0), _tyre(), _maintenance(),
            self._temps())
        deficits = [d for d in decisions if d.name == "inflation_deficit_kpa"]
        self.assertTrue(any("LOW" in c for d in deficits for c in d.caveats))

    def test_healthy_pressure_is_not_flagged(self):
        decisions = maintenance_view(
            _real_estimate(pressure_kpa=245.0), _tyre(), _maintenance(),
            self._temps(25.0))
        deficits = [d for d in decisions if d.name == "inflation_deficit_kpa"]
        self.assertFalse(any("LOW" in c for d in deficits for c in d.caveats))

    def test_gas_law_direction(self):
        """Cold-equivalent of a hot tyre is lower than its running pressure."""
        self.assertLess(cold_equivalent_pressure_kpa(240.0, 50.0, 25.0), 240.0)
        self.assertGreater(cold_equivalent_pressure_kpa(240.0, 5.0, 25.0), 240.0)


class RecoverableEnergyTests(unittest.TestCase):
    def test_pressure_component_is_available(self):
        decisions = recoverable_energy(
            _real_estimate(), _vehicle(), _tyre(), _physics(), 25.0)
        d = [x for x in decisions
             if x.name == "recoverable_energy_pressure_pct"][0]
        self.assertIs(d.status, DecisionStatus.OK)

    def test_underinflation_gives_positive_recoverable_energy(self):
        decisions = recoverable_energy(
            _real_estimate(pressure_kpa=180.0), _vehicle(), _tyre(),
            _physics(), 25.0)
        d = [x for x in decisions
             if x.name == "recoverable_energy_pressure_pct"][0]
        self.assertGreater(d.value, 0.0)

    def test_placard_pressure_gives_about_zero(self):
        decisions = recoverable_energy(
            _real_estimate(pressure_kpa=240.0), _vehicle(), _tyre(),
            _physics(), 25.0)
        d = [x for x in decisions
             if x.name == "recoverable_energy_pressure_pct"][0]
        self.assertAlmostEqual(d.value, 0.0, delta=0.5)

    def test_alignment_component_is_withheld(self):
        decisions = recoverable_energy(
            _real_estimate(), _vehicle(), _tyre(), _physics(), 25.0)
        d = [x for x in decisions
             if x.name == "recoverable_energy_alignment_pct"][0]
        self.assertIs(d.status, DecisionStatus.UNAVAILABLE)
        self.assertIn("toe", d.unavailable_reason.lower())

    def test_total_is_withheld_while_a_component_is_missing(self):
        decisions = recoverable_energy(
            _real_estimate(), _vehicle(), _tyre(), _physics(), 25.0)
        d = [x for x in decisions
             if x.name == "recoverable_energy_total_pct"][0]
        self.assertIs(d.status, DecisionStatus.UNAVAILABLE)

    def test_never_presents_tread_wear_as_recoverable(self):
        """Worn tread LOWERS rolling resistance. Counting it as waste would be
        physically backwards."""
        worn = _with_observed_tread(_real_estimate(), tread_mm=2.0)
        fresh = _with_observed_tread(_real_estimate(), tread_mm=7.0)
        w = [x for x in recoverable_energy(worn, _vehicle(), _tyre(),
                                           _physics(), 25.0)
             if x.name == "recoverable_energy_pressure_pct"][0]
        f = [x for x in recoverable_energy(fresh, _vehicle(), _tyre(),
                                           _physics(), 25.0)
             if x.name == "recoverable_energy_pressure_pct"][0]
        # Tread is held fixed in both legs, so it must cancel out entirely.
        self.assertAlmostEqual(w.value, f.value, places=6)


class MultiSnapshotFusionTests(unittest.TestCase):
    def test_single_snapshot_passes_through(self):
        est = _real_estimate()
        self.assertIs(fuse_snapshots([est]), est)

    def test_empty_is_rejected(self):
        with self.assertRaises(SnapshotFusionError):
            fuse_snapshots([])

    def test_fusing_reduces_pressure_uncertainty(self):
        snaps = [_real_estimate() for _ in range(4)]
        fused = fuse_snapshots(snaps)
        one = next(s for s in snaps[0].states if s.name == "press_FL")
        many = next(s for s in fused.states if s.name == "press_FL")
        self.assertLess(many.sigma, one.sigma)

    def test_weak_states_do_not_gain_confidence_from_repetition(self):
        """Averaging repeated priors must not manufacture certainty."""
        snaps = [_real_estimate() for _ in range(8)]
        fused = fuse_snapshots(snaps)
        one = next(s for s in snaps[0].states if s.name == "tread_FL")
        many = next(s for s in fused.states if s.name == "tread_FL")
        self.assertAlmostEqual(many.sigma, one.sigma, places=9)
        self.assertIsNot(many.observability, Observability.OBSERVED)

    def test_mixed_sources_are_refused(self):
        real = _real_estimate()
        sim = TyreStateEstimate(
            states=real.states, covariance_diag=real.covariance_diag,
            timestamp_s=11.0, odometer_km=real.odometer_km, source="real",
            model_version=real.model_version,
            config_fingerprint=real.config_fingerprint,
            n_measurements_available=real.n_measurements_available,
            n_states_observed=real.n_states_observed,
            mean_variance_reduction=real.mean_variance_reduction,
            converged=True, singular_matrix=False, iteration_count=1,
        )
        with self.assertRaises(SnapshotFusionError):
            fuse_snapshots([real, sim])

    def test_fused_result_is_stamped_at_the_newest_snapshot(self):
        a = _real_estimate()
        b = TyreStateEstimate(
            states=a.states, covariance_diag=a.covariance_diag,
            timestamp_s=99.0, odometer_km=16000.0, source=a.source,
            model_version=a.model_version,
            config_fingerprint=a.config_fingerprint,
            n_measurements_available=a.n_measurements_available,
            n_states_observed=a.n_states_observed,
            mean_variance_reduction=a.mean_variance_reduction,
            converged=True, singular_matrix=False, iteration_count=1,
        )
        fused = fuse_snapshots([a, b])
        self.assertEqual(fused.timestamp_s, 99.0)
        self.assertEqual(fused.odometer_km, 16000.0)


class FusedTyreReportTests(unittest.TestCase):
    def _report(self, **kw):
        return build_report(
            _real_estimate(**kw), _vehicle(), _tyre(), _physics(),
            _friction(), _maintenance(),
            {c: 30.0 for c in CORNERS}, 25.0,
        )

    def test_report_carries_time_and_distance_for_phase_5(self):
        r = self._report()
        self.assertEqual(r.timestamp_s, 10.0)
        self.assertEqual(r.odometer_km, 15200.0)

    def test_simulated_provenance_survives_to_the_report(self):
        r = self._report()
        self.assertEqual(r.source, "simulated")
        self.assertTrue(r.is_simulated)

    def test_report_has_both_actionable_and_withheld_decisions(self):
        r = self._report()
        self.assertGreater(len(r.actionable), 0)
        self.assertGreater(len(r.withheld), 0)

    def test_every_withheld_decision_explains_itself(self):
        for d in self._report().withheld:
            with self.subTest(decision=d.name):
                self.assertTrue(d.unavailable_reason.strip())

    def test_no_decision_is_a_bare_number(self):
        for d in self._report().decisions:
            with self.subTest(decision=d.name):
                self.assertIsInstance(d, Decision)
                if d.status is DecisionStatus.OK:
                    self.assertIsNotNone(d.value)
                else:
                    self.assertIsNone(d.value)

    def test_torque_ceiling_is_among_the_withheld(self):
        withheld = {d.name for d in self._report().withheld}
        self.assertIn("torque_ceiling_wet_nm", withheld)

    def test_lookup_by_name_and_corner(self):
        r = self._report()
        self.assertIsNotNone(
            r.by_name("cold_equivalent_pressure_kpa", corner="FL"))


if __name__ == "__main__":
    unittest.main()
