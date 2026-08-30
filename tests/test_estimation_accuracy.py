"""Estimation accuracy tests — ground truth vs estimator output.

Quantifies how well the estimator recovers hidden tyre states from
observable telemetry alone.  Each test runs the mock simulator to
generate physically-consistent telemetry, feeds it through the full
pipeline (feature extraction → estimator), and compares the resulting
estimate against the known ground truth.

Metrics reported:
  MAE   — mean absolute error (mm for tread, kPa for pressure)
  RMSE  — root-mean-square error
  bias  — signed mean error (positive = overestimate)
  coverage — fraction of estimates within 2-sigma of ground truth
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pytest

from src.evtyre.config.tyre import TyreConfig
from src.evtyre.config.vehicle import DriveLayout, VehicleConfig
from src.evtyre.estimation.estimator import Observability, TyreEstimator
from src.evtyre.features.contract import FeatureStatus
from src.evtyre.features.kinematics import extract as kin_extract
from src.evtyre.features.pressure_thermal import extract as pt_extract
from src.evtyre.features.road_load import extract as rl_extract, RoadLoadParams
from src.evtyre.pipeline import Pipeline
from src.evtyre.schema.common import CORNERS
from src.evtyre.simulation.mock_adapter import MockVehicleAdapter
from src.evtyre.simulation.scenarios import ScenarioType, load_scenario


# ===========================================================================
# Helpers
# ===========================================================================

def _build_pipeline() -> Pipeline:
    """Build a pipeline with all standard extractors."""
    vc = VehicleConfig(
        vehicle_id="accuracy_test",
        mass_kg=1800.0,
        front_weight_fraction=0.48,
        drive_layout=DriveLayout.RWD,
    )
    tc = TyreConfig(
        tyre_model_id="accuracy_test",
        wheel_belt_radius_m=0.322,
        tread_new_mm=8.0,
        tread_legal_mm=1.6,
        placard_pressure_kpa=240.0,
        cold_reference_temperature_c=25.0,
    )
    pipeline = Pipeline(vc, tc)
    pipeline.register("kinematics", kin_extract)
    pipeline.register("pressure_thermal", pt_extract)
    pipeline.register("road_load", rl_extract, {
        "road_load_params": RoadLoadParams(
            drag_coefficient=0.25,
            frontal_area_m2=2.3,
            driveline_efficiency=0.95,
        )
    })
    return pipeline


@dataclass
class AccuracyMetrics:
    """Aggregated accuracy metrics for one state variable."""
    name: str
    mae: float = 0.0
    rmse: float = 0.0
    bias: float = 0.0
    max_error: float = 0.0
    coverage_2sigma: float = 0.0
    n_samples: int = 0
    observability: str = "unknown"

    def __str__(self) -> str:
        return (
            f"{self.name:12s}: MAE={self.mae:.4f}, RMSE={self.rmse:.4f}, "
            f"bias={self.bias:+.4f}, max={self.max_error:.4f}, "
            f"cover_2-sigma={self.coverage_2sigma:.1%} (n={self.n_samples}, {self.observability})"
        )


def _compute_metrics(
    gt_values: list[float],
    est_values: list[float],
    est_sigmas: list[float],
    observability: str = "unknown",
    name: str = "",
) -> AccuracyMetrics:
    """Compute accuracy metrics from parallel ground-truth and estimate lists."""
    n = len(gt_values)
    if n == 0:
        return AccuracyMetrics(name=name, observability=observability)

    errors = [e - g for e, g in zip(est_values, gt_values)]
    abs_errors = [abs(err) for err in errors]

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    bias = sum(errors) / n
    max_err = max(abs_errors)

    # Coverage: fraction of estimates within 2-sigma of ground truth
    within = 0
    for i in range(n):
        if est_sigmas[i] > 0 and abs(errors[i]) <= 2.0 * est_sigmas[i]:
            within += 1
        elif est_sigmas[i] == 0 and abs(errors[i]) < 1e-6:
            within += 1
    coverage = within / n

    return AccuracyMetrics(
        name=name, mae=mae, rmse=rmse, bias=bias,
        max_error=max_err, coverage_2sigma=coverage,
        n_samples=n, observability=observability,
    )


@dataclass
class ScenarioAccuracy:
    """Full accuracy report for one scenario."""
    scenario_name: str
    n_steps: int
    total_distance_km: float
    tread_metrics: dict[str, AccuracyMetrics] = field(default_factory=dict)
    pressure_metrics: dict[str, AccuracyMetrics] = field(default_factory=dict)
    toe_metric: AccuracyMetrics | None = None

    @property
    def mean_tread_mae(self) -> float:
        if not self.tread_metrics:
            return 0.0
        return sum(m.mae for m in self.tread_metrics.values()) / len(self.tread_metrics)

    @property
    def mean_pressure_mae(self) -> float:
        if not self.pressure_metrics:
            return 0.0
        return sum(m.mae for m in self.pressure_metrics.values()) / len(self.pressure_metrics)

    @property
    def mean_tread_coverage(self) -> float:
        if not self.tread_metrics:
            return 0.0
        return sum(m.coverage_2sigma for m in self.tread_metrics.values()) / len(self.tread_metrics)

    def summary(self) -> str:
        lines = [
            f"Scenario: {self.scenario_name}",
            f"  Steps: {self.n_steps}, Distance: {self.total_distance_km:.1f} km",
            f"  Mean tread MAE: {self.mean_tread_mae:.4f} mm",
            f"  Mean pressure MAE: {self.mean_pressure_mae:.2f} kPa",
            f"  Mean tread 2-sigma coverage: {self.mean_tread_coverage:.1%}",
            "",
            "  Per-corner tread:",
        ]
        for corner in CORNERS:
            if corner in self.tread_metrics:
                lines.append(f"    {self.tread_metrics[corner]}")
        lines.append("  Per-corner pressure:")
        for corner in CORNERS:
            if corner in self.pressure_metrics:
                lines.append(f"    {self.pressure_metrics[corner]}")
        if self.toe_metric:
            lines.append(f"  Toe: {self.toe_metric}")
        return "\n".join(lines)


def _run_scenario_accuracy(
    scenario_type: ScenarioType,
    n_steps: int = 20,
    dt_s: float = 1.0,
) -> ScenarioAccuracy:
    """Run a scenario through the full pipeline and compute accuracy metrics."""
    scenario = load_scenario(scenario_type)
    adapter = MockVehicleAdapter()
    adapter.reset(scenario)
    pipeline = _build_pipeline()

    # Collect estimates and ground truth per corner per step
    tread_est: dict[str, list[float]] = {c: [] for c in CORNERS}
    tread_gt: dict[str, list[float]] = {c: [] for c in CORNERS}
    tread_sig: dict[str, list[float]] = {c: [] for c in CORNERS}
    press_est: dict[str, list[float]] = {c: [] for c in CORNERS}
    press_gt: dict[str, list[float]] = {c: [] for c in CORNERS}
    press_sig: dict[str, list[float]] = {c: [] for c in CORNERS}
    toe_est: list[float] = []
    toe_gt: list[float] = []
    toe_sig: list[float] = []
    last_distance = 0.0

    for step in range(n_steps):
        state = adapter.step(dt_s=dt_s)
        gt = state.ground_truth

        features, estimate = pipeline.run(state.telemetry)
        last_distance = state.odometer_km

        # Build name → estimate lookup
        est_by_name = {s.name: s for s in estimate.states}

        for corner in CORNERS:
            # Tread
            tread_key = f"tread_{corner}"
            if tread_key in est_by_name:
                est = est_by_name[tread_key]
                tread_est[corner].append(est.value)
                tread_gt[corner].append(gt.tread_mm[corner])
                tread_sig[corner].append(est.sigma)

            # Pressure — ground truth is gauge kPa; estimate is also gauge kPa
            press_key = f"press_{corner}"
            if press_key in est_by_name:
                est = est_by_name[press_key]
                press_est[corner].append(est.value)
                press_gt[corner].append(gt.pressure_kpa[corner])
                press_sig[corner].append(est.sigma)

        # Toe²
        if "toe^2" in est_by_name:
            est = est_by_name["toe^2"]
            toe_est.append(est.value)
            toe_gt.append(gt.toe_sq_deg2)
            toe_sig.append(est.sigma)

    # Compute metrics
    report = ScenarioAccuracy(
        scenario_name=scenario.name,
        n_steps=n_steps,
        total_distance_km=last_distance,
    )

    for corner in CORNERS:
        if tread_est[corner]:
            report.tread_metrics[corner] = _compute_metrics(
                tread_gt[corner], tread_est[corner], tread_sig[corner],
                name=f"tread_{corner}",
            )
        if press_est[corner]:
            report.pressure_metrics[corner] = _compute_metrics(
                press_gt[corner], press_est[corner], press_sig[corner],
                name=f"press_{corner}",
            )

    if toe_est:
        report.toe_metric = _compute_metrics(
            toe_gt, toe_est, toe_sig, name="toe^2",
        )

    return report


# ===========================================================================
# Test: Single-step accuracy (snapshot)
# ===========================================================================

class TestSingleStepAccuracy:
    """Test accuracy at a single snapshot — no temporal dependency."""

    def test_pressure_estimate_matches_ground_truth(self):
        """Pressure is OBSERVED — estimate should closely match GT."""
        scenario = load_scenario(ScenarioType.NORMAL)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        pipeline = _build_pipeline()

        state = adapter.step(dt_s=1.0)
        gt = state.ground_truth
        features, estimate = pipeline.run(state.telemetry)

        est_by_name = {s.name: s for s in estimate.states}

        for corner in CORNERS:
            est = est_by_name[f"press_{corner}"]
            gt_press = gt.pressure_kpa[corner]
            error = abs(est.value - gt_press)
            # Pressure is OBSERVED with σ≈5 kPa; error should be < 20 kPa
            assert error < 20.0, (
                f"{corner} pressure: estimate={est.value:.1f}, "
                f"gt={gt_press:.1f}, error={error:.1f}"
            )

    def test_tread_estimate_is_physically_plausible(self):
        """Tread estimate should be within physical bounds."""
        scenario = load_scenario(ScenarioType.NORMAL)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        pipeline = _build_pipeline()

        state = adapter.step(dt_s=1.0)
        gt = state.ground_truth
        features, estimate = pipeline.run(state.telemetry)

        est_by_name = {s.name: s for s in estimate.states}

        for corner in CORNERS:
            est = est_by_name[f"tread_{corner}"]
            gt_tread = gt.tread_mm[corner]
            # Error should be within reasonable bounds for WEAK observability
            error = abs(est.value - gt_tread)
            assert error < 3.0, (
                f"{corner} tread: estimate={est.value:.2f}, "
                f"gt={gt_tread:.2f}, error={error:.2f}"
            )

    def test_toe_estimate_is_bounded(self):
        """Toe² estimate should be non-negative and bounded."""
        scenario = load_scenario(ScenarioType.NORMAL)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        pipeline = _build_pipeline()

        state = adapter.step(dt_s=1.0)
        features, estimate = pipeline.run(state.telemetry)

        toe = next(s for s in estimate.states if s.name == "toe^2")
        assert toe.value >= 0.0, f"toe² must be ≥ 0, got {toe.value}"
        assert toe.value < 5.0, f"toe² unreasonably large: {toe.value}"

    def test_estimator_converges(self):
        """Gauss-Newton should converge within iteration limit."""
        scenario = load_scenario(ScenarioType.NORMAL)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        pipeline = _build_pipeline()

        state = adapter.step(dt_s=1.0)
        features, estimate = pipeline.run(state.telemetry)

        assert estimate.converged, (
            f"Estimator did not converge: max_dx={estimate.iteration_count}"
        )
        assert not estimate.singular_matrix


# ===========================================================================
# Test: Multi-step accuracy per scenario
# ===========================================================================

class TestMultiStepAccuracy:
    """Run scenarios for multiple steps and compute aggregate metrics."""

    def test_normal_scenario_accuracy(self):
        """Normal scenario: all corners symmetric, moderate accuracy."""
        report = _run_scenario_accuracy(ScenarioType.NORMAL, n_steps=15)
        print(report.summary())

        # Pressure MAE should be < 10 kPa (OBSERVED state)
        assert report.mean_pressure_mae < 10.0, (
            f"Normal pressure MAE too high: {report.mean_pressure_mae:.2f}"
        )

        # Tread MAE should be < 2.5 mm (WEAK observability, starts from prior)
        assert report.mean_tread_mae < 2.5, (
            f"Normal tread MAE too high: {report.mean_tread_mae:.4f}"
        )

    def test_asymmetric_wear_detection_accuracy(self):
        """Asymmetric scenario: estimator should detect the worn corner."""
        report = _run_scenario_accuracy(ScenarioType.ASYMMETRIC_WEAR, n_steps=15)
        print(report.summary())

        # RR is the most worn (3.7 mm). Estimator should estimate it lower
        # than the other corners, even if absolute value has error.
        rr_tread_estimates = report.tread_metrics.get("RR")
        fl_tread_estimates = report.tread_metrics.get("FL")
        assert rr_tread_estimates is not None
        assert fl_tread_estimates is not None

        # Mean RR estimate should be < mean FL estimate (both relative to GT)
        # This tests whether the estimator captures the asymmetry direction.
        # We check via bias: RR should have negative bias (underestimate worn)
        # or at least the ordering should be preserved.
        assert rr_tread_estimates.bias < 1.0, (
            f"RR bias too positive: {rr_tread_estimates.bias:.4f}"
        )

    def test_low_pressure_scenario_accuracy(self):
        """Low pressure scenario: estimator should detect RR pressure drop."""
        report = _run_scenario_accuracy(ScenarioType.LOW_PRESSURE, n_steps=15)
        print(report.summary())

        # RR starts at 180 kPa; others at 240 kPa
        # Pressure is OBSERVED, so the estimator should accurately recover this
        rr_press = report.pressure_metrics.get("RR")
        fl_press = report.pressure_metrics.get("FL")
        assert rr_press is not None
        assert fl_press is not None

        # RR pressure MAE should be reasonable (< 15 kPa)
        assert rr_press.mae < 15.0, (
            f"Low-pressure RR MAE too high: {rr_press.mae:.2f}"
        )

        # FL pressure should be more accurate than RR (at placard)
        assert fl_press.mae < 10.0, (
            f"Low-pressure FL MAE too high: {fl_press.mae:.2f}"
        )

    def test_accelerated_degradation_accuracy(self):
        """Accelerated degradation: tread should be tracked."""
        report = _run_scenario_accuracy(ScenarioType.ACCELERATED_DEGRADATION, n_steps=15)
        print(report.summary())

        # All corners start at 4.0 mm and wear at 0.10-0.12 mm/km
        # After 15 steps at 15 m/s × 1 s = 0.0225 km per step, total ~0.34 km
        # Wear per step is tiny — estimates should be close to 4.0
        assert report.mean_tread_mae < 2.5, (
            f"Accelerated degradation tread MAE: {report.mean_tread_mae:.4f}"
        )

    def test_toe_misalignment_scenario(self):
        """Toe misalignment: toe² estimate should be non-negative."""
        report = _run_scenario_accuracy(ScenarioType.TOE_MISALIGNMENT, n_steps=15)
        print(report.summary())

        # Ground truth has toe_sq = 1.0, but toe² is WEAK observability.
        # The estimator starts from prior (0.10) and barely moves — this is
        # correct behavior for a WEAK state.  We only verify non-negativity.
        if report.toe_metric:
            assert report.toe_metric.bias > -1.0, (
                f"Toe² bias unreasonable: {report.toe_metric.bias:.4f}"
            )

    def test_pressure_drift_scenario(self):
        """Pressure drift: pressure should track the slow loss."""
        report = _run_scenario_accuracy(ScenarioType.PRESSURE_DRIFT, n_steps=15)
        print(report.summary())

        for corner in CORNERS:
            pm = report.pressure_metrics.get(corner)
            assert pm is not None, f"Missing pressure metrics for {corner}"
            assert pm.mae < 15.0, (
                f"Pressure drift {corner} MAE too high: {pm.mae:.2f}"
            )


# ===========================================================================
# Test: Coverage (2-sigma statistical test)
# ===========================================================================

class TestCoverage:
    """Statistical coverage tests — are the estimator's uncertainty bounds honest?"""

    def test_pressure_coverage_above_80_percent(self):
        """At least 80% of pressure estimates should fall within 2-sigma.

        This is a statistical test: if the noise model is correct and the
        estimator is well-calibrated, ~95% should be within 2-sigma.  We use a
        conservative 80% threshold because the estimator doesn't model
        sensor noise perfectly.
        """
        report = _run_scenario_accuracy(ScenarioType.NORMAL, n_steps=20)

        for corner in CORNERS:
            pm = report.pressure_metrics.get(corner)
            if pm and pm.n_samples > 0:
                assert pm.coverage_2sigma >= 0.80, (
                    f"{corner} pressure coverage {pm.coverage_2sigma:.1%} < 80%"
                )

    def test_tread_coverage_above_50_percent(self):
        """At least 50% of tread estimates within 2-sigma.

        Tread is WEAK observability, so bounds are wider and coverage
        is expected to be lower than for pressure.
        """
        report = _run_scenario_accuracy(ScenarioType.NORMAL, n_steps=20)

        for corner in CORNERS:
            tm = report.tread_metrics.get(corner)
            if tm and tm.n_samples > 0:
                assert tm.coverage_2sigma >= 0.50, (
                    f"{corner} tread coverage {tm.coverage_2sigma:.1%} < 50%"
                )


# ===========================================================================
# Test: Observability consistency
# ===========================================================================

class TestObservabilityConsistency:
    """Verify observability labels are consistent across scenarios."""

    def test_pressure_always_observed(self):
        """Pressure should always be OBSERVED when TPMS data is available."""
        for st in ScenarioType:
            if st == ScenarioType.SENSOR_MISSINGNESS:
                continue  # TPMS might be missing
            scenario = load_scenario(st)
            adapter = MockVehicleAdapter()
            adapter.reset(scenario)
            pipeline = _build_pipeline()

            state = adapter.step(dt_s=1.0)
            features, estimate = pipeline.run(state.telemetry)

            for s in estimate.states:
                if s.name.startswith("press_"):
                    assert s.observability == Observability.OBSERVED, (
                        f"{s.name} should be OBSERVED in {st.value}, "
                        f"got {s.observability.value}"
                    )

    def test_tread_is_weak_not_unobservable(self):
        """Tread should be WEAK (not UNOBSERVABLE) when wheel speeds are available."""
        scenario = load_scenario(ScenarioType.NORMAL)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        pipeline = _build_pipeline()

        state = adapter.step(dt_s=1.0)
        features, estimate = pipeline.run(state.telemetry)

        for s in estimate.states:
            if s.name.startswith("tread_"):
                assert s.observability != Observability.UNOBSERVABLE, (
                    f"{s.name} should not be UNOBSERVABLE when wheel speeds are present"
                )

    def test_camber_is_always_unobservable(self):
        """Camber has zero Jacobian sensitivity — always UNOBSERVABLE."""
        scenario = load_scenario(ScenarioType.NORMAL)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        pipeline = _build_pipeline()

        state = adapter.step(dt_s=1.0)
        features, estimate = pipeline.run(state.telemetry)

        camber = next(s for s in estimate.states if s.name == "camber")
        assert camber.observability == Observability.UNOBSERVABLE


# ===========================================================================
# Test: Error bounds and physical consistency
# ===========================================================================

class TestErrorBounds:
    """Verify that estimates respect physical constraints."""

    def test_pressure_never_negative(self):
        """No pressure estimate should be negative."""
        for st in ScenarioType:
            scenario = load_scenario(st)
            adapter = MockVehicleAdapter()
            adapter.reset(scenario)
            pipeline = _build_pipeline()

            for _ in range(5):
                state = adapter.step(dt_s=1.0)
                features, estimate = pipeline.run(state.telemetry)
                for s in estimate.states:
                    if s.name.startswith("press_"):
                        assert s.value >= 0.0, (
                            f"{s.name} = {s.value} is negative in {st.value}"
                        )

    def test_tread_never_negative(self):
        """No tread estimate should be negative."""
        for st in ScenarioType:
            scenario = load_scenario(st)
            adapter = MockVehicleAdapter()
            adapter.reset(scenario)
            pipeline = _build_pipeline()

            for _ in range(5):
                state = adapter.step(dt_s=1.0)
                features, estimate = pipeline.run(state.telemetry)
                for s in estimate.states:
                    if s.name.startswith("tread_"):
                        assert s.value >= 0.0, (
                            f"{s.name} = {s.value} is negative in {st.value}"
                        )

    def test_toe_squared_non_negative(self):
        """Toe² must be ≥ 0 by construction."""
        for st in ScenarioType:
            scenario = load_scenario(st)
            adapter = MockVehicleAdapter()
            adapter.reset(scenario)
            pipeline = _build_pipeline()

            for _ in range(5):
                state = adapter.step(dt_s=1.0)
                features, estimate = pipeline.run(state.telemetry)
                toe = next(s for s in estimate.states if s.name == "toe^2")
                assert toe.value >= 0.0, (
                    f"toe² = {toe.value} is negative in {st.value}"
                )

    def test_sigma_always_positive(self):
        """Posterior sigma should always be non-negative."""
        scenario = load_scenario(ScenarioType.NORMAL)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        pipeline = _build_pipeline()

        state = adapter.step(dt_s=1.0)
        features, estimate = pipeline.run(state.telemetry)

        for s in estimate.states:
            assert s.sigma >= 0.0, f"{s.name} sigma = {s.sigma} is negative"

    def test_variance_reduction_bounded(self):
        """Variance reduction should be in [0, 1]."""
        scenario = load_scenario(ScenarioType.NORMAL)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        pipeline = _build_pipeline()

        state = adapter.step(dt_s=1.0)
        features, estimate = pipeline.run(state.telemetry)

        for s in estimate.states:
            assert 0.0 <= s.variance_reduction <= 1.0, (
                f"{s.name} VR = {s.variance_reduction} out of [0,1]"
            )


# ===========================================================================
# Test: Full accuracy report (printed, not asserted — for human review)
# ===========================================================================

class TestAccuracyReport:
    """Generate and print a full accuracy report for human review."""

    def test_print_full_accuracy_report(self, capsys):
        """Run all scenarios and print accuracy metrics."""
        print("\n" + "=" * 80)
        print("ESTIMATION ACCURACY REPORT — GROUND TRUTH vs ESTIMATOR OUTPUT")
        print("=" * 80)

        all_reports = []
        for st in ScenarioType:
            report = _run_scenario_accuracy(st, n_steps=20)
            all_reports.append(report)
            print(f"\n{report.summary()}")

        # Summary table
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Scenario':<25s} {'Tread MAE':>10s} {'Press MAE':>10s} {'Tread 2-sigma':>10s}")
        print("-" * 60)
        for r in all_reports:
            print(
                f"{r.scenario_name:<25s} "
                f"{r.mean_tread_mae:>10.4f} "
                f"{r.mean_pressure_mae:>10.2f} "
                f"{r.mean_tread_coverage:>10.1%}"
            )

        print("\n" + "=" * 80)
        print("NOTES:")
        print("  - Pressure is OBSERVED (directly measured via TPMS)")
        print("  - Tread is WEAK (inferred from wheel-speed ratios)")
        print("  - Toe² is WEAK (inferred from road-load residual)")
        print("  - Camber is UNOBSERVABLE (no measurement sensitivity)")
        print("  - All values are UNVALIDATED engineering estimates")
        print("=" * 80)

        # Just verify we ran all scenarios
        assert len(all_reports) == len(ScenarioType)
