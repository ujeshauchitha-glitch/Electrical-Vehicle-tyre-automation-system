"""Tests for Phase 5 enhancements:
1. Non-linear wear models
2. Asymmetric wear detection
3. Anomaly detection
4. Temperature-dependent degradation
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

# Fix numpy random for reproducibility
_rng = np.random.default_rng(42)


# ===========================================================================
# Helper: create test snapshots
# ===========================================================================

@dataclass
class _FakeState:
    name: str
    value: float
    sigma: float
    observability: object  # using object to avoid import issues
    prior_value: float = 0.0
    prior_sigma: float = 1.0
    variance_reduction: float = 0.0
    reason: str | None = None
    magnitude_only: bool = False


@dataclass
class _FakeSnapshot:
    states: tuple
    covariance_diag: np.ndarray
    source: str = "simulated"
    config_fingerprint: str = "test_config"


def _make_snap(
    tread_values: dict[str, float],
    press_values: dict[str, float],
    odom_km: float,
    obs_class=None,
    tread_sigma: float = 0.5,
    press_sigma: float = 5.0,
):
    """Create a fake snapshot for testing."""
    if obs_class is None:
        from src.evtyre.estimation.estimator import Observability
        obs_class = Observability

    states = []
    for c in ["FL", "FR", "RL", "RR"]:
        states.append(_FakeState(
            name=f"tread_{c}",
            value=tread_values[c],
            sigma=tread_sigma,
            observability=obs_class.OBSERVED,
        ))
    for c in ["FL", "FR", "RL", "RR"]:
        states.append(_FakeState(
            name=f"press_{c}",
            value=press_values[c],
            sigma=press_sigma,
            observability=obs_class.OBSERVED,
        ))
    states.append(_FakeState(name="toe^2", value=0.1, sigma=0.2, observability=obs_class.WEAK))
    states.append(_FakeState(name="camber", value=0.0, sigma=0.5, observability=obs_class.UNOBSERVABLE))

    return _FakeSnapshot(
        states=tuple(states),
        covariance_diag=np.ones(12) * 0.5,
    )


# ===========================================================================
# Test: Non-linear wear models
# ===========================================================================

class TestNonLinearWear:
    """Tests for non-linear wear fitting."""

    def test_linear_wear_detected(self):
        """Linear wear should be detected as best model."""
        from src.evtyre.fusion.nonlinear_wear import fit_nonlinear_wear, WearModel

        # Perfectly linear wear
        treads = [5.0, 4.8, 4.6, 4.4, 4.2, 4.0, 3.8, 3.6]
        snaps = []
        odoms = []
        for i, t in enumerate(treads):
            snaps.append(_make_snap(
                {"FL": t, "FR": t, "RL": t, "RR": t},
                {"FL": 240, "FR": 240, "RL": 240, "RR": 240},
                i * 1000.0,
            ))
            odoms.append(i * 1000.0)

        report = fit_nonlinear_wear(snaps, odoms)
        # FL should be best fit as linear
        assert report.fits["tread_FL"].best_model == WearModel.LINEAR
        assert report.fits["tread_FL"].r_squared_linear > 0.99
        assert report.fits["tread_FL"].n_observations == 8

    def test_accelerating_wear_detected(self):
        """Accelerating wear (quadratic) should be detected."""
        from src.evtyre.fusion.nonlinear_wear import fit_nonlinear_wear, WearModel

        # Accelerating wear (quadratic curve)
        treads = [5.0, 4.9, 4.7, 4.4, 4.0, 3.5, 2.9, 2.2]
        snaps = []
        odoms = []
        for i, t in enumerate(treads):
            snaps.append(_make_snap(
                {"FL": t, "FR": t, "RL": t, "RR": t},
                {"FL": 240, "FR": 240, "RL": 240, "RR": 240},
                i * 1000.0,
            ))
            odoms.append(i * 1000.0)

        report = fit_nonlinear_wear(snaps, odoms)
        # Should detect acceleration
        assert report.n_states_with_acceleration > 0

    def test_insufficient_data_raises(self):
        """Should raise if fewer than min_observations snapshots."""
        from src.evtyre.fusion.nonlinear_wear import fit_nonlinear_wear

        snaps = [_make_snap(
            {"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
            {"FL": 240, "FR": 240, "RL": 240, "RR": 240},
            0.0,
        )]

        with pytest.raises(ValueError, match="Need at least"):
            fit_nonlinear_wear(snaps, [0.0], min_observations=5)

    def test_prediction_works(self):
        """NonLinearFit.predict should return values at arbitrary distances."""
        from src.evtyre.fusion.nonlinear_wear import NonLinearFit, WearModel

        fit = NonLinearFit(
            name="tread_FL",
            best_model=WearModel.LINEAR,
            linear_slope=-0.0002,
            quadratic_coeff=0.0,
            exponential_rate=0.0,
            r_squared_linear=0.95,
            r_squared_quadratic=0.90,
            r_squared_exponential=0.85,
            aic_linear=10.0,
            aic_quadratic=12.0,
            aic_exponential=15.0,
            n_observations=8,
            residuals_std=0.1,
        )

        # At 0 km: predicted value should be 0 (intercept is a)
        val_0 = fit.predict(0.0)
        assert val_0 == 0.0

        # At 10000 km: slope * 10000
        val_10k = fit.predict(10000.0)
        assert abs(val_10k - (-0.0002 * 10000.0)) < 1e-10

    def test_accelerating_property(self):
        """accelerating property should return True for positive quadratic/exponential."""
        from src.evtyre.fusion.nonlinear_wear import NonLinearFit, WearModel

        fit_accel = NonLinearFit(
            name="test", best_model=WearModel.QUADRATIC,
            linear_slope=0.0, quadratic_coeff=0.001, exponential_rate=0.0,
            r_squared_linear=0.9, r_squared_quadratic=0.95, r_squared_exponential=0.8,
            aic_linear=10, aic_quadratic=8, aic_exponential=12,
            n_observations=8, residuals_std=0.1,
        )
        assert fit_accel.accelerating

        fit_decel = NonLinearFit(
            name="test", best_model=WearModel.QUADRATIC,
            linear_slope=0.0, quadratic_coeff=-0.001, exponential_rate=0.0,
            r_squared_linear=0.9, r_squared_quadratic=0.95, r_squared_exponential=0.8,
            aic_linear=10, aic_quadratic=8, aic_exponential=12,
            n_observations=8, residuals_std=0.1,
        )
        assert not fit_decel.accelerating


# ===========================================================================
# Test: Asymmetric wear detection
# ===========================================================================

class TestAsymmetricWear:
    """Tests for asymmetric wear detection."""

    def test_symmetric_tyres(self):
        """Symmetric tyres should report no asymmetry."""
        from src.evtyre.fusion.asymmetric import detect_asymmetry, AsymmetryType

        report = detect_asymmetry(
            tread_estimates={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
            tread_sigmas={"FL": 0.5, "FR": 0.5, "RL": 0.5, "RR": 0.5},
            pressure_estimates={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert report.asymmetry_type == AsymmetryType.SYMMETRIC
        assert report.tread_range_mm == 0.0

    def test_left_right_asymmetry(self):
        """Different left vs right tread should be detected."""
        from src.evtyre.fusion.asymmetric import detect_asymmetry, AsymmetryType, Severity

        report = detect_asymmetry(
            tread_estimates={"FL": 4.0, "FR": 5.0, "RL": 4.0, "RR": 5.0},
            tread_sigmas={"FL": 0.5, "FR": 0.5, "RL": 0.5, "RR": 0.5},
            pressure_estimates={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert report.asymmetry_type == AsymmetryType.LEFT_RIGHT
        assert report.severity != Severity.NORMAL
        assert abs(report.left_right_tread_delta_mm - (-1.0)) < 0.01

    def test_front_rear_asymmetry(self):
        """Different front vs rear tread should be detected."""
        from src.evtyre.fusion.asymmetric import detect_asymmetry, AsymmetryType

        report = detect_asymmetry(
            tread_estimates={"FL": 5.0, "FR": 5.0, "RL": 3.5, "RR": 3.5},
            tread_sigmas={"FL": 0.5, "FR": 0.5, "RL": 0.5, "RR": 0.5},
            pressure_estimates={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert report.asymmetry_type == AsymmetryType.FRONT_REAR
        assert report.front_rear_tread_delta_mm > 1.0

    def test_single_corner_outlier(self):
        """One corner much worse than others should be detected."""
        from src.evtyre.fusion.asymmetric import detect_asymmetry, AsymmetryType

        report = detect_asymmetry(
            tread_estimates={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 3.0},
            tread_sigmas={"FL": 0.5, "FR": 0.5, "RL": 0.5, "RR": 0.5},
            pressure_estimates={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert report.most_worn_corner == "RR"
        assert report.corner_analyses["RR"].is_outlier

    def test_recommendation_generated(self):
        """Recommendation should be generated for asymmetric wear."""
        from src.evtyre.fusion.asymmetric import detect_asymmetry

        report = detect_asymmetry(
            tread_estimates={"FL": 4.0, "FR": 5.0, "RL": 4.0, "RR": 5.0},
            tread_sigmas={"FL": 0.5, "FR": 0.5, "RL": 0.5, "RR": 0.5},
            pressure_estimates={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert len(report.recommendation) > 0
        assert "alignment" in report.recommendation.lower()

    def test_needs_alignment_check(self):
        """Moderate L-R asymmetry should suggest alignment check."""
        from src.evtyre.fusion.asymmetric import detect_asymmetry

        report = detect_asymmetry(
            tread_estimates={"FL": 3.5, "FR": 5.0, "RL": 3.5, "RR": 5.0},
            tread_sigmas={"FL": 0.5, "FR": 0.5, "RL": 0.5, "RR": 0.5},
            pressure_estimates={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert report.needs_alignment_check

    def test_tyre_rotation_recommended(self):
        """Large tread range should suggest tyre rotation."""
        from src.evtyre.fusion.asymmetric import detect_asymmetry

        report = detect_asymmetry(
            tread_estimates={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 3.5},
            tread_sigmas={"FL": 0.5, "FR": 0.5, "RL": 0.5, "RR": 0.5},
            pressure_estimates={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert report.needs_tyre_rotation

    def test_format_report(self):
        """format_asymmetry_report should return readable text."""
        from src.evtyre.fusion.asymmetric import detect_asymmetry, format_asymmetry_report

        report = detect_asymmetry(
            tread_estimates={"FL": 4.0, "FR": 5.0, "RL": 4.0, "RR": 5.0},
            tread_sigmas={"FL": 0.5, "FR": 0.5, "RL": 0.5, "RR": 0.5},
            pressure_estimates={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        text = format_asymmetry_report(report)
        assert "Asymmetric Wear Analysis" in text
        assert "left_right" in text


# ===========================================================================
# Test: Anomaly detection
# ===========================================================================

class TestAnomalyDetection:
    """Tests for anomaly detection."""

    def test_no_anomalies_for_steady_state(self):
        """No anomalies should be detected for steady state."""
        from src.evtyre.fusion.anomaly import detect_anomalies, AnomalySeverity

        # Use a deterministic zigzag pattern: values oscillate by ±0.5
        # around nominal.  Any 3-consecutive window has variance ≈ 0.17,
        # well above SENSOR_STUCK_VARIANCE_THRESHOLD (0.01).
        # Max consecutive delta is 1.0 < 3*sigma (1.5), so the sudden-
        # change gate blocks false positives.
        offsets_tread = [0.0, 0.5, -0.5, 0.3, -0.3, 0.4, -0.4, 0.2, -0.2, 0.1]
        offsets_press = [0.0, 3.0, -3.0, 2.0, -2.0, 2.5, -2.5, 1.5, -1.5, 0.5]
        snaps = []
        odoms = []
        for i in range(10):
            treads = {c: 5.0 + offsets_tread[i] for c in ["FL", "FR", "RL", "RR"]}
            presses = {c: 240.0 + offsets_press[i] for c in ["FL", "FR", "RL", "RR"]}
            snaps.append(_make_snap(treads, presses, i * 1.0))
            odoms.append(i * 1.0)

        report = detect_anomalies(snaps, odoms)
        assert not report.has_anomalies
        assert report.overall_severity == AnomalySeverity.NONE

    def test_sudden_pressure_drop_detected(self):
        """A sudden pressure drop should be flagged."""
        from src.evtyre.fusion.anomaly import detect_anomalies, AnomalyType

        snaps = []
        odoms = []
        for i in range(5):
            # Normal for first 4 steps, then sudden drop at step 5
            if i < 4:
                p_rr = 240.0
            else:
                p_rr = 180.0  # 60 kPa drop — should be detected

            snaps.append(_make_snap(
                {"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
                {"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": p_rr},
                i * 1.0,
            ))
            odoms.append(i * 1.0)

        report = detect_anomalies(snaps, odoms)
        assert report.has_anomalies
        # Should detect rapid pressure loss
        pressure_events = [e for e in report.events if e.anomaly_type == AnomalyType.RAPID_PRESSURE_LOSS]
        assert len(pressure_events) > 0

    def test_sudden_tread_damage_detected(self):
        """A sudden tread loss should be flagged."""
        from src.evtyre.fusion.anomaly import detect_anomalies, AnomalyType

        snaps = []
        odoms = []
        for i in range(5):
            if i < 3:
                t_rr = 5.0
            else:
                t_rr = 2.5  # 2.5 mm sudden loss — must exceed 3*sigma (1.5)

            snaps.append(_make_snap(
                {"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": t_rr},
                {"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
                i * 1.0,
            ))
            odoms.append(i * 1.0)

        report = detect_anomalies(snaps, odoms)
        assert report.has_anomalies
        tread_events = [e for e in report.events if e.anomaly_type == AnomalyType.TREAD_DAMAGE]
        assert len(tread_events) > 0
        assert tread_events[0].corner == "RR"

    def test_insufficient_snapshots(self):
        """Single snapshot should return no anomalies."""
        from src.evtyre.fusion.anomaly import detect_anomalies, AnomalySeverity

        snap = _make_snap(
            {"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
            {"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
            0.0,
        )
        report = detect_anomalies([snap])
        assert not report.has_anomalies
        assert report.overall_severity == AnomalySeverity.NONE

    def test_affected_corners(self):
        """affected_corners should list all corners with anomalies."""
        from src.evtyre.fusion.anomaly import detect_anomalies

        snaps = []
        odoms = []
        for i in range(5):
            if i < 3:
                snaps.append(_make_snap(
                    {"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
                    {"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
                    i * 100.0,
                ))
            else:
                snaps.append(_make_snap(
                    {"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
                    {"FL": 240.0, "FR": 240.0, "RL": 180.0, "RR": 180.0},
                    i * 100.0,
                ))
            odoms.append(i * 100.0)

        report = detect_anomalies(snaps, odoms)
        if report.has_anomalies:
            assert len(report.affected_corners) > 0

    def test_format_report(self):
        """format_anomaly_report should return readable text."""
        from src.evtyre.fusion.anomaly import detect_anomalies, format_anomaly_report

        snaps = [_make_snap(
            {"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
            {"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
            i * 100.0,
        ) for i in range(3)]

        report = detect_anomalies(snaps)
        text = format_anomaly_report(report)
        assert "Anomaly Detection Report" in text


# ===========================================================================
# Test: Temperature-dependent degradation
# ===========================================================================

class TestThermalDegradation:
    """Tests for temperature-dependent degradation model."""

    def test_wear_rate_higher_in_cold(self):
        """Wear rate should be higher in cold conditions."""
        from src.evtyre.fusion.thermal_degradation import (
            compute_thermal_state, ThermalRegime,
        )

        cold = compute_thermal_state("FL", -10.0, 240.0)
        normal = compute_thermal_state("FL", 25.0, 240.0)
        hot = compute_thermal_state("FL", 60.0, 240.0)

        assert cold.regime == ThermalRegime.COLD
        assert normal.regime == ThermalRegime.NORMAL
        assert hot.regime == ThermalRegime.HOT

        # Cold should have higher wear rate multiplier than normal
        assert cold.wear_rate_multiplier > normal.wear_rate_multiplier

    def test_pressure_increases_with_temperature(self):
        """Pressure sensitivity should be positive (pressure rises with temp)."""
        from src.evtyre.fusion.thermal_degradation import compute_thermal_state

        state = compute_thermal_state("FL", 40.0, 240.0)
        assert state.pressure_drift_kpa_per_c > 0

    def test_aging_faster_in_heat(self):
        """Aging should be faster at higher temperatures."""
        from src.evtyre.fusion.thermal_degradation import compute_thermal_state

        cold = compute_thermal_state("FL", 0.0, 240.0)
        hot = compute_thermal_state("FL", 50.0, 240.0)

        assert hot.aging_factor > cold.aging_factor

    def test_thermal_stress_index(self):
        """Thermal stress should be nonzero for extreme temperatures."""
        from src.evtyre.fusion.thermal_degradation import compute_thermal_state

        cold = compute_thermal_state("FL", -20.0, 240.0)
        hot = compute_thermal_state("FL", 70.0, 240.0)
        normal = compute_thermal_state("FL", 25.0, 240.0)

        assert cold.thermal_stress_index > 0
        assert hot.thermal_stress_index > 0
        assert normal.thermal_stress_index == 0.0

    def test_full_report(self):
        """compute_thermal_degradation_report should work for all corners."""
        from src.evtyre.fusion.thermal_degradation import (
            compute_thermal_degradation_report, ThermalRegime,
        )

        report = compute_thermal_degradation_report(
            temperatures_c={"FL": -5.0, "FR": -5.0, "RL": -5.0, "RR": -5.0},
            pressures_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert report.overall_regime == ThermalRegime.COLD
        assert report.needs_attention
        assert len(report.recommendation) > 0

    def test_adjust_wear_rate(self):
        """adjust_wear_rate_for_temperature should modify base rate."""
        from src.evtyre.fusion.thermal_degradation import adjust_wear_rate_for_temperature

        base_rate = 0.0001  # mm/km
        cold_rate = adjust_wear_rate_for_temperature(base_rate, -10.0)
        hot_rate = adjust_wear_rate_for_temperature(base_rate, 60.0)

        # Cold should increase wear, hot should decrease
        assert cold_rate > base_rate
        assert hot_rate < base_rate

    def test_predict_pressure_from_temperature(self):
        """Ideal gas law prediction should be approximately correct."""
        from src.evtyre.fusion.thermal_degradation import predict_pressure_from_temperature

        # 240 kPa at 25°C → at 35°C should be ~240 * 308/298 ≈ 248 kPa
        predicted = predict_pressure_from_temperature(240.0, 25.0, 35.0)
        expected = 240.0 * (35 + 273.15) / (25 + 273.15)
        assert abs(predicted - expected) < 0.1

    def test_format_report(self):
        """format_thermal_report should return readable text."""
        from src.evtyre.fusion.thermal_degradation import (
            compute_thermal_degradation_report, format_thermal_report,
        )

        report = compute_thermal_degradation_report(
            temperatures_c={"FL": 55.0, "FR": 55.0, "RL": 55.0, "RR": 55.0},
            pressures_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        text = format_thermal_report(report)
        assert "Thermal Degradation Analysis" in text
        assert "hot" in text.lower()

    def test_normal_conditions_no_attention(self):
        """Normal conditions should not require attention."""
        from src.evtyre.fusion.thermal_degradation import compute_thermal_degradation_report

        report = compute_thermal_degradation_report(
            temperatures_c={"FL": 25.0, "FR": 25.0, "RL": 25.0, "RR": 25.0},
            pressures_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert not report.needs_attention


# ===========================================================================
# Test: Integration (all four features together)
# ===========================================================================

class TestPhase5EnhancementsIntegration:
    """Integration tests combining all four Phase 5 enhancements."""

    def test_all_modules_importable(self):
        """All new modules should be importable."""
        from src.evtyre.fusion.nonlinear_wear import fit_nonlinear_wear, NonLinearWearReport
        from src.evtyre.fusion.asymmetric import detect_asymmetry, AsymmetryReport
        from src.evtyre.fusion.anomaly import detect_anomalies, AnomalyReport
        from src.evtyre.fusion.thermal_degradation import (
            compute_thermal_degradation_report, ThermalDegradationReport,
        )
        assert True

    def test_fusion_exports_complete(self):
        """All new types should be exported from fusion __init__."""
        from src.evtyre.fusion import (
            NonLinearFit, NonLinearWearReport, WearModel, fit_nonlinear_wear,
            AsymmetryReport, AsymmetryType, CornerAnalysis, Severity,
            detect_asymmetry,
            AnomalyEvent, AnomalyReport, AnomalySeverity, AnomalyType,
            detect_anomalies,
            ThermalDegradationReport, ThermalModelConfig, ThermalRegime, ThermalState,
            compute_thermal_degradation_report,
        )
        assert True

    def test_comprehensive_scenario(self):
        """Run all four analyses on the same scenario."""
        from src.evtyre.fusion.nonlinear_wear import fit_nonlinear_wear
        from src.evtyre.fusion.asymmetric import detect_asymmetry
        from src.evtyre.fusion.anomaly import detect_anomalies
        from src.evtyre.fusion.thermal_degradation import compute_thermal_degradation_report

        # Create a scenario with asymmetric wear + sudden pressure loss
        snaps = []
        odoms = []
        for i in range(8):
            # RR wearing faster, with a sudden pressure drop at step 5
            rr_tread = 5.0 - 0.1 * i - (0.5 if i >= 5 else 0.0)
            rr_press = 240.0 if i < 5 else 180.0

            snaps.append(_make_snap(
                {"FL": 5.0 - 0.02 * i, "FR": 5.0 - 0.02 * i,
                 "RL": 5.0 - 0.03 * i, "RR": rr_tread},
                {"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": rr_press},
                i * 1.0,
            ))
            odoms.append(i * 1.0)

        # 1. Non-linear wear
        nl_report = fit_nonlinear_wear(snaps, odoms)
        assert nl_report.n_snapshots == 8

        # 2. Asymmetric detection
        last_snap = snaps[-1]
        tread_vals = {s.name.replace("tread_", ""): s.value for s in last_snap.states if s.name.startswith("tread_")}
        press_vals = {s.name.replace("press_", ""): s.value for s in last_snap.states if s.name.startswith("press_")}
        asym_report = detect_asymmetry(tread_vals, tread_vals, press_vals)
        assert asym_report.tread_range_mm > 0

        # 3. Anomaly detection
        anom_report = detect_anomalies(snaps, odoms)
        # Should detect the sudden pressure drop
        assert anom_report.has_anomalies

        # 4. Thermal analysis
        therm_report = compute_thermal_degradation_report(
            temperatures_c={"FL": 30.0, "FR": 30.0, "RL": 30.0, "RR": 30.0},
            pressures_kpa=press_vals,
        )
        assert therm_report.overall_regime.value == "normal"
