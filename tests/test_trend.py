"""Phase 5 tests for trend detection and forecasting.

Tests the trend detection algorithm, significance testing, and
degradation forecasting.
"""

import numpy as np
import pytest

from src.evtyre.estimation.estimator import Observability, StateEstimate
from src.evtyre.estimation.schema import TyreStateEstimate
from src.evtyre.fusion.trend import (
    TrendClassification,
    TrendReport,
    TrendResult,
    detect_trends,
    format_trend_report,
)
from src.evtyre.fusion.forecast import (
    DegradationForecast,
    MaintenanceForecast,
    Urgency,
    compute_forecasts,
    format_forecast,
    TREAD_LEGAL_LIMIT_MM,
    PRESSURE_MIN_FRACTION,
)


# ===========================================================================
# Helper functions
# ===========================================================================

def _make_snapshot(
    tread: float,
    press: float,
    toe_sq: float = 0.1,
    sigma: float = 0.1,
    observability: Observability = Observability.OBSERVED,
) -> TyreStateEstimate:
    """Create a TyreStateEstimate for testing."""
    states = []
    for corner in ["FL", "FR", "RL", "RR"]:
        states.append(StateEstimate(
            name=f"tread_{corner}",
            value=tread,
            sigma=sigma,
            observability=observability,
            prior_value=5.0,
            prior_sigma=2.5,
            variance_reduction=0.5,
            reason=None if observability == Observability.OBSERVED else "test",
        ))
    for corner in ["FL", "FR", "RL", "RR"]:
        states.append(StateEstimate(
            name=f"press_{corner}",
            value=press,
            sigma=sigma,
            observability=observability,
            prior_value=240.0,
            prior_sigma=40.0,
            variance_reduction=0.5,
            reason=None if observability == Observability.OBSERVED else "test",
        ))
    states.append(StateEstimate(
        name="toe^2",
        value=toe_sq,
        sigma=sigma,
        observability=observability,
        prior_value=0.1,
        prior_sigma=0.4,
        variance_reduction=0.5,
        reason=None if observability == Observability.OBSERVED else "test",
    ))
    states.append(StateEstimate(
        name="camber",
        value=0.0,
        sigma=sigma,
        observability=Observability.UNOBSERVABLE,
        prior_value=0.0,
        prior_sigma=1.0,
        variance_reduction=0.0,
        reason="Camber has zero Jacobian sensitivity",
    ))

    return TyreStateEstimate(
        states=tuple(states),
        covariance_diag=tuple(s.sigma ** 2 for s in states),
        timestamp_s=0.0,
        odometer_km=None,
        source="simulated",
        model_version="5.0.0",
        config_fingerprint="test",
        n_measurements_available=6,
        n_states_observed=9,
        mean_variance_reduction=0.5,
        converged=True,
        singular_matrix=False,
        iteration_count=6,
    )


# ===========================================================================
# Trend detection tests
# ===========================================================================

class TestTrendDetection:
    def test_minimum_snapshots_required(self):
        """Must reject fewer than 5 snapshots (default min_observations)."""
        snapshots = [_make_snapshot(5.0, 240.0) for _ in range(4)]
        odometers = [1000.0 * i for i in range(4)]
        with pytest.raises(ValueError, match="Need at least"):
            detect_trends(snapshots, odometers)

    def test_constant_state_no_trend(self):
        """Constant state should produce no significant trend."""
        snapshots = [_make_snapshot(5.0, 240.0) for _ in range(6)]
        odometers = [1000.0 * i for i in range(6)]
        report = detect_trends(snapshots, odometers)
        assert report.overall_classification == TrendClassification.NONE

    def test_linear_wear_detected(self):
        """Linear wear should be detected as a trend."""
        treads = [7.0, 6.5, 6.0, 5.5, 5.0, 4.5]
        snapshots = [_make_snapshot(t, 240.0) for t in treads]
        odometers = [1000.0 * i for i in range(6)]
        report = detect_trends(snapshots, odometers)
        # Tread trend should be detected
        assert report.trends["tread_FL"].r_squared > 0.9
        assert report.trends["tread_FL"].slope < 0  # negative = wear

    def test_pressure_loss_detected(self):
        """Pressure loss should be detected as a trend."""
        pressures = [240.0, 238.0, 236.0, 234.0, 232.0, 230.0]
        snapshots = [_make_snapshot(5.0, p) for p in pressures]
        odometers = [1000.0 * i for i in range(6)]
        report = detect_trends(snapshots, odometers)
        # Pressure trend should be detected
        assert report.trends["press_FL"].r_squared > 0.9
        assert report.trends["press_FL"].slope < 0  # negative = loss

    def test_outlier_detection_executes(self):
        """Outlier detection path must execute without error."""
        treads = [7.0, 6.5, 6.0, 5.5, 5.0, 2.0]
        snapshots = [_make_snapshot(t, 240.0, sigma=0.01) for t in treads]
        odometers = [1000.0 * i for i in range(6)]
        report = detect_trends(snapshots, odometers)
        trend = report.trends["tread_FL"]
        assert trend.n_observations == 6
        assert trend.residuals_std > 0

    def test_trend_report_formatting(self):
        """Formatted report should be human-readable."""
        treads = [7.0, 6.5, 6.0, 5.5, 5.0, 4.5]
        snapshots = [_make_snapshot(t, 240.0) for t in treads]
        odometers = [1000.0 * i for i in range(6)]
        report = detect_trends(snapshots, odometers)
        formatted = format_trend_report(report)
        assert "Trend Analysis Report" in formatted
        assert "Snapshots analyzed: 6" in formatted


# ===========================================================================
# Forecast tests
# ===========================================================================

class TestForecast:
    def _make_estimate_with_trend(
        self,
        tread: float = 5.0,
        press: float = 240.0,
    ) -> TyreStateEstimate:
        """Create a TyreStateEstimate for forecasting."""
        return _make_snapshot(tread, press)

    def _make_trend_report(
        self,
        tread_slope: float = -0.0005,
        press_slope: float = 0.0,
    ) -> TrendReport:
        """Create a TrendReport for forecasting."""
        trends = {}
        for corner in ["FL", "FR", "RL", "RR"]:
            trends[f"tread_{corner}"] = TrendResult(
                name=f"tread_{corner}",
                slope=tread_slope,
                slope_unit="mm/km",
                r_squared=0.9,
                p_value=0.01,
                classification=TrendClassification.NORMAL,
                confidence=0.8,
                n_observations=6,
                residuals_std=0.1,
                is_outlier=False,
            )
            trends[f"press_{corner}"] = TrendResult(
                name=f"press_{corner}",
                slope=press_slope,
                slope_unit="kPa/km",
                r_squared=0.9 if press_slope != 0 else 0.0,
                p_value=0.01 if press_slope != 0 else 1.0,
                classification=TrendClassification.NORMAL if press_slope != 0 else TrendClassification.NONE,
                confidence=0.8 if press_slope != 0 else 0.0,
                n_observations=6,
                residuals_std=0.1,
                is_outlier=False,
            )
        trends["toe^2"] = TrendResult(
            name="toe^2",
            slope=0.0,
            slope_unit="deg²/km",
            r_squared=0.0,
            p_value=1.0,
            classification=TrendClassification.NONE,
            confidence=0.0,
            n_observations=6,
            residuals_std=0.1,
            is_outlier=False,
        )
        trends["camber"] = TrendResult(
            name="camber",
            slope=0.0,
            slope_unit="deg/km",
            r_squared=0.0,
            p_value=1.0,
            classification=TrendClassification.NONE,
            confidence=0.0,
            n_observations=0,
            residuals_std=0.0,
            is_outlier=False,
        )

        return TrendReport(
            trends=trends,
            overall_classification=TrendClassification.NORMAL if tread_slope != 0 else TrendClassification.NONE,
            mean_confidence=0.8 if tread_slope != 0 else 0.0,
            n_states_with_trends=4 if tread_slope != 0 else 0,
            n_snapshots=6,
            odometer_range_km=(0.0, 5000.0),
        )

    def test_forecast_with_trend(self):
        """Forecast should use detected trend."""
        estimate = self._make_estimate_with_trend(tread=5.0, press=240.0)
        trend = self._make_trend_report(tread_slope=-0.0005)
        forecast = compute_forecasts(estimate, trend, current_odometer_km=50000.0)
        assert forecast.n_parameters_forecast > 0
        # Tread should have a forecast
        assert "tread_FL" in forecast.forecasts
        # Should have positive remaining km (still above legal limit)
        assert forecast.forecasts["tread_FL"].remaining_km > 0

    def test_forecast_no_trend(self):
        """Forecast without trend should be inf."""
        estimate = self._make_estimate_with_trend(tread=5.0, press=240.0)
        trend = self._make_trend_report(tread_slope=0.0)
        forecast = compute_forecasts(estimate, trend, current_odometer_km=50000.0)
        # Without trend, remaining should be inf
        assert forecast.forecasts["tread_FL"].remaining_km == float('inf')

    def test_urgency_classification(self):
        """Urgency should be classified correctly."""
        estimate = self._make_estimate_with_trend(tread=5.0, press=240.0)
        # Very fast wear → high urgency
        trend = self._make_trend_report(tread_slope=-0.001)  # 1 mm per 1000 km
        forecast = compute_forecasts(estimate, trend, current_odometer_km=50000.0)
        # With 5mm remaining and -0.001 mm/km, ~5000 km left → HIGH urgency
        assert forecast.forecasts["tread_FL"].urgency in [Urgency.HIGH, Urgency.CRITICAL]

    def test_forecast_formatting(self):
        """Formatted forecast should be human-readable."""
        estimate = self._make_estimate_with_trend(tread=5.0, press=240.0)
        trend = self._make_trend_report(tread_slope=-0.0005)
        forecast = compute_forecasts(estimate, trend, current_odometer_km=50000.0)
        formatted = format_forecast(forecast)
        assert "Degradation Forecast" in formatted
        assert "Current odometer: 50000 km" in formatted


# ===========================================================================
# Significance testing tests
# ===========================================================================

class TestSignificanceTesting:
    def test_insufficient_data_not_significant(self):
        """Trend should not be significant with insufficient data."""
        # Only 5 snapshots (minimum) with noise
        rng = np.random.default_rng(42)
        treads = [5.0 + 0.1 * rng.normal() for _ in range(6)]
        snapshots = [_make_snapshot(t, 240.0) for t in treads]
        odometers = [1000.0 * i for i in range(6)]
        report = detect_trends(snapshots, odometers)
        # With noise, trend should not be significant
        assert report.trends["tread_FL"].p_value > 0.05

    def test_strong_signal_significant(self):
        """Strong linear signal should be significant."""
        treads = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
        snapshots = [_make_snapshot(t, 240.0, sigma=0.01) for t in treads]
        odometers = [1000.0 * i for i in range(6)]
        report = detect_trends(snapshots, odometers)
        # Strong signal should be significant
        assert report.trends["tread_FL"].p_value < 0.05
        assert report.trends["tread_FL"].r_squared > 0.99
