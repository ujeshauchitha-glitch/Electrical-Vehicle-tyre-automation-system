"""Phase 5 tests for temporal degradation-rate estimation.

Tests the temporal state extension, transition model, and multi-snapshot
trend estimation.
"""

import numpy as np
import pytest

from src.evtyre.estimation.temporal import (
    TemporalState,
    TemporalEstimate,
    DegradationLimits,
    DEFAULT_LIMITS,
    temporal_prior,
    temporal_transition,
    temporal_process_noise,
    estimate_temporal_trend,
)
from src.evtyre.estimation.estimator import Observability, StateEstimate
from src.evtyre.estimation.schema import TyreStateEstimate


# ===========================================================================
# TemporalState layout tests
# ===========================================================================

class TestTemporalStateLayout:
    def test_state_dimension(self):
        """Phase 5 state vector must be 18-dimensional."""
        assert TemporalState.N == 18

    def test_base_indices_match_phase3(self):
        """Base state indices must match Phase 3."""
        assert TemporalState.tread_fl == 0
        assert TemporalState.tread_fr == 1
        assert TemporalState.tread_rl == 2
        assert TemporalState.tread_rr == 3
        assert TemporalState.press_fl == 4
        assert TemporalState.press_fr == 5
        assert TemporalState.press_rl == 6
        assert TemporalState.press_rr == 7
        assert TemporalState.toe_sq == 8
        assert TemporalState.camber == 9

    def test_rate_indices(self):
        """Rate state indices must be contiguous after base states."""
        assert TemporalState.tread_rate_fl == 10
        assert TemporalState.tread_rate_fr == 11
        assert TemporalState.tread_rate_rl == 12
        assert TemporalState.tread_rate_rr == 13
        assert TemporalState.press_rate_fl == 14
        assert TemporalState.press_rate_fr == 15
        assert TemporalState.press_rate_rl == 16
        assert TemporalState.press_rate_rr == 17

    def test_slices(self):
        """Named slices must cover correct ranges."""
        assert TemporalState.TREAD == slice(0, 4)
        assert TemporalState.PRESS == slice(4, 8)
        assert TemporalState.TREAD_RATE == slice(10, 14)
        assert TemporalState.PRESS_RATE == slice(14, 18)


# ===========================================================================
# DegradationLimits tests
# ===========================================================================

class TestDegradationLimits:
    def test_default_limits_valid(self):
        """Default limits must be physically reasonable."""
        limits = DEFAULT_LIMITS
        assert limits.tread_max_rate_mm_per_1000km > 0
        assert limits.press_max_rate_kpa_per_month > 0
        assert limits.toe_max_rate_deg_per_1000km > 0

    def test_limits_are_frozen(self):
        """Limits must be immutable."""
        limits = DegradationLimits()
        with pytest.raises(AttributeError):
            limits.tread_max_rate_mm_per_1000km = 0.5


# ===========================================================================
# Temporal prior tests
# ===========================================================================

class TestTemporalPrior:
    def test_prior_shape(self):
        """Prior must have correct shape."""
        x0, P0 = temporal_prior()
        assert x0.shape == (18,)
        assert P0.shape == (18, 18)

    def test_prior_covariance_positive_definite(self):
        """Prior covariance must be positive definite."""
        _, P0 = temporal_prior()
        eigenvalues = np.linalg.eigvalsh(P0)
        assert np.all(eigenvalues >= 0)

    def test_rate_prior_zero(self):
        """Rate states must start at zero (no assumed trend)."""
        x0, _ = temporal_prior()
        assert np.all(x0[TemporalState.TREAD_RATE] == 0.0)
        assert np.all(x0[TemporalState.PRESS_RATE] == 0.0)

    def test_rate_prior_uncertainty(self):
        """Rate states must have positive uncertainty."""
        _, P0 = temporal_prior()
        for i in range(10, 18):
            assert P0[i, i] > 0


# ===========================================================================
# Temporal transition tests
# ===========================================================================

class TestTemporalTransition:
    def test_zero_elapsed_no_change(self):
        """With zero elapsed time/distance, state should not change."""
        x0, _ = temporal_prior()
        x_pred = temporal_transition(x0, dt_km=0.0, dt_months=0.0)
        np.testing.assert_array_almost_equal(x_pred, x0)

    def test_tread_wear(self):
        """Tread should decrease with negative wear rate."""
        x0, _ = temporal_prior()
        x0[TemporalState.TREAD_RATE] = -0.05  # -0.05 mm per 1000 km
        x_pred = temporal_transition(x0, dt_km=1000.0, dt_months=0.0)
        assert x_pred[TemporalState.tread_fl] < x0[TemporalState.tread_fl]

    def test_pressure_loss(self):
        """Pressure should decrease with negative loss rate."""
        x0, _ = temporal_prior()
        x0[TemporalState.PRESS_RATE] = -1.0  # -1 kPa per month
        x_pred = temporal_transition(x0, dt_km=0.0, dt_months=1.0)
        assert x_pred[TemporalState.press_fl] < x0[TemporalState.press_fl]

    def test_tread_clipped(self):
        """Tread must not go below physical minimum."""
        x0, _ = temporal_prior()
        x0[TemporalState.TREAD_RATE] = -10.0  # extreme wear
        x_pred = temporal_transition(x0, dt_km=1000.0, dt_months=0.0)
        assert np.all(x_pred[TemporalState.TREAD] >= 0.5)

    def test_pressure_clipped(self):
        """Pressure must stay within physical bounds."""
        x0, _ = temporal_prior()
        x0[TemporalState.PRESS_RATE] = -100.0  # extreme loss
        x_pred = temporal_transition(x0, dt_km=0.0, dt_months=10.0)
        assert np.all(x_pred[TemporalState.PRESS] >= 50.0)


# ===========================================================================
# Process noise tests
# ===========================================================================

class TestProcessNoise:
    def test_process_noise_shape(self):
        """Process noise must have correct shape."""
        Q = temporal_process_noise(dt_km=1000.0, dt_months=1.0)
        assert Q.shape == (18, 18)

    def test_process_noise_positive_semidefinite(self):
        """Process noise must be positive semi-definite."""
        Q = temporal_process_noise(dt_km=1000.0, dt_months=1.0)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues >= -1e-10)  # allow small numerical errors

    def test_process_noise_increases_with_elapsed(self):
        """Process noise should increase with elapsed time/distance."""
        Q1 = temporal_process_noise(dt_km=100.0, dt_months=0.1)
        Q2 = temporal_process_noise(dt_km=1000.0, dt_months=1.0)
        assert np.trace(Q2) > np.trace(Q1)


# ===========================================================================
# Temporal trend estimation tests
# ===========================================================================

class TestTemporalTrendEstimation:
    def _make_snapshot(
        self,
        tread: float,
        press: float,
        toe_sq: float = 0.1,
        sigma: float = 0.1,
        observability: Observability = Observability.OBSERVED,
    ) -> TyreStateEstimate:
        """Helper to create a TyreStateEstimate."""
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

    def test_insufficient_snapshots(self):
        """Must reject fewer than 3 snapshots."""
        snapshots = [self._make_snapshot(5.0, 240.0) for _ in range(2)]
        odometers = [1000.0 * i for i in range(2)]
        result = estimate_temporal_trend(snapshots, odometers)
        assert result is None

    def test_minimum_snapshots(self):
        """Must accept 3+ snapshots."""
        snapshots = [self._make_snapshot(5.0, 240.0) for _ in range(3)]
        odometers = [1000.0 * i for i in range(3)]
        result = estimate_temporal_trend(snapshots, odometers)
        assert result is not None
        assert result.n_snapshots_used == 3

    def test_constant_state_no_trend(self):
        """Constant state should produce zero degradation rate."""
        snapshots = [self._make_snapshot(5.0, 240.0) for _ in range(5)]
        odometers = [1000.0 * i for i in range(5)]
        result = estimate_temporal_trend(snapshots, odometers)
        assert result is not None
        # Tread rate should be approximately zero
        assert abs(result.degradation_rates.get("tread_FL", 1.0)) < 0.01

    def test_declining_tread_detected(self):
        """Declining tread should produce negative degradation rate."""
        treads = [7.0, 6.5, 6.0, 5.5, 5.0]
        snapshots = [self._make_snapshot(t, 240.0) for t in treads]
        odometers = [1000.0 * i for i in range(5)]
        result = estimate_temporal_trend(snapshots, odometers)
        assert result is not None
        # Tread rate should be negative (wear)
        assert result.degradation_rates.get("tread_FL", 0.0) < 0


# ===========================================================================
# Multiple source rejection tests
# ===========================================================================

class TestSourceRejection:
    def _make_compat_snap(self, source, fingerprint):
        """Create a minimal compatible snapshot with states."""
        states = []
        for corner in ["FL", "FR", "RL", "RR"]:
            states.append(StateEstimate(
                name=f"tread_{corner}", value=5.0, sigma=0.1,
                observability=Observability.OBSERVED, prior_value=5.0,
                prior_sigma=2.5, variance_reduction=0.5, reason=None,
            ))
        for corner in ["FL", "FR", "RL", "RR"]:
            states.append(StateEstimate(
                name=f"press_{corner}", value=240.0, sigma=0.1,
                observability=Observability.OBSERVED, prior_value=240.0,
                prior_sigma=40.0, variance_reduction=0.5, reason=None,
            ))
        states.append(StateEstimate(
            name="toe^2", value=0.1, sigma=0.1,
            observability=Observability.OBSERVED, prior_value=0.1,
            prior_sigma=0.4, variance_reduction=0.5, reason=None,
        ))
        states.append(StateEstimate(
            name="camber", value=0.0, sigma=0.1,
            observability=Observability.UNOBSERVABLE, prior_value=0.0,
            prior_sigma=1.0, variance_reduction=0.0,
            reason="unobservable",
        ))
        return TyreStateEstimate(
            states=tuple(states),
            covariance_diag=tuple(s.sigma ** 2 for s in states),
            timestamp_s=0.0, odometer_km=None, source=source,
            model_version="5.0.0", config_fingerprint=fingerprint,
            n_measurements_available=6, n_states_observed=9,
            mean_variance_reduction=0.5, converged=True,
            singular_matrix=False, iteration_count=6,
        )

    def test_reject_mixed_sources(self):
        """Must reject snapshots from different sources."""
        snap_real = self._make_compat_snap("real", "abc")
        snap_sim = self._make_compat_snap("simulated", "abc")
        with pytest.raises(ValueError, match="Cannot mix"):
            estimate_temporal_trend(
                [snap_real, snap_real, snap_sim],
                [0.0, 1000.0, 2000.0],
            )

    def test_reject_mixed_configs(self):
        """Must reject snapshots with different configurations."""
        snap_abc = self._make_compat_snap("simulated", "abc")
        snap_def = self._make_compat_snap("simulated", "def")
        with pytest.raises(ValueError, match="Cannot mix"):
            estimate_temporal_trend(
                [snap_abc, snap_abc, snap_def],
                [0.0, 1000.0, 2000.0],
            )
