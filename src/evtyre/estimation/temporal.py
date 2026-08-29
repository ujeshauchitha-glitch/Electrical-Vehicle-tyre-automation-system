"""Phase 5 — Temporal degradation-rate estimation.

Extends the Phase 3 state vector to include degradation rates, enabling
the estimator to track HOW STATE CHANGES OVER TIME (wear rate, pressure
loss rate, alignment drift).

CRITICAL DISTINCTION from Phase 4 multi-snapshot fusion:
  Fusion (Phase 4): "What is the state right now?" — reduces noise
  Degradation (Phase 5): "How fast is the state changing?" — detects trends

These are fundamentally different. Fusion averages noise from repeated
snapshots; degradation tracks actual physical trends. Conflating the two
is the specific error warned about in CLAUDE.md.

State vector (Phase 3): x = [tread(4), press(4), toe², camber]  (N=10)
State vector (Phase 5): x = [tread(4), press(4), toe², camber, tread_rate(4), press_rate(4)]  (N=18)

The rate states are modeled as slowly varying:
  tread(t+Δt) = tread(t) + tread_rate(t) * Δt
  press(t+Δt) = press(t) + press_rate(t) * Δt
  rate(t+Δt) = rate(t) + process_noise  (random walk)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np

from .estimator import Observability, StateEstimate
from .schema import TyreStateEstimate
from ..schema.common import CORNERS


# ===========================================================================
# Extended state layout (Phase 5)
# ===========================================================================

class TemporalState:
    """Index layout for the extended Phase 5 state vector.

    Inherits base indices from Phase 3 STATE and adds rate states.
    """
    # Phase 3 base states (indices 0-9)
    N_BASE = 10
    tread_fl = 0
    tread_fr = 1
    tread_rl = 2
    tread_rr = 3
    press_fl = 4
    press_fr = 5
    press_rl = 6
    press_rr = 7
    toe_sq = 8
    camber = 9

    # Phase 5 rate states (indices 10-17)
    tread_rate_fl = 10
    tread_rate_fr = 11
    tread_rate_rl = 12
    tread_rate_rr = 13
    press_rate_fl = 14
    press_rate_fr = 15
    press_rate_rl = 16
    press_rate_rr = 17

    N = 18  # total state dimension

    # Named slices for convenience
    TREAD = slice(0, 4)
    PRESS = slice(4, 8)
    TOE_SQ = 8
    CAMBER = 9
    TREAD_RATE = slice(10, 14)
    PRESS_RATE = slice(14, 18)


# State names for human-readable output
TEMPORAL_STATE_NAMES: list[str] = (
    [f"tread_{c}" for c in CORNERS]
    + [f"press_{c}" for c in CORNERS]
    + ["toe^2", "camber"]
    + [f"tread_rate_{c}" for c in CORNERS]
    + [f"press_rate_{c}" for c in CORNERS]
)


# ===========================================================================
# Degradation rate limits (UNVALIDATED — engineering estimates)
# ===========================================================================

@dataclass(frozen=True)
class DegradationLimits:
    """Physical bounds on degradation rates.

    All values are UNVALIDATED engineering estimates. These are not
    derived from empirical data — they represent expected ranges for
    normal driving conditions.
    """
    # Tread wear: typical 0.01-0.1 mm per 1000 km
    tread_max_rate_mm_per_1000km: float = 0.2
    """Maximum expected tread wear rate (mm per 1000 km)."""

    # Pressure loss: typical 0.5-2 kPa per month (slow leak)
    press_max_rate_kpa_per_month: float = 5.0
    """Maximum expected pressure loss rate (kPa per month)."""

    # Toe drift: typically 0.01-0.05 deg per 1000 km
    toe_max_rate_deg_per_1000km: float = 0.1
    """Maximum expected toe drift rate (deg per 1000 km)."""

    # Conversion factors
    KM_PER_MONTH: float = 1000.0
    """Assumed km per month for rate conversion (UNVALIDATED)."""


DEFAULT_LIMITS = DegradationLimits()


# ===========================================================================
# Temporal prior
# ===========================================================================

def temporal_prior(
    tread_new_mm: float = 7.5,
    placard_pressure_kpa: float = 240.0,
    limits: DegradationLimits = DEFAULT_LIMITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x0, P0) for the extended Phase 5 state.

    The prior for rate states is zero (no assumed trend) with wide
    uncertainty reflecting our ignorance of the actual wear rate.
    """
    x0 = np.zeros(TemporalState.N)

    # Base states: same as Phase 3
    mid_tread = (tread_new_mm + 1.6) / 2.0  # midpoint between new and legal
    x0[TemporalState.TREAD] = mid_tread
    x0[TemporalState.PRESS] = placard_pressure_kpa
    x0[TemporalState.TOE_SQ] = 0.10
    x0[TemporalState.CAMBER] = 0.0

    # Rate states: zero prior (no assumed trend)
    x0[TemporalState.TREAD_RATE] = 0.0
    x0[TemporalState.PRESS_RATE] = 0.0

    # Prior covariance
    P0 = np.eye(TemporalState.N) * 1e-10  # small diagonal to avoid singular

    # Base state uncertainties (same as Phase 3)
    for i in range(4):
        P0[i, i] = 2.5 ** 2
        P0[4 + i, 4 + i] = 40.0 ** 2
    P0[TemporalState.TOE_SQ, TemporalState.TOE_SQ] = 0.40 ** 2
    P0[TemporalState.CAMBER, TemporalState.CAMBER] = 1.0 ** 2

    # Rate state uncertainties (UNVALIDATED — wide priors)
    # Tread rate: ±0.1 mm per 1000 km (1σ)
    for i in range(4):
        P0[10 + i, 10 + i] = 0.1 ** 2

    # Pressure rate: ±2 kPa per month (1σ)
    for i in range(4):
        P0[14 + i, 14 + i] = 2.0 ** 2

    return x0, P0


# ===========================================================================
# Temporal state transition
# ===========================================================================

def temporal_transition(
    x: np.ndarray,
    dt_km: float,
    dt_months: float,
    limits: DegradationLimits = DEFAULT_LIMITS,
) -> np.ndarray:
    """Apply temporal state transition: x(t+Δt) = F * x(t).

    Parameters
    ----------
    x : (18,) state vector
    dt_km : distance elapsed (km)
    dt_months : time elapsed (months)
    limits : degradation rate limits

    Returns
    -------
    (18,) predicted state at t+Δt
    """
    x_pred = x.copy()

    # Tread evolves with distance: tread(t+Δt) = tread(t) + rate(t) * Δt
    x_pred[TemporalState.TREAD] = (
        x[TemporalState.TREAD] + x[TemporalState.TREAD_RATE] * dt_km
    )

    # Pressure evolves with time: press(t+Δt) = press(t) + rate(t) * Δt
    x_pred[TemporalState.PRESS] = (
        x[TemporalState.PRESS] + x[TemporalState.PRESS_RATE] * dt_months
    )

    # Rates are modeled as random walks (no drift term)
    # rate(t+Δt) = rate(t) + process_noise

    # Apply physical bounds
    x_pred[TemporalState.TREAD] = np.clip(
        x_pred[TemporalState.TREAD], 0.5, 9.0
    )
    x_pred[TemporalState.PRESS] = np.clip(
        x_pred[TemporalState.PRESS], 50.0, 500.0
    )
    x_pred[TemporalState.TOE_SQ] = max(0.0, x_pred[TemporalState.TOE_SQ])

    # Clip rates to physical limits
    max_tread_rate = limits.tread_max_rate_mm_per_1000km
    x_pred[TemporalState.TREAD_RATE] = np.clip(
        x_pred[TemporalState.TREAD_RATE], -max_tread_rate, max_tread_rate
    )
    max_press_rate = limits.press_max_rate_kpa_per_month
    x_pred[TemporalState.PRESS_RATE] = np.clip(
        x_pred[TemporalState.PRESS_RATE], -max_press_rate, max_press_rate
    )

    return x_pred


def temporal_process_noise(
    dt_km: float,
    dt_months: float,
    limits: DegradationLimits = DEFAULT_LIMITS,
) -> np.ndarray:
    """Compute process noise covariance Q for the temporal state.

    Process noise increases with elapsed time/distance, reflecting
    growing uncertainty about the state between measurements.

    Parameters
    ----------
    dt_km : distance elapsed (km)
    dt_months : time elapsed (months)
    limits : degradation rate limits

    Returns
    -------
    (18, 18) process noise covariance matrix
    """
    Q = np.zeros((TemporalState.N, TemporalState.N))

    # Base state process noise (small — state changes slowly)
    # Tread wear: ~0.01 mm per 1000 km uncertainty
    for i in range(4):
        Q[i, i] = (0.01 * dt_km / 1000.0) ** 2

    # Pressure drift: ~0.5 kPa per month uncertainty
    for i in range(4):
        Q[4 + i, 4 + i] = (0.5 * dt_months) ** 2

    # Toe and camber: essentially constant (no process noise)
    Q[TemporalState.TOE_SQ, TemporalState.TOE_SQ] = 0.0
    Q[TemporalState.CAMBER, TemporalState.CAMBER] = 0.0

    # Rate state process noise (random walk)
    # Tread rate: ±0.01 mm per 1000 km per 1000 km uncertainty
    for i in range(4):
        Q[10 + i, 10 + i] = (0.01 * dt_km / 1000.0) ** 2

    # Pressure rate: ±0.1 kPa per month per month uncertainty
    for i in range(4):
        Q[14 + i, 14 + i] = (0.1 * dt_months) ** 2

    return Q


# ===========================================================================
# Temporal measurement model
# ===========================================================================

def temporal_measurement_model(
    x: np.ndarray,
    tyre_config,
    physics,
    t_meas_c: float,
    vehicle_config,
) -> np.ndarray:
    """Predict measurements from the extended temporal state.

    The measurement model is the same as Phase 3 — rates are not
    directly measured. The estimator infers rates from how the
    state changes across snapshots.
    """
    from .estimator import _predict

    # Extract base states (indices 0-9)
    x_base = x[:TemporalState.N_BASE]

    # Use Phase 3 predict function
    return _predict(x_base, tyre_config, physics, t_meas_c, vehicle_config)


def temporal_jacobian(
    x: np.ndarray,
    tyre_config,
    physics,
    t_meas_c: float,
    vehicle_config,
    eps: float = 1e-5,
) -> np.ndarray:
    """Numerical Jacobian for the extended temporal state.

    The Jacobian has shape (N_MEAS, 18). The rate states (indices 10-17)
    have zero sensitivity in the current measurement model because rates
    are not directly measured.
    """
    from .estimator import _predict, MEAS

    H = np.zeros((MEAS.N, TemporalState.N))

    # Base states (0-9): same Jacobian as Phase 3
    x_base = x[:TemporalState.N_BASE]
    for j in range(TemporalState.N_BASE):
        dx = np.zeros(TemporalState.N_BASE)
        dx[j] = eps
        H[:, j] = (
            _predict(x_base + dx, tyre_config, physics, t_meas_c, vehicle_config)
            - _predict(x_base - dx, tyre_config, physics, t_meas_c, vehicle_config)
        ) / (2 * eps)

    # Rate states (10-17): zero sensitivity in current measurement model
    # (rates are inferred from temporal evolution, not direct measurement)

    return H


# ===========================================================================
# Multi-snapshot temporal estimator
# ===========================================================================

@dataclass
class TemporalEstimate:
    """Result of temporal degradation-rate estimation."""
    states: tuple[StateEstimate, ...]
    """Per-state estimates including rates."""

    covariance: np.ndarray
    """Posterior covariance matrix."""

    trend_significance: dict[str, float]
    """Statistical significance of detected trends (p-value)."""

    degradation_rates: dict[str, float]
    """Estimated degradation rates with units."""

    n_snapshots_used: int
    """Number of snapshots used for trend estimation."""

    convergence_quality: float
    """How well the temporal model fits the data (R²)."""


def estimate_temporal_trend(
    snapshots: Sequence[TyreStateEstimate],
    odometer_readings: Sequence[float],
    limits: DegradationLimits = DEFAULT_LIMITS,
) -> TemporalEstimate | None:
    """Estimate degradation rates from multiple snapshots.

    This is the core Phase 5 function. It takes a sequence of
    TyreStateEstimate objects and their odometer readings, then
    fits a linear trend to each OBSERVED state.

    Parameters
    ----------
    snapshots : sequence of TyreStateEstimate
        Must be from the same source and configuration.
    odometer_readings : sequence of float
        Odometer reading (km) for each snapshot.
    limits : degradation rate limits

    Returns
    -------
    TemporalEstimate or None if insufficient data
    """
    if len(snapshots) < 3:
        return None  # Need at least 3 snapshots for trend detection

    # Verify all snapshots are compatible
    sources = {s.source for s in snapshots}
    if len(sources) > 1:
        raise ValueError(f"Cannot mix snapshots from different sources: {sources}")

    fingerprints = {s.config_fingerprint for s in snapshots}
    if len(fingerprints) > 1:
        raise ValueError("Cannot mix snapshots with different configurations")

    # Sort by timestamp
    ordered = sorted(zip(snapshots, odometer_readings), key=lambda x: x[1])
    snaps = [s for s, _ in ordered]
    odoms = [o for _, o in ordered]

    # Extract state names from first snapshot
    state_names = [s.name for s in snaps[0].states]
    n_states = len(state_names)

    # For each state, fit linear regression if OBSERVED
    trend_significance = {}
    degradation_rates = {}
    estimated_states = []

    for i, name in enumerate(state_names):
        # Collect observations for this state
        values = []
        sigmas = []
        odoms_obs = []

        for snap, odom in zip(snaps, odoms):
            if i < len(snap.states):
                st = snap.states[i]
                if st.observability == Observability.OBSERVED:
                    values.append(st.value)
                    sigmas.append(st.sigma)
                    odoms_obs.append(odom)

        if len(values) < 3:
            # Insufficient observations for trend detection
            estimated_states.append(StateEstimate(
                name=name,
                value=snaps[-1].states[i].value if i < len(snaps[-1].states) else 0.0,
                sigma=snaps[-1].states[i].sigma if i < len(snaps[-1].states) else 1.0,
                observability=Observability.WEAK,
                prior_value=snaps[0].states[i].prior_value if i < len(snaps[0].states) else 0.0,
                prior_sigma=snaps[0].states[i].prior_sigma if i < len(snaps[0].states) else 1.0,
                variance_reduction=0.0,
                reason=f"Insufficient OBSERVED snapshots ({len(values)}/3) for trend detection",
                magnitude_only="toe" in name.lower(),
            ))
            trend_significance[name] = 1.0  # not significant
            degradation_rates[name] = 0.0
            continue

        # Weighted linear regression (inverse variance weighting)
        values_arr = np.array(values)
        sigmas_arr = np.array(sigmas)
        odoms_arr = np.array(odoms_obs)
        weights = 1.0 / (sigmas_arr ** 2)

        # Normalize odometer for numerical stability
        odom_center = odoms_arr.mean()
        odom_scale = max(odoms_arr.std(), 1.0)
        x_norm = (odoms_arr - odom_center) / odom_scale

        # Weighted least squares: y = a + b*x
        W = np.diag(weights)
        X = np.column_stack([np.ones_like(x_norm), x_norm])
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ values_arr)
        except np.linalg.LinAlgError:
            beta = np.array([values_arr.mean(), 0.0])

        slope_raw = beta[1] / odom_scale  # convert back to per-km

        # Compute R² (goodness of fit)
        y_pred = X @ beta
        ss_res = np.sum(weights * (values_arr - y_pred) ** 2)
        ss_tot = np.sum(weights * (values_arr - np.average(values_arr, weights=weights)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Compute p-value for slope (simplified t-test)
        if len(values) > 2:
            mse = ss_res / (len(values) - 2)
            try:
                cov_beta = mse * np.linalg.inv(X.T @ W @ X)
                slope_se = np.sqrt(max(0.0, cov_beta[1, 1])) / odom_scale
                if slope_se > 0:
                    t_stat = abs(slope_raw) / slope_se
                    # Rough p-value approximation (not exact)
                    p_value = max(0.0, min(1.0, 2.0 * np.exp(-0.7 * t_stat)))
                else:
                    p_value = 1.0
            except np.linalg.LinAlgError:
                p_value = 1.0
        else:
            p_value = 1.0

        trend_significance[name] = p_value
        degradation_rates[name] = slope_raw

        # Classify trend significance
        is_significant = p_value < 0.05 and r_squared > 0.3

        # Get posterior statistics from last snapshot
        last_snap = snaps[-1]
        prior_var = last_snap.states[i].prior_sigma ** 2 if i < len(last_snap.states) else 1.0
        post_var = last_snap.covariance_diag[i] ** 2 if i < len(last_snap.covariance_diag) else 1.0

        if is_significant:
            obs = Observability.OBSERVED
            reason = None
        else:
            obs = Observability.WEAK
            reason = f"Trend not statistically significant (p={p_value:.3f}, R²={r_squared:.3f})"

        estimated_states.append(StateEstimate(
            name=name,
            value=float(last_snap.states[i].value) if i < len(last_snap.states) else 0.0,
            sigma=float(last_snap.states[i].sigma) if i < len(last_snap.states) else 1.0,
            observability=obs,
            prior_value=float(last_snap.states[i].prior_value) if i < len(last_snap.states) else 0.0,
            prior_sigma=float(np.sqrt(max(0.0, prior_var))),
            variance_reduction=float(max(0.0, 1.0 - post_var / prior_var)) if prior_var > 0 else 0.0,
            reason=reason,
            magnitude_only="toe" in name.lower(),
        ))

    # Overall convergence quality
    r_squared_values = [v for v in trend_significance.values() if v < 1.0]
    mean_r_squared = np.mean(r_squared_values) if r_squared_values else 0.0

    return TemporalEstimate(
        states=tuple(estimated_states),
        covariance=snaps[-1].covariance_diag if hasattr(snaps[-1], 'covariance_diag') else np.eye(n_states),
        trend_significance=trend_significance,
        degradation_rates=degradation_rates,
        n_snapshots_used=len(snaps),
        convergence_quality=float(mean_r_squared),
    )
