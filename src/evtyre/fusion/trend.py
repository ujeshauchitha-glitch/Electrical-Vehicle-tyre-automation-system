"""Phase 5 — Trend detection for degradation-rate estimation.

Detects statistically significant trends in tyre state over time/distance.
This is NOT multi-snapshot fusion (Phase 4) — it detects actual physical
degradation trends, not just noise reduction.

Key distinction:
  Fusion: "What is the current state?" (average over noise)
  Trend: "How is the state changing?" (detect physical degradation)

The trend detector requires:
  1. Minimum 5 OBSERVED snapshots for statistical significance
  2. Linear regression with inverse-variance weighting
  3. Significance testing (p-value < 0.05, R² > 0.3)
  4. Outlier detection (residual > 3σ)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from ..estimation.estimator import Observability, StateEstimate
from ..estimation.schema import TyreStateEstimate


# ===========================================================================
# Trend classification
# ===========================================================================

class TrendClassification(Enum):
    """Classification of detected trends."""
    NONE = "none"
    """No significant trend detected."""

    NORMAL = "normal"
    """Trend within expected degradation bounds."""

    ACCELERATED = "accelerated"
    """Trend exceeds normal degradation bounds."""

    FAULT = "fault"
    """Anomalous trend suggesting component failure."""


@dataclass(frozen=True)
class TrendResult:
    """Result of trend detection for a single state."""
    name: str
    """State name (e.g., 'tread_FL', 'press_RL')."""

    slope: float
    """Estimated rate of change (per km or per month)."""

    slope_unit: str
    """Unit of the slope (e.g., 'mm/km', 'kPa/month')."""

    r_squared: float
    """Goodness of fit (0-1). Higher means better fit."""

    p_value: float
    """Statistical significance (0-1). Lower means more significant."""

    classification: TrendClassification
    """Classification of the trend."""

    confidence: float
    """Confidence in the trend estimate (0-1)."""

    n_observations: int
    """Number of OBSERVED snapshots used."""

    residuals_std: float
    """Standard deviation of residuals (noise estimate)."""

    is_outlier: bool
    """True if any observation is an outlier (>3σ residual)."""


@dataclass(frozen=True)
class TrendReport:
    """Complete trend analysis report."""
    trends: dict[str, TrendResult]
    """Per-state trend results."""

    overall_classification: TrendClassification
    """Worst-case classification across all states."""

    mean_confidence: float
    """Average confidence across detected trends."""

    n_states_with_trends: int
    """Number of states with statistically significant trends."""

    n_snapshots: int
    """Number of snapshots analyzed."""

    odometer_range_km: tuple[float, float]
    """Range of odometer readings (start, end)."""


# ===========================================================================
# Trend detection
# ===========================================================================

def detect_trends(
    snapshots: Sequence[TyreStateEstimate],
    odometer_readings: Sequence[float],
    min_observations: int = 5,
) -> TrendReport:
    """Detect statistically significant trends in tyre state.

    Parameters
    ----------
    snapshots : sequence of TyreStateEstimate
        Must be from the same source and configuration.
    odometer_readings : sequence of float
        Odometer reading (km) for each snapshot.
    min_observations : int
        Minimum number of OBSERVED snapshots required for trend detection.

    Returns
    -------
    TrendReport with per-state trend analysis
    """
    if len(snapshots) < min_observations:
        raise ValueError(
            f"Need at least {min_observations} snapshots for trend detection, "
            f"got {len(snapshots)}"
        )

    # Verify compatibility
    sources = {s.source for s in snapshots}
    if len(sources) > 1:
        raise ValueError(f"Cannot mix snapshots from different sources: {sources}")

    # Sort by odometer
    ordered = sorted(zip(snapshots, odometer_readings), key=lambda x: x[1])
    snaps = [s for s, _ in ordered]
    odoms = [o for _, o in ordered]

    odom_min = min(odoms)
    odom_max = max(odoms)

    # Extract state names
    state_names = [s.name for s in snaps[0].states]

    trends = {}
    n_trends = 0
    confidences = []

    for i, name in enumerate(state_names):
        # Collect OBSERVED values
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

        if len(values) < min_observations:
            trends[name] = TrendResult(
                name=name,
                slope=0.0,
                slope_unit="",
                r_squared=0.0,
                p_value=1.0,
                classification=TrendClassification.NONE,
                confidence=0.0,
                n_observations=len(values),
                residuals_std=0.0,
                is_outlier=False,
            )
            continue

        # Weighted linear regression
        values_arr = np.array(values)
        sigmas_arr = np.array(sigmas)
        odoms_arr = np.array(odoms_obs)
        weights = 1.0 / (sigmas_arr ** 2)

        # Normalize for numerical stability
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

        slope_raw = beta[1] / odom_scale  # per-km

        # Goodness of fit
        y_pred = X @ beta
        residuals = values_arr - y_pred
        ss_res = np.sum(weights * residuals ** 2)
        ss_tot = np.sum(weights * (values_arr - np.average(values_arr, weights=weights)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Residuals standard deviation (for reporting)
        residuals_std = float(np.std(residuals))

        # Outlier detection using MAD (median absolute deviation) for robustness
        # Standard deviation is inflated by outliers; MAD is not
        median_residual = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median_residual)))
        # MAD to sigma conversion: for Gaussian, MAD ≈ 0.6745 * sigma
        mad_sigma = mad / 0.6745 if mad > 0 else residuals_std
        is_outlier = any(abs(r - median_residual) > 3 * mad_sigma for r in residuals) if mad_sigma > 0 else False

        # P-value for slope significance
        if len(values) > 2:
            mse = ss_res / (len(values) - 2)
            try:
                cov_beta = mse * np.linalg.inv(X.T @ W @ X)
                slope_se = np.sqrt(max(0.0, cov_beta[1, 1])) / odom_scale
                if slope_se > 0:
                    t_stat = abs(slope_raw) / slope_se
                    # Approximate p-value (two-tailed)
                    p_value = max(0.0, min(1.0, 2.0 * np.exp(-0.7 * t_stat)))
                else:
                    p_value = 1.0
            except np.linalg.LinAlgError:
                p_value = 1.0
        else:
            p_value = 1.0

        # Classification
        is_significant = p_value < 0.05 and r_squared > 0.3

        if not is_significant:
            classification = TrendClassification.NONE
            confidence = 0.0
        else:
            # Check if trend is within normal bounds
            # (UNVALIDATED — would need empirical data for proper thresholds)
            classification = TrendClassification.NORMAL
            confidence = min(1.0, r_squared * (1.0 - p_value))
            n_trends += 1
            confidences.append(confidence)

        # Determine unit based on state name
        if "tread" in name and "rate" not in name:
            slope_unit = "mm/km"
        elif "press" in name and "rate" not in name:
            slope_unit = "kPa/km"
        elif "toe" in name:
            slope_unit = "deg/km"
        else:
            slope_unit = "unit/km"

        trends[name] = TrendResult(
            name=name,
            slope=float(slope_raw),
            slope_unit=slope_unit,
            r_squared=float(r_squared),
            p_value=float(p_value),
            classification=classification,
            confidence=float(confidence),
            n_observations=len(values),
            residuals_std=residuals_std,
            is_outlier=is_outlier,
        )

    # Overall classification (worst case)
    classifications = [t.classification for t in trends.values()]
    if TrendClassification.FAULT in classifications:
        overall = TrendClassification.FAULT
    elif TrendClassification.ACCELERATED in classifications:
        overall = TrendClassification.ACCELERATED
    elif TrendClassification.NORMAL in classifications:
        overall = TrendClassification.NORMAL
    else:
        overall = TrendClassification.NONE

    mean_conf = np.mean(confidences) if confidences else 0.0

    return TrendReport(
        trends=trends,
        overall_classification=overall,
        mean_confidence=float(mean_conf),
        n_states_with_trends=n_trends,
        n_snapshots=len(snaps),
        odometer_range_km=(odom_min, odom_max),
    )


# ===========================================================================
# Utility functions
# ===========================================================================

def format_trend_report(report: TrendReport) -> str:
    """Format a TrendReport as human-readable text."""
    lines = [
        f"Trend Analysis Report",
        f"{'='*50}",
        f"Snapshots analyzed: {report.n_snapshots}",
        f"Odometer range: {report.odometer_range_km[0]:.0f} - {report.odometer_range_km[1]:.0f} km",
        f"States with significant trends: {report.n_states_with_trends}",
        f"Overall classification: {report.overall_classification.value}",
        f"Mean confidence: {report.mean_confidence:.2f}",
        "",
        "Per-state trends:",
        "-" * 50,
    ]

    for name, trend in report.trends.items():
        if trend.classification != TrendClassification.NONE:
            lines.append(
                f"  {name}: {trend.slope:.6f} {trend.slope_unit} "
                f"(R²={trend.r_squared:.3f}, p={trend.p_value:.3f}, "
                f"class={trend.classification.value})"
            )

    return "\n".join(lines)
