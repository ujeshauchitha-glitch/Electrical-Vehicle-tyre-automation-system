"""Phase 5 — Degradation forecasting.

Predicts when maintenance is needed based on detected trends.
Provides forecasts with confidence intervals and urgency classification.

The forecast assumes constant degradation rates (linear extrapolation).
This is appropriate for short-term forecasting but may be inaccurate
for long-term predictions where wear rates may accelerate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from ..estimation.estimator import Observability, StateEstimate
from ..estimation.schema import TyreStateEstimate
from .trend import TrendReport, TrendClassification


# ===========================================================================
# Urgency classification
# ===========================================================================

class Urgency(Enum):
    """Urgency of maintenance action."""
    LOW = "low"
    """No action needed; current state is acceptable."""

    MEDIUM = "medium"
    """Action recommended within next service interval."""

    HIGH = "high"
    """Action should be taken soon."""

    CRITICAL = "critical"
    """Immediate action required."""


# ===========================================================================
# Forecast data structures
# ===========================================================================

@dataclass(frozen=True)
class MaintenanceForecast:
    """Forecast for a single maintenance parameter."""
    name: str
    """Parameter name (e.g., 'tread_FL', 'pressure_RL')."""

    current_value: float
    """Current estimated value."""

    threshold: float
    """Maintenance threshold (e.g., legal tread limit)."""

    threshold_unit: str
    """Unit of the threshold."""

    remaining_km: float
    """Estimated km until threshold is reached."""

    remaining_ci_km: tuple[float, float]
    """95% confidence interval on remaining km."""

    urgency: Urgency
    """Urgency classification."""

    confidence: float
    """Confidence in the forecast (0-1)."""

    based_on_trend: bool
    """True if forecast is based on detected trend; False if extrapolated from prior."""


@dataclass(frozen=True)
class DegradationForecast:
    """Complete degradation forecast."""
    forecasts: dict[str, MaintenanceForecast]
    """Per-parameter forecasts."""

    overall_urgency: Urgency
    """Worst-case urgency across all parameters."""

    mean_confidence: float
    """Average confidence across forecasts."""

    n_parameters_forecast: int
    """Number of parameters with valid forecasts."""

    current_odometer_km: float
    """Current odometer reading."""


# ===========================================================================
# Thresholds (UNVALIDATED — engineering estimates)
# ===========================================================================

# Tread: legal limit is 1.6 mm (varies by jurisdiction)
TREAD_LEGAL_LIMIT_MM = 1.6
"""Legal tread depth limit (mm). UNVALIDATED — check local regulations."""

# Pressure: minimum recommended is typically 80% of placard
PRESSURE_MIN_FRACTION = 0.80
"""Minimum recommended pressure as fraction of placard. UNVALIDATED."""

# Toe: typical alignment tolerance
TOE_MAX_TOLERANCE_DEG = 0.5
"""Maximum acceptable toe angle (deg). UNVALIDATED."""


# ===========================================================================
# Forecast computation
# ===========================================================================

def compute_forecasts(
    current_estimate: TyreStateEstimate,
    trend_report: TrendReport,
    current_odometer_km: float,
    placard_pressure_kpa: float = 240.0,
    tread_legal_mm: float = TREAD_LEGAL_LIMIT_MM,
) -> DegradationForecast:
    """Compute maintenance forecasts based on current state and trends.

    Parameters
    ----------
    current_estimate : TyreStateEstimate
        Current tyre state estimate.
    trend_report : TrendReport
        Detected trends from historical snapshots.
    current_odometer_km : float
        Current odometer reading (km).
    placard_pressure_kpa : float
        Placard pressure (kPa).
    tread_legal_mm : float
        Legal tread depth limit (mm).

    Returns
    -------
    DegradationForecast with per-parameter forecasts
    """
    forecasts = {}
    urgencies = []
    confidences = []

    for state in current_estimate.states:
        name = state.name

        # Determine threshold and unit based on state type
        if name.startswith("tread_"):
            threshold = tread_legal_mm
            threshold_unit = "mm"
            # Convert trend slope from per-km to per-mm
            trend_slope = trend_report.trends.get(name)
            if trend_slope and trend_slope.classification != TrendClassification.NONE:
                slope_per_km = trend_slope.slope  # mm/km (negative = wear)
            else:
                slope_per_km = 0.0

        elif name.startswith("press_"):
            corner = name.replace("press_", "")
            threshold = placard_pressure_kpa * PRESSURE_MIN_FRACTION
            threshold_unit = "kPa"
            trend_slope = trend_report.trends.get(name)
            if trend_slope and trend_slope.classification != TrendClassification.NONE:
                slope_per_km = trend_slope.slope  # kPa/km (negative = loss)
            else:
                slope_per_km = 0.0

        elif name == "toe^2":
            threshold = TOE_MAX_TOLERANCE_DEG ** 2
            threshold_unit = "deg²"
            trend_slope = trend_report.trends.get(name)
            if trend_slope and trend_slope.classification != TrendClassification.NONE:
                slope_per_km = trend_slope.slope
            else:
                slope_per_km = 0.0

        else:
            # Cannot forecast (e.g., camber, rates)
            continue

        # Current value
        current_value = state.value

        # Forecast remaining life
        if slope_per_km != 0 and state.observability == Observability.OBSERVED:
            # Linear extrapolation: remaining = (threshold - current) / slope
            remaining = (threshold - current_value) / slope_per_km
            remaining = max(0.0, remaining)  # negative means already past threshold

            # Confidence interval (simplified)
            slope_std = abs(slope_per_km) * 0.3  # UNVALIDATED uncertainty estimate
            if slope_std > 0:
                ci_lower = (threshold - current_value) / (slope_per_km + 1.96 * slope_std)
                ci_upper = (threshold - current_value) / (slope_per_km - 1.96 * slope_std)
                ci_lower = max(0.0, ci_lower)
                ci_upper = max(0.0, ci_upper)
                if ci_lower > ci_upper:
                    ci_lower, ci_upper = ci_upper, ci_lower
            else:
                ci_lower = remaining * 0.5
                ci_upper = remaining * 1.5

            confidence = trend_report.trends[name].confidence
            based_on_trend = True
        else:
            # No trend detected — cannot forecast meaningfully
            remaining = float('inf')
            ci_lower = 0.0
            ci_upper = float('inf')
            confidence = 0.0
            based_on_trend = False

        # Classify urgency
        if remaining == float('inf'):
            urgency = Urgency.LOW
        elif remaining < 1000:  # < 1000 km
            urgency = Urgency.CRITICAL
        elif remaining < 5000:  # < 5000 km
            urgency = Urgency.HIGH
        elif remaining < 20000:  # < 20000 km
            urgency = Urgency.MEDIUM
        else:
            urgency = Urgency.LOW

        forecasts[name] = MaintenanceForecast(
            name=name,
            current_value=current_value,
            threshold=threshold,
            threshold_unit=threshold_unit,
            remaining_km=remaining,
            remaining_ci_km=(ci_lower, ci_upper),
            urgency=urgency,
            confidence=confidence,
            based_on_trend=based_on_trend,
        )

        urgencies.append(urgency)
        if based_on_trend:
            confidences.append(confidence)

    # Overall urgency (worst case)
    urgency_order = [Urgency.CRITICAL, Urgency.HIGH, Urgency.MEDIUM, Urgency.LOW]
    overall_urgency = Urgency.LOW
    for u in urgency_order:
        if u in urgencies:
            overall_urgency = u
            break

    mean_conf = np.mean(confidences) if confidences else 0.0

    return DegradationForecast(
        forecasts=forecasts,
        overall_urgency=overall_urgency,
        mean_confidence=float(mean_conf),
        n_parameters_forecast=len(forecasts),
        current_odometer_km=current_odometer_km,
    )


# ===========================================================================
# Utility functions
# ===========================================================================

def format_forecast(forecast: DegradationForecast) -> str:
    """Format a DegradationForecast as human-readable text."""
    lines = [
        f"Degradation Forecast",
        f"{'='*50}",
        f"Current odometer: {forecast.current_odometer_km:.0f} km",
        f"Parameters forecasted: {forecast.n_parameters_forecast}",
        f"Overall urgency: {forecast.overall_urgency.value}",
        f"Mean confidence: {forecast.mean_confidence:.2f}",
        "",
        "Per-parameter forecasts:",
        "-" * 50,
    ]

    for name, f in forecast.forecasts.items():
        if f.remaining_km == float('inf'):
            remaining_str = "N/A (no trend detected)"
        else:
            remaining_str = f"{f.remaining_km:.0f} km (95% CI: {f.remaining_ci_km[0]:.0f}-{f.remaining_ci_km[1]:.0f})"

        lines.append(
            f"  {name}: {f.current_value:.2f} {f.threshold_unit} "
            f"→ {remaining_str} "
            f"[{f.urgency.value}]"
        )

    return "\n".join(lines)
