"""Phase 5 enhancement — Anomaly detection.

Detects sudden changes in tyre state that indicate:
  1. Punctures / rapid pressure loss
  2. Sensor failures (stuck readings, erratic noise)
  3. Sudden tread damage (chunking, delamination)
  4. Rapid temperature excursions

Approach:
  - Compute residuals between consecutive snapshots
  - Apply adaptive thresholds based on measurement uncertainty
  - Distinguish gradual degradation from sudden anomalies
  - Classify anomaly type for maintenance prioritisation

This is fundamentally different from trend detection:
  - Trend: gradual change over hundreds/thousands of km
  - Anomaly: sudden change in a single snapshot
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from ..estimation.estimator import Observability, StateEstimate
from ..estimation.schema import TyreStateEstimate
from ..schema.common import CORNERS


# ===========================================================================
# Anomaly classification
# ===========================================================================

class AnomalyType(Enum):
    """Type of detected anomaly."""
    NONE = "none"
    SUDDEN_DROP = "sudden_drop"
    SUDDEN_RISE = "sudden_rise"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_ERRATIC = "sensor_erratic"
    RAPID_PRESSURE_LOSS = "rapid_pressure_loss"
    TREAD_DAMAGE = "tread_damage"
    TEMPERATURE_SPIKE = "temperature_spike"


class AnomalySeverity(Enum):
    """Severity of detected anomaly."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ===========================================================================
# Thresholds (UNVALIDATED)
# ===========================================================================

# Maximum allowed change per step (per km driven)
MAX_TREAD_CHANGE_PER_STEP_MM = 0.5
MAX_PRESSURE_CHANGE_PER_STEP_KPA = 15.0
MAX_TEMPERATURE_CHANGE_PER_STEP_C = 10.0

# Sensor stuck detection: if variance < threshold over N steps
SENSOR_STUCK_VARIANCE_THRESHOLD = 0.01
SENSOR_STUCK_MIN_STEPS = 3

# Sensor erratic detection: if residual > N * sigma
SENSOR_ERRATIC_SIGMA_MULTIPLIER = 5.0

# Rapid pressure loss: if pressure drops > threshold in one step
RAPID_PRESSURE_LOSS_KPA = 20.0

# Tread damage: if tread drops > threshold in one step
TREAD_DAMAGE_MM = 0.3


# ===========================================================================
# Result data structures
# ===========================================================================

@dataclass(frozen=True)
class AnomalyEvent:
    """A single detected anomaly event."""
    corner: str
    """Which corner is affected."""

    anomaly_type: AnomalyType
    """Type of anomaly."""

    severity: AnomalySeverity
    """Severity."""

    state_name: str
    """Which state variable is affected."""

    previous_value: float
    """Value before the anomaly."""

    current_value: float
    """Value after the anomaly."""

    delta: float
    """Change magnitude."""

    threshold: float
    """Threshold that was exceeded."""

    sigma: float
    """Measurement uncertainty at time of anomaly."""

    description: str
    """Human-readable description."""


@dataclass(frozen=True)
class AnomalyReport:
    """Complete anomaly detection report."""
    events: list[AnomalyEvent]
    """Detected anomaly events."""

    n_anomalies: int
    """Total number of anomalies detected."""

    overall_severity: AnomalySeverity
    """Worst-case severity."""

    has_critical: bool
    """True if any critical anomaly detected."""

    n_snapshots_compared: int
    """Number of snapshot transitions analysed."""

    @property
    def has_anomalies(self) -> bool:
        return self.n_anomalies > 0

    @property
    def affected_corners(self) -> set[str]:
        """Set of corners with anomalies."""
        return {e.corner for e in self.events}


# ===========================================================================
# Anomaly detection
# ===========================================================================

def detect_anomalies(
    snapshots: Sequence[TyreStateEstimate],
    odometer_readings: Sequence[float] | None = None,
) -> AnomalyReport:
    """Detect sudden anomalies in sequential tyre state estimates.

    Compares consecutive snapshots and flags changes that exceed
    adaptive thresholds based on measurement uncertainty.

    Parameters
    ----------
    snapshots : sequence of TyreStateEstimate
        Ordered chronologically (or by odometer).
    odometer_readings : sequence of float, optional
        If provided, thresholds are scaled by distance between snapshots.

    Returns
    -------
    AnomalyReport with detected anomalies
    """
    if len(snapshots) < 2:
        return AnomalyReport(
            events=[],
            n_anomalies=0,
            overall_severity=AnomalySeverity.NONE,
            has_critical=False,
            n_snapshots_compared=0,
        )

    events = []
    state_names = [s.name for s in snapshots[0].states]

    for step_idx in range(1, len(snapshots)):
        prev_snap = snapshots[step_idx - 1]
        curr_snap = snapshots[step_idx]

        # Distance between steps (for threshold scaling)
        if odometer_readings and len(odometer_readings) > step_idx:
            dist_km = odometer_readings[step_idx] - odometer_readings[step_idx - 1]
            dist_km = max(dist_km, 0.1)  # avoid division by zero
        else:
            dist_km = 1.0  # default per-step

        for i, name in enumerate(state_names):
            if i >= len(prev_snap.states) or i >= len(curr_snap.states):
                continue

            prev_st = prev_snap.states[i]
            curr_st = curr_snap.states[i]

            # Only compare if both are OBSERVED
            if (prev_st.observability != Observability.OBSERVED or
                    curr_st.observability != Observability.OBSERVED):
                continue

            prev_val = prev_st.value
            curr_val = curr_st.value
            sigma = curr_st.sigma
            delta = curr_val - prev_val

            # Determine which corner this state belongs to
            corner = _extract_corner(name)
            if corner is None:
                continue  # skip non-corner states (toe, camber)

            # --- Check 1: Sudden magnitude change ---
            anomaly = _check_sudden_change(
                name, corner, prev_val, curr_val, delta, sigma, dist_km
            )
            if anomaly:
                events.append(anomaly)

            # --- Check 2: Sensor stuck ---
            anomaly = _check_sensor_stuck(
                name, corner, snapshots, step_idx, i
            )
            if anomaly:
                events.append(anomaly)

    # Deduplicate (same corner+type from different checks)
    events = _deduplicate_events(events)

    # Overall severity
    severity_order = [
        AnomalySeverity.CRITICAL,
        AnomalySeverity.HIGH,
        AnomalySeverity.MEDIUM,
        AnomalySeverity.LOW,
        AnomalySeverity.NONE,
    ]
    overall = AnomalySeverity.NONE
    for s in severity_order:
        if any(e.severity == s for e in events):
            overall = s
            break

    has_critical = any(e.severity == AnomalySeverity.CRITICAL for e in events)

    return AnomalyReport(
        events=events,
        n_anomalies=len(events),
        overall_severity=overall,
        has_critical=has_critical,
        n_snapshots_compared=len(snapshots) - 1,
    )


def _extract_corner(state_name: str) -> str | None:
    """Extract corner identifier from state name."""
    for corner in CORNERS:
        if state_name.endswith(f"_{corner}"):
            return corner
    return None


def _check_sudden_change(
    name: str,
    corner: str,
    prev_val: float,
    curr_val: float,
    delta: float,
    sigma: float,
    dist_km: float,
) -> AnomalyEvent | None:
    """Check for sudden magnitude change."""
    abs_delta = abs(delta)

    # Scale thresholds by distance
    if "press" in name:
        threshold = MAX_PRESSURE_CHANGE_PER_STEP_KPA * dist_km
        if abs_delta > threshold and abs_delta > 3.0 * sigma:
            severity = _severity_from_ratio(abs_delta / max(threshold, 1.0))
            return AnomalyEvent(
                corner=corner,
                anomaly_type=AnomalyType.RAPID_PRESSURE_LOSS if delta < 0 else AnomalyType.SUDDEN_RISE,
                severity=severity,
                state_name=name,
                previous_value=prev_val,
                current_value=curr_val,
                delta=delta,
                threshold=threshold,
                sigma=sigma,
                description=(
                    f"Pressure {'loss' if delta < 0 else 'gain'} of {abs_delta:.1f} kPa "
                    f"in {corner} over {dist_km:.1f} km (threshold: {threshold:.1f})"
                ),
            )

    elif "tread" in name and "rate" not in name:
        threshold = TREAD_DAMAGE_MM * dist_km
        if abs_delta > threshold and abs_delta > 3.0 * sigma:
            severity = _severity_from_ratio(abs_delta / max(threshold, 1.0))
            return AnomalyEvent(
                corner=corner,
                anomaly_type=AnomalyType.TREAD_DAMAGE if delta < 0 else AnomalyType.SUDDEN_RISE,
                severity=severity,
                state_name=name,
                previous_value=prev_val,
                current_value=curr_val,
                delta=delta,
                threshold=threshold,
                sigma=sigma,
                description=(
                    f"Tread {'loss' if delta < 0 else 'gain'} of {abs_delta:.2f} mm "
                    f"in {corner} over {dist_km:.1f} km (threshold: {threshold:.2f})"
                ),
            )

    elif "temp" in name:
        threshold = MAX_TEMPERATURE_CHANGE_PER_STEP_C * dist_km
        if abs_delta > threshold and abs_delta > 3.0 * sigma:
            severity = _severity_from_ratio(abs_delta / max(threshold, 1.0))
            return AnomalyEvent(
                corner=corner,
                anomaly_type=AnomalyType.TEMPERATURE_SPIKE,
                severity=severity,
                state_name=name,
                previous_value=prev_val,
                current_value=curr_val,
                delta=delta,
                threshold=threshold,
                sigma=sigma,
                description=(
                    f"Temperature {'spike' if delta > 0 else 'drop'} of {abs_delta:.1f}°C "
                    f"in {corner} over {dist_km:.1f} km"
                ),
            )

    return None


def _check_sensor_stuck(
    name: str,
    corner: str,
    snapshots: Sequence[TyreStateEstimate],
    current_idx: int,
    state_idx: int,
) -> AnomalyEvent | None:
    """Check if sensor appears stuck (no change over multiple steps).

    Uses absolute variance: if the observed readings have virtually no
    variation over several consecutive snapshots, the sensor may be
    frozen or the data pipeline may be replaying stale values.
    """
    lookback = min(SENSOR_STUCK_MIN_STEPS, current_idx + 1)
    if lookback < SENSOR_STUCK_MIN_STEPS:
        return None

    recent_values = []
    for j in range(current_idx - lookback + 1, current_idx + 1):
        snap = snapshots[j]
        if state_idx < len(snap.states):
            st = snap.states[state_idx]
            if st.observability == Observability.OBSERVED:
                recent_values.append(st.value)

    if len(recent_values) < SENSOR_STUCK_MIN_STEPS:
        return None

    variance = float(np.var(recent_values))
    if variance < SENSOR_STUCK_VARIANCE_THRESHOLD:
        mean_val = float(np.mean(recent_values))
        if abs(mean_val) > 0.01:
            return AnomalyEvent(
                corner=corner,
                anomaly_type=AnomalyType.SENSOR_STUCK,
                severity=AnomalySeverity.MEDIUM,
                state_name=name,
                previous_value=recent_values[0],
                current_value=recent_values[-1],
                delta=0.0,
                threshold=SENSOR_STUCK_VARIANCE_THRESHOLD,
                sigma=math.sqrt(variance),
                description=(
                    f"Sensor for {name} in {corner} appears stuck "
                    f"(variance={variance:.4f} over {len(recent_values)} steps)"
                ),
            )

    return None


def _severity_from_ratio(ratio: float) -> AnomalySeverity:
    """Map threshold exceedance ratio to severity."""
    if ratio < 2.0:
        return AnomalySeverity.LOW
    elif ratio < 4.0:
        return AnomalySeverity.MEDIUM
    elif ratio < 8.0:
        return AnomalySeverity.HIGH
    else:
        return AnomalySeverity.CRITICAL


def _deduplicate_events(events: list[AnomalyEvent]) -> list[AnomalyEvent]:
    """Remove duplicate events (same corner + type)."""
    seen = set()
    deduped = []
    for e in events:
        key = (e.corner, e.anomaly_type)
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped


def format_anomaly_report(report: AnomalyReport) -> str:
    """Format an AnomalyReport as human-readable text."""
    lines = [
        "Anomaly Detection Report",
        "=" * 55,
        f"Snapshots compared: {report.n_snapshots_compared}",
        f"Anomalies detected: {report.n_anomalies}",
        f"Overall severity: {report.overall_severity.value}",
        "",
    ]

    if not report.has_anomalies:
        lines.append("No anomalies detected.")
    else:
        for event in report.events:
            lines.append(f"  [{event.severity.value.upper()}] {event.description}")

    return "\n".join(lines)
