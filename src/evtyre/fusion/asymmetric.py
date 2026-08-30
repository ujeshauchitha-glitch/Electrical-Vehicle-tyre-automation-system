"""Phase 5 enhancement — Asymmetric wear detection.

Cross-correlates all four tyres to detect:
  1. Left-right asymmetry (suspension/alignment issues)
  2. Front-rear asymmetry (drivetrain/loading differences)
  3. Single-corner anomalies (localised damage/defect)
  4. Wear pattern classification (inner edge, outer edge, etc.)

Physical basis:
  - Uniform wear: all corners within tolerance → normal
  - L-R asymmetry: different camber/toe on one side → alignment
  - F-R asymmetry: different loading (RWD vs FWD) → normal or load issue
  - Single corner: puncture, defect, or localised issue

All thresholds are UNVALIDATED engineering estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from ..estimation.estimator import Observability
from ..estimation.schema import TyreStateEstimate
from ..schema.common import CORNERS


# ===========================================================================
# Asymmetry classification
# ===========================================================================

class AsymmetryType(Enum):
    """Type of detected asymmetry."""
    SYMMETRIC = "symmetric"
    LEFT_RIGHT = "left_right"
    FRONT_REAR = "front_rear"
    SINGLE_CORNER = "single_corner"
    COMPLEX = "complex"


class Severity(Enum):
    """Severity of detected asymmetry."""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.NORMAL: 0,
    Severity.MILD: 1,
    Severity.MODERATE: 2,
    Severity.SEVERE: 3,
}


def _max_severity(a: Severity, b: Severity) -> Severity:
    """Return the more severe of two Severity values."""
    return a if _SEVERITY_RANK[a] >= _SEVERITY_RANK[b] else b


# ===========================================================================
# Thresholds (UNVALIDATED)
# ===========================================================================

# Tread asymmetry thresholds (mm difference between corners)
TREAD_LR_THRESHOLD_MILD = 0.3
TREAD_LR_THRESHOLD_MODERATE = 0.6
TREAD_LR_THRESHOLD_SEVERE = 1.0

TREAD_FR_THRESHOLD_MILD = 0.3
TREAD_FR_THRESHOLD_MODERATE = 0.6
TREAD_FR_THRESHOLD_SEVERE = 1.0

# Single corner deviation from fleet mean
SINGLE_CORNER_MILD = 0.5
SINGLE_CORNER_MODERATE = 0.8
SINGLE_CORNER_SEVERE = 1.2

# Pressure asymmetry thresholds (kPa)
PRESSURE_LR_THRESHOLD_MILD = 10.0
PRESSURE_LR_THRESHOLD_MODERATE = 20.0
PRESSURE_LR_THRESHOLD_SEVERE = 35.0


# ===========================================================================
# Result data structures
# ===========================================================================

@dataclass(frozen=True)
class CornerAnalysis:
    """Analysis result for one corner."""
    corner: str
    tread_deviation_mm: float
    """Deviation from fleet mean tread depth."""

    pressure_deviation_kpa: float
    """Deviation from fleet mean pressure."""

    is_outlier: bool
    """True if this corner deviates significantly from others."""

    wear_rate_ratio: float
    """This corner's wear rate / fleet mean wear rate."""

    severity: Severity
    """Severity of this corner's deviation."""


@dataclass(frozen=True)
class AsymmetryReport:
    """Complete asymmetric wear analysis."""
    asymmetry_type: AsymmetryType
    """Type of detected asymmetry."""

    severity: Severity
    """Overall severity."""

    left_right_tread_delta_mm: float
    """Mean left tread - mean right tread (positive = left wears more)."""

    front_rear_tread_delta_mm: float
    """Mean front tread - mean rear tread (positive = front wears more)."""

    left_right_pressure_delta_kpa: float
    """Mean left pressure - mean right pressure."""

    front_rear_pressure_delta_kpa: float
    """Mean front pressure - mean rear pressure."""

    corner_analyses: dict[str, CornerAnalysis]
    """Per-corner analysis."""

    most_worn_corner: str
    """Corner with lowest tread depth."""

    least_worn_corner: str
    """Corner with highest tread depth."""

    tread_range_mm: float
    """Max - min tread depth across all corners."""

    fleet_mean_tread_mm: float
    """Mean tread depth across all corners."""

    fleet_mean_pressure_kpa: float
    """Mean pressure across all corners."""

    n_observations: int
    """Number of valid observations used."""

    recommendation: str
    """Human-readable recommendation."""

    @property
    def needs_alignment_check(self) -> bool:
        """True if asymmetry suggests alignment issue."""
        return self.asymmetry_type in (
            AsymmetryType.LEFT_RIGHT,
            AsymmetryType.COMPLEX,
        ) and self.severity in (Severity.MODERATE, Severity.SEVERE)

    @property
    def needs_tyre_rotation(self) -> bool:
        """True if tyre rotation recommended."""
        return self.tread_range_mm > 0.8 and self.severity != Severity.NORMAL


# ===========================================================================
# Asymmetry detection
# ===========================================================================

def detect_asymmetry(
    tread_estimates: dict[str, float],
    tread_sigmas: dict[str, float],
    pressure_estimates: dict[str, float],
    wear_rates: dict[str, float] | None = None,
) -> AsymmetryReport:
    """Detect asymmetric wear patterns across all four tyres.

    Parameters
    ----------
    tread_estimates : dict[str, float]
        Current tread depth estimate per corner (mm).
    tread_sigmas : dict[str, float]
        Uncertainty per corner (mm).
    pressure_estimates : dict[str, float]
        Current pressure estimate per corner (kPa).
    wear_rates : dict[str, float], optional
        Wear rate per corner (mm/km). If None, rate analysis is skipped.

    Returns
    -------
    AsymmetryReport with asymmetry classification and severity
    """
    # Fleet statistics
    treads = [tread_estimates.get(c, 0.0) for c in CORNERS]
    pressures = [pressure_estimates.get(c, 0.0) for c in CORNERS]

    mean_tread = float(np.mean(treads))
    mean_pressure = float(np.mean(pressures))

    most_worn = min(CORNERS, key=lambda c: tread_estimates.get(c, 0.0))
    least_worn = max(CORNERS, key=lambda c: tread_estimates.get(c, 0.0))
    tread_range = tread_estimates.get(least_worn, 0.0) - tread_estimates.get(most_worn, 0.0)

    # Left-Right deltas
    left_treads = [(tread_estimates.get("FL", 0.0) + tread_estimates.get("RL", 0.0)) / 2.0]
    right_treads = [(tread_estimates.get("FR", 0.0) + tread_estimates.get("RR", 0.0)) / 2.0]
    lr_tread_delta = float(np.mean(left_treads) - np.mean(right_treads))

    left_pressures = [(pressure_estimates.get("FL", 0.0) + pressure_estimates.get("RL", 0.0)) / 2.0]
    right_pressures = [(pressure_estimates.get("FR", 0.0) + pressure_estimates.get("RR", 0.0)) / 2.0]
    lr_pressure_delta = float(np.mean(left_pressures) - np.mean(right_pressures))

    # Front-Rear deltas
    front_treads = [(tread_estimates.get("FL", 0.0) + tread_estimates.get("FR", 0.0)) / 2.0]
    rear_treads = [(tread_estimates.get("RL", 0.0) + tread_estimates.get("RR", 0.0)) / 2.0]
    fr_tread_delta = float(np.mean(front_treads) - np.mean(rear_treads))

    front_pressures = [(pressure_estimates.get("FL", 0.0) + pressure_estimates.get("FR", 0.0)) / 2.0]
    rear_pressures = [(pressure_estimates.get("RL", 0.0) + pressure_estimates.get("RR", 0.0)) / 2.0]
    fr_pressure_delta = float(np.mean(front_pressures) - np.mean(rear_pressures))

    # Per-corner analysis
    corner_analyses = {}
    for corner in CORNERS:
        tread_dev = tread_estimates.get(corner, mean_tread) - mean_tread
        press_dev = pressure_estimates.get(corner, mean_pressure) - mean_pressure

        # Is this corner an outlier?
        sigma = tread_sigmas.get(corner, 1.0)
        is_outlier = abs(tread_dev) > 2.0 * sigma

        # Wear rate ratio
        if wear_rates and corner in wear_rates:
            mean_rate = float(np.mean(list(wear_rates.values())))
            rate_ratio = wear_rates[corner] / mean_rate if mean_rate != 0 else 1.0
        else:
            rate_ratio = 1.0

        # Severity based on deviation
        abs_dev = abs(tread_dev)
        if abs_dev < SINGLE_CORNER_MILD:
            sev = Severity.NORMAL
        elif abs_dev < SINGLE_CORNER_MODERATE:
            sev = Severity.MILD
        elif abs_dev < SINGLE_CORNER_SEVERE:
            sev = Severity.MODERATE
        else:
            sev = Severity.SEVERE

        corner_analyses[corner] = CornerAnalysis(
            corner=corner,
            tread_deviation_mm=tread_dev,
            pressure_deviation_kpa=press_dev,
            is_outlier=is_outlier,
            wear_rate_ratio=rate_ratio,
            severity=sev,
        )

    # Classify asymmetry type
    abs_lr = abs(lr_tread_delta)
    abs_fr = abs(fr_tread_delta)

    lr_severity = _classify_delta(abs_lr, TREAD_LR_THRESHOLD_MILD, TREAD_LR_THRESHOLD_MODERATE, TREAD_LR_THRESHOLD_SEVERE)
    fr_severity = _classify_delta(abs_fr, TREAD_FR_THRESHOLD_MILD, TREAD_FR_THRESHOLD_MODERATE, TREAD_FR_THRESHOLD_SEVERE)

    # Check for single-corner outlier
    n_outliers = sum(1 for ca in corner_analyses.values() if ca.is_outlier)

    # Overall classification
    if n_outliers == 1:
        asym_type = AsymmetryType.SINGLE_CORNER
        severity = _max_severity(lr_severity, fr_severity)
    elif lr_severity != Severity.NORMAL and fr_severity != Severity.NORMAL:
        asym_type = AsymmetryType.COMPLEX
        severity = _max_severity(lr_severity, fr_severity)
    elif lr_severity != Severity.NORMAL:
        asym_type = AsymmetryType.LEFT_RIGHT
        severity = lr_severity
    elif fr_severity != Severity.NORMAL:
        asym_type = AsymmetryType.FRONT_REAR
        severity = fr_severity
    else:
        asym_type = AsymmetryType.SYMMETRIC
        severity = Severity.NORMAL

    # Count valid observations
    n_obs = sum(
        1 for c in CORNERS
        if c in tread_estimates and c in pressure_estimates
    )

    # Recommendation
    recommendation = _generate_recommendation(asym_type, severity, most_worn, corner_analyses)

    return AsymmetryReport(
        asymmetry_type=asym_type,
        severity=severity,
        left_right_tread_delta_mm=lr_tread_delta,
        front_rear_tread_delta_mm=fr_tread_delta,
        left_right_pressure_delta_kpa=lr_pressure_delta,
        front_rear_pressure_delta_kpa=fr_pressure_delta,
        corner_analyses=corner_analyses,
        most_worn_corner=most_worn,
        least_worn_corner=least_worn,
        tread_range_mm=tread_range,
        fleet_mean_tread_mm=mean_tread,
        fleet_mean_pressure_kpa=mean_pressure,
        n_observations=n_obs,
        recommendation=recommendation,
    )


def _classify_delta(
    abs_delta: float, mild: float, moderate: float, severe: float,
) -> Severity:
    """Classify a delta magnitude into severity."""
    if abs_delta < mild:
        return Severity.NORMAL
    elif abs_delta < moderate:
        return Severity.MILD
    elif abs_delta < severe:
        return Severity.MODERATE
    else:
        return Severity.SEVERE


def _generate_recommendation(
    asym_type: AsymmetryType,
    severity: Severity,
    most_worn: str,
    corner_analyses: dict[str, CornerAnalysis],
) -> str:
    """Generate a human-readable recommendation."""
    if asym_type == AsymmetryType.SYMMETRIC:
        return "Wear is symmetric across all corners. No action required."

    if severity == Severity.NORMAL:
        return "Minor variations within normal tolerance."

    parts = []

    if asym_type == AsymmetryType.LEFT_RIGHT:
        parts.append("Left-right tread asymmetry detected.")
        if severity in (Severity.MODERATE, Severity.SEVERE):
            parts.append("Recommend wheel alignment check and suspension inspection.")

    elif asym_type == AsymmetryType.FRONT_REAR:
        parts.append("Front-rear tread asymmetry detected.")
        if severity in (Severity.MODERATE, Severity.SEVERE):
            parts.append("Consider tyre rotation to equalise wear.")

    elif asym_type == AsymmetryType.SINGLE_CORNER:
        outlier = [ca for ca in corner_analyses.values() if ca.is_outlier]
        if outlier:
            c = outlier[0].corner
            parts.append(f"Corner {c} is an outlier.")
            if outlier[0].severity == Severity.SEVERE:
                parts.append(f"Inspect {c} tyre for damage, defect, or unusual loading.")
            else:
                parts.append(f"Monitor {c} closely; may indicate localised issue.")

    elif asym_type == AsymmetryType.COMPLEX:
        parts.append("Complex asymmetry pattern detected (both L-R and F-R).")
        parts.append("Recommend full vehicle inspection including alignment, suspension, and loading.")

    if most_worn in [ca.corner for ca in corner_analyses.values() if ca.severity == Severity.SEVERE]:
        parts.append(f"Corner {most_worn} has critical wear — prioritise replacement.")

    return " ".join(parts)


def format_asymmetry_report(report: AsymmetryReport) -> str:
    """Format an AsymmetryReport as human-readable text."""
    lines = [
        "Asymmetric Wear Analysis",
        "=" * 55,
        f"Asymmetry type: {report.asymmetry_type.value}",
        f"Severity: {report.severity.value}",
        f"Fleet mean tread: {report.fleet_mean_tread_mm:.2f} mm",
        f"Tread range: {report.tread_range_mm:.2f} mm",
        f"Most worn: {report.most_worn_corner}",
        f"Least worn: {report.least_worn_corner}",
        "",
        f"L-R tread delta: {report.left_right_tread_delta_mm:+.2f} mm",
        f"F-R tread delta: {report.front_rear_tread_delta_mm:+.2f} mm",
        f"L-R pressure delta: {report.left_right_pressure_delta_kpa:+.1f} kPa",
        f"F-R pressure delta: {report.front_rear_pressure_delta_kpa:+.1f} kPa",
        "",
        "Per-corner:",
        "-" * 55,
    ]

    for corner in CORNERS:
        if corner in report.corner_analyses:
            ca = report.corner_analyses[corner]
            outlier = " [OUTLIER]" if ca.is_outlier else ""
            lines.append(
                f"  {corner}: tread={ca.tread_deviation_mm:+.2f} mm, "
                f"pressure={ca.pressure_deviation_kpa:+.1f} kPa, "
                f"rate_ratio={ca.wear_rate_ratio:.2f}, "
                f"{ca.severity.value}{outlier}"
            )

    lines.append("")
    lines.append(f"Recommendation: {report.recommendation}")

    return "\n".join(lines)
