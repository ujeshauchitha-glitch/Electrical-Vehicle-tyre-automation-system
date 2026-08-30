"""Phase 5 enhancement — Temperature-dependent degradation model.

Models how ambient temperature affects:
  1. Tyre wear rate (cold = brittle rubber → faster wear; hot = soft → slower)
  2. Pressure drift (ideal gas law: dP/dT ≈ P/T)
  3. Tread compound degradation (UV + heat accelerate aging)

Physical basis:
  - Wear rate: Arrhenius-like temperature dependence
    wear(T) = wear_ref * exp(-Ea/k * (1/T - 1/T_ref))
  - Pressure: Ideal gas law at constant volume
    dP/dT = P / T (per degree)
  - Rubber aging: heat + UV accelerates oxidative degradation

All coefficients are UNVALIDATED engineering estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from ..estimation.estimator import Observability
from ..estimation.schema import TyreStateEstimate


# ===========================================================================
# Temperature regime classification
# ===========================================================================

class ThermalRegime(Enum):
    """Classification of thermal conditions."""
    COLD = "cold"
    """Below 5°C — rubber is stiff, wear accelerates."""

    NORMAL = "normal"
    """5-35°C — nominal wear conditions."""

    WARM = "warm"
    """35-50°C — slightly elevated, wear slows but pressure rises."""

    HOT = "hot"
    """Above 50°C — risk of thermal degradation, pressure spikes."""


# ===========================================================================
# Thermal model parameters (UNVALIDATED)
# ===========================================================================

@dataclass(frozen=True)
class ThermalModelConfig:
    """Configuration for temperature-dependent degradation model."""

    # Reference temperature for wear rate (°C)
    reference_temp_c: float = 25.0
    """Temperature at which nominal wear rates apply."""

    # Arrhenius activation energy for rubber wear (eV)
    # UNVALIDATED — typical range 0.1-0.5 eV for elastomers
    activation_energy_ev: float = 0.3

    # Boltzmann constant (eV/K)
    boltzmann_ev_per_k: float = 8.617e-5

    # Cold stiffening factor: below reference, wear increases
    # wear(T) = wear_ref * exp(Ea/k * (1/T - 1/T_ref))
    # At T = 0°C vs 25°C: wear multiplier ≈ 1.5-2x (UNVALIDATED)

    # Pressure-temperature coefficient (fractional per °C)
    # From ideal gas law: dP/P = dT/T
    # At 25°C (298K): dP/dT ≈ P/298 ≈ 0.0034 * P per °C
    pressure_temp_coeff_per_c: float = 1.0 / 298.15
    """UNVALIDATED — ideal gas law at reference temperature."""

    # Rubber aging acceleration factor
    # For every 10°C above reference, aging accelerates by this factor
    aging_acceleration_per_10c: float = 2.0
    """UNVALIDATED — typical Arrhenius acceleration."""


DEFAULT_THERMAL_CONFIG = ThermalModelConfig()


# ===========================================================================
# Thermal analysis result
# ===========================================================================

@dataclass(frozen=True)
class ThermalState:
    """Thermal analysis for one corner at one time."""
    corner: str
    temperature_c: float
    regime: ThermalRegime

    wear_rate_multiplier: float
    """How much faster/slower wear is compared to reference temperature.
    1.0 = nominal, >1.0 = faster wear, <1.0 = slower wear."""

    pressure_drift_kpa_per_c: float
    """Expected pressure change per degree temperature change (kPa/°C)."""

    aging_factor: float
    """Rubber aging acceleration relative to reference."""

    thermal_stress_index: float
    """Combined thermal stress (0-1 scale, 0=none, 1=severe)."""


@dataclass(frozen=True)
class ThermalDegradationReport:
    """Complete thermal degradation analysis."""
    thermal_states: dict[str, ThermalState]
    """Per-corner thermal analysis."""

    ambient_temp_c: float
    """Ambient temperature."""

    overall_regime: ThermalRegime
    """Worst-case thermal regime."""

    max_wear_multiplier: float
    """Maximum wear rate multiplier across corners."""

    max_pressure_sensitivity: float
    """Maximum pressure sensitivity across corners (kPa/°C)."""

    temperature_compensation_applied: bool
    """True if temperature effects are being accounted for."""

    recommendation: str
    """Human-readable recommendation."""

    @property
    def needs_attention(self) -> bool:
        """True if thermal conditions require attention."""
        return self.overall_regime in (ThermalRegime.COLD, ThermalRegime.HOT)


# ===========================================================================
# Thermal analysis
# ===========================================================================

def compute_thermal_state(
    corner: str,
    temperature_c: float,
    current_pressure_kpa: float,
    config: ThermalModelConfig = DEFAULT_THERMAL_CONFIG,
) -> ThermalState:
    """Compute thermal degradation state for one corner.

    Parameters
    ----------
    corner : str
        Corner identifier (FL, FR, RL, RR).
    temperature_c : float
        Current tyre/ambient temperature (°C).
    current_pressure_kpa : float
        Current tyre pressure (kPa).
    config : ThermalModelConfig
        Model configuration.

    Returns
    -------
    ThermalState for this corner
    """
    # Convert to Kelvin for physics
    temp_k = temperature_c + 273.15
    ref_k = config.reference_temp_c + 273.15

    # Thermal regime
    regime = _classify_regime(temperature_c)

    # Wear rate multiplier (Arrhenius)
    # Higher temperature → lower wear (rubber is softer, more compliant)
    # Lower temperature → higher wear (rubber is stiff, brittle)
    if temp_k > 0 and ref_k > 0:
        wear_multiplier = math.exp(
            config.activation_energy_ev / config.boltzmann_ev_per_k *
            (1.0 / temp_k - 1.0 / ref_k)
        )
    else:
        wear_multiplier = 1.0

    # Pressure sensitivity (ideal gas law)
    # dP/dT = P / T (at constant volume)
    pressure_drift = current_pressure_kpa * config.pressure_temp_coeff_per_c

    # Aging factor
    temp_diff = temperature_c - config.reference_temp_c
    if temp_diff > 0:
        aging = config.aging_acceleration_per_10c ** (temp_diff / 10.0)
    else:
        aging = 1.0 / (config.aging_acceleration_per_10c ** (abs(temp_diff) / 10.0))

    # Thermal stress index (0-1)
    # Cold stress: increases below 0°C
    # Hot stress: increases above 45°C
    cold_stress = max(0.0, min(1.0, (5.0 - temperature_c) / 25.0))
    hot_stress = max(0.0, min(1.0, (temperature_c - 45.0) / 25.0))
    thermal_stress = max(cold_stress, hot_stress)

    return ThermalState(
        corner=corner,
        temperature_c=temperature_c,
        regime=regime,
        wear_rate_multiplier=wear_multiplier,
        pressure_drift_kpa_per_c=pressure_drift,
        aging_factor=aging,
        thermal_stress_index=thermal_stress,
    )


def _classify_regime(temp_c: float) -> ThermalRegime:
    """Classify temperature into thermal regime."""
    if temp_c < 5.0:
        return ThermalRegime.COLD
    elif temp_c < 35.0:
        return ThermalRegime.NORMAL
    elif temp_c < 50.0:
        return ThermalRegime.WARM
    else:
        return ThermalRegime.HOT


def compute_thermal_degradation_report(
    temperatures_c: dict[str, float],
    pressures_kpa: dict[str, float],
    config: ThermalModelConfig = DEFAULT_THERMAL_CONFIG,
) -> ThermalDegradationReport:
    """Compute thermal degradation analysis for all corners.

    Parameters
    ----------
    temperatures_c : dict[str, float]
        Temperature per corner (°C).
    pressures_kpa : dict[str, float]
        Pressure per corner (kPa).
    config : ThermalModelConfig
        Model configuration.

    Returns
    -------
    ThermalDegradationReport
    """
    thermal_states = {}
    regimes = []
    wear_multipliers = []
    pressure_sensitivities = []

    for corner in ["FL", "FR", "RL", "RR"]:
        temp = temperatures_c.get(corner, config.reference_temp_c)
        press = pressures_kpa.get(corner, 240.0)

        state = compute_thermal_state(corner, temp, press, config)
        thermal_states[corner] = state
        regimes.append(state.regime)
        wear_multipliers.append(state.wear_rate_multiplier)
        pressure_sensitivities.append(state.pressure_drift_kpa_per_c)

    # Overall regime (worst case)
    regime_priority = [ThermalRegime.HOT, ThermalRegime.COLD, ThermalRegime.WARM, ThermalRegime.NORMAL]
    overall = ThermalRegime.NORMAL
    for r in regime_priority:
        if r in regimes:
            overall = r
            break

    # Recommendation
    if overall == ThermalRegime.COLD:
        rec = (
            "Cold conditions detected. Tyre rubber is stiff, increasing wear rate. "
            "Allow extra warm-up time before spirited driving. "
            "Pressure readings may be low — cold inflation pressure should be checked."
        )
    elif overall == ThermalRegime.HOT:
        rec = (
            "Hot conditions detected. Risk of thermal degradation and pressure spikes. "
            "Monitor pressure closely. Avoid sustained high-load driving. "
            "Check for signs of thermal stress (blistering, uneven wear)."
        )
    elif overall == ThermalRegime.WARM:
        rec = (
            "Warm conditions. Slightly elevated temperature may increase pressure "
            "readings. Wear rate is nominally reduced. No immediate action required."
        )
    else:
        rec = "Normal thermal conditions. No temperature-related adjustments needed."

    return ThermalDegradationReport(
        thermal_states=thermal_states,
        ambient_temp_c=float(np.mean(list(temperatures_c.values()))) if temperatures_c else 25.0,
        overall_regime=overall,
        max_wear_multiplier=float(max(wear_multipliers)),
        max_pressure_sensitivity=float(max(pressure_sensitivities)),
        temperature_compensation_applied=True,
        recommendation=rec,
    )


def adjust_wear_rate_for_temperature(
    base_wear_rate_mm_per_km: float,
    temperature_c: float,
    config: ThermalModelConfig = DEFAULT_THERMAL_CONFIG,
) -> float:
    """Adjust a wear rate estimate for temperature effects.

    Parameters
    ----------
    base_wear_rate_mm_per_km : float
        Nominal wear rate at reference temperature.
    temperature_c : float
        Current temperature (°C).
    config : ThermalModelConfig
        Model configuration.

    Returns
    -------
    Temperature-adjusted wear rate (mm/km)
    """
    temp_k = temperature_c + 273.15
    ref_k = config.reference_temp_c + 273.15

    if temp_k > 0 and ref_k > 0:
        multiplier = math.exp(
            config.activation_energy_ev / config.boltzmann_ev_per_k *
            (1.0 / temp_k - 1.0 / ref_k)
        )
    else:
        multiplier = 1.0

    return base_wear_rate_mm_per_km * multiplier


def predict_pressure_from_temperature(
    reference_pressure_kpa: float,
    reference_temp_c: float,
    current_temp_c: float,
) -> float:
    """Predict pressure change due to temperature (ideal gas law).

    Parameters
    ----------
    reference_pressure_kpa : float
        Pressure at reference temperature.
    reference_temp_c : float
        Reference temperature (°C).
    current_temp_c : float
        Current temperature (°C).

    Returns
    -------
    Predicted pressure at current temperature (kPa)
    """
    ref_k = reference_temp_c + 273.15
    curr_k = current_temp_c + 273.15

    if ref_k > 0:
        return reference_pressure_kpa * (curr_k / ref_k)
    return reference_pressure_kpa


def format_thermal_report(report: ThermalDegradationReport) -> str:
    """Format a ThermalDegradationReport as human-readable text."""
    lines = [
        "Thermal Degradation Analysis",
        "=" * 55,
        f"Ambient temperature: {report.ambient_temp_c:.1f}°C",
        f"Overall regime: {report.overall_regime.value}",
        f"Max wear multiplier: {report.max_wear_multiplier:.3f}x",
        f"Max pressure sensitivity: {report.max_pressure_sensitivity:.3f} kPa/°C",
        "",
        "Per-corner thermal state:",
        "-" * 55,
    ]

    for corner in ["FL", "FR", "RL", "RR"]:
        if corner in report.thermal_states:
            ts = report.thermal_states[corner]
            lines.append(
                f"  {corner}: {ts.temperature_c:.1f}°C [{ts.regime.value}], "
                f"wear×{ts.wear_rate_multiplier:.3f}, "
                f"ΔP/ΔT={ts.pressure_drift_kpa_per_c:.3f} kPa/°C, "
                f"stress={ts.thermal_stress_index:.2f}"
            )

    lines.append("")
    lines.append(f"Recommendation: {report.recommendation}")

    return "\n".join(lines)
