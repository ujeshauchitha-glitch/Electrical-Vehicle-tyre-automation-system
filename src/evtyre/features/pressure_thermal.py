"""Per-corner pressure and temperature feature extraction.

Key project rules:
- running_pressure_pa (absolute) and cold_equivalent_pressure_pa are DISTINCT.
  Temperature compensation is a REPORTING step, never input-conditioning.
- The cold-equivalent normalisation uses Gay-Lussac's law (P/T = const at
  constant volume): P_cold = P_running * T_ref / T_running, where T_ref is
  a reference temperature (20 °C = 293.15 K).  Classification: B.
- Missing TPMS on a corner → that corner's features are UNAVAILABLE with a
  reason.  The other three corners still produce features normally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..schema.common import CORNERS, SensorReading, SensorStatus
from ..schema.telemetry import TelemetryFrame
from .contract import Classification, Directionality, Feature, FeatureStatus

if TYPE_CHECKING:
    pass

EXTRACTOR_VERSION = "0.1.0"

# Atmospheric pressure in Pa (standard sea-level).
ATMOSPHERIC_PRESSURE_PA: float = 101_325.0

# Reference temperature for cold-equivalent normalisation (20 °C in Kelvin).
_T_REF_K: float = 293.15
_C_TO_K: float = 273.15


def _celsius_to_kelvin(t_c: float) -> float:
    return t_c + _C_TO_K


def extract(
    frame: TelemetryFrame,
    vehicle_config: VehicleConfig,
    tyre_config: TyreConfig,
) -> tuple[Feature, ...]:
    """Extract per-corner pressure and temperature features.

    Returns a tuple of Feature objects.  When TPMS data is missing for a
    corner, that corner's features are UNAVAILABLE with a reason — other
    corners are unaffected.
    """
    features: list[Feature] = []
    ts = frame.timestamp_s
    prov = frame.source

    # Pre-extract usable pressures and temperatures per corner.
    pressures_kpa: dict[str, float | None] = {}
    temps_c: dict[str, float | None] = {}
    for corner in CORNERS:
        pr = frame.tpms_pressure_kpa[corner]
        pressures_kpa[corner] = pr.value if pr.is_usable else None
        tr = frame.tpms_temperature_c[corner]
        temps_c[corner] = tr.value if tr.is_usable else None

    # Ambient temperature (for temperature-rise calculation).
    ambient_c: float | None = (
        frame.ambient_temp_c.value if frame.ambient_temp_c.is_usable else None
    )

    for corner in CORNERS:
        p_kpa = pressures_kpa[corner]
        t_c = temps_c[corner]
        p_inputs = ("tpms_pressure_kpa",)
        t_inputs = ("tpms_temperature_c",)

        # --- Running pressure (absolute, Pa) ---
        # Classification A: purely telemetry-derived.
        if p_kpa is None:
            features.append(Feature(
                name=f"running_pressure_pa_{corner}",
                value=None,
                unit="Pa",
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason=f"TPMS pressure missing for {corner}",
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=p_inputs,
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))
        else:
            # Gauge → absolute by adding atmospheric pressure.
            p_abs_pa = (p_kpa * 1000.0) + ATMOSPHERIC_PRESSURE_PA
            features.append(Feature(
                name=f"running_pressure_pa_{corner}",
                value=p_abs_pa,
                unit="Pa",
                status=FeatureStatus.OK,
                unavailable_reason=None,
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=p_inputs,
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))

        # --- Cold-equivalent pressure (Pa) ---
        # Gay-Lussac's law at constant volume: P_cold = P * T_ref / T.
        # Classification B: literature-supported physical inference.
        if p_kpa is None or t_c is None:
            reason_parts = []
            if p_kpa is None:
                reason_parts.append(f"TPMS pressure missing for {corner}")
            if t_c is None:
                reason_parts.append(f"TPMS temperature missing for {corner}")
            features.append(Feature(
                name=f"cold_equivalent_pressure_pa_{corner}",
                value=None,
                unit="Pa",
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason="; ".join(reason_parts),
                directionality=Directionality.NATURAL,
                classification=Classification.B,
                inputs=("tpms_pressure_kpa", "tpms_temperature_c"),
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))
        else:
            t_k = _celsius_to_kelvin(t_c)
            if t_k <= 0.0:
                features.append(Feature(
                    name=f"cold_equivalent_pressure_pa_{corner}",
                    value=None,
                    unit="Pa",
                    status=FeatureStatus.UNAVAILABLE,
                    unavailable_reason=(
                        f"Temperature {t_c} °C converts to {t_k} K; "
                        f"Gay-Lussac normalisation undefined for T ≤ 0 K"
                    ),
                    directionality=Directionality.NATURAL,
                    classification=Classification.B,
                    inputs=("tpms_pressure_kpa", "tpms_temperature_c"),
                    corner=corner,
                    timestamp_s=ts,
                    provenance=prov,
                    extractor_version=EXTRACTOR_VERSION,
                ))
            else:
                # Gay-Lussac: P/T = const → P_cold = P_running * T_ref / T_running
                p_cold_pa = ((p_kpa * 1000.0) + ATMOSPHERIC_PRESSURE_PA) * _T_REF_K / t_k
                features.append(Feature(
                    name=f"cold_equivalent_pressure_pa_{corner}",
                    value=p_cold_pa,
                    unit="Pa",
                    status=FeatureStatus.OK,
                    unavailable_reason=None,
                    directionality=Directionality.NATURAL,
                    classification=Classification.B,
                    inputs=("tpms_pressure_kpa", "tpms_temperature_c"),
                    corner=corner,
                    timestamp_s=ts,
                    provenance=prov,
                    extractor_version=EXTRACTOR_VERSION,
                ))

        # --- Pressure deviation from placard (Pa) ---
        # Classification A: direct comparison of two telemetry-derived values.
        if p_kpa is None:
            features.append(Feature(
                name=f"pressure_deviation_from_placard_{corner}",
                value=None,
                unit="Pa",
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason=f"TPMS pressure missing for {corner}",
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=("tpms_pressure_kpa",),
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))
        else:
            dev_pa = (p_kpa - tyre_config.placard_pressure_kpa) * 1000.0
            features.append(Feature(
                name=f"pressure_deviation_from_placard_{corner}",
                value=dev_pa,
                unit="Pa",
                status=FeatureStatus.OK,
                unavailable_reason=None,
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=("tpms_pressure_kpa",),
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))

        # --- Per-corner temperature (°C) ---
        if t_c is None:
            features.append(Feature(
                name=f"tyre_temperature_c_{corner}",
                value=None,
                unit="°C",
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason=f"TPMS temperature missing for {corner}",
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=t_inputs,
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))
        else:
            features.append(Feature(
                name=f"tyre_temperature_c_{corner}",
                value=t_c,
                unit="°C",
                status=FeatureStatus.OK,
                unavailable_reason=None,
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=t_inputs,
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))

        # --- Temperature rise above ambient (°C) ---
        if t_c is None or ambient_c is None:
            reasons = []
            if t_c is None:
                reasons.append(f"TPMS temperature missing for {corner}")
            if ambient_c is None:
                reasons.append("ambient temperature missing")
            features.append(Feature(
                name=f"temperature_rise_above_ambient_{corner}",
                value=None,
                unit="°C",
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason="; ".join(reasons),
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=("tpms_temperature_c", "ambient_temp_c"),
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))
        else:
            rise = t_c - ambient_c
            features.append(Feature(
                name=f"temperature_rise_above_ambient_{corner}",
                value=rise,
                unit="°C",
                status=FeatureStatus.OK,
                unavailable_reason=None,
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=("tpms_temperature_c", "ambient_temp_c"),
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))

    # --- Cross-corner pressure spread (vehicle-level) ---
    # Classification A: max minus min of four readings.
    available_pressures = [p for p in pressures_kpa.values() if p is not None]
    if len(available_pressures) < 2:
        features.append(Feature(
            name="cross_corner_pressure_spread_kpa",
            value=None,
            unit="kPa",
            status=FeatureStatus.UNAVAILABLE,
            unavailable_reason=(
                f"Need ≥2 corners with TPMS pressure to compute spread; "
                f"only {len(available_pressures)} available"
            ),
            directionality=Directionality.MAGNITUDE_ONLY,
            classification=Classification.A,
            inputs=("tpms_pressure_kpa",),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))
    else:
        spread = max(available_pressures) - min(available_pressures)
        features.append(Feature(
            name="cross_corner_pressure_spread_kpa",
            value=spread,
            unit="kPa",
            status=FeatureStatus.OK,
            unavailable_reason=None,
            directionality=Directionality.MAGNITUDE_ONLY,
            classification=Classification.A,
            inputs=("tpms_pressure_kpa",),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))

    return tuple(features)
