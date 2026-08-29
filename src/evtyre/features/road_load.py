"""Road load decomposition and energy features.

Road load = rolling + aero + grade + acceleration.

CRITICAL — every physical parameter here must be a REQUIRED parameter with
NO default value and marked unvalidated.  VehicleConfig does not carry aero
or driveline parameters — they are supplied explicitly to this extractor.
A plausible-looking default drag coefficient would silently set the entire
rolling-resistance decomposition, which is the project's whole energy claim.

Worn tyres have LOWER rolling resistance (~20% lower new-to-worn).  Tread
depth is a wet-grip argument, not an energy one.  Do not write code or
comments implying wear costs energy.

Classification of road_load_coefficient: C (project hypothesis).
Its stated falsifier is an independent coast-down or dyno rolling-resistance
measurement.

Directionality of road_load_coefficient: NATURAL (signed-positive by
construction).  Do NOT mark it MAGNITUDE_ONLY.  The magnitude-only
property belongs to the map from toe to road load (toe drag is quadratic
and even, so toe's sign is not recoverable), and that inversion is Phase 3.
Do NOT emit any toe feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..schema.common import CORNERS, SensorReading, SensorStatus
from ..schema.telemetry import TelemetryFrame
from .contract import Classification, Directionality, Feature, FeatureStatus

if TYPE_CHECKING:
    pass

EXTRACTOR_VERSION = "0.1.0"

# Gravitational acceleration (SI).  This is a measured constant, not a guess.
_G_MS2: float = 9.80665

# Air density at sea level, 15 °C (ISA standard).  UNVALIDATED — listed as a
# guess because it varies with altitude and temperature.
_RHO_AIR_KG_M3: float = 1.225


@dataclass(frozen=True)
class RoadLoadParams:
    """Aero and driveline parameters that VehicleConfig does not carry.

    EVERY field is REQUIRED with NO default.  Every value is UNVALIDATED —
    it is a guess until checked against independent coast-down or dyno data.
    This is deliberately not a sensible-defaults pattern: a default drag
    coefficient would silently set the entire rolling-resistance decomposition.
    """

    drag_coefficient: float
    """Cd — UNVALIDATED guess.  Typical EV sedan: 0.23-0.28."""

    frontal_area_m2: float
    """A — UNVALIDATED guess.  Typical EV sedan: 2.2-2.5 m²."""

    driveline_efficiency: float
    """Fraction 0..1 — UNVALIDATED guess.  Typical single-speed EV: 0.92-0.97."""

    def __post_init__(self) -> None:
        if self.drag_coefficient <= 0:
            raise ValueError("drag_coefficient must be positive")
        if self.frontal_area_m2 <= 0:
            raise ValueError("frontal_area_m2 must be positive")
        if not 0.0 < self.driveline_efficiency <= 1.0:
            raise ValueError("driveline_efficiency must be in (0, 1]")


def extract(
    frame: TelemetryFrame,
    vehicle_config: VehicleConfig,
    tyre_config: TyreConfig,
    *,
    road_load_params: RoadLoadParams,
    air_density_kg_m3: float = _RHO_AIR_KG_M3,
    grade_rad: float = 0.0,
) -> tuple[Feature, ...]:
    """Extract road load decomposition features.

    Parameters
    ----------
    road_load_params : RoadLoadParams
        UNVALIDATED aero/driveline parameters.  Required — no defaults.
    air_density_kg_m3 : float
        Air density.  UNVALIDATED guess (ISA standard).  Provided as a
        keyword argument to make the guess explicit and overridable.
    grade_rad : float
        Road grade in radians.  Defaults to 0 (flat road) since no incline
        sensor is available.
    """
    features: list[Feature] = []
    ts = frame.timestamp_s
    prov = frame.source

    # Collect available data
    v_ms = frame.vehicle_speed_ms.value if frame.vehicle_speed_ms.is_usable else None
    accel = frame.accel_long_ms2.value if frame.accel_long_ms2.is_usable else None
    torque_nm = frame.motor_torque_nm.value if frame.motor_torque_nm.is_usable else None
    motor_speed = frame.motor_speed_rad_s.value if frame.motor_speed_rad_s.is_usable else None

    # Per-corner TPMS pressure (kPa) for rolling resistance estimation
    pressures_kpa: dict[str, float | None] = {}
    temps_c: dict[str, float | None] = {}
    for corner in CORNERS:
        pr = frame.tpms_pressure_kpa[corner]
        pressures_kpa[corner] = pr.value if pr.is_usable else None
        tr = frame.tpms_temperature_c[corner]
        temps_c[corner] = tr.value if tr.is_usable else None

    # --- Rolling resistance force (N) ---
    # F_roll = C_rr * m * g * cos(grade)
    # We estimate C_rr from tyre model parameters (UNVALIDATED functional form
    # from legacy code):
    #   C_rr = C_rr0 * (p_placard / p_actual)^p_exp * t_term * T_term
    # where t_term = (1 - span) + span * (tread / tread_new)
    # NOTE: worn tyres have LOWER rolling resistance (t_term < 1).
    # C_rr0 and exponents are UNVALIDATED guesses from legacy.
    _C_RR0 = 0.0090       # UNVALIDATED — legacy reference value
    _P_EXPONENT = 0.45    # UNVALIDATED — legacy reference value
    _TREAD_RR_SPAN = 0.20 # UNVALIDATED — legacy reference value
    _T_COEFF = 0.0015     # UNVALIDATED — legacy reference value
    _T_REF = 25.0         # UNVALIDATED — reference temperature (°C)

    available_c_rr = []
    for corner in CORNERS:
        p = pressures_kpa[corner]
        t = temps_c[corner]
        if p is not None and p > 0 and t is not None:
            p_term = (tyre_config.placard_pressure_kpa / p) ** _P_EXPONENT
            # t_term uses tread_new as reference — worn tread = lower Crr
            t_term = (1.0 - _TREAD_RR_SPAN) + _TREAD_RR_SPAN  # placeholder: tread unknown
            T_term = 1.0 + _T_COEFF * (t - _T_REF)
            c_rr = _C_RR0 * p_term * t_term * T_term
            available_c_rr.append(c_rr)

    if available_c_rr:
        c_rr_mean = sum(available_c_rr) / len(available_c_rr)
    else:
        c_rr_mean = None

    if c_rr_mean is not None:
        f_roll = c_rr_mean * vehicle_config.mass_kg * _G_MS2 * math.cos(grade_rad)
        features.append(Feature(
            name="rolling_resistance_force_n",
            value=f_roll,
            unit="N",
            status=FeatureStatus.OK,
            unavailable_reason=None,
            directionality=Directionality.NATURAL,
            classification=Classification.C,
            inputs=("tpms_pressure_kpa", "tpms_temperature_c"),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))
    else:
        features.append(Feature(
            name="rolling_resistance_force_n",
            value=None,
            unit="N",
            status=FeatureStatus.UNAVAILABLE,
            unavailable_reason="No corners with valid pressure and temperature data",
            directionality=Directionality.NATURAL,
            classification=Classification.C,
            inputs=("tpms_pressure_kpa", "tpms_temperature_c"),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))

    # --- Aerodynamic drag force (N) ---
    # F_aero = 0.5 * rho * CdA * v^2
    # UNVALIDATED: CdA = Cd * A
    cda = road_load_params.drag_coefficient * road_load_params.frontal_area_m2
    if v_ms is not None and v_ms >= 0:
        f_aero = 0.5 * air_density_kg_m3 * cda * v_ms ** 2
        features.append(Feature(
            name="aerodynamic_drag_force_n",
            value=f_aero,
            unit="N",
            status=FeatureStatus.OK,
            unavailable_reason=None,
            directionality=Directionality.NATURAL,
            classification=Classification.C,
            inputs=("vehicle_speed_ms",),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))
    else:
        features.append(Feature(
            name="aerodynamic_drag_force_n",
            value=None,
            unit="N",
            status=FeatureStatus.UNAVAILABLE,
            unavailable_reason="Vehicle speed unavailable",
            directionality=Directionality.NATURAL,
            classification=Classification.C,
            inputs=("vehicle_speed_ms",),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))

    # --- Grade resistance force (N) ---
    # F_grade = m * g * sin(grade)
    # Defaults to 0 on flat road.
    f_grade = vehicle_config.mass_kg * _G_MS2 * math.sin(grade_rad)
    features.append(Feature(
        name="grade_resistance_force_n",
        value=f_grade,
        unit="N",
        status=FeatureStatus.OK,
        unavailable_reason=None,
        directionality=Directionality.NATURAL,
        classification=Classification.C,
        inputs=(),
        corner=None,
        timestamp_s=ts,
        provenance=prov,
        extractor_version=EXTRACTOR_VERSION,
    ))

    # --- Inertial (acceleration) force (N) ---
    # F_inertia = m * a
    if accel is not None:
        f_inertia = vehicle_config.mass_kg * accel
        features.append(Feature(
            name="inertial_force_n",
            value=f_inertia,
            unit="N",
            status=FeatureStatus.OK,
            unavailable_reason=None,
            directionality=Directionality.NATURAL,
            classification=Classification.A,
            inputs=("accel_long_ms2",),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))
    else:
        features.append(Feature(
            name="inertial_force_n",
            value=None,
            unit="N",
            status=FeatureStatus.UNAVAILABLE,
            unavailable_reason="Longitudinal acceleration unavailable",
            directionality=Directionality.NATURAL,
            classification=Classification.A,
            inputs=("accel_long_ms2",),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))

    # --- Total road load force (N) ---
    # Sum of available components.  Missing components treated as 0 (their
    # contribution is unknown, not zero — this is flagged via individual
    # UNAVAILABLE features above).
    f_roll_val = next(
        (f.value for f in features if f.name == "rolling_resistance_force_n"
         and f.status == FeatureStatus.OK), 0.0
    )
    f_aero_val = next(
        (f.value for f in features if f.name == "aerodynamic_drag_force_n"
         and f.status == FeatureStatus.OK), 0.0
    )
    f_grade_val = next(
        (f.value for f in features if f.name == "grade_resistance_force_n"
         and f.status == FeatureStatus.OK), 0.0
    )
    f_inertia_val = next(
        (f.value for f in features if f.name == "inertial_force_n"
         and f.status == FeatureStatus.OK), 0.0
    )
    total_road_load = f_roll_val + f_aero_val + f_grade_val + f_inertia_val
    features.append(Feature(
        name="total_road_load_force_n",
        value=total_road_load,
        unit="N",
        status=FeatureStatus.OK,
        unavailable_reason=None,
        directionality=Directionality.NATURAL,
        classification=Classification.C,
        inputs=("tpms_pressure_kpa", "tpms_temperature_c",
                "vehicle_speed_ms", "accel_long_ms2"),
        corner=None,
        timestamp_s=ts,
        provenance=prov,
        extractor_version=EXTRACTOR_VERSION,
    ))

    # --- Road load coefficient (classification C) ---
    # C_road = F_total / (m * g * v) — the normalised road load.
    # Classification C: project hypothesis.
    # Falsifier: independent coast-down or dyno rolling-resistance measurement.
    # Directionality NATURAL: signed-positive by construction (energy is
    # consumed, not produced, in steady state).
    if v_ms is not None and v_ms > 1.0:  # avoid division by near-zero speed
        c_road = total_road_load / (vehicle_config.mass_kg * _G_MS2 * v_ms)
        features.append(Feature(
            name="road_load_coefficient",
            value=c_road,
            unit="",
            status=FeatureStatus.OK,
            unavailable_reason=None,
            directionality=Directionality.NATURAL,
            classification=Classification.C,
            inputs=("tpms_pressure_kpa", "tpms_temperature_c",
                    "vehicle_speed_ms", "accel_long_ms2"),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))
    else:
        features.append(Feature(
            name="road_load_coefficient",
            value=None,
            unit="",
            status=FeatureStatus.UNAVAILABLE,
            unavailable_reason=(
                "Vehicle speed unavailable or too low for coefficient calculation"
            ),
            directionality=Directionality.NATURAL,
            classification=Classification.C,
            inputs=("vehicle_speed_ms",),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))

    # --- CdA parameter feature (for transparency) ---
    features.append(Feature(
        name="effective_CdA_m2",
        value=cda,
        unit="m²",
        status=FeatureStatus.OK,
        unavailable_reason=None,
        directionality=Directionality.NATURAL,
        classification=Classification.C,
        inputs=(),
        corner=None,
        timestamp_s=ts,
        provenance=prov,
        extractor_version=EXTRACTOR_VERSION,
    ))

    return tuple(features)
