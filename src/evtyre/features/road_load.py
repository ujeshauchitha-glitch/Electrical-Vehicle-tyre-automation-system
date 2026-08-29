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

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..schema.common import CORNERS
from ..schema.telemetry import TelemetryFrame
from .contract import Classification, Directionality, Feature, FeatureStatus

EXTRACTOR_VERSION = "0.2.0"

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
    _T_COEFF = 0.0015     # UNVALIDATED — legacy reference value
    _T_REF = 25.0         # UNVALIDATED — reference temperature (°C)

    # NOTE ON THE OMITTED TREAD TERM.
    # Legacy's C_rr carries a tread factor  (1 - span) + span * (tread/tread_new).
    # It is deliberately absent here: tread depth is not known at Phase 2 — it is
    # a Phase 3 state, not a Phase 2 observable. Omitting the factor is equivalent
    # to holding it at its tread_new value of 1.0.
    # Do NOT reintroduce it as a constant expression: a previous revision wrote it
    # as `(1.0 - _TREAD_RR_SPAN) + _TREAD_RR_SPAN`, which is identically 1.0 while
    # looking like a computed term.
    available_c_rr = []
    for corner in CORNERS:
        p = pressures_kpa[corner]
        t = temps_c[corner]
        if p is not None and p > 0 and t is not None:
            p_term = (tyre_config.placard_pressure_kpa / p) ** _P_EXPONENT
            T_term = 1.0 + _T_COEFF * (t - _T_REF)
            c_rr = _C_RR0 * p_term * T_term
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
    # F_grade = m * g * sin(grade).
    #
    # ALWAYS UNAVAILABLE. TelemetryFrame carries no incline, pitch or elevation
    # channel (interface gap G6), so road grade is not measurable. The previous
    # revision emitted m*g*sin(0) = 0.0 with status OK, which reports the
    # flat-road ASSUMPTION as if it were a measurement.
    #
    # This is not the same category as the unvalidated aero constants: Cd and
    # frontal area are fixed vehicle properties that can be measured once, while
    # grade is a time-varying environmental state that changes continuously. On a
    # 2% incline the grade term is ~363 N against ~160 N of rolling resistance —
    # assuming it away silently would dominate everything downstream.
    features.append(Feature(
        name="grade_resistance_force_n",
        value=None,
        unit="N",
        status=FeatureStatus.UNAVAILABLE,
        unavailable_reason=(
            "No incline/pitch/elevation channel in TelemetryFrame (interface "
            "gap G6); road grade is not measurable, and a flat-road assumption "
            "is not a measurement"
        ),
        directionality=Directionality.NATURAL,
        classification=Classification.D,
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
    # A sum is only as available as its least available term. Every component
    # must be OK; a missing component makes the TOTAL unknown, not smaller.
    #
    # The previous revision defaulted each missing term to 0.0 via
    # `next(..., 0.0)` and emitted the sum with status OK — so a frame with every
    # sensor MISSING produced "total road load = 0.0 N, status OK". That is
    # fabrication, and it is what CLAUDE.md section 8 rule 2 forbids.
    _component_names = (
        "rolling_resistance_force_n",
        "aerodynamic_drag_force_n",
        "grade_resistance_force_n",
        "inertial_force_n",
    )
    _by_name = {f.name: f for f in features}
    _components = [_by_name[n] for n in _component_names]
    _missing = [c.name for c in _components if c.status is not FeatureStatus.OK]

    _total_inputs = ("tpms_pressure_kpa", "tpms_temperature_c",
                     "vehicle_speed_ms", "accel_long_ms2")

    if _missing:
        total_road_load = None
        features.append(Feature(
            name="total_road_load_force_n",
            value=None,
            unit="N",
            status=FeatureStatus.UNAVAILABLE,
            unavailable_reason=(
                "Cannot sum road load: component(s) unavailable — "
                + ", ".join(_missing)
            ),
            directionality=Directionality.NATURAL,
            classification=Classification.C,
            inputs=_total_inputs,
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))
    else:
        total_road_load = sum(c.value for c in _components)
        features.append(Feature(
            name="total_road_load_force_n",
            value=total_road_load,
            unit="N",
            status=FeatureStatus.OK,
            unavailable_reason=None,
            directionality=Directionality.NATURAL,
            classification=Classification.C,
            inputs=_total_inputs,
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))

        # --- Road load coefficient (dimensionless) ---
    # B1 FIX: This is the dimensionless C_rr mean, matching the legacy
    # estimator's predicted quantity: C_rr + F_toe/(m*g).
    # It does NOT include aero, inertia, or division by v -- those belong
    # to the force features above, not the estimator-facing coefficient.
    # Classification C: project hypothesis.
    # Falsifier: independent coast-down or dyno rolling-resistance measurement.
    if c_rr_mean is not None:
        features.append(Feature(
            name="road_load_coefficient",
            value=c_rr_mean,
            unit="",
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
            name="road_load_coefficient",
            value=None,
            unit="",
            status=FeatureStatus.UNAVAILABLE,
            unavailable_reason=(
                "No corners with valid pressure and temperature data"
            ),
            directionality=Directionality.NATURAL,
            classification=Classification.C,
            inputs=("tpms_pressure_kpa", "tpms_temperature_c"),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))

    # effective_CdA_m2 is deliberately NOT emitted: it is an echo of two
    # RoadLoadParams inputs (drag_coefficient * frontal_area_m2), not an
    # observable extracted from telemetry. Emitting config back as a "feature"
    # inflates the feature count with something no sensor contributed to.

    return tuple(features)
