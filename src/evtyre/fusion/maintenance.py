"""Maintenance view: cold-compensated pressure and inflation flags.

This is currently the only Phase 4 output that can be fully delivered, because
pressure is the only OBSERVED state in the current channel set.

Why cold-compensate for the report but never for the physics: a hot tyre reads
high. Judging inflation on running pressure makes every warm tyre look healthy
and every cold morning look like a puncture. So the REPORT is normalised to a
reference temperature and the estimator's physics is not — the same number
serving two different consumers. Feeding cold-equivalent pressure back into the
tyre model would be wrong; running pressure is what actually inflates the
carcass.
"""

from __future__ import annotations

from ..config.tyre import TyreConfig
from ..estimation.schema import TyreStateEstimate
from ..schema.common import CORNERS
from .config import MaintenanceConfig
from .contract import Decision, require_observed

_C_TO_K = 273.15
_P_ATM_KPA = 101.325


def cold_equivalent_pressure_kpa(
    running_gauge_kpa: float, tyre_temp_c: float, reference_temp_c: float
) -> float:
    """Gay-Lussac normalisation at constant volume: P/T = const.

    Works in ABSOLUTE pressure, because the gas law does. Converting to gauge
    before the ratio would be a physics error.
    """
    t_running_k = tyre_temp_c + _C_TO_K
    if t_running_k <= 0:
        raise ValueError(f"tyre temperature {tyre_temp_c} C is at or below 0 K")
    p_abs = running_gauge_kpa + _P_ATM_KPA
    p_ref_abs = p_abs * (reference_temp_c + _C_TO_K) / t_running_k
    return p_ref_abs - _P_ATM_KPA


def maintenance_view(
    estimate: TyreStateEstimate,
    tyre_config: TyreConfig,
    maintenance_config: MaintenanceConfig,
    corner_temps_c: dict[str, float | None],
) -> tuple[Decision, ...]:
    """Cold-equivalent pressure per corner, plus inflation flags.

    A corner is UNAVAILABLE if its pressure state is not OBSERVED or if its
    temperature is missing — cold compensation without a temperature is not
    possible, and assuming one would be fabrication.
    """
    decisions: list[Decision] = []
    cold_by_corner: dict[str, float] = {}

    for corner in CORNERS:
        state_name = f"press_{corner}"
        blocked = require_observed(estimate, [state_name])
        temp = corner_temps_c.get(corner)

        if blocked is not None:
            decisions.append(Decision.unavailable(
                name="cold_equivalent_pressure_kpa",
                unit="kPa",
                reason=f"Pressure not usable: {blocked}",
                basis=[state_name],
                corner=corner,
            ))
            continue

        if temp is None:
            decisions.append(Decision.unavailable(
                name="cold_equivalent_pressure_kpa",
                unit="kPa",
                reason=(
                    f"No tyre temperature for {corner}; cold compensation "
                    f"requires it and a substituted temperature would be an "
                    f"assumption presented as a measurement"
                ),
                basis=[state_name],
                corner=corner,
            ))
            continue

        running = next(s.value for s in estimate.states if s.name == state_name)
        cold = cold_equivalent_pressure_kpa(
            running, temp, tyre_config.cold_reference_temperature_c
        )
        cold_by_corner[corner] = cold
        decisions.append(Decision.ok(
            name="cold_equivalent_pressure_kpa",
            value=cold,
            unit="kPa",
            basis=[state_name],
            caveats=(
                f"normalised to {tyre_config.cold_reference_temperature_c:.0f} C "
                f"from a measured {temp:.0f} C; report only, never fed back "
                f"into the physics",
            ),
            corner=corner,
        ))

    # --- Under-inflation flags, on the cold-equivalent figure ---
    for corner in CORNERS:
        if corner not in cold_by_corner:
            decisions.append(Decision.unavailable(
                name="inflation_deficit_kpa",
                unit="kPa",
                reason=f"cold_equivalent_pressure_kpa unavailable for {corner}",
                basis=[f"press_{corner}"],
                corner=corner,
            ))
            continue
        deficit = tyre_config.placard_pressure_kpa - cold_by_corner[corner]
        caveats = ()
        if deficit > maintenance_config.low_pressure_margin_kpa:
            caveats = (
                f"LOW: {deficit:.0f} kPa below placard, over the "
                f"{maintenance_config.low_pressure_margin_kpa:.0f} kPa "
                f"action threshold — check for a leak",
            )
        decisions.append(Decision.ok(
            name="inflation_deficit_kpa",
            value=deficit,
            unit="kPa",
            basis=[f"press_{corner}"],
            caveats=caveats,
            corner=corner,
        ))

    # --- Cross-corner spread (vehicle level) ---
    if len(cold_by_corner) < 2:
        decisions.append(Decision.unavailable(
            name="cross_corner_pressure_spread_kpa",
            unit="kPa",
            reason=(
                f"Need at least 2 corners with a cold-equivalent pressure; "
                f"have {len(cold_by_corner)}"
            ),
            basis=[f"press_{c}" for c in CORNERS],
        ))
    else:
        spread = max(cold_by_corner.values()) - min(cold_by_corner.values())
        caveats = ()
        if spread > maintenance_config.cross_corner_spread_limit_kpa:
            caveats = (
                f"UNEVEN: {spread:.0f} kPa spread across corners, over the "
                f"{maintenance_config.cross_corner_spread_limit_kpa:.0f} kPa "
                f"threshold",
            )
        if len(cold_by_corner) < len(CORNERS):
            caveats = caveats + (
                f"computed over {len(cold_by_corner)} of {len(CORNERS)} corners",
            )
        decisions.append(Decision.ok(
            name="cross_corner_pressure_spread_kpa",
            value=spread,
            unit="kPa",
            basis=[f"press_{c}" for c in sorted(cold_by_corner)],
            caveats=caveats,
        ))

    return tuple(decisions)
