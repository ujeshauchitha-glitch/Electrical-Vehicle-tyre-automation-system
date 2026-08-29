"""Recoverable energy: how much of today's road load is avoidable.

THE SIGN MATTERS AND IT IS COUNTERINTUITIVE
-------------------------------------------
Worn tread REDUCES rolling resistance — roughly 20% lower from new to worn —
because there is less rubber to deform on every revolution. So "your worn tyres
are wasting your battery" is backwards: fitting new tyres COSTS an EV range.
Tread is a WET GRIP argument, which is a safety case, not an energy one. This
module must never present tread wear as recoverable energy, and tread is
deliberately held fixed in the comparison below.

The honest energy story is pressure and alignment, both recoverable in an
afternoon. Of those two, only pressure is currently observable, so only the
pressure component is reported. The toe component is UNAVAILABLE because toe is
UNOBSERVABLE — reporting a "total" that silently omits it would understate the
recoverable figure while looking complete.
"""

from __future__ import annotations

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..estimation.estimator import PhysicsConfig
from ..estimation.physics import rolling_resistance_coeff
from ..estimation.schema import TyreStateEstimate
from ..schema.common import CORNERS
from .contract import Decision, require_observed


def recoverable_energy(
    estimate: TyreStateEstimate,
    vehicle_config: VehicleConfig,
    tyre_config: TyreConfig,
    physics: PhysicsConfig,
    reference_temp_c: float,
) -> tuple[Decision, ...]:
    """Excess rolling resistance attributable to under-inflation.

    Compares current estimated pressure against placard, holding tread and
    temperature fixed so the comparison isolates inflation alone.
    """
    decisions: list[Decision] = []
    press_states = [f"press_{c}" for c in CORNERS]
    tread_states = [f"tread_{c}" for c in CORNERS]

    blocked = require_observed(estimate, press_states)
    if blocked is not None:
        decisions.append(Decision.unavailable(
            name="recoverable_energy_pressure_pct",
            unit="%",
            reason=f"Pressure not usable for all corners: {blocked}",
            basis=press_states,
        ))
    else:
        by_name = {s.name: s for s in estimate.states}
        # Tread is held FIXED at its current estimate in BOTH legs of the
        # comparison, so it cancels. Even when tread is only WEAK this is
        # sound: the same value appears on both sides. What must never happen
        # is counting tread wear itself as recoverable — see module docstring.
        tread_by_corner = {
            c: by_name[f"tread_{c}"].value for c in CORNERS
        }

        c_rr_now = [
            rolling_resistance_coeff(
                tread_by_corner[c], by_name[f"press_{c}"].value,
                reference_temp_c, tyre_config, physics,
            )
            for c in CORNERS
        ]
        c_rr_placard = [
            rolling_resistance_coeff(
                tread_by_corner[c], tyre_config.placard_pressure_kpa,
                reference_temp_c, tyre_config, physics,
            )
            for c in CORNERS
        ]
        now = sum(c_rr_now) / len(c_rr_now)
        ideal = sum(c_rr_placard) / len(c_rr_placard)
        pct = 100.0 * (now - ideal) / ideal

        decisions.append(Decision.ok(
            name="recoverable_energy_pressure_pct",
            value=pct,
            unit="%",
            basis=press_states + tread_states,
            caveats=(
                "rolling-resistance model and its constants are UNVALIDATED",
                "tread held fixed in the comparison: worn tread LOWERS rolling "
                "resistance, so counting wear as waste would be physically "
                "backwards and would inflate this number",
                "rolling resistance only; excludes aero, grade and inertia",
            ),
        ))

    # --- Toe component: structurally unavailable ---
    toe_blocked = require_observed(estimate, ["toe^2"])
    decisions.append(Decision.unavailable(
        name="recoverable_energy_alignment_pct",
        unit="%",
        reason=(
            f"Toe is not OBSERVED, so its drag contribution cannot be "
            f"quantified: {toe_blocked}. The road-load channel that could "
            f"carry toe is currently mean(C_rr) recomputed from pressure and "
            f"contains no toe information; a torque-derived road-load "
            f"measurement is required, and even then toe drag is confounded "
            f"with road grade (interface gap G6)."
        ),
        basis=["toe^2"],
    ))

    # --- Total: withheld while a component is missing ---
    decisions.append(Decision.unavailable(
        name="recoverable_energy_total_pct",
        unit="%",
        reason=(
            "Withheld: the alignment component is unavailable, so any 'total' "
            "would silently omit it and understate the recoverable figure "
            "while appearing complete. Report the pressure component on its "
            "own instead."
        ),
        basis=press_states + ["toe^2"],
    ))

    return tuple(decisions)
