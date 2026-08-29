"""FusedTyreReport: the Phase 4 -> Phase 5 contract, and the Phase 4 entry point.

Per CLAUDE.md section 5, this carries the decision-ready fields plus the
underlying TyreStateEstimate, a timestamp and an odometer reading. The last two
are what make Phase 5 possible at all: degradation RATE needs a time or
distance axis, and the legacy estimator had neither.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..estimation.estimator import PhysicsConfig
from ..estimation.schema import TyreStateEstimate
from .config import FrictionConfig, MaintenanceConfig
from .contract import Decision, DecisionStatus
from .energy import recoverable_energy
from .maintenance import maintenance_view
from .traction import torque_ceiling

REPORT_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class FusedTyreReport:
    """Decision-ready output plus the estimate it came from."""

    decisions: tuple[Decision, ...]
    estimate: TyreStateEstimate

    timestamp_s: float
    odometer_km: float | None
    source: str
    """'real', 'replay' or 'simulated', propagated unchanged from Phase 1."""

    report_schema_version: str = REPORT_SCHEMA_VERSION

    def by_name(self, name: str, corner: str | None = None) -> Decision | None:
        for d in self.decisions:
            if d.name == name and d.corner == corner:
                return d
        return None

    @property
    def actionable(self) -> tuple[Decision, ...]:
        return tuple(d for d in self.decisions if d.is_actionable)

    @property
    def withheld(self) -> tuple[Decision, ...]:
        """Decisions Phase 4 declined to make, each with its reason.

        Not an error list. This is the honest half of the report and should be
        surfaced, not filtered out: it is what stops a consumer assuming a
        missing field simply did not apply.
        """
        return tuple(
            d for d in self.decisions if d.status is DecisionStatus.UNAVAILABLE
        )

    @property
    def is_simulated(self) -> bool:
        """True if any part of this report derives from synthetic data.

        CLAUDE.md section 8 rule 4: a dashboard or log must never be able to
        present a simulated result as a real-vehicle one.
        """
        return self.source == "simulated"


def build_report(
    estimate: TyreStateEstimate,
    vehicle_config: VehicleConfig,
    tyre_config: TyreConfig,
    physics: PhysicsConfig,
    friction: FrictionConfig,
    maintenance: MaintenanceConfig,
    corner_temps_c: dict[str, float | None],
    reference_temp_c: float,
) -> FusedTyreReport:
    """Run every Phase 4 consumer over one estimate."""
    decisions: list[Decision] = []
    decisions.extend(maintenance_view(
        estimate, tyre_config, maintenance, corner_temps_c,
    ))
    decisions.extend(torque_ceiling(
        estimate, vehicle_config, tyre_config, physics, friction, wet=True,
    ))
    decisions.extend(torque_ceiling(
        estimate, vehicle_config, tyre_config, physics, friction, wet=False,
    ))
    decisions.extend(recoverable_energy(
        estimate, vehicle_config, tyre_config, physics, reference_temp_c,
    ))

    return FusedTyreReport(
        decisions=tuple(decisions),
        estimate=estimate,
        timestamp_s=estimate.timestamp_s,
        odometer_km=estimate.odometer_km,
        source=estimate.source,
    )
