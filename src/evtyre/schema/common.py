"""Shared primitives for the Phase 1 telemetry schema.

CORNERS mirrors legacy/ev_tyre_fusion.py's corner labelling (FL/FR/RL/RR) but is
defined independently here rather than imported from legacy - src/ code must not
depend on legacy/, per CLAUDE.md section 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

CORNERS: tuple[str, ...] = ("FL", "FR", "RL", "RR")


class SensorStatus(str, Enum):
    """Explicit status for every sensor reading in a TelemetryFrame.

    OK            - a value was present and within its plausible physical range.
    MISSING       - no value was available this frame (dropped CAN frame, an
                    unpopulated field, a sensor the vehicle doesn't have, ...).
    OUT_OF_RANGE  - a value was present but fell outside the plausible physical
                    range for its channel and is therefore not trusted.

    Phase 1 never invents a replacement value for a MISSING or OUT_OF_RANGE
    reading. See SensorReading below for what is preserved in each case.
    """

    OK = "ok"
    MISSING = "missing"
    OUT_OF_RANGE = "out_of_range"


@dataclass(frozen=True)
class SensorReading:
    """One normalized sensor value plus its validity status.

    `value` is the original number for OK and OUT_OF_RANGE readings (kept for
    diagnostics even when not trusted), and always None for MISSING readings.
    Downstream code must check `status` (or use `is_usable`) before trusting
    `value` - a non-None value alone does not mean the reading is usable.
    """

    value: Optional[float]
    status: SensorStatus

    def __post_init__(self) -> None:
        if self.status is SensorStatus.MISSING and self.value is not None:
            raise ValueError("a MISSING SensorReading must have value=None")
        if self.status is not SensorStatus.MISSING and self.value is None:
            raise ValueError(f"a {self.status.value} SensorReading must have a non-None value")

    @property
    def is_usable(self) -> bool:
        return self.status is SensorStatus.OK and self.value is not None

    @staticmethod
    def missing() -> "SensorReading":
        return SensorReading(value=None, status=SensorStatus.MISSING)
