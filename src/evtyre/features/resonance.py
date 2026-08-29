"""Resonance frequency feature extractor — STUB.

This extractor is classified **D** (not implementable with available
telemetry) because:

1. TelemetryFrame is a single-instant snapshot with no sample rate and
   no anti-alias contract (see docs/G1-windowed-sample-contract.md).
2. Resonance / vibration only exists across a span of time.
3. The windowed sample type (G1) has not been implemented yet.

It raises ``NotImplementedError`` with "G1" in the message.  This is
intentional: a stub that returns a value invites someone to fill it in
without reopening G1.  A stub that raises forces the caller to confront
the gap.

The resonance channel carries the whole tread estimate: ablation shows
0.10 mm tread error with it and 1.41 mm without, against a 1.6 mm
legal limit.  This is the highest-risk open item in the project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..schema.telemetry import TelemetryFrame
from .contract import Classification, Directionality, Feature, FeatureStatus

if TYPE_CHECKING:
    pass

EXTRACTOR_VERSION = "0.1.0"

_G1_MESSAGE = (
    "Resonance extraction requires a windowed sample type (G1) that does "
    "not yet exist.  TelemetryFrame is a single-instant snapshot with no "
    "sample rate and no anti-alias contract.  See "
    "docs/G1-windowed-sample-contract.md for the design.  "
    "DO NOT implement this without resolving the G1 physics question."
)


def extract(
    frame: TelemetryFrame,
    vehicle_config: VehicleConfig,
    tyre_config: TyreConfig,
) -> tuple[Feature, ...]:
    """Extract resonance features — BLOCKED by G1.

    Raises
    ------
    NotImplementedError
        Always, with "G1" in the message.  This extractor must never
        return a value — doing so would silently fill in the resonance
        channel without resolving the underlying physics question.
    """
    raise NotImplementedError(_G1_MESSAGE)
