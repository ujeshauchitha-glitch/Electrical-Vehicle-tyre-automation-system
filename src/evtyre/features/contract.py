"""Phase 2 feature contract — the authoritative data model for every
extracted observable.

Worker 1 implements this verbatim.  Every other Phase 2 extractor imports
and codes against this spec from the start.

Design note on future extensibility:
--------------------------------------
The ``extract`` signature accepts a ``TelemetryFrame``, which is a
single-instant snapshot.  When a windowed / time-series sample type is
added alongside TelemetryFrame (see Worker 5's G1 design doc), the
extractor signature will need to accept a Union or Protocol that covers
both types.  The contract itself — Feature, FeatureStatus, etc. — is
frame-type-agnostic and will survive that change unchanged.  Only the
``extract`` function signature will evolve.  This is an intentional
separation: the data contract is stable; the extractor dispatch is
expected to widen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.tyre import TyreConfig
    from ..config.vehicle import VehicleConfig
    from ..schema.telemetry import TelemetryFrame


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FeatureStatus(Enum):
    """Whether a feature value is usable."""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    OUT_OF_RANGE = "out_of_range"


class Directionality(Enum):
    """Whether a feature's sign is physically meaningful.

    NATURAL        — the sign carries information (e.g. signed acceleration).
    MAGNITUDE_ONLY — only the absolute value is physically meaningful (e.g.
                     toe-drag contribution, which is quadratic in toe angle).
    """

    NATURAL = "natural"
    MAGNITUDE_ONLY = "magnitude_only"


class Classification(Enum):
    """Confidence / provenance tier for a feature.

    A  — purely mathematical / telemetry-derived (e.g. pressure ratio).
    B  — literature-supported physical inference (e.g. cold-equivalent
         pressure via gas law, rolling-radius change from speed ratios).
    C  — project hypothesis, unvalidated against real data.
    D  — not implementable with available telemetry (e.g. resonance
         frequency — blocked by G1).
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Feature:
    """One extracted observable.

    Invariants enforced by ``__post_init__``:

    * ``status == OK`` ⟹ ``value is not None``.
    * ``status == UNAVAILABLE`` ⟹ ``value is None`` **and**
      ``unavailable_reason`` is a non-empty string.
    * ``status == OUT_OF_RANGE`` ⟹ ``value is not None`` (the raw value
      is kept for diagnostics, same convention as SensorReading).
    """

    name: str
    value: float | None          # None if and only if status is not OK
    unit: str                    # SI, explicit; "" for dimensionless
    status: FeatureStatus
    unavailable_reason: str | None   # required iff status is UNAVAILABLE
    directionality: Directionality
    classification: Classification
    inputs: tuple[str, ...]      # telemetry channel names actually consumed
    corner: str | None           # a member of CORNERS, or None if vehicle-level
    timestamp_s: float           # copied from the TelemetryFrame
    provenance: str              # copied from the frame's source label
    extractor_version: str       # your module's version constant

    def __post_init__(self) -> None:
        # Enforce both directions of the status/value invariant.
        if self.status is FeatureStatus.OK:
            if self.value is None:
                raise ValueError(
                    f"Feature {self.name!r}: status is OK but value is None"
                )
        elif self.status is FeatureStatus.UNAVAILABLE:
            if self.value is not None:
                raise ValueError(
                    f"Feature {self.name!r}: status is UNAVAILABLE but "
                    f"value is {self.value!r} (must be None)"
                )
            if not self.unavailable_reason:
                raise ValueError(
                    f"Feature {self.name!r}: status is UNAVAILABLE but "
                    f"unavailable_reason is missing or empty"
                )
        elif self.status is FeatureStatus.OUT_OF_RANGE:
            if self.value is None:
                raise ValueError(
                    f"Feature {self.name!r}: status is OUT_OF_RANGE but "
                    f"value is None"
                )

        # Corner must be a known corner or None.
        from ..schema.common import CORNERS
        if self.corner is not None and self.corner not in CORNERS:
            raise ValueError(
                f"Feature {self.name!r}: corner {self.corner!r} is not a "
                f"member of CORNERS {CORNERS}"
            )

        # inputs records which telemetry channel names were consumed.
        # An empty tuple is valid for features derived purely from config
        # parameters (e.g. grade resistance, effective CdA) — they consume
        # no telemetry channel at all.


# ---------------------------------------------------------------------------
# Extractor protocol
# ---------------------------------------------------------------------------

# The canonical extractor signature.  Every extractor module must define
# a function ``extract`` matching this signature:
#
#     def extract(
#         frame: TelemetryFrame,
#         vehicle_config: VehicleConfig,
#         tyre_config: TyreConfig,
#     ) -> tuple[Feature, ...]:
#
# This is documented here rather than enforced by a Protocol class, because
# Protocol enforcement would add a runtime dependency on typing_extensions
# for Python < 3.12 and the project is constrained to stdlib + numpy only.
# The tests below verify the signature structurally.
