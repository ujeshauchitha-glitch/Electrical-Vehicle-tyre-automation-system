"""The Phase 1 acquisition extension point.

A real backend (a live CAN bus + DBC decoder, a bench-rig reader, a fleet
telemetry API client, ...) only needs to implement `read_raw()` and
`source_label()` on a TelemetrySource subclass. `read_frame()` and everything
downstream of it (the schema, Phase 2+) never need to change when that
backend is added - building it is explicitly out of scope here, see CLAUDE.md
section 9.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Optional

from ..schema.telemetry import TelemetryFrame
from .normalize import build_telemetry_frame
from .raw import RawTelemetrySample
from .validation import DEFAULT_LIMITS, ValidationLimits


class TelemetrySource(ABC):
    """Base class for anything that can produce a stream of TelemetryFrame."""

    limits: ValidationLimits = DEFAULT_LIMITS

    @abstractmethod
    def read_raw(self) -> Optional[RawTelemetrySample]:
        """Return the next raw sample, or None if none is currently available.

        Must never fabricate a sample - if no data is available, return None
        rather than a guessed or default-filled RawTelemetrySample.
        """

    @abstractmethod
    def source_label(self) -> str:
        """One of "real", "replay", or "simulated" - see TelemetryFrame.source."""

    def read_frame(self) -> Optional[TelemetryFrame]:
        raw = self.read_raw()
        if raw is None:
            return None
        return build_telemetry_frame(raw, source=self.source_label(), limits=self.limits)


class ReplayTelemetrySource(TelemetrySource):
    """Replays a fixed, already-captured sequence of RawTelemetrySample.

    For feeding logged real (or bench-recorded) data through Phase 1 without a
    live CAN connection - it replays samples supplied by the caller, it does
    not invent any. This is the mechanism CLAUDE.md section 9's development
    order (step 4) means by "initially against logged/replayed real data".
    """

    def __init__(self, samples: Iterable[RawTelemetrySample], label: str = "replay") -> None:
        self._iterator: Iterator[RawTelemetrySample] = iter(samples)
        self._label = label

    def read_raw(self) -> Optional[RawTelemetrySample]:
        return next(self._iterator, None)

    def source_label(self) -> str:
        return self._label
