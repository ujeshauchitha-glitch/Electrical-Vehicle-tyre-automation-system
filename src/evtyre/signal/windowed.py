"""WindowedSample: the G1 time-series buffer (see docs/G1-windowed-sample-contract.md).

The sample rate is always supplied by the caller. Nothing in this module picks
a rate or a resonance band, because which torsional mode carries the tread
signal is still an open bench-experiment question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema.telemetry import FrameSource


@dataclass(frozen=True)
class WindowedSample:
    """A fixed-length window of per-channel time-series data.

    ``missing_mask`` is True where a sample is missing or corrupt; None means
    the window is complete. ``anti_alias_cutoff_hz`` is the analogue filter
    cutoff upstream of the ADC, or None if unknown (extractors then treat the
    whole band below Nyquist as untrusted for alias-sensitive claims).
    """

    data: np.ndarray
    channel_names: tuple[str, ...]
    sample_rate_hz: float
    start_time_s: float
    source: FrameSource
    provenance: str
    missing_mask: np.ndarray | None = None
    anti_alias_cutoff_hz: float | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("data must have shape (n_samples, n_channels)")
        if self.data.shape[1] != len(self.channel_names):
            raise ValueError("channel_names must match the number of data columns")
        if not self.sample_rate_hz > 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.missing_mask is not None and self.missing_mask.shape != self.data.shape:
            raise ValueError("missing_mask must have the same shape as data")
        if (
            self.anti_alias_cutoff_hz is not None
            and self.anti_alias_cutoff_hz > self.sample_rate_hz / 2
        ):
            raise ValueError("anti_alias_cutoff_hz must not exceed Nyquist")

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + (self.n_samples - 1) / self.sample_rate_hz

    def channel(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (values, valid_mask) for one channel."""
        j = self.channel_names.index(name)
        valid = (
            np.ones(self.n_samples, dtype=bool)
            if self.missing_mask is None
            else ~self.missing_mask[:, j]
        )
        return self.data[:, j], valid & np.isfinite(self.data[:, j])
