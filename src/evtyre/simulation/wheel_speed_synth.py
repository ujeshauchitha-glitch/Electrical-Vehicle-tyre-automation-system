"""Synthetic wheel-speed windows with a known resonance, for extractor tests.

The signal is a constant mean speed plus a lightly damped narrow-band
oscillation at ``resonance_hz`` plus white encoder noise, sampled at
``sample_rate_hz`` with no anti-alias filtering unless the caller simulates it
(``alias_hz`` injects an out-of-band tone that folds back). It is a test
fixture, not a model of real ABS encoder behaviour.
"""

from __future__ import annotations

import numpy as np

from ..schema.common import CORNERS
from ..signal.windowed import WindowedSample


def synth_window(
    resonance_hz: dict[str, float],
    sample_rate_hz: float,
    duration_s: float = 2.0,
    mean_rad_s: float = 70.0,
    amplitude_rad_s: float = 0.05,
    noise_rad_s: float = 0.03,
    q_factor: float = 20.0,
    alias_hz: float | None = None,
    seed: int = 0,
    missing_fraction: float = 0.0,
    gap: tuple[float, float] | None = None,
) -> WindowedSample:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate_hz)
    t = np.arange(n) / sample_rate_hz
    columns = []
    for corner in CORNERS:
        f0 = resonance_hz[corner]
        excitation = rng.normal(0.0, 1.0, n)
        # two-pole resonator driven by white noise: narrow band around f0
        r = np.exp(-np.pi * f0 / (q_factor * sample_rate_hz))
        a1, a2 = -2 * r * np.cos(2 * np.pi * f0 / sample_rate_hz), r * r
        y = np.zeros(n)
        for i in range(2, n):
            y[i] = excitation[i] - a1 * y[i - 1] - a2 * y[i - 2]
        y *= amplitude_rad_s / y.std()
        sig = mean_rad_s + y + rng.normal(0.0, noise_rad_s, n)
        if alias_hz is not None:
            sig = sig + 0.3 * amplitude_rad_s * np.sin(2 * np.pi * alias_hz * t)
        columns.append(sig)
    data = np.column_stack(columns)
    mask = np.zeros_like(data, dtype=bool)
    if missing_fraction > 0:
        mask |= rng.random(data.shape) < missing_fraction
    if gap is not None:
        mask[int(gap[0] * sample_rate_hz):int(gap[1] * sample_rate_hz), :] = True
    return WindowedSample(
        data=np.where(mask, np.nan, data),
        channel_names=tuple(f"wheel_speed_{c}" for c in CORNERS),
        sample_rate_hz=sample_rate_hz,
        start_time_s=0.0,
        source="simulated",
        provenance="wheel_speed_synth",
        missing_mask=mask if mask.any() else None,
    )
