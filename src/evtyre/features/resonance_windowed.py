"""Resonance peak extraction from a WindowedSample.

Classification C: the spectral estimator is standard, but that the peak it
finds is the tyre torsional mode (and which mode) is a project hypothesis
awaiting bench validation. The search band is a required argument so the
caller, not this module, decides between the first (~30-80 Hz) and second
(~100-300 Hz) mode hypotheses. frame-based ``resonance.extract`` still raises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema.common import CORNERS
from ..signal.windowed import WindowedSample
from .contract import Classification, Directionality, Feature, FeatureStatus

EXTRACTOR_VERSION = "0.1.0"
MAX_MISSING_FRACTION = 0.20


@dataclass(frozen=True)
class PeakEstimate:
    frequency_hz: float
    snr_db: float
    resolution_hz: float


def estimate_peak(
    x: np.ndarray,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    segment_resolution_hz: float = 1.0,
) -> PeakEstimate:
    """Welch-averaged (50 % overlap, ``segment_resolution_hz`` bins) Hann periodogram peak with a power-weighted centroid over the contiguous
    half-maximum region around the peak (less variable than the single-bin max
    for a broad, stochastically excited resonance).

    SNR is the peak power over the median power in the band.
    """
    lo, hi = band_hz
    if not 0 < lo < hi < sample_rate_hz / 2:
        raise ValueError("band_hz must satisfy 0 < lo < hi < Nyquist")
    seg_len = max(min(round(sample_rate_hz / segment_resolution_hz), len(x)), 8)
    hop = seg_len // 2
    window = np.hanning(seg_len)
    starts = range(0, len(x) - seg_len + 1, hop)
    psd = sum(
        np.abs(np.fft.rfft((x[s:s + seg_len] - x[s:s + seg_len].mean()) * window)) ** 2
        for s in starts
    )
    freqs = np.fft.rfftfreq(seg_len, 1.0 / sample_rate_hz)
    in_band = np.where((freqs >= lo) & (freqs <= hi))[0]
    if len(in_band) < 3:
        raise ValueError("band contains fewer than 3 frequency bins; lengthen the window")
    k = in_band[np.argmax(psd[in_band])]
    lo_i, hi_i = k, k
    while lo_i > in_band[0] and psd[lo_i - 1] >= 0.5 * psd[k]:
        lo_i -= 1
    while hi_i < in_band[-1] and psd[hi_i + 1] >= 0.5 * psd[k]:
        hi_i += 1
    weights = psd[lo_i:hi_i + 1]
    centroid = float(np.sum(freqs[lo_i:hi_i + 1] * weights) / np.sum(weights))
    df = freqs[1] - freqs[0]
    floor = np.median(psd[in_band])
    return PeakEstimate(
        frequency_hz=centroid,
        snr_db=float(10 * np.log10(psd[k] / max(floor, 1e-300))),
        resolution_hz=float(df),
    )


def extract(
    sample: WindowedSample,
    band_hz: tuple[float, float],
    min_snr_db: float = 6.0,
    channel_prefix: str = "wheel_speed_",
) -> tuple[Feature, ...]:
    """One ``resonance_frequency`` Feature per corner present in the sample."""
    features = []
    for corner in CORNERS:
        name = f"{channel_prefix}{corner}"
        base = dict(
            name="resonance_frequency",
            unit="Hz",
            directionality=Directionality.NATURAL,
            classification=Classification.C,
            inputs=(name,),
            corner=corner,
            timestamp_s=sample.start_time_s,
            provenance=sample.provenance,
            extractor_version=EXTRACTOR_VERSION,
        )

        def unavailable(reason: str) -> Feature:
            return Feature(value=None, status=FeatureStatus.UNAVAILABLE,
                           unavailable_reason=reason, **base)

        if name not in sample.channel_names:
            features.append(unavailable(f"channel {name} not in window"))
            continue
        values, valid = sample.channel(name)
        missing_fraction = 1.0 - valid.mean()
        if missing_fraction > MAX_MISSING_FRACTION:
            features.append(unavailable(
                f"{missing_fraction:.0%} of samples missing (limit {MAX_MISSING_FRACTION:.0%})"))
            continue
        good = np.flatnonzero(valid)
        values = values[good[0]:good[-1] + 1]
        if not valid[good[0]:good[-1] + 1].all():
            features.append(unavailable("mid-window gap; interpolation is not permitted"))
            continue
        try:
            peak = estimate_peak(values, sample.sample_rate_hz, band_hz)
        except ValueError as exc:
            features.append(unavailable(str(exc)))
            continue
        if peak.snr_db < min_snr_db:
            features.append(unavailable(
                f"no clear peak in band: SNR {peak.snr_db:.1f} dB < {min_snr_db} dB"))
            continue
        features.append(Feature(value=peak.frequency_hz, status=FeatureStatus.OK,
                                unavailable_reason=None, **base))
    return tuple(features)
