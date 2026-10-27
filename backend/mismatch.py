"""Model-mismatch layer: makes the simulated truth differ from the estimator's model.

The base simulator generates data with the same equations the estimator inverts,
so it cannot expose model error. This module perturbs the generated channels
with the kinds of errors a real vehicle would add. Every magnitude below is an
ASSUMED order of magnitude, not a measurement; the study is a stress test of
which estimator design degrades gracefully, not a prediction of real accuracy.

Perturbations (persistent per vehicle, so they are the same at install and later):

* per-tyre resonance scale (tyre-to-tyre stiffness/inertia spread)
* resonance temperature coefficient (rubber softening) that the model lacks
* per-tyre radius offset (tyre-size tolerance), which also shifts the axle ratio
* road-load scale (surface, wind, aero), applied to channels 10 and 11
* lateral-accelerometer and steering-angle zero offsets
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .estimator import CORNERS, TyreState
from .extended import ExtraNoise, measure_ext
from .physics import Vehicle


@dataclass(frozen=True)
class Mismatch:
    f_scale: np.ndarray = field(default_factory=lambda: np.zeros(4))
    f_temp_coeff: float = 0.0
    radius_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(4))
    crr_scale: float = 1.0
    latacc_bias: float = 0.0
    steer_bias_deg: float = 0.0

    @staticmethod
    def draw(rng: np.random.Generator, level: float = 1.0) -> "Mismatch":
        """level scales every spread; 0 is no mismatch, 1 is the default stress."""
        return Mismatch(
            f_scale=rng.normal(0.0, 0.02 * level, 4),
            f_temp_coeff=float(-0.002 + rng.normal(0.0, 0.001 * level)),
            radius_offset_m=rng.normal(0.0, 0.001 * level, 4),
            crr_scale=float(1.0 + rng.normal(0.0, 0.08 * level)),
            latacc_bias=float(rng.normal(0.0, 0.05 * level)),
            steer_bias_deg=float(rng.normal(0.0, 0.1 * level)),
        )


class _NoTolerance(ExtraNoise):
    radius_tolerance_mm = 0.0


def measure_mismatched(state: TyreState, rng: np.random.Generator, mm: Mismatch,
                       v_ms: float = 22.0, esc: bool = True, gnss: bool = True,
                       slip_bias: float = 0.0):
    """measure_ext with the Mismatch applied to the generated channels."""
    z, T = measure_ext(state, rng, v_ms=v_ms, esc=esc, gnss=gnss,
                       noise=_NoTolerance, slip_bias=slip_bias)
    for i, c in enumerate(CORNERS):
        z[4 + i] *= (1.0 + mm.f_scale[i]) * (1.0 + mm.f_temp_coeff * (state.temp[c] - 25.0))
    r0 = Vehicle.r_belt
    z[8] += (mm.radius_offset_m[1] - mm.radius_offset_m[0]) / r0
    z[9] += (mm.radius_offset_m[3] - mm.radius_offset_m[2]) / r0
    if gnss:
        z[14:18] += mm.radius_offset_m
    rolling_nominal = Vehicle.C_rr0 * Vehicle.mass * Vehicle.g
    z[10] *= mm.crr_scale
    if not np.isnan(z[11]):
        z[11] += (mm.crr_scale - 1.0) * rolling_nominal * r0 / (Vehicle.gear_ratio * Vehicle.drivetrain_eff)
    if esc:
        z[12] += mm.latacc_bias
        z[13] += mm.steer_bias_deg
    return z, T
