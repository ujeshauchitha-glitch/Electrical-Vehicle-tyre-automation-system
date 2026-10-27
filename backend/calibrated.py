"""Install-baseline calibration: remove per-vehicle persistent channel offsets.

Idea: at tyre install the state is largely known (new tread from the spec or a
gauge reading, TPMS pressure, alignment just checked so camber ~ 0). Predicting
the channels from that known state and comparing with the measured channels
gives per-tyre multiplicative/additive offsets (resonance scale, axle-ratio
offset, radius offset, accelerometer and steering zero offsets). Those offsets
are then removed from later measurements before the standard estimator runs.

What this does NOT remove: anything that changes after install (surface and
wind scale on road load, tyre swap, drift), the resonance temperature
coefficient (only compensated with an ASSUMED value), and errors in the install
state itself (a 0.5 mm error in the install tread becomes a 0.5 mm bias).
Every later estimate is therefore "tread relative to the install baseline".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimator import CORNERS, IDX_PRESS, IDX_TREAD, N_STATE
from .extended import (
    M_LATACC, M_RADIUS, M_STEER, estimate_ext, predict_ext_batch,
)
from .physics import Vehicle

ASSUMED_F_TEMP_COEFF = -0.002   # per deg C; assumed value used for compensation
ASSUMED_F_TEMP_SIGMA = 0.001    # per deg C; assumed 1-sigma error of that value
ASSUMED_CRR_SCALE_SIGMA = 0.08  # fractional road-load scale uncertainty (surface, wind)
T_REF = Vehicle.T_ref


@dataclass(frozen=True)
class Baseline:
    f_scale: np.ndarray          # measured/predicted resonance, temperature-compensated
    ratio_offset: np.ndarray     # (2,) additive on channels 8, 9
    radius_offset: np.ndarray    # (4,) additive on channels 14..17, NaN if unused
    latacc_bias: float           # NaN if unused
    steer_bias: float            # NaN if unused
    t_install: float
    f_pred: np.ndarray = None    # predicted install resonance (Hz), for model-variance terms
    n_snapshots: int = 1
    f_noise_hz: float = 0.15


def _install_state(tread_mm: float, pressures_kpa: np.ndarray) -> np.ndarray:
    x = np.zeros(N_STATE)
    x[IDX_TREAD] = tread_mm
    x[IDX_PRESS] = pressures_kpa
    return x


def calibrate(z_install: np.ndarray, t_install: float, tread_mm: float,
              coeff: float = ASSUMED_F_TEMP_COEFF) -> Baseline:
    """Learn the baseline from K install snapshots, shape (K, 18), NaN = unused.

    Install pressure is taken from the TPMS channels (its noise averages down
    with K); toe^2 and camber are taken as 0 (just aligned).
    """
    zbar = np.nanmean(z_install, axis=0)
    x = _install_state(tread_mm, zbar[0:4])
    pred = predict_ext_batch(x, t_install)[0]
    comp = 1.0 + coeff * (t_install - T_REF)
    return Baseline(
        f_scale=zbar[4:8] / (pred[4:8] * comp),
        ratio_offset=zbar[8:10] - pred[8:10],
        radius_offset=zbar[M_RADIUS] - pred[M_RADIUS],
        latacc_bias=float(zbar[M_LATACC] - pred[M_LATACC]),
        steer_bias=float(zbar[M_STEER] - pred[M_STEER]),
        t_install=t_install,
        f_pred=pred[4:8],
        n_snapshots=int(z_install.shape[0]),
    )


def apply_baseline(z: np.ndarray, T_meas: float, b: Baseline,
                   coeff: float = ASSUMED_F_TEMP_COEFF) -> np.ndarray:
    out = z.copy()
    out[4:8] = z[4:8] / (b.f_scale * (1.0 + coeff * (T_meas - T_REF)))
    out[8:10] = z[8:10] - b.ratio_offset
    out[M_RADIUS] = z[M_RADIUS] - b.radius_offset
    out[M_LATACC] = z[M_LATACC] - b.latacc_bias
    out[M_STEER] = z[M_STEER] - b.steer_bias
    return out


def model_variance(T_meas: float, b: Baseline, sigma_coeff: float = ASSUMED_F_TEMP_SIGMA,
                   crr_sigma: float = ASSUMED_CRR_SCALE_SIGMA) -> np.ndarray:
    """Extra 18x18 covariance for uncertainty left after calibration.

    Resonance: an error in the shared temperature coefficient moves all four
    tyres together (rank-one block, not four independent terms), plus the
    independent averaging error of the install baseline.
    Road load and torque: one shared surface/wind scale on the rolling part,
    so channels 10 and 11 are fully correlated.
    """
    extra = np.zeros((18, 18))
    f = b.f_pred
    common = f * sigma_coeff * (T_meas - b.t_install)
    extra[4:8, 4:8] = np.outer(common, common) + np.eye(4) * (b.f_noise_hz / np.sqrt(b.n_snapshots)) ** 2
    s_load = crr_sigma * Vehicle.C_rr0
    s_torque = s_load * Vehicle.mass * Vehicle.g * Vehicle.r_belt / (Vehicle.gear_ratio * Vehicle.drivetrain_eff)
    extra[10, 10] = s_load ** 2
    extra[11, 11] = s_torque ** 2
    extra[10, 11] = extra[11, 10] = s_load * s_torque
    return extra


def estimate_calibrated(z: np.ndarray, T_meas: float, baseline: Baseline, **kw):
    kw.setdefault("extra_variance", model_variance(T_meas, baseline))
    return estimate_ext(apply_baseline(z, T_meas, baseline), T_meas, **kw)
