"""Optional measurement channels layered on the frozen 12-channel estimator.

Adds, as channels 12..17 (NaN = unavailable, which leaves the base behaviour
unchanged):

  12     ESC lateral acceleration on a straight line (m/s^2)
  13     steering-angle offset needed to hold a straight line (deg)
  14-17  absolute rolling radius per corner, GNSS speed / wheel speed (m)

ALL constants here are assumed, none measured or fitted:

* Camber thrust uses C_gamma = 0.08 * C_alpha and the modelling assumption
  that the single camber state leans every wheel the same way in the vehicle
  frame. Mirrored left/right camber (the normal case) cancels and is invisible.
  Channels 12 and 13 are two readings of the same physical effect and their
  noise is treated as independent, which flatters the combined precision.
* Lateral-accelerometer bias, road crown/bank and wind are not modelled, and
  each maps directly onto a camber error, so channel 12 is the least
  trustworthy of the six.
* The radius channels need the nominal belt radius to be right. Tyre-size
  tolerance enters as a per-corner systematic sigma and GNSS speed scale error
  as a shared one; slip under torque is a bias that is only tested in
  scripts/ablation_ext.py, not absorbed in R.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .estimator import (
    IDX_PRESS, IDX_TOESQ, IDX_TREAD, N_STATE, TyreState,
    _FZ, measurement_covariance, measure, predict_batch, prior,
)
from .physics import CORNERS, Vehicle, effective_rolling_radius

N_EXT = 18
M_LATACC = 12
M_STEER = 13
M_RADIUS = slice(14, 18)

C_ALPHA = Vehicle.C_alpha
C_GAMMA = 0.08 * C_ALPHA


class ExtraNoise:
    latacc_sigma = 0.02          # m/s^2, assumed
    steer_sigma = 0.05           # deg, assumed
    gnss_scale_sigma = 0.002     # fractional speed-scale error, shared by corners
    wheel_speed_sigma = 0.0005   # fractional, per corner
    radius_tolerance_mm = 1.0    # per-corner tyre size tolerance, assumed


def camber_lateral_accel(camber_deg):
    return 4.0 * C_GAMMA * np.deg2rad(camber_deg) / Vehicle.mass


def steering_offset_deg(camber_deg):
    return -2.0 * (C_GAMMA / C_ALPHA) * np.asarray(camber_deg)


def predict_ext_batch(X: np.ndarray, T_meas=Vehicle.T_ref, v_ms=22.0,
                      accel_ms2=0.0, grade_rad=0.0) -> np.ndarray:
    X = np.atleast_2d(X)
    z = np.empty((X.shape[0], N_EXT))
    z[:, :12] = predict_batch(X, T_meas, v_ms, accel_ms2, grade_rad)
    z[:, M_LATACC] = camber_lateral_accel(X[:, 9])
    z[:, M_STEER] = steering_offset_deg(X[:, 9])
    z[:, M_RADIUS] = effective_rolling_radius(X[:, IDX_TREAD], X[:, IDX_PRESS], _FZ)
    return z


def jacobian_ext(x, T_meas=Vehicle.T_ref, v_ms=22.0, accel_ms2=0.0,
                 grade_rad=0.0, eps=1e-5) -> np.ndarray:
    steps = eps * np.eye(N_STATE)
    zp = predict_ext_batch(x + steps, T_meas, v_ms, accel_ms2, grade_rad)
    zm = predict_ext_batch(x - steps, T_meas, v_ms, accel_ms2, grade_rad)
    return ((zp - zm) / (2 * eps)).T


def extended_covariance(z: np.ndarray, noise=ExtraNoise) -> np.ndarray:
    R = np.zeros((N_EXT, N_EXT))
    R[:12, :12] = measurement_covariance(z_ref=z[:12])
    R[M_LATACC, M_LATACC] = noise.latacc_sigma ** 2
    R[M_STEER, M_STEER] = noise.steer_sigma ** 2
    r = Vehicle.r_belt
    random_var = (r * noise.wheel_speed_sigma) ** 2
    tol_var = (noise.radius_tolerance_mm / 1000.0) ** 2
    shared_var = (r * noise.gnss_scale_sigma) ** 2
    R[M_RADIUS, M_RADIUS] = shared_var + np.eye(4) * (random_var + tol_var)
    return R


def measure_ext(state: TyreState, rng: np.random.Generator, v_ms=22.0,
                esc=True, gnss=True, noise=ExtraNoise, slip_bias=0.0,
                include_motor_torque=True) -> Tuple[np.ndarray, float]:
    """Base measurement plus optional ESC and GNSS-radius channels.

    ``slip_bias`` is a fractional wheel-speed bias (e.g. driven-axle slip) that
    the estimator's R does not know about.
    """
    z12, T = measure(state, rng, v_ms=v_ms, include_motor_torque=include_motor_torque)
    z = np.full(N_EXT, np.nan)
    z[:12] = z12
    if esc:
        z[M_LATACC] = camber_lateral_accel(state.camber) + rng.normal(0, noise.latacc_sigma)
        z[M_STEER] = steering_offset_deg(state.camber) + rng.normal(0, noise.steer_sigma)
    if gnss:
        scale = rng.normal(0, noise.gnss_scale_sigma)
        for i, c in enumerate(CORNERS):
            r_true = effective_rolling_radius(state.tread[c], state.pressure[c], Vehicle.Fz(c))
            z[14 + i] = (r_true * (1 + scale + slip_bias + rng.normal(0, noise.wheel_speed_sigma))
                         + rng.normal(0, noise.radius_tolerance_mm / 1000.0))
    return z, T


def estimate_ext(z: np.ndarray, T_meas=Vehicle.T_ref, iters=6, v_ms=22.0,
                 accel_ms2=0.0, grade_rad=0.0, noise=ExtraNoise):
    """Iterated EKF over up to 18 channels; NaN channels are ignored."""
    x0, P0 = prior()
    R = extended_covariance(z, noise)
    nan_mask = np.isnan(z)
    idx = np.flatnonzero(nan_mask)
    R[idx, :] = 0.0
    R[:, idx] = 0.0
    R[idx, idx] = 1e30
    R_inv = np.linalg.inv(R)
    P0_inv = np.linalg.inv(P0)
    x = x0.copy()
    for _ in range(iters):
        H = jacobian_ext(x, T_meas, v_ms, accel_ms2, grade_rad)
        residual = z - predict_ext_batch(x, T_meas, v_ms, accel_ms2, grade_rad)[0]
        residual[nan_mask] = 0.0
        A = H.T @ R_inv @ H + P0_inv
        dx = np.linalg.solve(A, H.T @ R_inv @ residual + P0_inv @ (x0 - x))
        x = x + dx
        x[IDX_TREAD] = np.clip(x[IDX_TREAD], 0.5, 9.0)
        x[IDX_PRESS] = np.clip(x[IDX_PRESS], 100.0, 350.0)
        x[IDX_TOESQ] = max(0.0, x[IDX_TOESQ])
        if np.max(np.abs(dx)) < 1e-7:
            break
    H = jacobian_ext(x, T_meas, v_ms, accel_ms2, grade_rad)
    return x, np.linalg.inv(H.T @ R_inv @ H + P0_inv)


def reportable(x: np.ndarray, P: np.ndarray, min_variance_reduction: float = 1.5):
    """Per-state (value, sigma, variance_reduction); value is None when the data
    did not move the prior, so the prior mean is never shown as an estimate."""
    _, P0 = prior()
    out = []
    for i in range(N_STATE):
        vr = float(np.sqrt(P0[i, i] / P[i, i]))
        out.append((float(x[i]) if vr >= min_variance_reduction else None,
                    float(np.sqrt(P[i, i])), vr))
    return out
