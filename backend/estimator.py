"""
Estimator module for EV tyre state estimation.

Sensor model, state representation, and iterated extended Kalman filter.
Uses only standard-vehicle CAN signals: TPMS pressure, resonance frequency,
wheel-speed ratios, road-load estimate, and (Phase 4) motor torque.

Phase 4 adds an INDEPENDENT motor-torque-derived road-load channel that
breaks the circular dependency of the original road-load measurement:

  Original (circular):  pressure -> Crr model -> road load -> estimator
  Phase 4 (independent): motor torque -> drivetrain model -> road load -> estimator

The motor torque measurement comes from the motor controller CAN bus and is
physically independent of the tyre pressure model. The estimator's forward
model uses tyre state to PREDICT expected motor torque (for Jacobian computation),
but the actual measurement data is an independent observation from the drivetrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .physics import (
    CORNERS,
    Vehicle,
    DrivetrainConfig,
    effective_rolling_radius,
    first_mode_frequency,
    motor_torque_measurement,
    rolling_resistance_coeff,
    road_load,
    toe_drag_from_sq,
)


# ---------------------------------------------------------------------------
# State dimension and index slices
# ---------------------------------------------------------------------------

N_STATE = 10
IDX_TREAD = slice(0, 4)
IDX_PRESS = slice(4, 8)
IDX_TOESQ = 8
IDX_CAMBER = 9

STATE_NAMES: list[str] = (
    [f"tread_{c}" for c in CORNERS]
    + [f"press_{c}" for c in CORNERS]
    + ["toe^2", "camber"]
)

# Measurement dimension and index slices
N_MEAS = 12  # Phase 4: expanded from 11 to include motor torque channel
M_PRESS = slice(0, 4)
M_FREQ = slice(4, 8)
M_RATIO = slice(8, 10)
M_ROADLOAD = 10
M_MOTORTORQUE = 11  # Phase 4: independent motor torque measurement (N.m)


# ---------------------------------------------------------------------------
# TyreState -- ground truth or current estimate
# ---------------------------------------------------------------------------

class TyreState:
    """Per-vehicle tyre state container."""

    def __init__(
        self,
        tread: dict[str, float] | None = None,
        pressure: dict[str, float] | None = None,
        temp: dict[str, float] | None = None,
        toe: float = 0.0,
        camber: float = 0.0,
    ):
        self.tread    = dict(tread)    if tread    else {c: 6.0 for c in CORNERS}
        self.pressure = dict(pressure) if pressure else {c: Vehicle.p_placard for c in CORNERS}
        self.temp     = dict(temp)     if temp     else {c: 25.0 for c in CORNERS}
        self.toe      = toe
        self.camber   = camber

    @staticmethod
    def random(rng: np.random.Generator) -> TyreState:
        """Generate a physically plausible random tyre state."""
        base_front = rng.uniform(2.0, 7.8)
        base_rear  = max(1.7, base_front - rng.uniform(0.3, 2.0))
        tread = {
            "FL": base_front + rng.normal(0, 0.25),
            "FR": base_front + rng.normal(0, 0.25),
            "RL": base_rear  + rng.normal(0, 0.30),
            "RR": base_rear  + rng.normal(0, 0.30),
        }
        tread = {c: float(np.clip(v, 1.2, 8.0)) for c, v in tread.items()}
        pressure = {c: float(Vehicle.p_placard + rng.normal(0, 18.0)) for c in CORNERS}
        if rng.random() < 0.17:
            pressure[rng.choice(CORNERS)] -= rng.uniform(25, 65)
        soak = rng.uniform(0, 28)
        temp = {c: 25.0 + soak + rng.normal(0, 2.0) for c in CORNERS}
        toe = float(rng.normal(0, 0.20)) if rng.random() < 0.75 else float(rng.uniform(0.4, 1.1))
        camber = float(rng.normal(0, 0.6))
        return TyreState(tread, pressure, temp, toe, camber)


# ---------------------------------------------------------------------------
# Sensor noise model
# ---------------------------------------------------------------------------

class SensorNoise:
    """Quantises and corrupts ground-truth state into CAN-like measurements."""
    tpms_quantum      = 2.5
    tpms_sigma        = 3.0
    temp_sigma        = 1.5
    freq_sigma        = 0.15
    ratio_sigma       = 2.0e-4
    roadload_frac     = 0.040
    motortorque_sigma = 10.0  # Phase 4: motor torque noise (N.m)


def measure(
    state: TyreState,
    rng: np.random.Generator,
    noise: type = SensorNoise,
    v_ms: float = 22.0,
    accel_ms2: float = 0.0,
    grade_rad: float = 0.0,
    include_motor_torque: bool = False,
) -> Tuple[np.ndarray, float]:
    """Produce a noisy measurement vector from a true state.

    Parameters
    ----------
    include_motor_torque : bool
        If True, compute motor torque measurement from ground truth and add noise.
        If False, set motor torque channel to NaN (unavailable).
    """
    z = np.zeros(N_MEAS)  # 12 elements (Phase 4)

    # TPMS pressures (quantised + Gaussian noise)
    for i, c in enumerate(CORNERS):
        p_raw = np.round(state.pressure[c] / noise.tpms_quantum) * noise.tpms_quantum
        z[i] = p_raw + rng.normal(0, noise.tpms_sigma)

    # Resonance frequencies
    for i, c in enumerate(CORNERS):
        f_true = first_mode_frequency(state.tread[c], state.pressure[c])
        z[4 + i] = f_true + rng.normal(0, noise.freq_sigma)

    # Wheel-speed ratios (rear/front per axle)
    for k, (a, b) in enumerate((("FL", "FR"), ("RL", "RR"))):
        ra = effective_rolling_radius(state.tread[a], state.pressure[a], Vehicle.Fz(a))
        rb = effective_rolling_radius(state.tread[b], state.pressure[b], Vehicle.Fz(b))
        z[8 + k] = rb / ra + rng.normal(0, noise.ratio_sigma)

    # Road-load equivalent coefficient (original, kept for backward compatibility)
    F_total = road_load(state, v_ms)
    F_aero = 0.5 * Vehicle.rho_air * Vehicle.CdA * v_ms ** 2
    C_eq = (F_total - F_aero) / (Vehicle.mass * Vehicle.g)
    z[M_ROADLOAD] = C_eq * (1.0 + rng.normal(0, noise.roadload_frac))

    # Phase 4: Motor torque measurement (independent channel)
    if include_motor_torque:
        # Build state array for forward model (10 elements)
        state_arr = np.array([
            state.tread[c] for c in CORNERS
        ] + [
            state.pressure[c] for c in CORNERS
        ] + [
            state.toe ** 2,  # toe^2
            state.camber,
        ])
        T_true = motor_torque_measurement(state_arr, v_ms, accel_ms2, grade_rad)
        z[M_MOTORTORQUE] = T_true + rng.normal(0, noise.motortorque_sigma)
    else:
        z[M_MOTORTORQUE] = np.nan

    # Mean tyre temperature
    T_meas = float(np.mean([state.temp[c] for c in CORNERS]) + rng.normal(0, noise.temp_sigma))

    return z, T_meas


# ---------------------------------------------------------------------------
# Predict (forward model) and Jacobian
# ---------------------------------------------------------------------------

def predict(
    x: np.ndarray,
    T_meas: float = Vehicle.T_ref,
    v_ms: float = 22.0,
    accel_ms2: float = 0.0,
    grade_rad: float = 0.0,
) -> np.ndarray:
    """Forward measurement model: state vector -> expected measurements.

    Returns 12-element prediction vector:
    [0:4]  pressure predictions
    [4:8]  frequency predictions
    [8:10] ratio predictions
    [10]   road-load Crr-equivalent
    [11]   expected motor torque (N.m)
    """
    tread = {c: x[i] for i, c in enumerate(CORNERS)}
    press = {c: x[4 + i] for i, c in enumerate(CORNERS)}
    z = np.zeros(N_MEAS)  # 12 elements

    z[M_PRESS] = [press[c] for c in CORNERS]

    for i, c in enumerate(CORNERS):
        z[4 + i] = first_mode_frequency(tread[c], press[c])

    for k, (a, b) in enumerate((("FL", "FR"), ("RL", "RR"))):
        ra = effective_rolling_radius(tread[a], press[a], Vehicle.Fz(a))
        rb = effective_rolling_radius(tread[b], press[b], Vehicle.Fz(b))
        z[8 + k] = rb / ra

    C_rr = np.mean([rolling_resistance_coeff(tread[c], press[c], T_meas) for c in CORNERS])
    F_toe = toe_drag_from_sq(x[IDX_TOESQ])
    z[M_ROADLOAD] = C_rr + F_toe / (Vehicle.mass * Vehicle.g)

    # Phase 4: Expected motor torque from tyre state
    z[M_MOTORTORQUE] = motor_torque_measurement(x, v_ms, accel_ms2, grade_rad)

    return z


def jacobian(
    x: np.ndarray,
    T_meas: float = Vehicle.T_ref,
    v_ms: float = 22.0,
    accel_ms2: float = 0.0,
    grade_rad: float = 0.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """Numerical Jacobian of the forward model: d(measurements)/d(state)."""
    H = np.zeros((N_MEAS, N_STATE))  # (12, 10)
    for j in range(N_STATE):
        dx = np.zeros(N_STATE)
        dx[j] = eps
        H[:, j] = (
            predict(x + dx, T_meas, v_ms, accel_ms2, grade_rad)
            - predict(x - dx, T_meas, v_ms, accel_ms2, grade_rad)
        ) / (2 * eps)
    return H


# ---------------------------------------------------------------------------
# Measurement covariance
# ---------------------------------------------------------------------------

def measurement_covariance(
    noise: type = SensorNoise,
    z_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Diagonal measurement noise covariance matrix (12x12)."""
    R = np.zeros(N_MEAS)
    R[M_PRESS] = noise.tpms_sigma ** 2 + (noise.tpms_quantum ** 2) / 12.0
    R[M_FREQ] = noise.freq_sigma ** 2
    R[M_RATIO] = noise.ratio_sigma ** 2
    C_ref = z_ref[M_ROADLOAD] if z_ref is not None else Vehicle.C_rr0
    R[M_ROADLOAD] = (noise.roadload_frac * max(abs(C_ref), 1e-6)) ** 2
    R[M_MOTORTORQUE] = noise.motortorque_sigma ** 2  # Phase 4
    return np.diag(R)


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

def prior() -> Tuple[np.ndarray, np.ndarray]:
    """Initial state estimate and covariance."""
    x0 = np.zeros(N_STATE)
    x0[IDX_TREAD] = 5.0
    x0[IDX_PRESS] = Vehicle.p_placard
    x0[IDX_TOESQ] = 0.10
    x0[IDX_CAMBER] = 0.0

    P0 = np.diag([
        2.5 ** 2, 2.5 ** 2, 2.5 ** 2, 2.5 ** 2,
        40.0 ** 2, 40.0 ** 2, 40.0 ** 2, 40.0 ** 2,
        0.40 ** 2,
        1.0 ** 2,
    ])
    return x0, P0


# ---------------------------------------------------------------------------
# Observability analysis (Phase 4)
# ---------------------------------------------------------------------------

def observability_analysis(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    T_meas: float = Vehicle.T_ref,
    v_ms: float = 22.0,
    accel_ms2: float = 0.0,
    grade_rad: float = 0.0,
) -> dict:
    """Compute observability metrics for each state variable.

    Returns a dictionary with per-state variance reduction, toe-specific
    sensitivity metrics, Fisher information, and observability classification.
    """
    _, P0 = prior()
    s0 = np.sqrt(np.diag(P0))
    sigma = np.sqrt(np.diag(P))

    H = jacobian(x, T_meas, v_ms, accel_ms2, grade_rad)

    # Measurement covariance with NaN handling
    R = measurement_covariance(z_ref=z)
    nan_mask = np.isnan(z)
    nan_idx = np.where(nan_mask)[0]
    R[nan_idx, nan_idx] = 1e30  # set diagonal only, not entire rows
    R_inv = np.linalg.inv(R)

    # Fisher information matrix: F = H^T R^{-1} H
    F = H.T @ R_inv @ H

    # Per-state variance reduction
    variance_reduction = {}
    for i, name in enumerate(STATE_NAMES):
        if sigma[i] > 1e-15:
            variance_reduction[name] = float(s0[i] / sigma[i])
        else:
            variance_reduction[name] = float("inf")

    # Toe-specific analysis
    toe_idx = IDX_TOESQ
    toe_sensitivity_roadload = float(H[M_ROADLOAD, toe_idx])
    toe_sensitivity_motor_torque = float(H[M_MOTORTORQUE, toe_idx])
    toe_fisher = float(F[toe_idx, toe_idx])
    toe_var_prior = float(P0[toe_idx, toe_idx])
    toe_var_post = float(P[toe_idx, toe_idx])
    toe_vr = float(s0[toe_idx] / sigma[toe_idx]) if sigma[toe_idx] > 1e-15 else float("inf")

    # Observability classification
    if toe_fisher < 1e-10 or toe_vr < 1.05:
        toe_class = "UNOBSERVABLE"
    elif toe_vr < 1.5:
        toe_class = "WEAK"
    else:
        toe_class = "OBSERVED"

    n_available = int(np.sum(~nan_mask))

    return {
        "toe_sensitivity_roadload": toe_sensitivity_roadload,
        "toe_sensitivity_motor_torque": toe_sensitivity_motor_torque,
        "toe_variance_prior": toe_var_prior,
        "toe_variance_posterior": toe_var_post,
        "toe_variance_reduction": toe_vr,
        "toe_observability": toe_class,
        "toe_fisher_information": toe_fisher,
        "variance_reduction": variance_reduction,
        "n_measurements_available": n_available,
    }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

def estimate(
    z: np.ndarray,
    T_meas: float = Vehicle.T_ref,
    iters: int = 6,
    v_ms: float = 22.0,
    accel_ms2: float = 0.0,
    grade_rad: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterated extended Kalman filter: measurement vector -> (state, covariance).

    Handles NaN measurements by setting their R diagonal to a very large value,
    making those channels invisible to the estimator. This allows the same
    function to work with or without the motor torque channel.
    """
    x0, P0 = prior()

    # Build measurement covariance
    R = measurement_covariance(z_ref=z)

    # Handle NaN measurements: infinite variance = channel ignored
    nan_mask = np.isnan(z)
    nan_idx = np.where(nan_mask)[0]
    R[nan_idx, nan_idx] = 1e30  # set diagonal only, not entire rows

    R_inv = np.linalg.inv(R)
    P0_inv = np.linalg.inv(P0)

    x = x0.copy()
    for _ in range(iters):
        H = jacobian(x, T_meas, v_ms, accel_ms2, grade_rad)
        residual = z - predict(x, T_meas, v_ms, accel_ms2, grade_rad)

        # Zero out NaN residuals to avoid NaN propagation (R_inv weights are ~0)
        residual[nan_mask] = 0.0

        A = H.T @ R_inv @ H + P0_inv
        b = H.T @ R_inv @ residual + P0_inv @ (x0 - x)
        dx = np.linalg.solve(A, b)
        x = x + dx

        # Physical bounds
        x[IDX_TREAD] = np.clip(x[IDX_TREAD], 0.5, 9.0)
        x[IDX_PRESS] = np.clip(x[IDX_PRESS], 100.0, 350.0)
        x[IDX_TOESQ] = max(0.0, x[IDX_TOESQ])

        if np.max(np.abs(dx)) < 1e-7:
            break

    H = jacobian(x, T_meas, v_ms, accel_ms2, grade_rad)
    P = np.linalg.inv(H.T @ R_inv @ H + P0_inv)
    return x, P
