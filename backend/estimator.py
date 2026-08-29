"""
Estimator module for EV tyre state estimation.

Sensor model, state representation, and iterated extended Kalman filter.
Uses only standard-vehicle CAN signals: TPMS pressure, resonance frequency,
wheel-speed ratios, and road-load estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .physics import (
    CORNERS,
    Vehicle,
    effective_rolling_radius,
    first_mode_frequency,
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
N_MEAS = 11
M_PRESS = slice(0, 4)
M_FREQ = slice(4, 8)
M_RATIO = slice(8, 10)
M_ROADLOAD = 10


# ---------------------------------------------------------------------------
# TyreState — ground truth or current estimate
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
    tpms_quantum   = 2.5
    tpms_sigma     = 3.0
    temp_sigma     = 1.5
    freq_sigma     = 0.15
    ratio_sigma    = 2.0e-4
    roadload_frac  = 0.040


def measure(
    state: TyreState,
    rng: np.random.Generator,
    noise: type = SensorNoise,
    v_ms: float = 22.0,
) -> Tuple[np.ndarray, float]:
    """Produce a noisy measurement vector from a true state."""
    z = np.zeros(N_MEAS)

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

    # Road-load equivalent coefficient
    F_total = road_load(state, v_ms)
    F_aero = 0.5 * Vehicle.rho_air * Vehicle.CdA * v_ms ** 2
    C_eq = (F_total - F_aero) / (Vehicle.mass * Vehicle.g)
    z[M_ROADLOAD] = C_eq * (1.0 + rng.normal(0, noise.roadload_frac))

    # Mean tyre temperature
    T_meas = float(np.mean([state.temp[c] for c in CORNERS]) + rng.normal(0, noise.temp_sigma))

    return z, T_meas


# ---------------------------------------------------------------------------
# Predict (forward model) and Jacobian
# ---------------------------------------------------------------------------

def predict(x: np.ndarray, T_meas: float = Vehicle.T_ref, v_ms: float = 22.0) -> np.ndarray:
    """Forward measurement model: state vector -> expected measurements."""
    tread = {c: x[i] for i, c in enumerate(CORNERS)}
    press = {c: x[4 + i] for i, c in enumerate(CORNERS)}
    z = np.zeros(N_MEAS)

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

    return z


def jacobian(x: np.ndarray, T_meas: float = Vehicle.T_ref, v_ms: float = 22.0, eps: float = 1e-5) -> np.ndarray:
    """Numerical Jacobian of the forward model."""
    H = np.zeros((N_MEAS, N_STATE))
    for j in range(N_STATE):
        dx = np.zeros(N_STATE)
        dx[j] = eps
        H[:, j] = (predict(x + dx, T_meas, v_ms) - predict(x - dx, T_meas, v_ms)) / (2 * eps)
    return H


# ---------------------------------------------------------------------------
# Measurement covariance
# ---------------------------------------------------------------------------

def measurement_covariance(noise: type = SensorNoise, z_ref: np.ndarray | None = None) -> np.ndarray:
    """Diagonal measurement noise covariance matrix."""
    R = np.zeros(N_MEAS)
    R[M_PRESS] = noise.tpms_sigma ** 2 + (noise.tpms_quantum ** 2) / 12.0
    R[M_FREQ] = noise.freq_sigma ** 2
    R[M_RATIO] = noise.ratio_sigma ** 2
    C_ref = z_ref[M_ROADLOAD] if z_ref is not None else Vehicle.C_rr0
    R[M_ROADLOAD] = (noise.roadload_frac * C_ref) ** 2
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
# Estimator
# ---------------------------------------------------------------------------

def estimate(
    z: np.ndarray,
    T_meas: float = Vehicle.T_ref,
    iters: int = 6,
    v_ms: float = 22.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterated extended Kalman filter: measurement vector -> (state, covariance)."""
    x0, P0 = prior()
    R_inv = np.linalg.inv(measurement_covariance(z_ref=z))
    P0_inv = np.linalg.inv(P0)

    x = x0.copy()
    for _ in range(iters):
        H = jacobian(x, T_meas, v_ms)
        residual = z - predict(x, T_meas, v_ms)
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

    H = jacobian(x, T_meas, v_ms)
    P = np.linalg.inv(H.T @ R_inv @ H + P0_inv)
    return x, P
