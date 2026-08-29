"""Phase 3 tyre state estimator — skeleton.

Consumes Feature objects (Phase 2 output), not raw telemetry.  Ports the
STRUCTURE from legacy/ev_tyre_fusion.py — read as reference, not copied.

THREE DOCUMENTED BUGS FROM LEGACY — do not reintroduce:
1. Do NOT temperature-compensate pressure before the physics.
   Compensation is a reporting step.  Feed RUNNING pressure to stiffness.
2. Toe drag F = 2*C_alpha*toe^2.  Estimate toe^2 (enters observable
   linearly), take sqrt at end.  SIGN NOT RECOVERABLE — magnitude only.
3. State and measurement index slices are TEXTUALLY SEPARATE in state.py.

Known expected behaviour: camber has zero sensitivity in every available
observable — posterior equals prior exactly.  This is correct, not a bug.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..features.contract import Feature, FeatureStatus
from .state import MEAS, STATE, MeasurementLayout, StateLayout


# ===========================================================================
# Sensor noise model (UNVALIDATED — legacy reference values)
# ===========================================================================

@dataclass(frozen=True)
class SensorNoise:
    """Noise parameters for the measurement covariance.

    All values are UNVALIDATED guesses from legacy/ev_tyre_fusion.py.
    """

    tpms_sigma_kpa: float = 5.0
    """TPMS pressure noise (kPa, 1σ)."""

    freq_sigma_hz: float = 2.0
    """Resonance frequency noise (Hz, 1σ).  UNUSED until G1 resolves."""

    ratio_sigma: float = 0.005
    """Axle speed ratio noise (dimensionless, 1σ)."""

    roadload_frac: float = 0.05
    """Road load noise as fraction of reference value."""


DEFAULT_NOISE = SensorNoise()


# ===========================================================================
# Prior
# ===========================================================================

def prior(tyre_config: TyreConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (x0, P0) — initial state guess and covariance.

    x0 is a reasonable starting point given the tyre configuration.
    P0 reflects our uncertainty.
    """
    x0 = np.zeros(STATE.N)
    # Tread: start at midpoint between new and legal
    mid_tread = (tyre_config.tread_new_mm + tyre_config.tread_legal_mm) / 2.0
    x0[STATE.tread_fl] = mid_tread
    x0[STATE.tread_fr] = mid_tread
    x0[STATE.tread_rl] = mid_tread
    x0[STATE.tread_rr] = mid_tread
    # Pressure: start at placard
    x0[STATE.press_fl] = tyre_config.placard_pressure_kpa
    x0[STATE.press_fr] = tyre_config.placard_pressure_kpa
    x0[STATE.press_rl] = tyre_config.placard_pressure_kpa
    x0[STATE.press_rr] = tyre_config.placard_pressure_kpa
    # Toe^2: start at small positive value (not zero — avoids singularity)
    x0[STATE.toe_sq] = 0.10
    # Camber: start at zero (and will stay there — unobservable)
    x0[STATE.camber] = 0.0

    P0 = np.diag(np.array([
        2.5 ** 2, 2.5 ** 2, 2.5 ** 2, 2.5 ** 2,   # tread uncertainty
        40.0 ** 2, 40.0 ** 2, 40.0 ** 2, 40.0 ** 2,  # pressure uncertainty
        0.40 ** 2,                                     # toe^2 uncertainty
        1.0 ** 2,                                     # camber uncertainty
    ]))
    return x0, P0


# ===========================================================================
# Feature → measurement vector
# ===========================================================================

def features_to_measurement(
    features: Sequence[Feature],
    vehicle_config: VehicleConfig,
    tyre_config: TyreConfig,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Convert a tuple of Features into the measurement vector z and mask.

    Returns (z, R_diag, available_indices) where:
    - z: measurement vector (N_MEAS,)
    - R_diag: diagonal of measurement covariance (N_MEAS,)
    - available_indices: which z entries were actually populated

    Unavailable measurements are set to 0 in z and given a very large
    variance in R (effectively ignoring them in the update).
    """
    z = np.zeros(MEAS.N)
    R_diag = np.full(MEAS.N, 1e12)  # huge variance = ignored
    available: list[int] = []

    noise = DEFAULT_NOISE
    # Reference road load for noise scaling
    c_rr_ref = 0.0090  # UNVALIDATED — legacy C_rr0

    # Index features by name for quick lookup
    feat_map = {f.name: f for f in features}

    # Pressure measurements (per corner)
    press_corners = ["FL", "FR", "RL", "RR"]
    press_indices = [MEAS.press_fl, MEAS.press_fr, MEAS.press_rl, MEAS.press_rr]
    for corner, idx in zip(press_corners, press_indices):
        feat_name = f"running_pressure_pa_{corner}"
        # Try kPa variant too
        f = feat_map.get(feat_name)
        if f is None:
            # Try the kPa version
            f = feat_map.get(f"pressure_deviation_from_placard_{corner}")
        if f is not None and f.status == FeatureStatus.OK and f.value is not None:
            # Feature is ABSOLUTE Pa; state/predict use GAUGE kPa.
            z[idx] = (f.value - 101_325.0) / 1000.0
            R_diag[idx] = noise.tpms_sigma_kpa ** 2
            available.append(idx)
        # else: left at 0 with huge variance

    # Resonance frequency — BLOCKED by G1
    # All freq entries remain at 0 with huge variance
    # (the resonance extractor raises NotImplementedError)

    # Axle speed ratios
    for feat_name, idx in [
        ("axle_speed_ratio_front", MEAS.ratio_front),
        ("axle_speed_ratio_rear", MEAS.ratio_rear),
    ]:
        f = feat_map.get(feat_name)
        if f is not None and f.status == FeatureStatus.OK and f.value is not None:
            z[idx] = f.value
            R_diag[idx] = noise.ratio_sigma ** 2
            available.append(idx)

    # Road load coefficient
    f = feat_map.get("road_load_coefficient")
    if f is not None and f.status == FeatureStatus.OK and f.value is not None:
        z[MEAS.roadload] = f.value
        R_diag[MEAS.roadload] = (noise.roadload_frac * c_rr_ref) ** 2
        available.append(MEAS.roadload)

    return z, R_diag, available


# ===========================================================================
# Predict and Jacobian (simplified from legacy)
# ===========================================================================

def _predict(x: np.ndarray, tyre_config: TyreConfig) -> np.ndarray:
    """Predict the measurement vector from the state.

    Simplified from legacy/ev_tyre_fusion.py.  Only implements the
    channels that are currently available (pressure, speed ratio).
    Resonance is excluded — the G1 extractor raises.
    """
    z_pred = np.zeros(MEAS.N)

    # Pressure prediction: state pressure maps directly to measurement
    z_pred[MEAS.press_fl] = x[STATE.press_fl]
    z_pred[MEAS.press_fr] = x[STATE.press_fr]
    z_pred[MEAS.press_rl] = x[STATE.press_rl]
    z_pred[MEAS.press_rr] = x[STATE.press_rr]

    # Axle speed ratios: simplified model
    # ratio = omega_rear / omega_front ≈ 1.0 (same tyre size)
    # A more complete model would use effective rolling radius
    z_pred[MEAS.ratio_front] = 1.0
    z_pred[MEAS.ratio_rear] = 1.0

    # Road load: simplified C_rr from pressure
    # C_rr depends on pressure (UNVALIDATED functional form from legacy)
    _C_RR0 = 0.0090
    _P_EXP = 0.45
    p_refs = [x[STATE.press_fl], x[STATE.press_fr],
              x[STATE.press_rl], x[STATE.press_rr]]
    c_rr_vals = [_C_RR0 * (tyre_config.placard_pressure_kpa / max(p, 1.0)) ** _P_EXP
                 for p in p_refs]
    z_pred[MEAS.roadload] = np.mean(c_rr_vals)

    return z_pred


def _jacobian(x: np.ndarray, tyre_config: TyreConfig, eps: float = 1e-5) -> np.ndarray:
    """Numerical Jacobian of the predict function.

    H[i, j] = dz_i / dx_j ≈ (predict(x+dx_j) - predict(x-dx_j)) / (2*eps)
    """
    H = np.zeros((MEAS.N, STATE.N))
    for j in range(STATE.N):
        dx = np.zeros(STATE.N)
        dx[j] = eps
        H[:, j] = (_predict(x + dx, tyre_config) - _predict(x - dx, tyre_config)) / (2 * eps)
    return H


# ===========================================================================
# Estimator
# ===========================================================================

@dataclass
class EstimatorResult:
    """Result of one estimation step."""

    state: np.ndarray          # point estimate
    covariance: np.ndarray     # posterior covariance
    state_names: list[str]     # human-readable names for state entries
    n_available: int           # how many measurements were available
    n_total: int               # total measurement dimension
    confidence: float          # n_available / n_total
    toe_magnitude_deg: float   # sqrt(max(0, toe^2)), sign not recoverable
    camber_note: str           # explanatory note about camber unobservability


class TyreEstimator:
    """Phase 3 tyre state estimator.

    Consumes Feature objects from Phase 2, handles unavailable channels
    gracefully, and reports reduced confidence.
    """

    def __init__(
        self,
        vehicle_config: VehicleConfig,
        tyre_config: TyreConfig,
        noise: SensorNoise = DEFAULT_NOISE,
        n_iterations: int = 6,
        convergence_tol: float = 1e-7,
    ) -> None:
        self.vehicle_config = vehicle_config
        self.tyre_config = tyre_config
        self.noise = noise
        self.n_iterations = n_iterations
        self.convergence_tol = convergence_tol

    def estimate(
        self,
        features: Sequence[Feature],
    ) -> EstimatorResult:
        """Run one estimation step over a set of features.

        Handles unavailable channels gracefully: they are excluded from
        the update with reduced confidence reported.
        """
        x0, P0 = prior(self.tyre_config)

        # Convert features to measurement vector
        z, R_diag, available = features_to_measurement(
            features, self.vehicle_config, self.tyre_config,
        )

        # Validation: measurement vector must be fully populated before
        # the update.  "Fully populated" means every entry is either
        # filled with a real value OR explicitly marked with large
        # variance (which we do above for unavailable channels).
        # This assertion catches the legacy bug where z[0:4] was left
        # unfilled and produced a singular covariance matrix.
        assert len(z) == MEAS.N and len(R_diag) == MEAS.N, (
            f"Measurement vector has {len(z)} entries, expected {MEAS.N}"
        )
        assert np.all(np.isfinite(z)) and np.all(R_diag > 0.0), (
            "Measurement vector contains non-finite entries or non-positive variance"
        )
        # Every index is EITHER a real measurement (finite, small variance) OR
        # explicitly disabled (huge variance).  Nothing may be silently left at
        # zero with a usable variance - that was the legacy z[0:4] bug.
        for _i in range(MEAS.N):
            _enabled = _i in available
            assert _enabled == (R_diag[_i] < 1e11), (
                f"Measurement index {_i} is {'enabled' if _enabled else 'not enabled'} "
                f"but its variance is {R_diag[_i]:.3g} - a measurement was left "
                f"unpopulated with a usable variance (legacy z[0:4] bug class)"
            )

        # Gauss-Newton MAP update
        R_inv = np.diag(1.0 / R_diag)
        P0_inv = np.linalg.inv(P0)
        x = x0.copy()

        for _ in range(self.n_iterations):
            H = _jacobian(x, self.tyre_config)
            residual = z - _predict(x, self.tyre_config)
            A = H.T @ R_inv @ H + P0_inv
            b = H.T @ R_inv @ residual + P0_inv @ (x0 - x)
            try:
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Singular matrix — likely due to missing measurements
                break
            x = x + dx

            # Physical bounds
            x[STATE.tread_fl:STATE.tread_rr + 1] = np.clip(
                x[STATE.tread_fl:STATE.tread_rr + 1], 0.5, 9.0
            )
            x[STATE.press_fl:STATE.press_rr + 1] = np.clip(
                x[STATE.press_fl:STATE.press_rr + 1], 50.0, 500.0
            )
            x[STATE.toe_sq] = max(0.0, x[STATE.toe_sq])

            if np.max(np.abs(dx)) < self.convergence_tol:
                break

        # Posterior covariance
        H_final = _jacobian(x, self.tyre_config)
        try:
            P = np.linalg.inv(H_final.T @ R_inv @ H_final + P0_inv)
        except np.linalg.LinAlgError:
            P = P0.copy()  # Fall back to prior if singular

        # Toe: take sqrt of toe^2, sign not recoverable
        toe_magnitude = float(np.sqrt(max(0.0, x[STATE.toe_sq])))

        # Confidence
        confidence = len(available) / MEAS.N

        state_names = (
            [f"tread_{c}" for c in ["FL", "FR", "RL", "RR"]]
            + [f"press_{c}" for c in ["FL", "FR", "RL", "RR"]]
            + ["toe^2", "camber"]
        )

        return EstimatorResult(
            state=x,
            covariance=P,
            state_names=state_names,
            n_available=len(available),
            n_total=MEAS.N,
            confidence=confidence,
            toe_magnitude_deg=toe_magnitude,
            camber_note=(
                "Camber has zero sensitivity in every available observable. "
                "Its posterior equals its prior exactly. This is correct "
                "behaviour, not a bug — camber is unobservable from the "
                "current channel set."
            ),
        )
