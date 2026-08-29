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
from dataclasses import dataclass, field
from typing import Sequence

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..features.contract import Feature, FeatureStatus
from enum import Enum

import hashlib
from ..schema.common import CORNERS
from .state import MEAS, STATE, MeasurementLayout, StateLayout
from .physics import PhysicsConfig, corner_weight, effective_rolling_radius, rolling_resistance_coeff, toe_drag_from_sq


# ===========================================================================
# Observability classification
# ===========================================================================

class Observability(Enum):
    """Per-state observability classification, determined mechanically."""
    OBSERVED = "observed"       # posterior moved, variance shrank materially
    WEAK = "weak"               # some information, below threshold
    UNOBSERVABLE = "unobservable"  # zero Jacobian sensitivity



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


# Default physics constants (UNVALIDATED - legacy reference values)
MODEL_VERSION = "3.0.0"


DEFAULT_PHYSICS = PhysicsConfig(
    k_z0=210_000.0,
    cornering_stiffness=55_000.0,
    tread_rr_span=0.20,
    c_rr0=0.0090,
    p_exponent=0.45,
    t_coeff=0.0015,
    deflection_factor=1.0 / 3.0,  # UNVALIDATED — legacy value
)


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

    # Road load coefficient — DELIBERATELY NOT ADMITTED AS A MEASUREMENT.
    #
    # The Phase 2 road_load_coefficient is mean(C_rr) recomputed from TPMS
    # pressure and temperature through the same C_rr formula _predict() uses.
    # Measured on a live frame, it does not respond to motor torque, to
    # longitudinal acceleration, or to vehicle speed at all — only to pressure.
    # It is therefore not an independent observation of road load; it is the
    # pressure information restated.
    #
    # Admitting it would (a) double-count the pressure already in z[press_*],
    # shrinking pressure variance more than the data justifies, and (b) let the
    # estimator close the gap between the measured coefficient (which omits the
    # tread term) and the predicted one (which includes it) by moving TREAD —
    # producing a confident tread estimate built from no tread information.
    #
    # Re-enable ONLY when the coefficient is derived from an actual force
    # measurement (motor torque minus aero and inertia), which additionally
    # requires the missing road-grade channel (interface gap G6) to be
    # separable from it.
    _ = c_rr_ref  # retained for the future torque-derived channel

    return z, R_diag, available


# ===========================================================================
# Predict and Jacobian (simplified from legacy)
# ===========================================================================

def _predict(
    x: np.ndarray,
    tyre_config: TyreConfig,
    physics: PhysicsConfig,
    t_meas_c: float,
    vehicle_config: VehicleConfig,
) -> np.ndarray:
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

    # Axle speed ratios from effective rolling radius
    # The feature extractor emits omega_L / omega_R.
    # From pure rolling (v = omega*r for both wheels):
    #   omega_L / omega_R = r_R / r_L
    # So the predicted ratio is r_right / r_left = rb / ra.
    corners_left = ["FL", "RL"]
    corners_right = ["FR", "RR"]
    meas_indices = [MEAS.ratio_front, MEAS.ratio_rear]

    for left, right, idx in zip(corners_left, corners_right, meas_indices):
        tread_l = x[STATE.tread_fl] if left == "FL" else x[STATE.tread_rl]
        tread_r = x[STATE.tread_fr] if right == "FR" else x[STATE.tread_rr]
        press_l = x[STATE.press_fl] if left == "FL" else x[STATE.press_rl]
        press_r = x[STATE.press_fr] if right == "FR" else x[STATE.press_rr]

        fz_l = corner_weight(vehicle_config, left)
        fz_r = corner_weight(vehicle_config, right)

        ra = effective_rolling_radius(
            tread_l, press_l, fz_l, tyre_config.wheel_belt_radius_m,
            physics.k_z0, tyre_config.placard_pressure_kpa,
            physics.deflection_factor,
        )
        rb = effective_rolling_radius(
            tread_r, press_r, fz_r, tyre_config.wheel_belt_radius_m,
            physics.k_z0, tyre_config.placard_pressure_kpa,
            physics.deflection_factor,
        )
        z_pred[idx] = rb / max(ra, 1e-6)  # r_R / r_L = omega_L / omega_R

    # Road load prediction.
    #
    # NO TOE TERM. Legacy predicts C_rr + F_toe/(m*g) because its MEASURED
    # counterpart, C_eq = (F_total - F_aero)/(m*g), is derived from an actual
    # force and therefore contains toe drag. The Phase 2 road_load_coefficient
    # does not: it is mean(C_rr) recomputed from pressure, with no toe in it.
    #
    # Predicting a toe term against a measurement that has none makes the
    # residual converge to -F_toe/(m*g), so the estimator drives toe^2 -> 0,
    # shrinks its variance, and labels toe OBSERVED on zero toe information.
    # Do not reintroduce the toe term until the measured coefficient is
    # torque-derived. See features_to_measurement() for why the channel is
    # currently not admitted at all.
    c_rr_vals = [
        rolling_resistance_coeff(x[STATE.tread_fl], x[STATE.press_fl], t_meas_c, tyre_config, physics),
        rolling_resistance_coeff(x[STATE.tread_fr], x[STATE.press_fr], t_meas_c, tyre_config, physics),
        rolling_resistance_coeff(x[STATE.tread_rl], x[STATE.press_rl], t_meas_c, tyre_config, physics),
        rolling_resistance_coeff(x[STATE.tread_rr], x[STATE.press_rr], t_meas_c, tyre_config, physics),
    ]
    z_pred[MEAS.roadload] = float(np.mean(c_rr_vals))

    return z_pred


def _jacobian(
    x: np.ndarray,
    tyre_config: TyreConfig,
    physics: PhysicsConfig,
    t_meas_c: float,
    vehicle_config: VehicleConfig,
    eps: float = 1e-5,
) -> np.ndarray:
    """Numerical Jacobian of the predict function.

    H[i, j] = dz_i / dx_j ~ (predict(x+dx_j) - predict(x-dx_j)) / (2*eps)
    """
    H = np.zeros((MEAS.N, STATE.N))
    for j in range(STATE.N):
        dx = np.zeros(STATE.N)
        dx[j] = eps
        H[:, j] = (_predict(x + dx, tyre_config, physics, t_meas_c, vehicle_config) - _predict(x - dx, tyre_config, physics, t_meas_c, vehicle_config)) / (2 * eps)
    return H


# ===========================================================================
# Estimator
# ===========================================================================

@dataclass(frozen=True)
class StateEstimate:
    """Per-state estimate with observability classification.

    Invariants enforced by ``__post_init__``, following the same pattern as
    Phase 1's SensorReading and Phase 2's Feature:

    * not OBSERVED  =>  ``reason`` is a non-empty string.
    * OBSERVED      =>  ``reason`` is None.

    Without this, a WEAK or UNOBSERVABLE state could be constructed with no
    explanation, and a consumer would have no way to report why a number is
    not trustworthy — which is the entire purpose of this contract.
    """
    name: str
    value: float
    sigma: float                # posterior std dev
    observability: Observability
    prior_value: float
    prior_sigma: float
    variance_reduction: float   # 1 - (posterior_var / prior_var)
    reason: str | None          # required when not OBSERVED
    magnitude_only: bool = False  # True for toe (sign unrecoverable)
    jacobian_column_norm: float = 0.0  # 0 => no measurement depends on this state

    def __post_init__(self) -> None:
        if self.observability is Observability.OBSERVED:
            if self.reason is not None:
                raise ValueError(
                    f"StateEstimate {self.name!r}: OBSERVED must not carry a "
                    f"reason, got {self.reason!r}"
                )
        else:
            if not (self.reason and self.reason.strip()):
                raise ValueError(
                    f"StateEstimate {self.name!r}: observability is "
                    f"{self.observability.value} but no reason was given"
                )


@dataclass
class EstimatorResult:
    """Result of one estimation step."""

    states: tuple[StateEstimate, ...]  # per-state estimates
    covariance: np.ndarray              # posterior covariance matrix
    n_available: int                    # how many measurements were available
    n_total: int                        # total measurement dimension
    n_states_observed: int              # count of OBSERVED states
    mean_variance_reduction: float      # mean variance reduction over OBSERVED states
    singular_matrix: bool               # True if the update matrix was singular
    iteration_count: int                # number of Gauss-Newton iterations run
    converged: bool                     # True if max|dx| < tol before max iterations
    final_max_dx: float                 # max|dx| at last iteration


# ===========================================================================
# Observability classification
# ===========================================================================

# Threshold: Jacobian column norm below this is considered zero
_JACOBIAN_NORM_EPS = 1e-10

# Threshold: variance reduction below this is WEAK, above is OBSERVED
_VR_OBSERVED_THRESHOLD = 0.01  # 1% variance reduction = meaningful information

# Public aliases. These thresholds decide whether a number is presented to a
# user as an estimate, so tests and callers must be able to reference them by
# name rather than duplicating the literal.
JACOBIAN_ZERO_TOL = _JACOBIAN_NORM_EPS
WEAK_VARIANCE_REDUCTION = _VR_OBSERVED_THRESHOLD


def _classify_observability(
    H: np.ndarray,
    P0_diag: np.ndarray,
    P_diag: np.ndarray,
    state_names: list[str],
    x0: np.ndarray,
    x: np.ndarray,
    P0_full: np.ndarray | None = None,
    P_full: np.ndarray | None = None,
) -> tuple[StateEstimate, ...]:
    """Classify per-state observability from Jacobian norms and variance reduction.

    Parameters
    ----------
    H : (N_MEAS, N_STATE) Jacobian matrix at the posterior
    P0_diag : prior diagonal variances
    P_diag : posterior diagonal variances
    state_names : human-readable names
    x0 : prior state vector
    x : posterior state vector

    Returns
    -------
    tuple of StateEstimate, one per state variable
    """
    n_states = len(state_names)
    states: list[StateEstimate] = []

    # --- Common-mode (differential-only) detection -------------------------
    #
    # A per-state marginal variance can shrink a lot while the ABSOLUTE level
    # stays unknown. The axle-speed-ratio channel is exactly this case: it
    # predicts r_right/r_left, so it constrains the DIFFERENCE in tread across
    # an axle but says nothing about both corners moving together.
    #
    # Reporting such a state as OBSERVED would claim an absolute tread number
    # the data does not contain. So for each axle pair we measure the variance
    # reduction along the common-mode direction u = (e_i + e_j)/sqrt(2). If the
    # common mode is essentially unreduced, both members are demoted to WEAK.
    _differential_only: dict[int, float] = {}
    if P0_full is not None and P_full is not None:
        _name_to_idx = {n: i for i, n in enumerate(state_names)}
        for a, b in (("tread_FL", "tread_FR"), ("tread_RL", "tread_RR")):
            ia, ib = _name_to_idx.get(a), _name_to_idx.get(b)
            if ia is None or ib is None:
                continue
            u = np.zeros(n_states)
            u[ia] = u[ib] = 1.0 / np.sqrt(2.0)
            prior_cm = float(u @ P0_full @ u)
            post_cm = float(u @ P_full @ u)
            if prior_cm > 0:
                cm_reduction = max(0.0, min(1.0 - post_cm / prior_cm, 1.0))
                if cm_reduction < _VR_OBSERVED_THRESHOLD:
                    _differential_only[ia] = cm_reduction
                    _differential_only[ib] = cm_reduction

    for j in range(n_states):
        col_norm = float(np.linalg.norm(H[:, j]))
        prior_var = float(P0_diag[j])
        post_var = float(P_diag[j])

        # Variance reduction: how much did our uncertainty shrink?
        if prior_var > 0:
            vr = 1.0 - (post_var / prior_var)
            vr = max(0.0, min(vr, 1.0))  # clamp to [0, 1]
        else:
            vr = 0.0

        # Classify
        if col_norm < _JACOBIAN_NORM_EPS:
            obs = Observability.UNOBSERVABLE
            reason = f"Zero Jacobian sensitivity ({col_norm:.2e}) — posterior equals prior"
        elif vr < _VR_OBSERVED_THRESHOLD:
            obs = Observability.WEAK
            reason = f"Low information gain (variance reduction {vr:.4f})"
        elif j in _differential_only:
            # Marginal variance shrank, but only the within-axle DIFFERENCE is
            # constrained. The absolute level is not recoverable from this
            # channel set, so this must not be presented as an estimate.
            obs = Observability.WEAK
            reason = (
                f"Only the within-axle difference is constrained "
                f"(marginal variance reduction {vr:.4f}, but common-mode "
                f"reduction only {_differential_only[j]:.4f}); the absolute "
                f"level is not recoverable without an independent channel"
            )
        else:
            obs = Observability.OBSERVED
            reason = None

        # Toe is magnitude-only (sign not recoverable from drag)
        is_toe = ("toe" in state_names[j].lower())

        states.append(StateEstimate(
            name=state_names[j],
            value=float(x[j]),
            sigma=float(np.sqrt(max(0.0, post_var))),
            observability=obs,
            prior_value=float(x0[j]),
            prior_sigma=float(np.sqrt(max(0.0, prior_var))),
            variance_reduction=vr,
            reason=reason,
            magnitude_only=is_toe,
            jacobian_column_norm=col_norm,
        ))

    return tuple(states)


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
        physics: PhysicsConfig = DEFAULT_PHYSICS,
        n_iterations: int = 6,
        convergence_tol: float = 1e-7,
    ) -> None:
        self.vehicle_config = vehicle_config
        self.tyre_config = tyre_config
        self.noise = noise
        self.physics = physics
        self.n_iterations = n_iterations
        self.convergence_tol = convergence_tol

    def _extract_temperature(self, features: Sequence[Feature]) -> float:
        """Extract average tyre temperature from features."""
        temps = []
        for corner in CORNERS:
            name = f"tyre_temperature_c_{corner}"
            for f in features:
                if f.name == name and f.status == FeatureStatus.OK and f.value is not None:
                    temps.append(f.value)
                    break
        if temps:
            return sum(temps) / len(temps)
        return self.tyre_config.cold_reference_temperature_c

    def estimate(
        self,
        features: Sequence[Feature],
    ) -> EstimatorResult:
        """Run one estimation step over a set of features.

        Handles unavailable channels gracefully: they are excluded from
        the update with reduced confidence reported.
        """
        x0, P0 = prior(self.tyre_config)

        # Extract exogenous temperature from features
        t_meas_c = self._extract_temperature(features)

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
        if len(z) != MEAS.N or len(R_diag) != MEAS.N:
            raise ValueError(f"Measurement vector has {len(z)} entries, expected {MEAS.N}")
        if not np.all(np.isfinite(z)) or not np.all(R_diag > 0.0):
            raise ValueError("Measurement vector contains non-finite entries or non-positive variance")
        # Every index is EITHER a real measurement (finite, small variance) OR
        # explicitly disabled (huge variance).  Nothing may be silently left at
        # zero with a usable variance - that was the legacy z[0:4] bug.
        for _i in range(MEAS.N):
            _enabled = _i in available
            if _enabled != (R_diag[_i] < 1e11):
                raise ValueError(
                    f"Measurement index {_i} is {'enabled' if _enabled else 'not enabled'} "
                    f"but its variance is {R_diag[_i]:.3g} - a measurement was left "
                    f"unpopulated with a usable variance (legacy z[0:4] bug class)"
                )

        # Gauss-Newton MAP update
        R_inv = np.diag(1.0 / R_diag)
        P0_inv = np.linalg.inv(P0)
        x = x0.copy()
        singular_matrix = False
        n_iters = 0
        max_abs_dx = float("inf")

        for n_iters in range(1, self.n_iterations + 1):
            H = _jacobian(x, self.tyre_config, self.physics, t_meas_c, self.vehicle_config)
            residual = z - _predict(x, self.tyre_config, self.physics, t_meas_c, self.vehicle_config)
            A = H.T @ R_inv @ H + P0_inv
            b = H.T @ R_inv @ residual + P0_inv @ (x0 - x)
            try:
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Singular matrix — likely due to missing measurements
                singular_matrix = True
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

            max_abs_dx = float(np.max(np.abs(dx)))
            if max_abs_dx < self.convergence_tol:
                break

        # Posterior covariance
        H_final = _jacobian(x, self.tyre_config, self.physics, t_meas_c, self.vehicle_config)
        try:
            P = np.linalg.inv(H_final.T @ R_inv @ H_final + P0_inv)
        except np.linalg.LinAlgError:
            P = P0.copy()  # Fall back to prior if singular

        # Classify observability
        state_names = (
            [f"tread_{c}" for c in ["FL", "FR", "RL", "RR"]]
            + [f"press_{c}" for c in ["FL", "FR", "RL", "RR"]]
            + ["toe^2", "camber"]
        )
        states = _classify_observability(
            H_final, np.diag(P0), np.diag(P), state_names, x0, x, P0, P,
        )

        n_observed = sum(1 for s in states if s.observability == Observability.OBSERVED)
        # Averaged over OBSERVED states only: it answers "for the quantities we
        # actually claim to know, how much did we learn?". Averaging in states
        # we have just labelled WEAK or UNOBSERVABLE mixes that question with a
        # different one and makes the figure uninterpretable in either
        # direction. 0.0 when nothing is observed.
        observed_vr = [
            s.variance_reduction
            for s in states
            if s.observability is Observability.OBSERVED
        ]
        mean_vr = float(np.mean(observed_vr)) if observed_vr else 0.0

        return EstimatorResult(
            states=states,
            covariance=P,
            n_available=len(available),
            n_total=MEAS.N,
            n_states_observed=n_observed,
            mean_variance_reduction=mean_vr,
            singular_matrix=singular_matrix,
            iteration_count=n_iters,
            converged=max_abs_dx < self.convergence_tol,
            final_max_dx=float(max_abs_dx),
        )

    def to_schema(
        self,
        result: EstimatorResult,
        features: Sequence[Feature],
    ):
        """Convert EstimatorResult to TyreStateEstimate schema."""
        from .schema import TyreStateEstimate

        source = "simulated"
        timestamp_s = 0.0
        odometer_km = None
        for f in features:
            if f.provenance:
                source = f.provenance
            if f.timestamp_s > 0:
                timestamp_s = f.timestamp_s
            break

        config_str = (
            f"k_z0={self.physics.k_z0}:"
            f"c_alpha={self.physics.cornering_stiffness}:"
            f"c_rr0={self.physics.c_rr0}:"
            f"p_exp={self.physics.p_exponent}:"
            f"t_coeff={self.physics.t_coeff}:"
            f"defl={self.physics.deflection_factor}:"
            f"noise={self.noise.tpms_sigma_kpa}"
        )
        fingerprint = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        return TyreStateEstimate(
            states=result.states,
            covariance_diag=tuple(float(result.covariance[i, i]) for i in range(len(result.states))),
            timestamp_s=timestamp_s,
            odometer_km=odometer_km,
            source=source,
            model_version=MODEL_VERSION,
            config_fingerprint=fingerprint,
            n_measurements_available=result.n_available,
            n_states_observed=result.n_states_observed,
            mean_variance_reduction=result.mean_variance_reduction,
            converged=result.converged,
            singular_matrix=result.singular_matrix,
            iteration_count=result.iteration_count,
        )
