"""Phase 5 enhancement — Non-linear wear models.

Extends Phase 5's linear trend detection with exponential and quadratic
wear models. These capture accelerating wear (e.g., as rubber thins and
heat dissipation decreases) that linear models miss.

Models implemented:
  1. Linear:     y = a + b*x           (baseline, already in trend.py)
  2. Quadratic:  y = a + b*x + c*x²    (accelerating/decelerating wear)
  3. Exponential: y = a * exp(b*x)      (wear rate proportional to current state)

Each model is fit using weighted least squares and compared via AICc
(corrected Akaike Information Criterion) to select the best fit.

CRITICAL: These models describe how measured state changes over distance.
They do NOT change the EKF or the physical tyre model — they only
improve forecasting accuracy by fitting better curves to observed trends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from ..estimation.estimator import Observability
from ..estimation.schema import TyreStateEstimate


# ===========================================================================
# Model types
# ===========================================================================

class WearModel(Enum):
    """Supported wear curve models."""
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"


@dataclass(frozen=True)
class NonLinearFit:
    """Result of fitting a non-linear wear model to one state."""
    name: str
    """State name (e.g., 'tread_FL')."""

    best_model: WearModel
    """Model selected by AICc."""

    linear_slope: float
    """Linear component (mm/km)."""

    quadratic_coeff: float
    """Quadratic coefficient (mm/km²)."""

    exponential_rate: float
    """Exponential growth rate (per km)."""

    r_squared_linear: float
    """R² for linear model."""

    r_squared_quadratic: float
    """R² for quadratic model."""

    r_squared_exponential: float
    """R² for exponential model."""

    aic_linear: float
    """AICc for linear model."""

    aic_quadratic: float
    """AICc for quadratic model."""

    aic_exponential: float
    """AICc for exponential model."""

    n_observations: int
    """Number of OBSERVED snapshots used."""

    residuals_std: float
    """Standard deviation of residuals for best model."""

    @property
    def accelerating(self) -> bool:
        """True if the best model shows accelerating wear.

        For wear (decreasing values), acceleration means the rate of
        decrease is increasing in magnitude.  This happens when the
        linear slope and quadratic coefficient share the same sign
        (both negative for typical tread wear), or when there is a
        pure positive quadratic term (slope ≈ 0, quad > 0).

        For exponential, acceleration means |rate| grows with distance.
        """
        if self.best_model == WearModel.QUADRATIC:
            # Same sign → curve bends in the same direction as the trend
            if self.linear_slope * self.quadratic_coeff > 0:
                return True
            # Pure upward curvature (slope ≈ 0, positive quad)
            if abs(self.linear_slope) < 1e-12 and self.quadratic_coeff > 1e-12:
                return True
            return False
        elif self.best_model == WearModel.EXPONENTIAL:
            return self.exponential_rate > 0
        return False

    def predict(self, odometer_km: float) -> float:
        """Predict state value at a given odometer distance."""
        if self.best_model == WearModel.LINEAR:
            return self.linear_slope * odometer_km
        elif self.best_model == WearModel.QUADRATIC:
            return self.linear_slope * odometer_km + self.quadratic_coeff * odometer_km ** 2
        elif self.best_model == WearModel.EXPONENTIAL:
            return math.exp(self.exponential_rate * odometer_km)
        return 0.0


@dataclass(frozen=True)
class NonLinearWearReport:
    """Complete non-linear wear analysis."""
    fits: dict[str, NonLinearFit]
    """Per-state fit results."""

    n_states_with_acceleration: int
    """Number of states showing accelerating wear."""

    overall_model: WearModel
    """Most commonly selected model across states."""

    mean_r_squared: float
    """Mean R² of best-fit models."""

    n_snapshots: int
    """Number of snapshots analyzed."""


# ===========================================================================
# Model fitting
# ===========================================================================

def _fit_linear(
    x: np.ndarray, y: np.ndarray, w: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit y = a + b*x. Returns (a, b, r_squared, aicc)."""
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    try:
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    except np.linalg.LinAlgError:
        return float(y.mean()), 0.0, 0.0, float('inf')

    y_pred = X @ beta
    residuals = y - y_pred
    ss_res = float(np.sum(w * residuals ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = len(y)
    k = 2  # parameters: a, b
    aicc = _aicc(ss_res / max(n - k, 1), n, k)

    return float(beta[0]), float(beta[1]), r2, aicc


def _fit_quadratic(
    x: np.ndarray, y: np.ndarray, w: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Fit y = a + b*x + c*x². Returns (a, b, c, r_squared, aicc)."""
    X = np.column_stack([np.ones_like(x), x, x ** 2])
    W = np.diag(w)
    try:
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    except np.linalg.LinAlgError:
        return float(y.mean()), 0.0, 0.0, 0.0, float('inf')

    y_pred = X @ beta
    residuals = y - y_pred
    ss_res = float(np.sum(w * residuals ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = len(y)
    k = 3  # parameters: a, b, c
    aicc = _aicc(ss_res / max(n - k, 1), n, k)

    return float(beta[0]), float(beta[1]), float(beta[2]), r2, aicc


def _fit_exponential(
    x: np.ndarray, y: np.ndarray, w: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit y = a * exp(b*x) via log-linearization.

    Returns (a, b, r_squared, aicc).
    Only valid if all y > 0.
    """
    if np.any(y <= 0):
        return 0.0, 0.0, 0.0, float('inf')

    log_y = np.log(y)
    # Weight: w_log = w * y² (delta method for log transform)
    w_log = w * y ** 2

    X = np.column_stack([np.ones_like(x), x])
    W_log = np.diag(w_log)
    try:
        beta = np.linalg.solve(X.T @ W_log @ X, X.T @ W_log @ log_y)
    except np.linalg.LinAlgError:
        return float(np.exp(log_y.mean())), 0.0, 0.0, float('inf')

    log_y_pred = X @ beta
    residuals = log_y - log_y_pred
    ss_res = float(np.sum(w_log * residuals ** 2))
    ss_tot = float(np.sum(w_log * (log_y - np.average(log_y, weights=w_log)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = len(y)
    k = 2
    aicc = _aicc(ss_res / max(n - k, 1), n, k)

    return float(np.exp(beta[0])), float(beta[1]), r2, aicc


def _aicc(mse: float, n: int, k: int) -> float:
    """Corrected Akaike Information Criterion.

    AICc = n * ln(MSE) + 2k + 2k(k+1)/(n-k-1)
    Lower is better.
    """
    if mse <= 0:
        return float('-inf')
    if n - k - 1 <= 0:
        return float('inf')
    return n * math.log(mse) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


# ===========================================================================
# Main fitting function
# ===========================================================================

def fit_nonlinear_wear(
    snapshots: Sequence[TyreStateEstimate],
    odometer_readings: Sequence[float],
    min_observations: int = 5,
) -> NonLinearWearReport:
    """Fit non-linear wear models to tyre state evolution.

    For each state, fits linear, quadratic, and exponential models
    and selects the best via AICc.

    Parameters
    ----------
    snapshots : sequence of TyreStateEstimate
        Must be from the same source and configuration.
    odometer_readings : sequence of float
        Odometer reading (km) for each snapshot.
    min_observations : int
        Minimum number of OBSERVED snapshots required.

    Returns
    -------
    NonLinearWearReport with per-state fit results
    """
    if len(snapshots) < min_observations:
        raise ValueError(
            f"Need at least {min_observations} snapshots for non-linear wear fitting, "
            f"got {len(snapshots)}"
        )

    # Sort by odometer
    ordered = sorted(zip(snapshots, odometer_readings), key=lambda x: x[1])
    snaps = [s for s, _ in ordered]
    odoms = [o for _, o in ordered]

    state_names = [s.name for s in snaps[0].states]
    fits = {}
    model_counts = {m: 0 for m in WearModel}
    r2_values = []

    for i, name in enumerate(state_names):
        # Collect OBSERVED values
        values = []
        sigmas = []
        odoms_obs = []

        for snap, odom in zip(snaps, odoms):
            if i < len(snap.states):
                st = snap.states[i]
                if st.observability == Observability.OBSERVED:
                    values.append(st.value)
                    sigmas.append(st.sigma)
                    odoms_obs.append(odom)

        if len(values) < min_observations:
            fits[name] = NonLinearFit(
                name=name,
                best_model=WearModel.LINEAR,
                linear_slope=0.0,
                quadratic_coeff=0.0,
                exponential_rate=0.0,
                r_squared_linear=0.0,
                r_squared_quadratic=0.0,
                r_squared_exponential=0.0,
                aic_linear=float('inf'),
                aic_quadratic=float('inf'),
                aic_exponential=float('inf'),
                n_observations=len(values),
                residuals_std=0.0,
            )
            continue

        y = np.array(values)
        sigmas_arr = np.array(sigmas)
        x_raw = np.array(odoms_obs)
        w = 1.0 / (sigmas_arr ** 2)

        # Normalize for numerical stability
        x_center = x_raw.mean()
        x_scale = max(x_raw.std(), 1.0)
        x_norm = (x_raw - x_center) / x_scale

        # Fit all three models
        a_lin, b_lin, r2_lin, aicc_lin = _fit_linear(x_norm, y, w)
        a_quad, b_quad, c_quad, r2_quad, aicc_quad = _fit_quadratic(x_norm, y, w)
        a_exp, b_exp, r2_exp, aicc_exp = _fit_exponential(x_norm, y, w)

        # Select best model by AICc
        aicc_map = {
            WearModel.LINEAR: aicc_lin,
            WearModel.QUADRATIC: aicc_quad,
            WearModel.EXPONENTIAL: aicc_exp,
        }
        best = min(aicc_map, key=aicc_map.get)

        # Convert slopes back to per-km
        linear_slope = b_lin / x_scale

        # Residuals std for best model
        if best == WearModel.LINEAR:
            y_pred = a_lin + b_lin * x_norm
            residuals_std = float(np.std(y - y_pred))
        elif best == WearModel.QUADRATIC:
            y_pred = a_quad + b_quad * x_norm + c_quad * x_norm ** 2
            residuals_std = float(np.std(y - y_pred))
        else:
            y_pred = a_exp * np.exp(b_exp * x_norm)
            residuals_std = float(np.std(y - y_pred))

        model_counts[best] += 1
        r2_map = {WearModel.LINEAR: r2_lin, WearModel.QUADRATIC: r2_quad, WearModel.EXPONENTIAL: r2_exp}
        r2_values.append(r2_map[best])

        fits[name] = NonLinearFit(
            name=name,
            best_model=best,
            linear_slope=linear_slope,
            quadratic_coeff=c_quad / (x_scale ** 2),
            exponential_rate=b_exp / x_scale,
            r_squared_linear=r2_lin,
            r_squared_quadratic=r2_quad,
            r_squared_exponential=r2_exp,
            aic_linear=aicc_lin,
            aic_quadratic=aicc_quad,
            aic_exponential=aicc_exp,
            n_observations=len(values),
            residuals_std=residuals_std,
        )

    # Overall statistics
    overall_model = max(model_counts, key=model_counts.get)
    n_accel = sum(1 for f in fits.values() if f.accelerating)
    mean_r2 = float(np.mean(r2_values)) if r2_values else 0.0

    return NonLinearWearReport(
        fits=fits,
        n_states_with_acceleration=n_accel,
        overall_model=overall_model,
        mean_r_squared=mean_r2,
        n_snapshots=len(snaps),
    )


def format_nonlinear_report(report: NonLinearWearReport) -> str:
    """Format a NonLinearWearReport as human-readable text."""
    lines = [
        "Non-Linear Wear Analysis",
        "=" * 55,
        f"Snapshots analyzed: {report.n_snapshots}",
        f"Overall model: {report.overall_model.value}",
        f"States with accelerating wear: {report.n_states_with_acceleration}",
        f"Mean R²: {report.mean_r_squared:.3f}",
        "",
        "Per-state results:",
        "-" * 55,
    ]

    for name, fit in report.fits.items():
        if fit.n_observations < 5:
            lines.append(f"  {name}: insufficient data ({fit.n_observations} obs)")
            continue

        accel = " [ACCELERATING]" if fit.accelerating else ""
        lines.append(
            f"  {name}: best={fit.best_model.value}, "
            f"R²(lin)={fit.r_squared_linear:.3f}, "
            f"R²(quad)={fit.r_squared_quadratic:.3f}, "
            f"R²(exp)={fit.r_squared_exponential:.3f}"
            f"{accel}"
        )

    return "\n".join(lines)
