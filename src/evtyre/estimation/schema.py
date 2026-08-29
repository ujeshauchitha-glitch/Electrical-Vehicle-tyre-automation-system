"""Phase 3 -> Phase 4 contract: TyreStateEstimate.

Defined in src/evtyre/estimation/ (NOT src/evtyre/schema/) following the
Phase 2 precedent where Feature lives in features/contract.py. The
src/evtyre/schema/ directory is frozen and guarded against additions.

This is the object that Phase 4 consumes. Nothing downstream of the
estimator should ever see a raw numpy array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .estimator import Observability, StateEstimate


@dataclass(frozen=True)
class TyreStateEstimate:
    """Per-corner tyre state estimate with provenance and observability.

    This is the Phase 3 -> Phase 4 contract. Phase 4 should consume
    this object, never raw numpy arrays.
    """

    # --- State estimates ---
    states: tuple[StateEstimate, ...]
    """Per-state estimates with observability classification."""

    covariance_diag: tuple[float, ...]
    """Diagonal of posterior covariance (one std dev per state)."""

    # --- Metadata ---
    timestamp_s: float
    """Timestamp of the input telemetry frame."""

    odometer_km: float | None
    """Odometer reading at estimation time, if available."""

    source: str
    """Source label propagated from input frames: 'real', 'replay', or 'simulated'."""

    model_version: str
    """Version of the estimation model that produced this result."""

    config_fingerprint: str
    """Hash of the physics and noise config used, for reproducibility tracking."""

    # --- Diagnostics ---
    n_measurements_available: int
    """How many measurements were available for this estimate."""

    n_states_observed: int
    """Count of states classified as OBSERVED."""

    mean_variance_reduction: float
    """Mean variance reduction across all states."""

    converged: bool
    """True if the Gauss-Newton iteration converged."""

    singular_matrix: bool
    """True if the update matrix was singular (some states may be underdetermined)."""

    iteration_count: int
    """Number of Gauss-Newton iterations performed."""

    @property
    def tread_estimate(self) -> Mapping[str, StateEstimate]:
        """Per-corner tread estimates, keyed by corner name."""
        return {
            s.name.replace("tread_", ""): s
            for s in self.states
            if s.name.startswith("tread_")
        }

    @property
    def pressure_estimate(self) -> Mapping[str, StateEstimate]:
        """Per-corner pressure estimates, keyed by corner name."""
        return {
            s.name.replace("press_", ""): s
            for s in self.states
            if s.name.startswith("press_")
        }

    @property
    def toe_estimate(self) -> StateEstimate | None:
        """Toe estimate (magnitude-only)."""
        for s in self.states:
            if s.name == "toe^2":
                return s
        return None

    @property
    def all_unobservable(self) -> bool:
        """True if every state is UNOBSERVABLE (e.g. all features missing)."""
        return all(
            s.observability == Observability.UNOBSERVABLE for s in self.states
        )
