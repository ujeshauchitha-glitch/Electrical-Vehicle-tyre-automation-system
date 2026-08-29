"""Combine several recent Phase 3 estimates into one, to reduce sample noise.

This is the "reconciling multiple recent snapshots" part of CLAUDE.md section 3.
It is explicitly NOT a time trend: it treats the snapshots as repeated looks at
a state assumed constant over the window. Estimating how the state CHANGES over
time is Phase 5, and conflating the two is the specific error CLAUDE.md warns
about — averaging measurement noise is not the same as tracking wear.

Two rules make this safe:

1. Only OBSERVED states are fused. A WEAK or UNOBSERVABLE state is the prior
   showing through; averaging several copies of the same prior would shrink its
   variance and manufacture confidence from nothing. Repeated priors are not
   independent evidence.
2. Snapshots must share a `source`. Fusing a "real" estimate with a "simulated"
   one would produce a result that is neither, and CLAUDE.md section 8 rule 4
   requires that anything touched by synthetic data stay marked as such.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..estimation.estimator import Observability, StateEstimate
from ..estimation.schema import TyreStateEstimate


class SnapshotFusionError(ValueError):
    """Raised when a set of snapshots cannot legitimately be combined."""


def _fuse_one_state(
    name: str, states: Sequence[StateEstimate]
) -> StateEstimate:
    """Inverse-variance combination over the OBSERVED copies of one state.

    If no copy is OBSERVED, the state is returned as the first snapshot's
    version unchanged — deliberately not averaged, so a non-observed state can
    never gain confidence by being sampled repeatedly.
    """
    observed = [
        s for s in states
        if s.observability is Observability.OBSERVED and s.sigma > 0
    ]
    if not observed:
        base = states[0]
        n_weak = len(states)
        return StateEstimate(
            name=base.name,
            value=base.value,
            sigma=base.sigma,
            observability=base.observability,
            prior_value=base.prior_value,
            prior_sigma=base.prior_sigma,
            variance_reduction=base.variance_reduction,
            reason=(
                f"{base.reason or 'not observed'} "
                f"[not fused: none of {n_weak} snapshots observed this state; "
                f"averaging repeated priors would manufacture confidence]"
            ),
            magnitude_only=base.magnitude_only,
            jacobian_column_norm=base.jacobian_column_norm,
        )

    weights = [1.0 / (s.sigma ** 2) for s in observed]
    total_w = sum(weights)
    value = sum(w * s.value for w, s in zip(weights, observed)) / total_w
    sigma = math.sqrt(1.0 / total_w)

    base = observed[0]
    return StateEstimate(
        name=name,
        value=value,
        sigma=sigma,
        observability=Observability.OBSERVED,
        prior_value=base.prior_value,
        prior_sigma=base.prior_sigma,
        variance_reduction=(
            1.0 - (sigma ** 2) / (base.prior_sigma ** 2)
            if base.prior_sigma > 0 else 0.0
        ),
        reason=None,
        magnitude_only=base.magnitude_only,
        jacobian_column_norm=base.jacobian_column_norm,
    )


def fuse_snapshots(snapshots: Sequence[TyreStateEstimate]) -> TyreStateEstimate:
    """Combine recent snapshots of a state assumed constant over the window."""
    if not snapshots:
        raise SnapshotFusionError("Cannot fuse an empty set of snapshots")
    if len(snapshots) == 1:
        return snapshots[0]

    sources = {s.source for s in snapshots}
    if len(sources) > 1:
        raise SnapshotFusionError(
            f"Refusing to fuse snapshots from mixed sources {sorted(sources)}: "
            f"the result would be neither, and anything touched by simulated "
            f"data must stay marked as simulated (CLAUDE.md section 8 rule 4)"
        )

    fingerprints = {s.config_fingerprint for s in snapshots}
    models = {s.model_version for s in snapshots}
    if len(fingerprints) > 1 or len(models) > 1:
        raise SnapshotFusionError(
            f"Refusing to fuse snapshots produced by different configurations "
            f"(model versions {sorted(models)}, fingerprints "
            f"{sorted(fingerprints)}): the states would not be comparable"
        )

    ordered = sorted(snapshots, key=lambda s: s.timestamp_s)
    latest = ordered[-1]

    names = [s.name for s in latest.states]
    per_name: dict[str, list[StateEstimate]] = {n: [] for n in names}
    for snap in ordered:
        for st in snap.states:
            if st.name in per_name:
                per_name[st.name].append(st)

    fused_states = tuple(_fuse_one_state(n, per_name[n]) for n in names)
    n_observed = sum(
        1 for s in fused_states if s.observability is Observability.OBSERVED
    )
    observed_vr = [
        s.variance_reduction for s in fused_states
        if s.observability is Observability.OBSERVED
    ]

    return TyreStateEstimate(
        states=fused_states,
        covariance_diag=tuple(s.sigma ** 2 for s in fused_states),
        # The fused estimate is stamped at the NEWEST snapshot: it describes
        # the state as of now, using the window as supporting evidence.
        timestamp_s=latest.timestamp_s,
        odometer_km=latest.odometer_km,
        source=latest.source,
        model_version=latest.model_version,
        config_fingerprint=latest.config_fingerprint,
        n_measurements_available=sum(s.n_measurements_available for s in ordered),
        n_states_observed=n_observed,
        mean_variance_reduction=(
            sum(observed_vr) / len(observed_vr) if observed_vr else 0.0
        ),
        converged=all(s.converged for s in ordered),
        singular_matrix=any(s.singular_matrix for s in ordered),
        iteration_count=max(s.iteration_count for s in ordered),
    )
