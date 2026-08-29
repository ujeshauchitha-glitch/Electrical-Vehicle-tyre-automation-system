"""Phase 4 decision contract.

Phase 4 turns a Phase 3 TyreStateEstimate into decision-ready output. Every
such output is a Decision, never a bare float, for the same reason Phase 1 has
SensorReading, Phase 2 has Feature, and Phase 3 has StateEstimate: a consumer
must never be able to mistake "we could not determine this" for a number.

THE PHASE 4 RULE
----------------
A decision may only be OK if every state it consumes is OBSERVED.

This is stricter than it may look, and deliberately so. A WEAK state carries
some information but not enough to state a level; a decision built on one would
present the prior as a conclusion. The wet torque ceiling is the motivating
case: legacy computes it from ``tread_est - 2*sigma``, and with tread currently
WEAK that lower bound evaluates to ~0.7 mm — below the legal limit — purely
from the width of the prior, with no tread measurement in it at all. Emitting a
safety-relevant traction limit from that number would be the same fabrication
class this project has already had to fix twice, except acted upon by a
vehicle.

So Phase 4 refuses. `require_observed()` is the single gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from ..estimation.estimator import Observability
from ..estimation.schema import TyreStateEstimate


class DecisionStatus(str, Enum):
    """Whether a decision-layer output can be acted on."""

    OK = "ok"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Decision:
    """One decision-ready output.

    Invariants enforced by ``__post_init__``:

    * OK           => ``value is not None`` and ``unavailable_reason is None``
    * UNAVAILABLE  => ``value is None`` and a non-empty ``unavailable_reason``

    ``basis`` names the states consumed, so a reviewer can trace any number
    back to what it was computed from. ``caveats`` carries non-blocking
    warnings — things true of an OK value that a consumer still must not
    forget (e.g. that a constant behind it is unvalidated).
    """

    name: str
    value: float | None
    unit: str
    status: DecisionStatus
    unavailable_reason: str | None
    basis: tuple[str, ...]
    caveats: tuple[str, ...] = ()
    corner: str | None = None

    def __post_init__(self) -> None:
        if self.status is DecisionStatus.OK:
            if self.value is None:
                raise ValueError(f"Decision {self.name!r}: OK but value is None")
            if self.unavailable_reason is not None:
                raise ValueError(
                    f"Decision {self.name!r}: OK must not carry an "
                    f"unavailable_reason"
                )
        else:
            if self.value is not None:
                raise ValueError(
                    f"Decision {self.name!r}: UNAVAILABLE but value is "
                    f"{self.value!r} (must be None)"
                )
            if not (self.unavailable_reason and self.unavailable_reason.strip()):
                raise ValueError(
                    f"Decision {self.name!r}: UNAVAILABLE requires a reason"
                )

    @property
    def is_actionable(self) -> bool:
        return self.status is DecisionStatus.OK

    @staticmethod
    def unavailable(
        name: str,
        unit: str,
        reason: str,
        basis: Iterable[str] = (),
        corner: str | None = None,
    ) -> "Decision":
        return Decision(
            name=name,
            value=None,
            unit=unit,
            status=DecisionStatus.UNAVAILABLE,
            unavailable_reason=reason,
            basis=tuple(basis),
            corner=corner,
        )

    @staticmethod
    def ok(
        name: str,
        value: float,
        unit: str,
        basis: Iterable[str],
        caveats: Iterable[str] = (),
        corner: str | None = None,
    ) -> "Decision":
        return Decision(
            name=name,
            value=value,
            unit=unit,
            status=DecisionStatus.OK,
            unavailable_reason=None,
            basis=tuple(basis),
            caveats=tuple(caveats),
            corner=corner,
        )


def require_observed(
    estimate: TyreStateEstimate, state_names: Iterable[str]
) -> str | None:
    """Gate for every Phase 4 decision.

    Returns None when every named state is OBSERVED, otherwise a reason string
    naming the offending states and their actual observability — so the refusal
    explains itself rather than just declining.
    """
    by_name = {s.name: s for s in estimate.states}
    problems: list[str] = []

    for name in state_names:
        state = by_name.get(name)
        if state is None:
            problems.append(f"{name} is not present in the estimate")
        elif state.observability is not Observability.OBSERVED:
            detail = state.reason or "no reason recorded"
            problems.append(f"{name} is {state.observability.value} ({detail})")

    if not problems:
        return None
    return "; ".join(problems)
