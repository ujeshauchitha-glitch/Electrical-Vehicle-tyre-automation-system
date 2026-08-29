"""State and measurement index layouts for the Phase 3 estimator.

CRITICAL PROJECT RULE: Keep state index slices and measurement index slices
TEXTUALLY SEPARATE.  Sharing them previously left z[0:4] unfilled and
produced a singular covariance matrix.  These are defined independently
here, with no shared constants or aliases between them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema.common import CORNERS


# ===========================================================================
# State layout — what the estimator tracks
# ===========================================================================

@dataclass(frozen=True)
class StateLayout:
    """Named index slices into the state vector x.

    State variables:
    - tread[0:4]   — tread depth per corner (mm)
    - press[4:8]   — RUNNING pressure per corner (kPa, gauge)
                      NOTE: this is RUNNING pressure, not cold-equivalent.
                      Cold-equivalent is a reporting step, never an input
                      to the physics.  See Worker 2's critical project rule.
    - toe_sq       — toe angle SQUARED (deg^2)
                      Toe drag F = 2*C_alpha*toe^2.  The derivative of drag
                      with respect to toe vanishes at zero, so a linearised
                      estimator started at toe=0 sees no gradient.  Estimate
                      toe^2 instead, which enters the observable linearly.
                      Take sqrt at the end.  The SIGN IS NOT RECOVERABLE
                      because drag is even in toe — report magnitude only.
    - camber       — camber angle (deg)
                      Known to have zero sensitivity in every available
                      observable.  Its posterior equals its prior exactly.
                      That is correct behaviour, not a bug.
    """

    # Tread depth (mm), per corner
    tread_fl: int = 0
    tread_fr: int = 1
    tread_rl: int = 2
    tread_rr: int = 3

    # Running pressure (kPa, gauge), per corner
    press_fl: int = 4
    press_fr: int = 5
    press_rl: int = 6
    press_rr: int = 7

    # Toe squared (deg^2) — scalar
    toe_sq: int = 8

    # Camber (deg) — scalar, unobservable
    camber: int = 9

    N: int = 10  # total state dimension

    @property
    def tread_indices(self) -> list[int]:
        return [self.tread_fl, self.tread_fr, self.tread_rl, self.tread_rr]

    @property
    def press_indices(self) -> list[int]:
        return [self.press_fl, self.press_fr, self.press_rl, self.press_rr]


# ===========================================================================
# Measurement layout — what the estimator observes
# ===========================================================================

@dataclass(frozen=True)
class MeasurementLayout:
    """Named index slices into the measurement vector z.

    Measurement channels:
    - press[0:4]   — pressure per corner (kPa)
    - freq[4:8]    — resonance frequency per corner (Hz) — BLOCKED by G1
    - ratio[8:10]  — axle speed ratios (front, rear)
    - roadload     — road load coefficient (dimensionless)

    These indices are TEXTUALLY SEPARATE from StateLayout.
    No shared constants, no aliases, no overlap in meaning.
    """

    # Pressure measurement, per corner
    press_fl: int = 0
    press_fr: int = 1
    press_rl: int = 2
    press_rr: int = 3

    # Resonance frequency, per corner — BLOCKED by G1
    freq_fl: int = 4
    freq_fr: int = 5
    freq_rl: int = 6
    freq_rr: int = 7

    # Axle speed ratios
    ratio_front: int = 8
    ratio_rear: int = 9

    # Road load coefficient
    roadload: int = 10

    N: int = 11  # total measurement dimension

    @property
    def press_indices(self) -> list[int]:
        return [self.press_fl, self.press_fr, self.press_rl, self.press_rr]

    @property
    def freq_indices(self) -> list[int]:
        return [self.freq_fl, self.freq_fr, self.freq_rl, self.freq_rr]


# Singleton instances for convenience
STATE = StateLayout()
MEAS = MeasurementLayout()
