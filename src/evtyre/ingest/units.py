"""Pure unit-conversion helpers: no validation, no I/O, no dependencies.

Raw telemetry may arrive in whatever units the source system uses (rpm, psi,
Fahrenheit, g, km/h); everything downstream of Phase 1 works in SI (or
SI-derived) units. None of these fabricate a value - a missing input (None)
always produces a missing output (None).
"""

from __future__ import annotations

import math
from typing import Optional

_RPM_TO_RAD_S = 2.0 * math.pi / 60.0
_PSI_TO_KPA = 6.894757293168361
_G_TO_MS2 = 9.80665
_KMH_TO_MS = 1.0 / 3.6


def rpm_to_rad_s(rpm: Optional[float]) -> Optional[float]:
    return None if rpm is None else rpm * _RPM_TO_RAD_S


def psi_to_kpa(psi: Optional[float]) -> Optional[float]:
    return None if psi is None else psi * _PSI_TO_KPA


def fahrenheit_to_celsius(fahrenheit: Optional[float]) -> Optional[float]:
    return None if fahrenheit is None else (fahrenheit - 32.0) * 5.0 / 9.0


def g_to_ms2(g_units: Optional[float]) -> Optional[float]:
    return None if g_units is None else g_units * _G_TO_MS2


def kmh_to_ms(kmh: Optional[float]) -> Optional[float]:
    return None if kmh is None else kmh * _KMH_TO_MS
