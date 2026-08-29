# Phase 3 — Current State (for handoff)

**Date:** 29 August 2026
**Branch:** phase3-estimator
**Tests:** 225 passing
**Freeze guard:** passing (Phase 1 files untouched except TyreConfig amendment)

---

## What Phase 3 is

Phase 3 is NOT "write an estimator" — the Gauss-Newton skeleton from Phase 2 was
structurally fine. Phase 3 is:

1. **Make the estimator tell the truth** about what it can and cannot see
   (observability contract)
2. **Give it a channel that actually carries tread** (rolling-radius physics)
3. **Harden the numerics** (exceptions, convergence diagnostics, config)

---

## What was built

### New files created

```
src/evtyre/estimation/physics.py        — Tyre physics models
src/evtyre/estimation/schema.py         — TyreStateEstimate (Phase 3→4 contract)
docs/mathematical-audit.md              — Full derivation audit
docs/G1-windowed-sample-contract.md     — Resonance blocker (G1)
docs/review-backend-sprint.md           — Hostile reviewer report
```

### Files modified

```
src/evtyre/config/tyre.py               — Added cold_reference_temperature_c
src/evtyre/estimation/estimator.py      — Observability, physics wiring, hygiene
src/evtyre/estimation/state.py          — (unchanged from Phase 2)
src/evtyre/features/road_load.py        — E3 fix (fabrication eliminated)
src/evtyre/features/pressure_thermal.py — Uses config reference temperature
src/evtyre/pipeline.py                  — Returns TyreStateEstimate
scripts/run_demo.py                     — Shows observability per state
tests/phase1_frozen_hashes.json         — Regenerated manifest
```

---

## Key data structures

### PhysicsConfig (src/evtyre/estimation/physics.py)

```python
@dataclass(frozen=True)
class PhysicsConfig:
    k_z0: float                    # Vertical stiffness at placard (N/m). UNVALIDATED.
    cornering_stiffness: float     # C_alpha (N/rad) for toe drag. UNVALIDATED.
    tread_rr_span: float           # Fraction of C_rr from tread. UNVALIDATED.
    c_rr0: float                   # Reference rolling resistance. UNVALIDATED.
    p_exponent: float              # Pressure exponent for C_rr. UNVALIDATED.
    t_coeff: float                 # Temperature coefficient. UNVALIDATED.
    deflection_factor: float       # Fraction of deflection reducing r_eff. UNVALIDATED.
```

### Observability (src/evtyre/estimation/estimator.py)

```python
class Observability(Enum):
    OBSERVED     = "observed"       # posterior moved, variance shrank materially
    WEAK         = "weak"           # some information, below threshold
    UNOBSERVABLE = "unobservable"   # zero Jacobian sensitivity
```

### StateEstimate (src/evtyre/estimation/estimator.py)

```python
@dataclass(frozen=True)
class StateEstimate:
    name: str
    value: float
    sigma: float
    observability: Observability
    prior_value: float
    prior_sigma: float
    variance_reduction: float   # 1 - (posterior_var / prior_var)
    reason: str | None          # required when not OBSERVED
    magnitude_only: bool = False  # True for toe
```

### TyreStateEstimate (src/evtyre/estimation/schema.py)

```python
@dataclass(frozen=True)
class TyreStateEstimate:
    states: tuple[StateEstimate, ...]
    covariance_diag: tuple[float, ...]
    timestamp_s: float
    odometer_km: float | None
    source: str                    # "simulated" or propagated from features
    model_version: str
    config_fingerprint: str        # SHA-256 of physics config
    n_measurements_available: int
    n_states_observed: int
    mean_variance_reduction: float
    converged: bool
    singular_matrix: bool
    iteration_count: int
```

---

## The core physics (src/evtyre/estimation/physics.py)

```python
def effective_rolling_radius(tread_mm, p_kpa, fz_n, r_belt, k_z0,
                              p_placard_kpa, deflection_factor):
    r_free = r_belt + tread_mm / 1000.0
    k_z = k_z0 * (p_kpa / p_placard_kpa)
    delta = fz_n / max(k_z, 1.0)
    return r_free - deflection_factor * delta

def rolling_resistance_coeff(tread_mm, p_kpa, t_c, tyre_config, physics):
    p_term = (tyre_config.placard_pressure_kpa / max(p_kpa, 1.0)) ** physics.p_exponent
    s = physics.tread_rr_span
    t_term = (1.0 - s) + s * (tread_mm / tyre_config.tread_new_mm)
    T_term = 1.0 + physics.t_coeff * (t_c - tyre_config.cold_reference_temperature_c)
    return physics.c_rr0 * p_term * t_term * T_term

def toe_drag_from_sq(toe_sq_deg2, physics):
    return 2.0 * physics.cornering_stiffness * (math.pi / 180.0) ** 2 * toe_sq_deg2
```

---

## The estimator equation

**Measurement vector z** (11 slots, 5 active):
- z[0:4] = pressure per corner (kPa, gauge)
- z[4:8] = resonance frequency per corner (BLOCKED by G1)
- z[8:10] = axle speed ratios (omega_L / omega_R)
- z[10] = road load coefficient (DISABLED — circular)

**State vector x** (10 slots):
- x[0:4] = tread depth per corner (mm)
- x[4:8] = pressure per corner (kPa, gauge)
- x[8] = toe^2 (deg^2)
- x[9] = camber (deg)

**Predicted measurements h(x):**
- h[press_i] = x[press_i]
- h[ratio_front] = r_eff(FR) / r_eff(FL)   # r_R / r_L
- h[ratio_rear] = r_eff(RR) / r_eff(RL)
- h[roadload] = C_rr_mean + F_toe / (m*g)

**Estimator:**
x* = argmin_x  (z - h(x))^T R^-1 (z - h(x)) + (x - x0)^T P0^-1 (x - x0)

Solved iteratively via Gauss-Newton with A = H^T R^-1 H + P0^-1.

---

## Bugs found and fixed this session

### 1. Axle-ratio sign error (CRITICAL)
**Before:** `_predict()` computed `r_L / r_R` (left over right)
**After:** `_predict()` computes `r_R / r_L` (right over left)
**Why:** Feature extractor emits `omega_L / omega_R`, which by pure rolling = `r_R / r_L`.
Legacy had it right; Phase 3 had it flipped.

### 2. Road-load fabrication (E3, CRITICAL)
**Before:** Missing components defaulted to 0.0, emitted with status=OK
**After:** `total_road_load_force_n` is UNAVAILABLE unless every component is OK

### 3. Dead t_term (E5)
**Before:** `t_term = (1.0 - 0.20) + 0.20 * (tread / tread_new)` with tread unknown → always 1.0
**After:** Removed, omission stated in docstring

### 4. Road-load definition mismatch (B1)
**Before:** `road_load_coefficient` = F_total/(m*g*v) (units: s/m)
**After:** Dimensionless C_rr mean

**And then withdrawn as a measurement entirely.** Making the coefficient
dimensionless fixed the units but left it a pure function of TPMS pressure and
temperature — measured on a live frame it does not respond to motor torque, to
longitudinal acceleration, or to vehicle speed at all. It is the pressure
information restated through the C_rr formula, so admitting it double-counted
`z[press_*]` and let the estimator move tread to close the gap between the
measured coefficient (which omits the tread term) and the predicted one (which
includes it). The channel is no longer appended to `available`. Re-enable only
when it is derived from an actual force measurement — motor torque minus aero
and inertia — which additionally needs the missing grade channel (gap G6).

### 5. Confidence metric (E4)
**Before:** `confidence = len(available)/MEAS.N` (counts zero-info channels)
**After:** `n_states_observed` + `mean_variance_reduction` (honest)

### 6. Hardcoded reference temperature (E7)
**Before:** `_T_REF_K = 293.15` hardcoded
**After:** Uses `tyre_config.cold_reference_temperature_c`

---

## Current demo output

Pasted verbatim from `python scripts/run_demo.py`:

```
Estimator Output:
--------------------------------------------------
  tread_FL            4.9808  +/- 1.9238  WEAK
    Only the within-axle difference is constrained (marginal variance
    reduction 0.4079, but common-mode reduction only 0.0000); the absolute
    level is not recoverable without an independent channel
  tread_FR            4.1182  +/- 1.9206  WEAK
    Only the within-axle difference is constrained (marginal variance
    reduction 0.4098, but common-mode reduction only 0.0000); ...
  tread_RL            4.3428  +/- 1.9192  WEAK
    Only the within-axle difference is constrained (marginal variance
    reduction 0.4106, but common-mode reduction only 0.0000); ...
  tread_RR            4.7567  +/- 1.9226  WEAK
    Only the within-axle difference is constrained (marginal variance
    reduction 0.4086, but common-mode reduction only 0.0000); ...
  press_FL          238.0794  +/- 4.9581  OBSERVED
  press_FR          240.9370  +/- 4.9583  OBSERVED
  press_RL          232.0953  +/- 4.9567  OBSERVED
  press_RR          244.9479  +/- 4.9577  OBSERVED
  toe^2               0.1000  +/- 0.4000  UNOBSERVABLE [magnitude-only]
    Zero Jacobian sensitivity (0.00e+00) - posterior equals prior
  camber              0.0000  +/- 1.0000  UNOBSERVABLE
    Zero Jacobian sensitivity (0.00e+00) - posterior equals prior

  States observed:  4/10
  Mean var. reduction: 0.9846   (over OBSERVED states only)
  Measurements:     6 channels admitted
```

### How to read this

**Pressure is the only thing this system currently knows.** 4 of 10 states are
OBSERVED, and all four are pressures, each with a near-direct TPMS sensor.

**The four tread numbers are real but relative.** The axle ratio r_R/r_L
constrains the tread DIFFERENCE across an axle; the common mode — both corners
wearing together — is unconstrained, which is why the common-mode reduction
prints as 0.0000 while the marginal reduction is ~0.41. So the FL-vs-FR spread
is meaningful and the absolute 4.98 mm is not. Do not quote a tread depth from
this output.

**`Mean var. reduction: 0.9846` is over OBSERVED states only** — i.e. it
describes the pressure estimates, not tread. Averaging in states already
labelled WEAK or UNOBSERVABLE would mix two different questions.

An earlier revision of this document showed all four tread states as OBSERVED
with `States observed: 8/10` and `Mean var. reduction: 0.5575`. That was
recorded before the toe term was removed from the road-load prediction and
before the common-mode check existed. Those numbers are wrong; the values
themselves (4.9808 etc.) were and remain correct.

---

## What's NOT done (by design)

1. **Resonance channel** — blocked by G1 (TelemetryFrame has no sample rate)
2. **Phase 4** — torque_limit, recoverable_energy, decision layer
3. **Dynamic load transfer** — F_z is static only
4. **E6** — effective_rolling_radius_ratio in kinematics.py measures slip, not radius
5. **Bench validation** — every physical constant is UNVALIDATED

---

## Frozen files (DO NOT MODIFY)

```
src/evtyre/schema/          — Phase 1 schema (file-listing guard)
src/evtyre/config/          — Phase 1 config (hash guard, except TyreConfig)
src/evtyre/ingest/          — Phase 1 ingest (file-listing guard)
legacy/ev_tyre_fusion.py    — Reference implementation (hash guard)
```

---

## How to verify everything works

```bash
python -m unittest discover -s tests -t .   # Should show 225 passing
python -m unittest tests.test_phase1_frozen # Freeze guard, must stay green
python scripts/run_demo.py                  # Honest observability output
```

pytest is not a dependency of this project — the suite runs on the standard
library's `unittest`, per the stdlib + numpy constraint.

The two gate suites are the ones that matter most here:

- `tests/test_observability.py` — asserts that a state sitting at its prior is
  never labelled OBSERVED. This is the class whose absence let tread, toe and
  the confidence metric ship wrong; it caught the common-mode bug immediately.
- `tests/test_physics.py` — locks the axle-ratio direction. The measurement is
  omega_left/omega_right and omega is inversely proportional to radius, so the
  prediction must be r_right/r_left. If that inverts, every tread difference
  silently carries the wrong sign and nothing else would catch it.
