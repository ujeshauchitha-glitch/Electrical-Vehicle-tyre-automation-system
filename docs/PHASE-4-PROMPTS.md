# Phase 4 — Implementation Prompts (motor-torque-based road-load estimation)

**Date:** 29 August 2026
**Prerequisites:** Phase 3 complete — 225 tests passing, observability contract live, physics module wired in

---

## Context

Phase 3 established the estimator with two active measurement channels:
- **TPMS pressure** (4 channels) — directly observed
- **Axle speed ratios** (2 channels) — carry within-axle tread differentials

One channel is deliberately disabled:
- **Road load coefficient** — was computed from TPMS pressure/temperature using the same C_rr model the estimator uses, creating circularity. Disabled in Phase 3.0.

Phase 4 builds the first **genuinely independent** road-load measurement from motor torque, motor speed, vehicle speed, and longitudinal acceleration. This is the channel that:

1. **Enables toe estimation** — toe drag enters road load as F_toe/(m*g), currently invisible
2. **Separates tread from pressure in C_rr** — C_rr depends on tread and pressure; wheel-speed ratios only see rolling radius. Road load adds a second equation involving both.
3. **Closes the energy story** — quantifies avoidable energy waste from misalignment and under-inflation

---

## Shared constraint block

```
CONSTRAINTS (apply to all work in this task):
- Do NOT modify legacy/ev_tyre_fusion.py. It is frozen and hash-guarded.
- Do NOT modify anything under src/evtyre/schema/, src/evtyre/config/, or
  src/evtyre/ingest/. These are frozen Phase 1 paths, guarded by
  tests/test_phase1_frozen.py against modification, removal AND addition.
  Put new Phase 4 contracts in src/evtyre/estimation/ or src/evtyre/features/.
- Do NOT add dependencies. stdlib + numpy only.
- Do NOT implement resonance / spectral feature extraction (blocked by G1).
- Do NOT implement Phase 5 (temporal degradation-rate estimation).
- Every physical constant you introduce must be a REQUIRED config field with
  NO default, carrying an explicit "validated: no" note.
- python -m pytest tests/ -q must pass, including test_phase1_frozen,
  before you are done.

REPORT WHEN DONE:
- exact files created/changed
- test results (full count, not just "passing")
- any architectural decisions you made
- anything intentionally left unimplemented, and why
```

---

## Prompt 4a — Road-load feature extraction from motor torque (blocking, solo agent)

Build a feature extractor that computes a road-load coefficient from motor torque,
motor speed, vehicle speed, and longitudinal acceleration — the signals an EV
already has on its CAN bus.

### The physics

At constant speed on a flat road, the driving force balances road load:

    F_drive = F_roll + F_aero + F_grade + F_inertia + F_toe

where:
- F_drive = τ_motor / r_eff  (motor torque / effective rolling radius)
- F_roll = C_rr * m * g      (rolling resistance)
- F_aero = 0.5 * ρ * CdA * v² (aerodynamic drag)
- F_grade = m * g * sin(grade) (grade resistance — UNAVAILABLE, no incline sensor)
- F_inertia = m * a_x          (longitudinal acceleration)
- F_toe = 2 * C_alpha * (π/180)² * toe² (toe drag — we want to ESTIMATE this, not assume it)

Rearranging for the road-load coefficient that the estimator predicts:

    C_rr_measured = (F_drive - F_inertia) / (m * g)

This is the **independently measured** road-load coefficient. It contains:
- Rolling resistance (depends on tread and pressure — enters C_rr)
- Toe drag (depends on toe² — enters as F_toe/(m*g))
- NO aerodynamic drag (we subtracted it, or it's negligible at low speed)
- NO grade (assumed flat — gap G6)

### What to build

1. Create `src/evtyre/features/road_load_torque.py` with an `extract_road_load_from_motor()` function.

2. **Inputs** (from TelemetryFrame):
   - `motor_torque_nm` — traction inverter torque output
   - `motor_speed_rad_s` — traction inverter speed
   - `vehicle_speed_ms` — vehicle speed
   - `longitudinal_accel_m_s2` — ESC inertial unit
   - Plus the existing tyre/vehicle configs for m, g, r_belt, tread, pressure

3. **The extraction formula**:
   ```
   F_drive = motor_torque_nm / r_eff
   F_inertia = mass_kg * longitudinal_accel_m_s2
   C_rr_inferred = (F_drive - F_inertia) / (mass_kg * g)
   ```

4. **Noise model**: This measurement is noisy. Sources:
   - Motor torque accuracy (typically ±2-5% for production inverters)
   - Drivetrain losses (gearbox efficiency, bearing friction — NOT modelled, treated as noise)
   - Road gradient (assumed zero — any grade aliases into C_rr)
   - Speed-dependent aerodynamic drag (if not subtracted)
   Use a conservative noise estimate: σ_fraction = 0.10 (10%) as UNVALIDATED default.

5. **Availability rules** (following Phase 3's anti-fabrication pattern):
   - `road_load_from_motor` is UNAVAILABLE if motor_torque_nm is MISSING
   - UNAVAILABLE if vehicle_speed_ms is MISSING or < 2 m/s (low-speed noise dominates)
   - UNAVAILABLE if longitudinal_accel_m_s2 is MISSING
   - Otherwise OK

6. **Output**: A single `Feature` object:
   ```python
   Feature(
       name="road_load_from_motor",
       value=C_rr_inferred,        # dimensionless
       unit="",
       status=FeatureStatus.OK or UNAVAILABLE,
       unavailable_reason=...,
       directionality=Directionality.NATURAL,
       classification=Classification.A,
       inputs=("motor_torque_nm", "vehicle_speed_ms", "longitudinal_accel_m_s2"),
       corner=None,
       timestamp_s=...,
       provenance=...,
       extractor_version=...,
   )
   ```

### Tests

- At constant speed (a_x = 0), C_rr_inferred should be close to the true C_rr
- Under acceleration (a_x > 0), C_rr_inferred should decrease (some torque goes to inertia, not rolling)
- Under braking (regen, a_x < 0), C_rr_inferred should increase
- All-MISSING frame → Feature is UNAVAILABLE with reason
- Low speed (< 2 m/s) → UNAVAILABLE
- Noise test: add ±5% torque noise, assert C_rr_inferred is within ±2σ of truth

[+ shared constraint block]

---

## Prompt 4b — Wire road-load-from-motor into the estimator (parallel with 4a, after Prompt 4a's feature exists)

Replace the disabled circular road-load channel with the new independent
measurement from motor torque.

### What to change

1. **`src/evtyre/features/road_load.py`**: Keep the existing `road_load_coefficient` feature (it may still be useful for diagnostics), but the estimator should now consume `road_load_from_motor` instead.

2. **`src/evtyre/estimation/estimator.py`** `features_to_measurement()`:
   - Look for `road_load_from_motor` in the feature map (not `road_load_coefficient`)
   - When available, populate `z[MEAS.roadload]` with the motor-derived value
   - Use the new noise estimate (σ = 0.10 * C_rr_ref) instead of the old 5%

3. **`_predict()` road-load channel**: The prediction should be:
   ```
   z_pred[roadload] = C_rr_mean + F_toe/(m*g)
   ```
   This is already what the code computes. The key change is that the MEASUREMENT
   now comes from a genuinely independent source (motor torque), not from TPMS.

4. **The estimator can now see toe** — the road-load channel has non-zero
   sensitivity to toe² through F_toe/(m*g). The observability classifier should
   automatically detect this and upgrade toe² from UNOBSERVABLE to at least WEAK.

### Tests

- With motor-torque-derived road load available, toe² should become WEAK or OBSERVED
- The estimator should still work when road_load_from_motor is UNAVAILABLE
- Regression: all existing tests still pass
- Ablation: compare tread observability with and without the road-load channel

[+ shared constraint block]

---

## Prompt 4c — Update demo and integration (solo, last)

1. **`scripts/run_demo.py`**: Add motor-torque telemetry to the simulated data.
   Show the road-load channel as active when motor data is present.
   Display toe estimate with its observability.

2. **`src/evtyre/pipeline.py`**: Accept motor torque/speed in the feature pipeline.
   The pipeline should pass motor features through to the estimator.

3. **Cross-module tests**:
   - Full path: motor torque → feature → estimator → TyreStateEstimate with toe
   - Provenance propagation: motor features carry through to the estimate
   - All-missing motor data → road load UNAVAILABLE → toe UNOBSERVABLE (graceful degradation)

4. Run the FULL suite and report the real count.

[+ shared constraint block]

---

## Sequencing

```
Prompt 4a (motor-torque feature)  ──>  Prompt 4b (wire into estimator)
    blocking, solo                           blocking, solo
                                                    │
                                                    ▼
                                          Prompt 4c (demo + integration)
                                                    solo, last
```

## Verification

After each prompt, and again at the end:

```bash
python -m pytest tests/ -q
python scripts/run_demo.py
```

The demo is the acceptance test: after 4b, it should show `toe^2 WEAK` (not
UNOBSERVABLE), and the road-load channel should be active. After 4c, the full
pipeline should work end-to-end with motor torque as input.

## What this unlocks

| Before Phase 4 | After Phase 4 |
|----------------|---------------|
| Toe² UNOBSERVABLE | Toe² WEAK (road-load channel sees F_toe) |
| Road-load channel disabled (circular) | Road-load channel active (motor-torque derived) |
| C_rr_measured unavailable | Independent C_rr from motor torque |
| No energy story | Avoidable energy waste quantifiable |

## What this does NOT unlock

- Absolute tread depth (still needs resonance — G1 blocker)
- Grade resistance (still gap G6 — no incline sensor)
- Per-corner toe (only one lumped toe² scalar)
- Toe sign (drag is even in toe — magnitude only)
- Degradation rate (Phase 5)
