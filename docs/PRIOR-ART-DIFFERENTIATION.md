# Prior Art Differentiation Report

**Project:** EV Tyre Virtual-Sensing Estimation System
**Date:** 30 August 2026
**Prepared for:** Review 1 — patent positioning

---

## Executive Summary

Our system is a **software-only, zero-additional-hardware** method for estimating
hidden tyre states (tread depth, pressure, toe) from signals an EV already broadcasts
on its CAN bus. It is fundamentally different from every cited prior art reference in
three dimensions: **hardware requirements**, **estimation architecture**, and
**observability honesty**.

None of the three prior art patents attempt what we do. Two require dedicated
physical sensors or hardware installed on or near the tyre. The third (the closest
prior art) uses torsional mode frequency but (a) requires an accelerometer or
crown-mounted sensor, (b) uses a single-channel recursive least-squares estimator
with no observability analysis, and (c) makes no claim about multi-state fusion or
honest uncertainty reporting.

---

## Prior Art References

| Ref | Patent | Assignee | Approach | Hardware Required |
|-----|--------|----------|----------|-------------------|
| **PA1** | US20190184763A1 (US11090984B2) | Goodyear | Electroactive polymer sensors embedded in tread, RFID readout | Yes — embedded sensors + RFID reader |
| **PA2** | US9259976B2 | Goodyear | Torsional mode frequency + TPMS pressure + RLS algorithm | Partially — requires accelerometer or crown-mounted sensor |
| **PA3** | WO2020086698A1 | Tyrata Inc | IR stereoscopic camera tread profiling | Yes — drive-over IR camera rig |

---

## Differentiation by Dimension

### 1. Hardware Requirements

| Dimension | PA1 (Goodyear Sensor) | PA2 (Goodyear Torsional) | PA3 (Tyrata IR Camera) | **Our System** |
|-----------|----------------------|--------------------------|------------------------|----------------|
| Tyre-embedded sensor | Yes — electroactive polymer array in tread groove | No (uses TPMS module) | No | **No** |
| RFID reader | Yes — hub-mounted or vehicle-mounted | No | No | **No** |
| Accelerometer | No | Yes — hub-mounted or crown-mounted | No | **No** |
| Camera system | No | No | Yes — IR stereo cameras in drive-over enclosure | **No** |
| Drive-over infrastructure | No | No | Yes — enclosure with optical opening, shutter, air knife | **No** |
| TPMS (existing) | No (uses own RFID) | Yes | No | **Yes** |
| ABS wheel speed (existing) | No | Yes (as signal source) | No | **Yes** |
| Motor torque (existing) | No | No | No | **Yes (Phase 4)** |

**Key differentiator:** Our system requires **zero additional hardware** beyond
what every modern EV already has. PA1 requires manufacturing tyres with embedded
sensors. PA2 requires an accelerometer (not standard on all vehicles). PA3 requires
a drive-over camera rig (service lane only, not continuous monitoring).

### 2. Estimation Architecture

| Dimension | PA1 | PA2 | PA3 | **Our System** |
|-----------|-----|-----|-----|----------------|
| Method | Direct readout (binary: sensor abraded or not) | RLS with polynomial model | Direct measurement (optical triangulation) | **Gauss-Newton MAP with physics-based forward model** |
| Physics model | None (mechanical ablation) | Polynomial correlation (f(wear, pressure) = frequency) | None (geometric triangulation) | **Rolling radius + vertical stiffness + rolling resistance** |
| States estimated | Tread depth only (1 corner) | Tread depth only (1 corner) | Tread profile (geometry) | **10 states: 4 tread + 4 pressure + toe² + camber** |
| Multi-corner | Per-sensor, independently | Per-tyre with tire-specific coefficients | Per-tire via camera positioning | **Joint estimation across all 4 corners simultaneously** |
| Sensor fusion | None (single sensor) | TPMS pressure + torsional frequency | None (single modality) | **Multi-channel fusion: TPMS + wheel speed ratios + road load** |
| Prior / regularization | None | Forgetting factor in RLS | None | **Population prior with MAP regularization** |
| Iterative refinement | N/A | Recursive (per-sample update) | N/A | **Iterated Gauss-Newton (up to 10 iterations)** |

**Key differentiator:** Our system fuses multiple independent measurement channels
through a physics-based forward model. PA2's polynomial model captures a correlation
between torsional frequency and wear, but does not model the underlying physics
(rolling radius, vertical stiffness, pressure-dependent deflection). Our forward
model explicitly represents: tread → free radius → pressure-dependent stiffness →
effective rolling radius → wheel speed ratio. This physics grounding is what enables
the observability analysis (see Dimension 3).

### 3. Observability Honesty (the core inventive step)

| Dimension | PA1 | PA2 | PA3 | **Our System** |
|-----------|-----|-----|-----|----------------|
| Reports uncertainty | No — binary readout | No — point estimate only | No — geometric measurement | **Yes — per-state covariance** |
| Classifies observability | N/A (direct measurement) | No — assumes observable | N/A (direct measurement) | **Yes — OBSERVED / WEAK / UNOBSERVABLE per state** |
| Detects common-mode limitation | N/A | No | N/A | **Yes — projects posterior onto axle-pair basis** |
| Refuses to fabricate | N/A (sensor is either there or abraded) | No — always returns a number | N/A (camera either sees tread or doesn't) | **Yes — labels unobservable states with reason** |
| Jacobian analysis | None | None | None | **Column norm + variance reduction** |

**Key differentiator:** This is the patentable inventive step. No prior art system
mechanically determines which states are mathematically inferrable from the available
signals and honestly reports what it cannot see. Our system:

1. Computes the Jacobian column norm for every state at every estimation step.
2. Projects the posterior covariance onto the common-mode direction (u = (e_i + e_j)/√2)
   for each axle pair.
3. If the marginal variance shrinks but the common-mode variance does not (as happens
   for tread, where the left-right difference is constrained but the absolute level is
   not), the state is demoted from OBSERVED to WEAK.
4. If the Jacobian column is zero (as for camber), the state is labelled UNOBSERVABLE
   with a reason.
5. Every non-OBSERVED state carries a textual reason explaining why.

PA2's RLS estimator returns a tread depth number with no confidence interval, no
observability classification, and no detection of whether the estimate is constrained
or just returning the initial guess. A consumer cannot distinguish a real estimate
from a mathematical coincidence.

### 4. Measurement Model Completeness

| Measurement channel | PA1 | PA2 | PA3 | **Our System** |
|---------------------|-----|-----|-----|----------------|
| TPMS pressure | No (own RFID) | Yes (direct input) | No | **Yes — identity map in forward model** |
| Torsional mode frequency | No | Yes (primary channel) | No | **Blocked by G1 (unverified physics)** |
| Wheel speed ratio (axle partner) | No | No (uses raw wheel speed for spectral analysis) | No | **Yes — r_R/r_L via rolling radius model** |
| Road load / C_rr | No | No | No | **Planned: motor-torque derived (Phase 4)** |
| Tyre temperature | No | No | No | **Yes — cold-pressure compensation** |
| Optical tread profile | No | No | Yes (primary) | No |
| Electroactive polymer state | Yes (primary) | No | No | No |

**Key differentiator:** Our wheel speed ratio channel has a physics-based derivation:
ω_L/ω_R = r_R/r_L, where each r_eff depends on tread, pressure, and vertical load.
This is not a correlation — it is a physical equation. PA2 uses wheel speed only as
a signal source for spectral analysis (FFT to extract torsional mode), not as a
kinematic measurement.

### 5. What Happens When Data Is Missing

| Scenario | PA1 | PA2 | PA3 | **Our System** |
|----------|-----|-----|-----|----------------|
| Sensor fails | "Not available" — no estimate | Returns estimate with no indication of degraded quality | "No image" | **Labels affected states UNOBSERVABLE with reason; unaffected states remain estimated** |
| All sensors missing | No output | Returns prior as estimate (undete
ctable) | No output | **Every state labelled UNOBSERVABLE; no values presented as estimates** |
| One channel noisy | N/A (binary) | Weighted by forgetting factor, but no observability check | N/A | **R matrix inflates noise; Jacobian detects reduced sensitivity** |

**Key differentiator:** PA2's RLS estimator with forgetting factor will produce
a tread depth estimate from any combination of inputs, including only pressure data
with no torsional signal. The estimate will be the prior, but the consumer has no
way to know this. Our system explicitly refuses to present a prior as an estimate.

---

## Specific Claim-by-Claim Differentiation

### Against PA1 (US20190184763A1 -- Goodyear Electroactive Sensor)

| PA1 Claim Element | Our System | Differentiation |
|-------------------|------------|-----------------|
| "tread wear sensor mounted in the tread" | No sensor in tread | No hardware modification to tyre |
| "electroactive polymer for emitting a voltage in response to deformation" | No electroactive polymer | Uses existing vehicle signals |
| "RFID sensor tag that can be powered and read by an RFID reader" | No RFID | Uses CAN bus data already available |
| "sequential sacrificially abrade" | No physical ablation | Software estimation, non-destructive |

**Verdict:** Entirely different approach. PA1 is a hardware sensing system; ours is
a software estimation system. No overlap in claims.

### Against PA2 (US9259976B2 -- Goodyear Torsional Mode)

| PA2 Claim Element | Our System | Differentiation |
|-------------------|------------|-----------------|
| "tire torsional mode measuring means" | Torsional channel blocked (G1); uses wheel speed ratio instead | Different measurement channel |
| "recursive least squares algorithm based on a polynomial model" | Gauss-Newton MAP with physics-based forward model | Different estimation method; physics-grounded, not polynomial |
| "tire-specific torsional mode coefficients" | Physics constants are vehicle/tyre parameters, not empirical coefficients | Parameters have physical meaning (k_z0, deflection_factor), not curve-fit |
| Single-state output (tread depth) | 10-state vector (tread x4, pressure x4, toe-sq, camber) | Multi-state joint estimation |
| No observability analysis | Per-state observability classification | Jacobian-based, with common-mode check |

**Verdict:** Closest prior art, but fundamentally different in three ways:
(1) We use wheel speed ratios as a kinematic measurement, not spectral analysis;
(2) Our estimator is physics-based and multi-state; (3) We classify observability
mechanically, which PA2 does not attempt.

### Against PA3 (WO2020086698A1 -- Tyrata IR Camera)

| PA3 Claim Element | Our System | Differentiation |
|-------------------|------------|-----------------|
| "IR radiation source configured to project IR radiation on the tread" | No IR source | No hardware |
| "camera configured to generate an image of the tread" | No camera | Uses CAN bus signals |
| "drive-over enclosure" | No enclosure | Continuous monitoring, not service-lane only |
| "measure the profile of the tread based on the image" | Software estimation from kinematics | Different measurement principle entirely |
| Direct geometric measurement | Inverse estimation from vehicle dynamics | We infer tread from how the vehicle behaves, not from looking at it |

**Verdict:** Entirely different approach. PA3 is an optical measurement system;
ours is a software estimation system that runs continuously on existing vehicle hardware.

---

## Novelty Arguments

### Argument 1: Software-only multi-state tyre estimation from existing EV telemetry

No prior art estimates multiple tyre states (tread, pressure, toe, camber) jointly
from existing vehicle CAN bus signals without any additional hardware.

### Argument 2: Physics-based forward model connecting tread to wheel speed ratio

The chain tread -> free radius -> pressure-dependent stiffness -> effective rolling
radius -> wheel speed ratio (omega_L/omega_R = r_R/r_L) is not present in any prior
art. PA2 uses a polynomial correlation; our model is derived from first principles.

### Argument 3: Mechanical observability classification from Jacobian analysis

The method of computing Jacobian column norms per state, projecting posterior
covariance onto axle-pair subspaces, classifying each state as OBSERVED / WEAK /
UNOBSERVABLE, and refusing to present unverifiable estimates as facts is not present
in any prior art.

### Argument 4: Honest common-mode detection for axle-paired states

The technique of detecting that marginal variance reduction can coexist with zero
common-mode variance reduction, and using this to demote a state from OBSERVED to
WEAK, is novel. Axle speed ratios constrain the difference between left and right
tyres, not the absolute tread depth of either.

### Argument 5: Zero-hardware continuous monitoring

Our system provides continuous monitoring using signals always available on an EV
CAN bus, with no additional hardware, no service lane visit, and no tyre modification.

---

## What We Should NOT Claim

1. Absolute tread depth accuracy -- tread is currently WEAK, not OBSERVED.
2. Resonance-based estimation -- blocked by G1 (unverified physics).
3. Wear rate estimation -- Phase 5, not yet implemented.
4. Dynamic load transfer -- not modelled.
5. Bench-validated accuracy -- every physical constant is UNVALIDATED.

---

## Recommended Patent Claim Scope

**Independent claim 1 (broadest):** A method for estimating tyre states from vehicle
telemetry, comprising: receiving CAN bus signals comprising TPMS pressure and wheel
speeds; computing a physics-based forward model relating tread depth, pressure, and
vertical load to predicted wheel speed ratios; solving a regularised least-squares
estimation problem to find tyre states minimising prediction error; and classifying
each estimated state by computing Jacobian column norms and projecting posterior
covariance onto axle-pair subspaces to determine observability.

**Dependent claims:**
- The common-mode / differential-mode decomposition for axle-pair states
- The refusal to present unobservable states as estimates (reason labelling)
- The multi-state joint estimation across all four corners
- The physics-based rolling radius model
- The motor-torque-derived road load channel (Phase 4)
- The config-fingerprinted provenance chain

---

## References

1. US20190184763A1 / US11090984B2 -- "Sensor system for monitoring tire wear"
   Goodyear Tire & Rubber Co, 2018-12-19
2. US9259976B2 -- "Torsional mode tire wear state estimation system and method"
   Goodyear Tire & Rubber Co, 2013-08-12
3. WO2020086698A1 -- "Methods and systems used to measure tire treads"
   Tyrata Inc, 2019-10-23
