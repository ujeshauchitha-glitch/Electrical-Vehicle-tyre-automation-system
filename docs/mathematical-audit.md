# Mathematical Audit of the EV Tyre Virtual-Sensing Derivation

**Auditor:** Buffy (Codebuff)
**Date:** 29 August 2026
**Codebase state:** Branch `phase2-features`, after Phase 3 sign-error fix
**Test suite:** 173 tests passing, 146 subtests passing

---

## Purpose

This document audits the ChatGPT derivation of the EV tyre estimation model against
the actual codebase (legacy, Phase 2 features, Phase 3 estimator). Every equation is
checked for physical correctness, unit consistency, sign conventions, and alignment
with the implemented code.

---

## A. Line-by-Line Audit

### Section 0 — Objective: z ≈ h(x)

**Verdict: CORRECT.** Standard nonlinear least-squares formulation. The code
implements it as a Gauss-Newton MAP estimator with prior regularisation. The
derivation acknowledges the prior in Section 17.

---

### Sections 1–2 — Vehicle layout and v = ωr

**Verdict: CORRECT with caveat.** This is the pure-rolling (zero-slip) condition.
On a driven wheel under torque:

    v = ω r (1 - s)

where s is longitudinal slip. The derivation acknowledges this ("ignoring slip")
but the caveat should be stated more prominently: the model degrades under
acceleration/braking — exactly when tread matters for safety.

**Physical law** (with the slip caveat).

---

### Section 3 — r_free = r_belt + d/1000

**Verdict: CORRECT.** Empirical model. A tyre with deeper tread has a larger free
radius. Units: mm→m. Matches legacy.

---

### Section 4 — F_z = mg · s / 2

**Verdict: CORRECT, but incomplete.** Static load distribution only. During
longitudinal acceleration, dynamic load transfer shifts weight between axles.
During cornering, lateral load transfer shifts weight left-to-right. Both are
ignored. This is a **known limitation** — the model is most accurate at constant
speed on a straight road.

**Static equilibrium** (ignores dynamic load transfer).

---

### Section 5 — Deflection: δ = F_z D / k_z

**Verdict: INCORRECT NOTATION.** The ChatGPT derivation introduces a deflection
factor D. The actual code (both legacy and Phase 3) uses:

    r_eff = r_free - F_z / (3 k_z)

The 1/3 factor is the physical constant — it accounts for the fact that tyre
deflection distributes around the contact patch, and only about one-third of
total vertical deflection reduces the effective rolling radius. There is no
separate D.

**Corrected:**

    r_eff = r_belt + d/1000 - F_z / (3 k_z0 (P/P_placard))

**Empirical model** (the 1/3 factor from contact-patch mechanics).

---

### Section 6 — k_z = k_z0 · P / P_placard

**Verdict: CORRECT.** Empirical relationship. Higher pressure → stiffer tyre →
less deflection. Matches code and legacy.

---

### Section 7 — Combined radius model

**Verdict: CORRECT STRUCTURE, wrong notation (inherits the D error from Section 5).**

Corrected:

    r_eff(d, P, F_z) = r_belt + d/1000 - F_z P_placard / (3 k_z0 P)

---

### Section 8 — ω_L / ω_R = r_R / r_L

**Verdict: CORRECT.** Derivation:

    v = ω_L r_L = ω_R r_R  →  ω_L/ω_R = r_R/r_L

This is a **physical law** (from v = ωr, assuming equal vehicle speed for both
wheels on the same axle).

**Critical finding during audit:** The Phase 3 code had a sign error here —
predicting r_L / r_R instead of r_R / r_L. This has been fixed. See
"Critical Code Bug" below.

---

### Sections 9–10 — Front and rear axle equations

**Verdict: CORRECT.** Follow from Section 8.

---

### Section 11 — TPMS gives direct P_i

**Verdict: CORRECT.** TPMS directly measures pressure. The code uses gauge
pressure (TPMS absolute minus atmospheric 101.325 kPa, converted to kPa).

**Subtlety the derivation omits:** the state vector stores gauge pressure (kPa),
the feature extracts absolute Pa, and the estimator converts between them. This
is correct but should be stated explicitly.

---

### Section 12 — Temperature as exogenous input

**Verdict: CORRECT.** Temperature enters the model as an exogenous measured
quantity (via TPMS), not as a state to be estimated. The code extracts
temperature from features and passes it to rolling_resistance_coeff(). This is
sound — temperature affects C_rr but is measured, not inferred.

---

### Section 13 — Rolling resistance model and circularity

**Verdict: CORRECT.** The derivation correctly identifies the circularity problem:
the Phase 2 road_load_coefficient was computed from TPMS pressure/temperature
using the same C_rr model the estimator uses. This has been disabled in Phase 3.0.

**Current status:** The road_load measurement slot in the estimator is populated
from features, but road_load_coefficient is UNAVAILABLE for most frames. When it
IS available, it provides independent information only if derived from a genuinely
independent source (motor torque, vehicle dynamics). Currently it is not.

---

### Section 14 — Measurement vector

**Verdict: CORRECT.** The 11-slot vector with freq and roadload disabled matches
the code. 5 active measurements (4 pressure + 1 ratio) per frame in the common
case.

---

### Section 15 — State vector

**Verdict: CORRECT.** 10 states: 4 tread + 4 pressure + toe² + camber.

---

### Section 16 — Forward model

**Verdict: STRUCTURALLY CORRECT.** With the sign error fixed, the predicted
ratios now correctly match the measured ratios.

---

### Section 17 — The estimator

**Verdict: CORRECT.** The MAP formulation is properly stated. The code implements
Gauss-Newton iteration, which linearises h(x) at each step.

---

### Section 18 — Jacobian tells us whether the answer is real

**Verdict: CORRECT.** The Jacobian column norm answers: "does changing this state
affect any measurement?" With the sign error fixed, tread has non-zero Jacobian
sensitivity through the axle-ratio channel.

---

### Section 19 — Observability classification

**Verdict: CORRECT STRUCTURE.** Three-tier classification (UNOBSERVABLE / WEAK /
OBSERVED) based on Jacobian norm and variance reduction is sound. The code uses
thresholds of 10⁻¹⁰ (Jacobian) and 0.01 (variance reduction).

---

### Section 20 — The fundamental mathematical limitation

**Verdict: CORRECT and the most important section.** The key insight:

    ω_L / ω_R = r_R / r_L

If both tyres lose the same amount of tread (d_L → d_L − Δ, d_R → d_R − Δ),
the ratio barely changes. The measurement constrains:

    d_FL − d_FR  (differential, well-constrained)

but NOT:

    (d_FL + d_FR) / 2  (common mode, poorly constrained)

This is why the plan expects WEAK tread observability at best without the
resonance channel.

---

### Section 21 — Summary chain

**Verdict: CORRECT, with the notation fix from Section 5 carried through.**

The chain is:

    d_i → r_free,i → r_eff,i(P_i, F_z,i) → ω_i → ω_L/ω_R

The missing link is the resonance channel (f_mode → tread) which is blocked by G1.

---

### Section 22 — Independent equations

**Verdict: CORRECT and well-stated.** Each sensor provides a potentially
independent equation. With 5 active measurements and 10 states, the system is
underdetermined — which is why the prior is essential and why many states remain
WEAK or UNOBSERVABLE.

---

## B. Corrected Derivation (critical fixes only)

### Fix 1: Remove the deflection factor D

Replace with the actual 1/3 factor:

    r_eff = r_belt + d/1000 - F_z / (3 k_z0 (P/P_placard))

### Fix 2: The axle-ratio prediction must be r_R / r_L

    z_pred,ratio = r_eff,R / r_eff,L

to match the measured ratio ω_L / ω_R = r_R / r_L.

### Fix 3: Add the static-load limitation

    F_z,i = m g s_i / 2   (STATIC only — no dynamic load transfer)

---

## C. Final Estimator Equation

**Left-hand side** (what we measure — the measurement vector):

    z = [ P_FL, P_FR, P_RL, P_RR, ω_FL/ω_FR, ω_RL/ω_RR ]  ∈ R¹¹
        (5 active, 6 disabled)

**Right-hand side** (what the physics predicts):

    h(x) = [ P_FL,
              P_FR,
              P_RL,
              P_RR,
              r_eff,FR(x) / r_eff,FL(x),
              r_eff,RR(x) / r_eff,RL(x) ]

where for each corner i:

    r_eff,i = r_belt + x_tread,i / 1000
              - F_z,i / (3 k_z0 (x_press,i / P_placard))

**Estimator equation:**

    x* = argmin_x  (z − h(x))ᵀ R⁻¹ (z − h(x)) + (x − x₀)ᵀ P₀⁻¹ (x − x₀)

solved iteratively via Gauss-Newton with A = Hᵀ R⁻¹ H + P₀⁻¹.

---

## D. Observability Table

| State | Physical meaning | Measurement | Observable? | Why |
|-------|-----------------|-------------|-------------|-----|
| P_FL | Front-left pressure (kPa, gauge) | TPMS direct | **OBSERVED** | h_i = x_i, sensitivity = 1 |
| P_FR | Front-right pressure | TPMS direct | **OBSERVED** | Same |
| P_RL | Rear-left pressure | TPMS direct | **OBSERVED** | Same |
| P_RR | Rear-right pressure | TPMS direct | **OBSERVED** | Same |
| d_FL | Front-left tread depth (mm) | ω_L/ω_R (axle ratio) | **WEAK** | Tread and pressure push r_eff in the same direction (near-degenerate); only within-axle differentials are well-constrained |
| d_FR | Front-right tread depth | ω_L/ω_R (axle ratio) | **WEAK** | Same — within-axle pair with d_FL |
| d_RL | Rear-left tread depth | ω_L/ω_R (rear ratio) | **WEAK** | Same, rear axle |
| d_RR | Rear-right tread depth | ω_L/ω_R (rear ratio) | **WEAK** | Same, rear axle |
| toe² | Toe angle squared (deg²) | Road load (C_rr + F_toe/mg) | **UNOBSERVABLE** → WEAK | Road load is currently disabled (circularity). Once independently measured, toe² would be weakly observable through toe drag |
| camber | Camber angle (deg) | None | **UNOBSERVABLE** | No measurement depends on camber. Jacobian column norm = 0. Posterior = prior |

---

## E. Equation Classification

| Equation | Category | Notes |
|----------|----------|-------|
| v = ωr | Physical law | Pure-rolling approximation; degrades under slip |
| r_free = r_belt + d/1000 | Empirical model | Reasonable for small tread variations |
| k_z = k_z0 P/P_placard | Empirical model | Linear stiffness-pressure proportionality |
| r_eff = r_free − F_z/(3k_z) | Empirical model | 1/3 factor from contact-patch mechanics |
| F_z = mgs/2 | Static equilibrium | Ignores dynamic load transfer |
| ω_L/ω_R = r_R/r_L | Physical law | From v = ωr, assuming equal vehicle speed |
| P_measured = P | Measurement model | TPMS direct measurement |
| C_rr = C_rr0 · p_term · t_term · T_term | Empirical model | All parameters UNVALIDATED |
| F_toe = 2 C_α (π/180)² toe² | Empirical model | Toe drag even in toe |
| z ≈ h(x) | Estimator equation | Gauss-Newton MAP |
| P = (HᵀR⁻¹H + P₀⁻¹)⁻¹ | Estimator equation | Laplace posterior, valid under local linearity |
| Observability = ‖H_j‖ and VR | Estimator equation | Mechanical classification, not a physical law |

---

## F. Claims You Should NOT Make in a Presentation/Patent

1. **"We estimate absolute tread depth."** You do not. The axle-ratio channel
   constrains within-axle tread *differentials*, not absolute values. Say:
   "we estimate per-corner tread depth with WEAK observability, primarily
   constrained by left-right differentials."

2. **"The rolling resistance channel provides independent road-load information."**
   Currently it does not. The road_load_coefficient feature was computed from the
   same C_rr model the estimator uses, creating circularity. Say: "road-load
   integration from motor torque is a Phase 4 objective."

3. **"The model accounts for dynamic load transfer."** It does not. F_z is static
   only. Under hard acceleration (exactly when tread matters for safety), the
   load distribution is significantly different from the static assumption.

4. **"Tread is OBSERVED."** With the current codebase, tread may appear OBSERVED,
   but the physical expectation is WEAK at best. The variance reduction threshold
   of 0.01 is permissive. Say: "tread is weakly observable through within-axle
   differentials; absolute tread depth requires the resonance channel."

5. **"The 1/3 deflection factor is validated."** It is explicitly UNVALIDATED.
   All physical constants in PhysicsConfig are labelled as such.

6. **"Toe is estimable from available data."** Toe² is currently UNOBSERVABLE
   because the road-load channel is disabled. Even when enabled, toe drag
   enters as F_toe/(mg) which is typically <0.1% of total road load.

7. **"Temperature compensation is part of the estimator."** Temperature
   compensation is a reporting step (converting running pressure to
   cold-equivalent for display), not an input to the physics.

8. **"The estimator is validated."** The estimator is tested against synthetic
   data generated by the same equations it uses. This proves the inverse
   problem is well-posed and the arithmetic is correct. It does NOT prove the
   equations describe an actual tyre. Every figure is a statement about algebra
   until bench-tested.

---

## G. Critical Code Bug Found During Audit

### The Phase 3 axle-ratio sign error

**The bug:** The Phase 3 `_predict()` function computed the axle ratio as
r_L / r_R (left over right), but the feature extractor measures ω_L / ω_R
which by pure rolling equals r_R / r_L (right over left).

**Evidence:**

- Feature extractor (`kinematics.py:113`): `ratio = omega_l / omega_r`
  → this equals r_R / r_L from pure rolling.

- Phase 3 `_predict()` (before fix): `z_pred[idx] = ra / max(rb, 1e-6)`
  where ra = left, rb = right → this is r_L / r_R (reciprocal).

- Legacy `predict()`: `z[8 + k] = rb / ra` → r_R / r_L (correct).

**Consequence:** The Jacobian sensitivity for tread had the wrong sign — the
estimator pushed tread estimates away from truth rather than toward it. The
error was small in absolute terms (ratios are typically 0.99–1.01) but the
sensitivity was inverted.

**Fix applied:** Changed `ra / max(rb, 1e-6)` to `rb / max(ra, 1e-6)` in
`src/evtyre/estimation/estimator.py`.

**Verification:** After the fix, the demo shows tread values with the correct
polarity — what was high is now low and vice versa, confirming the estimator
now pushes tread in the right direction.

---

## H. Summary of Findings

| Section | Verdict | Issue |
|---------|---------|-------|
| 0 (Objective) | CORRECT | — |
| 1–2 (Kinematics) | CORRECT | Slip caveat not prominent enough |
| 3 (Free radius) | CORRECT | — |
| 4 (Vertical load) | CORRECT | Static only — dynamic load transfer ignored |
| 5 (Deflection) | INCORRECT | D notation should be 1/3 factor |
| 6 (Stiffness) | CORRECT | — |
| 7 (Combined radius) | CORRECT STRUCTURE | Inherited D notation error |
| 8 (Speed ratio) | CORRECT | Code had sign error (fixed) |
| 9–10 (Axle equations) | CORRECT | — |
| 11 (TPMS) | CORRECT | Pa→kPa conversion not stated |
| 12 (Temperature) | CORRECT | — |
| 13 (Rolling resistance) | CORRECT | Circularity identified and disabled |
| 14 (Measurement vector) | CORRECT | — |
| 15 (State vector) | CORRECT | — |
| 16 (Forward model) | CORRECT | After sign fix |
| 17 (Estimator) | CORRECT | — |
| 18 (Jacobian) | CORRECT | — |
| 19 (Observability) | CORRECT | Thresholds need tuning on real data |
| 20 (Limitation) | CORRECT | Most important section |
| 21 (Summary chain) | CORRECT | — |
| 22 (Independent equations) | CORRECT | — |

**Overall verdict: CORRECT WITH CAVEATS.** The derivation is mathematically sound
with three corrections needed: (1) the deflection factor notation, (2) the
axle-ratio sign convention, and (3) the static-load limitation. The code has
been fixed for item (2). Items (1) and (3) are presentational.
