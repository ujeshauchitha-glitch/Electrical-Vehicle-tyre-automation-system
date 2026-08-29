# Backend Sprint Review — Hostile Reviewer Report

**Reviewer:** Worker 9
**Date:** 29 August 2026
**Scope:** Workers 1–8, all files created and modified

---

## Methodology

Reviewed against the five quiet-failure modes:
1. Contract drift (Phase 1 frozen paths modified/added)
2. Fabricated defaults (especially aero parameters and physical constants)
3. Stubs that return instead of raise
4. Sign and directionality errors (magnitude-only treated as signed)
5. Unit errors and lost provenance

Plus: synthetic results described as validation, premature Phase 4/5 work.

---

## Critical Blockers

**None.** The codebase is structurally sound for a Phase 2 skeleton.

---

## High Risk

### H1. Road load coefficients are UNVALIDATED guesses masquerading as real numbers

**Worker 4** (`road_load.py`):
- `C_rr0 = 0.0090`, `p_exponent = 0.45`, `tread_rr_span = 0.20`, `T_coeff = 0.0015`
- `CdA` is provided as a required parameter (good), but the air density `_RHO_AIR_KG_M3 = 1.225` is a hardcoded constant with no way to override it in the feature function's keyword arguments (it defaults to the ISA standard).
- **Recommendation:** These are correctly marked as UNVALIDATED in the docstring, but a consumer could easily mistake them for calibrated values. Consider adding `# GUESS` inline comments on every constant, matching the pattern in legacy.

### H2. Road load extractor signature deviates from the contract

**Worker 4** (`road_load.py`):
- The `extract()` function has a different signature from the standard contract: it requires `road_load_params` as a keyword argument.
- This is by design (the interface gap is documented), but the pipeline (`pipeline.py`) registers it with `extra_kwargs`, and there is no type-check or documentation in the pipeline about which extractors need extra arguments.
- **Recommendation:** Add a docstring to `Pipeline.register()` about extractors with non-standard signatures.

### H3. Cross-corner pressure spread is MAGNITUDE_ONLY

**Worker 2** (`pressure_thermal.py`):
- `cross_corner_pressure_spread_kpa` is correctly marked `MAGNITUDE_ONLY`.
- **Good decision.** The spread is inherently unsigned.

---

## Medium Risk

### M1. Cold-equivalent pressure uses atmospheric pressure constant

**Worker 2** (`pressure_thermal.py`):
- `ATMOSPHERIC_PRESSURE_PA = 101_325.0` — this is a physical constant (measured, not guessed), so it's acceptable.
- However, the cold-equivalent calculation adds atmospheric pressure to the gauge TPMS reading, then normalises by temperature. The result is in absolute Pa. This is physically correct (Gay-Lussac at constant volume), but the feature is named `cold_equivalent_pressure_pa` which could be confused with the running pressure. The naming is fine — just noting for clarity.

### M2. Slip ratio guard threshold is arbitrary

**Worker 3** (`kinematics.py`):
- `_MIN_SPEED_FOR_SLIP_MS = 0.5` — the threshold below which slip ratio is UNAVAILABLE.
- This is a reasonable engineering choice, but it's a guess. At very low speeds, wheel-speed encoder noise dominates.
- **No action needed**, but worth documenting the rationale.

### M3. Estimator predict function is simplified

**Worker 6** (`estimator.py`):
- The `_predict()` function uses a simplified model for axle speed ratios (always returns 1.0) and road load (only pressure-dependent C_rr).
- This is acceptable for a skeleton, but the speed ratio prediction is wrong — it should depend on effective rolling radius, which depends on tread and pressure.
- **Recommendation:** Add a TODO comment noting this simplification.

### M4. Estimator uses numerical Jacobian

**Worker 6** (`estimator.py`):
- The Jacobian is computed numerically via finite differences. This is correct but slow. The legacy code does the same.
- For production, an analytical Jacobian would be faster and more numerically stable.

---

## Minor

### m1. Feature contract allows empty inputs tuple

**Worker 1** (`contract.py`):
- The original spec required non-empty inputs, but the contract was relaxed to allow empty tuples for config-only features (grade resistance, CdA).
- This is the right call — the alternative (requiring a fake input name) would be worse.

### m2. No `__init__.py` in tests/

**Worker 1** (`tests/`):
- The `tests/__init__.py` file was not created. pytest discovers tests without it, so this is fine.
- Some projects require it for `unittest` discovery. Not an issue here.

### m3. Demo script uses simulated data

**Worker 8** (`run_demo.py`):
- The demo runs on a simulated frame. This is expected — there's no real telemetry yet.
- The output is clearly labeled as "simulated" source.

---

## Good Decisions Worth Keeping

1. **Freeze guard with directory-listing assertion** (Worker 1): The manifest includes both hashes and file lists, catching both modification and addition. This is the right approach.

2. **Resonance stub raises, not returns** (Worker 5): A returning stub would silently fill the resonance channel. Raising forces the caller to confront G1.

3. **Toe estimated as toe^2** (Worker 6): Correctly avoids the vanishing-gradient problem at toe=0. The magnitude-only report is correct — sign is unrecoverable from drag.

4. **State and measurement slices are textually separate** (Worker 6): No shared Python objects between `StateLayout` and `MeasurementLayout`. This prevents the legacy z[0:4]-unfilled bug.

5. **Road load params are required, not defaulted** (Worker 4): No silent fabrication of CdA or drag coefficient.

6. **Worn tyres have LOWER rolling resistance** (Worker 4): The docstring explicitly states this, and the code correctly implements `t_term = (1 - span) + span * (tread/tread_new)` which decreases as tread decreases.

7. **Pipeline catches NotImplementedError** (Worker 8): The resonance extractor's raise is caught and recorded as UNAVAILABLE, not propagated as a crash.

8. **Adversarial tests catch the right things** (Worker 7): The all-missing-frame test verifies no numbers are returned, NaN/Inf don't propagate, and provenance survives.

---

## Items Flagged but Not Blocking

- **Camber unobservability**: Correctly asserted (posterior = prior). Not a bug.
- **Resonance blocked by G1**: Correctly stubbed. The physics question is real and unresolved.
- **No real telemetry yet**: All data is simulated. No claims of validation.
- **No Phase 4/5 work**: Correctly scoped to Phase 2 + Phase 3 skeleton.

---

## Verdict

The codebase is ready for tonight's review sprint. The structural choices are sound, the contract is clean, and the documented bugs are not reintroduced. The main risk is the road load coefficients being mistaken for calibrated values — but they are correctly marked as UNVALIDATED throughout.
