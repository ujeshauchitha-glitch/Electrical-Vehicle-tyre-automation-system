# CARLA + Real Dataset Demo — Claude Code Prompt

## Context

This is an EV tyre intelligence system that estimates per-corner tyre condition
(tread depth, pressure, toe, camber) from existing vehicle telemetry using a
Gauss-Newton MAP estimator. Phases 1-6 are complete with 441 passing tests.

**CARLA 0.9.16 is downloaded and extracted at `C:\CARLA`**. The Python API
package `carla` needs to be installed.

The open-source datasets below provide **real-world ground truth** to validate
the estimator against.

## Your Task

Build an **end-to-end visual demo** that:
1. Spins up CARLA with a vehicle driving through a town
2. Feeds CARLA telemetry into the existing estimator pipeline
3. Runs the Q-Motion dataset through the same pipeline for comparison
4. Produces a **live-updating HTML dashboard** showing both results side-by-side
5. Displays estimator accuracy metrics against real ground truth

---

## Step 0: Understand the existing code

Read these files first (DO NOT MODIFY any file under `src/evtyre/schema/` or
`src/evtyre/config/` -- those are frozen):

- `src/evtyre/estimation/estimator.py` -- Gauss-Newton MAP estimator
- `src/evtyre/estimation/physics.py` -- rolling radius, C_rr, toe drag
- `src/evtyre/estimation/state.py` -- state and measurement index layouts
- `src/evtyre/estimation/schema.py` -- TyreStateEstimate contract
- `src/evtyre/features/contract.py` -- Feature dataclass
- `src/evtyre/features/kinematics.py` -- wheel speed ratio extraction
- `src/evtyre/features/pressure_thermal.py` -- pressure features
- `src/evtyre/features/road_load.py` -- road load (DISABLED in estimator)
- `src/evtyre/simulation/interface.py` -- VehicleSimulator ABC
- `src/evtyre/simulation/mock_adapter.py` -- working reference implementation
- `src/evtyre/simulation/carla_adapter.py` -- STUB you will implement
- `src/evtyre/pipeline.py` -- Pipeline class (features -> estimator)
- `frontend/digital_twin.html` -- existing dashboard

Key architecture:
- VehicleSimulator produces SimulationState(telemetry, ground_truth)
- Pipeline.run(telemetry) produces TyreStateEstimate
- 10 states: tread_FL/FR/RL/RR, press_FL/FR/RL/RR, toe-squared, camber
- Measurement vector: 4 pressure + 4 resonance (BLOCKED G1) + 2 axle ratios + 1 roadload
- Road load is deliberately DISABLED (circularity -- see estimator.py)
- Tread is classified WEAK (not OBSERVED) due to common-mode check

---

## Step 1: Download Q-Motion dataset

Download from: https://data.mendeley.com/datasets/p5376m6xfc/1

Contains:
- 81 driving sessions, 233 km, 12.9 hours on Citroen Ami micro-EV
- CAN bus at 4-20 Hz (motor, vehicle status)
- TPMS pressure per wheel (ground truth)
- IMU at 180-200 Hz
- GNSS at 1 Hz
- Scenarios: baseline, aggressive driving, tyre pressure variation (1.5-2.6 bar)
- Occupant weight ground truth (gives Fz)

Map CSV columns to TelemetryFrame:
- tpms_pressure_kpa: from TPMS columns (convert bar->kPa)
- wheel_speed_rad_s: from motor speed and gear ratio
- vehicle_speed_ms: from GNSS
- accel_long_ms2: from IMU
- motor_torque_nm: from motor power and speed
- tpms_temperature_c: from temperature columns

---

## Step 2: Implement CarlaAdapter

CARLA is at C:\CARLA. Install the Python API:
  pip install --find-links=C:/CARLA/PythonAPI/carla/dist carla

Implement src/evtyre/simulation/carla_adapter.py:

1. Connect to CARLA at localhost:2000
2. Spawn vehicle (vehicle.tesla.model3 or similar EV)
3. Apply autopilot or scripted driving
4. At each tick read: wheel speeds, TPMS, IMU, motor torque, vehicle speed
5. Build TelemetryFrame via _make_frame() from interface.py
6. Build GroundTruthSnapshot from CARLA tyre state
7. Return SimulationState

Critical rules:
- Do NOT fabricate missing data -- mark as MISSING
- Noise model from Q-Motion: TPMS ~2 kPa, speed ~0.5%, accel ~0.1 m/s2
- If CARLA channel unavailable, it is MISSING -- estimator handles this

---

## Step 3: Build comparison engine

Create scripts/run_demo_comparison.py:

For each scenario (Normal, Asymmetric Wear, Low Pressure, Toe Misalignment):
  1. MockVehicleAdapter -> estimator -> results
  2. CarlaAdapter -> estimator -> results
  3. Q-Motion data -> estimator -> results
  4. Compare all three against ground truths
  5. Output JSON for dashboard

Metrics per scenario:
- Tread MAE (mm): mean |estimated - ground_truth|
- Pressure MAE (kPa): same for pressure
- 2-sigma coverage: fraction within 2-sigma of estimate
- Observability distribution: OBSERVED / WEAK / UNOBSERVABLE counts
- Convergence rate: % of timesteps where estimator converged

---

## Step 4: Build live dashboard

Update frontend/digital_twin.html with three tabs:

### Tab 1: Live CARLA (if server running)
- Top-down vehicle view (canvas is fine)
- Per-corner tyre state cards updating live
- Observability index with color coding
- Degradation trajectory chart

### Tab 2: Q-Motion Real Data
- Frame-by-frame playback of Q-Motion session
- Ground truth TPMS vs estimated pressure overlay
- Observability classification at each frame
- Accuracy metrics summary

### Tab 3: Accuracy Report
- Side-by-side comparison table (Mock / CARLA / Q-Motion)
- Per-corner error bar charts
- Estimated vs ground truth scatter plot

Design: dark theme, monospace numbers, green/amber/red observability colors,
responsive, works without CARLA (graceful fallback).

---

## Step 5: Tests

Add tests/test_carla_adapter.py and tests/test_qmotion_adapter.py:
- Interface compliance, field mapping, no fabrication
- Noise profiles match Q-Motion measured values
- Full pipeline runs on Q-Motion data without errors
- Accuracy metrics computation correct

---

## Step 6: Verify and commit

  python -m pytest tests/ -q         # ALL tests pass
  python -m pytest tests/test_phase1_frozen.py -q  # freeze guard passes
  python scripts/run_demo_comparison.py             # produces output

Commit on main with descriptive message.

---

## Constraints

1. DO NOT modify src/evtyre/schema/ or src/evtyre/config/ -- FROZEN
2. DO NOT fabricate missing sensor data -- mark MISSING
3. DO NOT hardcode tread as OBSERVED -- it is WEAK (common-mode check)
4. DO NOT claim estimator works on real data until it does -- show real numbers
5. Road load channel is DISABLED -- do not enable without motor-torque derivation
6. Resonance is BLOCKED by G1 -- freq channels will be MISSING
7. carla_adapter.py is currently a stub delegating to MockVehicleAdapter

## Success criteria

1. CARLA vehicle drives through town with telemetry streaming
2. Estimator runs real-time showing per-corner tyre state
3. Observability labels match Python (tread=WEAK, pressure=OBSERVED)
4. Q-Motion real data runs through same pipeline with ground truth comparison
5. Accuracy metrics are honest (not fabricated)
6. Dashboard is interactive and reviewer-friendly
7. If CARLA fails, demo falls back to Q-Motion + mock data
