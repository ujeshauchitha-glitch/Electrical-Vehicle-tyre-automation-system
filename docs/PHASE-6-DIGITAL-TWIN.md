# Phase 6 — EV Tyre Digital Twin / Simulation Integration

## Overview

Phase 6 connects a vehicle simulator to the existing estimator pipeline to create an interactive digital twin. The system demonstrates:

1. A virtual EV with hidden ground-truth tyre state
2. Observable telemetry derived from the ground truth
3. The estimator inferring tyre state from telemetry only
4. Observability classification responding to actual Jacobian values
5. Degradation accumulating over simulated distance
6. Asymmetric wear being detected
7. Ground truth vs estimate comparison for validation

## Architecture

```
Hidden Ground Truth (tyres, pressure, toe)
    ↓
Vehicle Dynamics Model
    ↓
Observable Telemetry (TelemetryFrame)
    ↓
Existing Feature Extraction (Phase 2)
    ↓
Existing Estimator (Phase 3)
    ↓
TyreStateEstimate
    ↓
Digital Twin Dashboard
```

The simulator NEVER feeds the estimator ground truth directly. The estimator only receives observable telemetry — the same TelemetryFrame interface used by real vehicle data.

## Simulator Choice

### CARLA (Optional)

CARLA is the primary simulator for photorealistic visualization. The `CarlaAdapter` implements the `VehicleSimulator` interface and converts CARLA's vehicle state into TelemetryFrame objects.

**CARLA does NOT directly provide:**
- Per-corner TPMS pressure (synthesized from ground truth)
- Motor torque (estimated from drivetrain model)
- Resonance frequency (not simulated)

These missing channels are clearly labeled as simulated/derived.

### Mock Adapter (Default)

The `MockVehicleAdapter` generates deterministic synthetic telemetry without requiring CARLA. It uses the same physics models as the estimator to compute observable signals from the hidden ground truth.

The mock adapter is the primary development and demo mode.

## Telemetry Mapping

The simulator outputs a standard TelemetryFrame:

| Simulator Output | TelemetryFrame Field | Source |
|---|---|---|
| Wheel angular velocity | `wheel_speed_rad_s` | v / r_eff per corner |
| TPMS pressure | `tpms_pressure_kpa` | Ground truth + noise |
| TPMS temperature | `tpms_temperature_c` | Load model + noise |
| Motor torque | `motor_torque_nm` | Drivetrain model |
| Motor speed | `motor_speed_rad_s` | v * gear_ratio / r_eff |
| Acceleration | `accel_long_ms2` | Scenario profile |
| Vehicle speed | `vehicle_speed_ms` | Scenario profile |

## Ground Truth Model

The simulator tracks a hidden `GroundTruthTyre` state:

```python
GroundTruthTyre(
    tread_mm={"FL": 5.0, "FR": 5.0, "RL": 4.5, "RR": 3.7},
    pressure_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 180.0},
    toe_sq_deg2=0.0,
)
```

This state evolves over distance via the `DegradationProfile`.

## Circularity Prevention

The motor torque measurement is derived from:
```
F_traction = F_rr + F_toe + m*a + F_aero
T_motor = F_traction * r_eff / (gear_ratio * efficiency)
```

Where F_rr is computed from the ground-truth rolling resistance. The estimator's predict function independently re-computes C_rr from its own state estimate. This is standard EKF architecture — the measurement comes from the drivetrain, not from the estimator's C_rr model.

## Scenarios

| Scenario | Description |
|---|---|
| Normal | All tyres healthy and symmetric |
| Uniform Wear | All tyres wearing at the same rate |
| Asymmetric Wear | RR tyre wearing faster (suspension issue) |
| Low Pressure | RR tyre has a slow leak |
| Pressure Drift | All tyres losing pressure slowly |
| Toe Misalignment | Front axle has excessive toe-out |
| Sensor Missing | Motor torque sensor unavailable |
| Accelerated Degradation | All tyres wearing fast |

## Observability Demonstration

The dashboard displays which states the estimator can actually observe from the available telemetry:

- **OBSERVED**: Posterior variance shrank materially from prior
- **WEAK**: Some information, below threshold
- **UNOBSERVABLE**: Zero Jacobian sensitivity

These classifications come from the actual Jacobian computation, not hard-coded values. The observability bars respond to real sensitivity values.

## Degradation Visualization

The degradation chart shows per-corner tread depth over distance, with:
- Color-coded lines per corner
- Legal tread limit line (1.6 mm)
- Current replay position marker
- Asymmetric wear immediately visible from diverging lines

## Validation Methodology

The ground-truth vs estimate comparison is labeled **SIMULATION ONLY**. It demonstrates that:
1. The estimator math is internally consistent
2. The estimator converges to values close to ground truth
3. Asymmetric degradation is detectable

This is NOT real-world validation. It confirms the pipeline runs end-to-end.

## Limitations

- Mock simulator uses simplified physics (no tyre dynamics, no load transfer)
- No CARLA visualization unless CARLA is installed
- Degradation model is linear (no accelerating wear near end-of-life)
- No environmental factors (temperature, season)
- Grade is not simulated (no incline sensor)

## Files Added

| File | Purpose |
|---|---|
| `src/evtyre/simulation/__init__.py` | Module init |
| `src/evtyre/simulation/interface.py` | VehicleSimulator ABC, SimulationState |
| `src/evtyre/simulation/ground_truth.py` | Hidden tyre/vehicle state |
| `src/evtyre/simulation/scenarios.py` | Pre-defined scenarios |
| `src/evtyre/simulation/mock_adapter.py` | Mock telemetry generator |
| `src/evtyre/simulation/carla_adapter.py` | CARLA adapter (optional) |
| `src/evtyre/simulation/replay.py` | Replay engine |
| `tests/test_phase6.py` | 28 Phase 6 tests |
| `frontend/digital_twin.html` | Interactive dashboard |
| `docs/PHASE-6-DIGITAL-TWIN.md` | This document |

## Test Coverage

28 tests covering:
- Mock simulator produces valid telemetry
- All scenario types are loadable
- Missing channels represented as missing (not zero)
- Estimator receives simulator telemetry correctly
- Replay is deterministic
- Ground truth comparison works
- Physical bounds enforcement
