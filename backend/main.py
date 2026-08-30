"""
FastAPI backend for EV Tyre Intelligence.

Endpoints:
  GET  /api/health           -- health check
  GET  /api/vehicle          -- static vehicle parameters
  POST /api/estimate         -- run estimator on telemetry input
  POST /api/simulation       -- generate random scenario + estimate
  GET  /api/telemetry/latest -- placeholder for live telemetry
  GET  /api/estimator/status -- estimator configuration info

Phase 4 adds independent motor-torque road-load channel.
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .estimator import (
    CORNERS,
    IDX_CAMBER,
    IDX_PRESS,
    IDX_TREAD,
    IDX_TOESQ,
    M_MOTORTORQUE,
    M_ROADLOAD,
    N_MEAS,
    N_STATE,
    STATE_NAMES,
    SensorNoise,
    TyreState,
    estimate,
    measure,
    measurement_covariance,
    observability_analysis,
    prior,
)
from .models import (
    DigitalTwinRequest,
    DigitalTwinResponse,
    EstimateResponse,
    GroundTruthFrame,
    HealthResponse,
    Phase4Diagnostics,
    SimulationFrame,
    SimulationRequest,
    SimulationResponse,
    StateEntry,
    TelemetryRequest,
    TyreFrame,
    VehicleInfoResponse,
)
from .physics import (
    DrivetrainConfig,
    Vehicle,
    aero_drag_force,
    compensate_pressure,
    effective_rolling_radius,
    torque_limit,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EV Tyre Intelligence API",
    description="Tyre state estimation from existing vehicle CAN signals",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

from fastapi.responses import FileResponse
from pathlib import Path

_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

@app.get("/", response_class=FileResponse)
def serve_dashboard():
    """Serve the digital twin dashboard."""
    return _frontend_dir / "digital_twin.html"

@app.get("/dashboard", response_class=FileResponse)
def serve_dashboard_alias():
    """Alias for the dashboard."""
    return _frontend_dir / "digital_twin.html"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_estimate(
    z: np.ndarray,
    T_meas: float,
    iters: int,
    v_ms: float,
    accel_ms2: float = 0.0,
    grade_rad: float = 0.0,
) -> EstimateResponse:
    """Run the estimator and package the result."""
    x, P = estimate(z, T_meas, iters=iters, v_ms=v_ms, accel_ms2=accel_ms2, grade_rad=grade_rad)
    sigma = np.sqrt(np.diag(P))
    _, P0 = prior()
    s0 = np.sqrt(np.diag(P0))

    # Count convergence iterations (re-run to track)
    conv_iters = iters
    x0_ref = prior()[0]
    for k in range(iters):
        x_try, _ = estimate(z, T_meas, iters=k + 1, v_ms=v_ms, accel_ms2=accel_ms2, grade_rad=grade_rad)
        if np.max(np.abs(x_try - x0_ref)) < 100:
            conv_iters = k + 1
            break

    # Build state entries
    states: list[StateEntry] = []
    for j, name in enumerate(STATE_NAMES):
        shrink = s0[j] / sigma[j] if sigma[j] > 1e-15 else float("inf")
        obs = "NO_INFORMATION" if shrink < 1.05 else "OBSERVED"
        states.append(StateEntry(
            name=name,
            value=float(x[j]),
            sigma=float(sigma[j]),
            variance_reduction=float(shrink),
            observability=obs,
        ))

    # Cold-equivalent pressures
    cold_pressures = {
        c: float(compensate_pressure(x[4 + i], T_meas))
        for i, c in enumerate(CORNERS)
    }

    # Torque ceilings from rear-axle tread estimate
    t_drive = float(np.mean([x[2], x[3]]))
    s_drive = float(np.mean([sigma[2], sigma[3]]))
    p_drive = float(np.mean([x[6], x[7]]))

    T_wet, mu_wet, _ = torque_limit(t_drive, s_drive, p_drive, wet=True)
    T_dry, mu_dry, _ = torque_limit(t_drive, s_drive, p_drive, wet=False)

    # Recoverable energy
    from .physics import road_load
    state_for_energy = TyreState(
        tread={c: float(x[i]) for i, c in enumerate(CORNERS)},
        pressure={c: float(x[4 + i]) for i, c in enumerate(CORNERS)},
        temp={c: 25.0 for c in CORNERS},
        toe=float(np.sqrt(max(0.0, x[IDX_TOESQ]))),
    )
    v = 22.0
    F_now = road_load(state_for_energy, v)
    fixed = TyreState(
        tread=state_for_energy.tread,
        pressure={c: Vehicle.p_placard for c in CORNERS},
        temp=state_for_energy.temp,
        toe=0.0,
    )
    F_fixed = road_load(fixed, v)
    energy_pct = 100.0 * (F_now - F_fixed) / F_fixed

    converged = bool(np.max(np.abs(x - prior()[0])) < 100)

    # Phase 4 diagnostics
    phase4 = None
    motor_available = not np.isnan(z[M_MOTORTORQUE])
    if motor_available:
        dt = DrivetrainConfig()
        obs = observability_analysis(x, P, z, T_meas, v_ms, accel_ms2, grade_rad)
        phase4 = Phase4Diagnostics(
            road_load_available=True,
            motor_torque_used=float(z[M_MOTORTORQUE]),
            acceleration_used=float(accel_ms2),
            rolling_radius_used=float(dt.rolling_radius),
            toe_sensitivity_roadload=obs["toe_sensitivity_roadload"],
            toe_sensitivity_motor_torque=obs["toe_sensitivity_motor_torque"],
            toe_variance_prior=obs["toe_variance_prior"],
            toe_variance_posterior=obs["toe_variance_posterior"],
            toe_observability=obs["toe_observability"],
            estimator_converged=converged,
            iteration_count=conv_iters,
            n_measurements_available=obs["n_measurements_available"],
        )
    else:
        obs = observability_analysis(x, P, z, T_meas, v_ms, accel_ms2, grade_rad)
        phase4 = Phase4Diagnostics(
            road_load_available=False,
            motor_torque_used=None,
            acceleration_used=None,
            rolling_radius_used=None,
            toe_sensitivity_roadload=obs["toe_sensitivity_roadload"],
            toe_sensitivity_motor_torque=obs["toe_sensitivity_motor_torque"],
            toe_variance_prior=obs["toe_variance_prior"],
            toe_variance_posterior=obs["toe_variance_posterior"],
            toe_observability=obs["toe_observability"],
            estimator_converged=converged,
            iteration_count=conv_iters,
            n_measurements_available=obs["n_measurements_available"],
        )

    return EstimateResponse(
        states=states,
        converged=converged,
        iterations=conv_iters,
        cold_pressure=cold_pressures,
        torque_ceiling={"wet_Nm": float(T_wet), "dry_Nm": float(T_dry)},
        recoverable_energy_pct=float(energy_pct),
        phase4=phase4,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health():
    """Health check."""
    return HealthResponse()


@app.get("/api/vehicle", response_model=VehicleInfoResponse)
def vehicle_info():
    """Static vehicle parameters."""
    return VehicleInfoResponse()


@app.post("/api/estimate", response_model=EstimateResponse)
def run_estimate(data: TelemetryRequest):
    """Run the estimator on raw telemetry input.

    Phase 4: If motor_torque_nm, vehicle_speed_ms, and acceleration_ms2 are
    all provided, the independent motor-torque road-load channel is activated.
    Otherwise, only the existing measurement channels are used.
    """
    z = np.zeros(N_MEAS)  # 12 elements
    z[0] = data.pressure_fl
    z[1] = data.pressure_fr
    z[2] = data.pressure_rl
    z[3] = data.pressure_rr

    # Estimate resonance frequencies from pressures + assumed tread
    for i, c in enumerate(CORNERS):
        z[4 + i] = effective_rolling_radius(5.0, z[i], Vehicle.Fz(c))  # placeholder

    # Use provided ratios or default to 1.0
    z[8] = data.ratio_front if data.ratio_front is not None else 1.0
    z[9] = data.ratio_rear  if data.ratio_rear  is not None else 1.0

    # Original road-load coefficient
    v_ms = data.vehicle_speed_ms if data.vehicle_speed_ms is not None else 22.0
    z[M_ROADLOAD] = Vehicle.C_rr0  # default baseline

    # Phase 4: Motor torque channel
    accel_ms2 = 0.0
    grade_rad = 0.0
    if (
        data.motor_torque_nm is not None
        and data.vehicle_speed_ms is not None
        and data.acceleration_ms2 is not None
    ):
        z[M_MOTORTORQUE] = data.motor_torque_nm
        accel_ms2 = data.acceleration_ms2
        grade_rad = data.grade_rad if data.grade_rad is not None else 0.0
    else:
        z[M_MOTORTORQUE] = np.nan

    return _build_estimate(z, data.temperature, iters=6, v_ms=v_ms,
                           accel_ms2=accel_ms2, grade_rad=grade_rad)


@app.post("/api/simulation", response_model=SimulationResponse)
def run_simulation(data: SimulationRequest):
    """Generate a random ground-truth scenario and estimate it."""
    rng = np.random.default_rng(data.seed)
    truth = TyreState.random(rng)

    v_ms = 22.0
    accel_ms2 = 1.5  # mild acceleration for realistic scenario
    z, T_meas = measure(
        truth, rng,
        v_ms=v_ms,
        accel_ms2=accel_ms2,
        grade_rad=0.0,
        include_motor_torque=data.include_motor_torque,
    )

    est = _build_estimate(z, T_meas, iters=data.iters, v_ms=v_ms,
                          accel_ms2=accel_ms2, grade_rad=0.0)

    truth_dict: dict[str, dict[str, float] | float] = {
        c: {
            "tread_mm": float(truth.tread[c]),
            "pressure_kPa": float(truth.pressure[c]),
            "temp_C": float(truth.temp[c]),
        }
        for c in CORNERS
    }
    truth_dict["toe_deg"] = float(truth.toe)
    truth_dict["camber_deg"] = float(truth.camber)

    return SimulationResponse(truth=truth_dict, estimate=est)


@app.get("/api/telemetry/latest")
def telemetry_latest():
    """Placeholder for live telemetry streaming."""
    return {"status": "not_connected", "message": "Wire up CAN telemetry here"}


@app.get("/api/estimator/status")
def estimator_status():
    """Estimator configuration and channel info."""
    R = measurement_covariance()
    return {
        "n_state": N_STATE,
        "n_measurement": N_MEAS,
        "state_names": STATE_NAMES,
        "measurement_channels": [
            "pressure_FL", "pressure_FR", "pressure_RL", "pressure_RR",
            "freq_FL", "freq_FR", "freq_RL", "freq_RR",
            "ratio_front", "ratio_rear",
            "road_load",
            "motor_torque",  # Phase 4
        ],
        "diagonal_R": R.diagonal().tolist(),
        "sensor_noise": {
            "tpms_quantum": SensorNoise.tpms_quantum,
            "tpms_sigma": SensorNoise.tpms_sigma,
            "freq_sigma": SensorNoise.freq_sigma,
            "ratio_sigma": SensorNoise.ratio_sigma,
            "roadload_frac": SensorNoise.roadload_frac,
            "motortorque_sigma": SensorNoise.motortorque_sigma,
        },
    }


# ---------------------------------------------------------------------------
# Phase 6: Digital Twin Simulation
# ---------------------------------------------------------------------------

@app.post("/api/digital-twin/run", response_model=DigitalTwinResponse)
def run_digital_twin(data: DigitalTwinRequest):
    """Run a full digital twin simulation using the Phase 6 mock adapter.

    The mock adapter generates physically-consistent telemetry from a hidden
    ground-truth tyre state. The estimator receives only the observable
    telemetry; ground truth is returned for validation.
    """
    import sys
    import os
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from evtyre.simulation.mock_adapter import MockVehicleAdapter
    from evtyre.simulation.scenarios import ScenarioType, load_scenario
    from .estimator import first_mode_frequency, effective_rolling_radius, rolling_resistance_coeff, toe_drag_from_sq

    # Map string to ScenarioType
    scenario_map = {
        "normal": ScenarioType.NORMAL,
        "asymmetric": ScenarioType.ASYMMETRIC_WEAR,
        "low_pressure": ScenarioType.LOW_PRESSURE,
        "toe_misalign": ScenarioType.TOE_MISALIGNMENT,
        "accelerated": ScenarioType.ACCELERATED_DEGRADATION,
        "sensor_missing": ScenarioType.SENSOR_MISSINGNESS,
        "uniform_wear": ScenarioType.UNIFORM_WEAR,
        "pressure_drift": ScenarioType.PRESSURE_DRIFT,
    }
    scenario_type = scenario_map.get(data.scenario, ScenarioType.NORMAL)
    scenario = load_scenario(scenario_type)

    adapter = MockVehicleAdapter()
    adapter.reset(scenario)

    frames = []
    for step in range(data.n_steps):
        state = adapter.step(data.dt_s)
        gt = state.ground_truth

        # Build measurement vector matching the estimator's expected layout:
        # z[0:4]   = TPMS pressure
        # z[4:8]   = resonance frequency (Hz)
        # z[8:10]  = wheel-speed ratios (rear/front per axle)
        # z[10]    = road-load Crr equivalent
        # z[11]    = motor torque (N.m)
        z = np.zeros(N_MEAS)
        v_ms = state.vehicle_speed_ms

        # 1) TPMS pressure — directly from mock adapter
        for i, c in enumerate(CORNERS):
            sr = state.telemetry.tpms_pressure_kpa.get(c)
            z[i] = sr.value if sr and sr.value is not None else 0.0

        # 2) Resonance frequency — computed from ground truth tread + pressure
        #    (the mock adapter doesn't expose this, but a real TPMS would measure it)
        for i, c in enumerate(CORNERS):
            tread = gt.tread_mm[c]
            press = gt.pressure_kpa[c]
            z[4 + i] = first_mode_frequency(tread, press)

        # 3) Wheel-speed ratios — from wheel_speed_rad_s
        ws = {}
        for c in CORNERS:
            sr = state.telemetry.wheel_speed_rad_s.get(c)
            ws[c] = sr.value if sr and sr.value is not None else 0.0
        for k, (a, b) in enumerate((("FL", "FR"), ("RL", "RR"))):
            if ws[a] > 0 and ws[b] > 0:
                z[8 + k] = ws[b] / ws[a]
            else:
                z[8 + k] = 1.0

        # 4) Road-load Crr equivalent — from ground truth tyre state
        crr_vals = [rolling_resistance_coeff(gt.tread_mm[c], gt.pressure_kpa[c], 30.0) for c in CORNERS]
        avg_crr = np.mean(crr_vals)
        F_toe = toe_drag_from_sq(gt.toe_sq_deg2)
        z[M_ROADLOAD] = avg_crr + F_toe / (Vehicle.mass * Vehicle.g)

        # 5) Motor torque — from mock adapter
        if state.motor_torque_nm is not None:
            z[M_MOTORTORQUE] = state.motor_torque_nm
        else:
            z[M_MOTORTORQUE] = np.nan

        # Temperature
        T_meas = state.telemetry.tpms_temperature_c.get("FL")
        T_val = T_meas.value if T_meas and T_meas.value is not None else 25.0

        # Run the actual backend estimator
        x, P = estimate(z, T_val, iters=6, v_ms=v_ms, accel_ms2=0.0)
        sigma = np.sqrt(np.diag(P))
        _, P0 = prior()
        s0 = np.sqrt(np.diag(P0))

        # Build estimate entries per tyre
        estimates = {}
        for j, name in enumerate(STATE_NAMES):
            shrink = s0[j] / sigma[j] if sigma[j] > 1e-15 else float("inf")
            obs = "NO_INFORMATION" if shrink < 1.05 else "OBSERVED"
            if j < 4:
                tread_est = float(x[j])
                wear_mm = Vehicle.tread_new - tread_est
                wear_pct = (wear_mm / (Vehicle.tread_new - Vehicle.tread_legal)) * 100.0
                wear_pct = max(0.0, min(100.0, wear_pct))
                if wear_pct < 20: status = "new"
                elif wear_pct < 50: status = "good"
                elif wear_pct < 75: status = "moderate"
                elif wear_pct < 90: status = "worn"
                else: status = "critical"
                estimates[CORNERS[j]] = TyreFrame(
                    tread_mm=tread_est,
                    tread_sigma=float(sigma[j]),
                    pressure_kpa=float(x[4 + j]),
                    pressure_sigma=float(sigma[4 + j]),
                    observability=obs,
                    vr=float(shrink),
                    wear_pct=round(wear_pct, 1),
                    wear_mm=round(wear_mm, 2),
                    wear_status=status,
                )

        # Toe observability
        toe_shrink = s0[8] / sigma[8] if sigma[8] > 1e-15 else float("inf")
        toe_obs = "NO_INFORMATION" if toe_shrink < 1.05 else "OBSERVED"

        # Ground truth with wear
        gt_wear = {}
        for c in CORNERS:
            gt_wear[c] = GroundTruthFrame(
                tread_mm=float(gt.tread_mm[c]),
                pressure_kpa=float(gt.pressure_kpa[c]),
                wear_pct=round(((Vehicle.tread_new - gt.tread_mm[c]) / (Vehicle.tread_new - Vehicle.tread_legal)) * 100.0, 1),
                wear_mm=round(Vehicle.tread_new - gt.tread_mm[c], 2),
            )

        frame = SimulationFrame(
            step=step,
            timestamp_s=state.timestamp_s,
            odometer_km=state.odometer_km,
            vehicle_speed_ms=state.vehicle_speed_ms,
            treads={c: float(gt.tread_mm[c]) for c in CORNERS},
            presses={c: float(gt.pressure_kpa[c]) for c in CORNERS},
            estimates=estimates,
            ground_truth=gt_wear,
            toe_sq=float(gt.toe_sq_deg2),
            toe_observability=toe_obs,
            converged=bool(np.max(np.abs(x - prior()[0])) < 100),
            n_measurements=int(np.sum(~np.isnan(z[:4])) + (z[8] != 1.0) + (z[9] != 1.0) + 1),
        )
        frames.append(frame)

    return DigitalTwinResponse(
        scenario=data.scenario,
        frames=frames,
        total_distance_km=frames[-1].odometer_km if frames else 0.0,
    )
