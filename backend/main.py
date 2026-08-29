"""
FastAPI backend for EV Tyre Intelligence.

Endpoints:
  GET  /api/health           – health check
  GET  /api/vehicle          – static vehicle parameters
  POST /api/estimate         – run estimator on telemetry input
  POST /api/simulation       – generate random scenario + estimate
  GET  /api/telemetry/latest – placeholder for live telemetry
  GET  /api/estimator/status – estimator configuration info
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
    N_STATE,
    STATE_NAMES,
    SensorNoise,
    TyreState,
    estimate,
    measure,
    measurement_covariance,
    prior,
)
from .models import (
    EstimateResponse,
    HealthResponse,
    SimulationRequest,
    SimulationResponse,
    StateEntry,
    TelemetryRequest,
    VehicleInfoResponse,
)
from .physics import (
    Vehicle,
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
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_estimate(z: np.ndarray, T_meas: float, iters: int, v_ms: float) -> EstimateResponse:
    """Run the estimator and package the result."""
    x, P = estimate(z, T_meas, iters=iters, v_ms=v_ms)
    sigma = np.sqrt(np.diag(P))
    _, P0 = prior()
    s0 = np.sqrt(np.diag(P0))

    # Build state entries
    states: list[StateEntry] = []
    for j, name in enumerate(STATE_NAMES):
        shrink = s0[j] / sigma[j]
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

    converged = bool(np.max(np.abs(x - prior()[0])) < 100)  # heuristic

    return EstimateResponse(
        states=states,
        converged=converged,
        iterations=iters,
        cold_pressure=cold_pressures,
        torque_ceiling={"wet_Nm": float(T_wet), "dry_Nm": float(T_dry)},
        recoverable_energy_pct=float(energy_pct),
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
    """Run the estimator on raw telemetry input."""
    # Build measurement vector from raw telemetry
    z = np.zeros(11)
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

    # Road-load coefficient estimate
    v_ms = 22.0
    z[10] = Vehicle.C_rr0  # default baseline

    return _build_estimate(z, data.temperature, iters=6, v_ms=v_ms)


@app.post("/api/simulation", response_model=SimulationResponse)
def run_simulation(data: SimulationRequest):
    """Generate a random ground-truth scenario and estimate it."""
    rng = np.random.default_rng(data.seed)
    truth = TyreState.random(rng)
    z, T_meas = measure(truth, rng)

    est = _build_estimate(z, T_meas, iters=data.iters, v_ms=22.0)

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
        "n_measurement": 11,
        "state_names": STATE_NAMES,
        "measurement_channels": [
            "pressure_FL", "pressure_FR", "pressure_RL", "pressure_RR",
            "freq_FL", "freq_FR", "freq_RL", "freq_RR",
            "ratio_front", "ratio_rear",
            "road_load",
        ],
        "diagonal_R": R.diagonal().tolist(),
        "sensor_noise": {
            "tpms_quantum": SensorNoise.tpms_quantum,
            "tpms_sigma": SensorNoise.tpms_sigma,
            "freq_sigma": SensorNoise.freq_sigma,
            "ratio_sigma": SensorNoise.ratio_sigma,
            "roadload_frac": SensorNoise.roadload_frac,
        },
    }
