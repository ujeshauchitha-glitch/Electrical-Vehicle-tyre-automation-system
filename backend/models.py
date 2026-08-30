"""
Pydantic models for API request/response validation.

Phase 4 adds motor-torque telemetry fields and observability diagnostics.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TelemetryRequest(BaseModel):
    """Raw sensor telemetry from the vehicle CAN bus."""
    pressure_fl: float = Field(..., description="TPMS pressure, FL (kPa gauge)")
    pressure_fr: float = Field(..., description="TPMS pressure, FR (kPa gauge)")
    pressure_rl: float = Field(..., description="TPMS pressure, RL (kPa gauge)")
    pressure_rr: float = Field(..., description="TPMS pressure, RR (kPa gauge)")

    ratio_front: Optional[float] = Field(None, description="Wheel-speed ratio FL/FR")
    ratio_rear:  Optional[float] = Field(None, description="Wheel-speed ratio RL/RR")

    temperature: float = Field(25.0, description="Mean tyre temperature (deg C)")

    # Phase 4: independent motor-torque road-load channel
    motor_torque_nm: Optional[float] = Field(
        None,
        description="Motor torque from controller (N.m). Required for Phase 4 road load.",
    )
    vehicle_speed_ms: Optional[float] = Field(
        None,
        description="Vehicle speed (m/s). Required for Phase 4 aero drag.",
    )
    acceleration_ms2: Optional[float] = Field(
        None,
        description="Longitudinal acceleration from IMU (m/s^2). Required for Phase 4.",
    )
    grade_rad: Optional[float] = Field(
        None,
        description="Road grade (rad). Optional; omit if unavailable.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pressure_fl": 240,
                "pressure_fr": 238,
                "pressure_rl": 242,
                "pressure_rr": 240,
                "ratio_front": 1.003,
                "ratio_rear": 0.998,
                "temperature": 25,
            }
        }


class SimulationRequest(BaseModel):
    """Request a random ground-truth scenario and estimate it."""
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    iters: int = Field(6, ge=1, le=20, description="Estimator iterations")
    include_motor_torque: bool = Field(
        False,
        description="If True, generate motor torque measurement for Phase 4 test.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class StateEntry(BaseModel):
    """One estimated state variable with uncertainty."""
    name: str
    value: float
    sigma: float
    variance_reduction: float
    observability: str = "OBSERVED"


class Phase4Diagnostics(BaseModel):
    """Phase 4 observability diagnostics for the motor-torque channel."""
    road_load_available: bool = Field(
        False,
        description="True if independent motor torque measurement was used.",
    )
    motor_torque_used: Optional[float] = Field(
        None, description="Motor torque value used in estimate (N.m).",
    )
    acceleration_used: Optional[float] = Field(
        None, description="Acceleration value used (m/s^2).",
    )
    rolling_radius_used: Optional[float] = Field(
        None, description="Effective rolling radius used (m).",
    )
    toe_sensitivity_roadload: Optional[float] = Field(
        None, description="Jacobian d(road_load)/d(toe^2) from existing channel.",
    )
    toe_sensitivity_motor_torque: Optional[float] = Field(
        None, description="Jacobian d(motor_torque)/d(toe^2) from Phase 4 channel.",
    )
    toe_variance_prior: Optional[float] = Field(
        None, description="Prior toe^2 variance.",
    )
    toe_variance_posterior: Optional[float] = Field(
        None, description="Posterior toe^2 variance after estimation.",
    )
    toe_observability: Optional[str] = Field(
        None, description="Toe observability classification: UNOBSERVABLE/WEAK/OBSERVED.",
    )
    estimator_converged: bool = Field(False, description="Whether estimator converged.")
    iteration_count: int = Field(0, description="Iterations used.")
    n_measurements_available: int = Field(11, description="Number of measurement channels used.")


class EstimateResponse(BaseModel):
    """Full estimator output for a single measurement."""
    states: list[StateEntry]
    converged: bool
    iterations: int
    cold_pressure: dict[str, float] = Field(
        ...,
        description="Temperature-compensated pressures per corner (kPa)",
    )
    torque_ceiling: dict[str, float] = Field(
        ...,
        description="Wet and dry torque ceilings (N.m)",
    )
    recoverable_energy_pct: float = Field(
        ...,
        description="Excess road load vs ideal pressure and zero toe (%)",
    )
    phase4: Optional[Phase4Diagnostics] = Field(
        None,
        description="Phase 4 motor-torque observability diagnostics (None if not available).",
    )


class SimulationResponse(BaseModel):
    """Ground truth + estimate for a random scenario."""
    truth: dict[str, object]
    estimate: EstimateResponse


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "0.2.0"


class VehicleInfoResponse(BaseModel):
    """Static vehicle parameters."""
    mass_kg: float = 1800.0
    front_weight: float = 0.48
    placard_pressure_kPa: float = 240.0
    tread_new_mm: float = 8.0
    tread_legal_mm: float = 1.6


# ---------------------------------------------------------------------------
# Digital Twin (Phase 6)
# ---------------------------------------------------------------------------

class DigitalTwinRequest(BaseModel):
    """Run a full digital twin simulation with the Phase 6 mock adapter."""
    scenario: str = Field(
        "normal",
        description="Scenario type: normal, asymmetric, low_pressure, toe_misalign, accelerated, sensor_missing",
    )
    n_steps: int = Field(100, ge=10, le=500, description="Number of simulation steps")
    dt_s: float = Field(360.0, gt=0, description="Time step in seconds (default 6 min)")


class TyreFrame(BaseModel):
    """One tyre's state at a point in time."""
    tread_mm: float
    tread_sigma: float
    pressure_kpa: float
    pressure_sigma: float
    observability: str
    vr: float
    wear_pct: float = Field(0.0, description="Tread wear as percentage of usable range (0=new, 100=legal limit)")
    wear_mm: float = Field(0.0, description="Tread wear in mm from new (8.0mm)")
    wear_status: str = Field("new", description="Wear status: new, good, moderate, worn, critical")


class GroundTruthFrame(BaseModel):
    """Ground truth for one tyre."""
    tread_mm: float
    pressure_kpa: float
    wear_pct: float = 0.0
    wear_mm: float = 0.0


class SimulationFrame(BaseModel):
    """One time-step from the digital twin."""
    step: int
    timestamp_s: float
    odometer_km: float
    vehicle_speed_ms: float
    treads: dict[str, float]
    presses: dict[str, float]
    estimates: dict[str, TyreFrame]
    ground_truth: dict[str, GroundTruthFrame]
    toe_sq: float
    toe_observability: str
    converged: bool
    n_measurements: int


class DigitalTwinResponse(BaseModel):
    """Full digital twin simulation result."""
    scenario: str
    frames: list[SimulationFrame]
    total_distance_km: float
