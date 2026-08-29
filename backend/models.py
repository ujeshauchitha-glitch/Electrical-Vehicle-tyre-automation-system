"""
Pydantic models for API request/response validation.
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


class SimulationResponse(BaseModel):
    """Ground truth + estimate for a random scenario."""
    truth: dict[str, object]
    estimate: EstimateResponse


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "0.1.0"


class VehicleInfoResponse(BaseModel):
    """Static vehicle parameters."""
    mass_kg: float = 1800.0
    front_weight: float = 0.48
    placard_pressure_kPa: float = 240.0
    tread_new_mm: float = 8.0
    tread_legal_mm: float = 1.6
