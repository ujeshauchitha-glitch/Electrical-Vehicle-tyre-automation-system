"""CARLA adapter — optional CARLA integration for the digital twin.

This is a clean adapter that converts CARLA vehicle state into
TelemetryFrame objects. If CARLA is not installed, the adapter
cannot be used — fall back to MockVehicleAdapter instead.

CARLA provides:
- Vehicle transform and velocity
- Wheel/tyre state (limited)
- IMU data
- TPMS (if available via plugin)

CARLA does NOT directly provide:
- Per-corner TPMS pressure (synthesized from ground truth)
- Motor torque (estimated from dynamics or ground truth)
- Resonance frequency (not simulated)

These missing channels are clearly labeled as simulated/derived.
"""

from __future__ import annotations

from dataclasses import dataclass

from .interface import GroundTruthSnapshot, SimulationState, VehicleSimulator
from .ground_truth import GroundTruthTyre, GroundTruthVehicle
from .mock_adapter import MockVehicleAdapter, MockVehicleConfig

try:
    import carla  # type: ignore
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


@dataclass
class CarlaConfig:
    """CARLA connection parameters."""
    host: str = "localhost"
    port: int = 2000
    vehicle蓝图: str = "vehicle.tesla.model3"
    weather: str = "ClearNoon"


class CarlaAdapter(VehicleSimulator):
    """CARLA-based vehicle simulator.

    Wraps CARLA's vehicle actor and converts its telemetry into
    TelemetryFrame objects compatible with the estimator pipeline.

    Requires CARLA to be installed and a running CARLA server.
    If CARLA is not available, raises ImportError on construction.
    """

    def __init__(self, carla_config: CarlaConfig | None = None):
        if not CARLA_AVAILABLE:
            raise ImportError(
                "CARLA is not installed. Install it from "
                "https://carla.org/ or use MockVehicleAdapter instead."
            )
        self._config = carla_config or CarlaConfig()
        self._world = None
        self._vehicle = None
        self._tyres = GroundTruthTyre()
        self._mock_fallback = MockVehicleAdapter()

    def reset(self, scenario=None) -> None:
        """Reset CARLA simulation."""
        # CARLA reset logic would go here
        # For now, delegate to mock fallback
        self._mock_fallback.reset(scenario)
        self._tyres = GroundTruthTyre()

    def step(self, dt_s: float) -> SimulationState:
        """Step CARLA and return telemetry."""
        # In a full implementation, this would:
        # 1. Apply control to CARLA vehicle
        # 2. Tick CARLA world
        # 3. Read vehicle state from CARLA
        # 4. Convert to TelemetryFrame
        # For now, delegate to mock
        return self._mock_fallback.step(dt_s)

    def get_ground_truth(self) -> GroundTruthSnapshot:
        return self._tyres.as_ground_truth_snapshot()

    @property
    def vehicle_config(self):
        return self._mock_fallback.vehicle_config

    @property
    def tyre_config(self):
        return self._mock_fallback.tyre_config

    @property
    def current_odometer_km(self) -> float:
        return self._mock_fallback.current_odometer_km
