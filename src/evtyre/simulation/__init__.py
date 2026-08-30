"""Phase 6 — EV Tyre Digital Twin / Simulation Integration.

Provides a vehicle simulator interface that feeds observable telemetry
into the existing estimator pipeline. The simulator generates measurements
from a hidden ground-truth tyre state; the estimator never sees ground
truth directly.

Architecture:
    GroundTruth → VehicleSimulator → TelemetryFrame → Existing Pipeline → Estimator
                                                                    ↓
                                                              Dashboard

CARLA and Mock adapters conform to the same VehicleSimulator interface,
so the downstream estimator is adapter-agnostic.
"""

from .interface import VehicleSimulator, SimulationState
from .ground_truth import GroundTruthTyre, GroundTruthVehicle
from .scenarios import Scenario, ScenarioType, load_scenario
from .mock_adapter import MockVehicleAdapter
from .carla_adapter import CarlaAdapter, CARLA_AVAILABLE
from .replay import ReplayEngine, ReplayFrame

__all__ = [
    "VehicleSimulator",
    "SimulationState",
    "GroundTruthTyre",
    "GroundTruthVehicle",
    "Scenario",
    "ScenarioType",
    "load_scenario",
    "MockVehicleAdapter",
    "CarlaAdapter",
    "CARLA_AVAILABLE",
    "ReplayEngine",
    "ReplayFrame",
]
