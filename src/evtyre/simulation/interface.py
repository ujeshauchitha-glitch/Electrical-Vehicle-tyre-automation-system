"""Simulation interface: the contract between vehicle simulators and the estimator.

A VehicleSimulator produces SimulationState objects at each time step.
SimulationState contains a TelemetryFrame (what the estimator sees) plus
optional ground-truth data (for validation only, never consumed by the estimator).

Both CARLA and Mock adapters implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping

from ..config.tyre import TyreConfig
from ..config.vehicle import VehicleConfig
from ..schema.common import CORNERS, SensorReading, SensorStatus
from ..schema.telemetry import TelemetryFrame


@dataclass(frozen=True)
class GroundTruthSnapshot:
    """Hidden ground-truth tyre state at one instant.

    This is NEVER passed to the estimator. It exists solely for
    validation: comparing estimated state vs actual state.
    """

    tread_mm: Mapping[str, float]  # per corner, keyed by CORNERS
    pressure_kpa: Mapping[str, float]  # per corner, gauge
    toe_sq_deg2: float = 0.0
    camber_deg: float = 0.0
    temperature_c: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_asymmetric(self) -> bool:
        """True if any corner differs from any other by more than 0.1 mm."""
        treads = list(self.tread_mm.values())
        return max(treads) - min(treads) > 0.1


@dataclass(frozen=True)
class SimulationState:
    """One time-step from a vehicle simulator.

    Contains:
    - telemetry: the TelemetryFrame the estimator will consume
    - ground_truth: the actual tyre state (validation only)
    - metadata: time, distance, speed for replay and display
    """

    telemetry: TelemetryFrame
    ground_truth: GroundTruthSnapshot
    timestamp_s: float
    odometer_km: float
    vehicle_speed_ms: float
    motor_torque_nm: float | None = None
    step_index: int = 0


class VehicleSimulator(ABC):
    """Abstract interface for a vehicle simulator.

    Concrete implementations:
    - MockVehicleAdapter: deterministic synthetic telemetry
    - CarlaAdapter: CARLA-based simulation (optional dependency)
    """

    @abstractmethod
    def reset(self, scenario: object | None = None) -> None:
        """Reset the simulator to initial state.

        Parameters
        ----------
        scenario : object, optional
            A Scenario object defining initial conditions.
        """

    @abstractmethod
    def step(self, dt_s: float) -> SimulationState:
        """Advance simulation by dt_s seconds.

        Returns a SimulationState containing the TelemetryFrame
        the estimator should consume, plus ground truth for validation.
        """

    @abstractmethod
    def get_ground_truth(self) -> GroundTruthSnapshot:
        """Return the current hidden ground-truth state."""

    @property
    @abstractmethod
    def vehicle_config(self) -> VehicleConfig:
        """Return the vehicle configuration."""

    @property
    @abstractmethod
    def tyre_config(self) -> TyreConfig:
        """Return the tyre configuration."""

    @property
    @abstractmethod
    def current_odometer_km(self) -> float:
        """Return current odometer reading."""

    def _make_sensor(
        self,
        value: float | None,
        status: SensorStatus = SensorStatus.OK,
    ) -> SensorReading:
        """Helper to build a SensorReading."""
        if status is SensorStatus.MISSING:
            return SensorReading.missing()
        return SensorReading(value=value, status=status)

    def _make_frame(
        self,
        timestamp_s: float,
        wheel_speeds: dict[str, float | None],
        pressures_kpa: dict[str, float | None],
        temps_c: dict[str, float | None],
        motor_torque: float | None,
        motor_speed: float | None,
        accel: float | None,
        ambient_temp: float,
        vehicle_speed: float,
        odometer_km: float,
        missing_channels: set[str] | None = None,
    ) -> TelemetryFrame:
        """Build a TelemetryFrame from raw values.

        Missing channels are explicitly marked, never silently zeroed.
        """
        missing = missing_channels or set()

        def _sr(channel: str, val: float | None) -> SensorReading:
            if channel in missing or val is None:
                return SensorReading.missing()
            return SensorReading(value=val, status=SensorStatus.OK)

        wheel_speed_map = {c: _sr("wheel_speed", wheel_speeds.get(c)) for c in CORNERS}
        press_map = {c: _sr("tpms_pressure", pressures_kpa.get(c)) for c in CORNERS}
        temp_map = {c: _sr("tpms_temperature", temps_c.get(c)) for c in CORNERS}

        return TelemetryFrame(
            timestamp_s=timestamp_s,
            source="simulated",
            wheel_speed_rad_s=wheel_speed_map,
            tpms_pressure_kpa=press_map,
            tpms_temperature_c=temp_map,
            motor_torque_nm=_sr("motor_torque", motor_torque),
            motor_speed_rad_s=_sr("motor_speed", motor_speed),
            accel_long_ms2=_sr("accel", accel),
            ambient_temp_c=_sr("ambient", ambient_temp),
            vehicle_speed_ms=_sr("vehicle_speed", vehicle_speed),
            odometer_km=_sr("odometer", odometer_km),
        )
