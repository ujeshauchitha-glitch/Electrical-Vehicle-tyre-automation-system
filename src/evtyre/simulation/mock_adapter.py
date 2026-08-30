"""Mock vehicle adapter — deterministic synthetic telemetry without CARLA.

Generates physically-consistent telemetry from a hidden ground-truth
tyre state. The estimator receives only the observable telemetry;
ground truth is available for validation.

The forward model:
    Ground truth (tread, pressure, toe) → wheel speeds, pressure, motor torque
    → TelemetryFrame → Estimator → estimated state

Circularity check: motor torque is derived from the drivetrain model
(F_rr + F_toe + m*a + F_aero), NOT from the pressure-derived C_rr
used in the estimator's predict function. The estimator's forward model
re-predicts C_rr from its own state estimate, which is standard EKF
architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config.tyre import TyreConfig
from ..config.vehicle import DriveLayout, VehicleConfig
from ..estimation.physics import (
    corner_weight,
    effective_rolling_radius,
    rolling_resistance_coeff,
    toe_drag_from_sq,
)
from ..estimation.estimator import DEFAULT_PHYSICS, PhysicsConfig
from ..schema.common import CORNERS, SensorStatus
from .ground_truth import GroundTruthTyre, GroundTruthVehicle
from .interface import GroundTruthSnapshot, SimulationState, VehicleSimulator
from .scenarios import Scenario, ScenarioType, load_scenario


@dataclass
class MockVehicleConfig:
    """Mock simulation vehicle parameters (UNVALIDATED)."""

    mass_kg: float = 1800.0
    front_weight_fraction: float = 0.48
    drive_layout: DriveLayout = DriveLayout.RWD
    gear_ratio: float = 9.0
    driveline_efficiency: float = 0.95
    drag_coefficient: float = 0.25
    frontal_area_m2: float = 2.3
    ambient_temp_c: float = 25.0

    # Tyre defaults
    wheel_belt_radius_m: float = 0.322
    tread_new_mm: float = 8.0
    tread_legal_mm: float = 1.6
    placard_pressure_kpa: float = 240.0
    cold_reference_temperature_c: float = 25.0

    # Sensor noise
    pressure_noise_kpa: float = 2.0
    speed_noise_fraction: float = 0.005
    accel_noise_ms2: float = 0.1
    torque_noise_fraction: float = 0.03
    temperature_noise_c: float = 0.5


class MockVehicleAdapter(VehicleSimulator):
    """Deterministic mock vehicle simulator.

    Generates telemetry from a hidden ground-truth state.
    No CARLA required.
    """

    def __init__(
        self,
        mock_config: MockVehicleConfig | None = None,
        vehicle_config: VehicleConfig | None = None,
        tyre_config: TyreConfig | None = None,
        physics: PhysicsConfig | None = None,
    ) -> None:
        self._mock_config = mock_config or MockVehicleConfig()
        mc = self._mock_config

        self._vehicle_config = vehicle_config or VehicleConfig(
            vehicle_id="mock_ev",
            mass_kg=mc.mass_kg,
            front_weight_fraction=mc.front_weight_fraction,
            drive_layout=mc.drive_layout,
        )
        self._tyre_config = tyre_config or TyreConfig(
            tyre_model_id="mock_tyre",
            wheel_belt_radius_m=mc.wheel_belt_radius_m,
            tread_new_mm=mc.tread_new_mm,
            tread_legal_mm=mc.tread_legal_mm,
            placard_pressure_kpa=mc.placard_pressure_kpa,
            cold_reference_temperature_c=mc.cold_reference_temperature_c,
        )
        self._physics = physics or DEFAULT_PHYSICS

        self._tyres = GroundTruthTyre()
        self._vehicle = GroundTruthVehicle()
        self._scenario: Scenario | None = None
        self._step_count = 0
        self._distance_km = 0.0

    def reset(self, scenario: Scenario | None = None) -> None:
        """Reset to initial state with given scenario."""
        if scenario is None:
            scenario = load_scenario(ScenarioType.NORMAL)
        self._scenario = scenario
        self._tyres = scenario.tyre_state
        self._vehicle = GroundTruthVehicle()
        self._vehicle.speed_ms = 15.0
        self._vehicle.acceleration_ms2 = 0.0
        self._vehicle.odometer_km = 0.0
        self._vehicle.timestamp_s = 0.0
        self._step_count = 0
        self._distance_km = 0.0

    def step(self, dt_s: float) -> SimulationState:
        """Advance simulation by dt_s seconds."""
        self._step_count += 1
        dt_h = dt_s / 3600.0  # hours
        v = self._vehicle.speed_ms
        dist_step_km = v * dt_s / 1000.0
        self._distance_km += dist_step_km
        self._vehicle.odometer_km = self._distance_km
        self._vehicle.timestamp_s += dt_s

        # Apply degradation from scenario
        if self._scenario:
            self._scenario.apply_degradation(dist_step_km)

        # Motor speed from vehicle speed and gear ratio
        avg_r = sum(
            effective_rolling_radius(
                self._tyres.tread_mm[c],
                self._tyres.pressure_kpa[c],
                corner_weight(self._vehicle_config, c),
                self._tyre_config.wheel_belt_radius_m,
                self._physics.k_z0,
                self._tyre_config.placard_pressure_kpa,
                self._physics.deflection_factor,
            )
            for c in CORNERS
        ) / 4.0
        omega_wheel = v / max(avg_r, 0.01)
        motor_speed = omega_wheel * self._mock_config.gear_ratio

        # Compute motor torque from ground truth
        torque_nm = self._compute_motor_torque()

        # Compute wheel speeds from ground truth
        wheel_speeds = {}
        for corner in CORNERS:
            r_eff = effective_rolling_radius(
                self._tyres.tread_mm[corner],
                self._tyres.pressure_kpa[corner],
                corner_weight(self._vehicle_config, corner),
                self._tyre_config.wheel_belt_radius_m,
                self._physics.k_z0,
                self._tyre_config.placard_pressure_kpa,
                self._physics.deflection_factor,
            )
            # Add small noise
            noise = 1.0 + self._mock_config.speed_noise_fraction * self._noise_tick()
            wheel_speeds[corner] = (v * noise) / max(r_eff, 0.01)

        # Compute pressures (TPMS readings)
        pressures = {}
        for corner in CORNERS:
            noise = self._mock_config.pressure_noise_kpa * self._noise_tick()
            pressures[corner] = self._tyres.pressure_kpa[corner] + noise

        # Compute temperatures
        temps = {}
        for corner in CORNERS:
            base_temp = self._mock_config.ambient_temp_c + 10.0
            load_heating = corner_weight(self._vehicle_config, corner) / 8000.0 * 5.0
            noise = self._mock_config.temperature_noise_c * self._noise_tick()
            temps[corner] = base_temp + load_heating + noise

        # Build TelemetryFrame
        missing = self._scenario.missing_channels if self._scenario else set()

        # Check which channels are missing
        mt = None if "motor_torque" in missing else torque_nm
        ms = None if "motor_speed" in missing else motor_speed

        frame = self._make_frame(
            timestamp_s=self._vehicle.timestamp_s,
            wheel_speeds=wheel_speeds,
            pressures_kpa=pressures,
            temps_c=temps,
            motor_torque=mt,
            motor_speed=ms,
            accel=self._vehicle.acceleration_ms2,
            ambient_temp=self._mock_config.ambient_temp_c,
            vehicle_speed=v,
            odometer_km=self._vehicle.odometer_km,
            missing_channels=missing,
        )

        return SimulationState(
            telemetry=frame,
            ground_truth=self._tyres.as_ground_truth_snapshot(),
            timestamp_s=self._vehicle.timestamp_s,
            odometer_km=self._vehicle.odometer_km,
            vehicle_speed_ms=v,
            motor_torque_nm=torque_nm,
            step_index=self._step_count,
        )

    def get_ground_truth(self) -> GroundTruthSnapshot:
        return self._tyres.as_ground_truth_snapshot()

    @property
    def vehicle_config(self) -> VehicleConfig:
        return self._vehicle_config

    @property
    def tyre_config(self) -> TyreConfig:
        return self._tyre_config

    @property
    def current_odometer_km(self) -> float:
        return self._distance_km

    def _compute_motor_torque(self) -> float:
        """Compute motor torque from drivetrain model.

        This is the OBSERVABLE measurement, not the estimator's prediction.
        The estimator re-predicts this from its own state estimate.
        """
        g = 9.80665
        v = self._vehicle.speed_ms
        mc = self._mock_config

        # Total rolling resistance
        total_frr = 0.0
        for corner in CORNERS:
            c_rr = rolling_resistance_coeff(
                self._tyres.tread_mm[corner],
                self._tyres.pressure_kpa[corner],
                self._tyres.temperature_c.get(corner, 35.0),
                self._tyre_config,
                self._physics,
            )
            fz = corner_weight(self._vehicle_config, corner)
            total_frr += c_rr * fz

        # Toe drag
        f_toe = toe_drag_from_sq(self._tyres.toe_sq_deg2, self._physics)

        # Inertial force
        f_inertia = self._vehicle_config.mass_kg * self._vehicle.acceleration_ms2

        # Aerodynamic drag
        cda = mc.drag_coefficient * mc.frontal_area_m2
        rho = 1.225
        f_aero = 0.5 * rho * cda * v * v

        f_traction = total_frr + f_toe + f_inertia + f_aero

        return f_traction * max(self._tyre_config.wheel_belt_radius_m, 0.01) / (
            mc.gear_ratio * mc.driveline_efficiency
        )

    def _noise_tick(self) -> float:
        """Simple deterministic pseudo-noise (not truly random, but varies per step)."""
        import math
        return math.sin(self._step_count * 12.9898 + 78.233) * 0.5
