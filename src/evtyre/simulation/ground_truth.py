"""Ground-truth tyre and vehicle models for simulation.

These define the HIDDEN state the simulator tracks. The estimator never
sees these directly — only the observable telemetry derived from them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config.tyre import TyreConfig
from ..config.vehicle import DriveLayout, VehicleConfig
from ..schema.common import CORNERS


@dataclass
class GroundTruthTyre:
    """Hidden tyre state — the actual physical condition."""

    tread_mm: dict[str, float] = field(default_factory=lambda: {
        "FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0,
    })
    pressure_kpa: dict[str, float] = field(default_factory=lambda: {
        "FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0,
    })
    toe_sq_deg2: float = 0.0
    camber_deg: float = 0.0
    temperature_c: dict[str, float] = field(default_factory=lambda: {
        "FL": 35.0, "FR": 35.0, "RL": 35.0, "RR": 35.0,
    })

    def apply_wear(self, corner: str, delta_mm: float) -> None:
        """Apply tread wear to one corner (negative = wear)."""
        self.tread_mm[corner] = max(0.5, self.tread_mm[corner] + delta_mm)

    def apply_pressure_change(self, corner: str, delta_kpa: float) -> None:
        """Apply pressure change to one corner."""
        self.pressure_kpa[corner] = max(50.0, self.pressure_kpa[corner] + delta_kpa)

    def as_ground_truth_snapshot(self):
        """Convert to GroundTruthSnapshot for the interface."""
        from .interface import GroundTruthSnapshot
        return GroundTruthSnapshot(
            tread_mm=dict(self.tread_mm),
            pressure_kpa=dict(self.pressure_kpa),
            toe_sq_deg2=self.toe_sq_deg2,
            camber_deg=self.camber_deg,
            temperature_c=dict(self.temperature_c),
        )


@dataclass
class GroundTruthVehicle:
    """Vehicle dynamics state for simulation."""

    speed_ms: float = 15.0  # m/s (~54 km/h)
    acceleration_ms2: float = 0.0
    odometer_km: float = 0.0
    timestamp_s: float = 0.0

    # Drivetrain
    gear_ratio: float = 9.0  # single-speed EV
    driveline_efficiency: float = 0.95
    motor_speed_rad_s: float = 0.0  # computed from speed

    def compute_wheel_speeds(
        self,
        tyre_config: TyreConfig,
        tyres: GroundTruthTyre,
    ) -> dict[str, float]:
        """Compute wheel angular velocities from vehicle speed and tyre state.

        omega = v / r_eff, where r_eff depends on tread and pressure.
        """
        from ..estimation.physics import effective_rolling_radius, corner_weight

        wheel_speeds = {}
        for corner in CORNERS:
            r_eff = effective_rolling_radius(
                tyres.tread_mm[corner],
                tyres.pressure_kpa[corner],
                corner_weight(self._vehicle_config(), corner),
                tyre_config.wheel_belt_radius_m,
                210_000.0,  # k_z0 — UNVALIDATED
                tyre_config.placard_pressure_kpa,
                1.0 / 3.0,  # deflection_factor
            )
            wheel_speeds[corner] = self.speed_ms / max(r_eff, 0.01)
        return wheel_speeds

    def compute_motor_torque(
        self,
        tyres: GroundTruthTyre,
        vehicle_config: VehicleConfig,
        tyre_config: TyreConfig,
    ) -> float:
        """Compute motor torque from drivetrain model.

        T_motor = (F_rr + F_toe + m*a) * r_eff / (gear_ratio * efficiency)
        """
        from ..estimation.physics import (
            corner_weight,
            effective_rolling_radius,
            rolling_resistance_coeff,
            toe_drag_from_sq,
        )

        g = 9.80665
        total_frr = 0.0
        for corner in CORNERS:
            c_rr = rolling_resistance_coeff(
                tyres.tread_mm[corner],
                tyres.pressure_kpa[corner],
                tyres.temperature_c.get(corner, 30.0),
                tyre_config,
                self._physics_config(),
            )
            fz = corner_weight(vehicle_config, corner)
            total_frr += c_rr * fz

        f_toe = toe_drag_from_sq(tyres.toe_sq_deg2, self._physics_config())
        f_inertia = vehicle_config.mass_kg * self.acceleration_ms2

        # Approximate aero drag
        cda = 0.25 * 2.3  # UNVALIDATED
        rho = 1.225
        f_aero = 0.5 * rho * cda * self.speed_ms ** 2

        f_traction = total_frr + f_toe + f_inertia + f_aero

        # Average rolling radius
        avg_r = sum(
            effective_rolling_radius(
                tyres.tread_mm[c], tyres.pressure_kpa[c],
                corner_weight(vehicle_config, c),
                tyre_config.wheel_belt_radius_m,
                210_000.0, tyre_config.placard_pressure_kpa, 1.0 / 3.0,
            )
            for c in CORNERS
        ) / 4.0

        return f_traction * avg_r / (self.gear_ratio * self.driveline_efficiency)

    def _vehicle_config(self) -> VehicleConfig:
        return VehicleConfig(
            vehicle_id="sim_ev",
            mass_kg=1800.0,
            front_weight_fraction=0.48,
            drive_layout=DriveLayout.RWD,
        )

    def _physics_config(self):
        from ..estimation.estimator import DEFAULT_PHYSICS
        return DEFAULT_PHYSICS
