"""Deterministic scenario generator for digital twin simulation.

Each scenario defines a hidden ground-truth tyre state and optional
time-varying evolution. The estimator never sees these definitions —
it only receives the resulting telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .ground_truth import GroundTruthTyre, GroundTruthVehicle


class ScenarioType(str, Enum):
    """Pre-defined simulation scenarios."""

    NORMAL = "normal"
    UNIFORM_WEAR = "uniform_wear"
    ASYMMETRIC_WEAR = "asymmetric_wear"
    LOW_PRESSURE = "low_pressure"
    PRESSURE_DRIFT = "pressure_drift"
    TOE_MISALIGNMENT = "toe_misalignment"
    SENSOR_MISSINGNESS = "sensor_missingness"
    ACCELERATED_DEGRADATION = "accelerated_degradation"


@dataclass
class DegradationProfile:
    """Defines how ground truth changes over distance.

    Rates are per 1000 km unless otherwise noted.
    """

    tread_wear_rate_mm_per_1000km: dict[str, float] = field(
        default_factory=lambda: {"FL": 0.03, "FR": 0.03, "RL": 0.04, "RR": 0.05}
    )
    pressure_drift_rate_kpa_per_month: dict[str, float] = field(
        default_factory=lambda: {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}
    )
    temperature_drift_c_per_1000km: float = 0.0


@dataclass
class Scenario:
    """A complete simulation scenario definition."""

    name: str
    description: str
    tyre_state: GroundTruthTyre
    vehicle_state: GroundTruthVehicle = field(default_factory=GroundTruthVehicle)
    degradation: DegradationProfile = field(default_factory=DegradationProfile)
    missing_channels: set[str] = field(default_factory=set)
    total_distance_km: float = 500.0
    """Total simulated distance in km."""

    def apply_degradation(self, distance_km: float) -> None:
        """Evolve ground truth by driving distance_km."""
        for corner in ["FL", "FR", "RL", "RR"]:
            wear = self.degradation.tread_wear_rate_mm_per_1000km.get(corner, 0.0)
            self.tyre_state.tread_mm[corner] = max(
                0.5,
                self.tyre_state.tread_mm[corner] - wear * distance_km / 1000.0,
            )
            drift = self.degradation.pressure_drift_rate_kpa_per_month.get(corner, 0.0)
            # Convert to per-km (assume ~1000 km/month)
            self.tyre_state.pressure_kpa[corner] = max(
                50.0,
                self.tyre_state.pressure_kpa[corner] - drift * distance_km / 1000.0,
            )


def load_scenario(scenario_type: ScenarioType) -> Scenario:
    """Create a pre-defined scenario."""

    if scenario_type == ScenarioType.NORMAL:
        return Scenario(
            name="Normal",
            description="All tyres healthy and symmetric",
            tyre_state=GroundTruthTyre(
                tread_mm={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
                pressure_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
                toe_sq_deg2=0.0,
            ),
            degradation=DegradationProfile(
                tread_wear_rate_mm_per_1000km={
                    "FL": 0.03, "FR": 0.03, "RL": 0.04, "RR": 0.05,
                },
            ),
        )

    elif scenario_type == ScenarioType.UNIFORM_WEAR:
        return Scenario(
            name="Uniform Wear",
            description="All tyres wearing at the same rate",
            tyre_state=GroundTruthTyre(
                tread_mm={"FL": 4.5, "FR": 4.5, "RL": 4.5, "RR": 4.5},
                pressure_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
                toe_sq_deg2=0.0,
            ),
            degradation=DegradationProfile(
                tread_wear_rate_mm_per_1000km={
                    "FL": 0.05, "FR": 0.05, "RL": 0.05, "RR": 0.05,
                },
            ),
        )

    elif scenario_type == ScenarioType.ASYMMETRIC_WEAR:
        return Scenario(
            name="Asymmetric Wear",
            description="RR tyre wearing faster (suspension issue)",
            tyre_state=GroundTruthTyre(
                tread_mm={"FL": 4.8, "FR": 4.8, "RL": 4.5, "RR": 3.7},
                pressure_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
                toe_sq_deg2=0.0,
            ),
            degradation=DegradationProfile(
                tread_wear_rate_mm_per_1000km={
                    "FL": 0.02, "FR": 0.02, "RL": 0.03, "RR": 0.08,
                },
            ),
        )

    elif scenario_type == ScenarioType.LOW_PRESSURE:
        return Scenario(
            name="Low Pressure",
            description="RR tyre has a slow leak",
            tyre_state=GroundTruthTyre(
                tread_mm={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
                pressure_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 180.0},
                toe_sq_deg2=0.0,
            ),
            degradation=DegradationProfile(
                pressure_drift_rate_kpa_per_month={
                    "FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": -3.0,
                },
            ),
        )

    elif scenario_type == ScenarioType.PRESSURE_DRIFT:
        return Scenario(
            name="Pressure Drift",
            description="All tyres losing pressure slowly",
            tyre_state=GroundTruthTyre(
                tread_mm={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
                pressure_kpa={"FL": 230.0, "FR": 230.0, "RL": 230.0, "RR": 230.0},
                toe_sq_deg2=0.0,
            ),
            degradation=DegradationProfile(
                pressure_drift_rate_kpa_per_month={
                    "FL": -1.0, "FR": -1.0, "RL": -1.5, "RR": -2.0,
                },
            ),
        )

    elif scenario_type == ScenarioType.TOE_MISALIGNMENT:
        return Scenario(
            name="Toe Misalignment",
            description="Front axle has excessive toe-out",
            tyre_state=GroundTruthTyre(
                tread_mm={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
                pressure_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
                toe_sq_deg2=1.0,  # |toe| = 1.0 deg
            ),
            degradation=DegradationProfile(
                tread_wear_rate_mm_per_1000km={
                    "FL": 0.06, "FR": 0.06, "RL": 0.03, "RR": 0.03,
                },
            ),
        )

    elif scenario_type == ScenarioType.SENSOR_MISSINGNESS:
        return Scenario(
            name="Sensor Missing",
            description="Motor torque and temperature sensors unavailable",
            tyre_state=GroundTruthTyre(
                tread_mm={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
                pressure_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
                toe_sq_deg2=0.0,
            ),
            missing_channels={"motor_torque", "tpms_temperature"},
        )

    elif scenario_type == ScenarioType.ACCELERATED_DEGRADATION:
        return Scenario(
            name="Accelerated Degradation",
            description="All tyres wearing fast (aggressive driving)",
            tyre_state=GroundTruthTyre(
                tread_mm={"FL": 4.0, "FR": 4.0, "RL": 4.0, "RR": 4.0},
                pressure_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
                toe_sq_deg2=0.0,
            ),
            degradation=DegradationProfile(
                tread_wear_rate_mm_per_1000km={
                    "FL": 0.10, "FR": 0.10, "RL": 0.12, "RR": 0.12,
                },
            ),
            total_distance_km=300.0,
        )

    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")
