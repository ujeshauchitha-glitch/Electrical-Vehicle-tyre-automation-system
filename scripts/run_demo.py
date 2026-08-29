"""End-to-end demo: TelemetryFrame → extractors → estimator.

Every number printed comes from an actual run.  No hardcoded example output.

Usage:
    python scripts/run_demo.py
"""

from __future__ import annotations

import sys
import os

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.features import pressure_thermal, kinematics, resonance
from evtyre.features.contract import FeatureStatus
from evtyre.pipeline import Pipeline
from evtyre.schema.common import CORNERS, SensorReading, SensorStatus
from evtyre.schema.telemetry import TelemetryFrame


def make_demo_frame() -> TelemetryFrame:
    """Create a realistic-ish TelemetryFrame for the demo."""
    return TelemetryFrame(
        timestamp_s=10.0,
        source="simulated",
        wheel_speed_rad_s={
            "FL": SensorReading(68.0, SensorStatus.OK),
            "FR": SensorReading(68.2, SensorStatus.OK),
            "RL": SensorReading(72.0, SensorStatus.OK),
            "RR": SensorReading(71.8, SensorStatus.OK),
        },
        tpms_pressure_kpa={
            "FL": SensorReading(238.0, SensorStatus.OK),
            "FR": SensorReading(241.0, SensorStatus.OK),
            "RL": SensorReading(232.0, SensorStatus.OK),
            "RR": SensorReading(245.0, SensorStatus.OK),
        },
        tpms_temperature_c={
            "FL": SensorReading(28.0, SensorStatus.OK),
            "FR": SensorReading(29.0, SensorStatus.OK),
            "RL": SensorReading(35.0, SensorStatus.OK),
            "RR": SensorReading(31.0, SensorStatus.OK),
        },
        motor_torque_nm=SensorReading(180.0, SensorStatus.OK),
        motor_speed_rad_s=SensorReading(250.0, SensorStatus.OK),
        accel_long_ms2=SensorReading(0.3, SensorStatus.OK),
        ambient_temp_c=SensorReading(22.0, SensorStatus.OK),
        vehicle_speed_ms=SensorReading(25.0, SensorStatus.OK),
        odometer_km=SensorReading(15200.0, SensorStatus.OK),
    )


def main() -> None:
    print("=" * 70)
    print("  EV Tyre Fusion — End-to-End Demo")
    print("=" * 70)
    print()

    # --- Configuration ---
    vehicle = VehicleConfig(
        vehicle_id="demo-vehicle-001",
        mass_kg=1850.0,
        front_weight_fraction=0.47,
        drive_layout=DriveLayout.RWD,
    )
    tyre = TyreConfig(
        tyre_model_id="demo-tyre-225-45R18",
        wheel_belt_radius_m=0.333,
        tread_new_mm=7.5,
        tread_legal_mm=1.6,
        placard_pressure_kpa=240.0,
        cold_reference_temperature_c=25.0,
    )

    print("Vehicle Configuration:")
    print(f"  ID:                    {vehicle.vehicle_id}")
    print(f"  Mass:                  {vehicle.mass_kg} kg")
    print(f"  Front weight fraction: {vehicle.front_weight_fraction}")
    print(f"  Drive layout:          {vehicle.drive_layout.value}")
    print()
    print("Tyre Configuration:")
    print(f"  Model:                 {tyre.tyre_model_id}")
    print(f"  Belt radius:           {tyre.wheel_belt_radius_m} m")
    print(f"  Tread new:             {tyre.tread_new_mm} mm")
    print(f"  Tread legal limit:     {tyre.tread_legal_mm} mm")
    print(f"  Placard pressure:      {tyre.placard_pressure_kpa} kPa")
    print()

    # --- Build pipeline ---
    pipe = Pipeline(vehicle, tyre)
    pipe.register("pressure_thermal", pressure_thermal.extract)
    pipe.register("kinematics", kinematics.extract)
    pipe.register("resonance", resonance.extract)  # will raise — caught by pipeline

    # --- Create frame ---
    frame = make_demo_frame()
    print("Input TelemetryFrame:")
    print(f"  Timestamp:   {frame.timestamp_s} s")
    print(f"  Source:      {frame.source}")
    print(f"  Vehicle speed: {frame.vehicle_speed_ms.value} m/s "
          f"({frame.vehicle_speed_ms.value * 3.6:.1f} km/h)")
    print(f"  Motor torque:  {frame.motor_torque_nm.value} Nm")
    for corner in CORNERS:
        p = frame.tpms_pressure_kpa[corner].value
        t = frame.tpms_temperature_c[corner].value
        ws = frame.wheel_speed_rad_s[corner].value
        print(f"  {corner}: P={p:.1f} kPa, T={t:.1f} C, w={ws:.1f} rad/s")
    print()

    # --- Run pipeline ---
    features, result = pipe.run(frame)

    # --- Summary ---
    ok_features = [f for f in features if f.status == FeatureStatus.OK]
    unavail_features = [f for f in features if f.status == FeatureStatus.UNAVAILABLE]

    print(f"Features produced:   {len(ok_features)}")
    print(f"Features unavailable: {len(unavail_features)}")
    print()

    # --- Feature summary ---
    print("Key Features (OK):")
    print("-" * 50)
    for f in sorted(ok_features, key=lambda x: x.name):
        if f.corner is not None:
            label = f"{f.name} [{f.corner}]"
        else:
            label = f.name
        unit_str = f" {f.unit}" if f.unit else ""
        print(f"  {label:<45} {f.value:>12.4f}{unit_str}  ({f.classification.value})")
    print()

    if unavail_features:
        print("Unavailable Features:")
        print("-" * 50)
        for f in sorted(unavail_features, key=lambda x: x.name):
            reason = (f.unavailable_reason or "unknown")[:60]
            print(f"  {f.name:<45} {reason}")
        print()

    # --- Estimator output ---
    print("Estimator Output:")
    print("-" * 50)
    for s in result.states:
        obs_tag = s.observability.value.upper()
        if s.magnitude_only:
            obs_tag += " [magnitude-only]"
        if s.reason:
            print(f"  {s.name:<15} {s.value:>10.4f}  +/- {s.sigma:.4f}  {obs_tag}")
            print(f"    {s.reason}")
        else:
            print(f"  {s.name:<15} {s.value:>10.4f}  +/- {s.sigma:.4f}  {obs_tag}")
    print()
    print(f"  States observed:  {result.n_states_observed}/{len(result.states)}")
    print(f"  Mean var. reduction: {result.mean_variance_reduction:.4f}"
          f"   (over OBSERVED states only)")
    print(f"  Measurements:     {result.n_measurements_available} channels admitted")
    if result.singular_matrix:
        print("  WARNING: Singular update matrix")

    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
