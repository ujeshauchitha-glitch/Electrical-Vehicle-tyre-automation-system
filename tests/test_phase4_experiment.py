"""
Phase 4 Definitive Experiment: Motor-Torque Road-Load Observability Comparison

Runs the exact BEFORE vs AFTER experiment requested in the specification:
  Case A: Existing measurements only (pressure, freq, wheel-speed ratios, road_load)
  Case B: Same telemetry PLUS motor-torque-derived road load

Reports: Jacobian, toe observability, toe covariance, variance reduction, convergence.
"""

import numpy as np

from backend.estimator import (
    CORNERS,
    IDX_TOESQ,
    M_MOTORTORQUE,
    M_ROADLOAD,
    N_MEAS,
    N_STATE,
    STATE_NAMES,
    TyreState,
    estimate,
    jacobian,
    measure,
    measurement_covariance,
    observability_analysis,
    prior,
)
from backend.physics import (
    DrivetrainConfig,
    Vehicle,
    motor_torque_measurement,
    toe_drag_from_sq,
)


def run_experiment():
    """Run the definitive BEFORE vs AFTER observability experiment."""

    # ── Fixed scenario ──────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    truth = TyreState(
        tread={"FL": 5.0, "FR": 5.2, "RL": 4.0, "RR": 4.1},
        pressure={"FL": 235.0, "FR": 238.0, "RL": 242.0, "RR": 240.0},
        temp={"FL": 28.0, "FR": 29.0, "RL": 32.0, "RR": 31.0},
        toe=0.6,
        camber=0.3,
    )
    v_ms = 22.0
    accel_ms2 = 1.0
    grade_rad = 0.0

    # ── CASE A: Without motor torque (Phase 1-3 only) ──────────────────
    rng_a = np.random.default_rng(42)
    z_a, T_meas_a = measure(truth, rng_a, v_ms=v_ms, accel_ms2=accel_ms2,
                             include_motor_torque=False)

    x_a, P_a = estimate(z_a, T_meas_a, v_ms=v_ms, accel_ms2=accel_ms2)
    sigma_a = np.sqrt(np.diag(P_a))
    obs_a = observability_analysis(x_a, P_a, z_a, T_meas_a, v_ms, accel_ms2, grade_rad)
    H_a = jacobian(x_a, T_meas_a, v_ms, accel_ms2, grade_rad)

    # ── CASE B: With motor torque (Phase 4) ────────────────────────────
    rng_b = np.random.default_rng(42)
    z_b, T_meas_b = measure(truth, rng_b, v_ms=v_ms, accel_ms2=accel_ms2,
                             include_motor_torque=True)

    x_b, P_b = estimate(z_b, T_meas_b, v_ms=v_ms, accel_ms2=accel_ms2)
    sigma_b = np.sqrt(np.diag(P_b))
    obs_b = observability_analysis(x_b, P_b, z_b, T_meas_b, v_ms, accel_ms2, grade_rad)
    H_b = jacobian(x_b, T_meas_b, v_ms, accel_ms2, grade_rad)

    # ── PRINT RESULTS ──────────────────────────────────────────────────
    print("=" * 72)
    print("  PHASE 4 DEFINITIVE EXPERIMENT: MOTOR-TORQUE ROAD-LOAD OBSERVABILITY")
    print("=" * 72)

    print("\n── Ground Truth ──")
    print(f"  toe (true):     {truth.toe:.4f} deg")
    print(f"  toe^2 (true):   {truth.toe**2:.6f} deg^2")
    print(f"  tread (true):   {dict((c, f'{truth.tread[c]:.1f}') for c in CORNERS)}")
    print(f"  pressure (true):{dict((c, f'{truth.pressure[c]:.1f}') for c in CORNERS)}")

    print("\n── Measurement Vector ──")
    labels = [f"press_{c}" for c in CORNERS] + [f"freq_{c}" for c in CORNERS] + \
             ["ratio_front", "ratio_rear", "road_load", "motor_torque"]
    print(f"  {'Channel':<20} {'CASE A':>12} {'CASE B':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    for i, label in enumerate(labels):
        val_a = z_a[i] if not np.isnan(z_a[i]) else float('nan')
        val_b = z_b[i] if not np.isnan(z_b[i]) else float('nan')
        print(f"  {label:<20} {val_a:>12.6f} {val_b:>12.6f}")

    print(f"\n── State Estimation ──")
    print(f"  {'State':<12} {'True':>8} {'CASE A':>12} {'CASE B':>12} {'σ_A':>8} {'σ_B':>8} {'VR_A':>8} {'VR_B':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    _, P0 = prior()
    s0 = np.sqrt(np.diag(P0))
    for i, name in enumerate(STATE_NAMES):
        true_val = None
        if name.startswith("tread_"):
            true_val = truth.tread[name.split("_")[1]]
        elif name.startswith("press_"):
            true_val = truth.pressure[name.split("_")[1]]
        elif name == "toe^2":
            true_val = truth.toe ** 2
        elif name == "camber":
            true_val = truth.camber
        tv_str = f"{true_val:.2f}" if true_val is not None else "—"
        vr_a = s0[i] / sigma_a[i] if sigma_a[i] > 1e-15 else float('inf')
        vr_b = s0[i] / sigma_b[i] if sigma_b[i] > 1e-15 else float('inf')
        marker = " ◄" if name == "toe^2" else ""
        print(f"  {name:<12} {tv_str:>8} {x_a[i]:>12.4f} {x_b[i]:>12.4f} "
              f"{sigma_a[i]:>8.4f} {sigma_b[i]:>8.4f} {vr_a:>8.2f} {vr_b:>8.2f}{marker}")

    print(f"\n── Toe (toe^2) Observability ──")
    print(f"  True toe^2:            {truth.toe**2:.6f}")
    print(f"  True toe:              {truth.toe:.4f} deg")
    print()
    print(f"  {'Metric':<35} {'CASE A':>14} {'CASE B':>14} {'Δ':>14}")
    print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*14}")
    metrics = [
        ("toe^2 estimate", "toe^2 estimate", x_a[IDX_TOESQ], x_b[IDX_TOESQ]),
        ("toe^2 posterior variance", "toe_variance_posterior",
         obs_a["toe_variance_posterior"], obs_b["toe_variance_posterior"]),
        ("toe^2 variance reduction", "toe_variance_reduction",
         obs_a["toe_variance_reduction"], obs_b["toe_variance_reduction"]),
        ("Fisher information", "toe_fisher_information",
         obs_a["toe_fisher_information"], obs_b["toe_fisher_information"]),
        ("Jacobian d(roadload)/d(toe^2)", "toe_sensitivity_roadload",
         obs_a["toe_sensitivity_roadload"], obs_b["toe_sensitivity_roadload"]),
        ("Jacobian d(motor_torque)/d(toe^2)", "toe_sensitivity_motor_torque",
         obs_a["toe_sensitivity_motor_torque"], obs_b["toe_sensitivity_motor_torque"]),
        ("Observability classification", "toe_observability",
         obs_a["toe_observability"], obs_b["toe_observability"]),
    ]
    for label, key, va, vb in metrics:
        if isinstance(va, str):
            print(f"  {label:<35} {va:>14} {vb:>14} {'—':>14}")
        else:
            delta = vb - va if isinstance(vb, (int, float)) else "—"
            delta_str = f"{delta:+.6e}" if isinstance(delta, float) else str(delta)
            print(f"  {label:<35} {va:>14.6f} {vb:>14.6f} {delta_str:>14}")

    print(f"\n── Jacobian Structure (toe^2 row) ──")
    print(f"  H[{M_ROADLOAD}, {IDX_TOESQ}] (road_load):      {H_b[M_ROADLOAD, IDX_TOESQ]:+.6e}")
    print(f"  H[{M_MOTORTORQUE}, {IDX_TOESQ}] (motor_torque): {H_b[M_MOTORTORQUE, IDX_TOESQ]:+.6e}")
    print(f"  Total toe^2 Fisher contribution from road_load:  "
          f"{H_b[M_ROADLOAD, IDX_TOESQ]**2 / (measurement_covariance()[M_ROADLOAD, M_ROADLOAD]):.6e}")
    print(f"  Total toe^2 Fisher contribution from motor_torque: "
          f"{H_b[M_MOTORTORQUE, IDX_TOESQ]**2 / (measurement_covariance()[M_MOTORTORQUE, M_MOTORTORQUE]):.6e}")

    print(f"\n── Diagnostics ──")
    print(f"  Measurements available:  CASE A = {obs_a['n_measurements_available']}, "
          f"CASE B = {obs_b['n_measurements_available']}")
    print(f"  Motor torque value:      {z_b[M_MOTORTORQUE]:.2f} N.m")
    print(f"  Acceleration used:       {accel_ms2:.2f} m/s²")
    dt = DrivetrainConfig()
    print(f"  Rolling radius:          {dt.rolling_radius:.4f} m")
    print(f"  Gear ratio:              {dt.gear_ratio:.1f}")
    print(f"  Drivetrain efficiency:   {dt.efficiency:.2f}")

    print(f"\n── Conclusion ──")
    if obs_a["toe_observability"] == obs_b["toe_observability"]:
        print(f"  Toe observability: {obs_a['toe_observability']} → {obs_b['toe_observability']} (unchanged)")
        print(f"  Fisher information increased by: "
              f"{obs_b['toe_fisher_information'] - obs_a['toe_fisher_information']:.4f} "
              f"({100*(obs_b['toe_fisher_information'] - obs_a['toe_fisher_information'])/max(obs_a['toe_fisher_information'],1e-10):.2f}%)")
        if obs_b["toe_observability"] == "OBSERVED":
            print("  The existing road load channel already provides toe sensitivity via toe_drag_from_sq().")
            print("  The motor torque channel adds a physically independent measurement path,")
            print("  but the additional Fisher information is marginal because both channels")
            print("  encode toe drag through the same physical mechanism.")
        else:
            print("  The motor torque channel does not provide sufficient additional information")
            print("  to change the toe observability classification.")
    else:
        print(f"  Toe observability CHANGED: {obs_a['toe_observability']} → {obs_b['toe_observability']}")
        print(f"  The motor torque channel provided sufficient additional information.")
    print("=" * 72)


def test_run_experiment():
    run_experiment()

if __name__ == "__main__":
    run_experiment()
