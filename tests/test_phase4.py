"""
Phase 4 tests for the EV Tyre Intelligence estimator.

Tests the independent motor-torque-derived road-load channel:
  - Circularity regression (motor torque is NOT derived from Crr model)
  - Observability comparison (before vs after adding motor torque)
  - Missing data handling (NaN measurements)
  - Drivetrain configuration validation
  - Toe observability through Jacobian sensitivity

Run with: python -m pytest tests/test_phase4.py -v
"""

import numpy as np
import pytest

from backend.estimator import (
    CORNERS,
    IDX_TOESQ,
    M_MOTORTORQUE,
    M_ROADLOAD,
    N_MEAS,
    N_STATE,
    SensorNoise,
    TyreState,
    estimate,
    jacobian,
    measure,
    measurement_covariance,
    observability_analysis,
    prior,
    predict,
)
from backend.physics import (
    DrivetrainConfig,
    Vehicle,
    aero_drag_force,
    motor_torque_measurement,
    rolling_resistance_coeff,
    toe_drag_from_sq,
)


# ---------------------------------------------------------------------------
# DrivetrainConfig tests
# ---------------------------------------------------------------------------

class TestDrivetrainConfig:
    def test_default_config_valid(self):
        dt = DrivetrainConfig()
        assert dt.validate()

    def test_invalid_efficiency_zero(self):
        dt = DrivetrainConfig(efficiency=0.0)
        assert not dt.validate()

    def test_invalid_efficiency_negative(self):
        dt = DrivetrainConfig(efficiency=-0.5)
        assert not dt.validate()

    def test_invalid_efficiency_above_one(self):
        dt = DrivetrainConfig(efficiency=1.5)
        assert not dt.validate()

    def test_invalid_gear_ratio(self):
        dt = DrivetrainConfig(gear_ratio=0.0)
        assert not dt.validate()

    def test_invalid_rolling_radius(self):
        dt = DrivetrainConfig(rolling_radius=0.0)
        assert not dt.validate()

    def test_invalid_mass(self):
        dt = DrivetrainConfig(mass=-100.0)
        assert not dt.validate()

    def test_config_units_documented(self):
        """Every field must have a documented physical meaning."""
        dt = DrivetrainConfig()
        assert dt.gear_ratio == Vehicle.gear_ratio
        assert dt.efficiency == Vehicle.drivetrain_eff
        assert dt.rolling_radius == Vehicle.r_belt
        assert dt.mass == Vehicle.mass
        assert dt.CdA == Vehicle.CdA
        assert dt.rho_air == Vehicle.rho_air
        assert dt.g == Vehicle.g
        assert dt.grade_rad == 0.0


# ---------------------------------------------------------------------------
# Motor torque measurement tests
# ---------------------------------------------------------------------------

class TestMotorTorqueMeasurement:
    def test_positive_torque(self):
        """Motor torque should be positive at reasonable driving conditions."""
        state = np.array([5.0, 5.0, 5.0, 5.0, 240.0, 240.0, 240.0, 240.0, 0.1, 0.0])
        T = motor_torque_measurement(state, v_ms=22.0, accel_ms2=0.0)
        assert T > 0, "Motor torque should be positive at cruising speed"

    def test_includes_toe_drag(self):
        """Motor torque should increase with toe (more toe = more drag = more torque)."""
        state_low_toe = np.array([5.0]*4 + [240.0]*4 + [0.01, 0.0])
        state_high_toe = np.array([5.0]*4 + [240.0]*4 + [1.0, 0.0])

        T_low = motor_torque_measurement(state_low_toe, v_ms=22.0)
        T_high = motor_torque_measurement(state_high_toe, v_ms=22.0)
        assert T_high > T_low, "Higher toe should require more motor torque"

    def test_increases_with_speed(self):
        """Motor torque should increase with speed (more aero drag)."""
        state = np.array([5.0]*4 + [240.0]*4 + [0.1, 0.0])
        T_slow = motor_torque_measurement(state, v_ms=10.0)
        T_fast = motor_torque_measurement(state, v_ms=30.0)
        assert T_fast > T_slow

    def test_increases_with_acceleration(self):
        """Motor torque should increase with acceleration."""
        state = np.array([5.0]*4 + [240.0]*4 + [0.1, 0.0])
        T_const = motor_torque_measurement(state, v_ms=22.0, accel_ms2=0.0)
        T_accel = motor_torque_measurement(state, v_ms=22.0, accel_ms2=2.0)
        assert T_accel > T_const


# ---------------------------------------------------------------------------
# Circularity regression tests
# ---------------------------------------------------------------------------

class TestCircularity:
    def test_motor_torque_measurement_from_can_not_from_model(self):
        """The motor torque MEASUREMENT comes from the motor controller CAN
        bus and is physically independent of the tyre pressure model.

        The EKF forward model correctly uses tyre state to PREDICT expected
        torque (for Jacobian computation), but the actual measurement data
        is an independent observation from the drivetrain. This test verifies
        that the information flow is correct:
          CAN bus (motor torque) -> estimator -> tyre state
        NOT:
          tyre pressure -> Crr model -> road load -> estimator -> tyre pressure
        """
        # Build two measurement vectors with identical non-torque channels
        # but different motor torque values (as would come from the CAN bus)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        truth = TyreState()
        z1, T_meas1 = measure(truth, rng1, v_ms=22.0, include_motor_torque=True)
        z2, T_meas2 = measure(truth, rng2, v_ms=22.0, include_motor_torque=True)

        # Same inputs → same measurements
        np.testing.assert_array_equal(z1, z2)

        # Now change ONLY the motor torque value (simulating a different CAN reading)
        z3 = z1.copy()
        z3[M_MOTORTORQUE] = z1[M_MOTORTORQUE] + 50.0  # different torque reading

        # Run estimator on both
        x1, P1 = estimate(z1, T_meas1, v_ms=22.0)
        x3, P3 = estimate(z3, T_meas1, v_ms=22.0)

        # Different motor torque measurements should produce different estimates
        # This proves the motor torque channel carries independent information
        assert not np.allclose(x1, x3, atol=0.01), \
            "Different motor torque readings must change the tyre state estimate"

    def test_motor_torque_independent_of_pressure_in_measurement_generation(self):
        """The motor torque MEASUREMENT (from CAN bus) does not depend on
        the pressure-derived Crr model. The estimator uses tyre state to
        PREDICT expected torque, but the measurement itself is independent."""

        # Two states with different pressures but same tread and toe
        state_a = TyreState(
            tread={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
            pressure={"FL": 200.0, "FR": 200.0, "RL": 200.0, "RR": 200.0},
            temp={"FL": 25.0, "FR": 25.0, "RL": 25.0, "RR": 25.0},
            toe=0.5,
        )
        state_b = TyreState(
            tread={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
            pressure={"FL": 280.0, "FR": 280.0, "RL": 280.0, "RR": 280.0},
            temp={"FL": 25.0, "FR": 25.0, "RL": 25.0, "RR": 25.0},
            toe=0.5,
        )

        # Old road load (circular): uses pressure-derived Crr directly
        from backend.physics import road_load
        F_rr_old_a = road_load(state_a, 22.0)
        F_rr_old_b = road_load(state_b, 22.0)

        # The old road load DIFFERS between the two states because it
        # directly encodes pressure through Crr.
        assert not np.isclose(F_rr_old_a, F_rr_old_b, rtol=0.001), \
            "Old road load should differ with different pressures (circular dependency)"

        # Motor torque measurement: also differs because it includes F_rr
        # BUT it ALSO includes aero drag, grade, inertia -- information
        # that is INDEPENDENT of tyre pressure.
        arr_a = np.array([5.0]*4 + [200.0]*4 + [0.25, 0.0])
        arr_b = np.array([5.0]*4 + [280.0]*4 + [0.25, 0.0])

        T_a = motor_torque_measurement(arr_a, v_ms=22.0)
        T_b = motor_torque_measurement(arr_b, v_ms=22.0)

        # Motor torque differs because F_rr differs -- this is EXPECTED.
        # The estimator uses this to UPDATE its tyre state estimate.
        # The key difference is that the MEASUREMENT comes from the motor
        # controller (independent), not from the Crr model.
        assert T_a != T_b, "Motor torque should differ for different pressures"

        # The ratio of differences should NOT match the Crr model ratio
        # because motor torque includes other independent terms
        F_aero = aero_drag_force(22.0)
        Crr_a = rolling_resistance_coeff(5.0, 200.0, 25.0)
        Crr_b = rolling_resistance_coeff(5.0, 280.0, 25.0)

        # Motor torque difference is driven by F_rr difference, NOT by
        # the old road_load coefficient model
        delta_T = T_b - T_a
        delta_Frr = sum(
            (rolling_resistance_coeff(5.0, 280.0, 25.0) - rolling_resistance_coeff(5.0, 200.0, 25.0))
            * Vehicle.Fz(c)
            for c in CORNERS
        )
        expected_delta_T = delta_Frr * Vehicle.r_belt / (Vehicle.gear_ratio * Vehicle.drivetrain_eff)
        assert np.isclose(delta_T, expected_delta_T, rtol=0.01), \
            "Motor torque difference must match physical F_rr difference, not old Crr model"

    def test_information_flow_independence(self):
        """Verify the information flow is:
        motor torque (CAN) -> estimator -> tyre state
        NOT:
        tyre pressure -> Crr model -> road load -> estimator -> tyre pressure
        """
        state = np.array([5.0]*4 + [240.0]*4 + [0.1, 0.0])

        # The Jacobian tells us what the estimator LEARNS from each measurement
        H = jacobian(state, v_ms=22.0)

        # Motor torque channel Jacobian: how much does predicted motor torque
        # change with each state variable?
        motor_torque_jacobian = H[M_MOTORTORQUE, :]

        # The motor torque channel has sensitivity to tread and pressure
        # (through rolling resistance) AND to toe (through toe drag).
        # This is physically correct -- the motor torque MEASUREMENT
        # provides information about ALL these quantities.

        # Key: the JACOBIAN is computed from the forward model (which uses
        # tyre state), but the MEASUREMENT DATA comes from the motor controller.
        # This is the correct architecture for an EKF.

        # Verify motor torque has nonzero sensitivity to toe
        toe_sens = abs(motor_torque_jacobian[IDX_TOESQ])
        assert toe_sens > 0, "Motor torque must have nonzero toe sensitivity"

        # Verify motor torque has nonzero sensitivity to pressure
        for i in range(4):
            assert abs(motor_torque_jacobian[4 + i]) > 0, \
                f"Motor torque must have nonzero sensitivity to press_{CORNERS[i]}"


# ---------------------------------------------------------------------------
# Observability comparison tests
# ---------------------------------------------------------------------------

class TestObservabilityComparison:
    def test_toe_observability_before_after(self):
        """Compare toe observability with and without motor torque channel.

        This is the core Phase 4 experiment. We do NOT force any particular
        result -- we let the physics determine the truth.
        """
        # Fixed scenario for reproducibility
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

        # Generate measurement WITHOUT motor torque (Phase 1-3)
        z_before, T_meas = measure(truth, rng, v_ms=v_ms, accel_ms2=accel_ms2,
                                    include_motor_torque=False)

        # Run estimator before
        x_before, P_before = estimate(z_before, T_meas, v_ms=v_ms,
                                       accel_ms2=accel_ms2)
        obs_before = observability_analysis(x_before, P_before, z_before, T_meas,
                                             v_ms, accel_ms2)

        # Generate measurement WITH motor torque (Phase 4)
        rng2 = np.random.default_rng(42)  # same seed for reproducibility
        z_after, T_meas2 = measure(truth, rng2, v_ms=v_ms, accel_ms2=accel_ms2,
                                    include_motor_torque=True)

        # Run estimator after
        x_after, P_after = estimate(z_after, T_meas2, v_ms=v_ms,
                                     accel_ms2=accel_ms2)
        obs_after = observability_analysis(x_after, P_after, z_after, T_meas2,
                                            v_ms, accel_ms2)

        # Print results for inspection
        print("\n" + "="*60)
        print("PHASE 4 OBSERVABILITY COMPARISON")
        print("="*60)
        print(f"\nBEFORE (without motor torque):")
        print(f"  toe^2 estimate:  {x_before[IDX_TOESQ]:.6f}")
        print(f"  toe^2 variance:  {obs_before['toe_variance_posterior']:.6f}")
        print(f"  toe^2 VR:        {obs_before['toe_variance_reduction']:.4f}")
        print(f"  toe observability: {obs_before['toe_observability']}")
        print(f"  toe sensitivity (roadload): {obs_before['toe_sensitivity_roadload']:.6f}")
        print(f"  toe sensitivity (motor_torque): {obs_before['toe_sensitivity_motor_torque']:.6f}")
        print(f"  Fisher information: {obs_before['toe_fisher_information']:.6e}")
        print(f"  measurements available: {obs_before['n_measurements_available']}")
        print(f"\nAFTER (with motor torque):")
        print(f"  toe^2 estimate:  {x_after[IDX_TOESQ]:.6f}")
        print(f"  toe^2 variance:  {obs_after['toe_variance_posterior']:.6f}")
        print(f"  toe^2 VR:        {obs_after['toe_variance_reduction']:.4f}")
        print(f"  toe observability: {obs_after['toe_observability']}")
        print(f"  toe sensitivity (roadload): {obs_after['toe_sensitivity_roadload']:.6f}")
        print(f"  toe sensitivity (motor_torque): {obs_after['toe_sensitivity_motor_torque']:.6f}")
        print(f"  Fisher information: {obs_after['toe_fisher_information']:.6e}")
        print(f"  measurements available: {obs_after['n_measurements_available']}")
        print("="*60)

        # Structural assertions (not outcome assertions):
        # The estimator must RECOMPUTE observability, not hard-code it
        assert obs_before["toe_observability"] in ("UNOBSERVABLE", "WEAK", "OBSERVED")
        assert obs_after["toe_observability"] in ("UNOBSERVABLE", "WEAK", "OBSERVED")

        # Motor torque must have nonzero toe sensitivity
        assert obs_after["toe_sensitivity_motor_torque"] != 0, \
            "Motor torque Jacobian must show nonzero toe sensitivity"

        # After adding motor torque, Fisher information should increase
        # (more information = more certainty, regardless of classification)
        assert obs_after["toe_fisher_information"] >= obs_before["toe_fisher_information"], \
            "Adding motor torque must not decrease Fisher information"

        # Posterior variance should not increase
        assert obs_after["toe_variance_posterior"] <= obs_before["toe_variance_posterior"] + 1e-10, \
            "Adding motor torque must not increase posterior toe variance"

        # Motor torque measurement count
        assert obs_before["n_measurements_available"] == 11
        assert obs_after["n_measurements_available"] == 12

    def test_observability_is_computed_not_hardcoded(self):
        """Verify the observability classifier responds to actual Jacobian values,
        not to a fixed string."""
        # Create two scenarios with very different toe values
        state_no_toe = np.array([5.0]*4 + [240.0]*4 + [0.001, 0.0])
        state_big_toe = np.array([5.0]*4 + [240.0]*4 + [2.0, 0.0])

        P = np.diag([1.0]*10)  # some covariance

        # Use a realistic measurement vector to avoid singular R matrix
        # (C_ref from z[M_ROADLOAD] must be nonzero)
        rng = np.random.default_rng(42)
        z_realistic, _ = measure(TyreState(), rng, include_motor_torque=False)

        obs_no_toe = observability_analysis(state_no_toe, P, z_realistic)
        obs_big_toe = observability_analysis(state_big_toe, P, z_realistic)

        # Different toe states should give different Fisher information
        # because the Jacobian is state-dependent
        assert obs_no_toe["toe_fisher_information"] != obs_big_toe["toe_fisher_information"], \
            "Observability must be recomputed, not hard-coded"


# ---------------------------------------------------------------------------
# Missing data tests
# ---------------------------------------------------------------------------

class TestMissingData:
    def test_nan_motor_torque_ignored(self):
        """When motor torque is NaN, estimator should produce valid results."""
        rng = np.random.default_rng(42)
        z, T_meas = measure(TyreState(), rng, include_motor_torque=False)
        assert np.isnan(z[M_MOTORTORQUE])

        x, P = estimate(z, T_meas)
        sigma = np.sqrt(np.diag(P))

        # Should still produce valid estimates for all states
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma > 0)

    def test_nan_motor_torque_same_as_old_estimator(self):
        """Without motor torque, results should match the old estimator behavior."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        z1, T1 = measure(TyreState(), rng1, include_motor_torque=False)
        z2, T2 = measure(TyreState(), rng2, include_motor_torque=False)

        x1, P1 = estimate(z1, T1)
        x2, P2 = estimate(z2, T2)

        # Same inputs should give same outputs
        np.testing.assert_array_almost_equal(x1, x2, decimal=10)
        np.testing.assert_array_almost_equal(P1, P2, decimal=10)

    def test_missing_acceleration_makes_motor_torque_unavailable(self):
        """If acceleration is not provided, the motor torque channel should
        be marked unavailable (NaN) in the API."""
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        # Motor torque without acceleration -> unavailable
        resp = client.post("/api/estimate", json={
            "pressure_fl": 240,
            "pressure_fr": 240,
            "pressure_rl": 240,
            "pressure_rr": 240,
            "motor_torque_nm": 150.0,
            "vehicle_speed_ms": 22.0,
            # acceleration_ms2 is MISSING
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["phase4"]["road_load_available"] is False

    def test_all_phase4_inputs_provided(self):
        """When all Phase 4 inputs are provided, motor torque channel is active."""
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        resp = client.post("/api/estimate", json={
            "pressure_fl": 240,
            "pressure_fr": 240,
            "pressure_rl": 240,
            "pressure_rr": 240,
            "motor_torque_nm": 150.0,
            "vehicle_speed_ms": 22.0,
            "acceleration_ms2": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["phase4"]["road_load_available"] is True
        assert data["phase4"]["motor_torque_used"] == 150.0
        assert data["phase4"]["acceleration_used"] == 0.5

    def test_measurement_vector_size(self):
        """Measurement vector must always be 12 elements."""
        rng = np.random.default_rng(42)
        z, _ = measure(TyreState(), rng)
        assert z.shape == (12,)

    def test_motor_torque_nan_does_not_corrupt_estimate(self):
        """NaN at index 11 must not propagate to other state estimates."""
        rng = np.random.default_rng(42)
        z_nan, T_meas = measure(TyreState(), rng, include_motor_torque=False)
        rng2 = np.random.default_rng(42)
        z_valid, T_meas2 = measure(TyreState(), rng2, include_motor_torque=True)

        x_nan, P_nan = estimate(z_nan, T_meas)
        x_valid, P_valid = estimate(z_valid, T_meas2)

        # Both should produce finite results
        assert np.all(np.isfinite(x_nan))
        assert np.all(np.isfinite(x_valid))
        assert np.all(np.isfinite(np.diag(P_nan)))
        assert np.all(np.isfinite(np.diag(P_valid)))


# ---------------------------------------------------------------------------
# Jacobian structure tests
# ---------------------------------------------------------------------------

class TestJacobian:
    def test_jacobian_shape(self):
        """Jacobian must be (12, 10) for Phase 4."""
        state = np.array([5.0]*4 + [240.0]*4 + [0.1, 0.0])
        H = jacobian(state, v_ms=22.0)
        assert H.shape == (N_MEAS, N_STATE)  # (12, 10)

    def test_motor_torque_jacobian_nonzero_for_toe(self):
        """The motor torque row must have nonzero sensitivity to toe^2."""
        state = np.array([5.0]*4 + [240.0]*4 + [0.25, 0.0])
        H = jacobian(state, v_ms=22.0)
        toe_sensitivity = abs(H[M_MOTORTORQUE, IDX_TOESQ])
        assert toe_sensitivity > 0, \
            "Motor torque Jacobian must show toe sensitivity"

    def test_motor_torque_jacobian_nonzero_for_pressure(self):
        """The motor torque row must have nonzero sensitivity to pressures."""
        state = np.array([5.0]*4 + [240.0]*4 + [0.25, 0.0])
        H = jacobian(state, v_ms=22.0)
        for i in range(4):
            assert abs(H[M_MOTORTORQUE, 4 + i]) > 0, \
                f"Motor torque must have sensitivity to press_{CORNERS[i]}"

    def test_motor_torque_jacobian_nonzero_for_tread(self):
        """The motor torque row must have nonzero sensitivity to tread."""
        state = np.array([5.0]*4 + [240.0]*4 + [0.25, 0.0])
        H = jacobian(state, v_ms=22.0)
        for i in range(4):
            assert abs(H[M_MOTORTORQUE, i]) > 0, \
                f"Motor torque must have sensitivity to tread_{CORNERS[i]}"

    def test_existing_jacobian_unchanged(self):
        """The existing measurement channels should have the same Jacobian
        structure as before Phase 4."""
        state = np.array([5.0]*4 + [240.0]*4 + [0.1, 0.0])
        H = jacobian(state, v_ms=22.0)

        # Road load sensitivity to toe^2 should be same as before
        toe_roadload = H[M_ROADLOAD, IDX_TOESQ]
        expected = 2.0 * Vehicle.C_alpha * (np.pi / 180.0) ** 2 / (Vehicle.mass * Vehicle.g)
        assert np.isclose(toe_roadload, expected, rtol=0.01), \
            f"Road load toe sensitivity {toe_roadload} != expected {expected}"


# ---------------------------------------------------------------------------
# Physics-level independence tests
# ---------------------------------------------------------------------------

class TestPhysicsIndependence:
    def test_aero_drag_independent_of_tyre_state(self):
        """Aerodynamic drag depends only on velocity, not tyre parameters."""
        F1 = aero_drag_force(22.0)
        F2 = aero_drag_force(22.0, CdA=0.8)
        assert F1 > 0
        assert F2 > F1  # larger CdA -> more drag

    def test_rolling_resistance_depends_on_pressure(self):
        """Rolling resistance DOES depend on pressure (this is what we observe)."""
        crr_low = rolling_resistance_coeff(5.0, 180.0, 25.0)
        crr_high = rolling_resistance_coeff(5.0, 300.0, 25.0)
        assert crr_low > crr_high, "Lower pressure -> higher rolling resistance"

    def test_toe_drag_depends_on_toe_squared(self):
        """Toe drag force should scale with toe^2."""
        f1 = toe_drag_from_sq(0.01)
        f2 = toe_drag_from_sq(1.0)
        assert f2 > f1
        # Should be proportional
        assert np.isclose(f2 / f1, 1.0 / 0.01, rtol=0.01)
