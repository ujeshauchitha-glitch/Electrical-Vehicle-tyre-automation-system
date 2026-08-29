"""
Basic tests for the EV Tyre Intelligence estimator.

Run with: python -m pytest tests/ -v
"""

import numpy as np
import pytest

from backend.estimator import (
    CORNERS,
    IDX_TREAD,
    N_MEAS,
    SensorNoise,
    TyreState,
    estimate,
    measure,
    measurement_covariance,
    prior,
)
from backend.physics import (
    Vehicle,
    compensate_pressure,
    effective_rolling_radius,
    first_mode_frequency,
    rolling_resistance_coeff,
    torque_limit,
    wet_friction,
)


# ---------------------------------------------------------------------------
# Physics tests
# ---------------------------------------------------------------------------

class TestPhysics:
    def test_rolling_radius_positive(self):
        r = effective_rolling_radius(5.0, 240.0, Vehicle.Fz("FL"))
        assert r > 0
        assert 0.2 < r < 0.5  # reasonable range in metres

    def test_frequency_increases_with_pressure(self):
        f_low = first_mode_frequency(5.0, 200.0)
        f_high = first_mode_frequency(5.0, 280.0)
        assert f_high > f_low

    def test_frequency_decreases_with_tread(self):
        f_new = first_mode_frequency(8.0, 240.0)
        f_worn = first_mode_frequency(2.0, 240.0)
        assert f_new < f_worn

    def test_rolling_resistance_positive(self):
        crr = rolling_resistance_coeff(5.0, 240.0, 25.0)
        assert crr > 0
        assert crr < 0.05  # should be small

    def test_wet_friction_monotonic(self):
        mu_2 = wet_friction(2.0)
        mu_6 = wet_friction(6.0)
        assert mu_6 > mu_2

    def test_compensate_pressure(self):
        cold = compensate_pressure(260.0, 45.0)
        assert cold < 260.0  # hot tyre reads high, cold-equivalent is lower

    def test_torque_limit_positive(self):
        T, mu, lcb = torque_limit(5.0, 0.3, 240.0, wet=True)
        assert T > 0
        assert mu > 0
        assert lcb > 0


# ---------------------------------------------------------------------------
# Estimator tests
# ---------------------------------------------------------------------------

class TestEstimator:
    def test_prior_shape(self):
        x0, P0 = prior()
        assert x0.shape == (10,)
        assert P0.shape == (10, 10)

    def test_measurement_dimension(self):
        """Phase 4: measurement vector is 12 (added motor torque channel)."""
        assert N_MEAS == 12

    def test_measurement_covariance_positive(self):
        R = measurement_covariance()
        assert R.shape == (12, 12)
        diag = np.diag(R)
        assert np.all(diag > 0)

    def test_estimate_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        z, T_meas = measure(TyreState(), rng)
        x, P = estimate(z, T_meas)
        assert x.shape == (10,)
        assert P.shape == (10, 10)

    def test_tread_within_bounds(self):
        rng = np.random.default_rng(42)
        truth = TyreState.random(rng)
        z, T_meas = measure(truth, rng)
        x, P = estimate(z, T_meas)
        assert np.all(x[IDX_TREAD] >= 0.5)
        assert np.all(x[IDX_TREAD] <= 9.0)

    def test_single_vehicle_coverage(self):
        """Run 50 random vehicles and check that ~95% of errors fall within 2 sigma."""
        rng = np.random.default_rng(12345)
        covered = 0
        total = 0
        for _ in range(50):
            truth = TyreState.random(rng)
            z, T_meas = measure(truth, rng)
            x, P = estimate(z, T_meas)
            sigma = np.sqrt(np.diag(P))
            for i in range(4):
                e = x[i] - truth.tread[CORNERS[i]]
                if abs(e) <= 2.0 * sigma[i]:
                    covered += 1
                total += 1
        coverage = covered / total
        assert coverage > 0.85, f"Coverage {coverage:.1%} too low (expected ~95%)"

    def test_pressure_estimate_close(self):
        """Pressure should be estimated within a few kPa."""
        rng = np.random.default_rng(99)
        truth = TyreState()
        z, T_meas = measure(truth, rng)
        x, P = estimate(z, T_meas)
        for i in range(4):
            err = abs(x[4 + i] - truth.pressure[CORNERS[i]])
            assert err < 10, f"Pressure error {err:.1f} kPa too large for {CORNERS[i]}"


# ---------------------------------------------------------------------------
# API tests (require httpx)
# ---------------------------------------------------------------------------

class TestAPI:
    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[testclient] not installed")
        from backend.main import app
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_vehicle_info(self, client):
        resp = client.get("/api/vehicle")
        assert resp.status_code == 200
        assert resp.json()["mass_kg"] == 1800.0

    def test_estimate_endpoint(self, client):
        resp = client.post("/api/estimate", json={
            "pressure_fl": 240,
            "pressure_fr": 240,
            "pressure_rl": 240,
            "pressure_rr": 240,
            "temperature": 25,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "states" in data
        assert len(data["states"]) == 10

    def test_simulation_endpoint(self, client):
        resp = client.post("/api/simulation", json={"iters": 6})
        assert resp.status_code == 200
        data = resp.json()
        assert "truth" in data
        assert "estimate" in data

    def test_estimator_status(self, client):
        resp = client.get("/api/estimator/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_state"] == 10
        assert data["n_measurement"] == 12  # Phase 4
        assert "motor_torque" in data["measurement_channels"]
