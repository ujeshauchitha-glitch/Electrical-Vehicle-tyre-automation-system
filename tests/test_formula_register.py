"""Hand-computed checks of the backend physics equations.

Each expected value is derived here from the equation as written in the
docstring or docs, not by calling the code under test. Constants are the
project's ASSUMED values (see backend/physics.py), so these tests pin the
equations, not their real-world accuracy.
"""
import math
import unittest

import numpy as np

from backend.estimator import TyreState, predict, predict_batch
from backend.physics import (
    CORNERS,
    Vehicle,
    effective_rolling_radius,
    first_mode_frequency,
    motor_torque_measurement,
    rolling_resistance_coeff,
    toe_drag_force,
    toe_drag_from_sq,
)


class RollingResistance(unittest.TestCase):
    def test_reference_value(self):
        self.assertAlmostEqual(rolling_resistance_coeff(8.0, 240.0, 25.0), 0.0090, places=12)

    def test_warm_tyre_rolls_easier(self):
        cold = rolling_resistance_coeff(5.0, 240.0, 25.0)
        warm = rolling_resistance_coeff(5.0, 240.0, 55.0)
        self.assertLess(warm, cold)
        self.assertAlmostEqual(warm / cold, 1.0 - 0.008 * 30.0, places=12)

    def test_temperature_coefficient_within_published_range(self):
        self.assertTrue(-0.014 <= Vehicle.T_coeff <= -0.005)

    def test_low_pressure_and_worn_tread(self):
        base = rolling_resistance_coeff(8.0, 240.0, 25.0)
        self.assertAlmostEqual(
            rolling_resistance_coeff(8.0, 180.0, 25.0) / base, (240 / 180) ** 0.45, places=12)
        self.assertAlmostEqual(
            rolling_resistance_coeff(4.0, 240.0, 25.0) / base, 0.8 + 0.2 * 0.5, places=12)


class ToeDrag(unittest.TestCase):
    def test_per_wheel_convention(self):
        alpha = math.radians(0.5)
        self.assertAlmostEqual(toe_drag_force(0.5), 2 * 55_000.0 * alpha ** 2, places=9)
        self.assertAlmostEqual(toe_drag_force(0.5), 8.3770, places=3)

    def test_total_toe_is_twice_per_wheel(self):
        """Total axle toe t gives per-wheel t/2: drag = C_alpha t^2 / 2."""
        t = 1.0
        self.assertAlmostEqual(
            toe_drag_force(t / 2), 55_000.0 * math.radians(t) ** 2 / 2, places=9)

    def test_sq_form_matches(self):
        self.assertAlmostEqual(toe_drag_from_sq(0.25), toe_drag_force(0.5), places=9)


class RadiusAndResonance(unittest.TestCase):
    def test_effective_radius(self):
        fz = Vehicle.Fz("RL")
        delta = fz / (210_000.0 * 200.0 / 240.0)
        expected = 0.322 + 0.005 - delta / 3.0
        self.assertAlmostEqual(effective_rolling_radius(5.0, 200.0, fz), expected, places=12)

    def test_tread_adds_one_for_one_to_radius(self):
        fz = Vehicle.Fz("FL")
        d = effective_rolling_radius(6.0, 240.0, fz) - effective_rolling_radius(5.0, 240.0, fz)
        self.assertAlmostEqual(d, 0.001, places=12)

    def test_first_mode_frequency(self):
        K = 39_121.0 + 108.67 * 240.0
        J = 0.550 + 0.0332 * 5.0
        self.assertAlmostEqual(first_mode_frequency(5.0, 240.0),
                               math.sqrt(K / J) / (2 * math.pi), places=9)
        self.assertTrue(30.0 < first_mode_frequency(5.0, 240.0) < 80.0)

    def test_resonance_rises_as_tread_wears_and_as_pressure_rises(self):
        self.assertGreater(first_mode_frequency(2.0, 240.0), first_mode_frequency(8.0, 240.0))
        self.assertGreater(first_mode_frequency(5.0, 280.0), first_mode_frequency(5.0, 200.0))

    def test_axle_ratio_orientation(self):
        x = np.array([5.0, 6.0, 5.0, 5.0] + [240.0] * 4 + [0.0, 0.0])
        z = predict(x)
        self.assertGreater(z[8], 1.0)
        self.assertAlmostEqual(z[9], 1.0, places=12)


class MotorTorque(unittest.TestCase):
    def _expected(self, x, v, accel, grade, temps):
        tread, press, toe_sq = x[:4], x[4:8], x[8]
        crr = [rolling_resistance_coeff(tread[i], press[i], temps[i]) for i in range(4)]
        f_rr = math.cos(grade) * sum(crr[i] * Vehicle.Fz(c) for i, c in enumerate(CORNERS))
        f = (0.5 * 1.2 * 0.65 * v ** 2 + f_rr + 1800.0 * 9.81 * math.sin(grade)
             + 1800.0 * accel + 2 * 55_000.0 * math.radians(1) ** 2 * toe_sq)
        r = np.mean([effective_rolling_radius(tread[i], press[i], Vehicle.Fz(c))
                     for i, c in enumerate(CORNERS) if c in ("RL", "RR")])
        return f * r / (9.0 * 0.92)

    def test_hand_computed_torque_with_temperature_grade_and_accel(self):
        x = np.array([6.0, 5.5, 4.0, 3.5] + [230.0, 240.0, 210.0, 250.0] + [0.3, 0.2])
        temps = [40.0, 42.0, 38.0, 45.0]
        got = motor_torque_measurement(x, 25.0, 0.8, 0.03, temp_c=temps)
        self.assertAlmostEqual(got, self._expected(x, 25.0, 0.8, 0.03, temps), places=9)

    def test_uses_effective_not_belt_radius(self):
        x = np.array([5.0] * 4 + [240.0] * 4 + [0.0, 0.0])
        belt_based = motor_torque_measurement(x, 22.0) * Vehicle.r_belt / effective_rolling_radius(
            5.0, 240.0, Vehicle.Fz("RL"))
        self.assertNotAlmostEqual(motor_torque_measurement(x, 22.0), belt_based, places=3)

    def test_warm_tyres_lower_torque(self):
        x = np.array([5.0] * 4 + [240.0] * 4 + [0.0, 0.0])
        self.assertLess(motor_torque_measurement(x, 22.0, temp_c=60.0),
                        motor_torque_measurement(x, 22.0, temp_c=20.0))

    def test_batch_prediction_matches_scalar_torque(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = np.r_[rng.uniform(2, 8, 4), rng.uniform(170, 290, 4), rng.uniform(0, 1), 0.0]
            T = float(rng.uniform(10, 60))
            zb = predict_batch(x, T_meas=T, v_ms=27.0, accel_ms2=0.4, grade_rad=0.02)[0]
            self.assertAlmostEqual(
                zb[11], motor_torque_measurement(x, 27.0, 0.4, 0.02, temp_c=T), places=9)

    def test_measure_and_predict_agree_when_noise_free(self):
        from backend.estimator import SensorNoise, measure

        class Quiet(SensorNoise):
            motortorque_sigma = 0.0

        s = TyreState(temp={c: 40.0 for c in CORNERS})
        z, T = measure(s, np.random.default_rng(1), noise=Quiet, include_motor_torque=True)
        x = np.array([s.tread[c] for c in CORNERS] + [s.pressure[c] for c in CORNERS]
                     + [s.toe ** 2, s.camber])
        self.assertAlmostEqual(
            z[11], predict_batch(x, T_meas=40.0)[0][11], places=9)


if __name__ == "__main__":
    unittest.main()
