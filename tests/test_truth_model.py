import unittest

import numpy as np

from backend.estimator import CORNERS, TyreState
from backend.extended import M_LATACC, predict_ext_batch
from backend.physics import Vehicle
from backend.study import (
    calibrated_estimator, evaluate, generate_cases, mean_ci, plain_estimator,
)
from backend.truth_model import (
    DRIVELINE_STIFFNESS, Conditions, HiddenVehicle, corner_loads,
    measure_truth, nominal_sidewall_stiffness, tread_inertia,
    tread_sensitivity_gain, upper_mode_hz,
)


class _Silent:
    tpms_quantum = 1e-9
    tpms_sigma = temp_sigma = freq_sigma = ratio_sigma = roadload_frac = motortorque_sigma = 0.0


class _SilentExtra:
    latacc_sigma = steer_sigma = gnss_scale_sigma = wheel_speed_sigma = 0.0


def _install_state():
    return TyreState(tread={c: Vehicle.tread_new for c in CORNERS},
                     pressure={c: Vehicle.p_placard for c in CORNERS},
                     temp={c: Vehicle.T_ref for c in CORNERS}, toe=0.0, camber=0.0)


class TruthModelStructureTests(unittest.TestCase):
    def test_nominal_truth_matches_estimator_model_at_install_state(self):
        rng = np.random.default_rng(0)
        hv = HiddenVehicle.draw(rng, np.inf, level=0.0)
        z, _, inputs = measure_truth(_install_state(), hv, Conditions(), rng,
                                     noise=_Silent, extra=_SilentExtra)
        x = np.r_[[Vehicle.tread_new] * 4, [Vehicle.p_placard] * 4, 0.0, 0.0]
        zp = predict_ext_batch(x, Vehicle.T_ref)[0]
        np.testing.assert_allclose(z[4:8], zp[4:8], rtol=0.01)
        np.testing.assert_allclose(z[14:18], zp[14:18], atol=1.5e-3)
        c_eq = z[10] + inputs["grade_rad"] + inputs["accel_ms2"] / Vehicle.g
        np.testing.assert_allclose(c_eq, zp[10], rtol=0.05)
        np.testing.assert_allclose(z[11], zp[11], rtol=0.04)

    def test_two_inertia_upper_mode_hits_the_anchor_and_a_lower_mode_exists(self):
        for rim in (0.45, 0.9, 2.0):
            ks = nominal_sidewall_stiffness(rim)
            jb = 0.55 + float(tread_inertia(Vehicle.tread_new))
            self.assertAlmostEqual(upper_mode_hz(jb, rim, ks), 45.0, delta=0.01)
            m = np.diag([jb, rim])
            k = np.array([[ks, -ks], [-ks, ks + DRIVELINE_STIFFNESS]])
            f = np.sort(np.sqrt(np.linalg.eigvals(np.linalg.solve(m, k)).real)) / (2 * np.pi)
            self.assertLess(f[0], 15.0)

    def test_tread_sensitivity_falls_as_rim_inertia_shrinks(self):
        gains = [tread_sensitivity_gain(r) for r in (np.inf, 2.0, 0.9, 0.45)]
        self.assertTrue(all(a > b for a, b in zip(gains, gains[1:])))
        self.assertAlmostEqual(gains[0], 1.0, delta=0.1)

    def test_road_crown_shows_up_as_lateral_acceleration_without_camber(self):
        rng = np.random.default_rng(0)
        hv = HiddenVehicle.draw(rng, np.inf, level=0.0)
        z, _, _ = measure_truth(_install_state(), hv, Conditions(crown_rad=0.015), rng,
                                noise=_Silent, extra=_SilentExtra)
        self.assertAlmostEqual(z[M_LATACC], Vehicle.g * 0.015, places=6)

    def test_payload_moves_load_to_the_rear(self):
        light, heavy = corner_loads(0.0), corner_loads(300.0)
        self.assertGreater(heavy[2] - light[2], heavy[0] - light[0])
        self.assertAlmostEqual(heavy.sum() - light.sum(), 300.0 * Vehicle.g, places=6)


class StudyHarnessTests(unittest.TestCase):
    def test_mean_ci_matches_hand_value(self):
        m, h = mean_ci([1.0, 2.0, 3.0])
        self.assertAlmostEqual(m, 2.0)
        self.assertAlmostEqual(h, 4.303 / np.sqrt(3), places=6)

    def test_cases_are_reproducible_per_seed(self):
        a = generate_cases(3, 5, n_install=4)
        b = generate_cases(3, 5, n_install=4)
        c = generate_cases(3, 6, n_install=4)
        np.testing.assert_array_equal(a[0].z, b[0].z)
        self.assertFalse(np.array_equal(a[0].z, c[0].z))

    def test_calibration_beats_plain_and_each_group_of_terms_matters(self):
        cases = generate_cases(25, 0)
        plain = evaluate(cases, plain_estimator)["tread"]["rmse"]
        full = evaluate(cases, calibrated_estimator())["tread"]["rmse"]
        no_offsets = evaluate(cases, calibrated_estimator(
            drop=["f_scale", "ratio", "radius", "latacc", "steer"]))["tread"]["rmse"]
        self.assertLess(full, 0.7 * plain)
        self.assertGreater(no_offsets, 1.2 * full)

    def test_model_variance_terms_improve_coverage(self):
        cases = generate_cases(25, 1)
        with_var = evaluate(cases, calibrated_estimator())["tread"]["cover1"]
        without = evaluate(cases, calibrated_estimator(variance="none"))["tread"]["cover1"]
        self.assertGreater(with_var, without)


if __name__ == "__main__":
    unittest.main()
