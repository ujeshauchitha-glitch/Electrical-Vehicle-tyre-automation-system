import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ablation_ext import run  # noqa: E402
from backend.estimator import M_FREQ, TyreState, estimate, measure  # noqa: E402
from backend.extended import (  # noqa: E402
    estimate_ext, jacobian_ext, measure_ext, reportable,
)


class ExtendedChannelTests(unittest.TestCase):
    def test_camber_column_is_zero_without_esc_and_nonzero_with(self):
        x = np.array([5.0] * 4 + [240.0] * 4 + [0.1, 0.0])
        H = jacobian_ext(x)
        self.assertGreater(abs(H[12, 9]), 0.0)
        self.assertGreater(abs(H[13, 9]), 0.0)
        self.assertTrue(np.all(H[:12, 9] == 0.0))

    def test_unavailable_extra_channels_leave_base_estimate_unchanged(self):
        rng = np.random.default_rng(3)
        z, T = measure(TyreState.random(rng), rng, include_motor_torque=True)
        z_ext = np.full(18, np.nan)
        z_ext[:12] = z
        xb, _ = estimate(z, T)
        xe, _ = estimate_ext(z_ext, T)
        np.testing.assert_allclose(xe[:9], xb[:9], atol=1e-4)

    def test_camber_not_reported_without_esc(self):
        rng = np.random.default_rng(4)
        z, T = measure_ext(TyreState.random(rng), rng, esc=False, gnss=False)
        x, P = estimate_ext(z, T)
        value, sigma, _ = reportable(x, P)[9]
        self.assertIsNone(value)
        self.assertAlmostEqual(sigma, 1.0, places=3)

    def test_camber_recovered_with_esc_and_no_bias(self):
        _, _, cam_rmse, cam_sigma = run(100, 11, True, False, False, 0.0, 0.0)
        self.assertLess(cam_rmse, 0.25)
        self.assertLess(cam_sigma, 0.2)

    def test_accelerometer_bias_breaks_camber_and_sigma_does_not_warn(self):
        """Known limitation: an unmodelled 0.15 m/s^2 bias gives an error far larger
        than the claimed sigma. Fails if someone makes R aware of bias; update then."""
        _, _, cam_rmse, cam_sigma = run(100, 11, True, False, False, 0.0, 0.15)
        self.assertGreater(cam_rmse, 3 * cam_sigma)

    def test_gnss_radius_improves_tread_without_resonance_in_simulation(self):
        base = run(100, 12, False, False, True, 0.0, 0.0)[0]
        gnss = run(100, 12, False, True, True, 0.0, 0.0)[0]
        self.assertLess(gnss, 0.8 * base)

    def test_slip_bias_biases_tread_and_is_not_absorbed(self):
        _, bias, _, _ = run(100, 12, False, True, True, 0.01, 0.0)
        self.assertGreater(bias, 1.5)

    def test_resonance_dropped_leaves_tread_channel_to_gnss(self):
        rng = np.random.default_rng(5)
        z, T = measure_ext(TyreState.random(rng), rng, esc=False, gnss=True)
        z[M_FREQ] = np.nan
        x, P = estimate_ext(z, T)
        self.assertTrue(np.all(np.isfinite(x)))


if __name__ == "__main__":
    unittest.main()
