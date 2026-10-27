import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mismatch_study import run  # noqa: E402
from backend.calibrated import Baseline, model_variance  # noqa: E402


class MismatchStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lvl1 = run(60, 21, 1.0)
        cls.lvl1_off = run(60, 21, 1.0, install_tread_error_mm=0.5)

    def test_plain_estimator_is_badly_overconfident_under_mismatch(self):
        p = self.lvl1["plain"]
        self.assertGreater(p["tread"], 5 * p["tread_sigma"])

    def test_calibration_cuts_tread_and_pressure_error(self):
        self.assertLess(self.lvl1["calibrated"]["tread"], 0.5 * self.lvl1["plain"]["tread"])
        self.assertLess(self.lvl1["calibrated"]["press"], 0.5 * self.lvl1["plain"]["press"])

    def test_calibrated_claimed_sigma_is_roughly_honest(self):
        c = self.lvl1["calibrated"]
        self.assertLess(c["tread"], 1.6 * c["tread_sigma"])
        self.assertLess(c["camber"], 1.6 * c["camber_sigma"])

    def test_calibration_fixes_accelerometer_bias_camber(self):
        self.assertLess(self.lvl1["calibrated"]["camber"], 0.5 * self.lvl1["plain"]["camber"])

    def test_install_tread_error_propagates(self):
        self.assertGreater(self.lvl1_off["calibrated"]["tread"],
                           self.lvl1["calibrated"]["tread"])


class ModelVarianceTests(unittest.TestCase):
    def test_resonance_block_is_common_mode_and_psd(self):
        b = Baseline(np.ones(4), np.zeros(2), np.zeros(4), 0.0, 0.0, 25.0,
                     f_pred=np.array([48.0, 48.0, 50.0, 50.0]), n_snapshots=20)
        v = model_variance(45.0, b)
        np.testing.assert_allclose(v, v.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(v).min(), -1e-12)
        self.assertGreater(v[4, 5], 0.0)
        self.assertAlmostEqual(v[10, 11] ** 2, v[10, 10] * v[11, 11], places=18)

    def test_no_temperature_distance_leaves_only_averaging_error(self):
        b = Baseline(np.ones(4), np.zeros(2), np.zeros(4), 0.0, 0.0, 25.0,
                     f_pred=np.full(4, 48.0), n_snapshots=20)
        v = model_variance(25.0, b)
        self.assertEqual(v[4, 5], 0.0)
        self.assertAlmostEqual(v[4, 4], 0.15 ** 2 / 20, places=12)


if __name__ == "__main__":
    unittest.main()
