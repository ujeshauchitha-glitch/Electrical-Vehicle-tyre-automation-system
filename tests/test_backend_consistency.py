import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ablation import VARIANTS, run  # noqa: E402


class AblationConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = run(150, 7, VARIANTS["full"])
        cls.no_res = run(150, 7, VARIANTS["no resonance"])
        cls.no_tpms = run(150, 7, VARIANTS["no TPMS"])

    def test_nees_near_one_per_dof(self):
        for r in (self.full, self.no_res, self.no_tpms):
            self.assertGreater(r["nees_per_dof"], 0.5)
            self.assertLess(r["nees_per_dof"], 1.5)

    def test_resonance_carries_tread(self):
        self.assertLess(self.full["tread_rmse_mm"], 0.3)
        self.assertGreater(self.no_res["tread_rmse_mm"], 5 * self.full["tread_rmse_mm"])

    def test_tpms_carries_pressure(self):
        self.assertGreater(self.no_tpms["pressure_rmse_kpa"], 3 * self.full["pressure_rmse_kpa"])


if __name__ == "__main__":
    unittest.main()
