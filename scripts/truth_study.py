"""Plain vs install-calibrated estimator against the structurally different truth model.

SIMULATION ONLY. Truth is backend/truth_model.py (assumed physics, same authors
as the estimator). 95 % CIs are over seeds. 'gain' is the truth model's
d(frequency)/d(tread) relative to the estimator's resonance model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.study import (  # noqa: E402
    calibrated_estimator, mean_ci, plain_estimator, run_study,
)
from backend.truth_model import tread_sensitivity_gain  # noqa: E402


def fmt(values):
    m, h = mean_ci(values)
    return f"{m:5.2f}+/-{h:4.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--rim", type=float, nargs="+", default=[np.inf, 2.0, 0.9, 0.45])
    a = ap.parse_args()
    seeds = range(a.seeds)
    print(f"SIMULATION ONLY, assumed truth physics; {a.seeds} seeds x {a.n} vehicles, level {a.level}")
    print("values are mean +/- 95% CI over seeds; claimed = RMS posterior sigma; "
          "cov1 = share of errors inside +/-1 sigma (0.68 if honest); NEES/dof for tread (1.0 if honest)")
    est = {"plain": plain_estimator, "calibrated": calibrated_estimator()}
    for rim in a.rim:
        for drop_res in (False, True):
            res = run_study(est, seeds, a.n, a.level, rim, drop_resonance=drop_res)
            tag = "no resonance" if drop_res else "resonance"
            print(f"\nrim inertia {rim:g} kg.m^2 (gain {tread_sensitivity_gain(rim):.2f}), {tag}")
            print(f"{'estimator':11s} {'tread':>11s} {'claimed':>8s} {'cov1':>5s} {'NEES':>6s} "
                  f"{'press':>11s} {'claimed':>8s} {'camber':>11s} {'claimed':>8s}")
            for name, r in res.items():
                print(f"{name:11s} {fmt(r['tread']['rmse']):>11s} {np.mean(r['tread']['sigma']):8.2f} "
                      f"{np.mean(r['tread']['cover1']):5.2f} {np.mean(r['tread']['nees']):6.1f} "
                      f"{fmt(r['press']['rmse']):>11s} {np.mean(r['press']['sigma']):8.2f} "
                      f"{fmt(r['camber']['rmse']):>11s} {np.mean(r['camber']['sigma']):8.2f}")


if __name__ == "__main__":
    main()
