"""Per-term ablation of the install-baseline calibration against the truth model.

SIMULATION ONLY. Each row switches off one calibration term (or part of the
model variance) and reports the change in RMSE versus the full calibrated
estimator, paired by vehicle, with a 95 % CI over seeds. A positive change
means the term was helping.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.study import (  # noqa: E402
    calibrated_estimator, mean_ci, paired_difference, plain_estimator, run_study,
)

VARIANTS = {
    "full": calibrated_estimator(),
    "- resonance scale": calibrated_estimator(drop=["f_scale"]),
    "- axle-ratio offset": calibrated_estimator(drop=["ratio"]),
    "- radius offset": calibrated_estimator(drop=["radius"]),
    "- lat-accel zero": calibrated_estimator(drop=["latacc"]),
    "- steering zero": calibrated_estimator(drop=["steer"]),
    "- temp compensation": calibrated_estimator(temp_comp=False),
    "- all offsets": calibrated_estimator(drop=["f_scale", "ratio", "radius", "latacc", "steer"]),
    "variance: none": calibrated_estimator(variance="none"),
    "variance: resonance only": calibrated_estimator(variance="resonance"),
    "variance: road-load only": calibrated_estimator(variance="roadload"),
    "plain (no calibration)": plain_estimator,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--rim", type=float, default=np.inf)
    a = ap.parse_args()
    res = run_study(VARIANTS, range(a.seeds), a.n, a.level, a.rim)
    print(f"SIMULATION ONLY, assumed truth physics; {a.seeds} seeds x {a.n} vehicles, "
          f"level {a.level}, rim inertia {a.rim:g}")
    print(f"{'variant':26s} | {'tread RMSE':>11s} {'delta vs full':>13s} {'cov1':>5s} | "
          f"{'press RMSE':>11s} {'delta vs full':>13s} | {'camber RMSE':>11s} {'delta vs full':>13s}")
    for name in VARIANTS:
        cells = []
        for group in ("tread", "press", "camber"):
            m, h = mean_ci(res[name][group]["rmse"])
            if name == "full":
                d = "-"
            else:
                dm, dh = paired_difference(res, name, "full", group)
                d = f"{dm:+.2f}+/-{dh:.2f}"
            cells.append((f"{m:.2f}+/-{h:.2f}", d))
        cov = np.mean(res[name]["tread"]["cover1"])
        print(f"{name:26s} | {cells[0][0]:>11s} {cells[0][1]:>13s} {cov:5.2f} | "
              f"{cells[1][0]:>11s} {cells[1][1]:>13s} | {cells[2][0]:>11s} {cells[2][1]:>13s}")


if __name__ == "__main__":
    main()
