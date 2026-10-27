"""Camber and tread study for backend.extended (simulation only).

Truth is drawn from TyreState.random; camber is N(0, 0.6 deg). Read the
slip-bias and accelerometer-bias rows: they show how fast the channels'
claimed precision disappears under errors the estimator's R does not model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.estimator import CORNERS, IDX_TREAD, M_FREQ, TyreState  # noqa: E402
from backend.extended import estimate_ext, measure_ext  # noqa: E402

CASES = [
    # name, esc, gnss, drop resonance, slip bias, latacc bias (m/s^2)
    ("base 12 ch", False, False, False, 0.0, 0.0),
    ("base, no resonance", False, False, True, 0.0, 0.0),
    ("+ESC", True, False, False, 0.0, 0.0),
    ("+GNSS radius, no resonance", False, True, True, 0.0, 0.0),
    ("+both, no resonance", True, True, True, 0.0, 0.0),
    ("+both, with resonance", True, True, False, 0.0, 0.0),
    ("+GNSS no res, slip +0.3%", False, True, True, 0.003, 0.0),
    ("+GNSS no res, slip +1.0%", False, True, True, 0.010, 0.0),
    ("+ESC, accel bias 0.05 m/s2", True, False, False, 0.0, 0.05),
    ("+ESC, accel bias 0.15 m/s2", True, False, False, 0.0, 0.15),
]


def run(n, seed, esc, gnss, drop_res, slip, ay_bias):
    rng = np.random.default_rng(seed)
    tread_err, cam_err, cam_sig = [], [], []
    for _ in range(n):
        s = TyreState.random(rng)
        z, T = measure_ext(s, rng, esc=esc, gnss=gnss, slip_bias=slip)
        if drop_res:
            z[M_FREQ] = np.nan
        if esc and ay_bias:
            z[12] += ay_bias
        x, P = estimate_ext(z, T)
        tread_err.append(x[IDX_TREAD] - np.array([s.tread[c] for c in CORNERS]))
        cam_err.append(x[9] - s.camber)
        cam_sig.append(np.sqrt(P[9, 9]))
    t = np.array(tread_err)
    return (float(np.sqrt(np.mean(t ** 2))), float(np.mean(t)),
            float(np.sqrt(np.mean(np.square(cam_err)))), float(np.mean(cam_sig)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"SIMULATION ONLY: {a.n} states, seed {a.seed}; camber prior sigma 1.0 deg, "
          f"truth sigma 0.6 deg (zero-estimate RMSE ~0.6)")
    print(f"{'case':30s} {'tread RMSE':>10s} {'tread bias':>10s} {'camb RMSE':>10s} {'camb sigma':>10s}")
    for name, *args in CASES:
        r = run(a.n, a.seed, *args)
        print(f"{name:30s} {r[0]:10.3f} {r[1]:10.3f} {r[2]:10.3f} {r[3]:10.3f}")


if __name__ == "__main__":
    main()
