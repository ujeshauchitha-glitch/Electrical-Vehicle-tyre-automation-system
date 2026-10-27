"""Stress test: estimator vs a truth model that differs from the estimator's model.

SIMULATION ONLY, with ASSUMED mismatch magnitudes (backend/mismatch.py). Compares
the plain 18-channel estimator with the install-baseline calibrated one.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.calibrated import calibrate, estimate_calibrated  # noqa: E402
from backend.estimator import CORNERS, IDX_PRESS, IDX_TREAD, M_FREQ, TyreState  # noqa: E402
from backend.extended import ExtraNoise, estimate_ext  # noqa: E402
from backend.mismatch import Mismatch, measure_mismatched  # noqa: E402

N_INSTALL = 20


class CalibratedNoise(ExtraNoise):
    radius_tolerance_mm = 0.3


def install_snapshots(rng, mm, **kw):
    rows, temps = [], []
    for _ in range(N_INSTALL):
        s = TyreState(tread={c: 8.0 for c in CORNERS}, toe=0.0, camber=0.0,
                      temp={c: 25.0 + rng.normal(0, 1.0) for c in CORNERS})
        z, T = measure_mismatched(s, rng, mm, **kw)
        rows.append(z)
        temps.append(T)
    return np.array(rows), float(np.mean(temps))


def run(n, seed, level, drop_res=False, gnss=True, install_tread_error_mm=0.0):
    rng = np.random.default_rng(seed)
    out = {"plain": [], "calibrated": []}
    for _ in range(n):
        mm = Mismatch.draw(rng, level)
        zi, ti = install_snapshots(rng, mm, gnss=gnss)
        baseline = calibrate(zi, ti, 8.0 + install_tread_error_mm)
        s = TyreState.random(rng)
        z, T = measure_mismatched(s, rng, mm, gnss=gnss)
        if drop_res:
            z[M_FREQ] = np.nan
        truth = np.r_[[s.tread[c] for c in CORNERS], [s.pressure[c] for c in CORNERS],
                      s.toe ** 2, s.camber]
        xp, Pp = estimate_ext(z, T)
        xc, Pc = estimate_calibrated(z, T, baseline, noise=CalibratedNoise)
        for key, x, P in (("plain", xp, Pp), ("calibrated", xc, Pc)):
            out[key].append((x - truth, np.sqrt(np.diag(P))))
    res = {}
    for key, rows in out.items():
        e = np.array([r[0] for r in rows])
        sg = np.array([r[1] for r in rows])
        rms = lambda a: float(np.sqrt(np.mean(a ** 2)))  # noqa: E731
        res[key] = {
            "tread": rms(e[:, IDX_TREAD]), "tread_sigma": float(np.mean(sg[:, IDX_TREAD])),
            "press": rms(e[:, IDX_PRESS]), "toe2": rms(e[:, 8]),
            "camber": rms(e[:, 9]), "camber_sigma": float(np.mean(sg[:, 9])),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"SIMULATION ONLY, assumed mismatch; {a.n} vehicles, {N_INSTALL} install snapshots, seed {a.seed}")
    print(f"{'case':34s} {'estimator':11s} {'tread':>6s} {'(claimed)':>9s} {'press':>6s} "
          f"{'toe^2':>6s} {'camber':>7s} {'(claimed)':>9s}")
    cases = [
        ("no mismatch (level 0)", dict(level=0.0)),
        ("mismatch level 1", dict(level=1.0)),
        ("mismatch level 2", dict(level=2.0)),
        ("level 1, no resonance", dict(level=1.0, drop_res=True)),
        ("level 1, install tread off by 0.5 mm", dict(level=1.0, install_tread_error_mm=0.5)),
    ]
    for name, kw in cases:
        r = run(a.n, a.seed, **kw)
        for key in ("plain", "calibrated"):
            v = r[key]
            print(f"{name if key == 'plain' else '':34s} {key:11s} {v['tread']:6.2f} "
                  f"{v['tread_sigma']:9.2f} {v['press']:6.2f} {v['toe2']:6.2f} "
                  f"{v['camber']:7.2f} {v['camber_sigma']:9.2f}")


if __name__ == "__main__":
    main()
