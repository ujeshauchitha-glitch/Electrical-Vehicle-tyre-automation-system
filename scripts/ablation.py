"""Channel ablation and filter-consistency report for backend.estimator (simulation only).

Each row drops measurement channels by setting them NaN, then scores the
estimator against the known simulated truth over seeded random states.
NEES is the mean normalised estimation error squared over the states listed
in ``--nees-states``; a consistent filter gives about the state count.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.estimator import (  # noqa: E402
    CORNERS, IDX_PRESS, IDX_TREAD, IDX_TOESQ, M_FREQ, M_MOTORTORQUE, M_PRESS,
    M_RATIO, M_ROADLOAD, TyreState, estimate, measure,
)

VARIANTS = {
    "full": [],
    "no resonance": [M_FREQ],
    "no motor torque": [M_MOTORTORQUE],
    "no road load, no torque": [M_ROADLOAD, M_MOTORTORQUE],
    "no TPMS": [M_PRESS],
    "no wheel-speed ratio": [M_RATIO],
    "radius+TPMS only": [M_FREQ, M_ROADLOAD, M_MOTORTORQUE],
}


def truth_vector(state):
    return np.array([state.tread[c] for c in CORNERS] + [state.pressure[c] for c in CORNERS]
                    + [state.toe ** 2, state.camber])


def run(n, seed, drop):
    rng = np.random.default_rng(seed)
    err, nees = [], []
    for _ in range(n):
        state = TyreState.random(rng)
        z, T = measure(state, rng, include_motor_torque=True)
        z = z.copy()
        for ch in drop:
            z[ch] = np.nan
        x, P = estimate(z, T)
        e = x - truth_vector(state)
        err.append(e)
        free = np.diag(P) > 0
        nees.append(float(e[free] @ np.linalg.solve(P[np.ix_(free, free)], e[free])) / free.sum())
    err = np.array(err)
    return {
        "tread_rmse_mm": float(np.sqrt(np.mean(err[:, IDX_TREAD] ** 2))),
        "pressure_rmse_kpa": float(np.sqrt(np.mean(err[:, IDX_PRESS] ** 2))),
        "toe_sq_rmse": float(np.sqrt(np.mean(err[:, IDX_TOESQ] ** 2))),
        "camber_rmse_deg": float(np.sqrt(np.mean(err[:, 9] ** 2))),
        "nees_per_dof": float(np.mean(nees)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"SIMULATION ONLY: {a.n} seeded random states, seed {a.seed}")
    print(f"{'variant':26s} {'tread mm':>9s} {'press kPa':>10s} {'toe^2':>7s} {'camber':>7s} {'NEES/dof':>9s}")
    for name, drop in VARIANTS.items():
        r = run(a.n, a.seed, drop)
        print(f"{name:26s} {r['tread_rmse_mm']:9.3f} {r['pressure_rmse_kpa']:10.2f} "
              f"{r['toe_sq_rmse']:7.3f} {r['camber_rmse_deg']:7.3f} {r['nees_per_dof']:9.2f}")


if __name__ == "__main__":
    main()
