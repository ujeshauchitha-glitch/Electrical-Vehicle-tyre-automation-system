"""Benchmark backend.estimator.estimate on seeded random states."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.estimator import TyreState, estimate, measure  # noqa: E402


def main(n: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    zs = [measure(TyreState.random(rng), rng, include_motor_torque=True) for _ in range(n)]
    estimate(*zs[0])
    t0 = time.perf_counter()
    xs = np.array([estimate(z, T)[0] for z, T in zs])
    dt = time.perf_counter() - t0
    print(f"{n} estimates: {dt:.3f} s ({1e3 * dt / n:.2f} ms each)")
    print(f"checksum {xs.sum():.9f}")


if __name__ == "__main__":
    main()
