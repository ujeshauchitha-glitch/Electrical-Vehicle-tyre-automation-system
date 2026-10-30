"""Multi-seed study harness: truth-model cases, estimator variants, metrics, CIs.

Cases are generated once per vehicle and shared by every estimator variant, so
differences between variants are paired and their confidence intervals are
tighter than for independent runs. The confidence interval is over seeds (each
seed is an independent batch of vehicles), not over vehicles within a seed.
SIMULATION ONLY; the truth model is assumed physics (backend/truth_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable

import numpy as np

from .calibrated import (
    ASSUMED_F_TEMP_COEFF, ASSUMED_F_TEMP_SIGMA, Baseline, apply_baseline,
    calibrate, model_variance,
)
from .estimator import CORNERS, M_FREQ, TyreState
from .extended import ExtraNoise, estimate_ext
from .physics import Vehicle
from .truth_model import Conditions, HiddenVehicle, measure_truth

N_INSTALL = 20
T_975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


class CalibratedNoise(ExtraNoise):
    radius_tolerance_mm = 0.3


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    """Mean and 95 % half-width (Student t) of independent per-seed values."""
    v = np.asarray(list(values), dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan")
    t = T_975[n - 2] if n - 1 <= len(T_975) else 1.96
    return float(v.mean()), float(t * v.std(ddof=1) / np.sqrt(n))


@dataclass
class Case:
    truth: np.ndarray
    z: np.ndarray
    T: float
    inputs: dict
    zi: np.ndarray
    ti: float
    _baselines: dict = field(default_factory=dict, repr=False)

    def baseline(self, coeff: float = ASSUMED_F_TEMP_COEFF) -> Baseline:
        if coeff not in self._baselines:
            self._baselines[coeff] = calibrate(self.zi, self.ti, Vehicle.tread_new, coeff)
        return self._baselines[coeff]


def generate_cases(n: int, seed: int, level: float = 1.0, rim_inertia: float = np.inf,
                   gnss: bool = True, n_install: int = N_INSTALL) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        hv = HiddenVehicle.draw(rng, rim_inertia, level)
        p_install = Vehicle.p_placard + rng.normal(0.0, 4.0 * level, 4)
        rows, temps = [], []
        for _ in range(n_install):
            s = TyreState(
                tread={c: Vehicle.tread_new + hv.install_tread_offset[i] for i, c in enumerate(CORNERS)},
                pressure={c: float(p_install[i]) for i, c in enumerate(CORNERS)},
                temp={c: 25.0 + rng.normal(0.0, 1.0) for c in CORNERS}, toe=0.0, camber=0.0)
            z, T, _ = measure_truth(s, hv, Conditions.draw_install(rng, level), rng, gnss=gnss)
            rows.append(z)
            temps.append(T)
        s = TyreState.random(rng)
        z, T, inputs = measure_truth(s, hv, Conditions.draw_service(rng, level), rng, gnss=gnss)
        truth = np.r_[[s.tread[c] for c in CORNERS], [s.pressure[c] for c in CORNERS],
                      s.toe ** 2, s.camber]
        cases.append(Case(truth, z, T, inputs, np.array(rows), float(np.mean(temps))))
    return cases


Estimator = Callable[[Case], tuple]


def plain_estimator(case: Case):
    return estimate_ext(case.z, case.T, **case.inputs)


ALL_TERMS = ("f_scale", "ratio", "radius", "latacc", "steer")
VARIANCE_MODES = ("none", "resonance", "roadload", "full")


def calibrated_estimator(drop: Iterable[str] = (), variance: str = "full",
                         temp_comp: bool = True, noise: type = CalibratedNoise) -> Estimator:
    """Install-baseline estimator with selected terms switched off.

    drop: subset of ALL_TERMS whose baseline correction is replaced by identity.
    variance: which parts of the extra model variance are kept.
    temp_comp: False removes the resonance temperature compensation and widens
    the assumed coefficient sigma to cover the uncompensated value.
    """
    drop = set(drop)
    coeff = ASSUMED_F_TEMP_COEFF if temp_comp else 0.0
    sigma_coeff = ASSUMED_F_TEMP_SIGMA if temp_comp else float(np.hypot(0.002, ASSUMED_F_TEMP_SIGMA))

    def run(case: Case):
        b = case.baseline(coeff)
        patch = {}
        if "f_scale" in drop:
            patch["f_scale"] = np.ones(4)
        if "ratio" in drop:
            patch["ratio_offset"] = np.zeros(2)
        if "radius" in drop:
            patch["radius_offset"] = np.zeros(4)
        if "latacc" in drop:
            patch["latacc_bias"] = 0.0
        if "steer" in drop:
            patch["steer_bias"] = 0.0
        b = replace(b, **patch)
        extra = model_variance(case.T, b, sigma_coeff=sigma_coeff)
        keep = np.zeros_like(extra)
        if variance in ("resonance", "full"):
            keep[4:8, 4:8] = extra[4:8, 4:8]
        if variance in ("roadload", "full"):
            keep[10:12, 10:12] = extra[10:12, 10:12]
        return estimate_ext(apply_baseline(case.z, case.T, b, coeff), case.T,
                            noise=noise, extra_variance=keep, **case.inputs)

    return run


def _group_metrics(errors: np.ndarray, covs: np.ndarray, idx) -> dict:
    e = errors[:, idx]
    var = np.stack([np.diag(c)[idx] for c in covs])
    sigma = np.sqrt(var)
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "sigma": float(np.sqrt(np.mean(var))),
        "cover1": float(np.mean(np.abs(e) < sigma)),
    }


def evaluate(cases: list[Case], estimator: Estimator, drop_resonance: bool = False) -> dict:
    errs, covs = [], []
    for case in cases:
        if drop_resonance:
            case = replace(case, z=case.z.copy())
            case.z[M_FREQ] = np.nan
        x, P = estimator(case)
        errs.append(x - case.truth)
        covs.append(P)
    errs, covs = np.array(errs), np.array(covs)
    out = {
        "tread": _group_metrics(errs, covs, list(range(0, 4))),
        "press": _group_metrics(errs, covs, list(range(4, 8))),
        "camber": _group_metrics(errs, covs, [9]),
    }
    nees = [e[:4] @ np.linalg.solve(P[:4, :4], e[:4]) / 4.0 for e, P in zip(errs, covs)]
    out["tread"]["nees"] = float(np.mean(nees))
    return out


def run_study(estimators: Dict[str, Estimator], seeds: Iterable[int], n: int,
              level: float = 1.0, rim_inertia: float = np.inf, gnss: bool = True,
              drop_resonance: bool = False) -> Dict[str, Dict[str, Dict[str, list]]]:
    """{estimator: {group: {metric: [value per seed]}}}."""
    res: Dict[str, Dict[str, Dict[str, list]]] = {k: {} for k in estimators}
    for seed in seeds:
        cases = generate_cases(n, seed, level, rim_inertia, gnss)
        for name, est in estimators.items():
            m = evaluate(cases, est, drop_resonance)
            for group, metrics in m.items():
                for key, val in metrics.items():
                    res[name].setdefault(group, {}).setdefault(key, []).append(val)
    return res


def paired_difference(res, name: str, baseline: str, group: str, metric: str = "rmse"):
    d = np.array(res[name][group][metric]) - np.array(res[baseline][group][metric])
    return mean_ci(d)
