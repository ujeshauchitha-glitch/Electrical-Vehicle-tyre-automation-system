"""Truth model with a different structure from the estimator's forward model.

backend/estimator.py inverts closed-form equations (single-inertia resonance,
linear stiffness, radius = free - deflection/3, constant efficiency, ...). A
simulator built from the same equations cannot show model error, and
backend/mismatch.py only rescales the simulator's outputs. This module
generates the same 18-channel measurement vector from physics with a
different structure:

* resonance: two-inertia belt/rim torsional model, upper mode, pressure
  stiffening as a power law of absolute pressure, centrifugal stiffening with
  speed. The tread sensitivity of the frequency depends on the rim inertia,
  which is a study parameter, because that is the least certain part of the
  resonance-to-tread link.
* rolling radius: nonlinear vertical stiffness, radius growth with speed,
  per-tyre size tolerance, wheel slip on the driven axle.
* rolling resistance: speed, load and tread dependence, exponential
  temperature law.
* drivetrain: load-dependent efficiency.
* operating conditions that differ between install and service: payload, wind,
  grade, road crown, surface, speed.

Everything is ASSUMED: each constant is an order of magnitude chosen by the
authors, not a measurement or a fit. This is a structurally different model,
not an independent or validated one (same authors, same assumptions about
which effects exist). Its use is a stress test of how far the estimator's
results depend on its own equations. The only anchor to the estimator's model
is that the nominal resonance frequency at new tread, placard pressure and
25 C matches physics.first_mode_frequency.

Not modelled: tyre ageing, load dependence of the resonance, camber that is
mirrored left/right, aquaplaning, temperature gradients, speed-dependent
sensor delays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimator import SensorNoise, TyreState
from .extended import ExtraNoise, M_LATACC, M_STEER, N_EXT
from .physics import CORNERS, Vehicle, first_mode_frequency

TREAD_DENSITY = 1100.0       # kg/m^3
TREAD_LAND_RATIO = 0.70
TREAD_WIDTH_M = 0.205
DRIVELINE_STIFFNESS = 8000.0  # N.m/rad, rim-to-vehicle coupling
CENTRIFUGAL_V_C = 150.0       # m/s, scale of centrifugal stiffening
RADIUS_GROWTH = 2.5e-6        # fractional radius growth per (m/s)^2
KZ_A = 120_000.0              # N/m
KZ_B = 370.0                  # N/m per kPa
RHO_AIR = 1.20
V_REF = 22.0

_AXLE_SHARE = np.array([Vehicle.front_weight, Vehicle.front_weight,
                        1.0 - Vehicle.front_weight, 1.0 - Vehicle.front_weight])
_PAYLOAD_SHARE = np.array([0.15, 0.15, 0.35, 0.35])


def corner_loads(payload_kg: float = 0.0) -> np.ndarray:
    return Vehicle.g * (Vehicle.mass * _AXLE_SHARE / 2.0 + payload_kg * _PAYLOAD_SHARE)


def tread_inertia(tread_mm):
    r = Vehicle.r_belt + np.asarray(tread_mm) / 2000.0
    mass = (TREAD_DENSITY * TREAD_LAND_RATIO * TREAD_WIDTH_M
            * 2.0 * np.pi * r * np.asarray(tread_mm) / 1000.0)
    return mass * r * r


def upper_mode_hz(jb, jr, ks, kd=DRIVELINE_STIFFNESS):
    """Upper natural frequency (Hz) of belt (jb) - sidewall (ks) - rim (jr) - driveline (kd)."""
    if np.isinf(jr):
        return np.sqrt(ks / jb) / (2.0 * np.pi)
    a = jb * jr
    b = jb * (ks + kd) + jr * ks
    c = ks * kd
    x = (b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return np.sqrt(x) / (2.0 * np.pi)


def nominal_sidewall_stiffness(rim_inertia: float) -> float:
    """ks that puts the upper mode at the estimator's nominal frequency for a new tyre."""
    w2 = (2.0 * np.pi * first_mode_frequency(Vehicle.tread_new, Vehicle.p_placard)) ** 2
    jb = 0.55 + float(tread_inertia(Vehicle.tread_new))
    if np.isinf(rim_inertia):
        return jb * w2
    kd = DRIVELINE_STIFFNESS
    return jb * w2 * (kd - rim_inertia * w2) / (kd - (jb + rim_inertia) * w2)


@dataclass(frozen=True, eq=False)
class Conditions:
    v_ms: float = V_REF
    payload_kg: float = 0.0
    wind_ms: float = 0.0
    grade_rad: float = 0.0
    crown_rad: float = 0.0
    surface_scale: float = 1.0
    accel_ms2: float = 0.0

    @staticmethod
    def draw_install(rng: np.random.Generator, level: float = 1.0) -> "Conditions":
        return Conditions(
            v_ms=V_REF + level * rng.uniform(-4.0, 4.0),
            payload_kg=level * rng.uniform(0.0, 80.0),
            wind_ms=float(rng.normal(0.0, 1.0 * level)),
            grade_rad=float(rng.normal(0.0, 0.003 * level)),
            crown_rad=float(rng.normal(0.0, 0.015 * level)),
            surface_scale=float(1.0 + rng.normal(0.0, 0.05 * level)),
            accel_ms2=float(rng.normal(0.0, 0.05 * level)),
        )

    @staticmethod
    def draw_service(rng: np.random.Generator, level: float = 1.0) -> "Conditions":
        return Conditions(
            v_ms=V_REF + level * rng.uniform(-8.0, 10.0),
            payload_kg=level * rng.uniform(0.0, 300.0),
            wind_ms=float(rng.normal(0.0, 1.5 * level)),
            grade_rad=float(rng.normal(0.0, 0.010 * level)),
            crown_rad=float(rng.normal(0.0, 0.015 * level)),
            surface_scale=float(1.0 + rng.normal(0.0, 0.08 * level)),
            accel_ms2=float(rng.normal(0.0, 0.10 * level)),
        )


@dataclass(frozen=True, eq=False)
class HiddenVehicle:
    """Persistent per-vehicle parameters the estimator does not know."""
    rim_inertia: float
    ks_nom: float
    j_belt0: np.ndarray
    ks_scale: np.ndarray
    beta: np.ndarray
    f_temp: float
    radius_tol_m: np.ndarray
    lam: float
    kz_scale: np.ndarray
    crr0: float
    crr_p_exp: float
    crr_span: float
    crr_T: float
    c_alpha_scale: float
    camber_ratio: float
    cda: float
    latacc_bias: float
    steer_bias_deg: float
    install_tread_offset: np.ndarray

    @staticmethod
    def draw(rng: np.random.Generator, rim_inertia: float = np.inf,
             level: float = 1.0) -> "HiddenVehicle":
        n = lambda s, size=None: rng.normal(0.0, s * level, size)  # noqa: E731
        return HiddenVehicle(
            rim_inertia=rim_inertia,
            ks_nom=nominal_sidewall_stiffness(rim_inertia),
            j_belt0=0.55 + n(0.03, 4),
            ks_scale=1.0 + n(0.02, 4),
            beta=np.clip(0.55 + n(0.05, 4), 0.2, 0.9),
            f_temp=float(-0.002 + n(0.001)),
            radius_tol_m=n(0.001, 4),
            lam=float(np.clip(0.34 + n(0.015), 0.2, 0.5)),
            kz_scale=1.0 + n(0.04, 4),
            crr0=float(0.0090 * (1.0 + n(0.08))),
            crr_p_exp=float(0.50 + n(0.05)),
            crr_span=float(np.clip(0.20 + n(0.05), 0.05, 0.40)),
            crr_T=float(-0.009 + n(0.0015)),
            c_alpha_scale=float(1.0 + n(0.12)),
            camber_ratio=float(0.08 * (1.0 + n(0.25))),
            cda=float(Vehicle.CdA * (1.0 + n(0.05))),
            latacc_bias=float(n(0.05)),
            steer_bias_deg=float(n(0.10)),
            install_tread_offset=n(0.25, 4),
        )

    def resonance_hz(self, tread, press, temp, v_ms):
        jb = self.j_belt0 + tread_inertia(tread)
        ks = (self.ks_nom * self.ks_scale
              * ((press + Vehicle.p_atm) / (Vehicle.p_placard + Vehicle.p_atm)) ** self.beta
              * (1.0 + 2.0 * self.f_temp * (temp - Vehicle.T_ref))
              * (1.0 + (v_ms / CENTRIFUGAL_V_C) ** 2) / (1.0 + (V_REF / CENTRIFUGAL_V_C) ** 2))
        return upper_mode_hz(jb, self.rim_inertia, ks)

    def rolling_radius(self, tread, press, fz, v_ms):
        kz = self.kz_scale * (KZ_A + KZ_B * press)
        r_free = (Vehicle.r_belt + tread / 1000.0) * (1.0 + RADIUS_GROWTH * v_ms ** 2)
        return r_free - self.lam * fz / kz + self.radius_tol_m

    def rolling_coeff(self, tread, press, temp, fz, v_ms, surface_scale):
        f_load = (fz / corner_loads(0.0)) ** -0.1
        f_tread = (1.0 - self.crr_span) + self.crr_span * tread / Vehicle.tread_new
        f_temp = np.exp(self.crr_T * (temp - Vehicle.T_ref))
        f_speed = (1.0 + 0.4 * (v_ms / 40.0) ** 2) / (1.0 + 0.4 * (V_REF / 40.0) ** 2)
        return (self.crr0 * (Vehicle.p_placard / press) ** self.crr_p_exp
                * f_load * f_tread * f_temp * f_speed * surface_scale)


def tread_sensitivity_gain(rim_inertia: float) -> float:
    """d(f)/d(tread) of the nominal truth model relative to the estimator's, at new tread."""
    h, dh = Vehicle.tread_new, 0.05
    hv = HiddenVehicle.draw(np.random.default_rng(0), rim_inertia, level=0.0)
    p, T = Vehicle.p_placard, Vehicle.T_ref
    truth = (hv.resonance_hz(np.array([h + dh]), np.array([p]), np.array([T]), V_REF)
             - hv.resonance_hz(np.array([h - dh]), np.array([p]), np.array([T]), V_REF)) / (2 * dh)
    est = (first_mode_frequency(h + dh, p) - first_mode_frequency(h - dh, p)) / (2 * dh)
    return float(truth[0] / est)


def measure_truth(state: TyreState, hv: HiddenVehicle, cond: Conditions,
                  rng: np.random.Generator, noise: type = SensorNoise,
                  extra: type = ExtraNoise, esc: bool = True, gnss: bool = True):
    """Return (z[18], T_meas, estimator_inputs) generated by the truth model."""
    tread = np.array([state.tread[c] for c in CORNERS])
    press = np.array([state.pressure[c] for c in CORNERS])
    temp = np.array([state.temp[c] for c in CORNERS])
    fz = corner_loads(cond.payload_kg)
    mass = Vehicle.mass + cond.payload_kg
    v = cond.v_ms

    z = np.full(N_EXT, np.nan)
    z[0:4] = (np.round(press / noise.tpms_quantum) * noise.tpms_quantum
              + rng.normal(0.0, noise.tpms_sigma, 4))
    z[4:8] = hv.resonance_hz(tread, press, temp, v) + rng.normal(0.0, noise.freq_sigma, 4)
    r = hv.rolling_radius(tread, press, fz, v)
    z[8] = r[1] / r[0] + rng.normal(0.0, noise.ratio_sigma)
    z[9] = r[3] / r[2] + rng.normal(0.0, noise.ratio_sigma)

    crr = hv.rolling_coeff(tread, press, temp, fz, v, cond.surface_scale)
    c_alpha = hv.c_alpha_scale * Vehicle.C_alpha
    f_aero = 0.5 * RHO_AIR * hv.cda * (v + cond.wind_ms) ** 2
    f_toe = 2.0 * c_alpha * np.deg2rad(state.toe) ** 2
    f_trac = (f_aero + np.cos(cond.grade_rad) * float(np.sum(crr * fz))
              + mass * Vehicle.g * np.sin(cond.grade_rad)
              + mass * cond.accel_ms2 + f_toe)

    accel_meas = cond.accel_ms2 + float(rng.normal(0.0, 0.02))
    grade_meas = cond.grade_rad + float(rng.normal(0.0, 0.001))
    f_aero_nominal = 0.5 * Vehicle.rho_air * Vehicle.CdA * v ** 2
    m_g = Vehicle.mass * Vehicle.g
    c_eq = (f_trac - f_aero_nominal - m_g * np.sin(grade_meas)
            - Vehicle.mass * accel_meas) / m_g
    z[10] = c_eq * (1.0 + rng.normal(0.0, noise.roadload_frac))

    eta = 0.93 - 0.06 * np.exp(-abs(f_trac) / 300.0)
    z[11] = (f_trac * float(r[2:].mean()) / (Vehicle.gear_ratio * eta)
             + rng.normal(0.0, noise.motortorque_sigma))

    if esc:
        cg = hv.camber_ratio * c_alpha
        a_true = (4.0 * cg * np.deg2rad(state.camber) / mass + Vehicle.g * cond.crown_rad)
        z[M_LATACC] = a_true + hv.latacc_bias + rng.normal(0.0, extra.latacc_sigma)
        z[M_STEER] = (-np.rad2deg(mass * a_true / (2.0 * c_alpha)) + hv.steer_bias_deg
                      + rng.normal(0.0, extra.steer_sigma))
    if gnss:
        slip = np.zeros(4)
        slip[2:] = (max(f_trac, 0.0) / 2.0) / (10.0 * fz[2:])
        scale = rng.normal(0.0, extra.gnss_scale_sigma)
        z[14:18] = r * (1.0 + scale - slip + rng.normal(0.0, extra.wheel_speed_sigma, 4))

    T_meas = float(temp.mean() + rng.normal(0.0, noise.temp_sigma))
    return z, T_meas, {"v_ms": v, "accel_ms2": accel_meas, "grade_rad": grade_meas}
