

import argparse
import numpy as np

class Vehicle:

    mass          = 1800.0
    front_weight  = 0.48
    g             = 9.81
    r_belt        = 0.322
    tread_new     = 8.0
    tread_legal   = 1.6

    p_placard     = 240.0
    p_atm         = 101.325

    k_z0          = 210_000.0

    J_carcass     = 0.550
    J_per_mm      = 0.0332
    K_carcass     = 39_121.0
    K_per_kPa     = 108.67

    C_rr0         = 0.0090
    p_exponent    = 0.45
    tread_rr_span = 0.20
    T_coeff       = 0.0015
    T_ref         = 25.0

    CdA           = 0.65
    rho_air       = 1.20

    C_alpha       = 55_000.0

    mu_dry        = 1.00
    wet_floor     = 0.45
    wet_tau       = 2.5

    @classmethod
    def Fz(cls, corner):

        share = cls.front_weight if corner in ("FL", "FR") else 1.0 - cls.front_weight
        return cls.mass * cls.g * share / 2.0

CORNERS = ("FL", "FR", "RL", "RR")

def compensate_pressure(p_gauge_kPa, T_tyre_C, T_ref_C=Vehicle.T_ref):
\
\
\
\
\
\
\
\
\
\
\

    p_abs = p_gauge_kPa + Vehicle.p_atm
    p_ref_abs = p_abs * (T_ref_C + 273.15) / (T_tyre_C + 273.15)
    return p_ref_abs - Vehicle.p_atm

def effective_rolling_radius(tread_mm, p_kPa, Fz_N):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    r_free = Vehicle.r_belt + tread_mm / 1000.0
    k_z = Vehicle.k_z0 * (p_kPa / Vehicle.p_placard)
    delta = Fz_N / k_z
    return r_free - delta / 3.0

def first_mode_frequency(tread_mm, p_kPa):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    J = Vehicle.J_carcass + Vehicle.J_per_mm * tread_mm
    K = Vehicle.K_carcass + Vehicle.K_per_kPa * p_kPa
    return np.sqrt(K / J) / (2.0 * np.pi)

def rolling_resistance_coeff(tread_mm, p_kPa, T_C):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    p_term = (Vehicle.p_placard / p_kPa) ** Vehicle.p_exponent
    s = Vehicle.tread_rr_span
    t_term = (1.0 - s) + s * (tread_mm / Vehicle.tread_new)
    T_term = 1.0 + Vehicle.T_coeff * (T_C - Vehicle.T_ref)
    return Vehicle.C_rr0 * p_term * t_term * T_term

def toe_drag_force(toe_deg):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    toe_rad = np.deg2rad(toe_deg)
    return 2.0 * Vehicle.C_alpha * toe_rad ** 2

def toe_drag_from_sq(toe_sq_deg2):

    return 2.0 * Vehicle.C_alpha * (np.pi / 180.0) ** 2 * toe_sq_deg2

def road_load(state, v_ms, accel_ms2=0.0, grade_rad=0.0):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    C_rr_mean = np.mean([
        rolling_resistance_coeff(state.tread[c], state.pressure[c], state.temp[c])
        for c in CORNERS
    ])
    F_roll = C_rr_mean * Vehicle.mass * Vehicle.g * np.cos(grade_rad)
    F_aero = 0.5 * Vehicle.rho_air * Vehicle.CdA * v_ms ** 2
    F_grade = Vehicle.mass * Vehicle.g * np.sin(grade_rad)
    F_inertia = Vehicle.mass * accel_ms2
    F_toe = toe_drag_force(state.toe)
    return F_roll + F_aero + F_grade + F_inertia + F_toe

def slip_stiffness(tread_mm, p_kPa, Fz_N):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    p_term = (p_kPa / Vehicle.p_placard) ** 0.2
    wear_term = 1.0 + 0.04 * (Vehicle.tread_new - tread_mm)
    return 12.0 * Fz_N * p_term * wear_term

def wet_friction(tread_mm):
\
\
\
\
\
\
\
\
\
\
\

    k = Vehicle.wet_floor
    return Vehicle.mu_dry * (k + (1 - k) * (1 - np.exp(-tread_mm / Vehicle.wet_tau)))

class TyreState:

    def __init__(self, tread=None, pressure=None, temp=None, toe=0.0, camber=0.0):
        self.tread    = dict(tread)    if tread    else {c: 6.0 for c in CORNERS}
        self.pressure = dict(pressure) if pressure else {c: Vehicle.p_placard for c in CORNERS}
        self.temp     = dict(temp)     if temp     else {c: 25.0 for c in CORNERS}
        self.toe      = toe
        self.camber   = camber

    @staticmethod
    def random(rng):

        base_front = rng.uniform(2.0, 7.8)
        base_rear  = max(1.7, base_front - rng.uniform(0.3, 2.0))
        tread = {
            "FL": base_front + rng.normal(0, 0.25),
            "FR": base_front + rng.normal(0, 0.25),
            "RL": base_rear  + rng.normal(0, 0.30),
            "RR": base_rear  + rng.normal(0, 0.30),
        }
        tread = {c: float(np.clip(v, 1.2, 8.0)) for c, v in tread.items()}
        pressure = {c: float(Vehicle.p_placard + rng.normal(0, 18.0)) for c in CORNERS}

        if rng.random() < 0.17:
            pressure[rng.choice(CORNERS)] -= rng.uniform(25, 65)
        soak = rng.uniform(0, 28)
        temp = {c: 25.0 + soak + rng.normal(0, 2.0) for c in CORNERS}
        toe = float(rng.normal(0, 0.20)) if rng.random() < 0.75 else float(rng.uniform(0.4, 1.1))
        camber = float(rng.normal(0, 0.6))
        return TyreState(tread, pressure, temp, toe, camber)

class SensorNoise:
    tpms_quantum   = 2.5
    tpms_sigma     = 3.0
    temp_sigma     = 1.5
    freq_sigma     = 0.15
    ratio_sigma    = 2.0e-4
    roadload_frac  = 0.040

def measure(state, rng, noise=SensorNoise, v_ms=22.0):
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    z = np.zeros(N_MEAS)

    for i, c in enumerate(CORNERS):
        p_raw = np.round(state.pressure[c] / noise.tpms_quantum) * noise.tpms_quantum
        z[i] = p_raw + rng.normal(0, noise.tpms_sigma)

    for i, c in enumerate(CORNERS):
        f_true = first_mode_frequency(state.tread[c], state.pressure[c])
        z[4 + i] = f_true + rng.normal(0, noise.freq_sigma)

    for k, (a, b) in enumerate((("FL", "FR"), ("RL", "RR"))):
        ra = effective_rolling_radius(state.tread[a], state.pressure[a], Vehicle.Fz(a))
        rb = effective_rolling_radius(state.tread[b], state.pressure[b], Vehicle.Fz(b))
        z[8 + k] = rb / ra + rng.normal(0, noise.ratio_sigma)

    F_total = road_load(state, v_ms)
    F_aero = 0.5 * Vehicle.rho_air * Vehicle.CdA * v_ms ** 2
    C_eq = (F_total - F_aero) / (Vehicle.mass * Vehicle.g)
    z[M_ROADLOAD] = C_eq * (1.0 + rng.normal(0, noise.roadload_frac))

    T_meas = np.mean([state.temp[c] for c in CORNERS]) + rng.normal(0, noise.temp_sigma)

    return z, float(T_meas)

N_STATE = 10
IDX_TREAD = slice(0, 4)
IDX_PRESS = slice(4, 8)
IDX_TOESQ = 8
IDX_CAMBER = 9

STATE_NAMES = [f"tread_{c}" for c in CORNERS] + [f"press_{c}" for c in CORNERS] \
              + ["toe^2", "camber"]

N_MEAS = 11
M_PRESS = slice(0, 4)
M_FREQ = slice(4, 8)
M_RATIO = slice(8, 10)
M_ROADLOAD = 10

def predict(x, T_meas=Vehicle.T_ref, v_ms=22.0):

    tread = {c: x[i] for i, c in enumerate(CORNERS)}
    press = {c: x[4 + i] for i, c in enumerate(CORNERS)}
    z = np.zeros(N_MEAS)

    z[M_PRESS] = [press[c] for c in CORNERS]

    for i, c in enumerate(CORNERS):
        z[4 + i] = first_mode_frequency(tread[c], press[c])

    for k, (a, b) in enumerate((("FL", "FR"), ("RL", "RR"))):
        ra = effective_rolling_radius(tread[a], press[a], Vehicle.Fz(a))
        rb = effective_rolling_radius(tread[b], press[b], Vehicle.Fz(b))
        z[8 + k] = rb / ra

    C_rr = np.mean([rolling_resistance_coeff(tread[c], press[c], T_meas)
                    for c in CORNERS])
    F_toe = toe_drag_from_sq(x[IDX_TOESQ])
    z[M_ROADLOAD] = C_rr + F_toe / (Vehicle.mass * Vehicle.g)

    return z

def jacobian(x, T_meas=Vehicle.T_ref, v_ms=22.0, eps=1e-5):
\
\
\
\
\
\

    H = np.zeros((N_MEAS, N_STATE))
    for j in range(N_STATE):
        dx = np.zeros(N_STATE)
        dx[j] = eps
        H[:, j] = (predict(x + dx, T_meas, v_ms)
                   - predict(x - dx, T_meas, v_ms)) / (2 * eps)
    return H

def measurement_covariance(noise=SensorNoise, z_ref=None):

    R = np.zeros(N_MEAS)

    R[M_PRESS] = noise.tpms_sigma ** 2 + (noise.tpms_quantum ** 2) / 12.0
    R[M_FREQ] = noise.freq_sigma ** 2
    R[M_RATIO] = noise.ratio_sigma ** 2
    C_ref = z_ref[M_ROADLOAD] if z_ref is not None else Vehicle.C_rr0
    R[M_ROADLOAD] = (noise.roadload_frac * C_ref) ** 2
    return np.diag(R)

def prior():
\
\
\
\
\

    x0 = np.zeros(N_STATE)
    x0[IDX_TREAD] = 5.0
    x0[IDX_PRESS] = Vehicle.p_placard
    x0[IDX_TOESQ] = 0.10
    x0[IDX_CAMBER] = 0.0

    P0 = np.diag([
        2.5 ** 2, 2.5 ** 2, 2.5 ** 2, 2.5 ** 2,
        40.0 ** 2, 40.0 ** 2, 40.0 ** 2, 40.0 ** 2,
        0.40 ** 2,
        1.0 ** 2,
    ])
    return x0, P0

def estimate(z, T_meas=Vehicle.T_ref, iters=6, v_ms=22.0):
\
\
\
\
\
\
\
\
\
\
\
\

    x0, P0 = prior()
    R_inv = np.linalg.inv(measurement_covariance(z_ref=z))
    P0_inv = np.linalg.inv(P0)

    x = x0.copy()
    for _ in range(iters):
        H = jacobian(x, T_meas, v_ms)
        residual = z - predict(x, T_meas, v_ms)
        A = H.T @ R_inv @ H + P0_inv
        b = H.T @ R_inv @ residual + P0_inv @ (x0 - x)
        dx = np.linalg.solve(A, b)
        x = x + dx

        x[IDX_TREAD] = np.clip(x[IDX_TREAD], 0.5, 9.0)
        x[IDX_PRESS] = np.clip(x[IDX_PRESS], 100.0, 350.0)
        x[IDX_TOESQ] = max(0.0, x[IDX_TOESQ])
        if np.max(np.abs(dx)) < 1e-7:
            break

    H = jacobian(x, T_meas, v_ms)
    P = np.linalg.inv(H.T @ R_inv @ H + P0_inv)
    return x, P

def torque_limit(tread_est_mm, tread_sigma_mm, pressure_kPa, wet=True, safety=0.85):
\
\
\
\
\
\
\
\
\
\

    tread_lcb = max(0.5, tread_est_mm - 2.0 * tread_sigma_mm)
    mu = wet_friction(tread_lcb) if wet else Vehicle.mu_dry
    Fz_drive = Vehicle.Fz("RL") + Vehicle.Fz("RR")
    r_eff = effective_rolling_radius(tread_est_mm, pressure_kPa, Vehicle.Fz("RL"))
    F_max = mu * Fz_drive * safety
    return F_max * r_eff, mu, tread_lcb

def recoverable_energy(state):
\
\
\
\
\
\
\

    v = 22.0
    F_now = road_load(state, v)
    fixed = TyreState(
        tread=state.tread,
        pressure={c: Vehicle.p_placard for c in CORNERS},
        temp=state.temp,
        toe=0.0,
    )
    F_fixed = road_load(fixed, v)
    return 100.0 * (F_now - F_fixed) / F_fixed

def sensitivity_table():
\
\
\
\
\
\

    t0, p0 = 5.0, Vehicle.p_placard
    Fz = Vehicle.Fz("FL")
    d_t, d_p = 0.5, 10.0

    def d(fn, dt, dp):
        return fn(t0 + dt, p0 + dp) - fn(t0, p0)

    f_dt = d(first_mode_frequency, d_t, 0) / d_t
    f_dp = d(first_mode_frequency, 0, d_p) / d_p
    r_dt = (effective_rolling_radius(t0 + d_t, p0, Fz)
            - effective_rolling_radius(t0, p0, Fz)) * 1000 / d_t
    r_dp = (effective_rolling_radius(t0, p0 + d_p, Fz)
            - effective_rolling_radius(t0, p0, Fz)) * 1000 / d_p
    c0 = rolling_resistance_coeff(t0, p0, 25)
    c_dt = 100 * (rolling_resistance_coeff(t0 + d_t, p0, 25) - c0) / c0 / d_t
    c_dp = 100 * (rolling_resistance_coeff(t0, p0 + d_p, 25) - c0) / c0 / d_p

    print("\n  SENSITIVITY OF EACH OBSERVABLE TO EACH FAULT")
    print("  " + "-" * 68)
    print(f"  {'observable':<28}{'per +1 mm tread':>20}{'per +10 kPa':>20}")
    print("  " + "-" * 68)
    print(f"  {'first mode frequency':<28}{f_dt:>+17.3f} Hz{f_dp * d_p:>+17.3f} Hz")
    print(f"  {'effective rolling radius':<28}{r_dt:>+17.3f} mm{r_dp * d_p:>+17.3f} mm")
    print(f"  {'rolling resistance':<28}{c_dt:>+17.2f} %{c_dp * d_p:>+18.2f} %")
    print(f"  {'camber':<28}{'(no dependence)':>20}{'(no dependence)':>20}")
    print("  " + "-" * 68)
    print("  Read the signs. Tread and pressure push the frequency in OPPOSITE")
    print("  directions, which is what lets one observable resolve two faults")
    print("  once TPMS pins the pressure term. Both push rolling radius the")
    print(f"  SAME way, and 1 mm of tread == {abs(r_dt / r_dp):.0f} kPa there, so the")
    print("  wheel-speed channel is useless on its own and excellent as a")
    print("  differential check between axle partners.")

def single_vehicle_demo(rng):
    print("\n" + "=" * 72)
    print("  DEMO 1  -  ONE VEHICLE, END TO END")
    print("=" * 72)

    truth = TyreState.random(rng)
    z, T_meas = measure(truth, rng)
    x, P = estimate(z, T_meas)
    sigma = np.sqrt(np.diag(P))

    print("\n  TYRE STATE  (estimator never saw the truth column)")
    print("  " + "-" * 68)
    print(f"  {'corner':<10}{'true tread':>12}{'estimated':>14}{'error':>10}"
          f"{'true kPa':>11}{'est kPa':>10}")
    print("  " + "-" * 68)
    for i, c in enumerate(CORNERS):
        print(f"  {c:<10}{truth.tread[c]:>9.2f} mm{x[i]:>9.2f} mm"
              f"{x[i] - truth.tread[c]:>+8.2f} mm"
              f"{truth.pressure[c]:>9.0f}{x[4 + i]:>10.0f}")
    print("  " + "-" * 68)
    toe_est = np.sqrt(max(0.0, x[IDX_TOESQ]))
    print(f"  {'|toe| deg':<10}{abs(truth.toe):>12.2f}{toe_est:>14.2f}"
          f"{toe_est - abs(truth.toe):>+10.2f}   <- magnitude only, sign is even")
    print(f"  {'camber':<10}{truth.camber:>12.2f}{x[IDX_CAMBER]:>14.2f}"
          f"{x[IDX_CAMBER] - truth.camber:>+10.2f}   <- unobservable, see below")

    print("\n  MAINTENANCE VIEW  -  pressures normalised to 25 C")
    print("  " + "-" * 68)
    print(f"  measured tyre temperature: {T_meas:.0f} C"
          f"   (placard is {Vehicle.p_placard:.0f} kPa cold)")
    for i, c in enumerate(CORNERS):
        cold = compensate_pressure(x[4 + i], T_meas)
        flag = "  LOW - check for a leak" if cold < Vehicle.p_placard - 20 else ""
        print(f"  {c:<6} running {x[4 + i]:>6.0f} kPa   ->   cold-equivalent"
              f" {cold:>6.0f} kPa{flag}")
    print("  A hot tyre reads high. Judging inflation on the running pressure")
    print("  makes every warm tyre look healthy and every cold morning look")
    print("  like a puncture, so the report is normalised and the physics is")
    print("  not. Same number, two different consumers.")

    print("\n  WHAT THE ESTIMATOR LEARNED  (prior sigma -> posterior sigma)")
    print("  " + "-" * 68)
    _, P0 = prior()
    s0 = np.sqrt(np.diag(P0))
    for j, name in enumerate(STATE_NAMES):
        shrink = s0[j] / sigma[j]
        bar = "#" * min(40, int(shrink))
        note = "  NO INFORMATION" if shrink < 1.05 else ""
        print(f"  {name:<12}{s0[j]:>8.2f} -> {sigma[j]:>6.3f}   x{shrink:>6.1f}  {bar}{note}")
    print("  " + "-" * 68)
    print("  Camber's variance did not move. Nothing the vehicle measures")
    print("  depends on it. This is not a bug in the estimator - it is the")
    print("  reason a drive-over depot scanner has to exist.")

    print("\n  LAYER 2  -  TORQUE CEILING FROM THE ESTIMATED STATE")
    print("  " + "-" * 68)
    t_drive = float(np.mean([x[2], x[3]]))
    s_drive = float(np.mean([sigma[2], sigma[3]]))
    p_drive = float(np.mean([x[6], x[7]]))
    T_wet, mu_wet_est, lcb = torque_limit(t_drive, s_drive, p_drive, wet=True)
    T_dry, mu_dry_est, _ = torque_limit(t_drive, s_drive, p_drive, wet=False)
    T_ref_new, _, _ = torque_limit(Vehicle.tread_new, 0.0, Vehicle.p_placard, wet=True)

    print(f"  drive-axle tread estimate      {t_drive:>8.2f} mm  +/- {s_drive:.2f}")
    print(f"  lower confidence bound (2 s)   {lcb:>8.2f} mm   <- the number used")
    print(f"  peak wet friction at that LCB  {mu_wet_est:>8.2f}")
    print(f"  torque ceiling, wet            {T_wet:>8.0f} N.m")
    print(f"  torque ceiling, dry            {T_dry:>8.0f} N.m")
    print(f"  vs a new tyre in the wet       {100 * (T_wet / T_ref_new - 1):>+8.1f} %")
    print("\n  Every EV already cuts torque AFTER a wheel slips. Knowing the")
    print("  tread lets it cut torque BEFORE. That is the only thing Layer 1")
    print("  buys Layer 2, and it is enough.")

    waste = recoverable_energy(truth)
    print("\n  ENERGY  -  HOW MUCH OF THE ROAD LOAD IS AVOIDABLE TODAY")
    print("  " + "-" * 68)
    print(f"  excess road load vs placard pressure and zero toe:  {waste:>+6.2f} %")
    print("  Tread is deliberately held fixed in that comparison. Worn tread")
    print("  LOWERS rolling resistance, so counting it as waste would be")
    print("  physically backwards and would inflate the number dishonestly.")

def validation_sweep(rng, n_trials):
    print("\n" + "=" * 72)
    print(f"  DEMO 2  -  VALIDATION OVER {n_trials} RANDOM VEHICLES")
    print("=" * 72)

    err_t, err_p, err_toe, err_cam = [], [], [], []
    sig_t = []
    covered = 0

    for _ in range(n_trials):
        truth = TyreState.random(rng)
        z, T_meas = measure(truth, rng)
        x, P = estimate(z, T_meas)
        s = np.sqrt(np.diag(P))
        for i, c in enumerate(CORNERS):
            e = x[i] - truth.tread[c]
            err_t.append(e)
            sig_t.append(s[i])
            if abs(e) <= 2.0 * s[i]:
                covered += 1
            err_p.append(x[4 + i] - truth.pressure[c])
        err_toe.append(np.sqrt(max(0.0, x[IDX_TOESQ])) - abs(truth.toe))
        err_cam.append(x[IDX_CAMBER] - truth.camber)

    err_t = np.array(err_t)
    err_p = np.array(err_p)
    err_toe = np.array(err_toe)
    err_cam = np.array(err_cam)

    print(f"\n  {'quantity':<22}{'MAE':>12}{'RMSE':>12}{'p95 |err|':>14}")
    print("  " + "-" * 68)
    print(f"  {'tread depth  [mm]':<22}{np.mean(np.abs(err_t)):>12.3f}"
          f"{np.sqrt(np.mean(err_t ** 2)):>12.3f}"
          f"{np.percentile(np.abs(err_t), 95):>14.3f}")
    print(f"  {'pressure     [kPa]':<22}{np.mean(np.abs(err_p)):>12.2f}"
          f"{np.sqrt(np.mean(err_p ** 2)):>12.2f}"
          f"{np.percentile(np.abs(err_p), 95):>14.2f}")
    print(f"  {'|toe| magnitude [deg]':<22}{np.mean(np.abs(err_toe)):>12.3f}"
          f"{np.sqrt(np.mean(err_toe ** 2)):>12.3f}"
          f"{np.percentile(np.abs(err_toe), 95):>14.3f}")
    print(f"  {'camber       [deg]':<22}{np.mean(np.abs(err_cam)):>12.3f}"
          f"{np.sqrt(np.mean(err_cam ** 2)):>12.3f}"
          f"{np.percentile(np.abs(err_cam), 95):>14.3f}")
    print("  " + "-" * 68)
    print(f"  mean reported sigma on tread      {np.mean(sig_t):.3f} mm")
    print(f"  errors inside the 2-sigma band    {100 * covered / len(err_t):.1f} %"
          f"   (should be near 95)")
    print("\n  The camber row is the control. Its error is just the spread of")
    print("  the truth itself, because the estimate never moves off the prior.")
    print("  If a channel is ever added that DOES see camber, this row moving")
    print("  is how you would know it worked.")

    return {
        "tread_mae": float(np.mean(np.abs(err_t))),
        "tread_rmse": float(np.sqrt(np.mean(err_t ** 2))),
        "tread_p95": float(np.percentile(np.abs(err_t), 95)),
        "press_mae": float(np.mean(np.abs(err_p))),
        "toe_mae": float(np.mean(np.abs(err_toe))),
        "coverage": float(100 * covered / len(err_t)),
        "sigma_mean": float(np.mean(sig_t)),
    }

def ablation_study(rng, n_trials=150):
\
\
\
\
\
\

    print("\n" + "=" * 72)
    print(f"  DEMO 3  -  ABLATION: WHICH CHANNELS ACTUALLY EARN THEIR PLACE")
    print("=" * 72)

    configs = {
        "all three channels":           [True,  True,  True],
        "no rolling resistance":        [True,  True,  False],
        "no wheel-speed ratios":        [True,  False, True],
        "no resonance (TPMS + ratios)": [False, True,  True],
    }

    print(f"\n  {'configuration':<32}{'tread MAE':>14}{'tread p95':>14}")
    print("  " + "-" * 68)

    for label, (use_freq, use_ratio, use_rr) in configs.items():
        local = np.random.default_rng(20240828)
        errs = []
        for _ in range(n_trials):
            truth = TyreState.random(local)
            z, T_meas = measure(truth, local)
            R = measurement_covariance(z_ref=z).copy()

            if not use_freq:
                for i in range(4, 8):
                    R[i, i] = 1e12
            if not use_ratio:
                for i in range(8, 10):
                    R[i, i] = 1e12
            if not use_rr:
                R[10, 10] = 1e12

            x0, P0 = prior()
            R_inv = np.linalg.inv(R)
            P0_inv = np.linalg.inv(P0)
            x = x0.copy()
            for _ in range(6):
                H = jacobian(x, T_meas)
                dx = np.linalg.solve(
                    H.T @ R_inv @ H + P0_inv,
                    H.T @ R_inv @ (z - predict(x, T_meas)) + P0_inv @ (x0 - x),
                )
                x = x + dx
                x[IDX_TREAD] = np.clip(x[IDX_TREAD], 0.5, 9.0)
                x[IDX_PRESS] = np.clip(x[IDX_PRESS], 100.0, 350.0)
                x[IDX_TOESQ] = max(0.0, x[IDX_TOESQ])
            for i, c in enumerate(CORNERS):
                errs.append(x[i] - truth.tread[c])

        errs = np.array(errs)
        print(f"  {label:<32}{np.mean(np.abs(errs)):>11.3f} mm"
              f"{np.percentile(np.abs(errs), 95):>11.3f} mm")

    print("  " + "-" * 68)
    print("  Resonance is doing the heavy lifting. Wheel-speed ratios sharpen")
    print("  the within-axle comparison. Rolling resistance is nearly free -")
    print("  it needs no new signal - but it contributes little to TREAD; its")
    print("  real job is the energy and alignment story, not the wear number.")

def fault_energy_table():
\
\
\
\
\

    print("\n" + "=" * 72)
    print("  DEMO 4  -  WHAT EACH FAULT COSTS, FROM THE PHYSICS")
    print("=" * 72)

    v = 22.0
    healthy = TyreState()
    F0 = road_load(healthy, v)

    def variant(label, **kw):
        s = TyreState(**{k: v_ for k, v_ in kw.items()})
        F = road_load(s, v)
        return label, 100 * (F - F0) / F0

    p_placard = Vehicle.p_placard
    rows = [
        variant("under-inflated 40 kPa, all four",
                pressure={c: p_placard - 40 for c in CORNERS}),
        variant("under-inflated 70 kPa, all four",
                pressure={c: p_placard - 70 for c in CORNERS}),
        variant("front toe 0.5 deg", toe=0.5),
        variant("front toe 1.0 deg", toe=1.0),
        variant("tread worn to 4 mm, all four",
                tread={c: 4.0 for c in CORNERS}),
        variant("tread worn to 2 mm, all four",
                tread={c: 2.0 for c in CORNERS}),
    ]

    print(f"\n  baseline: 6.0 mm tread, {p_placard:.0f} kPa, zero toe, {v:.0f} m/s")
    print(f"\n  {'change from baseline':<38}{'road load':>14}")
    print("  " + "-" * 68)
    for label, pct in rows:
        print(f"  {label:<38}{pct:>+12.1f} %")
    print("  " + "-" * 68)
    print("""
  LOOK AT THE SIGN ON THE LAST TWO ROWS.

  Worn tread REDUCES road load. There is less rubber to deform on every
  revolution, so a bald tyre rolls more freely than a new one. The published
  industry figure is roughly a 20 percent fall in rolling resistance from new
  to worn, which is why fitting new tyres COSTS an EV up to 10 percent of its
  range rather than restoring it.

  So the pitch "your worn tyres are wasting your battery" is backwards, and
  any deck built on it is built on a physics error.

  The true energy story is the middle four rows: pressure and alignment. Both
  are recoverable in an afternoon. "Three percent of your energy bill is
  avoidable, here is which vehicle and which corner" is a far better line
  than "your tyres are old" - and it has the advantage of being true.

  Tread still matters enormously. It just matters for WET GRIP, which is a
  safety argument, not an energy one. Keep the two arguments separate and
  both of them stay honest.
""")

def main():
    ap = argparse.ArgumentParser(description="EV tyre state estimation demo")
    ap.add_argument("--trials", type=int, default=300, help="validation trials")
    ap.add_argument("--seed", type=int, default=20260828, help="random seed")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 72)
    print("  EV TYRE STATE ESTIMATION FROM EXISTING VEHICLE SENSORS")
    print("  physics -> sensor model -> estimator -> decision")
    print("=" * 72)
    print("\n  Inputs, all of them already on a production EV's CAN bus:")
    print("    wheel speed x4     ABS / ESC encoders")
    print("    TPMS pressure x4   slow, coarse, temperature-contaminated")
    print("    motor torque       traction inverter")
    print("    motor speed        traction inverter")
    print("    longitudinal accel ESC inertial unit")
    print("    temperature        BMS and TPMS sensor")
    print("\n  Outputs: tread depth and pressure per corner, front toe,")
    print("           a wet-weather torque ceiling, and a recoverable-energy")
    print("           figure - with an honest uncertainty on each.")

    sensitivity_table()
    fault_energy_table()

    rng = np.random.default_rng(args.seed)
    single_vehicle_demo(rng)
    stats = validation_sweep(rng, args.trials)
    if not args.skip_ablation:
        ablation_study(rng)

    print("\n" + "=" * 72)
    print("  READ THIS BEFORE QUOTING ANY NUMBER ABOVE")
    print("=" * 72)
    print("""
  These results are TRUE OF THE MODEL, not yet of a tyre.

  The estimator is being tested against the same equations that generated
  the data. That is a real and necessary test - it proves the inverse
  problem is well posed and that the fusion arithmetic is correct - but it
  cannot tell you whether Equation 3 describes an actual tyre.

  Everything rests on one unverified claim: that the first structural mode
  moves measurably and repeatably with tread depth, with the pressure
  confound removable. Until that is measured on a bench, every figure above
  is a statement about algebra.

  That bench test is the next thing to build, and it is cheap. Sensor noise
  IS the experiment, so the accelerometer choice matters more than anything
  else in the bill of materials.
""")
    print("=" * 72 + "\n")
    return stats

if __name__ == "__main__":
    main()
