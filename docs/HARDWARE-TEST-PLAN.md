# Hardware test plan

Everything in this repository is currently validated against simulation only.
This plan lists what must happen on real hardware before any claim about the
real world is made. Acceptance criteria below are proposals; fix them before
looking at data.

## Simulation baseline to beat or match (seeded, 300 states, `scripts/ablation.py`)

| Variant | Tread RMSE (mm) | Pressure RMSE (kPa) | NEES/dof |
|---|---|---|---|
| Full | 0.125 | 2.46 | 0.90 |
| No resonance | 1.704 | 3.01 | 0.95 |
| No TPMS | 0.573 | 16.48 | 0.79 |
| Radius + TPMS only | 1.869 | 3.02 | 0.92 |

Reading: in simulation, tread depends almost entirely on the resonance
channel (about 14x worse without it, against a 1.6 mm legal limit). Motor
torque adds nothing once road load is present (they are redundant in the
model), and camber is not observable from these channels (RMSE stays at the
prior). Simulation uses the same equations as the estimator, so these numbers
are optimistic by construction.

## Phase 0: before touching a vehicle

- [ ] Fix acceptance criteria (suggested: tread MAE <= 0.5 mm, pressure MAE
      <= 10 kPa vs reference gauge, no false "OBSERVED" label on a channel
      shown unobservable).
- [ ] Decide the logging hardware and sample rate (see Phase 1).
- [ ] Check legal/insurance position for instrumenting the vehicle and for
      the test location. Read-only CAN tap; the logger must not transmit.

## Phase 1: signal path (settles G1)

Goal: find which torsional mode appears in wheel-speed data, and the rate and
filter needed.

1. Log raw ABS wheel-speed pulses (not the decoded CAN speed, which is
   usually 10-100 Hz) at >= 1 kHz per wheel, plus decoded CAN, TPMS, IMU.
   Record anti-alias characteristics of the capture chain.
2. Constant-speed runs, 3 speeds, 3 pressures. Look for a pressure-dependent
   spectral peak: first mode ~30-80 Hz or second ~100-300 Hz (model assumes
   first). Deliverable: spectra plots and a go/no-go on the resonance channel.
3. Fit `K_carcass`, `K_per_kPa`, `J_carcass`, `J_per_mm` from the peak vs
   pressure (and later vs tread). These are currently assumed values.
4. Estimate encoder tooth-ring error and drivetrain torsional lines that
   contaminate the band.

Stop condition: no pressure-dependent peak => resonance is unusable and the
tread estimate falls back to the "no resonance" row above.

## Phase 2: controlled matrix

Vary one factor at a time and log reference truth:

- Pressure: placard, -20, -40, +20 kPa (calibrated gauge).
- Tread: new, mid, near-limit tyres (depth gauge at several points).
- Load: empty, loaded. Surface: smooth asphalt (and one rough). Temperature:
  cold start vs warm.
- Alignment: nominal and a known toe offset on a rack (to test toe^2).

Each run: >= 60 s at steady speed, plus accel/decel segments for the motor
torque and slip behaviour.

## Phase 3: model fitting and validation

- Fit on one tyre set, validate on another set and another day. Never tune on
  the validation runs.
- Report MAE, bias, NEES, and per-channel ablation from real data next to the
  simulation table above.
- Check slip, regen, grade, wet surface: conditions the model assumes away.

## Phase 4: degradation

Wear needs mileage. Log the same tyres over weeks to months with periodic
depth measurements; test whether the temporal tracker's degradation rate
matches measured mm/km. Start early; it is the longest task.

## Software work this plan needs

- [x] `WindowedSample` and spectral extractor (synthetic tests only).
- [x] CSV log replay (`src/evtyre/logio/csv_log.py`) for decoded logs.
- [ ] Raw CAN decoding from a DBC (needs the vehicle's DBC or reverse
      engineering; vehicle-specific).
- [ ] Raw wheel-pulse timestamps -> uniformly sampled wheel speed.
- [ ] Wire resonance features into the estimator measurement vector.
- [ ] Timestamp alignment between window and frame.
- [ ] Per-vehicle calibration storage.

## Risks

- No pressure-dependent peak in wheel-speed data (kills the tread channel).
- Estimator equations were written alongside the simulator; real tyres
  may differ in ways that break observability.
- Tread information may be weaker than the model assumes (2 % radius change).
- Vehicle CAN access and DBC availability.
