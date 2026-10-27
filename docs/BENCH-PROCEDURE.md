# Bench procedure: Phase 1 signal-path experiment

Goal: decide whether a tread- and pressure-dependent resonance exists in
wheel-speed data, and fit the assumed constants. Everything here is a plan;
nothing has been run. Pass/fail criteria are fixed before looking at data.

## Equipment
- Vehicle (or a single-wheel rig) with readable ABS wheel-speed pulses.
- Logger capturing each wheel's raw pulse timestamps (or an ADC on the sensor
  line) at an effective sample rate >= 1 kHz, with the anti-alias chain known.
- Decoded CAN: wheel speeds, TPMS pressure and temperature, motor torque and
  speed, steering angle, lateral and longitudinal acceleration.
- RTK or dual-frequency GNSS speed, an IMU, calibrated tyre-pressure gauge,
  tread-depth gauge, infrared thermometer, level surface for static zero-offset.
- Read-only CAN tap. The logger must not transmit.

## Runs
Run A, signal path (decides G1)
1. Steady 40, 60, 80 km/h on straight smooth road, 60 s each, at placard
   pressure. Repeat at placard -40 kPa. Log raw wheel speed.
2. Compute spectra per wheel. Pass: a peak whose frequency changes with
   pressure in the predicted direction (up with pressure) and that is not a
   driveline order (does not scale with speed). Fail: no such peak => the
   resonance channel is unusable; report the no-resonance performance.
3. Record the band (first mode ~30-80 Hz or second ~100-300 Hz) and the
   sample rate needed.

Run B, install baseline
4. With new or known-tread tyres: record the install-state truth (tread gauge
   at 3 points per tyre, pressures, alignment-rack toe and camber). Log 20+
   steady snapshots at known conditions. These feed `calibrate()`.

Run C, tread and pressure matrix
5. Tyres at new, mid-wear, near-limit (three sets or shaved tyres). Pressures
   placard, -20, -40. Temperatures cold start and warmed up (log tyre
   temperature). For each: 60 s at 60 km/h.
6. Fit `K_carcass`, `K_per_kPa`, `J_carcass`, `J_per_mm` and the resonance
   temperature coefficient. Hold out one tyre set for validation.

Run D, camber and alignment
7. On a rack set camber to 0 and +/-1 deg (same direction both sides, then
   mirrored), toe nominal. Straight-line runs in both directions. Record
   steering offset and lateral acceleration. Fit `C_gamma / C_alpha`.

Run E, GNSS radius
8. Coasting segments at several speeds; measure rolling circumference
   independently; compare with GNSS-speed / wheel-speed radius; record slip
   bias on driven vs undriven axles.

## Acceptance criteria (proposed)
- Tread MAE <= 0.5 mm on held-out tyres, relative to the install baseline.
- Pressure MAE <= 10 kPa versus the reference gauge.
- Claimed 1-sigma contains the error about 68 % of the time (NEES near 1).
- No state labelled observable that the Jacobian shows is not.

## Log format
See `data/log_template.csv` (decoded channels, native units). Raw pulse
captures go in separate files with a sidecar noting sample rate, filter, and
sensor type. Replay with `evtyre.logio.csv_log.CsvLogSource`.

## Safety and permissions
Closed test track or controlled road, insurance cleared, passive logging only,
no modification of vehicle control signals.
