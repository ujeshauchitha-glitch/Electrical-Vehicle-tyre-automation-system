# EV Tyre Intelligence System — Architecture

Status: **architecture only, nothing implemented yet.** The only working code in this
repository is the reference/demo simulation at [`legacy/ev_tyre_fusion.py`](legacy/ev_tyre_fusion.py).
Everything below describes the target modular system and how the legacy code maps into it —
it is a plan, not a description of code that currently exists under `src/`.

---

## 1. Project objective

Build a modular system that estimates **per-corner EV tyre condition** — tread depth,
inflation pressure, front toe angle (camber acknowledged as currently unobservable from
onboard signals) — by fusing signals an EV already has, or can derive, from its own
telemetry/CAN bus, using an explicit physical model plus a Bayesian estimator, with a
calibrated uncertainty on every output. Beyond a single snapshot, the system should
eventually track that state **over time and distance** to estimate degradation **rate**
(wear rate, pressure-drift rate), not just current condition, so it can support forward-
looking maintenance and safety decisions (e.g. distance-to-legal-tread-limit, pre-emptive
torque limiting, leak detection).

The system must be honest about what is currently **proven** (the estimator's math is
well-posed and internally consistent) versus what is currently only **assumed** (that the
underlying physical models — most critically, the resonance-vs-tread-and-pressure
relationship — hold on a real tyre). Nothing here should be presented as validated until
it has been checked against real bench or vehicle data.

---

## 2. The five-phase architecture

| Phase | Name | One-line purpose |
|---|---|---|
| 1 | EV telemetry/input layer | Ingest and normalize real vehicle sensor/CAN data — no physics applied |
| 2 | Observable/feature extraction | Turn normalized telemetry into the derived observables the estimator consumes (pressure, resonance frequency, wheel-speed ratio, road-load coefficient) |
| 3 | Tyre state estimation | Fuse one snapshot of observables + physics model + prior into a per-corner state estimate with uncertainty |
| 4 | Multi-variable fusion | Turn the raw state estimate into decision-ready output (maintenance view, torque ceiling, energy-waste figure), reconciling multiple recent snapshots and auxiliary context |
| 5 | Temporal degradation-rate estimation | Track state over time/distance to estimate wear and pressure-drift **rate**, not just current level |

Phases 1–2 do not exist in any form today — the legacy script only ever fabricates their
output. Phase 3 exists as a working reference implementation in `legacy/`, using assumed
(not bench-validated) physics. Phase 4 exists partially as demo/reporting functions in
`legacy/`. Phase 5 does not exist in any form — the legacy estimator is stateless and
single-shot; see [§7](#7-important-assumptions-and-known-limitations).

---

## 3. Responsibilities of each phase

### Phase 1 — EV telemetry/input layer
- Read raw signals from a real or logged CAN/telemetry source: wheel speed ×4, TPMS
  pressure ×4, motor torque, motor speed, longitudinal acceleration, temperature (BMS/TPMS),
  vehicle speed, odometer, timestamp.
- Normalize units, timestamp everything, and flag missing/out-of-range/dropped-frame data.
- **Must not** apply any tyre-physics interpretation, and **must not** silently substitute
  simulated values for missing real data (see [§8](#8-rules-for-keeping-synthetic-simulation-separate-from-real-telemetry)).

### Phase 2 — Observable/feature extraction
- Compute the observation vector the estimator needs from a window of Phase 1 data:
  denoised per-corner pressure, per-corner first-mode resonance frequency (requires new
  signal-processing work — e.g. spectral/order-tracking analysis of wheel-speed-encoder
  ripple, which does not exist in the legacy code), axle-partner wheel-speed/rolling-radius
  ratio, and a road-load/rolling-resistance-equivalent coefficient (requires a new
  energy-balance derivation from torque/speed/acceleration; the legacy code only fabricates
  this from a known simulated state, it never derives it).
- Attach an empirically grounded noise/uncertainty estimate to each channel — not an
  assumed constant, once real data is available.
- **Must not** invent or assume observable values when a channel cannot currently be
  derived from available signals; it should be reported as unavailable, not backfilled.

### Phase 3 — Tyre state estimation
- Take one Phase 2 observation snapshot (plus a prior) and produce a per-corner state
  estimate (tread, pressure, toe magnitude, camber) with covariance, using the physical
  model and the Gauss-Newton/MAP estimator pattern already proven out in
  `legacy/ev_tyre_fusion.py`.
- Must accept live inputs (measured temperature, vehicle speed) as required arguments, not
  silently-defaulted constants, and must source physical constants from a per-vehicle
  config rather than hardcoded values.
- Must preserve and surface the known unobservability of camber and the sign of toe rather
  than hiding it.

### Phase 4 — Multi-variable fusion
- Combine Phase 3 output with additional context: multiple recent snapshots (robust
  combination to reduce single-sample noise), and — where available — auxiliary
  non-telemetry context (service records, last known tyre-change date).
- Produce the decision-ready outputs already sketched in `legacy/`: cold-compensated
  ("maintenance view") pressure, a wet/dry torque ceiling using a conservative lower-
  confidence tread bound, and a recoverable-energy figure.
- Must generalize legacy's hardcoded assumptions (e.g. rear-drive-only torque ceiling) into
  configuration rather than baked-in constants.
- Still describes the **current** condition only — no time trend.

### Phase 5 — Temporal degradation-rate estimation
- Persist Phase 3/4 output over time (state, covariance, timestamp, odometer, and ideally
  driving-context covariates) per vehicle, per corner.
- Fit or filter a wear/drift-rate model from that history — this is new capability the
  legacy code has no equivalent of at all, since it is stateless and single-shot.
- Produce degradation rate (e.g. mm/1000 km, kPa/week), projected time/distance to the
  legal tread limit with uncertainty, and anomaly flags (e.g. a sudden rate change
  suggesting damage rather than normal wear, or a slow leak).
- Must not conflate measurement noise in a single snapshot with a genuine trend; any rate
  claim needs enough history and an explicit uncertainty band.

---

## 4. Repository structure

```
legacy/
  ev_tyre_fusion.py        # frozen reference/demo — do not modify; see §6

src/
  evtyre/
    config/                # vehicle & tyre configuration schema + loader (used by Phase 3/4)
    schema/                # shared data contracts between phases (see §5)
    ingest/                # Phase 1 — telemetry adapters, validation, normalization
    features/              # Phase 2 — resonance/ratio/road-load extraction
    estimation/            # Phase 3 — physics model, predict/jacobian/estimate
    fusion/                # Phase 4 — multi-snapshot fusion, decision-layer consumers
    degradation/            # Phase 5 — history persistence interfaces, wear-rate models
  simulation/                # synthetic ground-truth + fake-sensor generators, for tests only
                              #  — never imported by ingest/, features/, estimation/ in production paths

tests/
  unit/                     # per-module tests
  fixtures/                 # synthetic data fixtures (built on src/simulation, or on legacy's
                              #  TyreState.random()/measure() until those are ported)

docs/
  # equation references, validation status per assumption, bench-test writeups

data/
  # real captured telemetry / bench-validation data, once it exists — never synthetic data

experiments/
  # bench-validation experiments (e.g. the resonance-vs-tread rig test called for in §7)

demo/
  # CLI/demo entrypoints — eventual replacement for legacy's main(), operating on real
  #  or replayed-log data rather than fabricated data

CLAUDE.md                   # this file
```

Empty directories (`data/`, `demo/`, `docs/`, `experiments/`, `tests/`, `src/evtyre/`)
already exist as scaffolding; none are populated yet.

---

## 5. Interfaces between phases

Each phase boundary is a typed data contract, not a shared mutable object. Every contract
should carry a timestamp and a model/config version tag, so a stored or logged value can
always be traced back to the physics/config version that produced it (the legacy code has
no such versioning, which is a known gap — see [§7](#7-important-assumptions-and-known-limitations)).

| Boundary | Contract (conceptual shape) | Carries |
|---|---|---|
| Phase 1 → 2 | `TelemetryFrame` / stream | Normalized raw signals, per-signal validity flags, timestamp |
| Phase 2 → 3 | `Observation` | The observation vector `z` (pressure ×4, resonance ×4, ratio ×2, road-load ×1 — legacy's 11-element layout, or its evolution), per-channel noise/covariance, measured temperature, vehicle speed, timestamp |
| Phase 3 → 4 | `TyreStateEstimate` | Per-corner tread/pressure/toe/camber point estimate + covariance, timestamp, odometer, model version |
| Phase 4 → 5 | `FusedTyreReport` | Decision-ready fields (cold pressure, torque ceiling, energy-waste %) plus the underlying `TyreStateEstimate`, timestamp, odometer |
| Phase 5 → consumers | `DegradationRateEstimate` | Rate per corner, projected time/distance to limit with uncertainty, anomaly flags |

Contracts should be defined once under `src/evtyre/schema/` and imported by every phase
that touches them, rather than each phase inventing its own ad hoc shape — this replaces
legacy's approach of magic-number index slices (`IDX_TREAD`, `M_PRESS`, etc.) into the
state/measurement vectors.

---

## 6. What belongs in `legacy/` vs `src/`

### Stays in `legacy/` (reference/demo only, not production code)
- `ev_tyre_fusion.py` in its entirety, **unmodified**.
- Its synthetic ground-truth generator (`TyreState.random`) and its measurement fabricator
  (`measure`) — useful as a reference for the observation contract's shape and as a
  regression fixture, but not a stand-in for real ingestion.
- Its demo/reporting functions (`sensitivity_table`, `fault_energy_table`,
  `single_vehicle_demo`, `validation_sweep`, `ablation_study`, `main`/CLI) — useful
  documentation of the model's behavior and a self-consistency check, not library code.
- Its specific numeric constants on `Vehicle` — they describe one fictional example
  vehicle and must not be treated as validated defaults for a real vehicle/tyre.

### Moves into `src/evtyre/` (once decoupled, configurable, and documented)
- The physical equations (pressure compensation, effective rolling radius, resonance
  frequency, rolling resistance, toe drag, wet friction) — each ported equation should
  carry an explicit "validated: yes/no" note pointing at supporting evidence (or its
  absence).
- `slip_stiffness` specifically needs a decision before it moves anywhere: it is currently
  dead code in `legacy/` (fully implemented, never called by the pipeline) — either
  integrate it into a real channel or drop it; do not carry dead code into `src/`.
- Vehicle/tyre configuration, replacing the hardcoded `Vehicle` class with a loadable,
  per-vehicle-model schema under `src/evtyre/config/`.
- The estimator core (`predict`, `jacobian`, `measurement_covariance`, `prior`,
  `estimate`), reworked to take live inputs as required arguments.
- The state/measurement schema, formalized under `src/evtyre/schema/`.
- The decision-layer consumers (`torque_limit`, `recoverable_energy`), generalized off
  legacy's hardcoded assumptions (e.g. the rear-drive-only torque ceiling).
- The sensor-noise model, documented, and eventually replaced by empirically measured
  noise once real sensors are involved.

---

## 7. Important assumptions and known limitations

These are carried over unchanged from the legacy implementation and **remain unvalidated**
until checked against real bench or vehicle data. None of the results the legacy demo
prints (low error, good 2σ coverage, etc.) should be read as evidence these hold on a real
tyre — the demo validates its own internal math against its own generative model, which is
a self-consistency check, not external validation.

- **Core unvalidated claim**: that a tyre's first structural resonance mode moves
  measurably and repeatably with tread depth, in a way separable from its dependence on
  pressure. Everything the tread estimate leans on assumes this.
- The effective-rolling-radius model's 1/3-deflection factor and vertical-stiffness
  constant are generic tyre-mechanics approximations, not fitted to any specific
  vehicle/tyre.
- The rolling-resistance model's functional form and constants (including the sign of its
  temperature term) are assumed, not fitted from dynamometer or coast-down data.
- The toe-drag model's cornering-stiffness constant is a single assumed value applied
  uniformly regardless of load or tyre condition.
- The wet-friction model (feeding a safety-relevant torque ceiling) is an assumed curve
  shape with no wet-braking/cornering test data behind it.
- Extracting a per-corner resonance frequency from a real ABS/ESC wheel-speed signal, and
  a clean road-load coefficient from motor torque/speed/acceleration, are both assumed
  feasible at specific noise levels — neither has been implemented or demonstrated.
- The system assumes static per-corner load (no dynamic load transfer), a single
  vehicle-wide temperature standing in for four independent corner temperatures, and (in
  the legacy code) a hardcoded constant vehicle speed rather than a live value.
- Camber is not observable from any currently modeled signal — this is a structural
  limitation of the current channel set, not a tuning gap, and would require a new sensor
  (e.g. a depot alignment scanner) to resolve.
- Toe is only ever recoverable as an unsigned magnitude from a single lumped scalar — not
  per-side, per-axle, or signed.
- No degradation-**rate** capability exists yet at any layer; Phase 5 is entirely new work.

---

## 8. Rules for keeping synthetic simulation separate from real telemetry

Because the only working code today is a simulator, and because conflating simulated and
real data is the single easiest way to accidentally claim something is validated when it
isn't, the following rules apply to all future code under `src/`:

1. **Simulation code lives only under `src/simulation/`** (or stays in `legacy/`) and is
   never imported by `src/evtyre/ingest/`, `features/`, `estimation/`, or `fusion/` in a
   production code path. It may be imported by `tests/`.
2. **Phase 1 (ingestion) never fabricates data.** If a real signal is missing or invalid,
   it must be reported as missing/invalid, not filled in with a simulated or assumed
   value, except in an explicitly named demo/test mode that cannot be reached from a
   production entrypoint.
3. **Any function that generates fake ground truth or fake measurements must be named and
   located so this is obvious** (e.g. under `simulation/`, or prefixed `synthetic_`/`fake_`)
   — never given a name that could be mistaken for a real ingestion or estimation function.
4. **Every output that was derived even partially from synthetic data must carry an
   explicit marker** (e.g. a `source: "simulated"` field on the report) so a downstream
   consumer, dashboard, or log can never present it as a real-vehicle result.
5. **Validation language matters.** Tests and docs that use synthetic data should describe
   what they actually show — e.g. "estimator math is internally consistent" — and must not
   describe results as "validated" or "proven" for a real tyre. That claim requires bench
   or real-vehicle data specifically, per [§7](#7-important-assumptions-and-known-limitations).
6. **`legacy/ev_tyre_fusion.py` itself is exempt from these rules as a whole** — it is a
   self-contained demo that already documents its own simulated nature — but nothing new
   written under `src/` may quietly reuse its fabrication logic as if it were real
   ingestion.

---

## 9. Development order

Recommended build order, front-loading the riskiest and most foundational pieces:

1. **Shared schema + config layer** (`src/evtyre/schema/`, `src/evtyre/config/`) — define
   the data contracts from [§5](#5-interfaces-between-phases) and a real vehicle/tyre
   configuration format first, since every later phase depends on them and legacy's magic
   index slices are exactly what this replaces.
2. **Port Phase 3's estimator core from `legacy/`** (physics equations, `predict`,
   `jacobian`, `estimate`, `prior`) into `src/evtyre/estimation/`, generalized to the new
   schema/config, with `legacy/ev_tyre_fusion.py`'s own demo runs kept as regression tests
   confirming the ported version reproduces the same behavior on the same synthetic inputs.
   This is largely a refactor of already-working logic, so it's low-risk and unblocks
   everything downstream from a testable estimator early.
3. **Bench-validate the core physical assumption in parallel** (§7's "core unvalidated
   claim" — resonance frequency vs. tread depth and pressure) via a small rig experiment
   under `experiments/`. This is the single highest-risk assumption in the whole system;
   it should be checked before investing heavily in Phase 2's resonance-extraction
   pipeline, not after.
4. **Phase 1 — telemetry ingestion**, initially against logged/replayed real or bench data
   rather than a live vehicle, to get a real `TelemetryFrame` stream flowing and validated.
5. **Phase 2 — feature extraction**, built against Phase 1's real data: start with the
   parts that are close to direct signals (pressure, wheel-speed ratio), then build the
   genuinely new resonance-frequency and road-load-derivation pipelines, informed by the
   Step 3 bench results.
6. **Wire Phase 3 to real Phase 2 output** and validate the estimator's output against
   independent ground truth (bench measurements or manual tread/pressure checks) — not
   just self-consistency — before trusting any number it produces on a real vehicle.
7. **Phase 4 — fusion/decision layer**, built on top of a validated Phase 3, porting and
   generalizing legacy's `torque_limit`/`recoverable_energy`-style consumers.
8. **Phase 5 — degradation-rate estimation**, once enough real historical Phase 3/4 output
   exists across time/mileage to make trend-fitting meaningful; this is the last phase
   specifically because it depends on accumulated real data that the earlier phases must
   first produce reliably.

---

*No files other than this one were changed to produce this document.
`legacy/ev_tyre_fusion.py` was not modified.*
