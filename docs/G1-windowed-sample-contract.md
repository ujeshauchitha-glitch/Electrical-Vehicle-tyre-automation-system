# G1 — Windowed Sample Contract

**Status:** Design only. No implementation. No modification to Phase 1.

---

## Why this exists

`TelemetryFrame` is a single-instant snapshot with no sample rate and no
anti-alias contract.  Vibration and resonance only exist across a span of
time, so the frozen Phase 1 schema structurally cannot carry the resonance
channel.

This matters because resonance carries the whole tread estimate: ablation
shows **0.10 mm tread error with it** and **1.41 mm without**, against a
1.6 mm legal limit.  The gap between "resonance available" and "resonance
unavailable" is the gap between a useful system and a useless one.

---

## Proposed type: `WindowedSample`

A `WindowedSample` wraps a fixed-length buffer of raw ADC / encoder counts
for one or more channels, alongside metadata that makes the buffer
interpretable.  It sits *alongside* `TelemetryFrame`, not replacing it.
Feature extractors that need instantaneous values (pressure, wheel speed)
consume `TelemetryFrame`; extractors that need spectral content (resonance)
consume `WindowedSample`.

```python
@dataclass(frozen=True)
class WindowedSample:
    """A fixed-length window of time-series data for one or more channels.

    This is a DESIGN ONLY — do not implement until G1 is resolved.
    """

    # --- Buffer ---
    data: np.ndarray           # shape (n_samples, n_channels), float64
    channel_names: tuple[str, ...]  # column labels for data

    # --- Timing ---
    sample_rate_hz: float      # samples per second (PARAMETER, see below)
    start_time_s: float        # absolute timestamp of first sample
    end_time_s: float          # absolute timestamp of last sample

    # --- Provenance ---
    source: FrameSource        # "real", "replay", or "simulated"
    provenance: str            # source label

    # --- Missing-data semantics ---
    missing_mask: np.ndarray | None  # shape (n_samples, n_channels), bool
        # True = sample is missing / corrupt.  None means no missing data.
        # Feature extractors must handle mid-window gaps gracefully.
```

---

## Key design parameters (all unresolved)

### Sample rate — a parameter, not a chosen value

The sample rate is **the** critical unresolved parameter.  There is an
unresolved physics question that determines what rate is needed:

**Question:** Does the tread signal live in the **first** tyre torsional
mode or the **second**?

| Hypothesis | Approximate frequency range | Minimum Nyquist rate | Recommended sample rate |
|---|---|---|---|
| First torsional mode | 30–80 Hz | 160 Hz | 500 Hz |
| Second torsional mode | 100–300 Hz | 600 Hz | 1000+ Hz |

Picking a rate would silently answer this question.  Both hypotheses remain
open.  The `sample_rate_hz` field is therefore a **parameter that must be
set by the caller** — the `WindowedSample` constructor does not choose it.

Until the physics question is resolved (via bench experiment — see
`experiments/`), any implementation must support both rate ranges.  The
recommended approach is to design for the higher rate (1000 Hz) and
downsample if the first-mode hypothesis wins, rather than designing for the
lower rate and being unable to recover the second-mode signal.

### Window length

Window length determines frequency resolution: Δf = 1/T.

| Window length | Δf at 1000 Hz | Δf at 500 Hz | Use case |
|---|---|---|---|
| 0.5 s | 2 Hz | 2 Hz | Fast update, coarse resolution |
| 1.0 s | 1 Hz | 1 Hz | Balanced |
| 2.0 s | 0.5 Hz | 0.5 Hz | Fine resolution, slow update |

Recommendation: 1.0 s as default, with the option to override.  The
frequency resolution must be fine enough to separate the resonance peak
from nearby spectral features (e.g. driveline orders), which requires
Δf ≤ 2 Hz at minimum.

### Anti-alias bandwidth

The anti-alias filter cutoff must be below the Nyquist frequency of the
sampling rate.  For a 1000 Hz sample rate, the anti-alias cutoff should
be ≤ 500 Hz.  For a 500 Hz rate, ≤ 250 Hz.

If the second-mode hypothesis (100–300 Hz) is correct, the anti-alias
filter must not attenuate signals in that band.  This constrains the
filter design: cutoff must be above 300 Hz for the second-mode signal
to survive.

### Per-corner vs vehicle-level

The wheel-speed encoder signal is per-corner (four independent encoders).
A `WindowedSample` may carry:

- **Per-corner windows**: four independent buffers, one per wheel.
  This is the most informative but requires four synchronized ADC
  channels.
- **Vehicle-level window**: a single buffer (e.g. from a chassis-mounted
  accelerometer).  This loses per-corner discrimination.

Recommendation: per-corner as the primary design.  Vehicle-level as a
fallback when only one sensor is available.

### Missing-data semantics mid-window

Real telemetry drops samples.  A `WindowedSample` may have gaps mid-window.
The `missing_mask` field marks which samples are present.  Feature
extractors must:

1. Never interpolate across gaps without explicit opt-in.
2. Report UNAVAILABLE if the fraction of missing samples exceeds a
   threshold (e.g. > 20% of the window).
3. Handle gaps at the start or end of the window gracefully (e.g. by
   reducing the effective window length).

---

## Composition with TelemetryFrame

A `WindowedSample` does **not** replace a `TelemetryFrame`.  The two
coexist:

| Aspect | TelemetryFrame | WindowedSample |
|---|---|---|
| Content | Instantaneous sensor values | Time-series buffer |
| Timestamp | Single float | start_time_s + end_time_s |
| Resonance | Cannot carry it | Can carry it |
| Pressure / speed | Native | Not its purpose |
| Phase | Phase 1 output | Future (G1-blocked) |

Feature extractors consume one or both:

```python
def extract(
    frame: TelemetryFrame,
    sample: WindowedSample | None,  # None until G1 is implemented
    vehicle_config: VehicleConfig,
    tyre_config: TyreConfig,
) -> tuple[Feature, ...]:
```

Extractors that only need instantaneous data ignore `sample`.
Extractors that need spectral content (resonance) require `sample` and
emit UNAVAILABLE when it is `None`.

---

## What blocks implementation

1. **The physics question**: first or second torsional mode?  Requires a
   bench experiment (see `experiments/`).
2. **The anti-alias filter design**: depends on the answer to (1).
3. **ADC / encoder interface**: how to actually get the time-series data
   from the vehicle's CAN bus or a dedicated sensor.
4. **Synchronisation**: aligning the windowed sample with the
   instantaneous TelemetryFrame.

None of these are software problems — they are physics and hardware
problems.  The software contract is defined above so that when the
physics is resolved, the implementation path is clear.

---

## Resonance extractor (stub)

The resonance feature extractor is implemented as a stub that raises
`NotImplementedError` with "G1" in the message.  This is intentional:
a stub that returns a value invites someone to fill it in without
reopening G1.  See `src/evtyre/features/resonance.py`.
