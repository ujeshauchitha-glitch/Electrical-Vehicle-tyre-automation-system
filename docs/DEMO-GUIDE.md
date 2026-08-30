# 3D Digital Twin — what it is, how to run it, how to explain it

A plain-English guide to `frontend/tyre3d.html`.

---

## 1. What we actually built, in one paragraph

A car drives down a road. We are **not** allowed to look at its tyres. All we get
are the signals the car already sends to its own computer — how fast each wheel
is spinning, what the tyre-pressure sensors say, how hot the tyres are. From
those signals alone, the software tries to work out the condition of each tyre:
how much tread is left, what the pressure is, and whether the wheels are
pointing and leaning the way they should.

The demo shows two things side by side: the **truth** (the 3D car, because we
are simulating it, so we know the real answer) and the **estimate** (every card,
table and chart, which only ever sees the telemetry). The interesting part is
the gap between them — and, more importantly, *which parts of that gap can be
closed and which cannot*.

---

## 2. The single most important idea: observability

Some things simply cannot be worked out from the signals available, no matter
how good the maths is. The demo calls this **observability**, and it labels
every quantity with one of three verdicts:

| Verdict | Meaning |
|---|---|
| `OBSERVED` | The measurements genuinely pin this down. |
| `WEAK` | There is some information, but not enough to trust the number. |
| `UNOBSERVABLE` | Nothing on the car responds to this at all. The number you see is just our starting guess handed back. |

**Analogy.** Imagine a ruler that can only tell you the *difference in height*
between two people, never anyone's actual height. Stand two people on it and it
will tell you "the left one is 4 cm taller" perfectly. Ask it "how tall is the
left one?" and it has nothing to offer. That is exactly the situation with tyre
tread: the wheel-speed sensors compare the left and right wheel, so they see the
**difference** in tread across an axle beautifully and the **absolute depth**
not at all.

This is why the demo will confidently tell you *"your rear-right is 1 mm more
worn than your rear-left"* but will not tell you *"you have 4 mm of tread left."*
That is a limitation of the sensors, not of the algorithm — and being upfront
about it is the entire point of the project.

---

## 3. What it can and cannot measure

Numbers are computed live by sweeping all ten scenarios; open the **Accuracy**
tab and they are on screen.

| Quantity | Typical error | Needs | Verdict |
|---|---|---|---|
| Tyre pressure, per corner | ~0.31 kPa | base sensors | **measured** |
| Tread *difference* within an axle | ~0.03 mm | base sensors | **measured** |
| Camber angle, including which way it leans | ~0.03° | ESC sensors | **measured** |
| Toe angle, magnitude only | ~0.14° | + motor torque | **measured** |
| Absolute tread depth | ~0.5 mm typical, 3 mm worst | — | **not recoverable** |

The last row does not improve with a better prior, a different sensor package,
or more solver iterations. We tested all of them.

---

## 4. Why camber and toe matter — the question worth asking

> *"Why would we need camber and toe signals? How does that help with finding
> tyre wear?"*

Because **misalignment is a cause of wear, not a symptom of it.**

- **Toe** is how much the wheels point inward or outward relative to straight
  ahead. If a wheel is pointing even slightly sideways while the car drives
  forward, it is scrubbing across the road surface every metre you travel.
  Toe error is the single biggest cause of rapid, uneven tyre wear — a
  meaningful toe error can cut tyre life dramatically.
- **Camber** is how much the wheel leans in or out when viewed from the front.
  A leaning wheel puts most of the load on one shoulder of the tyre, so that
  shoulder wears out while the rest of the tread is still fine.

So the logic runs:

> Measuring tread tells you a tyre is *already* ruined.
> Measuring alignment tells you *why*, and *while you can still fix it*.

A system that says "your rear tyre is at 2 mm" is delivering bad news too late.
A system that says "your toe is out by 1°, which is roughly doubling your wear
rate — get an alignment" is delivering something the driver can act on, before
the tyre is scrap. That is what makes camber and toe worth chasing even though
neither is a wear measurement itself.

There is a second, structural reason. Tread is the quantity we *cannot* observe
well. Alignment is a quantity we *can* — to a few hundredths of a degree. Given
that misalignment drives wear, estimating alignment accurately is a far better
route to predicting tyre life than trying to squeeze an absolute tread depth out
of sensors that cannot provide one.

---

## 5. The three sensor packages

The demo has one control that changes everything: the **BASE / ESC / FULL**
buttons in the top bar (or press `P`).

- **BASE** — wheel-speed sensors, tyre-pressure sensors, temperature.
  Pressure is measured well. Tread difference is measured well. Camber and toe
  are completely invisible.

- **ESC** — adds the lateral accelerometer and the steering-angle sensor.
  **These are not hypothetical extras.** Electronic Stability Control has been
  legally required on new passenger cars since 2012 in the US (FMVSS 126) and
  2014 in the EU, and ESC cannot work without exactly these sensors. So almost
  every car on the road already has them.
  Result: **camber becomes measurable, including which way it leans**, with no
  new hardware at all. The physics is the one every alignment shop uses — a
  car with a camber fault *pulls* to one side, and the driver holds a small
  steering offset to keep it straight. That offset is the giveaway.

- **FULL** — adds an estimate of road-load force from motor torque, and assumes
  the wheel-speed ratio is averaged over a window rather than taken sample by
  sample. Toe becomes measurable (magnitude only — drag is the same whether the
  wheel points left or right, so the direction is genuinely unrecoverable).

---

## 6. How to run it

**Simplest — just open the file.** Double-click `frontend/tyre3d.html`. It needs
an internet connection (it pulls the 3D library from a CDN).

**For a live presentation, use the offline copy.** Open
`frontend/tyre3d.offline.html` instead. Everything is baked in — it makes zero
network requests, so bad conference wifi cannot break it. Copy it onto a USB
stick and it will work on any machine with Chrome.

**To serve it locally:**

```bash
python -m http.server 8765
```

then open `http://localhost:8765/frontend/tyre3d.html`.

**To rebuild after changing anything:**

```bash
python scripts/build_offline.py
```

The offline copy is generated from the online one, so always rebuild after an
edit or the two will drift apart. The script fails loudly if any external
reference survives.

**To regenerate the accuracy figures from the Python test suite:**

```bash
python scripts/export_accuracy.py
```

---

## 7. How to explain it — press `T`

There is a **guided tour** built in. Click `TOUR` in the top bar or press `T`.
It walks through the argument in nine steps, setting the scenario, sensor
package and tab for you, so you can talk while it drives itself. Arrow keys
move between steps, `Esc` exits.

The tour follows this arc:

1. **The setup** — 3D car is truth, panels are estimates.
2. **Pressure works** — recovered to ~0.3 kPa. This is what "solved" looks like.
3. **Tread does not** — the rear tyres are really at 5.0 and 4.0 mm; watch the
   Error column.
4. **Here is why** — the wheel-speed ratio only sees differences.
5. **Alignment is invisible too** — a real −1.5° camber fault reads as exactly
   0.000, because that is our starting guess, unchanged.
6. **The payoff** — switch to ESC and camber lands within ~0.05° of the true −1.50°. Hardware the car
   already has.
7. **Toe** — needs one more channel, and its direction is never recoverable.
8. **Why this matters** — misalignment causes the wear.
9. **The honest summary** — what is measured, what is not.

### If you only have two minutes

Press `T`, then jump to step 5, then step 6. The camber fault going from
"invisible" to "measured to 0.03°" with a single click, using sensors already
mandated by law, is the strongest thirty seconds in the demo.

---

## 8. Controls

| Key | Action |
|---|---|
| `T` | Guided tour on/off |
| `→` `←` | Next / previous tour step |
| `Esc` | Exit tour |
| `Space` | Play / pause |
| `R` | Reset the current scenario |
| `C` | Toggle orbit / fixed front camera |
| `P` | Cycle sensor package (BASE → ESC → FULL) |
| `1`–`9`, `0` | Jump to scenario 1–10 |

Also on screen: a scenario dropdown, speed and ambient-temperature sliders, a
distance slider that scrubs to any point in the first 1000 km, sliders for the
*true* toe and camber so you can dial in a fault and watch the estimator react,
and a steering control with an automatic S-curve.

Drag in the 3D view to orbit; scroll to zoom.

---

## 9. Honesty notes — say these out loud

Worth stating plainly if anyone asks, because they are the difference between a
demo and a claim:

- **Everything here is simulated.** No physical tyre has been measured. The
  comparison is model-versus-model: our estimator against the simulation that
  generated the data. It is not real-world validation.
- **Every physical constant is an unvalidated guess** carried over from the
  earlier prototype — stiffness, cornering stiffness, rolling-resistance
  coefficients, all of it. None has been checked against bench data. The
  Physics tab labels them as such.
- **Two on-screen numbers are deliberately accelerated for the demo**: wear rate
  (×10) and the odometer (×500), so a 1000 km run fits in a demo slot. Both are
  named and displayed in the Physics tab so nothing is silently sped up.
- **The Python suite's "Tread MAE 0.20 mm" is not an accuracy figure.** The
  Accuracy tab shows why: across every scenario the spread of the tread error is
  under 0.007 mm and the offset fraction is 100%, meaning the estimate never
  moves. That column is measuring the distance from our starting guess to a
  ground truth that happened to be set near it. The suite's own assertion agrees
  — it allows up to 2.5 mm and comments "WEAK observability, starts from prior".
- **Cornering was tested and did not help.** Turning the car swings the corner
  weights hard (from 4263 N to 5350 N at 4 m/s²), and in principle that is an
  independent probe of absolute tread. Measured contribution: common-mode
  variance reduction 0.0489 while cornering versus 0.0492 in a straight line.
  Real effect, too small to matter. Reported because a negative result is still
  a result.
