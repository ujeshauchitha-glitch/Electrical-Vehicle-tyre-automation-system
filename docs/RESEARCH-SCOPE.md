# Field scoping: indirect tyre-state estimation from vehicle signals

Status: first pass, 27 Oct 2026. Built from web search snippets only; no paper
or patent below has been read in full. Patent text states what is claimed, not
what was validated. Every item marked **verify** needs a primary-source read
before it is cited in a report or deck.

## 1. Approach families found

| Family | Signal used | Targets | Evidence found |
|---|---|---|---|
| Rolling-radius / DLR | Wheel speed vs. GNSS or vehicle speed | Pressure loss, radius change (wear) | Academic: [Özkan, Lundquist, Gustafsson](https://people.isy.liu.se/rt/lundquist/Publications/OzkanLG2011.pdf) (marginalized particle filter, mm-level absolute radius, sub-mm relative radius on an axle). Patents: [US 11707952](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11707952), [US 11840242](https://patents.google.com/patent/US11840242B2/fr) |
| Resonance (RFM) | Spectrum of ABS wheel speed | Pressure (absolute, so detects 4-wheel deflation) | [SAE 2016-01-0459](https://saemobilus.sae.org/articles/indirect-tpms-algorithm-based-tire-resonance-frequency-estimated-ar-model-2016-01-0459) (AR/MESE after tooth-ring error removal); patents e.g. [US 9527352](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9527352) |
| Wheel-speed feature classification | Features of wheel-speed signal | Tread wear state | Goodyear patents [US 9821611](https://patents.justia.com/patent/9821611), [US 9610810](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9610810), [US 10603962](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10603962) |
| Wheel-speed FFT + regression | FFT residual vs. reference curve | Tread depth | [US 2024/0190179](https://patents.justia.com/patent/20240190179) |
| CAN-bus rolling radius + slip | Wheel/vehicle speed, accel, throttle | Wear via radius | Goodyear-linked application, via the [US 9719886](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9719886) family |
| In-tyre sensors | Inner-liner accelerometer, P/T/load | Load, friction, wear, mileage | Pirelli Cyber Tyre ([Yahoo report](https://sg.news.yahoo.com/pirelli-launching-smart-tires-monitor-213200500.html), [thesis](https://www.politesi.polimi.it/handle/10589/109723)); Goodyear SightLine ([tread-moulded sensors](https://autos.yahoo.com/tread-moulded-sensors-revolutionising-tyres-201129214.html)); [US 11650134](https://patents.google.com/patent/US11650134) |

## 2. What this means for our project

- Our estimator already combines two of these (rolling radius via wheel-speed
  ratios, resonance frequency) and adds an EV-specific motor-torque road-load
  channel. The combination, plus honest observability reporting, is the
  differentiator to check. **Verify** novelty against the patents above before
  claiming it.
- Mapping to the CLAUDE.md rule "do not claim a variable is observable if the
  measurements cannot identify it": the published rolling-radius work reports
  radius, not tread depth. Tread-to-radius is a ~6.4 mm change on ~330 mm
  (about 2 %) over the legal life in our model, so radius needs pressure,
  load and temperature separated out first. Our WEAK/OBSERVED labelling is the
  right framing; keep it.
- Pressure is the dominant confound for both radius and resonance. Direct TPMS
  gives us a pressure anchor, but the hybrid TPMS+resonance idea is itself
  patented ([US 9016116](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9016116)).
- Everything we have is simulation. No source found gives a head-to-head
  accuracy comparison across these methods, so there is no external benchmark
  number to quote against.

## 3. Motivation / market context

- Euro 7 (Reg. (EU) 2024/1257) is the first regulation to cap tyre abrasion
  particles; one source gives new-type dates of 1 Jul 2028 (cars), and limits of
  3 mg/km for pure EVs vs 7 mg/km otherwise. **Verify** figures against the
  regulation text. Sources: [KIT/TFEIP](https://tfeip-secretariat.org/_files/ugd/e5a9c7_657c779c4163481789a43c4075c8887f.pdf),
  [HERTH+BUSS](https://herthundbuss.com/en/industry-more/the-new-euro-7-emissions-standard/),
  [ETRMA position](https://tyrepress.com/2022/11/etrma-supports-holistic-approach-to-tyre-inclusive-euro-7-proposals).
- EVs are heavier and so abrade more ([Michelin](https://evmagazine.com/news/michelin-the-secret-microplastics-pollution-from-evs)).
  The torque contribution was not quantified in anything found. This is a gap
  our torque channel could speak to, but only with real data.

## 4. Gaps and next searches

1. Peer-reviewed surveys of indirect wear estimation (none found). Try Google
   Scholar / IEEE Xplore: "tyre tread depth estimation wheel speed",
   "tire wear virtual sensor", "vehicle-based tire wear estimation".
2. Tyre-mode resonance physics for wear: how does resonance shift with tread
   mass? Our model uses `J_per_mm = 0.0332 kg·m²/mm` (assumed, not fitted).
3. Quantitative accuracy of any indirect tread-depth method on real vehicles.
4. EV-specific wear: regen braking, instant torque, load. Peer-reviewed only.
5. Public datasets: any CAN + measured-tread dataset we could validate on.
   Nothing identified yet.
6. Freedom-to-operate: have someone read the claims of the Goodyear and
   resonance patent families. This is not legal advice.
7. Bridgestone CAIS, Continental, Bosch, Michelin offerings (only a passing
   mention of Bridgestone was found).

## 5. Suggested project actions

- Add a validation plan item: obtain or collect real wheel-speed + measured
  tread data; until then keep all results labelled simulated.
- Add a baseline estimator (rolling-radius-only with GNSS speed) for ablation,
  so we can show what resonance and torque add.
- Fit or justify the assumed constants (`J_per_mm`, `tread_rr_span`, `p_exponent`).
