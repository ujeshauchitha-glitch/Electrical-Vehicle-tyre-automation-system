# MASTER PROMPT: Build the EV Tyre Intelligence 3D Demo

## What we built

An EV tyre intelligence system that estimates hidden tyre condition from
signals an EV already has on its CAN bus -- NO new hardware. Gauss-Newton
MAP estimator with an observability contract that mechanically determines
what it can and cannot see.

GitHub: https://github.com/ujeshauchitha-glitch/Electrical-Vehicle-tyre-automation-system
Branch: main | Tests: 441 passing

## YOUR MISSION

Build a SINGLE self-contained HTML file that is visually STUNNING.
Not a dashboard. A cinematic, interactive 3D experience.

Requirements:
1. 3D car model with visible tyre state per corner
2. Estimator running in real-time in JavaScript
3. Observability contract: tread=WEAK, pressure=OBSERVED, toe/camber=UNOBSERVABLE
4. Degradation animation over distance
5. Scenario switching (Normal, Low Pressure, Asymmetric Wear, etc.)
6. Validated accuracy metrics displayed
7. Looks like a product demo, not homework

## TECH STACK (single HTML file, no build step)
- Three.js (CDN) for 3D rendering
- Tailwind CSS (CDN) for layout
- Google Fonts: Space Grotesk + JetBrains Mono
- Just open the HTML file -- no server needed

## THE 3D SCENE

Build a stylized low-poly car from Three.js primitives (boxes, cylinders, spheres).

1. Dark reflective floor (showroom style)
2. 4 tyres that change color by tread health: Green(>5mm) Yellow(2-5mm) Red(<2mm)
3. Floating holographic labels per tyre: tread X.XX mm +/- Y.YY, pressure XXX kPa, observability badge
4. Turntable rotation + mouse drag orbit
5. Subtle particle system, grid floor pulses
6. Glassmorphism UI panels with backdrop-blur

## THE ESTIMATOR (port Python to JS)

10 states: [tread_FL/FR/RL/RR, press_FL/FR/RL/RR, toe_sq, camber]
11 measurements: [press x4, freq x4 (MISSING), ratio_front, ratio_rear, roadload (disabled)]

Physics functions to port:
- effectiveRadius(tread_mm, p_kpa, fz_n): r_free = 0.333 + tread/1000, k_z = 210000*(p/240), delta = fz/k_z, r_eff = r_free - delta/3
- cornerWeight(corner): mass*9.80665*frontFrac/2 or (1-frontFrac)/2. mass=1850, frontFrac=0.47
- rollingResistance(tread, p, temp): 0.009 * (240/p)^0.45 * (0.8+0.2*tread/7.5) * (1+0.0015*(temp-25))
- toeDrag(toe_sq): 2*55000*(PI/180)^2 * toe_sq
- predictRatio(tl, pl, tr, pr): effectiveRadius(tr,pr,...) / effectiveRadius(tl,pl,...)

Gauss-Newton MAP:
- Prior: x0 = [4.55,4.55,4.55,4.55, 240,240,240,240, 0.1, 0]
- P0 = diag([6.25x4, 1600x4, 0.16, 1.0])
- R = [25,25,25,25, 1e12x4, 0.000025, 0.000025, 1e12]
- Max 6 iterations, clip tread[0.5,9], press[50,500], toe_sq>=0
- Numerical Jacobian eps=1e-5

Observability: colNorm > 1e-10 && vr >= 0.01 -> check common-mode for tread pairs -> WEAK or OBSERVED

## VALIDATED ACCURACY (show in demo)

| Scenario           | Tread MAE | Press MAE | Coverage |
|--------------------|-----------|-----------|----------|
| Normal             | 0.20 mm   | 0.64 kPa  | 100%     |
| Uniform Wear       | 0.30 mm   | 0.64 kPa  | 100%     |
| Asymmetric Wear    | 0.35 mm   | 0.64 kPa  | 100%     |
| Low Pressure       | 0.20 mm   | 0.67 kPa  | 100%     |
| Pressure Drift     | 0.20 mm   | 0.61 kPa  | 100%     |
| Toe Misalignment   | 0.20 mm   | 0.64 kPa  | 100%     |
| Sensor Missing     | 0.20 mm   | 0.64 kPa  | 100%     |
| Accel. Degradation | 0.80 mm   | 0.64 kPa  | 100%     |

DISCLAIMER: All values UNVALIDATED. Pressure=OBSERVED, Tread=WEAK, model-vs-model only.

## SCENARIOS

Normal: tread=[7.5x4], press=[240x4], wear=[0.03,0.03,0.04,0.05] mm/1000km
Asymmetric: tread=[7.5,7.5,5.0,4.0], wear=[0.02,0.02,0.06,0.08]
Low Pressure: tread=[7.5x4], press=[240,240,180,240], wear=[0.03,0.03,0.08,0.03]
Accelerated: tread=[6.0x4], wear=[0.15x4]
Pressure Drift: tread=[7.5x4], press_drift=[0,0,-5,0] kPa/1000km

## UI LAYOUT

Top bar: TYRE_INTEL_EV logo | Phase 6 badge | Scenario dropdown | Distance slider
Center: 3D car turntable + 4 floating tyre cards (tread bar, pressure, observability)
Bottom tabs: Estimator | Observability | Accuracy | Physics equations
Right sidebar: Sensor grid (8 sensors), States observed X/10, variance reduction

## VISUAL DESIGN

Dark #0a0a0f, panels #141420. Accents: cyan #00d4ff, magenta #ff00aa, green #00ff88
Space Grotesk headings, JetBrains Mono numbers. Glassmorphism, glow effects, particles.
Animated gradient borders. Smooth transitions 0.3s.

## CONSTRAINTS
1. NO fabricating data -- WEAK means WEAK
2. NO tread as OBSERVED -- it is WEAK
3. JS estimator must match Python numbers
4. Include UNVALIDATED disclaimer
5. Works by opening HTML file (no server)
6. Amazing on projector/screen share
7. Keyboard: Space=pause, 1-5=scenario, R=reset

## INSPIRATION
Tesla Service Mode meets Iron Man HUD meets F1 telemetry.
Reviewer leans forward and says wait show me that again.
