# Reservoir Computing Web Demo — Plan

## Context
The user wants a hands-on demo of reservoir computing (RC) showing it doing well at tasks
that are normally handled by other AI techniques (LSTMs, CNNs, backprop). The repo
`m1omg/ReservoirWebDemo` is empty apart from a GPL-3.0 `LICENSE`. The user wants:
- a **web app** (no install, runs anywhere, shareable), with **all four demos**: chaos
  forecasting, gesture recognition, spoken word recognition, and a "bucket of water"
  physical reservoir;
- a **GitHub Pages** deploy workflow, a push to `ccr-9e9978e2-j5ywp6`, and a **PR to main**;
- everything **independent of display refresh rate** (user preference).

Main point the app makes: only the linear readout is trained, so learning is one ridge
regression. That takes milliseconds in the browser, where the usual approaches need minutes
to hours of backprop.

## Tech decisions
- Vanilla JS **ES modules**, HTML, CSS and Canvas 2D. No build step, no runtime
  dependencies, no CDN, so it works offline.
- `package.json` (`"type": "module"`, no deps) only so that `node --test` can import the
  same modules the browser uses.
- Heavy math stays on the main thread, but long jobs (chaos training) run as **chunked
  async loops that yield on a time budget** (about 12 ms), with a progress bar.
- **Refresh-rate independence:** one `FixedStepLoop` drives every animation. It
  accumulates `performance.now()` deltas and runs a fixed number of simulation steps per
  *second* (`dt = 1/stepsPerSecond`). It renders once per rAF, caps catch-up (max 250 ms)
  and resets after the tab was hidden. Live audio is driven by audio callbacks, which are
  already real-time. All throttles (live gesture guessing, etc.) are time-based.
- Theming with CSS variables (light/dark via `prefers-color-scheme`). Canvases read their
  colors from CSS variables and are HiDPI-aware (`devicePixelRatio` + `ResizeObserver`).
  The layout is responsive and touch-friendly.
- Hash routing (`#chaos`, `#gestures`, `#speech`, `#water`) so a single demo can be linked.
  Each demo initializes the first time its tab opens and pauses its loop when hidden.

## File layout
```
index.html                 app shell, tabs, intro page (inline SVG diagram: input → fixed random reservoir → trained readout)
css/style.css
js/main.js                 tab routing, lazy demo init, pause/resume
js/core/rng.js             seeded PRNG (mulberry32) + gaussian
js/core/linalg.js          Cholesky solve, RidgeAccumulator (streams XᵀX/XᵀY upper-triangle), ridgeSolve(X,Y,λ) auto primal/dual
js/core/esn.js             EchoStateNetwork: sparse CSR W, Win, bias, leak, spectral-radius scaling (Gelfand log-growth estimate), step/run, feature map [1, x, x_even²]
js/core/sequence-classifier.js  shared by gestures+speech: ESN states at fixed fractions of the sequence + mean → dual ridge, softmax scores, augmentation hooks
js/core/loop.js            FixedStepLoop (refresh-rate independent)
js/ui/canvas.js            HiDPI canvas helper, line plots, heatmap (neurons×time), 3D orbit projection
js/ui/controls.js          slider/select/toggle builders (incl. log-scale slider)
js/systems/chaos.js        Lorenz, Rössler (RK4), Mackey–Glass (delay buffer, τ=17)
js/systems/gestures-synth.js  procedural gesture generator + augmentation
js/systems/wave.js         2-D damped wave-equation "pond"
js/audio/features.js       FFT, mel filterbank, log-mel frames, per-utterance mean norm
js/audio/capture-worklet.js  AudioWorkletProcessor posting sample blocks
js/audio/vad.js            energy VAD with adaptive noise floor, pre-roll, hang-over
js/demos/{chaos,gestures,speech,water}.js
tests/*.test.js            node --test unit/behaviour tests
tests/e2e/smoke.mjs        Playwright smoke test + screenshots (dev-only, uses preinstalled Chromium)
run.sh                     POSIX sh: python3 -m http.server on 127.0.0.1 + xdg-open (any distro)
README.md, .nojekyll
.github/workflows/ci.yml   node --test on push/PR
.github/workflows/pages.yml  test → assemble _site → upload-pages-artifact → deploy-pages (on push to main + manual)
```

## Demos

### 1. Chaos forecasting (flagship)
- Systems: Lorenz (default), Rössler, Mackey–Glass. Data z-scored. Teacher-forced run →
  `RidgeAccumulator` with target u(t+1). Then **closed loop**: the prediction is fed back
  as input.
- Default ESN of about 300–400 nodes, sparse (about 6 links/node), with squared-feature
  readout (Pathak/Lu-Hunt-Ott trick). Defaults get tuned by a Node test.
- Views:
  - true vs predicted time series, with markers for prediction start and **valid
    prediction time** (normalized error > 0.4), shown in **Lyapunov times** for
    Lorenz/Rössler;
  - rotating, draggable **3D attractor**: faint true attractor plus a bright ESN trail that
    keeps **"dreaming"** forever;
  - live **reservoir activity heatmap**.
- Controls: system, N, spectral radius, leak, input scaling, ridge λ (log), training
  length, seed / "new random reservoir", squared-features toggle, **"compare: no reservoir
  (linear autoregression)"** baseline, and a **Perturb** button that kicks the dream
  trajectory so you can watch it fall back onto the attractor.
- Stats: training time in ms, sample count, feature count, valid time.

### 2. Gesture recognition
- Pointer Events canvas (mouse/touch/pen), multi-stroke with a pen-up channel. A gesture
  ends 600 ms after the last stroke (time-based) or on button press.
- Preprocessing: arc-length resampling to about 40 points, centering and scale
  normalization. Inputs per step: [x, y, dx, dy, penUp].
- **Built-in classes generated procedurally:** circle (both directions), triangle, square,
  check, zigzag, spiral, star, wave. Augmentation covers rotation, anisotropic scale, shear,
  smooth noise and start offset. Held-out accuracy and training time are shown at startup.
- **Teach your own gesture:** name it, draw 3–5 examples. Each example is augmented ×10 and
  the model retrains instantly. Delete a class with a button. User gestures persist in
  `localStorage` (try/catch).
- **Guesses while you draw** (time-throttled re-run on the partial stroke; training
  includes prefixes ≥60%). Confidence bars, plus "not sure" below a threshold.
- Visual: reservoir "fingerprint" heatmap (neurons × time) for the last gesture.

### 3. Spoken word recognition
- `getUserMedia` → AudioWorklet capture → log-mel frames (25 ms window / 10 ms hop, about
  24 bands, CMN) → VAD segments utterances.
- Default words up / down / left / right (renameable, add/remove). The user records about 5
  examples per word, each augmented (time-stretch, noise). Classification uses
  `SequenceClassifier` (about 300 nodes).
- Listen mode: shows the recognized word and confidence bars. A small **voice-controlled
  grid token** moves on up/down/left/right, so it's a practical use. There's also a live
  scrolling mel spectrogram, a noise-floor meter and the reservoir heatmap of the last
  utterance.
- Privacy: all processing is local. Only feature vectors (never audio) are stored, and only
  if the user ticks "remember on this device". There's a "Forget" button.
- Shows clear messages when the mic is denied or the page isn't a secure context.

### 4. Bucket of water
- About 72×48 damped wave-equation pond, after Fernando & Sojakka 2003. Input "motors" poke
  the surface and about 48 probe points are read out. Rendering uses shaded ImageData with
  visible probes and motors.
- Tasks:
  - **sine vs square** classification (scrolling trace of input, true label and readout);
  - **memory/recall** (reconstruct the input from d steps ago).
- **"Nonlinear sensor (h²)"** toggle. The wave medium is linear, so sine/square fails
  without it. That point is explained in the UI.
- Training is chunked and fast; then live test at a fixed rate with a speed slider and a
  running accuracy figure.

### Intro page
- Plain-language RC explanation and a diagram.
- One card per demo: the task, what's usually used for it, and what's used here (random
  fixed network + one linear solve).
- Every literature claim is **verified via WebSearch** before writing (Jaeger & Haas 2004;
  Pathak et al. 2018; Vlachas et al. 2020; Fernando & Sojakka 2003; Verstraeten et al.
  2005; Gauthier et al. 2021 NG-RC; Lyapunov exponents; Speech Commands dataset size).
- Also check the current versions of the GitHub Pages actions via WebSearch.

## Implementation order (commit after each step)
0. Save this plan into the repo as `PLAN.md` (requested by the user).
1. Core: rng, linalg, esn, loop, plus tests.
2. Chaos systems and demo; tune defaults with a test.
3. Gestures: generator, classifier, demo.
4. Water: simulator, tasks, demo.
5. Speech: features, worklet, VAD, demo.
6. App shell, intro, styles, theming, responsiveness.
7. README (what RC is, how to run: Pages URL / `./run.sh` / any static server, browser
   support, privacy, references), `run.sh`, workflows, `.nojekyll`.
8. Verify (below), then commit and `git push -u origin ccr-9e9978e2-j5ywp6` (retry with
   backoff on network errors). Open a PR to `main` (there's no PR template), then offer to
   watch it.

## Verification
- `node --test tests/` must pass:
  - RNG determinism;
  - Cholesky and ridge recover a known linear map, primal ≈ dual;
  - measured spectral radius within 10% of the target;
  - echo-state property: two initial states converge when ρ < 1;
  - Lorenz valid prediction time ≥ about 3 Lyapunov times with the defaults, and the linear
    baseline is much worse;
  - Mackey–Glass closed-loop NRMSE is small;
  - synthetic gesture held-out accuracy ≥ 90%;
  - mel features: a sine peaks in the right band, and the synthetic chirp/tone "words"
    classify at ≥ 90%;
  - VAD finds bursts in a synthetic signal;
  - water: sine/square ≥ 85% with h², clearly worse without;
  - `FixedStepLoop` gives the same step count over 1 s at simulated 30/60/144/240 Hz frame
    rates.
- Playwright smoke test (`tests/e2e/smoke.mjs`):
  - serve with `python3 -m http.server`, then open every tab in Chromium
    (`/opt/pw-browsers`) with fake media flags;
  - assert there are no console errors;
  - click Train on chaos, draw a synthetic gesture via mouse events and check a prediction
    appears, run the water demo, and start the speech demo with the fake mic;
  - take desktop and phone-width (390 px) screenshots, light and dark, and review them
    visually;
  - also run the loop with a simulated frame rate and confirm simulation speed doesn't
    change.
- Run `run.sh` once to confirm it serves the app.
