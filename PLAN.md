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

---

# Follow-up: ELI5 explanations, Simple/Detailed switch (+ URL option), Reset settings buttons, Slovak translation

## Context
The Reservoir Computing Playground (PR #1) has been merged into `main`. The user wants:
1. A plain-language **ELI5 ("Simple")** version of every description, next to the current
   technical text ("Detailed").
2. A **Simple / Detailed switch** in the descriptions to flip between them. **Simple is the
   default** for first-time visitors (user's choice), and the last choice is remembered.
3. A **URL option** for the same choice, so a shared link can open in Simple or Detailed:
   `…/?explain=simple#chaos` or `…/?explain=detailed`.
4. A **"Reset settings" button for each simulation**.
5. A **natural-sounding Slovak translation** of the whole app. It opens automatically when the
   browser's preferred language is Slovak.
6. The simulations themselves stay exactly as they are: no changes to the models, defaults,
   tuning or visuals.

## Branch
PR #1 is merged, so restart the designated branch from the latest `main` and open a **new** PR:
`git fetch origin main && git checkout -B ccr-9e9978e2-j5ywp6 origin/main`. The local branch
only holds already-merged history, so a `--force-with-lease` push is fine.

## Design

### Reading level (new `js/ui/explain.js`)
- The state lives on `<html data-explain="simple|detailed">`. Text exists in both versions,
  wrapped in `data-level="simple"` / `data-level="detailed"`, and CSS hides the inactive one:
  `[data-explain='simple'] [data-level='detailed'], [data-explain='detailed'] [data-level='simple'] { display: none }`.
- Which level applies: URL `?explain=` first, then the saved choice (`localStorage`
  `rc-explain`, inside try/catch), then the default `simple`.
- An inline script in `index.html` `<head>` (next to the existing theme script) applies the
  level before first paint, so there's no flash of the wrong text.
- `levelSwitch()`: a small segmented control `Simple | Detailed` (two buttons with
  `aria-pressed`, keyboard-focusable). Every copy of it stays in sync. Clicking one sets
  `data-explain`, saves the choice, and updates the address bar's `?explain=` with
  `history.replaceState`, keeping the `#demo` hash. Copying the address bar therefore shares
  the current mode.
- Helper `levels(simpleChildren, detailedChildren)` returns the two wrapped variants, for use in
  the demo modules.
- Where the switch appears: the intro hero, and each demo's `view-head`, right under the title
  and above the description. Each demo's "What am I looking at?" `<details>` follows the same
  global setting.

### What gets a Simple version
- **Intro** (`index.html`):
  - the hero paragraphs;
  - each demo card's description, plus the "Usually / Here" lines in plain words;
  - "Why does a random network work?": a crowd/pond analogy, with the four settings explained in
    everyday terms;
  - "Where it's weaker".
  The diagram, headings and Further reading are shared.
- **Each demo** (`js/demos/{chaos,gestures,speech,water}.js`):
  - the `view-head` paragraph, plus a short "Try it: …" line in Simple mode;
  - the explainer's paragraphs.
  The current text becomes the Detailed variant, unchanged. Speech's privacy paragraph
  appears in both. Control labels, stats and charts are unchanged.
- Simple-mode wording: short sentences, no jargon, everyday analogies (pebbles and ripples,
  weather forecasts, a crowd with memories, fingerprints). It must stay factually correct and
  never invent numbers; any figures must match the Detailed text.

### Reset settings (one button per demo: `↺ Reset settings`, small, at the end of the settings card)
- Pull each demo's initial control values into one `DEFAULTS` object. The slider/toggle
  constructors and the reset both use it, so the two can't drift apart.
- Reset sets every control through the existing `slider.value` / `toggle.checked` setters,
  then runs the demo's existing retrain path **once**:
  - **Chaos**: `applyPreset()` for the currently selected system, squared features on, linear
    comparison on, dream speed 120 (`S.dreamRate`), reservoir seed back to 1, then `train()`.
    The chosen system stays selected.
  - **Gestures**: neurons 200, spectral radius 0.9, leak 0.3, then `rebuild()`. Taught gestures
    are kept; the existing "Reset to built-in gestures" button handles those.
  - **Speech**: sensitivity 12 dB (also applied to a live VAD), neurons 300, leak 0.25, then
    `scheduleTrain(0, true)`. Recordings are kept; "Forget all recordings" handles those.
  - **Water**: nonlinear camera on, delay 8, damping 0.06, probes 48, speed 30 (also
    `loop.setRate`), then `train()`. The selected task stays.
- After a reset, a status line confirms what happened.

### Slovak translation (new `js/i18n.js` + `js/content/en.js`, `js/content/sk.js`)
- **Which language opens:**
  1. URL `?lang=sk|en`, the same kind of shareable option as `?explain=`; both combine with
     `#demo`;
  2. otherwise the saved choice (`rc-lang`);
  3. otherwise **Slovak if the browser's first preferred language starting with `sk`
     appears in `navigator.languages`**;
  4. otherwise English.

  The inline head script sets `<html lang>` before the app loads.
- **Switch:** a compact `EN | SK` control in the header next to the theme button. Switching
  saves the choice, updates `?lang=` and reloads the page. That's the simplest reliable way
  to rebuild every demo's text. Taught gestures and saved recordings survive in storage.
- **`t(key, params)`** looks strings up in the active content table, falls back to English for
  any missing key, and interpolates `{name}` placeholders.
- **Number format:** a `fmt(n, digits)` helper based on `Intl.NumberFormat`. Displayed numbers
  in Slovak use a decimal comma and a space as the thousands separator (8,2 · 3 899).
  Computations are untouched.
- **What gets translated:** all user-visible text.
  - The intro page, in both Simple and Detailed versions. The intro body moves from static
    HTML into a small renderer (`js/intro.js`) fed by the content tables, so it isn't
    duplicated four times in HTML.
  - Header, footer, page title and meta description.
  - Every demo's title, descriptions, explainers, control labels, hints, buttons, stats,
    legends, status and warning messages, aria-labels, tooltip text, and text drawn on
    canvases (axis and unit labels such as "training ends" / "koniec trénovania", and
    LT → LČ for Ljapunovove časy).
  - Gesture names (kruh, trojuholník, štvorec, fajka ✓, cikcak, špirála, hviezda, vlnovka,
    krížik ✕).
- **Speech in Slovak:** the default words become **hore / dole / vľavo / vpravo**. The voice
  grid accepts both these and the English words, so saved English words still steer it.
- **Style:**
  - Natural, conversational Slovak with correct diacritics, informal "ty" in instructions,
    and standard technical terms: hrebeňová regresia, spektrálny polomer, atraktor,
    Ljapunovov čas, spektrogram, neurón, škálovanie vstupu, miera úniku (leak).
  - "Reservoir computing" becomes "rezervoárové počítanie (reservoir computing)" on first
    mention. There's no established Slovak term.
  - The Simple Slovak versions use the same everyday analogies as English, written natively
    rather than translated word for word.
- `README.md` gets a short note on the language and explanation options (the README itself
  stays in English).

### Files
- New: `js/ui/explain.js`, `js/i18n.js`, `js/content/en.js`, `js/content/sk.js`, `js/intro.js`.
- Edit:
  - `index.html`: the inline head script handles theme, explain level and language, and the
    intro body becomes a mount point;
  - `js/main.js`: render the intro, the header language switch and translated tab names; keep
    `?explain`/`?lang` intact on hash navigation;
  - `css/style.css`: visibility rules, `.level-switch` and `.try` styles;
  - the four `js/demos/*.js`: description variants, switch, DEFAULTS, reset button;
  - `README.md`: mention the switch, the URL option and reset;
  - `tests/e2e/smoke.mjs`: new checks;
  - `PLAN.md`: append this follow-up plan as a new section.
- Reuse the existing `h`, `button`, `storage` helpers in `js/ui/dom.js` and each demo's
  current retrain functions. No simulation code (`js/core`, `js/systems`, `js/audio`)
  changes.

## Verification
- `npm test`: all 29 existing tests still pass, which shows the simulations are untouched.
- Extend `tests/e2e/smoke.mjs`, running in Chromium in all three sessions:
  - a fresh visit shows the Simple text (Detailed hidden). Clicking "Detailed" swaps the
    text, every switch on the page follows, and the URL gains `?explain=detailed`.
  - `?explain=detailed#water` opens the water demo in Detailed mode.
  - the choice survives a reload.
  - a context with `locale: 'sk-SK'` opens in Slovak: `<html lang="sk">`, Slovak title and tab
    names, Slovak gesture name for a drawn circle ("kruh"), decimal commas in stats.
  - `?lang=en` overrides the Slovak locale, and the EN/SK switch flips the language.
  - no English leftovers in Slovak mode: scan the page text for a list of common English UI
    words.
  - every demo runs without console errors in Slovak.
  - per demo: change a control (set a slider through the page), click Reset settings, and
    check the read-outs return to the defaults with no console errors.
- Review screenshots of the intro and one demo in Simple and Detailed, desktop and 390 px
  phone, light and dark. Check the switch fits on phone width and nothing overlaps. Also
  review Slovak screenshots: longer Slovak words must not overflow buttons, tabs or stat
  tiles, and the header must still fit at 390 px with the EN/SK switch.
- Commit, push to `ccr-9e9978e2-j5ywp6`, open a new PR to `main`, and tell the user the
  Pages site updates once it's merged.
