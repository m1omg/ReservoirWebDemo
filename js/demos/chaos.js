// Demo 1 — forecasting chaos, then letting the reservoir "dream" the attractor.

import { SYSTEMS, zscore } from '../systems/chaos.js';
import { ReservoirForecaster, LinearForecaster, forecast } from '../core/forecaster.js';
import { FixedStepLoop } from '../core/loop.js';
import { Rng } from '../core/rng.js';
import { h, slider, select, toggle, button, stat, legend, progressBar } from '../ui/dom.js';
import { Canvas2D, colors, onThemeChange, yAxis, strokeSeries, drawHeatmap, makeCamera, project, orbitControls, withAlpha } from '../ui/canvas.js';

const TICKS = 240; // simulation ticks per second (independent of display rate)
const HORIZON = { lorenz: 1200, rossler: 1500, mackeyGlass: 1500 };
const CONTEXT = { lorenz: 220, rossler: 260, mackeyGlass: 260 };
const DATA_SEED = 2024;
const RASTER_COLS = 240;
const RASTER_ROWS = 48;
const TRAIL = 2400;

export function mount(root) {
  const S = {
    system: 'lorenz',
    seed: 1,
    model: null,
    linear: null,
    result: null,
    linResult: null,
    data: null,
    train: 0,
    horizon: 0,
    context: 0,
    revealed: 0,
    revealRate: 300,
    dreamRate: 120,
    dreamAcc: 0,
    revealAcc: 0,
    trail: new Float32Array(TRAIL * 3),
    trailLen: 0,
    trailHead: 0,
    raster: new Float32Array(RASTER_COLS * RASTER_ROWS),
    rasterHead: 0,
    rasterNeurons: [],
    rowMean: new Float64Array(RASTER_ROWS),
    rowScale: new Float64Array(RASTER_ROWS),
    truthCloud: null,
    hist: [],
    training: false,
    hover: null,
    rng: new Rng(99),
  };
  const cam = makeCamera(0.7, 0.3);

  // ---------- DOM ----------
  const forecastCanvas = h('canvas', { class: 'plot', 'aria-label': 'Forecast compared with the true trajectory' });
  const tooltip = h('div', { class: 'tooltip', hidden: true });
  const legendBox = h('div');
  const attractorCanvas = h('canvas', { class: 'plot', 'aria-label': 'Three-dimensional view of the attractor; drag to rotate' });
  const rasterCanvas = h('canvas', { class: 'plot', 'aria-label': 'Activity of reservoir neurons over time' });
  const progress = progressBar();
  const status = h('p', { class: 'hint', 'aria-live': 'polite' }, '');

  const stTrain = stat('Training time');
  const stValid = stat('Valid forecast');
  const stLinear = stat('Linear model, no reservoir');
  const stFeat = stat('Trained weights');

  const sysSelect = select({
    label: 'Chaotic system',
    value: S.system,
    options: Object.entries(SYSTEMS).map(([k, v]) => [k, v.name]),
    onChange: (v) => {
      S.system = v;
      applyPreset();
      describe();
      train();
    },
  });
  const sysDesc = h('p', { class: 'hint' });

  const fmt = (d) => (v) => Number(v).toFixed(d);
  const sz = slider({ label: 'Neurons', min: 50, max: 800, step: 50, value: 300, onChange: () => train() });
  const sr = slider({ label: 'Spectral radius (memory / echo)', min: 0.05, max: 1.6, step: 0.05, value: 0.3, format: fmt(2), onChange: () => train() });
  const lk = slider({ label: 'Leak rate', min: 0.05, max: 1, step: 0.05, value: 1, format: fmt(2), onChange: () => train() });
  const ins = slider({ label: 'Input scaling', min: 0.05, max: 2, step: 0.05, value: 0.3, format: fmt(2), onChange: () => train() });
  const lam = slider({ label: 'Ridge regularisation λ', min: 1e-14, max: 1e-2, log: true, value: 1e-11, format: (v) => v.toExponential(0), onChange: () => train() });
  const tl = slider({ label: 'Training steps', min: 500, max: 10000, step: 500, value: 4000, onChange: () => train() });
  const sq = toggle({ label: 'Squared readout features', hint: 'Lets the linear readout see x² for half the neurons', checked: true, onChange: () => train() });
  const cmp = toggle({ label: 'Compare with linear model', hint: 'Same training data and same ridge regression, but with no reservoir', checked: true, onChange: () => train() });
  const speed = slider({
    label: 'Dream speed',
    min: 10,
    max: 600,
    step: 10,
    value: S.dreamRate,
    format: (v) => `${v} steps/s`,
    onInput: (v) => (S.dreamRate = v),
  });

  const trainBtn = button('Train & forecast', () => train(), 'primary');
  const newBtn = button('New random reservoir', () => {
    S.seed++;
    train();
  });
  const perturbBtn = button('Perturb the dream', () => {
    if (!S.model) return;
    S.model.perturb(S.rng, 1.2);
  });
  const restartBtn = button('Restart dream', () => {
    if (!S.model) return;
    // Rewind to the end of training and replay the (deterministic) forecast.
    S.model.restore(S.snapshot);
    S.result = forecast(S.model, S.data, SYSTEMS[S.system].dim, 0, S.train, S.horizon);
    startDream();
  });

  const panel = h(
    'div',
    { class: 'card panel' },
    sysSelect.el,
    sysDesc,
    h('div', { class: 'btn-row' }, trainBtn, newBtn),
    progress.el,
    status,
    h('h2', {}, 'Reservoir'),
    sz.el,
    sr.el,
    lk.el,
    ins.el,
    h('h2', {}, 'Readout (the only trained part)'),
    lam.el,
    tl.el,
    sq.el,
    cmp.el,
    h('h2', {}, 'Dreaming'),
    speed.el,
    h('div', { class: 'btn-row' }, perturbBtn, restartBtn),
  );

  const explain = h(
    'details',
    { class: 'explain' },
    h('summary', {}, 'What am I looking at?'),
    h('p', {}, 'The reservoir is a network of a few hundred randomly connected neurons with fixed weights. During training it is fed the true trajectory of a chaotic system, and a ', h('b', {}, 'single ridge regression'), ' learns how to read the next point from the neurons’ states. That is the whole training process. There is no backpropagation and there are no epochs.'),
    h('p', {}, 'Then the input is disconnected and each prediction is fed back in as the next input. The network now runs on its own and keeps generating the future. Chaos makes small errors grow exponentially. A ', h('b', {}, 'Lyapunov time'), ' is how long an error takes to grow e-fold (≈2.7×). Weather forecasts lose accuracy after a few of them, so predicting about 8 Lyapunov times ahead from 4,000 samples is very good.'),
    h('p', {}, 'Once the forecast has drifted from the true path, the reservoir keeps “dreaming” a trajectory that stays on the right attractor. It learned the system’s ', h('i', {}, 'climate'), ' as well as its weather. Press ', h('b', {}, 'Perturb'), ' to knock it off the attractor and watch it return.'),
    h('p', {}, 'The linear model gets exactly the same data and training procedure. Its forecast falls apart almost immediately, which shows that the reservoir’s random nonlinear dynamics are what make the difference.'),
  );

  root.append(
    h(
      'div',
      { class: 'view-head' },
      h('h1', {}, 'Forecasting chaos'),
      h('p', {}, 'A random, untrained network learns to predict a chaotic system in about a third of a second, then keeps generating its attractor on its own. LSTMs and other backprop-trained networks are the usual tool for this task.'),
    ),
    h(
      'div',
      { class: 'demo' },
      h(
        'div',
        { class: 'stack' },
        h('div', { class: 'stats' }, stTrain.el, stValid.el, stLinear.el, stFeat.el),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'Forecast vs. reality'), h('span', { class: 'sub' }, 'left of the line: training data · right: the network predicting on its own')), legendBox, h('div', { class: 'rel' }, forecastCanvas, tooltip)),
        h(
          'div',
          { class: 'grid-2' },
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'The attractor, dreamed'), h('span', { class: 'sub' }, 'drag to rotate')), legend([['--axis', 'true attractor'], ['--s2', 'reservoir running freely']]), attractorCanvas),
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'Inside the reservoir'), h('span', { class: 'sub' }, `${RASTER_ROWS} of the neurons, last ${RASTER_COLS} steps`)), legend([['--div-neg', 'below'], ['--div-pos', 'above the neuron’s own average']]), rasterCanvas),
        ),
        explain,
      ),
      panel,
    ),
  );

  const fc = new Canvas2D(forecastCanvas, { aspect: 2.3, minHeight: 260, maxHeight: 420, onResize: () => drawForecast() });
  const ac = new Canvas2D(attractorCanvas, { aspect: 1.1, minHeight: 240, maxHeight: 420 });
  const rc = new Canvas2D(rasterCanvas, { aspect: 1.1, minHeight: 240, maxHeight: 420 });
  orbitControls(attractorCanvas, cam, () => drawAttractor());
  onThemeChange(() => {
    drawForecast();
    drawAttractor();
    drawRaster();
  });

  // ---------- Behaviour ----------

  function applyPreset() {
    const p = SYSTEMS[S.system].preset;
    sz.value = p.size;
    sr.value = p.spectralRadius;
    lk.value = p.leak;
    ins.value = p.inputScaling;
    lam.value = p.lambda;
    tl.value = p.train;
  }

  function describe() {
    const sys = SYSTEMS[S.system];
    sysDesc.textContent = sys.description;
    legendBox.replaceChildren(
      legend([['--s1', 'true trajectory'], ['--s2', 'reservoir forecast'], ...(cmp.checked ? [['--s3', 'linear model (no reservoir)', true]] : [])]),
    );
  }

  const unit = () => {
    const sys = SYSTEMS[S.system];
    return sys.lyapunov ? { f: (steps) => steps * sys.dt * sys.lyapunov, name: 'Lyapunov times', short: 'LT' } : { f: (steps) => steps * sys.dt, name: 'time units', short: 't' };
  };

  async function train() {
    if (S.training) {
      S.pending = true;
      return;
    }
    S.training = true;
    S.pending = false;
    trainBtn.disabled = true;
    loop.stop();
    const sys = SYSTEMS[S.system];
    const T = tl.value;
    const H = HORIZON[S.system];
    status.textContent = 'Generating data…';
    progress.set(0);
    await new Promise((r) => setTimeout(r, 0));
    const raw = sys.generate(T + H + 20, DATA_SEED);
    const { data } = zscore(raw, sys.dim, T);
    S.data = data;
    S.train = T;
    S.horizon = H;
    S.context = Math.min(CONTEXT[S.system], T - 1);
    buildCloud(sys, data, T);

    const model = new ReservoirForecaster({
      dim: sys.dim,
      esn: { size: sz.value, degree: sys.preset.degree, spectralRadius: sr.value, leak: lk.value, inputScaling: ins.value, biasScaling: sys.preset.biasScaling, seed: S.seed },
      squared: sq.checked,
      lambda: lam.value,
    });
    status.textContent = `Running ${T.toLocaleString()} steps through ${sz.value} neurons and solving for the readout…`;
    const t0 = performance.now();
    await model.train(data, 0, T, { onProgress: (p) => progress.set(p) });
    const ms = performance.now() - t0;
    S.snapshot = model.snapshot();
    S.result = forecast(model, data, sys.dim, 0, T, H);
    S.model = model;

    S.linResult = null;
    if (cmp.checked) {
      const lin = new LinearForecaster({ dim: sys.dim, delays: 8, lambda: lam.value });
      await lin.train(data, 0, T, {});
      S.linResult = forecast(lin, data, sys.dim, 0, T, H);
    }

    const u = unit();
    stTrain.set(ms < 1000 ? `${Math.round(ms)}` : (ms / 1000).toFixed(2), ms < 1000 ? 'ms' : 's');
    const valid = u.f(S.result.valid);
    stValid.set(S.result.valid >= H ? `> ${u.f(H).toFixed(u.short === 't' ? 0 : 1)}` : valid.toFixed(u.short === 't' ? 0 : 1), u.name);
    if (S.linResult) stLinear.set(u.f(S.linResult.valid).toFixed(u.short === 't' ? 0 : 2), u.name);
    else stLinear.set('off');
    stFeat.set((model.featureCount * sys.dim).toLocaleString(), `(${model.samples.toLocaleString()} samples)`);
    status.textContent = `Trained in ${Math.round(ms)} ms on ${model.samples.toLocaleString()} samples. The forecast is being revealed below.`;
    progress.set(1);
    describe();

    S.revealed = 0;
    S.revealAcc = 0;
    S.revealRate = Math.max(150, H / 4);
    startDream();
    S.training = false;
    trainBtn.disabled = false;
    if (S.pending) train();
  }

  function buildCloud(sys, data, T) {
    const pts = [];
    const lag = sys.embedLag || 0;
    for (let t = Math.max(2 * lag, 0); t < T; t += 2) pts.push(point(sys, (k) => data[k], t));
    S.truthCloud = pts;
  }

  // 3-D coordinates for a time index; 1-D series use a delay embedding.
  function point(sys, get, t) {
    if (sys.dim === 3) {
      const [a, b, c] = sys.view;
      return [get(t * 3 + a), get(t * 3 + b), get(t * 3 + c)];
    }
    const L = sys.embedLag;
    return [get(t), get(t - L), get(t - 2 * L)];
  }

  function startDream() {
    if (!S.model || !S.result) return;
    const sys = SYSTEMS[S.system];
    S.trailLen = 0;
    S.trailHead = 0;
    S.hist = [];
    // Seed the trail with the forecast itself so the dream continues it.
    const H = S.horizon;
    const pred = S.result.pred;
    for (let k = 0; k < H; k++) pushDream(sys, pred.subarray(k * sys.dim, (k + 1) * sys.dim));
    const N = S.model.esn.size;
    S.rasterNeurons = Array.from({ length: RASTER_ROWS }, (_, i) => Math.floor((i * N) / RASTER_ROWS));
    S.raster.fill(0);
    S.rasterHead = 0;
    loop.start();
  }

  function pushDream(sys, y) {
    if (sys.dim === 1) {
      S.hist.push(y[0]);
      if (S.hist.length > 64) S.hist.shift();
      const L = sys.embedLag;
      const n = S.hist.length;
      if (n <= 2 * L) return;
      addTrail(S.hist[n - 1], S.hist[n - 1 - L], S.hist[n - 1 - 2 * L]);
    } else {
      const [a, b, c] = sys.view;
      addTrail(y[a], y[b], y[c]);
    }
  }

  function addTrail(x, y, z) {
    const i = S.trailHead * 3;
    S.trail[i] = x;
    S.trail[i + 1] = y;
    S.trail[i + 2] = z;
    S.trailHead = (S.trailHead + 1) % TRAIL;
    S.trailLen = Math.min(TRAIL, S.trailLen + 1);
  }

  function dreamStep() {
    const sys = SYSTEMS[S.system];
    const y = S.model.dream();
    // Keep a runaway (rare, with extreme settings) from breaking the view.
    for (let j = 0; j < y.length; j++) if (!Number.isFinite(y[j]) || Math.abs(y[j]) > 50) y[j] = Math.sign(y[j] || 1) * 50;
    pushDream(sys, y);
    const x = S.model.state;
    const col = S.rasterHead;
    for (let r = 0; r < RASTER_ROWS; r++) S.raster[r * RASTER_COLS + col] = x[S.rasterNeurons[r]];
    S.rasterHead = (col + 1) % RASTER_COLS;
  }

  const loop = new FixedStepLoop({
    stepsPerSecond: TICKS,
    step: () => {
      if (!S.model) return;
      if (S.revealed < S.horizon) {
        S.revealAcc += S.revealRate / TICKS;
        const n = Math.floor(S.revealAcc);
        S.revealAcc -= n;
        S.revealed = Math.min(S.horizon, S.revealed + n);
      }
      S.dreamAcc += S.dreamRate / TICKS;
      while (S.dreamAcc >= 1) {
        dreamStep();
        S.dreamAcc -= 1;
      }
      if (!cam.dragging) cam.yaw += 0.12 / TICKS;
    },
    render: () => {
      drawForecast();
      drawAttractor();
      drawRaster();
    },
  });

  // ---------- Drawing ----------

  function drawForecast() {
    const c = colors();
    const { ctx, width: W, height: Hh } = fc;
    fc.clear();
    if (!S.result) return;
    const sys = SYSTEMS[S.system];
    const D = sys.dim;
    const ctxN = S.context;
    const total = ctxN + S.horizon;
    const left = 34;
    const right = 10;
    const top = 20;
    const bottom = 26;
    const gap = 10;
    const ph = (Hh - top - bottom - gap * (D - 1)) / D;
    const pw = W - left - right;
    const toX = (i) => left + (i / (total - 1)) * pw;
    const start = S.train - 1 - ctxN; // first plotted time index
    const nowX = toX(ctxN);
    const labels = sys.dim === 3 ? ['x', 'y', 'z'] : ['x'];
    const u = unit();

    for (let d = 0; d < D; d++) {
      const y0 = top + d * (ph + gap);
      const toY = yAxis(ctx, { x: left, y: y0, w: pw, h: ph, min: -3, max: 3, ticks: [-2, 0, 2], c });
      ctx.fillStyle = c.text2;
      ctx.font = '600 11px system-ui, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(labels[d], left + 4, y0 + 10);
      // Truth (training context + the future it should have predicted).
      strokeSeries(ctx, total, (i) => S.data[(start + i) * D + d], toX, toY, c.s1, 1.5);
      const rev = S.revealed;
      // Linear baseline.
      if (S.linResult) {
        const lp = S.linResult.pred;
        strokeSeries(ctx, rev, (k) => clampV(lp[k * D + d]), (k) => toX(ctxN + 1 + k), toY, c.s3, 1.5, [5, 4]);
      }
      // Reservoir forecast.
      const pred = S.result.pred;
      strokeSeries(ctx, rev, (k) => clampV(pred[k * D + d]), (k) => toX(ctxN + 1 + k), toY, c.s2, 2);
    }

    // "Now" line and valid-time marker.
    ctx.save();
    ctx.strokeStyle = c.text2;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(nowX) + 0.5, top);
    ctx.lineTo(Math.round(nowX) + 0.5, Hh - bottom);
    ctx.stroke();
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = c.text2;
    ctx.textAlign = 'right';
    ctx.fillText('training ends', nowX - 4, Hh - bottom + 14);
    if (S.result.valid < S.horizon && S.revealed > S.result.valid) {
      const vx = Math.round(toX(ctxN + 1 + S.result.valid)) + 0.5;
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = c.s2;
      ctx.beginPath();
      ctx.moveTo(vx, top);
      ctx.lineTo(vx, Hh - bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.textAlign = 'left';
      ctx.fillStyle = c.text;
      const v = u.f(S.result.valid).toFixed(u.short === 't' ? 0 : 1);
      let txt = `still accurate up to here: ${v} ${u.name}`;
      if (ctx.measureText(txt).width > W - left - right) txt = `accurate for ${v} ${u.short}`;
      const tw = ctx.measureText(txt).width;
      let tx = vx + 4 + tw > W - right ? vx - 4 - tw : vx + 4;
      tx = Math.max(left, Math.min(W - right - tw, tx));
      ctx.fillText(txt, tx, 12);
    }
    // Time ticks in Lyapunov times (or time units) after "now".
    ctx.fillStyle = c.muted;
    ctx.textAlign = 'center';
    const totalUnits = u.f(S.horizon);
    const stepU = niceStep(totalUnits / 6);
    for (let v = stepU; v < totalUnits; v += stepU) {
      const k = Math.round(v / u.f(1));
      const x = toX(ctxN + 1 + k);
      if (Math.abs(x - nowX) < 60) continue;
      ctx.fillText(`${+v.toFixed(2)} ${u.short}`, x, Hh - 6);
    }
    ctx.restore();

    // Hover crosshair.
    if (S.hover !== null) {
      const i = S.hover;
      const x = Math.round(toX(i)) + 0.5;
      ctx.save();
      ctx.strokeStyle = c.axis;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, Hh - bottom);
      ctx.stroke();
      ctx.restore();
    }
  }

  function clampV(v) {
    return Number.isFinite(v) ? Math.max(-3.2, Math.min(3.2, v)) : NaN;
  }

  function niceStep(raw) {
    const p = 10 ** Math.floor(Math.log10(raw));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) return m * p;
    return 10 * p;
  }

  forecastCanvas.addEventListener('pointermove', (e) => {
    if (!S.result) return;
    const rect = forecastCanvas.getBoundingClientRect();
    const sys = SYSTEMS[S.system];
    const total = S.context + S.horizon;
    const left = 34;
    const pw = fc.width - left - 10;
    const i = Math.round(((e.clientX - rect.left - left) / pw) * (total - 1));
    if (i < 0 || i >= total) {
      S.hover = null;
      tooltip.hidden = true;
      drawForecast();
      return;
    }
    S.hover = i;
    const D = sys.dim;
    const t = S.train - 1 - S.context + i;
    const k = i - S.context - 1;
    const u = unit();
    const rows = [];
    const names = D === 3 ? ['x', 'y', 'z'] : ['x'];
    rows.push(h('div', {}, k >= 0 ? `${u.f(k + 1).toFixed(2)} ${u.name} into the forecast` : 'training data'));
    for (let d = 0; d < D; d++) {
      const parts = [`${names[d]}: true ${S.data[t * D + d].toFixed(2)}`];
      if (k >= 0 && k < S.revealed) {
        parts.push(`reservoir ${S.result.pred[k * D + d].toFixed(2)}`);
        if (S.linResult) parts.push(`linear ${fmtNum(S.linResult.pred[k * D + d])}`);
      }
      rows.push(h('div', { class: 'row' }, parts.join(' · ')));
    }
    tooltip.replaceChildren(...rows);
    tooltip.hidden = false;
    const x = e.clientX - rect.left;
    tooltip.style.left = `${Math.min(x + 12, rect.width - tooltip.offsetWidth - 4)}px`;
    tooltip.style.top = `${Math.max(0, e.clientY - rect.top - 40)}px`;
    drawForecast();
  });
  forecastCanvas.addEventListener('pointerleave', () => {
    S.hover = null;
    tooltip.hidden = true;
    drawForecast();
  });

  function fmtNum(v) {
    if (!Number.isFinite(v)) return '∞';
    return Math.abs(v) > 99 ? v.toExponential(0) : v.toFixed(2);
  }

  const P = [0, 0, 0];
  function drawAttractor() {
    const c = colors();
    const { ctx, width: W, height: Hh } = ac;
    ac.clear();
    if (!S.truthCloud) return;
    const s = Math.min(W, Hh) / 6.2;
    const cx = W / 2;
    const cy = Hh / 2;
    ctx.fillStyle = withAlpha(c.text2, c.dark ? 0.22 : 0.18);
    for (const p of S.truthCloud) {
      project(cam, p[0], p[1], p[2], cx, cy, s, P);
      ctx.fillRect(P[0], P[1], 1.2, 1.2);
    }
    // Dream trail: older segments fade out.
    const n = S.trailLen;
    if (n > 1) {
      const segs = 12;
      const per = Math.ceil(n / segs);
      ctx.lineWidth = 1.6;
      ctx.lineJoin = 'round';
      for (let sg = 0; sg < segs; sg++) {
        const a0 = sg * per;
        const a1 = Math.min(n - 1, (sg + 1) * per);
        if (a1 <= a0) continue;
        ctx.strokeStyle = withAlpha(c.s2, 0.12 + 0.88 * ((sg + 1) / segs) ** 2);
        ctx.beginPath();
        for (let k = a0; k <= a1; k++) {
          const idx = ((S.trailHead - n + k + TRAIL) % TRAIL) * 3;
          project(cam, S.trail[idx], S.trail[idx + 1], S.trail[idx + 2], cx, cy, s, P);
          if (k === a0) ctx.moveTo(P[0], P[1]);
          else ctx.lineTo(P[0], P[1]);
        }
        ctx.stroke();
      }
      const idx = ((S.trailHead - 1 + TRAIL) % TRAIL) * 3;
      project(cam, S.trail[idx], S.trail[idx + 1], S.trail[idx + 2], cx, cy, s, P);
      ctx.fillStyle = c.s2;
      ctx.strokeStyle = c.surface;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(P[0], P[1], 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  function drawRaster() {
    const c = colors();
    const { ctx, width: W, height: Hh } = rc;
    rc.clear();
    if (!S.model) return;
    const head = S.rasterHead;
    // Each neuron sits at its own bias level, so show how far it swings
    // away from its own average over the window — that's the "echo".
    for (let r = 0; r < RASTER_ROWS; r++) {
      let m = 0;
      let v = 0;
      const o = r * RASTER_COLS;
      for (let k = 0; k < RASTER_COLS; k++) m += S.raster[o + k];
      m /= RASTER_COLS;
      for (let k = 0; k < RASTER_COLS; k++) v += (S.raster[o + k] - m) ** 2;
      S.rowMean[r] = m;
      S.rowScale[r] = 1 / (2.2 * Math.sqrt(v / RASTER_COLS) + 1e-6);
    }
    drawHeatmap(ctx, RASTER_COLS, RASTER_ROWS, (col, r) => (S.raster[r * RASTER_COLS + ((head + col) % RASTER_COLS)] - S.rowMean[r]) * S.rowScale[r], 0, 0, W, Hh - 18, c.lut);
    ctx.fillStyle = c.muted;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('← older', 2, Hh - 4);
    ctx.textAlign = 'right';
    ctx.fillText('now →', W - 2, Hh - 4);
  }

  applyPreset();
  describe();

  let started = false;
  return {
    show() {
      if (!started) {
        started = true;
        train();
      } else if (S.model) loop.start();
    },
    hide() {
      loop.stop();
    },
  };
}
