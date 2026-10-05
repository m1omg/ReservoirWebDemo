// Demo 1 — forecasting chaos, then letting the reservoir "dream" the attractor.

import { SYSTEMS, zscore } from '../systems/chaos.js';
import { ReservoirForecaster, LinearForecaster, forecast } from '../core/forecaster.js';
import { FixedStepLoop } from '../core/loop.js';
import { Rng } from '../core/rng.js';
import { h, slider, select, toggle, button, stat, legend, progressBar, rich, paras } from '../ui/dom.js';
import { t, tp, fmt, fmtInt, fmtTick } from '../i18n.js';
import { levelSwitch, levels } from '../ui/explain.js';
import { Canvas2D, colors, onThemeChange, yAxis, strokeSeries, drawHeatmap, makeCamera, project, orbitControls, withAlpha } from '../ui/canvas.js';

const TICKS = 240; // simulation ticks per second (independent of display rate)
const HORIZON = { lorenz: 1200, rossler: 1500, mackeyGlass: 1500 };
const CONTEXT = { lorenz: 220, rossler: 260, mackeyGlass: 260 };
const DATA_SEED = 2024;
const RASTER_COLS = 240;
const RASTER_ROWS = 48;
const TRAIL = 2400;
// Settings that aren't part of a system's preset (see SYSTEMS[*].preset).
const DEFAULTS = { squared: true, compare: true, dreamRate: 120, seed: 1 };

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
  const forecastCanvas = h('canvas', { class: 'plot', 'aria-label': t('chaos.aria.forecast') });
  const tooltip = h('div', { class: 'tooltip', hidden: true });
  const legendBox = h('div');
  const attractorCanvas = h('canvas', { class: 'plot', 'aria-label': t('chaos.aria.attractor') });
  const rasterCanvas = h('canvas', { class: 'plot', 'aria-label': t('chaos.aria.raster') });
  const progress = progressBar();
  const status = h('p', { class: 'hint', 'aria-live': 'polite' }, '');

  const stTrain = stat(t('common.trainingTime'));
  const stValid = stat(t('chaos.stats.valid'));
  const stLinear = stat(t('chaos.stats.linear'));
  const stFeat = stat(t('common.trainedWeights'));

  const sysSelect = select({
    label: t('chaos.controls.system'),
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

  const f2 = (v) => fmt(v, 2);
  const sz = slider({ label: t('common.neurons'), min: 50, max: 800, step: 50, value: 300, format: fmtInt, onChange: () => train() });
  const sr = slider({ label: t('chaos.controls.spectral'), min: 0.05, max: 1.6, step: 0.05, value: 0.3, format: f2, onChange: () => train() });
  const lk = slider({ label: t('common.leak'), min: 0.05, max: 1, step: 0.05, value: 1, format: f2, onChange: () => train() });
  const ins = slider({ label: t('chaos.controls.input'), min: 0.05, max: 2, step: 0.05, value: 0.3, format: f2, onChange: () => train() });
  const lam = slider({ label: t('chaos.controls.lambda'), min: 1e-14, max: 1e-2, log: true, value: 1e-11, format: (v) => v.toExponential(0), onChange: () => train() });
  const tl = slider({ label: t('chaos.controls.steps'), min: 500, max: 10000, step: 500, value: 4000, format: fmtInt, onChange: () => train() });
  const sq = toggle({ label: t('chaos.controls.squared'), hint: t('chaos.controls.squaredHint'), checked: DEFAULTS.squared, onChange: () => train() });
  const cmp = toggle({ label: t('chaos.controls.compare'), hint: t('chaos.controls.compareHint'), checked: DEFAULTS.compare, onChange: () => train() });
  const speed = slider({
    label: t('chaos.controls.speed'),
    min: 10,
    max: 600,
    step: 10,
    value: S.dreamRate,
    format: (v) => t('chaos.controls.speedUnit', { v: fmtInt(v) }),
    onInput: (v) => (S.dreamRate = v),
  });

  const trainBtn = button(t('chaos.controls.train'), () => train(), 'primary');
  const newBtn = button(t('chaos.controls.newReservoir'), () => {
    S.seed++;
    train();
  });
  const perturbBtn = button(t('chaos.controls.perturb'), () => {
    if (!S.model) return;
    S.model.perturb(S.rng, SYSTEMS[S.system].kick);
    dreamStatus.textContent = t('chaos.status.kicked');
  });
  const restartBtn = button(t('chaos.controls.restart'), () => restartDream());
  const dreamStatus = h('p', { class: 'hint', 'aria-live': 'polite' });
  const resetNote = h('p', { class: 'hint', 'aria-live': 'polite' });
  const resetBtn = button(t('common.reset'), () => resetSettings(), 'small');

  function restartDream() {
    if (!S.model) return;
    // Rewind to the end of training and replay the (deterministic) forecast.
    S.model.restore(S.snapshot);
    S.result = forecast(S.model, S.data, SYSTEMS[S.system].dim, 0, S.train, S.horizon);
    startDream();
  }

  // Back to the defaults for the selected system (the system itself stays).
  function resetSettings() {
    applyPreset();
    sq.checked = DEFAULTS.squared;
    cmp.checked = DEFAULTS.compare;
    speed.value = DEFAULTS.dreamRate;
    S.dreamRate = DEFAULTS.dreamRate;
    S.seed = DEFAULTS.seed;
    describe();
    train();
    resetNote.textContent = t('common.resetDone');
  }

  const panel = h(
    'div',
    { class: 'card panel' },
    sysSelect.el,
    sysDesc,
    h('div', { class: 'btn-row' }, trainBtn, newBtn),
    progress.el,
    status,
    h('h2', {}, t('common.reservoir')),
    sz.el,
    sr.el,
    lk.el,
    ins.el,
    h('h2', {}, t('chaos.controls.readout')),
    lam.el,
    tl.el,
    sq.el,
    cmp.el,
    h('h2', {}, t('chaos.controls.dreaming')),
    speed.el,
    h('div', { class: 'btn-row' }, perturbBtn, restartBtn),
    dreamStatus,
    h('div', { class: 'panel-foot' }, resetBtn, resetNote),
  );

  const explain = h('details', { class: 'explain' }, h('summary', {}, t('chaos.explain.summary')), levels(paras(t('chaos.explain.simple')), paras(t('chaos.explain.detailed'))));

  root.append(
    h(
      'div',
      { class: 'view-head' },
      h('h1', {}, t('chaos.title')),
      levelSwitch(),
      levels([h('p', {}, rich(t('chaos.lead.simple'))), h('p', { class: 'try' }, h('b', {}, t('common.tryIt')), ' ', rich(t('chaos.try')))], h('p', {}, rich(t('chaos.lead.detailed')))),
    ),
    h(
      'div',
      { class: 'demo' },
      h(
        'div',
        { class: 'stack' },
        h('div', { class: 'stats' }, stTrain.el, stValid.el, stLinear.el, stFeat.el),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('chaos.cards.forecast')), h('span', { class: 'sub' }, t('chaos.cards.forecastSub'))), legendBox, h('div', { class: 'rel' }, forecastCanvas, tooltip)),
        h(
          'div',
          { class: 'grid-2' },
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('chaos.cards.attractor')), h('span', { class: 'sub' }, t('chaos.cards.drag'))), legend([['--axis', t('chaos.legend.trueAttractor')], ['--s2', t('chaos.legend.free')]]), attractorCanvas),
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('chaos.cards.inside')), h('span', { class: 'sub' }, t('chaos.cards.insideSub', { rows: RASTER_ROWS, cols: RASTER_COLS }))), legend([['--div-neg', t('chaos.legend.below')], ['--div-pos', t('chaos.legend.above')]]), rasterCanvas),
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
    sysDesc.textContent = t(`chaos.systems.${S.system}`);
    legendBox.replaceChildren(
      legend([['--s1', t('chaos.legend.truth')], ['--s2', t('chaos.legend.forecast')], ...(cmp.checked ? [['--s3', t('chaos.legend.linear'), true]] : [])]),
    );
  }

  const unit = () => {
    const sys = SYSTEMS[S.system];
    const base = sys.lyapunov ? 'chaos.units.lt' : 'chaos.units.tu';
    return {
      f: sys.lyapunov ? (steps) => steps * sys.dt * sys.lyapunov : (steps) => steps * sys.dt,
      short: t(sys.lyapunov ? 'chaos.units.ltShort' : 'chaos.units.tuShort'),
      digits: sys.lyapunov ? 1 : 0,
      // Unit words agree with the number shown (Slovak declines them).
      name: (value, digits) => (digits > 0 ? t(`${base}.dec`) : tp(base, Math.round(value))),
    };
  };

  async function train() {
    if (S.training) {
      S.pending = true;
      return;
    }
    S.training = true;
    S.pending = false;
    trainBtn.disabled = true;
    dreamStatus.textContent = '';
    loop.stop();
    const sys = SYSTEMS[S.system];
    const T = tl.value;
    const H = HORIZON[S.system];
    status.textContent = t('chaos.status.generating');
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
    status.textContent = t('chaos.status.running', { steps: fmtInt(T), n: fmtInt(sz.value) });
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
    stTrain.set(ms < 1000 ? fmtInt(ms) : fmt(ms / 1000, 2), ms < 1000 ? t('common.ms') : t('common.s'));
    const valid = u.f(S.result.valid);
    const shown = S.result.valid >= H ? u.f(H) : valid;
    stValid.set(S.result.valid >= H ? `> ${fmt(shown, u.digits)}` : fmt(shown, u.digits), u.name(shown, u.digits));
    if (S.linResult) {
      const lin = u.f(S.linResult.valid);
      stLinear.set(fmt(lin, u.digits ? 2 : 0), u.name(lin, u.digits ? 2 : 0));
    }
    else stLinear.set('–');
    stFeat.set(fmtInt(model.featureCount * sys.dim), t('chaos.stats.samples', { n: fmtInt(model.samples) }));
    status.textContent = t('chaos.status.trained', { ms: fmtInt(ms), samples: fmtInt(model.samples) });
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
    S.health = { n: 0, mean: 0, sq: 1, escapes: 0 };
    if (S.visible) loop.start();
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
    // The data is z-scored, so the real attractor lives within about ±3.
    // Clip what is fed back so a runaway can't explode, and watch the dream's
    // health: if it escapes or collapses to a point, say so and restart.
    let escaped = false;
    for (let j = 0; j < y.length; j++) {
      if (!Number.isFinite(y[j])) y[j] = 0;
      if (Math.abs(y[j]) > 4.5) {
        y[j] = Math.sign(y[j]) * 4.5;
        escaped = true;
      }
    }
    const H = S.health;
    H.n++;
    H.mean += (y[0] - H.mean) / 200;
    H.sq += (y[0] * y[0] - H.sq) / 200;
    if (escaped) H.escapes++;
    if (H.n > 400 && (H.escapes > 20 || H.sq - H.mean * H.mean < 0.03)) {
      dreamStatus.textContent = H.escapes > 20 ? t('chaos.status.escaped') : t('chaos.status.collapsed');
      restartDream();
      return;
    }
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
      const toY = yAxis(ctx, { x: left, y: y0, w: pw, h: ph, min: -3, max: 3, ticks: [-2, 0, 2], format: fmtTick, c });
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
    ctx.fillText(t('chaos.canvas.trainingEnds'), nowX - 4, Hh - bottom + 14);
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
      const v = fmt(u.f(S.result.valid), u.digits);
      let txt = t('chaos.canvas.accurate', { v, unit: u.name(u.f(S.result.valid), u.digits) });
      if (ctx.measureText(txt).width > W - left - right) txt = t('chaos.canvas.accurateShort', { v, unit: u.short });
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
      ctx.fillText(`${fmtTick(v)} ${u.short}`, x, Hh - 6);
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
    const ti = S.train - 1 - S.context + i;
    const k = i - S.context - 1;
    const u = unit();
    const rows = [];
    const names = D === 3 ? ['x', 'y', 'z'] : ['x'];
    rows.push(h('div', {}, k >= 0 ? t('chaos.tooltip.into', { v: fmt(u.f(k + 1), 2), unit: u.name(u.f(k + 1), 2) }) : t('chaos.tooltip.training')));
    for (let d = 0; d < D; d++) {
      const parts = [`${names[d]}: ${t('chaos.tooltip.true')} ${fmt(S.data[ti * D + d], 2)}`];
      if (k >= 0 && k < S.revealed) {
        parts.push(`${t('chaos.tooltip.reservoir')} ${fmt(S.result.pred[k * D + d], 2)}`);
        if (S.linResult) parts.push(`${t('chaos.tooltip.linear')} ${fmtNum(S.linResult.pred[k * D + d])}`);
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
    return Math.abs(v) > 99 ? v.toExponential(0) : fmt(v, 2);
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
    ctx.fillText(t('common.older'), 2, Hh - 4);
    ctx.textAlign = 'right';
    ctx.fillText(t('common.now'), W - 2, Hh - 4);
  }

  applyPreset();
  describe();

  let started = false;
  return {
    show() {
      S.visible = true;
      if (!started) {
        started = true;
        train();
      } else if (S.model && !S.training) loop.start();
    },
    hide() {
      S.visible = false;
      loop.stop();
    },
  };
}
