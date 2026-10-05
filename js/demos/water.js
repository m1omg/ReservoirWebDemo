// Demo 4 — a simulated bucket of water used as the reservoir.

import { Pond, sineSquareStream } from '../systems/wave.js';
import { RidgeAccumulator, matVec } from '../core/linalg.js';
import { FixedStepLoop, runChunked } from '../core/loop.js';
import { Rng } from '../core/rng.js';
import { h, slider, select, toggle, button, stat, legend, progressBar, rich, paras } from '../ui/dom.js';
import { t, tp, fmt, fmtInt, fmtTick, fmtPct } from '../i18n.js';
import { levelSwitch, levels } from '../ui/explain.js';
import { Canvas2D, colors, onThemeChange, yAxis, strokeSeries, withAlpha } from '../ui/canvas.js';

const TRAIN_SAMPLES = 3000;
const WASHOUT = 100;
const TRACE = 240;
const WINDOW = 600;
const MAX_DELAY = 30;
const DEFAULTS = { camera: true, delay: 8, damping: 0.06, probes: 48, rate: 30 };

export function mount(root) {
  const S = {
    task: 'shape',
    pond: null,
    W: null,
    F: 0,
    f: null,
    stream: null,
    rng: new Rng(5),
    hist: new Float32Array(MAX_DELAY + 1),
    trace: { u: new Float32Array(TRACE), target: new Float32Array(TRACE), y: new Float32Array(TRACE), n: 0, head: 0 },
    win: [],
    since: 0,
    lastLabel: -1,
    curve: null,
    training: false,
    running: true,
    image: null,
  };

  // ---------- DOM ----------
  const pondCanvas = h('canvas', { class: 'plot', 'aria-label': t('water.aria.pond') });
  const traceCanvas = h('canvas', { class: 'plot', 'aria-label': t('water.aria.trace') });
  const curveCanvas = h('canvas', { class: 'plot', 'aria-label': t('water.aria.curve') });
  const traceLegend = h('div');
  const curveCard = h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('water.curve')), h('span', { class: 'sub' }, t('water.curveSub'))), curveCanvas);
  const progress = progressBar();
  const status = h('p', { class: 'hint', 'aria-live': 'polite' });

  const stAcc = stat(t('water.stats.accuracy'));
  const stSettled = stat(t('water.stats.settled'));
  const stTime = stat(t('common.trainingTime'));
  const stW = stat(t('common.trainedWeights'));

  const taskSel = select({
    label: t('water.task'),
    value: S.task,
    options: [
      ['shape', t('water.tasks.shape')],
      ['memory', t('water.tasks.memory')],
    ],
    onChange: (v) => {
      S.task = v;
      layout();
      train();
    },
  });
  const cam = toggle({ label: t('water.camera'), hint: t('water.cameraHint'), checked: DEFAULTS.camera, onChange: () => train() });
  const delay = slider({
    label: t('water.delay'),
    min: 0,
    max: MAX_DELAY,
    step: 1,
    value: DEFAULTS.delay,
    format: (v) => tp('water.delayUnit', v),
    // Every delay already has its own trained readout; just switch to it.
    onInput: (v) => {
      if (!S.allW || S.task !== 'memory') return;
      S.W = S.allW[v];
      S.win = [];
      drawCurve();
    },
  });
  const damp = slider({ label: t('water.damping'), min: 0.01, max: 0.2, step: 0.01, value: DEFAULTS.damping, format: (v) => fmt(v, 2), onChange: () => train() });
  const probes = slider({ label: t('water.probes'), min: 8, max: 96, step: 8, value: DEFAULTS.probes, format: fmtInt, onChange: () => train() });
  const rate = slider({ label: t('water.speed'), min: 5, max: 120, step: 5, value: DEFAULTS.rate, format: (v) => t('water.speedUnit', { v: fmtInt(v) }), onInput: (v) => loop.setRate(v) });
  const pauseBtn = button(t('water.pause'), () => {
    S.running = !S.running;
    pauseBtn.textContent = S.running ? t('water.pause') : t('water.resume');
    if (S.running && S.visible) loop.start();
    else loop.stop();
  });
  const pokeBtn = button(t('water.splash'), () => splash());
  const delayWrap = h('div', {}, delay.el);
  const resetNote = h('p', { class: 'hint', 'aria-live': 'polite' });
  const resetBtn = button(t('common.reset'), () => {
    cam.checked = DEFAULTS.camera;
    delay.value = DEFAULTS.delay;
    damp.value = DEFAULTS.damping;
    probes.value = DEFAULTS.probes;
    rate.value = DEFAULTS.rate;
    loop.setRate(DEFAULTS.rate);
    train();
    resetNote.textContent = t('common.resetDone');
  }, 'small');

  const panel = h(
    'div',
    { class: 'card panel' },
    taskSel.el,
    cam.el,
    delayWrap,
    progress.el,
    status,
    h('h2', {}, t('water.bucket')),
    damp.el,
    probes.el,
    h('h2', {}, t('water.live')),
    rate.el,
    h('div', { class: 'btn-row' }, pauseBtn, pokeBtn),
    h('div', { class: 'panel-foot' }, resetBtn, resetNote),
  );

  const explain = h('details', { class: 'explain' }, h('summary', {}, t('water.explain.summary')), levels(paras(t('water.explain.simple')), paras(t('water.explain.detailed'))));

  root.append(
    h(
      'div',
      { class: 'view-head' },
      h('h1', {}, t('water.title')),
      levelSwitch(),
      levels([h('p', {}, rich(t('water.lead.simple'))), h('p', { class: 'try' }, h('b', {}, t('common.tryIt')), ' ', rich(t('water.try')))], h('p', {}, rich(t('water.lead.detailed')))),
    ),
    h(
      'div',
      { class: 'demo' },
      h(
        'div',
        { class: 'stack' },
        h('div', { class: 'stats' }, stAcc.el, stSettled.el, stTime.el, stW.el),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('water.pond')), h('span', { class: 'sub' }, t('water.pondSub'))), pondCanvas),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('water.readout')), h('span', { class: 'sub' }, t('water.readoutSub'))), traceLegend, traceCanvas),
        curveCard,
        explain,
      ),
      panel,
    ),
  );

  const pc = new Canvas2D(pondCanvas, { aspect: 1.5, minHeight: 200, maxHeight: 520, onResize: () => drawPond() });
  const tc = new Canvas2D(traceCanvas, { aspect: 3.2, minHeight: 160, maxHeight: 260, onResize: () => drawTrace() });
  const cc = new Canvas2D(curveCanvas, { aspect: 3.2, minHeight: 150, maxHeight: 240, onResize: () => drawCurve() });
  onThemeChange(() => {
    drawPond();
    drawTrace();
    drawCurve();
  });

  // ---------- Training ----------

  function layout() {
    delayWrap.hidden = S.task !== 'memory';
    curveCard.hidden = S.task !== 'memory';
    stSettled.el.hidden = S.task !== 'shape';
    if (S.task === 'shape') {
      traceLegend.replaceChildren(legend([['--s1', t('water.legend.input')], ['--s3', t('water.legend.answer')], ['--s2', t('water.legend.readout')]]));
      stAcc.el.querySelector('.label').textContent = t('water.stats.accuracy');
    } else {
      traceLegend.replaceChildren(legend([['--s3', t('water.legend.past')], ['--s2', t('water.legend.readout')]]));
      stAcc.el.querySelector('.label').textContent = t('water.stats.recall');
    }
  }

  function newPond() {
    return new Pond({ damping: damp.value, probes: probes.value });
  }

  async function train() {
    if (S.training) {
      S.pending = true;
      return;
    }
    S.training = true;
    S.pending = false;
    loop.stop();
    const pond = newPond();
    const squared = cam.checked;
    const F = pond.featureLength(squared);
    const f = new Float64Array(F);
    const memory = S.task === 'memory';
    const stream = memory ? null : sineSquareStream(1);
    const rng = new Rng(11);
    const hist = new Float32Array(MAX_DELAY + 1);
    // For the memory task every delay shares the same features, so one
    // multi-output regression trains all 31 readouts (one per delay) at once.
    const acc = new RidgeAccumulator(F, memory ? MAX_DELAY + 1 : 1);
    const target1 = new Float64Array(1);
    status.textContent = t('water.status.making');
    const t0 = performance.now();
    await runChunked(
      TRAIN_SAMPLES,
      (t) => {
        let u;
        let target;
        if (memory) {
          u = rng.uniform(-1, 1);
          hist.copyWithin(1, 0, MAX_DELAY);
          hist[0] = u;
        } else {
          const s = stream.next();
          u = s.u;
          target = s.label;
        }
        pond.step([u]);
        if (t < WASHOUT) return;
        pond.read(f, squared);
        if (memory) acc.add(f, hist);
        else {
          target1[0] = target;
          acc.add(f, target1);
        }
      },
      { onProgress: (p) => progress.set(p) },
    );
    const Wall = acc.solve(1e-4);
    const weights = Array.from({ length: acc.nOut }, (_, d) => Wall.subarray(d * F, (d + 1) * F));
    const ms = performance.now() - t0;

    S.pond = pond;
    S.F = F;
    S.f = f;
    S.squared = squared;
    S.W = memory ? weights[delay.value] : weights[0];
    S.allW = weights;
    S.stream = memory ? null : sineSquareStream(777);
    S.hist = new Float32Array(MAX_DELAY + 1);
    S.win = [];
    S.trace.n = 0;
    S.trace.head = 0;
    S.since = 0;
    S.lastLabel = -1;
    S.settledOk = 0;
    S.settledN = 0;
    stTime.set(fmtInt(ms), t('common.ms'));
    stW.set(fmtInt(F), t('water.stats.weights', { p: fmtInt(pond.probes.length), x: squared ? ' × 2' : '' }));
    stAcc.set('…');
    stSettled.set('…');
    status.textContent = t('water.status.trained', { samples: fmtInt(TRAIN_SAMPLES), cells: fmtInt(pond.w * pond.h), ms: fmtInt(ms) });
    if (memory) await computeCurve(pond, weights, squared);
    S.training = false;
    if (S.pending) return train();
    if (S.running && S.visible) loop.start();
  }

  // Memory curve measured on a fresh random signal.
  async function computeCurve(trainedPond, weights, squared) {
    const pond = newPond();
    const F = pond.featureLength(squared);
    const f = new Float64Array(F);
    const rng = new Rng(4242);
    const hist = new Float32Array(MAX_DELAY + 1);
    const se = new Float64Array(MAX_DELAY + 1);
    const sv = new Float64Array(MAX_DELAY + 1);
    const y = new Float64Array(1);
    await runChunked(1500, (t) => {
      const u = rng.uniform(-1, 1);
      hist.copyWithin(1, 0, MAX_DELAY);
      hist[0] = u;
      pond.step([u]);
      if (t < WASHOUT + MAX_DELAY) return;
      pond.read(f, squared);
      for (let d = 0; d <= MAX_DELAY; d++) {
        matVec(weights[d], 1, F, f, y);
        se[d] += (y[0] - hist[d]) ** 2;
        sv[d] += hist[d] ** 2;
      }
    });
    S.curve = Array.from(se, (e, d) => Math.max(0, 1 - e / sv[d]));
    drawCurve();
  }

  // ---------- Live run ----------

  function liveStep() {
    if (!S.pond) return;
    const memory = S.task === 'memory';
    let u;
    let target;
    if (memory) {
      u = S.rng.uniform(-1, 1);
      S.hist.copyWithin(1, 0, MAX_DELAY);
      S.hist[0] = u;
      target = S.hist[delay.value];
    } else {
      const s = S.stream.next();
      u = s.u;
      target = s.label;
      if (target !== S.lastLabel) S.since = 0;
      else S.since++;
      S.lastLabel = target;
    }
    S.pond.step([u]);
    S.pond.read(S.f, S.squared);
    const y = matVec(S.W, 1, S.F, S.f)[0];
    const tr = S.trace;
    tr.u[tr.head] = u;
    tr.target[tr.head] = target;
    tr.y[tr.head] = y;
    tr.head = (tr.head + 1) % TRACE;
    tr.n = Math.min(TRACE, tr.n + 1);

    if (memory) {
      S.win.push([y, target]);
    } else {
      const ok = (y > 0.5 ? 1 : 0) === target;
      S.win.push([ok ? 1 : 0, S.since > 1.5 * S.stream.period ? 1 : 0]);
    }
    if (S.win.length > WINDOW) S.win.shift();
    S.tick = (S.tick || 0) + 1;
    if (S.tick % 10 === 0) updateStats();
  }

  function updateStats() {
    if (S.win.length < 30) return;
    if (S.task === 'memory') {
      let se = 0;
      let sv = 0;
      for (const [y, t] of S.win) {
        se += (y - t) ** 2;
        sv += t * t;
      }
      stAcc.set(fmt(Math.max(0, 1 - se / sv), 2), t('water.stats.forD', { d: delay.value }));
    } else {
      let ok = 0;
      let sOk = 0;
      let sN = 0;
      for (const [c, settled] of S.win) {
        ok += c;
        if (settled) {
          sN++;
          sOk += c;
        }
      }
      stAcc.set(fmtPct(ok / S.win.length));
      stSettled.set(sN ? fmtPct(sOk / sN) : '…');
    }
  }

  function splash() {
    if (!S.pond) return;
    const p = S.pond;
    const x = 20 + S.rng.int(p.w - 26);
    const y = 4 + S.rng.int(p.h - 8);
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) p.u[(y + dy) * p.w + x + dx] += 0.6 * Math.exp(-(dx * dx + dy * dy) / 3);
  }

  const loop = new FixedStepLoop({
    stepsPerSecond: rate.value,
    step: liveStep,
    render: () => {
      drawPond();
      drawTrace();
    },
  });

  // ---------- Drawing ----------

  function drawPond() {
    const c = colors();
    const { ctx, width: W, height: H } = pc;
    pc.clear();
    const p = S.pond;
    if (!p) return;
    if (!S.image || S.image.width !== p.w || S.image.height !== p.h) {
      const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(p.w, p.h) : Object.assign(document.createElement('canvas'), { width: p.w, height: p.h });
      S.image = { cv, cx: cv.getContext('2d'), width: p.w, height: p.h };
      S.image.img = S.image.cx.createImageData(p.w, p.h);
    }
    const d = S.image.img.data;
    const lut = c.lut;
    const u = p.u;
    for (let y = 0; y < p.h; y++) {
      for (let x = 0; x < p.w; x++) {
        const i = y * p.w + x;
        const v = Math.tanh(u[i] * 2.5);
        const k = Math.round((v + 1) * 127.5) * 3;
        // Light from the top-left gives the surface some relief.
        const gx = u[Math.min(i + 1, u.length - 1)] - u[Math.max(i - 1, 0)];
        const gy = u[Math.min(i + p.w, u.length - 1)] - u[Math.max(i - p.w, 0)];
        const light = 1 + Math.max(-0.35, Math.min(0.35, (gx + gy) * 6));
        const o = i * 4;
        d[o] = lut[k] * light;
        d[o + 1] = lut[k + 1] * light;
        d[o + 2] = lut[k + 2] * light;
        d[o + 3] = 255;
      }
    }
    S.image.cx.putImageData(S.image.img, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    const r = 10;
    ctx.beginPath();
    ctx.roundRect(0, 0, W, H, r);
    ctx.clip();
    ctx.drawImage(S.image.cv, 0, 0, W, H);
    ctx.restore();
    const sx = W / p.w;
    const sy = H / p.h;
    // Probes.
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = withAlpha(c.text, 0.75);
    for (const idx of p.probes) {
      const px = ((idx % p.w) + 0.5) * sx;
      const py = (Math.floor(idx / p.w) + 0.5) * sy;
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }
    // Motors bob with their input.
    const last = S.trace.n ? S.trace.u[(S.trace.head - 1 + TRACE) % TRACE] : 0;
    p.motors.forEach((m) => {
      const px = (m.x + 0.5) * sx;
      const py = (m.y + 0.5) * sy;
      const rr = 7 + 3 * last * Math.sign(m.gain);
      ctx.fillStyle = c.s1;
      ctx.strokeStyle = c.surface;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, Math.max(3, rr), 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
    ctx.fillStyle = c.text;
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(t('water.motors'), (p.motors[0].x + 0.5) * sx - 22, (p.motors[0].y + 0.5) * sy - 14);
  }

  function drawTrace() {
    const c = colors();
    const { ctx, width: W, height: H } = tc;
    tc.clear();
    const tr = S.trace;
    if (!tr.n) return;
    const left = 30;
    const memory = S.task === 'memory';
    const min = memory ? -1.4 : -1.2;
    const max = memory ? 1.4 : 1.6;
    const toY = yAxis(ctx, { x: left, y: 6, w: W - left - 6, h: H - 12, min, max, ticks: memory ? [-1, 0, 1] : [-1, 0, 0.5, 1], format: fmtTick, c });
    const n = tr.n;
    const toX = (i) => left + ((TRACE - n + i) / (TRACE - 1)) * (W - left - 6);
    const at = (arr) => (i) => arr[(tr.head - n + i + TRACE) % TRACE];
    if (memory) {
      strokeSeries(ctx, n, at(tr.target), toX, toY, c.s3, 2);
      strokeSeries(ctx, n, (i) => Math.max(min, Math.min(max, at(tr.y)(i))), toX, toY, c.s2, 2);
    } else {
      strokeSeries(ctx, n, at(tr.u), toX, toY, withAlpha(c.s1, 0.55), 1.5);
      strokeSeries(ctx, n, at(tr.target), toX, toY, c.s3, 2);
      strokeSeries(ctx, n, (i) => Math.max(min, Math.min(max, at(tr.y)(i))), toX, toY, c.s2, 2);
      // Decision threshold.
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = c.muted;
      ctx.beginPath();
      ctx.moveTo(left, Math.round(toY(0.5)) + 0.5);
      ctx.lineTo(W - 6, Math.round(toY(0.5)) + 0.5);
      ctx.stroke();
      ctx.restore();
    }
  }

  function drawCurve() {
    const c = colors();
    const { ctx, width: W, height: H } = cc;
    cc.clear();
    if (!S.curve) return;
    const left = 30;
    const bottom = 20;
    const toY = yAxis(ctx, { x: left, y: 6, w: W - left - 6, h: H - bottom - 6, min: 0, max: 1, ticks: [0, 0.5, 1], format: fmtTick, c });
    const n = S.curve.length;
    const band = (W - left - 6) / n;
    const bw = Math.min(24, band - 2);
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    for (let d = 0; d < n; d++) {
      const x = left + d * band + (band - bw) / 2;
      const y0 = toY(0);
      const y1 = toY(S.curve[d]);
      ctx.fillStyle = d === delay.value ? c.s2 : c.s1;
      const hgt = Math.max(0, y0 - y1);
      const r = Math.min(4, bw / 2, hgt);
      ctx.beginPath();
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1 + r);
      ctx.quadraticCurveTo(x, y1, x + r, y1);
      ctx.lineTo(x + bw - r, y1);
      ctx.quadraticCurveTo(x + bw, y1, x + bw, y1 + r);
      ctx.lineTo(x + bw, y0);
      ctx.closePath();
      ctx.fill();
      if (d % 5 === 0) {
        ctx.fillStyle = c.muted;
        ctx.fillText(`d=${d}`, x + bw / 2, H - 5);
      }
    }
  }

  layout();
  let started = false;
  return {
    show() {
      S.visible = true;
      if (!started) {
        started = true;
        train();
      } else if (S.running && !S.training) loop.start();
    },
    hide() {
      S.visible = false;
      loop.stop();
    },
  };
}
