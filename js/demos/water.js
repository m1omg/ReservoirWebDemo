// Demo 4 — a simulated bucket of water used as the reservoir.

import { Pond, sineSquareStream } from '../systems/wave.js';
import { RidgeAccumulator, matVec } from '../core/linalg.js';
import { FixedStepLoop, runChunked } from '../core/loop.js';
import { Rng } from '../core/rng.js';
import { h, slider, select, toggle, button, stat, legend, progressBar } from '../ui/dom.js';
import { Canvas2D, colors, onThemeChange, yAxis, strokeSeries, withAlpha } from '../ui/canvas.js';

const TRAIN_SAMPLES = 3000;
const WASHOUT = 100;
const TRACE = 240;
const WINDOW = 600;
const MAX_DELAY = 30;

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
  const pondCanvas = h('canvas', { class: 'plot', 'aria-label': 'Simulated water surface with motors and probe points' });
  const traceCanvas = h('canvas', { class: 'plot', 'aria-label': 'Input signal, correct answer and the readout over time' });
  const curveCanvas = h('canvas', { class: 'plot', 'aria-label': 'How well the past can be recalled, by delay' });
  const traceLegend = h('div');
  const curveCard = h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'How long does the water remember?'), h('span', { class: 'sub' }, 'R² of recalling the input from d steps ago')), curveCanvas);
  const progress = progressBar();
  const status = h('p', { class: 'hint', 'aria-live': 'polite' });

  const stAcc = stat('Accuracy, all samples');
  const stSettled = stat('Accuracy after 1.5 periods');
  const stTime = stat('Training time');
  const stW = stat('Trained weights');

  const taskSel = select({
    label: 'Task',
    value: S.task,
    options: [
      ['shape', 'Tell a sine wave from a square wave'],
      ['memory', 'Remember the past (recall the input from d steps ago)'],
    ],
    onChange: (v) => {
      S.task = v;
      layout();
      train();
    },
  });
  const cam = toggle({ label: 'Nonlinear camera', hint: 'Sees brightness ≈ height², averaged over its exposure, as well as the height itself', checked: true, onChange: () => train() });
  const delay = slider({
    label: 'Delay d',
    min: 0,
    max: MAX_DELAY,
    step: 1,
    value: 8,
    format: (v) => `${v} steps`,
    // Every delay already has its own trained readout; just switch to it.
    onInput: (v) => {
      if (!S.allW || S.task !== 'memory') return;
      S.W = S.allW[v];
      S.win = [];
      drawCurve();
    },
  });
  const damp = slider({ label: 'Damping (how quickly ripples die)', min: 0.01, max: 0.2, step: 0.01, value: 0.06, format: (v) => v.toFixed(2), onChange: () => train() });
  const probes = slider({ label: 'Probe points', min: 8, max: 96, step: 8, value: 48, onChange: () => train() });
  const rate = slider({ label: 'Simulation speed', min: 5, max: 120, step: 5, value: 30, format: (v) => `${v} samples/s`, onInput: (v) => loop.setRate(v) });
  const pauseBtn = button('Pause', () => {
    S.running = !S.running;
    pauseBtn.textContent = S.running ? 'Pause' : 'Resume';
    if (S.running) loop.start();
    else loop.stop();
  });
  const pokeBtn = button('Splash!', () => splash());
  const delayWrap = h('div', {}, delay.el);

  const panel = h(
    'div',
    { class: 'card panel' },
    taskSel.el,
    cam.el,
    delayWrap,
    progress.el,
    status,
    h('h2', {}, 'The bucket'),
    damp.el,
    probes.el,
    h('h2', {}, 'Live run'),
    rate.el,
    h('div', { class: 'btn-row' }, pauseBtn, pokeBtn),
  );

  const explain = h(
    'details',
    { class: 'explain' },
    h('summary', {}, 'Why would water compute anything?'),
    h('p', {}, 'Two “motors” on the left push the water up and down following the input signal. Ripples spread out, bounce off the walls and overlap. At any moment the surface holds a mixture of what the motors did over the last few moments, so it is a physical memory of the input’s recent history.'),
    h('p', {}, 'The probe points are the camera. A linear readout, trained with one ridge regression, combines their readings into the answer. Nothing about the water is trained, and the water doesn’t “know” anything about sine or square waves.'),
    h('p', {}, 'Ripples in this simulation obey a linear wave equation, so from the height alone a linear readout can only produce another linear filter of the input. That is fine for remembering the past, but a filtered sine is still a sine, so it can never settle on a steady “this is a square” answer. Turn the nonlinear camera off and the readout swings around wildly. A real camera looking at water through a light source sees brightness that depends nonlinearly on the surface, and it averages light over its exposure. That one nonlinearity is enough for the readout to tell the shapes apart.'),
    h('p', {}, 'Fernando & Sojakka (2003) did this for real: a tank of water on an overhead projector, LEGO motors and a webcam. It solved XOR and told the spoken words “zero” and “one” apart.'),
  );

  root.append(
    h(
      'div',
      { class: 'view-head' },
      h('h1', {}, 'A bucket of water'),
      h('p', {}, 'Any system with rich enough dynamics can serve as the reservoir. Here a simulated pond does the computing, and the only thing trained is how to read 48 points on its surface.'),
    ),
    h(
      'div',
      { class: 'demo' },
      h(
        'div',
        { class: 'stack' },
        h('div', { class: 'stats' }, stAcc.el, stSettled.el, stTime.el, stW.el),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'The pond'), h('span', { class: 'sub' }, 'motors on the left · probe points shown as rings')), pondCanvas),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'Live readout'), h('span', { class: 'sub' }, 'newest on the right')), traceLegend, traceCanvas),
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
      traceLegend.replaceChildren(legend([['--s1', 'input signal'], ['--s3', 'right answer (up = square)'], ['--s2', 'readout']]));
      stAcc.el.querySelector('.label').textContent = 'Accuracy, all samples';
    } else {
      traceLegend.replaceChildren(legend([['--s3', 'input d steps ago (to recall)'], ['--s2', 'readout']]));
      stAcc.el.querySelector('.label').textContent = 'Recall quality R²';
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
    status.textContent = 'Making waves…';
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
    stTime.set(`${Math.round(ms)}`, 'ms');
    stW.set(String(F), `(${pond.probes.length} probes${squared ? ' × 2' : ''} + bias)`);
    stAcc.set('…');
    stSettled.set('…');
    status.textContent = `${TRAIN_SAMPLES.toLocaleString()} input samples sent through ${(pond.w * pond.h).toLocaleString()} cells of water, readout solved in ${Math.round(ms)} ms. Now testing on a fresh signal.`;
    if (memory) await computeCurve(pond, weights, squared);
    S.training = false;
    if (S.pending) return train();
    if (S.running) loop.start();
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
      stAcc.set(Math.max(0, 1 - se / sv).toFixed(2), `for d = ${delay.value}`);
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
      stAcc.set(`${((ok / S.win.length) * 100).toFixed(1)}%`);
      stSettled.set(sN ? `${((sOk / sN) * 100).toFixed(1)}%` : '…');
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
    ctx.fillText('motors', (p.motors[0].x + 0.5) * sx - 22, (p.motors[0].y + 0.5) * sy - 14);
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
    const toY = yAxis(ctx, { x: left, y: 6, w: W - left - 6, h: H - 12, min, max, ticks: memory ? [-1, 0, 1] : [-1, 0, 0.5, 1], format: (v) => v, c });
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
    const toY = yAxis(ctx, { x: left, y: 6, w: W - left - 6, h: H - bottom - 6, min: 0, max: 1, ticks: [0, 0.5, 1], format: (v) => v, c });
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
      if (!started) {
        started = true;
        train();
      } else if (S.running && !S.training) loop.start();
    },
    hide() {
      loop.stop();
    },
  };
}
