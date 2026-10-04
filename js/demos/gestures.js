// Demo 2 — gesture recognition with a reservoir, including few-shot teaching.

import { BUILTIN, TEMPLATES, makeDataset, expandExamples, preprocess, compact, RESAMPLE, INPUTS } from '../systems/gestures.js';
import { SequenceClassifier } from '../core/sequence-classifier.js';
import { runChunked } from '../core/loop.js';
import { Rng } from '../core/rng.js';
import { h, slider, button, stat, probBars, storage, progressBar } from '../ui/dom.js';
import { Canvas2D, colors, onThemeChange, drawHeatmap } from '../ui/canvas.js';

const STORE_KEY = 'rc-gestures-v1';
const SYNTH_PER_CLASS = 50;
const TEST_PER_CLASS = 40;
const FINISH_MS = 650;
const GUESS_EVERY_MS = 60;
const MIN_EXAMPLES = 3;
const HEAT_ROWS = 48;
const LABELS = { circle: 'circle', triangle: 'triangle', square: 'square', check: 'check ✓', zigzag: 'zigzag', spiral: 'spiral', star: 'star', wave: 'wave', cross: 'cross ✕' };
const pretty = (n) => LABELS[n] || n;

export function mount(root) {
  const saved = storage.get(STORE_KEY, null);
  const S = {
    user: saved?.user || {}, // name → [strokes]
    removed: new Set(saved?.removed || []),
    clf: null,
    synth: null, // cached synthetic training examples (with features)
    test: null,
    strokes: [],
    drawing: false,
    finishTimer: 0,
    lastGuess: 0,
    finished: false,
    teaching: null, // { name, examples: [] }
    fingerprint: null,
    lastSeq: null,
    busy: false,
    rng: new Rng(17),
  };

  // ---------- DOM ----------
  const drawCanvas = h('canvas', { class: 'draw', 'aria-label': 'Drawing area' });
  const drawHint = h('div', { class: 'draw-hint' }, 'Draw a shape here: circle, triangle, square, check, zigzag, spiral, star, wave or cross.');
  const banner = h('div', { class: 'mode-banner', hidden: true });
  const verdict = h('div', { class: 'verdict', 'aria-live': 'polite' }, h('small', {}, 'Draw something…'));
  const bars = probBars();
  const heatCanvas = h('canvas', { class: 'plot', 'aria-label': 'Reservoir activity while reading the gesture' });
  const classList = h('div', { class: 'class-list' });
  const nameInput = h('input', { type: 'text', placeholder: 'Name, e.g. heart, arrow, M', maxlength: 24, 'aria-label': 'Name of the new gesture' });
  const progress = progressBar();
  const status = h('p', { class: 'hint', 'aria-live': 'polite' });
  const readyMark = h('span', { hidden: true });

  const stAcc = stat('Accuracy on unseen drawings');
  const stTime = stat('Training time');
  const stN = stat('Training examples');

  const teachBtn = button('Teach it', () => startTeaching(nameInput.value));
  const doneBtn = button('Done', () => finishTeaching(), 'primary');
  const cancelBtn = button('Cancel', () => cancelTeaching());
  const clearBtn = button('Clear', () => clearDrawing());
  const resetBtn = button('Reset to built-in gestures', () => {
    S.user = {};
    S.removed = new Set();
    persist();
    retrain({ features: false });
  }, 'small');

  nameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') startTeaching(nameInput.value);
  });

  const sz = slider({ label: 'Neurons', min: 50, max: 500, step: 25, value: 200, onChange: () => rebuild() });
  const sr = slider({ label: 'Spectral radius', min: 0.1, max: 1.5, step: 0.05, value: 0.9, format: (v) => v.toFixed(2), onChange: () => rebuild() });
  const lk = slider({ label: 'Leak rate', min: 0.05, max: 1, step: 0.05, value: 0.3, format: (v) => v.toFixed(2), onChange: () => rebuild() });

  const teachRow = h('div', { class: 'btn-row' }, doneBtn, cancelBtn);
  teachRow.hidden = true;

  const panel = h(
    'div',
    { class: 'stack' },
    h(
      'div',
      { class: 'card panel' },
      h('h2', {}, 'Gestures it knows'),
      h('p', { class: 'hint' }, 'Tap + to add more of your own examples to a gesture.'),
      classList,
      h('h2', {}, 'Teach a new gesture'),
      h('div', { class: 'word-row', style: { gridTemplateColumns: 'minmax(0,1fr) auto' } }, nameInput, teachBtn),
      banner,
      teachRow,
      progress.el,
      status,
      resetBtn,
    ),
    h('div', { class: 'card panel' }, h('h2', {}, 'Reservoir'), sz.el, sr.el, lk.el, h('p', { class: 'hint', style: { margin: 0 } }, 'Changing these builds a new random reservoir and retrains everything.')),
  );

  const explain = h(
    'details',
    { class: 'explain' },
    h('summary', {}, 'How does it recognise a drawing?'),
    h('p', {}, `Your stroke is resampled to ${RESAMPLE} evenly spaced points. They are fed to the reservoir one at a time as position, direction of travel, how sharply it bends and a “pen lifted” flag. Like a pond after a stone drops in, the reservoir’s neurons keep echoing what came earlier, so by the end the state holds a fingerprint of the whole movement.`),
    h('p', {}, 'The readout takes snapshots of that state at 25%, 50%, 75% and 100% of the gesture, plus its average, and a single ridge regression maps those snapshots to gesture names. Adding a new gesture means re-solving that one linear system, which takes well under a second. No gradient descent is involved.'),
    h('p', {}, 'The built-in gestures are trained on drawings generated by the computer, so it has never seen your handwriting. It is tested on fresh generated drawings it has never seen. Partly drawn shapes are in the training data too, which is why it can guess before you finish.'),
  );

  root.append(
    h(
      'div',
      { class: 'view-head' },
      h('h1', {}, 'Gesture recognition'),
      h('p', {}, 'Draw with a mouse, finger or pen. A fixed random network turns your movement into a pattern of activity, and a linear readout names it. You can teach it new gestures from three examples.'),
    ),
    h(
      'div',
      { class: 'demo' },
      h(
        'div',
        { class: 'stack' },
        h('div', { class: 'stats' }, stAcc.el, stTime.el, stN.el),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'Draw here'), h('span', { class: 'btn-row' }, clearBtn)), h('div', { class: 'draw-wrap' }, drawCanvas, drawHint)),
        h(
          'div',
          { class: 'grid-2' },
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'Its guess')), verdict, h('div', { style: { height: '8px' } }), bars.el),
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'Reservoir fingerprint'), h('span', { class: 'sub' }, `${HEAT_ROWS} neurons × ${RESAMPLE} steps`)), heatCanvas, h('p', { class: 'hint', style: { margin: '6px 0 0' } }, 'The readout classifies this pattern of activity. It never looks at the picture itself.')),
        ),
        explain,
      ),
      panel,
    ),
    readyMark,
  );

  const dc = new Canvas2D(drawCanvas, { aspect: 4 / 3, minHeight: 240, maxHeight: Math.max(300, Math.round(window.innerHeight * 0.62)), onResize: () => redraw() });
  const hc = new Canvas2D(heatCanvas, { aspect: 1.25, minHeight: 160, maxHeight: 300, onResize: () => drawFingerprint() });
  onThemeChange(() => {
    redraw();
    drawFingerprint();
    renderClassList();
  });

  // ---------- Classes & training ----------

  function activeClasses() {
    const names = BUILTIN.filter((n) => !S.removed.has(n));
    for (const n of Object.keys(S.user)) if (!BUILTIN.includes(n) && S.user[n].length && !names.includes(n)) names.push(n);
    return names;
  }

  function persist() {
    storage.set(STORE_KEY, { user: S.user, removed: [...S.removed] });
  }

  function buildClassifier() {
    S.clf = new SequenceClassifier({ inputs: INPUTS, size: sz.value, spectralRadius: sr.value, leak: lk.value, inputScaling: 1, lambda: 1e-3, seed: 7 });
    S.synth = null;
    S.test = null;
  }

  async function rebuild() {
    buildClassifier();
    await retrain({ features: true });
  }

  async function retrain({ features = true } = {}) {
    if (S.busy) {
      S.again = true;
      return;
    }
    S.busy = true;
    const clf = S.clf;
    const t0 = performance.now();
    status.textContent = 'Training…';
    // Synthetic examples for all built-ins (cached; features computed once per reservoir).
    if (!S.synth || features) {
      const all = makeDataset(BUILTIN, SYNTH_PER_CLASS, 1);
      await runChunked(all.length, (i) => (all[i].features = clf.features(all[i].seq)), { onProgress: (p) => progress.set(p * 0.6) });
      S.synth = all;
      S.test = makeDataset(BUILTIN, TEST_PER_CLASS, 999, { prefixes: false });
    }
    const classes = activeClasses();
    const examples = S.synth.filter((e) => classes.includes(e.label));
    const rng = new Rng(3);
    let userCount = 0;
    for (const name of classes) {
      const list = S.user[name] || [];
      userCount += list.length;
      const copies = BUILTIN.includes(name) ? 6 : 14;
      for (const ex of expandExamples(list, name, copies, rng)) {
        ex.features = clf.features(ex.seq);
        examples.push(ex);
      }
    }
    clf.train(examples, classes);
    const ms = performance.now() - t0;
    progress.set(0.7);
    stTime.set(ms < 1000 ? `${Math.round(ms)}` : (ms / 1000).toFixed(2), ms < 1000 ? 'ms' : 's');
    stN.set(examples.length.toLocaleString(), userCount ? `(${userCount} yours)` : '');
    renderClassList();

    // Accuracy on fresh synthetic drawings of the built-in shapes.
    const test = S.test.filter((e) => classes.includes(e.label));
    let ok = 0;
    await runChunked(test.length, (i) => {
      if (clf.predict(test[i].seq).label === test[i].label) ok++;
    }, { onProgress: (p) => progress.set(0.7 + 0.3 * p) });
    if (test.length) stAcc.set(`${((ok / test.length) * 100).toFixed(1)}%`, `of ${test.length}`);
    else stAcc.set('–');
    status.textContent = `Retrained the readout on ${examples.length.toLocaleString()} examples in ${Math.round(ms)} ms.`;
    readyMark.dataset.ready = '1';
    S.busy = false;
    if (S.again) {
      S.again = false;
      retrain({ features: false });
    }
  }

  function renderClassList() {
    const c = colors();
    const classes = activeClasses();
    classList.replaceChildren(
      ...classes.map((name) => {
        const thumb = h('canvas', { width: 80, height: 80 });
        const strokes = S.user[name]?.length && !BUILTIN.includes(name) ? S.user[name][0] : TEMPLATES[name] ? TEMPLATES[name](new Rng(4)) : null;
        if (strokes) drawThumb(thumb, strokes, c);
        const mine = S.user[name]?.length || 0;
        const meta = BUILTIN.includes(name) ? (mine ? `built-in + ${mine} of yours` : 'built-in') : `${mine} of your examples`;
        const add = button('+', () => startTeaching(name), 'small');
        add.setAttribute('aria-label', `Add examples of ${pretty(name)}`);
        add.title = 'Add examples';
        const del = button('×', () => removeClass(name), 'small danger');
        del.setAttribute('aria-label', `Remove ${pretty(name)}`);
        del.title = 'Remove';
        return h('div', { class: `class-item${S.teaching?.name === name ? ' active' : ''}` }, thumb, h('div', {}, h('div', {}, pretty(name)), h('div', { class: 'meta' }, meta)), h('div', { class: 'btn-row', style: { gap: '4px', flexWrap: 'nowrap' } }, add, del));
      }),
    );
  }

  function drawThumb(cv, strokes, c) {
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, 80, 80);
    const all = strokes.flat();
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const [x, y] of all) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    const k = 56 / Math.max(maxX - minX, maxY - minY, 1e-9);
    const ox = 40 - ((minX + maxX) / 2) * k;
    const oy = 40 - ((minY + maxY) / 2) * k;
    ctx.strokeStyle = c.text;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const st of strokes) {
      ctx.beginPath();
      st.forEach(([x, y], i) => (i ? ctx.lineTo(x * k + ox, y * k + oy) : ctx.moveTo(x * k + ox, y * k + oy)));
      ctx.stroke();
    }
    // Start dot so the drawing direction is visible.
    const [sx, sy] = strokes[0][0];
    ctx.fillStyle = c.s2;
    ctx.beginPath();
    ctx.arc(sx * k + ox, sy * k + oy, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  function removeClass(name) {
    if (activeClasses().length <= 2) {
      status.textContent = 'Keep at least two gestures, or there is nothing to choose between.';
      return;
    }
    if (BUILTIN.includes(name)) S.removed.add(name);
    delete S.user[name];
    persist();
    retrain({ features: false });
  }

  // ---------- Teaching ----------

  function startTeaching(rawName) {
    const name = String(rawName || '').trim().toLowerCase().slice(0, 24);
    if (!name) {
      status.textContent = 'Type a name for your gesture first.';
      nameInput.focus();
      return;
    }
    if (BUILTIN.includes(name)) S.removed.delete(name);
    S.teaching = { name, examples: [] };
    nameInput.value = '';
    clearDrawing();
    updateBanner();
    renderClassList();
  }

  function updateBanner() {
    const t = S.teaching;
    banner.hidden = !t;
    teachRow.hidden = !t;
    if (!t) return;
    const n = t.examples.length;
    banner.textContent = n < MIN_EXAMPLES ? `Draw “${pretty(t.name)}” on the canvas: example ${n + 1} of at least ${MIN_EXAMPLES}.` : `Got ${n} examples of “${pretty(t.name)}”. Draw more for better accuracy, or press Done.`;
    doneBtn.disabled = n < MIN_EXAMPLES && !(S.user[t.name]?.length >= MIN_EXAMPLES);
    drawHint.textContent = `Teaching mode: draw “${pretty(t.name)}”`;
  }

  function finishTeaching() {
    const t = S.teaching;
    if (!t) return;
    S.user[t.name] = [...(S.user[t.name] || []), ...t.examples].slice(-20);
    S.teaching = null;
    persist();
    updateBanner();
    drawHint.textContent = 'Draw a shape here, or teach it your own.';
    retrain({ features: false });
  }

  function cancelTeaching() {
    S.teaching = null;
    updateBanner();
    drawHint.textContent = 'Draw a shape here, or teach it your own.';
    renderClassList();
  }

  // ---------- Drawing ----------

  function pointFrom(e) {
    const r = drawCanvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  drawCanvas.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    drawCanvas.setPointerCapture(e.pointerId);
    clearTimeout(S.finishTimer);
    if (S.finished) clearDrawing();
    S.drawing = true;
    S.strokes.push([pointFrom(e)]);
    drawHint.hidden = true;
    redraw();
  });

  drawCanvas.addEventListener('pointermove', (e) => {
    if (!S.drawing) return;
    const st = S.strokes[S.strokes.length - 1];
    const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of events.length ? events : [e]) {
      const p = pointFrom(ev);
      const q = st[st.length - 1];
      if (Math.hypot(p[0] - q[0], p[1] - q[1]) >= 1.5) st.push(p);
    }
    redraw();
    const now = performance.now();
    if (!S.teaching && now - S.lastGuess > GUESS_EVERY_MS) {
      S.lastGuess = now;
      guess(false);
    }
  });

  const endStroke = () => {
    if (!S.drawing) return;
    S.drawing = false;
    clearTimeout(S.finishTimer);
    S.finishTimer = setTimeout(finishGesture, FINISH_MS);
  };
  drawCanvas.addEventListener('pointerup', endStroke);
  drawCanvas.addEventListener('pointercancel', endStroke);

  function inkLength() {
    let L = 0;
    for (const st of S.strokes) for (let i = 1; i < st.length; i++) L += Math.hypot(st[i][0] - st[i - 1][0], st[i][1] - st[i - 1][1]);
    return L;
  }

  function finishGesture() {
    S.finished = true;
    if (inkLength() < 12) {
      clearDrawing();
      return;
    }
    if (S.teaching) {
      S.teaching.examples.push(compact(S.strokes));
      updateBanner();
      flashInk();
      return;
    }
    guess(true);
  }

  function guess(final) {
    if (!S.clf?.trained || S.strokes.length === 0) return;
    if (inkLength() < 20) return;
    const seq = preprocess(S.strokes);
    if (!seq) return;
    const record = [];
    const p = S.clf.predict(seq, record);
    S.fingerprint = record;
    S.lastSeq = seq;
    bars.update(S.clf.classes.map(pretty), p.probs);
    const name = pretty(p.label);
    if (final) {
      verdict.replaceChildren(p.confidence > 0.45 ? name : h('span', {}, name, ' ', h('small', {}, '(not sure)')));
    } else {
      verdict.replaceChildren(h('span', {}, name, ' ', h('small', {}, '…still drawing')));
    }
    drawFingerprint();
    if (final) redraw();
  }

  function clearDrawing() {
    clearTimeout(S.finishTimer);
    S.strokes = [];
    S.finished = false;
    S.drawing = false;
    S.lastSeq = null;
    drawHint.hidden = false;
    redraw();
  }

  let flashUntil = 0;
  function flashInk() {
    flashUntil = performance.now() + 350;
    redraw();
    setTimeout(() => {
      clearDrawing();
    }, 350);
  }

  function redraw() {
    const c = colors();
    const { ctx } = dc;
    dc.clear();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 5;
    ctx.strokeStyle = performance.now() < flashUntil ? c.good : c.text;
    for (const st of S.strokes) {
      ctx.beginPath();
      st.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (st.length === 1) ctx.lineTo(st[0][0] + 0.1, st[0][1]);
      ctx.stroke();
    }
    // After recognition, show the resampled points the reservoir actually saw.
    if (S.finished && S.lastSeq && S.strokes.length) {
      const all = S.strokes.flat();
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      for (const [x, y] of all) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
      const k = Math.max(maxX - minX, maxY - minY) / 2;
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      S.lastSeq.forEach((u, i) => {
        ctx.fillStyle = c.s2;
        ctx.globalAlpha = 0.35 + 0.65 * (i / (S.lastSeq.length - 1));
        ctx.beginPath();
        ctx.arc(cx + u[0] * k, cy + u[1] * k, 4, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    }
  }

  function drawFingerprint() {
    const c = colors();
    const { ctx, width: W, height: Hh } = hc;
    hc.clear();
    const rec = S.fingerprint;
    if (!rec || !rec.length) {
      ctx.fillStyle = c.muted;
      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Appears when you draw', W / 2, Hh / 2);
      return;
    }
    const N = rec[0].length;
    const stride = Math.max(1, Math.floor(N / HEAT_ROWS));
    drawHeatmap(ctx, rec.length, HEAT_ROWS, (col, r) => rec[col][(r * stride) % N], 0, 0, W, Hh - 18, c.lut);
    ctx.fillStyle = c.muted;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('start of stroke', 2, Hh - 4);
    ctx.textAlign = 'right';
    ctx.fillText('end →', W - 2, Hh - 4);
  }

  buildClassifier();
  renderClassList();
  redraw();
  drawFingerprint();
  let started = false;

  return {
    show() {
      if (!started) {
        started = true;
        retrain({ features: true });
      }
    },
    hide() {
      clearTimeout(S.finishTimer);
    },
  };
}
