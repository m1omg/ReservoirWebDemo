// Demo 2 — gesture recognition with a reservoir, including few-shot teaching.

import { BUILTIN, TEMPLATES, makeDataset, expandExamples, preprocess, compact, RESAMPLE, INPUTS } from '../systems/gestures.js';
import { SequenceClassifier } from '../core/sequence-classifier.js';
import { runChunked } from '../core/loop.js';
import { Rng } from '../core/rng.js';
import { h, slider, button, stat, probBars, storage, progressBar, rich, paras } from '../ui/dom.js';
import { t, tp, fmt, fmtInt, fmtPct } from '../i18n.js';
import { levelSwitch, levels } from '../ui/explain.js';
import { Canvas2D, colors, onThemeChange, drawHeatmap } from '../ui/canvas.js';

const STORE_KEY = 'rc-gestures-v1';
const SYNTH_PER_CLASS = 50;
const TEST_PER_CLASS = 40;
const FINISH_MS = 650;
const GUESS_EVERY_MS = 60;
const MIN_EXAMPLES = 3;
const HEAT_ROWS = 48;
const DEFAULTS = { size: 200, spectralRadius: 0.9, leak: 0.3 };
const pretty = (n) => (BUILTIN.includes(n) ? t(`gestures.names.${n}`) : n);

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
  const drawCanvas = h('canvas', { class: 'draw', 'aria-label': t('gestures.aria.draw') });
  const drawHint = h('div', { class: 'draw-hint' }, t('gestures.hint'));
  const banner = h('div', { class: 'mode-banner', hidden: true });
  const verdict = h('div', { class: 'verdict', 'aria-live': 'polite' }, h('small', {}, t('gestures.drawSomething')));
  const bars = probBars();
  const heatCanvas = h('canvas', { class: 'plot', 'aria-label': t('gestures.aria.heat') });
  const classList = h('div', { class: 'class-list' });
  const nameInput = h('input', { type: 'text', placeholder: t('gestures.namePlaceholder'), maxlength: 24, 'aria-label': t('gestures.nameAria') });
  const progress = progressBar();
  const status = h('p', { class: 'hint', 'aria-live': 'polite' });
  const readyMark = h('span', { hidden: true });

  const stAcc = stat(t('gestures.stats.accuracy'));
  const stTime = stat(t('common.trainingTime'));
  const stN = stat(t('gestures.stats.examples'));

  const teachBtn = button(t('gestures.teach'), () => startTeaching(nameInput.value));
  const doneBtn = button(t('gestures.done'), () => finishTeaching(), 'primary');
  const cancelBtn = button(t('gestures.cancel'), () => cancelTeaching());
  const clearBtn = button(t('gestures.clear'), () => clearDrawing());
  const resetBtn = button(t('gestures.resetGestures'), () => {
    S.user = {};
    S.removed = new Set();
    persist();
    retrain({ features: false });
  }, 'small');

  nameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') startTeaching(nameInput.value);
  });

  const f2 = (v) => fmt(v, 2);
  const sz = slider({ label: t('common.neurons'), min: 50, max: 500, step: 25, value: DEFAULTS.size, format: fmtInt, onChange: () => rebuild() });
  const sr = slider({ label: t('common.spectralRadius'), min: 0.1, max: 1.5, step: 0.05, value: DEFAULTS.spectralRadius, format: f2, onChange: () => rebuild() });
  const lk = slider({ label: t('common.leak'), min: 0.05, max: 1, step: 0.05, value: DEFAULTS.leak, format: f2, onChange: () => rebuild() });
  const resetNote = h('p', { class: 'hint', 'aria-live': 'polite' });
  const resetSettingsBtn = button(t('common.reset'), () => {
    sz.value = DEFAULTS.size;
    sr.value = DEFAULTS.spectralRadius;
    lk.value = DEFAULTS.leak;
    clearDrawing();
    rebuild();
    resetNote.textContent = t('common.resetDone');
  }, 'small');

  const teachRow = h('div', { class: 'btn-row' }, doneBtn, cancelBtn);
  teachRow.hidden = true;

  const panel = h(
    'div',
    { class: 'stack' },
    h(
      'div',
      { class: 'card panel' },
      h('h2', {}, t('gestures.known')),
      h('p', { class: 'hint' }, t('gestures.knownHint')),
      classList,
      h('h2', {}, t('gestures.teachTitle')),
      h('div', { class: 'word-row', style: { gridTemplateColumns: 'minmax(0,1fr) auto' } }, nameInput, teachBtn),
      banner,
      teachRow,
      progress.el,
      status,
      resetBtn,
    ),
    h(
      'div',
      { class: 'card panel' },
      h('h2', {}, t('common.reservoir')),
      sz.el,
      sr.el,
      lk.el,
      h('p', { class: 'hint', style: { margin: 0 } }, t('gestures.reservoirNote')),
      h('div', { class: 'panel-foot' }, resetSettingsBtn, resetNote),
    ),
  );

  const explain = h('details', { class: 'explain' }, h('summary', {}, t('gestures.explain.summary')), levels(paras(t('gestures.explain.simple')), paras(t('gestures.explain.detailed', { n: RESAMPLE }))));

  root.append(
    h(
      'div',
      { class: 'view-head' },
      h('h1', {}, t('gestures.title')),
      levelSwitch(),
      levels([h('p', {}, rich(t('gestures.lead.simple'))), h('p', { class: 'try' }, h('b', {}, t('common.tryIt')), ' ', rich(t('gestures.try')))], h('p', {}, rich(t('gestures.lead.detailed')))),
    ),
    h(
      'div',
      { class: 'demo' },
      h(
        'div',
        { class: 'stack' },
        h('div', { class: 'stats' }, stAcc.el, stTime.el, stN.el),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('gestures.drawHere')), h('span', { class: 'btn-row' }, clearBtn)), h('div', { class: 'draw-wrap' }, drawCanvas, drawHint)),
        h(
          'div',
          { class: 'grid-2' },
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('gestures.guess'))), verdict, h('div', { style: { height: '8px' } }), bars.el),
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('gestures.fingerprint')), h('span', { class: 'sub' }, t('gestures.fingerprintSub', { rows: HEAT_ROWS, cols: RESAMPLE }))), heatCanvas, h('p', { class: 'hint', style: { margin: '6px 0 0' } }, t('gestures.fingerprintNote'))),
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
    status.textContent = t('gestures.status.training');
    // Synthetic examples for all built-ins (cached; features computed once per reservoir).
    // Cached features belong to one reservoir; recompute them for a new one
    // (e.g. when the reservoir was rebuilt while a previous run was busy).
    if (!S.synth || features || S.synthFor !== clf) {
      const all = makeDataset(BUILTIN, SYNTH_PER_CLASS, 1);
      await runChunked(all.length, (i) => (all[i].features = clf.features(all[i].seq)), { onProgress: (p) => progress.set(p * 0.6) });
      S.synth = all;
      S.synthFor = clf;
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
    stTime.set(ms < 1000 ? fmtInt(ms) : fmt(ms / 1000, 2), ms < 1000 ? t('common.ms') : t('common.s'));
    stN.set(fmtInt(examples.length), userCount ? t('gestures.stats.yours', { n: fmtInt(userCount) }) : '');
    renderClassList();

    // Accuracy on fresh synthetic drawings of the built-in shapes.
    const test = S.test.filter((e) => classes.includes(e.label));
    let ok = 0;
    await runChunked(test.length, (i) => {
      if (clf.predict(test[i].seq).label === test[i].label) ok++;
    }, { onProgress: (p) => progress.set(0.7 + 0.3 * p) });
    if (test.length) stAcc.set(fmtPct(ok / test.length), t('gestures.stats.of', { n: fmtInt(test.length) }));
    else stAcc.set('–');
    status.textContent = t('gestures.status.retrained', { n: fmtInt(examples.length), ms: fmtInt(ms) });
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
        const meta = BUILTIN.includes(name) ? (mine ? tp('gestures.builtinPlus', mine) : t('gestures.builtin')) : tp('gestures.yourExamples', mine);
        const add = button('+', () => startTeaching(name), 'small');
        add.setAttribute('aria-label', t('gestures.addAria', { name: pretty(name) }));
        add.title = t('gestures.addTitle');
        const del = button('×', () => removeClass(name), 'small danger');
        del.setAttribute('aria-label', t('gestures.removeAria', { name: pretty(name) }));
        del.title = t('gestures.removeTitle');
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
      status.textContent = t('gestures.status.keepTwo');
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
      status.textContent = t('gestures.status.needName');
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
    const tc = S.teaching;
    banner.hidden = !tc;
    teachRow.hidden = !tc;
    if (!tc) return;
    const n = tc.examples.length;
    const name = pretty(tc.name);
    banner.textContent = n < MIN_EXAMPLES ? t('gestures.banner.draw', { name, k: n + 1, min: MIN_EXAMPLES }) : tp('gestures.banner.got', n, { name });
    doneBtn.disabled = n < MIN_EXAMPLES && !(S.user[tc.name]?.length >= MIN_EXAMPLES);
    drawHint.textContent = t('gestures.teachingHint', { name });
  }

  function finishTeaching() {
    const tc = S.teaching;
    if (!tc) return;
    S.user[tc.name] = [...(S.user[tc.name] || []), ...tc.examples].slice(-20);
    S.teaching = null;
    persist();
    updateBanner();
    drawHint.textContent = t('gestures.hintAgain');
    retrain({ features: false });
  }

  function cancelTeaching() {
    S.teaching = null;
    updateBanner();
    drawHint.textContent = t('gestures.hintAgain');
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
      verdict.replaceChildren(p.confidence > 0.45 ? name : h('span', {}, name, ' ', h('small', {}, t('gestures.notSure'))));
    } else {
      verdict.replaceChildren(h('span', {}, name, ' ', h('small', {}, t('gestures.stillDrawing'))));
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
      ctx.fillText(t('gestures.appears'), W / 2, Hh / 2);
      return;
    }
    const N = rec[0].length;
    const stride = Math.max(1, Math.floor(N / HEAT_ROWS));
    drawHeatmap(ctx, rec.length, HEAT_ROWS, (col, r) => rec[col][(r * stride) % N], 0, 0, W, Hh - 18, c.lut);
    ctx.fillStyle = c.muted;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(t('gestures.startStroke'), 2, Hh - 4);
    ctx.textAlign = 'right';
    ctx.fillText(t('gestures.end'), W - 2, Hh - 4);
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
