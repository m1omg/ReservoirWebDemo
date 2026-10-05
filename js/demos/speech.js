// Demo 3 — spoken word recognition from a handful of your own recordings.

import { MelFrontend, utteranceInputs, stretch } from '../audio/features.js';
import { Vad } from '../audio/vad.js';
import { SequenceClassifier } from '../core/sequence-classifier.js';
import { Rng } from '../core/rng.js';
import { h, slider, toggle, button, stat, probBars, storage, rich, paras } from '../ui/dom.js';
import { LANGS, t, fmt, fmtInt, fmtPct } from '../i18n.js';
import { levelSwitch, levels } from '../ui/explain.js';
import { Canvas2D, colors, onThemeChange, drawHeatmap } from '../ui/canvas.js';

const STORE_KEY = 'rc-speech-v1';
const TARGET = 5;
const SPEC_FRAMES = 300; // 3 s of 10 ms frames
const BANDS = 24;
const HEAT_ROWS = 48;
const GRID = 7;
const COPIES = 8;
const DEFAULTS = { sensitivity: 12, size: 300, leak: 0.25 };
const DIRS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
// Words that steer the grid, in every language (so "up" and "hore" both work).
const MOVES = {};
for (const table of Object.values(LANGS)) for (const [dir, word] of Object.entries(table.speech?.moves || {})) MOVES[word.toLowerCase()] = DIRS[dir];
for (const dir of Object.keys(DIRS)) MOVES[dir] = DIRS[dir];

export function mount(root) {
  const saved = storage.get(STORE_KEY, null);
  const S = {
    words: saved?.words?.length ? saved.words : t('speech.defaultWords').map((name) => ({ name, examples: [] })),
    remember: !!saved,
    clf: null,
    audio: null,
    recordingFor: null,
    spec: new Float32Array(SPEC_FRAMES * BANDS),
    speakFlags: new Uint8Array(SPEC_FRAMES),
    specHead: 0,
    level: -90,
    dirty: true,
    lastStates: null,
    token: [3, 3],
    goal: [5, 1],
    goals: 0,
    rng: new Rng(Date.now() % 100000),
    trainTimer: 0,
  };

  // ---------- DOM ----------
  const startBtn = button(t('speech.start'), () => startMic(), 'primary');
  const stopBtn = button(t('speech.stop'), () => stopMic());
  stopBtn.hidden = true;
  const micNote = h('p', { class: 'note', hidden: true });
  const meterLevel = h('div', { class: 'level' });
  const meterThresh = h('div', { class: 'thresh' });
  const meter = h('div', { class: 'meter', role: 'meter', 'aria-label': t('speech.aria.meter') }, meterLevel, meterThresh);
  const specCanvas = h('canvas', { class: 'plot', 'aria-label': t('speech.aria.spec') });
  const heatCanvas = h('canvas', { class: 'plot', 'aria-label': t('speech.aria.heat') });
  const verdict = h('div', { class: 'verdict', 'aria-live': 'polite' }, h('small', {}, t('speech.recordFirst')));
  const bars = probBars();
  const wordList = h('div', { class: 'class-list', style: { maxHeight: 'none' } });
  const banner = h('div', { class: 'mode-banner', hidden: true });
  const status = h('p', { class: 'hint', 'aria-live': 'polite' });
  const gridEl = h('div', { class: 'token-grid', 'aria-label': t('speech.aria.grid') });
  const goalsEl = h('span', { class: 'sub' }, t('speech.voiceGoal'));
  for (let i = 0; i < GRID * GRID; i++) gridEl.append(h('div'));

  const stAcc = stat(t('speech.stats.selfCheck'));
  const stTime = stat(t('common.trainingTime'));
  const stN = stat(t('speech.stats.recordings'));

  const sens = slider({
    label: t('speech.sensitivity'),
    min: 6,
    max: 24,
    step: 1,
    value: DEFAULTS.sensitivity,
    format: (v) => `${fmtInt(v)} dB`,
    onInput: (v) => {
      if (S.audio) S.audio.vad.sensitivityDb = v;
    },
  });
  const sz = slider({ label: t('common.neurons'), min: 100, max: 600, step: 50, value: DEFAULTS.size, format: fmtInt, onChange: () => scheduleTrain(0, true) });
  const lk = slider({ label: t('common.leak'), min: 0.05, max: 1, step: 0.05, value: DEFAULTS.leak, format: (v) => fmt(v, 2), onChange: () => scheduleTrain(0, true) });
  const remember = toggle({
    label: t('speech.remember'),
    hint: t('speech.rememberHint'),
    checked: S.remember,
    onChange: (v) => {
      S.remember = v;
      persist();
    },
  });
  const forgetBtn = button(t('speech.forget'), () => {
    for (const w of S.words) w.examples = [];
    storage.remove(STORE_KEY);
    S.clf = null;
    renderWords();
    verdict.replaceChildren(h('small', {}, t('speech.recordFirst')));
    bars.clear();
    stAcc.set('–');
    stN.set('0');
  }, 'small danger');
  const addWordBtn = button(t('speech.addWord'), () => {
    let n = S.words.length + 1;
    while (S.words.some((w) => w.name === t('speech.newWord', { n }))) n++;
    S.words.push({ name: t('speech.newWord', { n }), examples: [] });
    renderWords();
  }, 'small');
  const stopRecBtn = button(t('speech.stopRecording'), () => {
    S.recordingFor = null;
    updateBanner();
    renderWords();
  }, 'small');
  const resetNote = h('p', { class: 'hint', 'aria-live': 'polite' });
  const resetBtn = button(t('common.reset'), () => {
    sens.value = DEFAULTS.sensitivity;
    if (S.audio) S.audio.vad.sensitivityDb = DEFAULTS.sensitivity;
    sz.value = DEFAULTS.size;
    lk.value = DEFAULTS.leak;
    scheduleTrain(0, true);
    resetNote.textContent = t('common.resetDone');
  }, 'small');

  const panel = h(
    'div',
    { class: 'stack' },
    h(
      'div',
      { class: 'card panel' },
      h('h2', {}, t('speech.mic')),
      h('div', { class: 'btn-row' }, startBtn, stopBtn),
      micNote,
      meter,
      sens.el,
      h('h2', {}, t('speech.teachTitle', { n: TARGET })),
      h('p', { class: 'hint' }, t('speech.teachHint')),
      banner,
      wordList,
      h('div', { class: 'btn-row' }, addWordBtn, stopRecBtn),
      status,
      h('h2', {}, t('speech.talkTitle')),
      h('p', { class: 'hint', style: { margin: '-6px 0 0' } }, t('speech.talkHint')),
    ),
    h('div', { class: 'card panel' }, h('h2', {}, t('common.reservoir')), sz.el, lk.el, remember.el, forgetBtn, h('div', { class: 'panel-foot' }, resetBtn, resetNote)),
  );

  const explain = h(
    'details',
    { class: 'explain' },
    h('summary', {}, t('speech.explain.summary')),
    levels(paras(t('speech.explain.simple')), paras(t('speech.explain.detailed'))),
    ...paras(t('speech.explain.privacy')),
  );

  const mv = t('speech.moves');
  root.append(
    h(
      'div',
      { class: 'view-head' },
      h('h1', {}, t('speech.title')),
      levelSwitch(),
      levels([h('p', {}, rich(t('speech.lead.simple'))), h('p', { class: 'try' }, h('b', {}, t('common.tryIt')), ' ', rich(t('speech.try', mv)))], h('p', {}, rich(t('speech.lead.detailed')))),
    ),
    h(
      'div',
      { class: 'demo controls-first' },
      h(
        'div',
        { class: 'stack' },
        h('div', { class: 'stats' }, stAcc.el, stTime.el, stN.el),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('speech.hears')), h('span', { class: 'sub' }, t('speech.hearsSub'))), specCanvas),
        h(
          'div',
          { class: 'grid-2' },
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('speech.heard'))), verdict, h('div', { style: { height: '8px' } }), bars.el),
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('speech.voiceControl')), goalsEl), h('div', { style: { display: 'flex', justifyContent: 'center' } }, gridEl), h('p', { class: 'hint', style: { margin: '8px 0 0', textAlign: 'center' } }, t('speech.sayWords', mv))),
        ),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, t('speech.response')), h('span', { class: 'sub' }, t('speech.responseSub', { rows: HEAT_ROWS }))), heatCanvas),
        explain,
      ),
      panel,
    ),
  );

  const sc = new Canvas2D(specCanvas, { aspect: 3.4, minHeight: 140, maxHeight: 240, onResize: () => (S.dirty = true) });
  const hc = new Canvas2D(heatCanvas, { aspect: 3.4, minHeight: 120, maxHeight: 220, onResize: () => drawHeat() });
  onThemeChange(() => {
    S.dirty = true;
    drawHeat();
  });

  // ---------- Words ----------

  function persist() {
    if (!S.remember) {
      storage.remove(STORE_KEY);
      return;
    }
    const round = (a) => Array.from(a, (v) => Math.round(v * 100) / 100);
    const ok = storage.set(STORE_KEY, {
      words: S.words.map((w) => ({ name: w.name, examples: w.examples.map((e) => ({ frames: e.frames.map(round), energies: round(e.energies) })) })),
    });
    if (!ok) status.textContent = t('speech.status.saveFailed');
  }

  function renderWords() {
    wordList.replaceChildren(
      ...S.words.map((w, i) => {
        const input = h('input', { type: 'text', value: w.name, maxlength: 20, 'aria-label': t('speech.wordAria', { n: i + 1 }) });
        input.addEventListener('change', () => {
          const v = input.value.trim().toLowerCase();
          if (!v || S.words.some((o, j) => j !== i && o.name === v)) {
            input.value = w.name;
            return;
          }
          w.name = v;
          persist();
          scheduleTrain(0);
        });
        const dots = h('span', { class: 'dots', 'aria-label': t('speech.recordingsAria', { n: w.examples.length }) }, Array.from({ length: TARGET }, (_, k) => h('i', { class: k < w.examples.length ? 'on' : '' })));
        const rec = button(S.recordingFor === i ? t('speech.listening') : w.examples.length ? t('speech.recordMore') : t('speech.record'), () => armRecording(i), `small${S.recordingFor === i ? ' recording' : ''}`);
        const del = button('×', () => {
          if (S.words.length <= 2) return;
          S.words.splice(i, 1);
          if (S.recordingFor === i) S.recordingFor = null;
          persist();
          renderWords();
          scheduleTrain(0);
        }, 'small danger');
        del.setAttribute('aria-label', t('speech.removeAria', { name: w.name }));
        return h('div', { class: 'word-row' }, h('div', {}, input, h('div', { style: { marginTop: '4px' } }, dots, w.examples.length > TARGET ? h('small', { class: 'meta' }, ` +${w.examples.length - TARGET}`) : null)), rec, del);
      }),
    );
    const total = S.words.reduce((a, w) => a + w.examples.length, 0);
    stN.set(fmtInt(total), t('speech.stats.suggested', { n: fmtInt(S.words.length * TARGET) }));
  }

  function armRecording(i) {
    if (!S.audio) {
      status.textContent = t('speech.status.startFirst');
      return;
    }
    S.recordingFor = S.recordingFor === i ? null : i;
    updateBanner();
    renderWords();
  }

  function updateBanner() {
    const i = S.recordingFor;
    banner.hidden = i === null;
    if (i === null) return;
    const w = S.words[i];
    banner.textContent = t('speech.banner', { name: w.name, k: Math.min(w.examples.length + 1, TARGET), n: TARGET });
  }

  // ---------- Training ----------

  function scheduleTrain(delay = 300, rebuild = false) {
    if (rebuild) S.clf = null;
    clearTimeout(S.trainTimer);
    S.trainTimer = setTimeout(train, delay);
  }

  function ready() {
    return S.words.filter((w) => w.examples.length >= 1).length >= 2;
  }

  function train() {
    if (!ready()) {
      S.clf = null;
      return;
    }
    const t0 = performance.now();
    const clf = new SequenceClassifier({ inputs: BANDS + 1, size: sz.value, spectralRadius: 0.9, leak: lk.value, inputScaling: 1, lambda: 1e-3, snapshots: [0.2, 0.4, 0.6, 0.8, 1], seed: 21 });
    const words = S.words.filter((w) => w.examples.length);
    const rng = new Rng(9);
    const examples = [];
    for (const w of words) {
      for (const e of w.examples) {
        examples.push({ seq: utteranceInputs(e.frames, e.energies), label: w.name, src: e, original: true });
        for (let k = 0; k < COPIES; k++) {
          const s = stretch(e.frames, e.energies, rng.uniform(0.8, 1.25), rng);
          examples.push({ seq: utteranceInputs(s.frames, s.energies), label: w.name, src: e });
        }
      }
    }
    for (const ex of examples) ex.features = clf.features(ex.seq);
    const names = words.map((w) => w.name);
    clf.train(examples, names);
    const ms = performance.now() - t0;
    S.clf = clf;
    stTime.set(fmtInt(ms), t('common.ms'));

    // Self-check: leave each recording (and its stretched copies) out,
    // retrain without it, and see if it's still recognised.
    let ok = 0;
    let n = 0;
    if (words.filter((w) => w.examples.length >= 2).length >= 2) {
      ({ ok, n } = clf.leaveOneOut(examples, names, (ex) => ex.src, (ex) => ex.original));
    }
    if (n) stAcc.set(fmtPct(ok / n, 0), t('speech.stats.leftOut', { n }));
    status.textContent = t('speech.status.trained', { n: fmtInt(examples.length), r: fmtInt(words.reduce((a, w) => a + w.examples.length, 0)), ms: fmtInt(ms) });
  }

  // ---------- Audio ----------

  async function startMic() {
    micNote.hidden = true;
    if (!window.isSecureContext) {
      showNote(t('speech.errors.insecure'));
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof AudioWorkletNode === 'undefined') {
      showNote(t('speech.errors.unsupported'));
      return;
    }
    startBtn.disabled = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: true, autoGainControl: false } });
      const ctx = new AudioContext({ latencyHint: 'interactive' });
      await ctx.audioWorklet.addModule(new URL('../audio/capture-worklet.js', import.meta.url));
      const source = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, 'rc-capture');
      const mute = ctx.createGain();
      mute.gain.value = 0;
      source.connect(node);
      node.connect(mute);
      mute.connect(ctx.destination);
      const fe = new MelFrontend(ctx.sampleRate, { bands: BANDS });
      const vad = new Vad({ sensitivityDb: sens.value, onUtterance: onUtterance });
      node.port.onmessage = (e) => fe.push(e.data, onFrame);
      await ctx.resume();
      S.audio = { stream, ctx, node, fe, vad };
      startBtn.hidden = true;
      stopBtn.hidden = false;
      status.textContent = t('speech.status.on', { khz: fmtInt(ctx.sampleRate / 1000) });
      renderLoop();
    } catch (err) {
      console.warn(err);
      showNote(err?.name === 'NotAllowedError' ? t('speech.errors.denied') : t('speech.errors.failed', { error: err?.message || err }));
    } finally {
      startBtn.disabled = false;
    }
  }

  function stopMic() {
    const a = S.audio;
    if (!a) return;
    a.node.port.onmessage = null;
    for (const track of a.stream.getTracks()) track.stop();
    a.ctx.close().catch(() => {});
    S.audio = null;
    S.recordingFor = null;
    startBtn.hidden = false;
    stopBtn.hidden = true;
    updateBanner();
    renderWords();
    status.textContent = t('speech.status.off');
  }

  function showNote(text) {
    micNote.textContent = text;
    micNote.hidden = false;
    micNote.classList.add('warn');
  }

  function onFrame(mel, db) {
    const a = S.audio;
    if (!a) return;
    a.vad.push(mel, db);
    const o = S.specHead * BANDS;
    S.spec.set(mel, o);
    S.speakFlags[S.specHead] = a.vad.speaking ? 1 : 0;
    S.specHead = (S.specHead + 1) % SPEC_FRAMES;
    S.level = db;
    S.dirty = true;
  }

  function onUtterance(frames, energies) {
    if (S.recordingFor !== null) {
      const w = S.words[S.recordingFor];
      w.examples.push({ frames, energies });
      if (w.examples.length >= TARGET && w.examples.length % TARGET === 0) S.recordingFor = null;
      updateBanner();
      renderWords();
      persist();
      scheduleTrain(250);
      flashVerdict(t('speech.got', { name: w.name, n: w.examples.length }));
      return;
    }
    if (!S.clf) {
      flashVerdict(ready() ? t('speech.trainingNow') : t('speech.recordSome'));
      return;
    }
    const record = [];
    const p = S.clf.predict(utteranceInputs(frames, energies), record);
    S.lastStates = record;
    drawHeat();
    bars.update(S.clf.classes, p.probs);
    if (p.confidence < 0.55) {
      verdict.replaceChildren(h('span', {}, p.label, ' ', h('small', {}, t('speech.notSure'))));
      return;
    }
    verdict.replaceChildren(p.label);
    act(p.label);
  }

  function flashVerdict(text) {
    verdict.replaceChildren(h('small', {}, text));
  }

  function act(word) {
    const m = MOVES[String(word).toLowerCase()];
    if (!m) return;
    S.token = [Math.max(0, Math.min(GRID - 1, S.token[0] + m[0])), Math.max(0, Math.min(GRID - 1, S.token[1] + m[1]))];
    if (S.token[0] === S.goal[0] && S.token[1] === S.goal[1]) {
      S.goals++;
      goalsEl.textContent = t('speech.goals', { n: S.goals });
      do S.goal = [S.rng.int(GRID), S.rng.int(GRID)];
      while (S.goal[0] === S.token[0] && S.goal[1] === S.token[1]);
    }
    renderGrid();
  }

  function renderGrid() {
    const cells = gridEl.children;
    for (let i = 0; i < cells.length; i++) {
      const x = i % GRID;
      const y = Math.floor(i / GRID);
      cells[i].className = (x === S.token[0] && y === S.token[1] ? 'on' : '') + (x === S.goal[0] && y === S.goal[1] ? ' goal' : '');
    }
  }

  // Rendering is driven by incoming audio frames (100 per second), so the
  // spectrogram scrolls at the same speed on every display.
  function renderLoop() {
    if (!S.audio) return;
    if (S.dirty) {
      S.dirty = false;
      drawSpec();
      const a = S.audio;
      const toPct = (db) => Math.max(0, Math.min(100, ((db + 90) / 90) * 100));
      meterLevel.style.width = `${toPct(S.level)}%`;
      meterThresh.style.left = `${toPct(a.vad.threshold)}%`;
    }
    requestAnimationFrame(renderLoop);
  }

  function drawSpec() {
    const c = colors();
    const { ctx, width: W, height: H } = sc;
    sc.clear();
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < S.spec.length; i++) {
      const v = S.spec[i];
      if (v === 0) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!(hi > lo)) return;
    lo = Math.max(lo, hi - 14);
    const head = S.specHead;
    const barH = 6;
    drawHeatmap(ctx, SPEC_FRAMES, BANDS, (col, r) => {
      const v = S.spec[((head + col) % SPEC_FRAMES) * BANDS + (BANDS - 1 - r)];
      return ((v - lo) / (hi - lo)) * 2 - 1;
    }, 0, barH + 2, W, H - barH - 2, c.seqLut, { smooth: true });
    ctx.fillStyle = c.s2;
    const cw = W / SPEC_FRAMES;
    for (let col = 0; col < SPEC_FRAMES; col++) if (S.speakFlags[(head + col) % SPEC_FRAMES]) ctx.fillRect(col * cw, 0, cw + 0.5, barH);
  }

  function drawHeat() {
    const c = colors();
    const { ctx, width: W, height: H } = hc;
    hc.clear();
    const rec = S.lastStates;
    if (!rec?.length) {
      ctx.fillStyle = c.muted;
      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(t('speech.sayOnce'), W / 2, H / 2);
      return;
    }
    const N = rec[0].length;
    const stride = Math.max(1, Math.floor(N / HEAT_ROWS));
    drawHeatmap(ctx, rec.length, HEAT_ROWS, (col, r) => rec[col][(r * stride) % N], 0, 0, W, H, c.lut);
  }

  renderWords();
  renderGrid();
  drawHeat();
  if (ready()) scheduleTrain(0);

  return {
    show() {},
    hide() {
      // Privacy: never keep listening in the background.
      stopMic();
    },
  };
}
