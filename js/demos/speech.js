// Demo 3 — spoken word recognition from a handful of your own recordings.

import { MelFrontend, utteranceInputs, stretch } from '../audio/features.js';
import { Vad } from '../audio/vad.js';
import { SequenceClassifier } from '../core/sequence-classifier.js';
import { Rng } from '../core/rng.js';
import { h, slider, toggle, button, stat, probBars, storage } from '../ui/dom.js';
import { Canvas2D, colors, onThemeChange, drawHeatmap } from '../ui/canvas.js';

const STORE_KEY = 'rc-speech-v1';
const TARGET = 5;
const SPEC_FRAMES = 300; // 3 s of 10 ms frames
const BANDS = 24;
const HEAT_ROWS = 48;
const GRID = 7;
const COPIES = 8;
const DEFAULT_WORDS = ['up', 'down', 'left', 'right'];

export function mount(root) {
  const saved = storage.get(STORE_KEY, null);
  const S = {
    words: saved?.words?.length ? saved.words : DEFAULT_WORDS.map((name) => ({ name, examples: [] })),
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
  const startBtn = button('Start microphone', () => startMic(), 'primary');
  const stopBtn = button('Stop', () => stopMic());
  stopBtn.hidden = true;
  const micNote = h('p', { class: 'note', hidden: true });
  const meterLevel = h('div', { class: 'level' });
  const meterThresh = h('div', { class: 'thresh' });
  const meter = h('div', { class: 'meter', role: 'meter', 'aria-label': 'Microphone level' }, meterLevel, meterThresh);
  const specCanvas = h('canvas', { class: 'plot', 'aria-label': 'Live mel spectrogram of the microphone' });
  const heatCanvas = h('canvas', { class: 'plot', 'aria-label': 'Reservoir activity for the last word' });
  const verdict = h('div', { class: 'verdict', 'aria-live': 'polite' }, h('small', {}, 'Record a few examples of each word first.'));
  const bars = probBars();
  const wordList = h('div', { class: 'class-list', style: { maxHeight: 'none' } });
  const banner = h('div', { class: 'mode-banner', hidden: true });
  const status = h('p', { class: 'hint', 'aria-live': 'polite' });
  const gridEl = h('div', { class: 'token-grid', 'aria-label': 'Voice-controlled grid' });
  const goalsEl = h('span', { class: 'sub' }, 'Steer the orange square onto the green cell');
  for (let i = 0; i < GRID * GRID; i++) gridEl.append(h('div'));

  const stAcc = stat('Self-check accuracy');
  const stTime = stat('Training time');
  const stN = stat('Your recordings');

  const sens = slider({
    label: 'Sensitivity (dB above background)',
    min: 6,
    max: 24,
    step: 1,
    value: 12,
    format: (v) => `${v} dB`,
    onInput: (v) => {
      if (S.audio) S.audio.vad.sensitivityDb = v;
    },
  });
  const sz = slider({ label: 'Neurons', min: 100, max: 600, step: 50, value: 300, onChange: () => scheduleTrain(0, true) });
  const lk = slider({ label: 'Leak rate', min: 0.05, max: 1, step: 0.05, value: 0.25, format: (v) => v.toFixed(2), onChange: () => scheduleTrain(0, true) });
  const remember = toggle({
    label: 'Remember my recordings on this device',
    hint: 'Stores sound features (not audio) in this browser only',
    checked: S.remember,
    onChange: (v) => {
      S.remember = v;
      persist();
    },
  });
  const forgetBtn = button('Forget all recordings', () => {
    for (const w of S.words) w.examples = [];
    storage.remove(STORE_KEY);
    S.clf = null;
    renderWords();
    verdict.replaceChildren(h('small', {}, 'Record a few examples of each word first.'));
    bars.clear();
    stAcc.set('–');
    stN.set('0');
  }, 'small danger');
  const addWordBtn = button('+ Add word', () => {
    let n = S.words.length + 1;
    while (S.words.some((w) => w.name === `word ${n}`)) n++;
    S.words.push({ name: `word ${n}`, examples: [] });
    renderWords();
  }, 'small');
  const stopRecBtn = button('Stop recording', () => {
    S.recordingFor = null;
    updateBanner();
    renderWords();
  }, 'small');

  const panel = h(
    'div',
    { class: 'stack' },
    h(
      'div',
      { class: 'card panel' },
      h('h2', {}, '1 · Microphone'),
      h('div', { class: 'btn-row' }, startBtn, stopBtn),
      micNote,
      meter,
      sens.el,
      h('h2', {}, `2 · Teach it your words (${TARGET} each)`),
      h('p', { class: 'hint' }, 'Press Record, then say the word several times with short pauses in between.'),
      banner,
      wordList,
      h('div', { class: 'btn-row' }, addWordBtn, stopRecBtn),
      status,
      h('h2', {}, '3 · Talk to it'),
      h('p', { class: 'hint', style: { margin: '-6px 0 0' } }, 'Once every word has a few recordings, just say a word.'),
    ),
    h('div', { class: 'card panel' }, h('h2', {}, 'Reservoir'), sz.el, lk.el, remember.el, forgetBtn),
  );

  const explain = h(
    'details',
    { class: 'explain' },
    h('summary', {}, 'How does this work, and what happens to my voice?'),
    h('p', {}, 'Every 10 ms the microphone signal is turned into 24 numbers that describe how much energy is in each frequency band, like the spectrogram above. A voice-activity detector cuts out each word, and the frames are fed one by one into a random reservoir of a few hundred neurons. The readout looks at the reservoir’s state at five moments during the word plus its average, and is trained with one ridge regression on your recordings. Each recording is also time-stretched into a few slightly faster and slower versions.'),
    h('p', {}, 'Commercial keyword spotters are deep CNNs trained on tens of thousands of clips from thousands of speakers. This model has only heard you, about five times per word, so it works best in the same room, with the same mic and voice. A different person will probably confuse it. That is the trade-off: almost no data, almost no training time.'),
    h('p', {}, h('b', {}, 'Privacy: '), 'the audio is processed inside this page and never leaves your device. No audio is stored. If you tick “remember”, only the 24-number summaries are saved, in this browser’s local storage. The microphone turns off when you leave this tab.'),
  );

  root.append(
    h(
      'div',
      { class: 'view-head' },
      h('h1', {}, 'Spoken word recognition'),
      h('p', {}, 'Keyword spotting usually takes a CNN trained on more than 100,000 clips. This uses a random reservoir trained on about five of your own recordings per word, inside your browser.'),
    ),
    h(
      'div',
      { class: 'demo controls-first' },
      h(
        'div',
        { class: 'stack' },
        h('div', { class: 'stats' }, stAcc.el, stTime.el, stN.el),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'What the microphone hears'), h('span', { class: 'sub' }, 'last 3 seconds · low pitch at the bottom · orange bar = word detected')), specCanvas),
        h(
          'div',
          { class: 'grid-2' },
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'It heard')), verdict, h('div', { style: { height: '8px' } }), bars.el),
          h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'Voice control'), goalsEl), h('div', { style: { display: 'flex', justifyContent: 'center' } }, gridEl), h('p', { class: 'hint', style: { margin: '8px 0 0', textAlign: 'center' } }, 'Say “up”, “down”, “left” or “right”.')),
        ),
        h('div', { class: 'card' }, h('div', { class: 'card-title' }, h('span', {}, 'Reservoir response to the last word'), h('span', { class: 'sub' }, `${HEAT_ROWS} neurons × time`)), heatCanvas),
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
    if (!ok) status.textContent = 'Could not save to this browser’s storage (it may be full or disabled).';
  }

  function renderWords() {
    wordList.replaceChildren(
      ...S.words.map((w, i) => {
        const input = h('input', { type: 'text', value: w.name, maxlength: 20, 'aria-label': `Word ${i + 1}` });
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
        const dots = h('span', { class: 'dots', 'aria-label': `${w.examples.length} recordings` }, Array.from({ length: TARGET }, (_, k) => h('i', { class: k < w.examples.length ? 'on' : '' })));
        const rec = button(S.recordingFor === i ? 'Listening…' : w.examples.length ? 'Record more' : 'Record', () => armRecording(i), `small${S.recordingFor === i ? ' recording' : ''}`);
        const del = button('×', () => {
          if (S.words.length <= 2) return;
          S.words.splice(i, 1);
          if (S.recordingFor === i) S.recordingFor = null;
          persist();
          renderWords();
          scheduleTrain(0);
        }, 'small danger');
        del.setAttribute('aria-label', `Remove ${w.name}`);
        return h('div', { class: 'word-row' }, h('div', {}, input, h('div', { style: { marginTop: '4px' } }, dots, w.examples.length > TARGET ? h('small', { class: 'meta' }, ` +${w.examples.length - TARGET}`) : null)), rec, del);
      }),
    );
    const total = S.words.reduce((a, w) => a + w.examples.length, 0);
    stN.set(String(total), `of ${S.words.length * TARGET} suggested`);
  }

  function armRecording(i) {
    if (!S.audio) {
      status.textContent = 'Start the microphone first.';
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
    banner.textContent = `Say “${w.name}” (${Math.min(w.examples.length + 1, TARGET)} of ${TARGET})…`;
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
    stTime.set(`${Math.round(ms)}`, 'ms');

    // Self-check: leave each recording (and its stretched copies) out,
    // retrain without it, and see if it's still recognised.
    let ok = 0;
    let n = 0;
    if (words.filter((w) => w.examples.length >= 2).length >= 2) {
      ({ ok, n } = clf.leaveOneOut(examples, names, (ex) => ex.src, (ex) => ex.original));
    }
    if (n) stAcc.set(`${Math.round((ok / n) * 100)}%`, `of ${n}, each left out`);
    status.textContent = `Readout trained on ${examples.length} examples (${words.reduce((a, w) => a + w.examples.length, 0)} recordings plus stretched copies) in ${Math.round(ms)} ms.`;
  }

  // ---------- Audio ----------

  async function startMic() {
    micNote.hidden = true;
    if (!window.isSecureContext) {
      showNote('The microphone only works on a secure page: https:// or http://localhost. Open the hosted version, or run ./run.sh and use the localhost link.');
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof AudioWorkletNode === 'undefined') {
      showNote('This browser lacks the audio features this demo needs (getUserMedia and AudioWorklet). Try a current Firefox, Chrome, Edge or Safari.');
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
      status.textContent = `Microphone on (${Math.round(ctx.sampleRate / 1000)} kHz). Speak normally; the background level adapts automatically.`;
      renderLoop();
    } catch (err) {
      console.warn(err);
      showNote(err?.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow it in the address bar and try again.' : `Could not start the microphone: ${err?.message || err}`);
    } finally {
      startBtn.disabled = false;
    }
  }

  function stopMic() {
    const a = S.audio;
    if (!a) return;
    a.node.port.onmessage = null;
    for (const t of a.stream.getTracks()) t.stop();
    a.ctx.close().catch(() => {});
    S.audio = null;
    S.recordingFor = null;
    startBtn.hidden = false;
    stopBtn.hidden = true;
    updateBanner();
    renderWords();
    status.textContent = 'Microphone off.';
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
      flashVerdict(`Got it: “${w.name}” #${w.examples.length}`);
      return;
    }
    if (!S.clf) {
      flashVerdict(ready() ? 'Training…' : 'Record some examples first');
      return;
    }
    const record = [];
    const p = S.clf.predict(utteranceInputs(frames, energies), record);
    S.lastStates = record;
    drawHeat();
    bars.update(S.clf.classes, p.probs);
    if (p.confidence < 0.55) {
      verdict.replaceChildren(h('span', {}, p.label, ' ', h('small', {}, '(not sure)')));
      return;
    }
    verdict.replaceChildren(p.label);
    act(p.label);
  }

  function flashVerdict(text) {
    verdict.replaceChildren(h('small', {}, text));
  }

  function act(word) {
    const moves = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
    const m = moves[word];
    if (!m) return;
    S.token = [Math.max(0, Math.min(GRID - 1, S.token[0] + m[0])), Math.max(0, Math.min(GRID - 1, S.token[1] + m[1]))];
    if (S.token[0] === S.goal[0] && S.token[1] === S.goal[1]) {
      S.goals++;
      goalsEl.textContent = `Goals reached: ${S.goals}`;
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
      ctx.fillText('Say a word once the model is trained', W / 2, H / 2);
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
