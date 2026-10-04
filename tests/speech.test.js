import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MelFrontend, utteranceInputs, stretch, FFT } from '../js/audio/features.js';
import { Vad } from '../js/audio/vad.js';
import { SequenceClassifier } from '../js/core/sequence-classifier.js';
import { Rng } from '../js/core/rng.js';
import { synthWord, SYNTH_WORDS } from './helpers/synth-speech.js';

const SR = 16000;

test('FFT finds the frequency of a pure tone', () => {
  const n = 512;
  const fft = new FFT(n);
  const x = Float64Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * 40 * i) / n));
  const p = fft.power(x, new Float64Array(n / 2 + 1));
  let best = 0;
  for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
  assert.equal(best, 40);
});

test('a 1 kHz tone lights up the mel band centred near 1 kHz', () => {
  const fe = new MelFrontend(SR);
  const tone = Float32Array.from({ length: SR / 2 }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 1000 * i) / SR));
  let last = null;
  fe.push(tone, (mel) => (last = mel));
  let best = 0;
  for (let b = 1; b < last.length; b++) if (last[b] > last[best]) best = b;
  const centre = fe.centersHz[best];
  assert.ok(Math.abs(centre - 1000) < 150, `peak band centred at ${centre.toFixed(0)} Hz`);
  assert.equal(fe.hop, 160);
});

test('voice activity detector finds separate bursts in background noise', () => {
  const rng = new Rng(3);
  const sig = new Float32Array(SR * 3);
  for (let i = 0; i < sig.length; i++) sig[i] = rng.gauss() * 0.002;
  for (const [a, b] of [[0.5, 0.9], [1.6, 2.1]]) for (let i = a * SR; i < b * SR; i++) sig[i] += 0.3 * Math.sin((2 * Math.PI * 300 * i) / SR);
  const fe = new MelFrontend(SR);
  const found = [];
  const vad = new Vad({ onUtterance: (frames) => found.push(frames.length) });
  fe.push(sig, (mel, db) => vad.push(mel, db));
  assert.equal(found.length, 2, `found ${found.length} utterances`);
  for (const len of found) assert.ok(len > 35 && len < 75, `utterance of ${len} frames`);
});

function utterances(audio) {
  const fe = new MelFrontend(SR);
  const found = [];
  const vad = new Vad({ onUtterance: (f, e) => found.push({ frames: f, energies: e }) });
  fe.push(audio, (mel, db) => vad.push(mel, db));
  return found;
}

test('learns synthetic spoken words from 5 examples each', () => {
  const rng = new Rng(1);
  const train = [];
  const testSet = [];
  for (const w of SYNTH_WORDS) {
    for (let i = 0; i < 17; i++) {
      const u = utterances(synthWord(w, rng));
      assert.equal(u.length, 1, `VAD found ${u.length} utterances in one word`);
      (i < 5 ? train : testSet).push({ ...u[0], label: w });
    }
  }
  const clf = new SequenceClassifier({ inputs: 25, size: 300, spectralRadius: 0.9, leak: 0.25, inputScaling: 1, lambda: 1e-3, snapshots: [0.2, 0.4, 0.6, 0.8, 1], seed: 21 });
  const aug = new Rng(2);
  const ex = [];
  for (const t of train) {
    ex.push({ seq: utteranceInputs(t.frames, t.energies), label: t.label, src: t, original: true });
    for (let k = 0; k < 8; k++) {
      const s = stretch(t.frames, t.energies, aug.uniform(0.8, 1.25), aug);
      ex.push({ seq: utteranceInputs(s.frames, s.energies), label: t.label, src: t });
    }
  }
  for (const e of ex) e.features = clf.features(e.seq);
  clf.train(ex, SYNTH_WORDS);
  let ok = 0;
  for (const t of testSet) if (clf.predict(utteranceInputs(t.frames, t.energies)).label === t.label) ok++;
  assert.ok(ok / testSet.length >= 0.9, `accuracy ${((ok / testSet.length) * 100).toFixed(1)}%`);

  const loo = clf.leaveOneOut(ex, SYNTH_WORDS, (e) => e.src, (e) => e.original);
  assert.equal(loo.n, train.length);
  assert.ok(loo.ok / loo.n >= 0.85, `leave-one-out ${loo.ok}/${loo.n}`);
});

test('leave-one-out matches retraining from scratch', () => {
  const rng = new Rng(4);
  const mk = (label, shift) => ({ seq: Array.from({ length: 12 }, (_, t) => Float64Array.of(Math.sin(t * 0.5 + shift) + rng.gauss() * 0.3)), label });
  const ex = [];
  for (let i = 0; i < 6; i++) ex.push({ ...mk('a', 0), id: i, original: true }, { ...mk('b', 2), id: 100 + i, original: true });
  const clf = new SequenceClassifier({ inputs: 1, size: 40, seed: 3, lambda: 1e-2 });
  for (const e of ex) e.features = clf.features(e.seq);
  const fast = clf.leaveOneOut(ex, ['a', 'b'], (e) => e.id, () => true);
  let ok = 0;
  for (const held of ex) {
    clf.train(ex.filter((e) => e !== held), ['a', 'b']);
    if (clf.predictFeatures(held.features).label === held.label) ok++;
  }
  assert.equal(fast.ok, ok);
  assert.equal(fast.n, ex.length);
});
