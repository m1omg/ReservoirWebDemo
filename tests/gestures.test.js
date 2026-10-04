import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BUILTIN, INPUTS, makeDataset, preprocess, expandExamples, synthesize, truncate } from '../js/systems/gestures.js';
import { SequenceClassifier } from '../js/core/sequence-classifier.js';
import { Rng } from '../js/core/rng.js';

const clf = new SequenceClassifier({ inputs: INPUTS, size: 200, spectralRadius: 0.9, leak: 0.3, inputScaling: 1, lambda: 1e-3, seed: 7 });
clf.train(makeDataset(BUILTIN, 50, 1), BUILTIN);

test('preprocess gives a fixed-length, normalised sequence', () => {
  const seq = preprocess([[[10, 10], [110, 10], [110, 60]], [[0, 0], [5, 5]]]);
  assert.equal(seq.length, 40);
  for (const u of seq) {
    assert.equal(u.length, INPUTS);
    assert.ok(Math.abs(u[0]) <= 1.0001 && Math.abs(u[1]) <= 1.0001);
    assert.ok(Math.abs(Math.hypot(u[2], u[3]) - 1) < 1e-9);
  }
  assert.ok(seq.some((u) => u[4] === 1), 'pen-up jump is flagged');
  assert.equal(preprocess([[[1, 1]]]), null);
});

test('built-in gestures: ≥ 95% accuracy on unseen synthetic drawings', () => {
  const testSet = makeDataset(BUILTIN, 40, 4242, { prefixes: false });
  let ok = 0;
  for (const ex of testSet) if (clf.predict(ex.seq).label === ex.label) ok++;
  assert.ok(ok / testSet.length >= 0.95, `accuracy ${(ok / testSet.length * 100).toFixed(1)}%`);
});

test('a circle is recognised wherever it starts and whichever way it goes', () => {
  for (let k = 0; k < 8; k++) {
    for (const dir of [1, -1]) {
      const pts = [];
      for (let i = 0; i <= 48; i++) {
        const a = (k * Math.PI) / 4 + (dir * i * Math.PI * 2) / 48;
        pts.push([300 + 100 * Math.cos(a), 200 + 100 * Math.sin(a)]);
      }
      assert.equal(clf.predict(preprocess([pts])).label, 'circle', `start ${k}, dir ${dir}`);
    }
  }
});

test('guesses most gestures correctly when only 75% drawn', () => {
  const rng = new Rng(77);
  let ok = 0;
  let n = 0;
  for (const name of BUILTIN) {
    for (let i = 0; i < 20; i++) {
      n++;
      if (clf.predict(preprocess(truncate(synthesize(name, rng), 0.75))).label === name) ok++;
    }
  }
  assert.ok(ok / n >= 0.85, `early accuracy ${(ok / n * 100).toFixed(1)}%`);
});

test('few-shot: learns a new gesture (heart) from 3 examples', () => {
  const heart = (rng, jitter = 0.03) => {
    const pts = [];
    for (let i = 0; i <= 80; i++) {
      const t = (i / 80) * Math.PI * 2;
      const x = 16 * Math.sin(t) ** 3;
      const y = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t));
      pts.push([x / 17 + rng.gauss() * jitter, y / 17 + rng.gauss() * jitter]);
    }
    return [pts];
  };
  const rng = new Rng(8);
  const own = new SequenceClassifier({ inputs: INPUTS, size: 200, spectralRadius: 0.9, leak: 0.3, inputScaling: 1, lambda: 1e-3, seed: 7 });
  const classes = [...BUILTIN, 'heart'];
  const data = makeDataset(BUILTIN, 50, 1).concat(expandExamples([heart(rng), heart(rng), heart(rng)], 'heart', 14, rng));
  own.train(data, classes);
  let ok = 0;
  for (let i = 0; i < 20; i++) if (own.predict(preprocess(heart(rng, 0.05))).label === 'heart') ok++;
  assert.ok(ok >= 18, `heart recognised ${ok}/20 times`);
  // ...without forgetting the others.
  const testSet = makeDataset(BUILTIN, 20, 4343, { prefixes: false });
  let okOld = 0;
  for (const ex of testSet) if (own.predict(ex.seq).label === ex.label) okOld++;
  assert.ok(okOld / testSet.length >= 0.93, `built-ins after teaching: ${(okOld / testSet.length * 100).toFixed(1)}%`);
});
