import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Pond, sineSquareStream } from '../js/systems/wave.js';
import { RidgeAccumulator, matVec } from '../js/core/linalg.js';
import { Rng } from '../js/core/rng.js';

function sineSquare(squared) {
  const pond = new Pond();
  const F = pond.featureLength(squared);
  const f = new Float64Array(F);
  const train = sineSquareStream(1);
  const acc = new RidgeAccumulator(F, 1);
  for (let t = 0; t < 3000; t++) {
    const { u, label } = train.next();
    pond.step([u]);
    if (t >= 100) acc.add(pond.read(f, squared), [label]);
  }
  const W = acc.solve(1e-4);
  const testStream = sineSquareStream(2);
  let ok = 0;
  let n = 0;
  let sOk = 0;
  let sN = 0;
  let since = 0;
  let last = -1;
  for (let t = 0; t < 2000; t++) {
    const { u, label } = testStream.next();
    since = label === last ? since + 1 : 0;
    last = label;
    pond.step([u]);
    const y = matVec(W, 1, F, pond.read(f, squared))[0];
    if (t < 50) continue;
    const c = (y > 0.5 ? 1 : 0) === label;
    n++;
    if (c) ok++;
    if (since > 18) {
      sN++;
      if (c) sOk++;
    }
  }
  return { acc: ok / n, settled: sOk / sN };
}

test('wave simulation stays bounded and calm without input', () => {
  const pond = new Pond();
  for (let t = 0; t < 200; t++) pond.step([t < 20 ? 1 : 0]);
  let max = 0;
  for (const v of pond.u) max = Math.max(max, Math.abs(v));
  assert.ok(Number.isFinite(max) && max < 1, `max height ${max}`);
  for (let t = 0; t < 800; t++) pond.step([0]);
  let late = 0;
  for (const v of pond.u) late = Math.max(late, Math.abs(v));
  assert.ok(late < max * 0.05, 'ripples die out with damping');
});

test('water + nonlinear camera tells sine from square waves', () => {
  const { acc, settled } = sineSquare(true);
  assert.ok(acc > 0.85, `overall accuracy ${(acc * 100).toFixed(1)}%`);
  assert.ok(settled > 0.95, `settled accuracy ${(settled * 100).toFixed(1)}%`);
});

test('with a purely linear sensor the same task fails', () => {
  const { acc } = sineSquare(false);
  assert.ok(acc < 0.75, `linear sensor reached ${(acc * 100).toFixed(1)}%`);
});

test('the water remembers recent input', () => {
  const pond = new Pond();
  const F = pond.featureLength(false);
  const f = new Float64Array(F);
  const rng = new Rng(1);
  const D = 10;
  const hist = new Float64Array(D + 1);
  const acc = new RidgeAccumulator(F, D + 1);
  const feats = [];
  const targets = [];
  for (let t = 0; t < 3000; t++) {
    hist.copyWithin(1, 0, D);
    hist[0] = rng.uniform(-1, 1);
    pond.step([hist[0]]);
    pond.read(f, false);
    if (t < 100) continue;
    if (t < 2000) acc.add(f, hist);
    else {
      feats.push(Float64Array.from(f));
      targets.push(Float64Array.from(hist));
    }
  }
  const W = acc.solve(1e-4);
  for (const d of [3, 5, 10]) {
    let se = 0;
    let sv = 0;
    for (let k = 0; k < feats.length; k++) {
      const y = matVec(W.subarray(d * F, (d + 1) * F), 1, F, feats[k])[0];
      se += (y - targets[k][d]) ** 2;
      sv += targets[k][d] ** 2;
    }
    assert.ok(1 - se / sv > 0.7, `R² for delay ${d}: ${(1 - se / sv).toFixed(2)}`);
  }
});
