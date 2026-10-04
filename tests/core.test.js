import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Rng } from '../js/core/rng.js';
import { cholesky, choleskySolve, RidgeAccumulator, ridgeSolve, matVec } from '../js/core/linalg.js';
import { EchoStateNetwork, SparseMatrix, estimateSpectralRadius } from '../js/core/esn.js';
import { FixedStepLoop, runChunked } from '../js/core/loop.js';

test('rng is deterministic per seed', () => {
  const a = new Rng(42);
  const b = new Rng(42);
  const c = new Rng(43);
  const sa = Array.from({ length: 10 }, () => a.next());
  const sb = Array.from({ length: 10 }, () => b.next());
  const sc = Array.from({ length: 10 }, () => c.next());
  assert.deepEqual(sa, sb);
  assert.notDeepEqual(sa, sc);
  for (const v of sa) assert.ok(v >= 0 && v < 1);
});

test('gaussian has roughly zero mean and unit variance', () => {
  const r = new Rng(5);
  let s = 0;
  let s2 = 0;
  const n = 20000;
  for (let i = 0; i < n; i++) {
    const g = r.gauss();
    s += g;
    s2 += g * g;
  }
  assert.ok(Math.abs(s / n) < 0.03);
  assert.ok(Math.abs(s2 / n - 1) < 0.05);
});

test('cholesky solves an SPD system', () => {
  const A = new Float64Array([4, 2, 0.4, 2, 5, 1, 0.4, 1, 3]);
  const x = [1, -2, 0.5];
  const b = new Float64Array(3);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) b[i] += A[i * 3 + j] * x[j];
  const L = new Float64Array(A);
  assert.ok(cholesky(L, 3));
  const sol = choleskySolve(L, 3, b, 1);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(sol[i] - x[i]) < 1e-10);
});

test('ridge regression recovers a known linear map (primal and dual agree)', () => {
  const rng = new Rng(3);
  const F = 12;
  const C = 2;
  const Wtrue = Float64Array.from({ length: C * F }, () => rng.uniform(-1, 1));
  const make = (S) => {
    const X = new Float64Array(S * F);
    const Y = new Float64Array(S * C);
    for (let s = 0; s < S; s++) {
      for (let k = 0; k < F; k++) X[s * F + k] = rng.gauss();
      const y = matVec(Wtrue, C, F, X.subarray(s * F, (s + 1) * F));
      Y.set(y, s * C);
    }
    return { X, Y };
  };
  // Primal (S > F)
  const big = make(200);
  const Wp = ridgeSolve(big.X, 200, F, big.Y, C, 1e-9);
  for (let i = 0; i < Wtrue.length; i++) assert.ok(Math.abs(Wp[i] - Wtrue[i]) < 1e-5);

  // Streaming accumulator gives the same answer.
  const acc = new RidgeAccumulator(F, C);
  for (let s = 0; s < 200; s++) acc.add(big.X.subarray(s * F, (s + 1) * F), big.Y.subarray(s * C, (s + 1) * C));
  const Wa = acc.solve(1e-9);
  for (let i = 0; i < Wtrue.length; i++) assert.ok(Math.abs(Wa[i] - Wtrue[i]) < 1e-5);

  // Dual (S < F) equals primal on the same data.
  const small = make(8);
  const lam = 1e-2;
  const Wd = ridgeSolve(small.X, 8, F, small.Y, C, lam);
  const acc2 = new RidgeAccumulator(F, C);
  for (let s = 0; s < 8; s++) acc2.add(small.X.subarray(s * F, (s + 1) * F), small.Y.subarray(s * C, (s + 1) * C));
  const Wp2 = acc2.solve(lam);
  for (let i = 0; i < Wd.length; i++) assert.ok(Math.abs(Wd[i] - Wp2[i]) < 1e-8, `${Wd[i]} vs ${Wp2[i]}`);
});

test('spectral radius estimate is exact on a scaled cyclic permutation', () => {
  const n = 50;
  const rowPtr = Int32Array.from({ length: n + 1 }, (_, i) => i);
  const cols = Int32Array.from({ length: n }, (_, i) => (i + 1) % n);
  const vals = new Float64Array(n).fill(0.7);
  const W = new SparseMatrix(n, rowPtr, cols, vals);
  assert.ok(Math.abs(estimateSpectralRadius(W) - 0.7) < 1e-6);
});

test('reservoir is scaled to the requested spectral radius', () => {
  for (const rho of [0.5, 0.9, 1.3]) {
    const esn = new EchoStateNetwork({ size: 400, degree: 8, spectralRadius: rho, seed: 11 });
    const measured = estimateSpectralRadius(esn.W, 1500, 300, 999);
    assert.ok(Math.abs(measured - rho) / rho < 0.1, `wanted ${rho}, got ${measured}`);
  }
});

test('echo state property: different starting states forget their past', () => {
  const esn = new EchoStateNetwork({ size: 200, spectralRadius: 0.8, inputs: 1, seed: 2 });
  const rng = new Rng(9);
  const inputs = Array.from({ length: 300 }, () => [rng.uniform(-1, 1)]);
  esn.x.fill(0.9);
  for (const u of inputs) esn.step(u);
  const a = Float64Array.from(esn.x);
  esn.x.fill(-0.9);
  for (const u of inputs) esn.step(u);
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff = Math.max(diff, Math.abs(a[i] - esn.x[i]));
  assert.ok(diff < 1e-6, `states still differ by ${diff}`);
});

test('fixed-step loop runs at the same speed for any display refresh rate', () => {
  const results = [];
  for (const hz of [30, 60, 75, 144, 240]) {
    let clock = 0;
    let pending = null;
    let steps = 0;
    const loop = new FixedStepLoop({
      stepsPerSecond: 100,
      step: () => steps++,
      now: () => clock,
      requestFrame: (cb) => {
        pending = cb;
        return 1;
      },
      cancelFrame: () => {},
    });
    loop.start();
    const frameMs = 1000 / hz;
    while (clock < 2000 - 1e-9) {
      clock += frameMs;
      const cb = pending;
      pending = null;
      cb(clock);
    }
    loop.stop();
    results.push(steps);
  }
  for (const s of results) assert.ok(Math.abs(s - 200) <= 1, `steps per 2 s: ${results}`);
});

test('fixed-step loop does not try to replay a long stall', () => {
  let clock = 0;
  let pending = null;
  let steps = 0;
  const loop = new FixedStepLoop({
    stepsPerSecond: 100,
    step: () => steps++,
    now: () => clock,
    requestFrame: (cb) => {
      pending = cb;
      return 1;
    },
    cancelFrame: () => {},
  });
  loop.start();
  clock = 60000; // tab was hidden for a minute
  pending(clock);
  assert.ok(steps <= 25, `ran ${steps} catch-up steps`);
});

test('runChunked visits every index and reports progress', async () => {
  const seen = [];
  let last = 0;
  await runChunked(100, (i) => seen.push(i), { budgetMs: 0, onProgress: (p) => (last = p) });
  assert.equal(seen.length, 100);
  assert.equal(seen[99], 99);
  assert.equal(last, 1);
});
