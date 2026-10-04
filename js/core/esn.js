// Echo State Network: a big, fixed, random recurrent network (the "reservoir").
// Nothing in here is ever trained — only a linear readout on top of it is.

import { Rng } from './rng.js';

// Sparse matrix in compressed-sparse-row form: row i holds the incoming
// connections of neuron i.
export class SparseMatrix {
  constructor(n, rowPtr, cols, vals) {
    this.n = n;
    this.rowPtr = rowPtr;
    this.cols = cols;
    this.vals = vals;
  }

  mul(x, out) {
    const { n, rowPtr, cols, vals } = this;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let p = rowPtr[i], e = rowPtr[i + 1]; p < e; p++) s += vals[p] * x[cols[p]];
      out[i] = s;
    }
    return out;
  }

  scale(f) {
    for (let p = 0; p < this.vals.length; p++) this.vals[p] *= f;
  }

  static random(n, degree, rng) {
    const k = Math.min(degree, n);
    const rowPtr = new Int32Array(n + 1);
    const cols = new Int32Array(n * k);
    const vals = new Float64Array(n * k);
    const chosen = new Set();
    for (let i = 0; i < n; i++) {
      rowPtr[i] = i * k;
      chosen.clear();
      while (chosen.size < k) chosen.add(rng.int(n));
      let p = i * k;
      for (const c of [...chosen].sort((a, b) => a - b)) {
        cols[p] = c;
        vals[p] = rng.uniform(-1, 1);
        p++;
      }
    }
    rowPtr[n] = n * k;
    return new SparseMatrix(n, rowPtr, cols, vals);
  }
}

// Spectral radius via Gelfand's formula, rho = lim ||W^k v||^(1/k):
// average the per-step log growth of a repeatedly multiplied vector. Unlike
// plain power iteration this also works when the dominant eigenvalues are a
// complex pair (the vector then rotates, but its growth rate is still rho).
export function estimateSpectralRadius(W, iters = 600, burn = 150, seed = 12345) {
  const rng = new Rng(seed);
  let v = new Float64Array(W.n);
  let w = new Float64Array(W.n);
  for (let i = 0; i < W.n; i++) v[i] = rng.gauss();
  let logSum = 0;
  let counted = 0;
  for (let it = 0; it < iters; it++) {
    W.mul(v, w);
    let norm = 0;
    for (let i = 0; i < W.n; i++) norm += w[i] * w[i];
    norm = Math.sqrt(norm);
    if (norm === 0) return 0;
    for (let i = 0; i < W.n; i++) w[i] /= norm;
    if (it >= burn) {
      logSum += Math.log(norm);
      counted++;
    }
    [v, w] = [w, v];
  }
  return Math.exp(logSum / counted);
}

export const DEFAULT_ESN = {
  size: 300,
  inputs: 1,
  degree: 6,
  spectralRadius: 0.9,
  leak: 1,
  inputScaling: 0.5,
  biasScaling: 0.2,
  // 'dense': every neuron sees every input. 'single': each neuron sees one input.
  inputMode: 'dense',
  seed: 1,
};

export class EchoStateNetwork {
  constructor(options = {}) {
    const o = { ...DEFAULT_ESN, ...options };
    this.options = o;
    this.size = o.size;
    this.inputs = o.inputs;
    this.leak = o.leak;
    const rng = new Rng(o.seed);
    const N = o.size;

    this.W = SparseMatrix.random(N, o.degree, rng);
    const rho = estimateSpectralRadius(this.W);
    this.W.scale(rho > 0 ? o.spectralRadius / rho : 0);

    this.Win = new Float64Array(N * o.inputs);
    for (let i = 0; i < N; i++) {
      if (o.inputMode === 'single') {
        this.Win[i * o.inputs + (i % o.inputs)] = rng.uniform(-1, 1) * o.inputScaling;
      } else {
        for (let j = 0; j < o.inputs; j++) this.Win[i * o.inputs + j] = rng.uniform(-1, 1) * o.inputScaling;
      }
    }
    this.bias = new Float64Array(N);
    for (let i = 0; i < N; i++) this.bias[i] = rng.uniform(-1, 1) * o.biasScaling;

    this.x = new Float64Array(N);
    this.pre = new Float64Array(N);
  }

  reset() {
    this.x.fill(0);
  }

  // One time step: x ← (1−a)·x + a·tanh(W·x + Win·u + b)
  step(u) {
    const { size: N, inputs: D, Win, bias, x, pre, leak } = this;
    this.W.mul(x, pre);
    for (let i = 0; i < N; i++) {
      let s = pre[i] + bias[i];
      const r = i * D;
      for (let j = 0; j < D; j++) s += Win[r + j] * u[j];
      x[i] = (1 - leak) * x[i] + leak * Math.tanh(s);
    }
    return x;
  }
}

// Readout features built from a reservoir state: [1, u?, x, x_even²?].
// Squaring half the neurons (Pathak et al. 2018; Lu, Hunt & Ott 2018) breaks
// the odd symmetry of tanh, which matters for systems like Lorenz.
export class FeatureMap {
  constructor(size, inputs, { squared = true, includeInput = false } = {}) {
    this.size = size;
    this.inputs = inputs;
    this.squared = squared;
    this.includeInput = includeInput;
    this.length = 1 + (includeInput ? inputs : 0) + size + (squared ? Math.ceil(size / 2) : 0);
  }

  fill(x, u, out) {
    let p = 0;
    out[p++] = 1;
    if (this.includeInput) for (let j = 0; j < this.inputs; j++) out[p++] = u[j];
    for (let i = 0; i < this.size; i++) out[p++] = x[i];
    if (this.squared) for (let i = 0; i < this.size; i += 2) out[p++] = x[i] * x[i];
    return out;
  }
}
