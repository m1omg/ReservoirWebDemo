// Time-series forecasters with a common interface:
//   observe(u)  feed a true value, returns the prediction of the next value
//   dream()     feed the model's own last prediction back in (closed loop)
// Training is teacher forcing: show the true series, learn to predict u(t+1).

import { EchoStateNetwork, FeatureMap } from './esn.js';
import { RidgeAccumulator, matVec } from './linalg.js';
import { runChunked } from './loop.js';

export class ReservoirForecaster {
  constructor({ dim, esn = {}, squared = true, lambda = 1e-6, washout = 100 }) {
    this.dim = dim;
    this.esn = new EchoStateNetwork({ ...esn, inputs: dim });
    this.map = new FeatureMap(this.esn.size, dim, { squared, includeInput: true });
    this.lambda = lambda;
    this.washout = washout;
    this.f = new Float64Array(this.map.length);
    this.y = new Float64Array(dim);
    this.Wout = null;
  }

  get featureCount() {
    return this.map.length;
  }

  // data: flat series (count × dim) starting at `start`.
  async train(data, start, count, { onProgress, budgetMs = 12 } = {}) {
    const { dim, esn, map, f } = this;
    const acc = new RidgeAccumulator(map.length, dim);
    esn.reset();
    const u = new Float64Array(dim);
    await runChunked(
      count - 1,
      (t) => {
        const o = (start + t) * dim;
        for (let j = 0; j < dim; j++) u[j] = data[o + j];
        const x = esn.step(u);
        if (t >= this.washout) acc.add(map.fill(x, u, f), data.subarray(o + dim, o + 2 * dim));
      },
      { onProgress, budgetMs },
    );
    this.Wout = acc.solve(this.lambda);
    this.samples = acc.count;
    // Leave the reservoir having seen everything except the last value, so
    // the caller can observe() it and start forecasting from there.
  }

  observe(u) {
    const x = this.esn.step(u);
    matVec(this.Wout, this.dim, this.map.length, this.map.fill(x, u, this.f), this.y);
    return this.y;
  }

  dream() {
    return this.observe(Float64Array.from(this.y));
  }

  perturb(rng, amount) {
    for (let j = 0; j < this.dim; j++) this.y[j] += rng.gauss() * amount;
    const x = this.esn.x;
    for (let i = 0; i < x.length; i++) x[i] += rng.gauss() * amount * 0.1;
  }

  get state() {
    return this.esn.x;
  }
}

// Baseline without a reservoir: a linear autoregressive model that predicts
// u(t+1) from the last `delays` values, trained with the very same ridge
// regression. It shows what the reservoir's nonlinear memory adds.
export class LinearForecaster {
  constructor({ dim, delays = 8, lambda = 1e-6 }) {
    this.dim = dim;
    this.delays = delays;
    this.lambda = lambda;
    this.length = 1 + dim * delays;
    this.hist = new Float64Array(dim * delays);
    this.f = new Float64Array(this.length);
    this.y = new Float64Array(dim);
    this.Wout = null;
  }

  get featureCount() {
    return this.length;
  }

  push(u) {
    this.hist.copyWithin(this.dim, 0, this.hist.length - this.dim);
    this.hist.set(u.subarray ? u.subarray(0, this.dim) : u, 0);
    this.f[0] = 1;
    this.f.set(this.hist, 1);
    return this.f;
  }

  async train(data, start, count, { onProgress, budgetMs = 12 } = {}) {
    const { dim } = this;
    const acc = new RidgeAccumulator(this.length, dim);
    this.hist.fill(0);
    await runChunked(
      count - 1,
      (t) => {
        const o = (start + t) * dim;
        const f = this.push(data.subarray(o, o + dim));
        if (t >= this.delays) acc.add(f, data.subarray(o + dim, o + 2 * dim));
      },
      { onProgress, budgetMs },
    );
    this.Wout = acc.solve(this.lambda);
    this.samples = acc.count;
  }

  observe(u) {
    matVec(this.Wout, this.dim, this.length, this.push(u), this.y);
    return this.y;
  }

  dream() {
    return this.observe(Float64Array.from(this.y));
  }

  perturb(rng, amount) {
    for (let j = 0; j < this.dim; j++) this.y[j] += rng.gauss() * amount;
  }

  get state() {
    return this.hist;
  }
}

// Closed-loop forecast after training on data[start .. start+count).
// Returns the predictions and the "valid time": the first step at which the
// normalised error exceeds `threshold` (Pathak et al. 2018 style).
export function forecast(model, data, dim, start, count, horizon, threshold = 0.4) {
  const last = start + count - 1;
  const pred = new Float64Array(horizon * dim);
  let y = model.observe(data.subarray(last * dim, last * dim + dim));
  let valid = horizon;
  let sumSq = 0;
  for (let k = 0; k < horizon; k++) {
    pred.set(y, k * dim);
    const o = (last + 1 + k) * dim;
    let err = 0;
    for (let j = 0; j < dim; j++) err += (y[j] - data[o + j]) ** 2;
    sumSq += err;
    // z-scored data: mean squared norm of the signal is `dim`.
    if (valid === horizon && Math.sqrt(err / dim) > threshold) valid = k;
    if (k < horizon - 1) y = model.dream();
  }
  return { pred, valid, rmse: Math.sqrt(sumSq / (horizon * dim)) };
}
