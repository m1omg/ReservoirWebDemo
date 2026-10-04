// Classifies variable-length sequences (pen strokes, spoken words) with a
// reservoir. The reservoir turns a whole sequence into one fixed-size
// "fingerprint" (its state at a few moments plus its average state), and a
// linear readout trained by ridge regression maps fingerprints to classes.

import { EchoStateNetwork } from './esn.js';
import { ridgeSolve, matVec } from './linalg.js';

export class SequenceClassifier {
  constructor({
    inputs,
    size = 200,
    degree = 8,
    spectralRadius = 0.9,
    leak = 0.3,
    inputScaling = 1,
    biasScaling = 0.2,
    seed = 7,
    snapshots = [0.25, 0.5, 0.75, 1],
    useMean = true,
    lambda = 1e-3,
  }) {
    this.esn = new EchoStateNetwork({ size, inputs, degree, spectralRadius, leak, inputScaling, biasScaling, seed });
    this.inputs = inputs;
    this.snapshots = snapshots;
    this.useMean = useMean;
    this.lambda = lambda;
    this.featureLength = 1 + size * (snapshots.length + (useMean ? 1 : 0));
    this.classes = [];
    this.W = null;
  }

  // seq: array of per-step input vectors. If `record` is given, every state
  // is copied into it (used for the "fingerprint" heatmaps).
  features(seq, record = null) {
    const esn = this.esn;
    const N = esn.size;
    const T = seq.length;
    const f = new Float64Array(this.featureLength);
    f[0] = 1;
    esn.reset();
    const marks = this.snapshots.map((s) => Math.max(0, Math.min(T - 1, Math.round(s * (T - 1)))));
    const meanOff = 1 + N * this.snapshots.length;
    for (let t = 0; t < T; t++) {
      const x = esn.step(seq[t]);
      if (record) record.push(Float32Array.from(x));
      for (let k = 0; k < marks.length; k++) {
        if (marks[k] === t) f.set(x, 1 + k * N);
      }
      if (this.useMean) for (let i = 0; i < N; i++) f[meanOff + i] += x[i] / T;
    }
    return f;
  }

  // examples: [{ seq, label }] where label is a class name.
  train(examples, classes) {
    this.classes = classes.slice();
    const C = classes.length;
    const F = this.featureLength;
    const S = examples.length;
    if (S === 0 || C === 0) {
      this.W = null;
      return;
    }
    const X = new Float64Array(S * F);
    const Y = new Float64Array(S * C);
    examples.forEach((ex, s) => {
      X.set(ex.features || this.features(ex.seq), s * F);
      Y[s * C + classes.indexOf(ex.label)] = 1;
    });
    this.W = ridgeSolve(X, S, F, Y, C, this.lambda);
  }

  get trained() {
    return this.W !== null;
  }

  scoresFromFeatures(f) {
    return matVec(this.W, this.classes.length, this.featureLength, f);
  }

  predict(seq, record = null) {
    const f = this.features(seq, record);
    return this.predictFeatures(f);
  }

  predictFeatures(f) {
    const scores = this.scoresFromFeatures(f);
    const probs = softmax(scores, 10);
    let best = 0;
    for (let c = 1; c < scores.length; c++) if (scores[c] > scores[best]) best = c;
    return { scores, probs, index: best, label: this.classes[best], confidence: probs[best] };
  }
}

// Readout scores are roughly one-hot targets (0 or 1), so a sharpening factor
// turns them into readable confidences.
export function softmax(scores, sharpness = 1) {
  let max = -Infinity;
  for (const s of scores) if (s > max) max = s;
  const out = new Float64Array(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) {
    out[i] = Math.exp(sharpness * (scores[i] - max));
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}
