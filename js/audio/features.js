// Audio front end: short-time FFT → log-mel energies, computed frame by
// frame as samples arrive (25 ms windows every 10 ms).

export class FFT {
  constructor(n) {
    if (n & (n - 1)) throw new Error('FFT size must be a power of two');
    this.n = n;
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / n);
    }
    this.re = new Float64Array(n);
    this.im = new Float64Array(n);
  }

  // Power spectrum |X[k]|² for k = 0..n/2 of a real signal.
  power(x, out) {
    const { n, rev, re, im } = this;
    for (let i = 0; i < n; i++) {
      re[rev[i]] = i < x.length ? x[i] : 0;
      im[rev[i]] = 0;
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const c = this.cos[k * step];
          const s = this.sin[k * step];
          const a = start + k;
          const b = a + half;
          const tr = re[b] * c - im[b] * s;
          const ti = re[b] * s + im[b] * c;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
    for (let k = 0; k <= n / 2; k++) out[k] = re[k] * re[k] + im[k] * im[k];
    return out;
  }
}

const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);

export function melFilterbank(bands, nfft, sampleRate, fmin, fmax) {
  const bins = nfft / 2 + 1;
  const lo = hzToMel(fmin);
  const hi = hzToMel(Math.min(fmax, sampleRate / 2));
  const centers = Array.from({ length: bands + 2 }, (_, i) => (melToHz(lo + ((hi - lo) * i) / (bands + 1)) * nfft) / sampleRate);
  const filters = [];
  for (let b = 0; b < bands; b++) {
    const [l, c, r] = [centers[b], centers[b + 1], centers[b + 2]];
    const w = new Float64Array(bins);
    for (let k = Math.floor(l); k <= Math.ceil(r) && k < bins; k++) {
      if (k <= l || k >= r) continue;
      w[k] = k < c ? (k - l) / (c - l) : (r - k) / (r - c);
    }
    filters.push(w);
  }
  return { filters, centersHz: centers.slice(1, -1).map((k) => (k * sampleRate) / nfft) };
}

export class MelFrontend {
  constructor(sampleRate, { frameMs = 25, hopMs = 10, bands = 24, fmin = 80, fmax = 7600 } = {}) {
    this.sampleRate = sampleRate;
    this.frameLen = Math.round((sampleRate * frameMs) / 1000);
    this.hop = Math.round((sampleRate * hopMs) / 1000);
    let nfft = 1;
    while (nfft < this.frameLen) nfft <<= 1;
    this.nfft = nfft;
    this.fft = new FFT(nfft);
    this.bands = bands;
    const fb = melFilterbank(bands, nfft, sampleRate, fmin, fmax);
    this.filters = fb.filters;
    this.centersHz = fb.centersHz;
    this.window = Float64Array.from({ length: this.frameLen }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (this.frameLen - 1)));
    this.buf = new Float64Array(this.frameLen + this.hop * 8);
    this.fill = 0;
    this.frame = new Float64Array(this.frameLen);
    this.spec = new Float64Array(nfft / 2 + 1);
  }

  // Feed samples; calls onFrame(logMel: Float32Array, energyDb) per hop.
  push(samples, onFrame) {
    let i = 0;
    while (i < samples.length) {
      const n = Math.min(samples.length - i, this.buf.length - this.fill);
      this.buf.set(samples.subarray(i, i + n), this.fill);
      this.fill += n;
      i += n;
      while (this.fill >= this.frameLen) {
        onFrame(...this.analyse(this.buf));
        this.buf.copyWithin(0, this.hop, this.fill);
        this.fill -= this.hop;
      }
    }
  }

  analyse(buf) {
    const { frameLen, frame, window, spec, filters, bands } = this;
    let energy = 0;
    let mean = 0;
    for (let i = 0; i < frameLen; i++) mean += buf[i];
    mean /= frameLen;
    for (let i = 0; i < frameLen; i++) {
      const v = buf[i] - mean;
      energy += v * v;
      frame[i] = v * window[i];
    }
    this.fft.power(frame, spec);
    const mel = new Float32Array(bands);
    for (let b = 0; b < bands; b++) {
      const w = filters[b];
      let s = 0;
      for (let k = 0; k < w.length; k++) if (w[k]) s += w[k] * spec[k];
      mel[b] = Math.log(s + 1e-9);
    }
    const db = 10 * Math.log10(energy / frameLen + 1e-12);
    return [mel, db];
  }
}

// Turns an utterance (array of log-mel frames + their energies in dB) into
// reservoir inputs: per-band mean removed (so microphone colour and loudness
// don't matter) plus a relative-loudness channel.
export function utteranceInputs(frames, energies, scale = 0.25) {
  const T = frames.length;
  const B = frames[0].length;
  const mean = new Float64Array(B);
  for (const f of frames) for (let b = 0; b < B; b++) mean[b] += f[b] / T;
  let maxE = -Infinity;
  for (const e of energies) maxE = Math.max(maxE, e);
  return frames.map((f, t) => {
    const u = new Float64Array(B + 1);
    for (let b = 0; b < B; b++) u[b] = (f[b] - mean[b]) * scale;
    u[B] = Math.max(-3, (energies[t] - maxE) / 15);
    return u;
  });
}

// Augmentation for few-shot training: random time stretch plus a little
// noise on the features.
export function stretch(frames, energies, factor, rng, noise = 0.08) {
  const T = frames.length;
  const T2 = Math.max(4, Math.round(T * factor));
  const B = frames[0].length;
  const outF = [];
  const outE = [];
  for (let j = 0; j < T2; j++) {
    const x = (j / (T2 - 1)) * (T - 1);
    const i = Math.floor(x);
    const a = x - i;
    const i2 = Math.min(T - 1, i + 1);
    const f = new Float32Array(B);
    for (let b = 0; b < B; b++) f[b] = frames[i][b] * (1 - a) + frames[i2][b] * a + rng.gauss() * noise;
    outF.push(f);
    outE.push(energies[i] * (1 - a) + energies[i2] * a);
  }
  return { frames: outF, energies: outE };
}
