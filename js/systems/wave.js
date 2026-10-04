// A simulated "bucket of water": a 2-D damped wave equation on a grid.
// Fernando & Sojakka (2003) used a real tank of water, LEGO motors and a
// camera as a reservoir. Here the motors push the surface, a set of probe
// points plays the camera, and the waves do the computing.

import { Rng } from '../core/rng.js';

export class Pond {
  constructor({ width = 72, height = 48, speed = 0.6, damping = 0.06, substeps = 4, motors = null, probes = 48, exposure = 0.05 } = {}) {
    this.w = width;
    this.h = height;
    this.c2 = speed * speed; // (c·dt/dx)², must stay < 0.5 for stability
    this.damping = damping;
    this.relax = 0.004;
    this.substeps = substeps;
    this.u = new Float32Array(width * height);
    this.prev = new Float32Array(width * height);
    this.next = new Float32Array(width * height);
    this.motors = motors || [
      { x: width * 0.22, y: height * 0.32, r: 2.2, gain: 1 },
      { x: width * 0.22, y: height * 0.7, r: 2.2, gain: -0.8 },
    ];
    // Precomputed Gaussian footprints for the motors.
    this.footprints = this.motors.map((m) => {
      const cells = [];
      const R = Math.ceil(m.r * 2.5);
      for (let dy = -R; dy <= R; dy++) {
        for (let dx = -R; dx <= R; dx++) {
          const x = Math.round(m.x) + dx;
          const y = Math.round(m.y) + dy;
          if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) continue;
          const wgt = Math.exp(-(dx * dx + dy * dy) / (2 * m.r * m.r));
          if (wgt > 0.02) cells.push(y * width + x, wgt);
        }
      }
      return cells;
    });
    this.probes = probeGrid(width, height, probes);
    this.exposure = exposure;
    this.glow = new Float32Array(this.probes.length);
    this.time = 0;
  }

  reset() {
    this.u.fill(0);
    this.prev.fill(0);
    this.next.fill(0);
    this.glow.fill(0);
  }

  // Advance one input sample: `drive[k]` is the push of motor k.
  step(drive) {
    for (let s = 0; s < this.substeps; s++) this.substep(drive);
  }

  substep(drive) {
    const { w, h, u, prev, next, c2 } = this;
    const keep = 1 - this.damping;
    const relax = this.relax;
    for (let y = 1; y < h - 1; y++) {
      const row = y * w;
      for (let x = 1; x < w - 1; x++) {
        const i = row + x;
        const lap = u[i - 1] + u[i + 1] + u[i - w] + u[i + w] - 4 * u[i];
        // The last term slowly pulls the level back to rest, so pushes that
        // don't average out can't raise the whole bucket forever.
        next[i] = u[i] + keep * (u[i] - prev[i]) + c2 * lap - relax * u[i];
      }
    }
    // Reflecting walls: copy the neighbour (zero slope at the edge).
    for (let x = 0; x < w; x++) {
      next[x] = next[w + x];
      next[(h - 1) * w + x] = next[(h - 2) * w + x];
    }
    for (let y = 0; y < h; y++) {
      next[y * w] = next[y * w + 1];
      next[y * w + w - 1] = next[y * w + w - 2];
    }
    // Motors push the surface.
    for (let k = 0; k < this.footprints.length; k++) {
      const f = (drive[k] ?? drive[0] ?? 0) * this.motors[k].gain * 0.06;
      if (f === 0) continue;
      const fp = this.footprints[k];
      for (let j = 0; j < fp.length; j += 2) next[fp[j]] += f * fp[j + 1];
    }
    // Rotate buffers: prev ← u ← next.
    this.prev = u;
    this.u = next;
    this.next = prev;
    this.time++;
    // The "camera" integrates brightness over its exposure: a running
    // average of the squared surface height at every probe.
    const a = this.exposure;
    const pr = this.probes;
    const glow = this.glow;
    for (let p = 0; p < pr.length; p++) {
      const v = this.u[pr[p]] * 4;
      glow[p] += a * (v * v - glow[p]);
    }
  }

  // Probe readings: the surface height at each probe and, with a nonlinear
  // sensor, the brightness a camera would record (squared height averaged
  // over the exposure time).
  read(out, squared) {
    const P = this.probes.length;
    for (let p = 0; p < P; p++) {
      out[1 + p] = this.u[this.probes[p]] * 4;
      if (squared) out[1 + P + p] = this.glow[p];
    }
    out[0] = 1;
    return out;
  }

  featureLength(squared) {
    return 1 + this.probes.length * (squared ? 2 : 1);
  }
}

function probeGrid(w, h, n) {
  // An evenly spread rows×cols grid of probe points over the pond.
  const cols = Math.round(Math.sqrt(n * (w / h)));
  const rows = Math.ceil(n / cols);
  const out = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols && out.length < n; c++) {
      const x = Math.round(((c + 0.5) / cols) * (w - 2)) + 1;
      const y = Math.round(((r + 0.5) / rows) * (h - 2)) + 1;
      out.push(y * w + x);
    }
  }
  return out;
}

// ---------- Input signals for the tasks ----------

// Endless stream of sine and square waves of the same period, switching at
// random moments. next() returns { u, label } with label 1 for square.
export function sineSquareStream(seed, period = 12) {
  const rng = new Rng(seed);
  let kind = rng.int(2);
  let left = 0;
  let k = 0;
  return {
    period,
    next() {
      if (left === 0) {
        kind = 1 - kind;
        left = (4 + rng.int(6)) * period;
        k = 0;
      }
      const ph = (k++ / period) * Math.PI * 2;
      left--;
      // Same loudness (RMS) for both, so only the shape differs.
      const u = kind ? (Math.sin(ph) >= 0 ? 1 : -1) * Math.SQRT1_2 : Math.sin(ph);
      return { u, label: kind };
    },
  };
}

export function sineSquareSignal(length, seed, period = 12) {
  const s = sineSquareStream(seed, period);
  const input = new Float32Array(length);
  const label = new Float32Array(length);
  for (let t = 0; t < length; t++) {
    const { u, label: l } = s.next();
    input[t] = u;
    label[t] = l;
  }
  return { input, label };
}

export function randomSignal(length, seed) {
  const rng = new Rng(seed);
  return Float32Array.from({ length }, () => rng.uniform(-1, 1));
}
