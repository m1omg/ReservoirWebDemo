// Classic chaotic systems used to benchmark forecasters.

import { Rng } from '../core/rng.js';

function rk4(f, s, h, dim, k) {
  const [k1, k2, k3, k4, tmp] = k;
  f(s, k1);
  for (let i = 0; i < dim; i++) tmp[i] = s[i] + 0.5 * h * k1[i];
  f(tmp, k2);
  for (let i = 0; i < dim; i++) tmp[i] = s[i] + 0.5 * h * k2[i];
  f(tmp, k3);
  for (let i = 0; i < dim; i++) tmp[i] = s[i] + h * k3[i];
  f(tmp, k4);
  for (let i = 0; i < dim; i++) s[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}

function odeSystem({ deriv, dim, dt, substeps, start, transient }) {
  return (steps, seed) => {
    const rng = new Rng(seed);
    const s = Float64Array.from(start, (v) => v + rng.uniform(-1, 1));
    const k = Array.from({ length: 5 }, () => new Float64Array(dim));
    const h = dt / substeps;
    for (let t = 0; t < transient; t++) for (let j = 0; j < substeps; j++) rk4(deriv, s, h, dim, k);
    const out = new Float64Array(steps * dim);
    for (let t = 0; t < steps; t++) {
      out.set(s, t * dim);
      for (let j = 0; j < substeps; j++) rk4(deriv, s, h, dim, k);
    }
    return out;
  };
}

// Mackey–Glass delay differential equation:
//   dx/dt = β·x(t−τ) / (1 + x(t−τ)^n) − γ·x(t)
// integrated with RK4 on a fine grid (the delayed term is linearly
// interpolated at half steps) and sampled every `dt` time units.
function mackeyGlass({ tau = 17, beta = 0.2, gamma = 0.1, n = 10, h = 0.1, dt = 1, transient = 1000 }) {
  return (steps, seed) => {
    const rng = new Rng(seed);
    const lag = Math.round(tau / h);
    const per = Math.round(dt / h);
    const total = lag + 1 + Math.round(transient / h) + steps * per;
    const x = new Float64Array(total);
    for (let i = 0; i <= lag; i++) x[i] = 1.2 + rng.uniform(-0.1, 0.1);
    const f = (xt, xd) => (beta * xd) / (1 + Math.pow(xd, n)) - gamma * xt;
    for (let i = lag; i < total - 1; i++) {
      const d0 = x[i - lag];
      const d1 = x[i - lag + 1];
      const dm = 0.5 * (d0 + d1);
      const xi = x[i];
      const k1 = f(xi, d0);
      const k2 = f(xi + 0.5 * h * k1, dm);
      const k3 = f(xi + 0.5 * h * k2, dm);
      const k4 = f(xi + h * k3, d1);
      x[i + 1] = xi + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    }
    const out = new Float64Array(steps);
    const first = total - steps * per;
    for (let t = 0; t < steps; t++) out[t] = x[first + t * per];
    return out;
  };
}

export const SYSTEMS = {
  lorenz: {
    name: 'Lorenz',
    dim: 3,
    dt: 0.02,
    lyapunov: 0.9056, // largest Lyapunov exponent (Sprott)
    description: 'Lorenz (1963): a toy model of atmospheric convection — the original "butterfly effect".',
    generate: odeSystem({
      dim: 3,
      dt: 0.02,
      substeps: 2,
      start: [1, 1, 25],
      transient: 1500,
      deriv: (s, d) => {
        d[0] = 10 * (s[1] - s[0]);
        d[1] = s[0] * (28 - s[2]) - s[1];
        d[2] = s[0] * s[1] - (8 / 3) * s[2];
      },
    }),
    view: [0, 2, 1],
    preset: { size: 300, degree: 3, spectralRadius: 0.3, leak: 1, inputScaling: 0.3, biasScaling: 0.5, lambda: 1e-11, train: 4000 },
    kick: 1.2,
  },
  rossler: {
    name: 'Rössler',
    dim: 3,
    dt: 0.15,
    lyapunov: 0.0714, // largest Lyapunov exponent (Sprott)
    description: 'Rössler (1976): a spiral that occasionally flings itself upward — chaotic but gentle.',
    generate: odeSystem({
      dim: 3,
      dt: 0.15,
      substeps: 3,
      start: [1, 1, 0],
      transient: 1000,
      deriv: (s, d) => {
        d[0] = -s[1] - s[2];
        d[1] = s[0] + 0.2 * s[1];
        d[2] = 0.2 + s[2] * (s[0] - 5.7);
      },
    }),
    view: [0, 1, 2],
    preset: { size: 300, degree: 3, spectralRadius: 0.3, leak: 1, inputScaling: 0.3, biasScaling: 0.5, lambda: 1e-11, train: 4000 },
    kick: 0.5,
  },
  mackeyGlass: {
    name: 'Mackey–Glass',
    dim: 1,
    dt: 1,
    lyapunov: null,
    description: 'Mackey–Glass (1977, τ = 17): a delay equation from blood-cell production; a standard reservoir benchmark.',
    generate: mackeyGlass({}),
    // 1-D series: shown as a delay embedding (x(t), x(t−6), x(t−12)).
    embedLag: 6,
    preset: { size: 300, degree: 3, spectralRadius: 0.9, leak: 0.5, inputScaling: 0.3, biasScaling: 0.5, lambda: 1e-11, train: 4000 },
    kick: 0.3,
  },
};

export function zscore(data, dim, count) {
  const mean = new Float64Array(dim);
  const std = new Float64Array(dim);
  for (let t = 0; t < count; t++) for (let j = 0; j < dim; j++) mean[j] += data[t * dim + j];
  for (let j = 0; j < dim; j++) mean[j] /= count;
  for (let t = 0; t < count; t++) for (let j = 0; j < dim; j++) std[j] += (data[t * dim + j] - mean[j]) ** 2;
  for (let j = 0; j < dim; j++) std[j] = Math.sqrt(std[j] / count) || 1;
  const out = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = (data[i] - mean[i % dim]) / std[i % dim];
  return { data: out, mean, std };
}
