import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SYSTEMS, zscore } from '../js/systems/chaos.js';
import { ReservoirForecaster, LinearForecaster, forecast } from '../js/core/forecaster.js';

async function setup(name, seed, model = 'reservoir') {
  const sys = SYSTEMS[name];
  const p = sys.preset;
  const raw = sys.generate(p.train + 2010, seed);
  const { data } = zscore(raw, sys.dim, p.train);
  const m =
    model === 'reservoir'
      ? new ReservoirForecaster({ dim: sys.dim, esn: { ...p, seed }, lambda: p.lambda, squared: true })
      : new LinearForecaster({ dim: sys.dim, delays: 8, lambda: 1e-9 });
  await m.train(data, 0, p.train, { budgetMs: 1e9 });
  const r = forecast(m, data, sys.dim, 0, p.train, 2000);
  return { sys, m, r, lyap: (steps) => steps * sys.dt * (sys.lyapunov || 1) };
}

test('Lorenz: reservoir forecasts several Lyapunov times ahead', async () => {
  for (const seed of [7, 21]) {
    const { r, lyap } = await setup('lorenz', seed);
    assert.ok(lyap(r.valid) > 5, `valid for only ${lyap(r.valid).toFixed(2)} Lyapunov times`);
  }
});

test('Lorenz: linear autoregression without a reservoir fails quickly', async () => {
  const { r, lyap } = await setup('lorenz', 7, 'linear');
  assert.ok(lyap(r.valid) < 1, `linear baseline lasted ${lyap(r.valid).toFixed(2)} Lyapunov times`);
});

test('Lorenz: dreaming reservoir stays on the attractor (visits both wings)', async () => {
  const { m } = await setup('lorenz', 14);
  let pos = 0;
  const sq = [0, 0, 0];
  const n = 20000;
  for (let k = 0; k < n; k++) {
    const y = m.dream();
    for (let j = 0; j < 3; j++) {
      assert.ok(Number.isFinite(y[j]) && Math.abs(y[j]) < 10);
      sq[j] += y[j] * y[j];
    }
    if (y[0] > 0) pos++;
  }
  for (const s of sq) assert.ok(Math.abs(Math.sqrt(s / n) - 1) < 0.3);
  assert.ok(pos / n > 0.3 && pos / n < 0.7, `fraction on +x wing: ${pos / n}`);
});

test('Rössler: reservoir forecasts several Lyapunov times ahead', async () => {
  const { r, lyap } = await setup('rossler', 7);
  assert.ok(lyap(r.valid) > 3, `valid for only ${lyap(r.valid).toFixed(2)} Lyapunov times`);
});

test('Mackey–Glass: reservoir forecasts hundreds of steps ahead', async () => {
  const { r } = await setup('mackeyGlass', 7);
  assert.ok(r.valid > 300, `valid for only ${r.valid} steps`);
});
