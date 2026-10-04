// Pen gestures: procedural templates, augmentation, and conversion of raw
// strokes into the sequence the reservoir listens to.
//
// A gesture is an array of strokes; a stroke is an array of [x, y] points
// (screen coordinates, y pointing down).

import { Rng } from '../core/rng.js';

export const RESAMPLE = 40;
export const INPUTS = 6;

// ---------- Preprocessing ----------

// Turns strokes into RESAMPLE steps of [x, y, dx, dy, penUp, turn]:
// positions are centred and scaled to about [-1, 1], (dx, dy) is the unit
// direction of travel, penUp marks the invisible jumps between strokes, and
// turn is how sharply the path is bending.
export function preprocess(strokes, n = RESAMPLE) {
  const pts = [];
  for (let s = 0; s < strokes.length; s++) {
    const st = strokes[s];
    for (let i = 0; i < st.length; i++) pts.push({ x: st[i][0], y: st[i][1], up: i === 0 && s > 0 ? 1 : 0 });
  }
  if (pts.length < 2) return null;

  // Normalise using the inked points only.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let cx = 0;
  let cy = 0;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
    cx += p.x;
    cy += p.y;
  }
  const size = Math.max(maxX - minX, maxY - minY);
  if (!(size > 1e-9)) return null;
  cx = (minX + maxX) / 2;
  cy = (minY + maxY) / 2;
  const k = 2 / size;

  // Cumulative arc length; a segment ending at a point with up=1 is a jump.
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y));
  const L = cum[cum.length - 1];
  if (!(L > 0)) return null;

  const out = [];
  let seg = 1;
  const pos = [];
  for (let j = 0; j < n; j++) {
    const target = (j / (n - 1)) * L;
    while (seg < pts.length - 1 && cum[seg] < target) seg++;
    const a = pts[seg - 1];
    const b = pts[seg];
    const span = cum[seg] - cum[seg - 1];
    const t = span > 0 ? (target - cum[seg - 1]) / span : 0;
    pos.push([(a.x + (b.x - a.x) * t - cx) * k, (a.y + (b.y - a.y) * t - cy) * k, b.up]);
  }
  const dirs = [];
  for (let j = 0; j < n; j++) {
    const q = pos[Math.min(n - 1, j + 1)];
    const r = pos[Math.max(0, j - 1)];
    const dx = q[0] - r[0];
    const dy = q[1] - r[1];
    const m = Math.hypot(dx, dy) || 1;
    dirs.push([dx / m, dy / m]);
  }
  for (let j = 0; j < n; j++) {
    const p = pos[j];
    const [dx, dy] = dirs[j];
    // Signed turning angle: constant for a circle wherever you start it.
    const [px, py] = dirs[Math.max(0, j - 1)];
    const turn = Math.atan2(px * dy - py * dx, px * dx + py * dy);
    out.push(Float64Array.of(p[0], p[1], dx, dy, p[2], Math.max(-3, Math.min(3, turn * 6))));
  }
  return out;
}

// Keeps only the first `frac` of the total ink length (for early guessing).
export function truncate(strokes, frac) {
  let L = 0;
  for (const st of strokes) for (let i = 1; i < st.length; i++) L += Math.hypot(st[i][0] - st[i - 1][0], st[i][1] - st[i - 1][1]);
  const limit = L * frac;
  const out = [];
  let acc = 0;
  for (const st of strokes) {
    const cur = [st[0]];
    for (let i = 1; i < st.length; i++) {
      const d = Math.hypot(st[i][0] - st[i - 1][0], st[i][1] - st[i - 1][1]);
      if (acc + d > limit) {
        const t = (limit - acc) / d;
        cur.push([st[i - 1][0] + (st[i][0] - st[i - 1][0]) * t, st[i - 1][1] + (st[i][1] - st[i - 1][1]) * t]);
        out.push(cur);
        return out;
      }
      acc += d;
      cur.push(st[i]);
    }
    out.push(cur);
  }
  return out;
}

// ---------- Templates ----------

function polyline(vertices, perEdge = 16) {
  const out = [];
  for (let i = 0; i < vertices.length - 1; i++) {
    const [ax, ay] = vertices[i];
    const [bx, by] = vertices[i + 1];
    for (let k = 0; k < perEdge; k++) out.push([ax + ((bx - ax) * k) / perEdge, ay + ((by - ay) * k) / perEdge]);
  }
  out.push(vertices[vertices.length - 1]);
  return out;
}

function jitterVerts(vs, rng, amt) {
  return vs.map(([x, y]) => [x + rng.gauss() * amt, y + rng.gauss() * amt]);
}

const maybeReverse = (rng, s) => (rng.next() < 0.5 ? s.slice().reverse() : s);

// Each template returns strokes in a unit-ish box, y down.
export const TEMPLATES = {
  circle: (rng) => {
    // People start circles anywhere, though most often near the top.
    const a0 = rng.next() < 0.5 ? -Math.PI / 2 + rng.uniform(-0.7, 0.7) : rng.uniform(0, Math.PI * 2);
    const dir = rng.next() < 0.5 ? 1 : -1;
    const sweep = Math.PI * 2 * rng.uniform(0.97, 1.08);
    const pts = [];
    for (let i = 0; i <= 64; i++) {
      const a = a0 + dir * sweep * (i / 64);
      pts.push([Math.cos(a), Math.sin(a)]);
    }
    return [pts];
  },
  triangle: (rng) => {
    const v = jitterVerts([[0, -1], [-1, 0.8], [1, 0.8], [0, -1]], rng, 0.06);
    return [maybeReverse(rng, polyline(v))];
  },
  square: (rng) => {
    const corners = [[-1, -1], [-1, 1], [1, 1], [1, -1]];
    const start = rng.next() < 0.75 ? 0 : rng.int(4);
    const order = [0, 1, 2, 3, 4].map((i) => corners[(start + i) % 4]);
    return [maybeReverse(rng, polyline(jitterVerts(order, rng, 0.06)))];
  },
  check: (rng) => {
    const v = jitterVerts([[-1, 0], [-0.35, 0.8], [1, -1]], rng, 0.07);
    return [polyline(v, 20)];
  },
  zigzag: (rng) => {
    const v = jitterVerts([[-1, -0.6], [-0.5, 0.6], [0, -0.6], [0.5, 0.6], [1, -0.6]], rng, 0.06);
    return [polyline(v, 12)];
  },
  spiral: (rng) => {
    const dir = rng.next() < 0.5 ? 1 : -1;
    const turns = rng.uniform(2.2, 3);
    const a0 = rng.uniform(0, Math.PI * 2);
    const pts = [];
    for (let i = 0; i <= 120; i++) {
      const t = i / 120;
      const a = a0 + dir * t * turns * Math.PI * 2;
      const r = 0.08 + 0.92 * t;
      pts.push([r * Math.cos(a), r * Math.sin(a)]);
    }
    return [pts];
  },
  star: (rng) => {
    // Pentagram in one stroke, starting bottom-left, going up to the top.
    const p = (k) => {
      const a = -Math.PI / 2 + (k * 4 * Math.PI) / 5;
      return [Math.cos(a), Math.sin(a)];
    };
    const v = [p(3), p(0), p(2), p(4), p(1), p(3)];
    return [polyline(jitterVerts(v, rng, 0.05), 12)];
  },
  wave: (rng) => {
    const pts = [];
    const ph = rng.uniform(-0.3, 0.3);
    const cycles = rng.uniform(1.8, 2.2);
    for (let i = 0; i <= 80; i++) {
      const t = i / 80;
      pts.push([-1 + 2 * t, -0.45 * Math.sin(ph + t * cycles * Math.PI * 2)]);
    }
    return [pts];
  },
  cross: (rng) => {
    const a = polyline(jitterVerts([[-1, -1], [1, 1]], rng, 0.06), 24);
    const b = polyline(jitterVerts([[1, -1], [-1, 1]], rng, 0.06), 24);
    return rng.next() < 0.8 ? [a, b] : [b, a];
  },
};

export const BUILTIN = Object.keys(TEMPLATES);

// ---------- Augmentation ----------

// Random affine transform plus smooth wobble, like a different hand.
export function augment(strokes, rng, strength = 1) {
  const rot = rng.uniform(-0.2, 0.2) * strength;
  const sx = 1 + rng.uniform(-0.25, 0.25) * strength;
  const sy = 1 + rng.uniform(-0.25, 0.25) * strength;
  const sh = rng.uniform(-0.15, 0.15) * strength;
  const c = Math.cos(rot);
  const s = Math.sin(rot);
  const waves = Array.from({ length: 3 }, () => ({ f: rng.uniform(0.5, 3), p: rng.uniform(0, 6.3), ax: rng.gauss() * 0.03 * strength, ay: rng.gauss() * 0.03 * strength }));
  let total = 0;
  for (const st of strokes) total += st.length;
  let idx = 0;
  return strokes.map((st) =>
    st.map(([x, y]) => {
      const t = idx++ / Math.max(1, total - 1);
      let wx = 0;
      let wy = 0;
      for (const w of waves) {
        wx += w.ax * Math.sin(w.p + t * w.f * Math.PI * 2);
        wy += w.ay * Math.cos(w.p + t * w.f * Math.PI * 2);
      }
      const x1 = (x + sh * y) * sx;
      const y1 = y * sy;
      return [c * x1 - s * y1 + wx, s * x1 + c * y1 + wy];
    }),
  );
}

export function synthesize(name, rng) {
  return augment(TEMPLATES[name](rng), rng);
}

// Training set: full gestures plus some partial ones (≥65% drawn) so the
// classifier can already guess while you're drawing.
export function makeDataset(names, perClass, seed, { prefixes = true } = {}) {
  const rng = new Rng(seed);
  const out = [];
  for (const name of names) {
    for (let i = 0; i < perClass; i++) {
      let strokes = synthesize(name, rng);
      if (prefixes && i % 4 === 3) strokes = truncate(strokes, rng.uniform(0.65, 0.9));
      const seq = preprocess(strokes);
      if (seq) out.push({ seq, label: name });
    }
  }
  return out;
}

// Light augmentation for the user's own examples (few-shot learning).
export function expandExamples(strokesList, label, copies, rng) {
  const out = [];
  for (const strokes of strokesList) {
    const base = preprocess(strokes);
    if (base) out.push({ seq: base, label });
    for (let k = 0; k < copies; k++) {
      let s = augment(strokes, rng, 0.45);
      if (k % 4 === 3) s = truncate(s, rng.uniform(0.7, 0.92));
      const seq = preprocess(s);
      if (seq) out.push({ seq, label });
    }
  }
  return out;
}

// Re-centre raw strokes into a compact form for storage.
export function compact(strokes) {
  const seq = strokes.flat();
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of seq) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const k = 2 / Math.max(maxX - minX, maxY - minY, 1e-9);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return strokes.map((st) => {
    const step = Math.max(1, Math.floor(st.length / 64));
    const kept = st.filter((_, i) => i % step === 0 || i === st.length - 1);
    return kept.map(([x, y]) => [Math.round((x - cx) * k * 1000) / 1000, Math.round((y - cy) * k * 1000) / 1000]);
  });
}
