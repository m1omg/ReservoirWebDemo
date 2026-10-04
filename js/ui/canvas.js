// Canvas plumbing: crisp HiDPI canvases, theme colours, plots and heatmaps.

// ---------- Theme colours ----------

let cached = null;
const listeners = new Set();

function hexToRgb(hex) {
  const m = hex.trim().match(/^#?([0-9a-f]{6})$/i);
  if (!m) return [128, 128, 128];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function colors() {
  if (cached) return cached;
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  cached = {
    page: v('--page'),
    surface: v('--surface'),
    surface2: v('--surface-2'),
    text: v('--text'),
    text2: v('--text-2'),
    muted: v('--muted'),
    grid: v('--grid'),
    axis: v('--axis'),
    s1: v('--s1'),
    s2: v('--s2'),
    s3: v('--s3'),
    s7: v('--s7'),
    good: v('--good'),
    critical: v('--critical'),
    accent: v('--accent'),
    divNeg: hexToRgb(v('--div-neg')),
    divMid: hexToRgb(v('--div-mid')),
    divPos: hexToRgb(v('--div-pos')),
    dark: (v('--surface') || '#fff').toLowerCase() === '#1a1a19',
  };
  cached.lut = divergingLut(cached.divNeg, cached.divMid, cached.divPos);
  // Sequential single-hue ramp (blue) for magnitudes such as spectrograms:
  // near-zero recedes into the surface, large values are the strongest blue.
  const seq = cached.dark ? ['#1a1a19', '#104281', '#3987e5', '#cde2fb'] : ['#fcfcfb', '#b7d3f6', '#3987e5', '#0d366b'];
  cached.seqLut = rampLut(seq.map(hexToRgb));
  return cached;
}

function rampLut(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(x));
    const f = x - k;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * f;
  }
  return lut;
}

export function invalidateColors() {
  cached = null;
  for (const cb of listeners) cb();
}

export function onThemeChange(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

if (typeof window !== 'undefined' && window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', invalidateColors);
}

// 256-entry RGB lookup: index 0 → negative pole, 128 → neutral, 255 → positive.
export function divergingLut(neg, mid, pos) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const [a, b, f] = t < 0.5 ? [mid, neg, 1 - t * 2] : [mid, pos, (t - 0.5) * 2];
    // Ease so small activations still show a tint.
    const e = Math.pow(f, 0.75);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = a[c] + (b[c] - a[c]) * e;
  }
  return lut;
}

export function withAlpha(color, alpha) {
  const [r, g, b] = hexToRgb(color);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ---------- HiDPI canvas ----------

export class Canvas2D {
  constructor(canvas, { height = null, aspect = null, minHeight = 120, maxHeight = Infinity, onResize = null } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.fixedHeight = height;
    this.aspect = aspect;
    this.minHeight = minHeight;
    this.maxHeight = maxHeight;
    this.onResize = null;
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    this.resize();
    // Attach the callback after the first sizing so callers can reference
    // the instance inside it.
    this.onResize = onResize;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas.parentElement || canvas);
  }

  resize() {
    const parent = this.canvas.parentElement;
    const w = Math.max(1, Math.floor((parent ? parent.clientWidth : this.canvas.clientWidth) || 300));
    let hgt = this.fixedHeight ?? (this.aspect ? w / this.aspect : this.canvas.clientHeight || 200);
    hgt = Math.round(Math.min(this.maxHeight, Math.max(this.minHeight, hgt)));
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    if (w === this.width && hgt === this.height && dpr === this.dpr) return;
    this.width = w;
    this.height = hgt;
    this.dpr = dpr;
    this.canvas.style.height = `${hgt}px`;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(hgt * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.onResize?.(this);
  }

  clear(color = null) {
    const { ctx } = this;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (color) {
      ctx.fillStyle = color;
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    } else ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.restore();
  }

  destroy() {
    this.ro.disconnect();
  }
}

// ---------- Plots ----------

// Draws horizontal hairline gridlines and returns a y → pixel mapping.
export function yAxis(ctx, { x, y, w, h, min, max, ticks = [], format = (v) => v, c }) {
  const toY = (v) => y + h - ((v - min) / (max - min)) * h;
  ctx.save();
  ctx.lineWidth = 1;
  ctx.font = '11px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (const t of ticks) {
    const py = Math.round(toY(t)) + 0.5;
    ctx.strokeStyle = t === 0 ? c.axis : c.grid;
    ctx.beginPath();
    ctx.moveTo(x, py);
    ctx.lineTo(x + w, py);
    ctx.stroke();
    ctx.fillStyle = c.muted;
    ctx.fillText(format(t), x - 6, py);
  }
  ctx.restore();
  return toY;
}

// Strokes one series. `get(i)` returns the value at sample i (or NaN to break).
export function strokeSeries(ctx, n, get, toX, toY, color, width = 2, dash = null) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < n; i++) {
    const v = get(i);
    if (!Number.isFinite(v)) {
      pen = false;
      continue;
    }
    const px = toX(i);
    const py = toY(v);
    if (pen) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
    pen = true;
  }
  ctx.stroke();
  ctx.restore();
}

// ---------- Heatmaps ----------

const scratch = new Map();
function scratchCanvas(w, h) {
  const key = `${w}x${h}`;
  let s = scratch.get(key);
  if (!s) {
    const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
    const cx = cv.getContext('2d');
    s = { cv, cx, img: cx.createImageData(w, h) };
    scratch.set(key, s);
  }
  return s;
}

// values(col, row) ∈ [-1, 1] mapped through the diverging LUT, drawn as a
// cols×rows grid of crisp cells into the rectangle (x, y, w, h).
export function drawHeatmap(ctx, cols, rows, values, x, y, w, h, lut, { smooth = false, scale = 1 } = {}) {
  if (cols <= 0 || rows <= 0) return;
  const { cv, cx, img } = scratchCanvas(cols, rows);
  const d = img.data;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let v = values(c, r) * scale;
      if (!Number.isFinite(v)) v = 0;
      const k = Math.max(0, Math.min(255, Math.round((v + 1) * 127.5))) * 3;
      const p = (r * cols + c) * 4;
      d[p] = lut[k];
      d[p + 1] = lut[k + 1];
      d[p + 2] = lut[k + 2];
      d[p + 3] = 255;
    }
  }
  cx.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = smooth;
  ctx.drawImage(cv, x, y, w, h);
  ctx.restore();
}

// ---------- 3-D orbit projection ----------

export function makeCamera(yaw = 0.6, pitch = 0.35) {
  return { yaw, pitch, scale: 1 };
}

// Project a point (already centred/normalised) to screen space.
export function project(cam, px, py, pz, cx, cy, s, out) {
  const cyaw = Math.cos(cam.yaw);
  const syaw = Math.sin(cam.yaw);
  const cp = Math.cos(cam.pitch);
  const sp = Math.sin(cam.pitch);
  const x1 = cyaw * px - syaw * pz;
  const z1 = syaw * px + cyaw * pz;
  const y2 = cp * py - sp * z1;
  const z2 = sp * py + cp * z1;
  const persp = 1 / (1 + z2 * 0.08);
  out[0] = cx + x1 * s * persp;
  out[1] = cy - y2 * s * persp;
  out[2] = z2;
  return out;
}

// Drag-to-rotate on a canvas. Calls onChange() after each move.
export function orbitControls(canvas, cam, onChange) {
  let drag = null;
  canvas.style.touchAction = 'none';
  canvas.style.cursor = 'grab';
  canvas.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY, yaw: cam.yaw, pitch: cam.pitch };
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
    cam.dragging = true;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    cam.yaw = drag.yaw + (e.clientX - drag.x) * 0.01;
    cam.pitch = Math.max(-1.4, Math.min(1.4, drag.pitch + (e.clientY - drag.y) * 0.01));
    onChange?.();
  });
  const end = () => {
    drag = null;
    cam.dragging = false;
    canvas.style.cursor = 'grab';
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
}
