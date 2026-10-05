// Tiny DOM helpers and form controls.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

let uid = 0;
const nextId = (p) => `${p}-${++uid}`;

// Range slider with a live value read-out. With `log: true` the slider moves
// in powers of ten between min and max.
export function slider({ label, min, max, step = 1, value, log = false, format = (v) => v, onInput, onChange }) {
  const id = nextId('sl');
  const toSlider = (v) => (log ? Math.log10(v) : v);
  const fromSlider = (s) => (log ? 10 ** s : s);
  const input = h('input', {
    type: 'range',
    id,
    min: toSlider(min),
    max: toSlider(max),
    step: log ? 0.1 : step,
    value: toSlider(value),
  });
  const out = h('output', { for: id }, format(value));
  const el = h('div', { class: 'control' }, h('label', { class: 'control-label', for: id }, h('span', {}, label), out), input);
  const api = {
    el,
    input,
    get value() {
      const v = fromSlider(parseFloat(input.value));
      return log ? v : Math.round(v / step) * step;
    },
    set value(v) {
      input.value = toSlider(v);
      out.textContent = format(api.value);
    },
  };
  input.addEventListener('input', () => {
    out.textContent = format(api.value);
    onInput?.(api.value);
  });
  input.addEventListener('change', () => onChange?.(api.value));
  return api;
}

export function select({ label, options, value, onChange }) {
  const id = nextId('sel');
  const sel = h(
    'select',
    { id },
    options.map(([v, text]) => h('option', { value: v, selected: v === value }, text)),
  );
  sel.addEventListener('change', () => onChange?.(sel.value));
  const el = h('div', { class: 'control' }, h('label', { class: 'control-label', for: id }, label), sel);
  return {
    el,
    get value() {
      return sel.value;
    },
    set value(v) {
      sel.value = v;
    },
  };
}

export function toggle({ label, hint, checked = false, onChange }) {
  const input = h('input', { type: 'checkbox', checked });
  input.addEventListener('change', () => onChange?.(input.checked));
  const el = h('label', { class: 'check' }, input, h('span', {}, label, hint ? h('small', {}, hint) : null));
  return {
    el,
    get checked() {
      return input.checked;
    },
    set checked(v) {
      input.checked = v;
    },
  };
}

export function button(text, onClick, cls = '') {
  return h('button', { type: 'button', class: `btn ${cls}`.trim(), onclick: onClick }, text);
}

export function stat(label, value = '–') {
  const v = h('div', { class: 'value' }, value);
  const el = h('div', { class: 'stat' }, h('div', { class: 'label' }, label), v);
  return {
    el,
    set(text, unit) {
      v.textContent = text;
      if (unit) v.append(' ', h('small', {}, unit));
    },
  };
}

export function legend(items) {
  return h(
    'div',
    { class: 'legend' },
    items.map(([color, text, dashed]) =>
      h('span', { class: 'key', style: { color: `var(${color})` } }, h('span', { class: `swatch${dashed ? ' dashed' : ''}` }), h('span', { style: { color: 'var(--text-2)' } }, text)),
    ),
  );
}

export function progressBar() {
  const fill = h('div');
  const el = h('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100 }, fill);
  return {
    el,
    set(frac) {
      fill.style.width = `${Math.round(frac * 100)}%`;
      el.setAttribute('aria-valuenow', Math.round(frac * 100));
    },
  };
}

// Probability bars for a classifier.
export function probBars() {
  const el = h('div', { class: 'bars', 'aria-live': 'polite' });
  return {
    el,
    update(labels, probs) {
      let best = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      const order = labels.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
      el.replaceChildren(
        ...order.map((i) => {
          const fill = h('div', { class: 'fill', style: { width: `${(probs[i] * 100).toFixed(1)}%` } });
          return h(
            'div',
            { class: `bar-row${i === best ? ' top' : ''}` },
            h('span', { class: 'name', title: labels[i] }, labels[i]),
            h('div', { class: 'track' }, fill),
            h('span', { class: 'pct' }, `${Math.round(probs[i] * 100)}%`),
          );
        }),
      );
    },
    clear() {
      el.replaceChildren();
    },
  };
}

export const storage = {
  get(key, fallback = null) {
    try {
      const v = localStorage.getItem(key);
      return v === null ? fallback : JSON.parse(v);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch {
      return false;
    }
  },
  remove(key) {
    try {
      localStorage.removeItem(key);
    } catch {
      /* storage unavailable */
    }
  },
};

// Inline rich text from the content tables: **bold**, *italic* and
// [text](url). Built as DOM nodes, never as HTML strings.
export function rich(text) {
  const out = [];
  const re = /\*\*(.+?)\*\*|\*(.+?)\*|\[(.+?)\]\((.+?)\)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1] !== undefined) out.push(h('b', {}, m[1]));
    else if (m[2] !== undefined) out.push(h('i', {}, m[2]));
    else out.push(h('a', { href: m[4], ...(m[4].startsWith('#') ? {} : { rel: 'noopener' }) }, m[3]));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// One <p> per string.
export function paras(list, attrs = {}) {
  return (Array.isArray(list) ? list : [list]).map((s) => h('p', attrs, rich(s)));
}
