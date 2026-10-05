// Two reading levels for every description: "simple" (explain like I'm five)
// and "detailed". The level lives on <html data-explain>, CSS hides the
// other variant, and ?explain= in the URL makes a link open in either mode.

import { h } from './dom.js';
import { t } from '../i18n.js';

export const LEVELS = ['simple', 'detailed'];
const KEY = 'rc-explain';
const switches = new Set();

export function detectLevel() {
  try {
    const url = new URLSearchParams(location.search).get('explain');
    if (LEVELS.includes(url)) return url;
  } catch {
    /* no URL */
  }
  try {
    const v = JSON.parse(localStorage.getItem(KEY));
    if (LEVELS.includes(v)) return v;
  } catch {
    /* storage unavailable */
  }
  return 'simple';
}

export function currentLevel() {
  const v = document.documentElement.dataset.explain;
  return LEVELS.includes(v) ? v : 'simple';
}

export function setLevel(level, { updateUrl = true } = {}) {
  if (!LEVELS.includes(level)) return;
  document.documentElement.dataset.explain = level;
  try {
    localStorage.setItem(KEY, JSON.stringify(level));
  } catch {
    /* storage unavailable */
  }
  if (updateUrl) {
    try {
      const url = new URL(location.href);
      url.searchParams.set('explain', level);
      history.replaceState(history.state, '', url.toString());
    } catch {
      /* URL can't be changed here (e.g. a sandboxed preview) */
    }
  }
  for (const sync of switches) sync();
}

// Segmented "Simple | Detailed" control. Every copy stays in sync.
export function levelSwitch() {
  const buttons = LEVELS.map((level) =>
    h('button', { type: 'button', class: 'seg', 'data-value': level, onclick: () => setLevel(level) }, t(`common.level.${level}`)),
  );
  const el = h('div', { class: 'level-switch', role: 'group', 'aria-label': t('common.level.label') }, h('span', { class: 'level-label' }, t('common.level.label')), h('div', { class: 'segmented' }, buttons));
  const sync = () => {
    const cur = currentLevel();
    for (const b of buttons) b.setAttribute('aria-pressed', String(b.dataset.value === cur));
  };
  switches.add(sync);
  sync();
  return el;
}

// Both variants of a piece of content; CSS shows the active one.
export function levels(simple, detailed, tag = 'div') {
  return [h(tag, { 'data-level': 'simple' }, simple), h(tag, { 'data-level': 'detailed' }, detailed)];
}
