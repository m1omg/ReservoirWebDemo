// App shell: hash routing between demos, lazy loading, theme toggle.

import { invalidateColors } from './ui/canvas.js';
import { storage } from './ui/dom.js';

const VIEWS = ['intro', 'chaos', 'gestures', 'speech', 'water'];
const loaders = {
  chaos: () => import('./demos/chaos.js'),
  gestures: () => import('./demos/gestures.js'),
  speech: () => import('./demos/speech.js'),
  water: () => import('./demos/water.js'),
};
const mounted = {};
let current = null;

async function route() {
  const name = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'intro';
  if (name === current) return;
  if (current && mounted[current]) mounted[current].hide?.();
  current = name;
  for (const v of VIEWS) document.getElementById(`view-${v}`).hidden = v !== name;
  for (const a of document.querySelectorAll('.tab')) {
    if (a.dataset.view === name) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
  document.title = name === 'intro' ? 'Reservoir Computing Playground' : `${document.querySelector(`.tab[data-view="${name}"]`).textContent} · Reservoir Computing Playground`;
  window.scrollTo({ top: 0 });
  if (name === 'intro') return;
  if (!mounted[name]) {
    const container = document.getElementById(`view-${name}`);
    try {
      const mod = await loaders[name]();
      mounted[name] = mod.mount(container);
    } catch (err) {
      console.error(err);
      container.textContent = `Sorry, this demo failed to load: ${err.message}`;
      return;
    }
  }
  if (current === name) mounted[name].show?.();
}

// Pause animations when the page is hidden; resume when it comes back.
document.addEventListener('visibilitychange', () => {
  const m = current && mounted[current];
  if (!m) return;
  if (document.hidden) m.hide?.();
  else m.show?.();
});

// Theme: auto → light → dark.
const themeBtn = document.getElementById('theme-toggle');
const THEMES = ['auto', 'light', 'dark'];
function applyTheme(t) {
  if (t === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
  themeBtn.textContent = { auto: '◐', light: '☀', dark: '☾' }[t];
  themeBtn.title = `Colour theme: ${t} (click to change)`;
  themeBtn.setAttribute('aria-label', `Colour theme: ${t}`);
  invalidateColors();
}
let theme = storage.get('rc-theme', 'auto');
if (!THEMES.includes(theme)) theme = 'auto';
applyTheme(theme);
themeBtn.addEventListener('click', () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  storage.set('rc-theme', theme);
  applyTheme(theme);
});

window.addEventListener('hashchange', route);
route();
