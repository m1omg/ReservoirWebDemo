// App shell: hash routing between demos, lazy loading, language, reading
// level and theme.

import { invalidateColors } from './ui/canvas.js';
import { storage } from './ui/dom.js';
import { lang, t, setLang } from './i18n.js';
import { detectLevel, setLevel } from './ui/explain.js';
import { renderIntro } from './intro.js';

const VIEWS = ['intro', 'chaos', 'gestures', 'speech', 'water'];
const loaders = {
  chaos: () => import('./demos/chaos.js'),
  gestures: () => import('./demos/gestures.js'),
  speech: () => import('./demos/speech.js'),
  water: () => import('./demos/water.js'),
};
const mounted = {};
let current = null;

// ---------- Language & static text ----------

document.documentElement.lang = lang;
document.querySelector('meta[name="description"]').setAttribute('content', t('meta.description'));
document.querySelector('.brand').setAttribute('aria-label', t('nav.home'));
document.getElementById('brand-name').textContent = t('nav.brand');
document.getElementById('tabs').setAttribute('aria-label', t('nav.demos'));
for (const a of document.querySelectorAll('.tab')) a.textContent = t(`nav.${a.dataset.view}`);
document.getElementById('footer-text').textContent = t('common.footer');
document.getElementById('footer-source').textContent = t('common.source');

const langSwitch = document.getElementById('lang-switch');
langSwitch.setAttribute('aria-label', t('nav.language'));
for (const b of langSwitch.querySelectorAll('button')) {
  b.setAttribute('aria-pressed', String(b.dataset.lang === lang));
  b.addEventListener('click', () => setLang(b.dataset.lang));
}

// Reading level: the head script already applied it; keep it in sync with
// storage without touching the URL unless the visitor asked for it there.
setLevel(detectLevel(), { updateUrl: false });

renderIntro(document.getElementById('view-intro'));

// ---------- Routing ----------

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
  document.title = name === 'intro' ? t('meta.title') : `${t(`nav.${name}`)} · ${t('meta.title')}`;
  window.scrollTo({ top: 0 });
  if (name === 'intro') return;
  if (!mounted[name]) {
    const container = document.getElementById(`view-${name}`);
    try {
      const mod = await loaders[name]();
      mounted[name] = mod.mount(container);
    } catch (err) {
      console.error(err);
      container.textContent = t('common.loadFailed', { error: err.message });
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

// ---------- Theme: auto → light → dark ----------

const themeBtn = document.getElementById('theme-toggle');
const THEMES = ['auto', 'light', 'dark'];
function applyTheme(th) {
  if (th === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = th;
  themeBtn.textContent = { auto: '◐', light: '☀', dark: '☾' }[th];
  const label = t('nav.theme', { theme: t(`nav.themes.${th}`) });
  themeBtn.title = label;
  themeBtn.setAttribute('aria-label', label);
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
