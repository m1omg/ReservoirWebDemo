// The intro page, rendered from the content tables so it exists in every
// language and at both reading levels.

import { h, rich, paras } from './ui/dom.js';
import { t } from './i18n.js';
import { levelSwitch, levels } from './ui/explain.js';

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

function diagram() {
  const d = (k) => esc(t(`intro.diagram.${k}`));
  const wrap = h('div');
  wrap.innerHTML = `
  <svg class="diagram" viewBox="0 0 520 300" role="img" aria-labelledby="dia-title dia-desc">
    <title id="dia-title">${d('title')}</title>
    <desc id="dia-desc">${d('desc')}</desc>
    <defs>
      <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="var(--text-2)" />
      </marker>
    </defs>
    <text x="40" y="118" text-anchor="middle" font-size="13" fill="var(--text-2)">${d('input')}</text>
    <path d="M8 150 q8 -40 16 0 t16 0 t16 0 t16 0" fill="none" stroke="var(--s1)" stroke-width="2.5" stroke-linecap="round" />
    <line x1="76" y1="150" x2="118" y2="150" stroke="var(--text-2)" stroke-width="1.5" marker-end="url(#arr)" />
    <circle cx="235" cy="150" r="112" fill="var(--surface-2)" stroke="var(--border)" />
    <g stroke="var(--axis)" stroke-width="1.2" fill="none">
      <path d="M175 110 L230 85 L290 115 L300 175 L245 215 L185 190 Z" />
      <path d="M175 110 L245 150 L300 175 M230 85 L245 150 L185 190 M290 115 L245 215 M175 110 L185 190" />
      <path d="M200 150 L245 150 M230 85 L200 150 L245 215" />
    </g>
    <g>
      <circle cx="175" cy="110" r="9" fill="var(--div-neg)" />
      <circle cx="230" cy="85" r="9" fill="var(--div-pos)" />
      <circle cx="290" cy="115" r="9" fill="var(--div-mid)" stroke="var(--axis)" />
      <circle cx="300" cy="175" r="9" fill="var(--div-pos)" />
      <circle cx="245" cy="215" r="9" fill="var(--div-neg)" />
      <circle cx="185" cy="190" r="9" fill="var(--div-pos)" />
      <circle cx="245" cy="150" r="9" fill="var(--div-neg)" />
      <circle cx="200" cy="150" r="9" fill="var(--div-mid)" stroke="var(--axis)" />
    </g>
    <text x="235" y="26" text-anchor="middle" font-size="14" font-weight="600" fill="var(--text)">${d('reservoir')}</text>
    <text x="235" y="44" text-anchor="middle" font-size="12" fill="var(--text-2)">${d('reservoirSub')}</text>
    <g stroke="var(--s2)" stroke-width="1.6">
      <line x1="300" y1="175" x2="420" y2="150" />
      <line x1="290" y1="115" x2="420" y2="150" />
      <line x1="245" y1="215" x2="420" y2="150" />
    </g>
    <circle cx="425" cy="150" r="12" fill="var(--s2)" />
    <text x="430" y="212" text-anchor="middle" font-size="14" font-weight="600" fill="var(--text)">${d('readout')}</text>
    <text x="430" y="230" text-anchor="middle" font-size="12" fill="var(--text-2)">${d('readoutSub1')}</text>
    <text x="430" y="246" text-anchor="middle" font-size="12" fill="var(--text-2)">${d('readoutSub2')}</text>
    <line x1="440" y1="150" x2="480" y2="150" stroke="var(--text-2)" stroke-width="1.5" marker-end="url(#arr)" />
    <text x="495" y="155" text-anchor="middle" font-size="13" fill="var(--text-2)">${d('out')}</text>
  </svg>`;
  return wrap.firstElementChild;
}

function card(id) {
  const c = (k) => t(`intro.cards.${id}.${k}`);
  const dl = (u, hr) => h('dl', {}, h('dt', {}, t('intro.usually')), h('dd', {}, u), h('dt', {}, t('intro.here')), h('dd', {}, hr));
  return h(
    'a',
    { class: 'card demo-card', href: `#${id}` },
    h('h3', {}, c('title')),
    levels([h('p', {}, rich(c('simple'))), dl(c('usuallySimple'), c('hereSimple'))], [h('p', {}, rich(c('detailed'))), dl(c('usuallyDetailed'), c('hereDetailed'))]),
    h('span', { class: 'go' }, t('intro.open')),
  );
}

const REFS = [
  ['H. Jaeger & H. Haas (2004).', 'Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication', 'https://doi.org/10.1126/science.1091277', '*Science* 304, 78–80.', 'jaeger'],
  ['W. Maass, T. Natschläger & H. Markram (2002).', 'Real-time computing without stable states', null, '*Neural Computation* 14(11).', 'maass'],
  ['C. Fernando & S. Sojakka (2003).', 'Pattern recognition in a bucket', 'https://doi.org/10.1007/978-3-540-39432-7_63', '*ECAL 2003*, LNCS 2801.', 'bucket'],
  ['J. Pathak, Z. Lu, B. R. Hunt, M. Girvan & E. Ott (2017).', 'Using machine learning to replicate chaotic attractors and calculate Lyapunov exponents from data', 'https://doi.org/10.1063/1.5010300', '*Chaos* 27, 121102.', null],
  ['P. R. Vlachas et al. (2020).', 'Backpropagation algorithms and reservoir computing in recurrent neural networks for the forecasting of complex spatiotemporal dynamics', 'https://doi.org/10.1016/j.neunet.2020.02.016', '*Neural Networks* 126, 191–217.', 'vlachas'],
  ['D. J. Gauthier, E. Bollt, A. Griffith & W. A. S. Barbosa (2021).', 'Next generation reservoir computing', 'https://doi.org/10.1038/s41467-021-25801-2', '*Nature Communications* 12, 5564.', null],
  ['D. Verstraeten, B. Schrauwen, D. Stroobandt & J. Van Campenhout (2005).', 'Isolated word recognition with the liquid state machine: a case study', null, '*Information Processing Letters* 95, 521–528.', null],
  ['K. Vandoorne et al. (2014).', 'Experimental demonstration of reservoir computing on a silicon photonics chip', 'https://doi.org/10.1038/ncomms4541', '*Nature Communications* 5, 3541.', null],
  ['M. Lukoševičius (2012).', 'A practical guide to applying echo state networks', null, '*Neural Networks: Tricks of the Trade*, LNCS 7700.', 'luko'],
];

export function renderIntro(root) {
  const w = (k) => t(`intro.why.${k}`);
  root.replaceChildren(
    h(
      'div',
      { class: 'hero' },
      h('div', {}, h('h1', {}, t('intro.title')), levelSwitch(), levels(paras(t('intro.hero.simple')), paras(t('intro.hero.detailed')))),
      diagram(),
    ),
    h('div', { class: 'cards' }, ['chaos', 'gestures', 'speech', 'water'].map(card)),
    h(
      'div',
      { class: 'prose' },
      h('h2', {}, w('title')),
      levels(
        [...paras(w('simple')), h('p', {}, rich(w('knobsIntro.simple'))), h('ul', {}, w('knobs.simple').map((s) => h('li', {}, rich(s)))), h('p', {}, rich(w('physical.simple')))],
        [...paras(w('detailed')), h('p', {}, rich(w('knobsIntro.detailed'))), h('ul', {}, w('knobs.detailed').map((s) => h('li', {}, rich(s)))), h('p', {}, rich(w('physical.detailed')))],
      ),
      h('h2', {}, t('intro.weaker.title')),
      levels(paras(t('intro.weaker.simple')), paras(t('intro.weaker.detailed'))),
      h('h2', {}, t('intro.reading.title')),
      h(
        'ol',
        { class: 'refs' },
        REFS.map(([authors, title, url, venue, note]) =>
          h('li', {}, `${authors} `, url ? h('a', { href: url, rel: 'noopener' }, title) : title, '. ', rich(venue), note ? ` ${t(`intro.reading.notes.${note}`)}` : ''),
        ),
      ),
    ),
  );
}
