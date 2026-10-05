// Languages: picks English or Slovak, looks up strings and formats numbers.
//
// Order of precedence: ?lang= in the URL, then the visitor's saved choice,
// then Slovak if the browser prefers Slovak, otherwise English.

import en from './content/en.js';
import sk from './content/sk.js';

export const LANGS = { en, sk };
const KEY = 'rc-lang';

function fromUrl(param) {
  try {
    return new URLSearchParams(location.search).get(param);
  } catch {
    return null;
  }
}

function saved(key) {
  try {
    const v = localStorage.getItem(key);
    return v ? JSON.parse(v) : null;
  } catch {
    return null;
  }
}

export function detectLang() {
  const url = fromUrl('lang');
  if (url && LANGS[url]) return url;
  const s = saved(KEY);
  if (s && LANGS[s]) return s;
  const prefs = (typeof navigator !== 'undefined' && (navigator.languages?.length ? navigator.languages : [navigator.language])) || [];
  // The first listed language decides: a Slovak speaker with English as a
  // fallback gets Slovak, an English speaker who also reads Slovak doesn't.
  const first = String(prefs[0] || '').toLowerCase();
  return first.startsWith('sk') ? 'sk' : 'en';
}

export const lang = typeof document !== 'undefined' ? detectLang() : 'en';
const table = LANGS[lang];

function lookup(tbl, key) {
  let v = tbl;
  for (const part of key.split('.')) {
    if (v == null) return undefined;
    v = v[part];
  }
  return v;
}

function interpolate(s, params) {
  return s.replace(/\{(\w+)\}/g, (_, k) => (params && k in params ? String(params[k]) : `{${k}}`));
}

// t('chaos.title') → string (or array of strings), English as a fallback.
export function t(key, params) {
  let v = lookup(table, key);
  if (v === undefined) v = lookup(en, key);
  if (v === undefined) return key;
  if (typeof v === 'string') return interpolate(v, params);
  if (Array.isArray(v)) return v.map((s) => (typeof s === 'string' ? interpolate(s, params) : s));
  return v;
}

// Plural-aware lookup: the key holds { one, few, many, other }.
const pluralRules = typeof Intl !== 'undefined' ? new Intl.PluralRules(lang) : null;
export function tp(key, n, params = {}) {
  const forms = lookup(table, key) || lookup(en, key) || {};
  const cat = pluralRules ? pluralRules.select(n) : n === 1 ? 'one' : 'other';
  const s = forms[cat] ?? forms.other ?? forms.one ?? key;
  return interpolate(s, { n: fmtInt(n), ...params });
}

const fixedCache = new Map();
// Fixed number of decimals, in the local style (8.2 / 8,2).
export function fmt(n, digits = 0) {
  if (!Number.isFinite(n)) return n > 0 ? '∞' : n < 0 ? '−∞' : '–';
  let f = fixedCache.get(digits);
  if (!f) {
    f = new Intl.NumberFormat(lang, { minimumFractionDigits: digits, maximumFractionDigits: digits });
    fixedCache.set(digits, f);
  }
  return f.format(n);
}

// Whole numbers with local thousands separators (3,899 / 3 899).
export function fmtInt(n) {
  return fmt(Math.round(n), 0);
}

// Short trimmed decimal for axis ticks (0.5 / 0,5; 2 / 2).
export function fmtTick(n) {
  return new Intl.NumberFormat(lang, { maximumFractionDigits: 2 }).format(n);
}

export function setLang(next) {
  if (!LANGS[next] || next === lang) return;
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* storage unavailable: the URL still carries the choice */
  }
  const url = new URL(location.href);
  url.searchParams.set('lang', next);
  location.replace(url.toString());
}

// Percentages in the local style (99.4% / 99,4 %); takes a fraction.
export function fmtPct(fraction, digits = 1) {
  return new Intl.NumberFormat(lang, { style: 'percent', minimumFractionDigits: digits, maximumFractionDigits: digits }).format(fraction);
}
