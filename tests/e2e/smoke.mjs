// Browser smoke test: serves the app, opens every demo in Chromium, checks
// for console errors, exercises each demo and saves screenshots.
//
//   node tests/e2e/smoke.mjs [--out dir] [--only chaos,water]
//
// Needs Playwright (a local or global install) and a Chromium build.

import { spawn, execSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdirSync, writeFileSync } from 'node:fs';
import { synthWord, SYNTH_WORDS } from '../helpers/synth-speech.js';
import { Rng } from '../../js/core/rng.js';
import en from '../../js/content/en.js';
import sk from '../../js/content/sk.js';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = process.argv.slice(2);
const outDir = args.includes('--out') ? args[args.indexOf('--out') + 1] : join(root, 'tests', 'e2e', 'screenshots');
const only = args.includes('--only') ? args[args.indexOf('--only') + 1].split(',') : null;
mkdirSync(outDir, { recursive: true });

async function loadPlaywright() {
  try {
    return await import('playwright');
  } catch {
    const globalRoot = execSync('npm root -g').toString().trim();
    return createRequire(join(globalRoot, 'noop.js'))('playwright');
  }
}

const { chromium } = await loadPlaywright();
const port = 8000 + Math.floor(Math.random() * 900);
const server = spawn('python3', ['-m', 'http.server', String(port), '--bind', '127.0.0.1'], { cwd: root, stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 800));
const base = `http://127.0.0.1:${port}/`;

// A fake microphone that "says" synthetic words over and over.
function writeWav(path) {
  const sr = 16000;
  const rng = new Rng(5);
  const parts = [];
  for (let k = 0; k < 8; k++) parts.push(synthWord(SYNTH_WORDS[k % SYNTH_WORDS.length], rng, sr));
  const total = parts.reduce((a, p) => a + p.length, 0);
  const buf = Buffer.alloc(44 + total * 2);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + total * 2, 4);
  buf.write('WAVEfmt ', 8);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sr, 24);
  buf.writeUInt32LE(sr * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(total * 2, 40);
  let o = 44;
  for (const p of parts) for (const v of p) (buf.writeInt16LE(Math.max(-32767, Math.min(32767, Math.round(v * 32767))), o), (o += 2));
  writeFileSync(path, buf);
}
const wav = join(outDir, 'fake-mic.wav');
writeWav(wav);

const launchOpts = {
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${wav}`, '--autoplay-policy=no-user-gesture-required'],
};
if (process.env.CHROMIUM_PATH) launchOpts.executablePath = process.env.CHROMIUM_PATH;
const browser = await chromium.launch(launchOpts);

const failures = [];
const want = (name) => !only || only.includes(name);

// Visible = rendered and not hidden by the reading-level CSS.
const visible = (page, sel) => page.evaluate((q) => [...document.querySelectorAll(q)].some((el) => el.offsetParent !== null), sel);

// Moves a slider away from its value, presses the demo's reset button and
// checks the slider is back where it started.
async function checkReset(page, view, L, tag) {
  const range = page.locator(`#view-${view} input[type=range]`).first();
  const before = await range.inputValue();
  await range.evaluate((el) => {
    el.value = el.max;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.waitForTimeout(300);
  await page.locator(`#view-${view} button`, { hasText: L.common.reset.replace('↺ ', '') }).first().click();
  await page.waitForTimeout(300);
  const after = await range.inputValue();
  if (after !== before) failures.push(`${tag}: ${view} reset left the first slider at ${after} (expected ${before})`);
  const note = await page.locator(`#view-${view} .panel-foot .hint`).first().textContent();
  if (note !== L.common.resetDone) failures.push(`${tag}: ${view} reset note is "${note}"`);
}

// English words that must not appear in the Slovak demo pages.
const ENGLISH_UI = /\b(Training|Neurons|Reset|Simple|Detailed|Draw|Record|Accuracy|Microphone|Leak rate|Spectral radius|Settings|Pause|Splash|Forecast|Readout|Dream)\b/;

async function session({ width, height, scheme, tag, locale = 'en-US' }) {
  const L = locale.startsWith('sk') ? sk : en;
  const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: scheme, deviceScaleFactor: 1, locale });
  await ctx.grantPermissions(['microphone'], { origin: base });
  const page = await ctx.newPage();
  const errors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(String(e)));
  const shot = (name) => page.screenshot({ path: join(outDir, `${tag}-${name}.png`), fullPage: true });
  const noEnglish = async (view) => {
    if (L !== sk) return;
    const text = await page.locator(`#view-${view}`).innerText();
    const m = text.match(ENGLISH_UI);
    if (m) failures.push(`${tag}: English word "${m[0]}" left in the Slovak ${view} page`);
  };

  await page.goto(base + '#intro');
  await page.waitForTimeout(300);
  const lang = await page.evaluate(() => document.documentElement.lang);
  if (lang !== (L === sk ? 'sk' : 'en')) failures.push(`${tag}: page language is "${lang}"`);
  const title = await page.title();
  if (title !== L.meta.title) failures.push(`${tag}: title is "${title}"`);

  if (want('intro')) {
    // Simple is the default; Detailed is one click away and lands in the URL.
    if (!(await visible(page, '#view-intro [data-level=simple]')) || (await visible(page, '#view-intro [data-level=detailed]'))) failures.push(`${tag}: intro does not open in Simple mode`);
    await shot('intro');
    await page.locator('.level-switch .seg', { hasText: L.common.level.detailed }).first().click();
    if (!page.url().includes('explain=detailed')) failures.push(`${tag}: URL lacks explain=detailed after switching`);
    if (!(await visible(page, '#view-intro [data-level=detailed]'))) failures.push(`${tag}: Detailed text not shown after switching`);
    await page.reload();
    await page.waitForTimeout(300);
    if ((await page.evaluate(() => document.documentElement.dataset.explain)) !== 'detailed') failures.push(`${tag}: Detailed choice lost on reload`);
    await shot('intro-detailed');
    await page.locator('.level-switch .seg', { hasText: L.common.level.simple }).first().click();
    const pressed = await page.evaluate(() => [...document.querySelectorAll('.level-switch .seg[aria-pressed=true]')].map((b) => b.dataset.value));
    if (pressed.some((v) => v !== 'simple')) failures.push(`${tag}: level switches out of sync: ${pressed}`);
  }

  if (want('chaos')) {
    await page.goto(base + '#chaos');
    await page.waitForFunction(() => /\d/.test(document.querySelector('#view-chaos .stat .value')?.textContent || ''), null, { timeout: 30000 });
    await page.waitForTimeout(4500);
    if (L === sk) {
      const valid = await page.locator('#view-chaos .stat .value').nth(1).textContent();
      if (!/\d,\d/.test(valid)) failures.push(`${tag}: Slovak forecast stat lacks a decimal comma: "${valid}"`);
    }
    await noEnglish('chaos');
    await shot('chaos');
    await checkReset(page, 'chaos', L, tag);
  }

  if (want('gestures')) {
    await page.goto(base + '#gestures');
    await page.waitForFunction(() => document.querySelector('#view-gestures [data-ready]'), null, { timeout: 30000 });
    const box = await page.locator('#view-gestures canvas.draw').boundingBox();
    // Draw a circle.
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    const r = Math.min(box.width, box.height) * 0.3;
    await page.mouse.move(cx + r, cy);
    await page.mouse.down();
    for (let i = 1; i <= 48; i++) {
      const a = (-i / 48) * Math.PI * 2;
      await page.mouse.move(cx + r * Math.cos(a), cy + r * Math.sin(a));
    }
    await page.mouse.up();
    await page.waitForTimeout(1200);
    const verdict = await page.locator('#view-gestures .verdict').textContent();
    if (!verdict.includes(L.gestures.names.circle)) failures.push(`${tag}: gesture circle recognised as "${verdict}"`);
    await noEnglish('gestures');
    await shot('gestures');
    await checkReset(page, 'gestures', L, tag);
  }

  if (want('speech')) {
    await page.goto(base + '#speech');
    await page.waitForTimeout(500);
    await page.locator('#view-speech button', { hasText: L.speech.start }).first().click();
    await page.locator('#view-speech button', { hasText: L.speech.stop }).first().waitFor({ state: 'visible', timeout: 10000 });
    // Arm recording for the first word and wait for the fake mic to say something.
    await page.locator('#view-speech .word-row button', { hasText: L.speech.record }).first().click();
    try {
      await page.waitForFunction(() => document.querySelectorAll('#view-speech .dots i.on').length >= 2, null, { timeout: 15000 });
    } catch {
      failures.push(`${tag}: speech demo never detected an utterance from the fake microphone`);
    }
    await page.waitForTimeout(500);
    await noEnglish('speech');
    await shot('speech');
    await checkReset(page, 'speech', L, tag);
  }

  if (want('water')) {
    await page.goto(base + '#water');
    await page.waitForFunction(() => /%/.test(document.querySelector('#view-water .stat .value')?.textContent || ''), null, { timeout: 30000 });
    await page.waitForTimeout(3000);
    await noEnglish('water');
    await shot('water');
    await checkReset(page, 'water', L, tag);
  }

  // A shared link can force the reading level...
  await page.goto(base + '?explain=detailed#water');
  await page.waitForTimeout(500);
  if (!(await visible(page, '#view-water .view-head [data-level=detailed]'))) failures.push(`${tag}: ?explain=detailed#water did not open in Detailed mode`);
  // ...and the language.
  if (L === sk) {
    await page.goto(base + '?lang=en#intro');
    await page.waitForTimeout(300);
    if ((await page.evaluate(() => document.documentElement.lang)) !== 'en') failures.push(`${tag}: ?lang=en did not override the Slovak locale`);
    await page.locator('#lang-switch button', { hasText: 'SK' }).click();
    await page.waitForLoadState('load');
    await page.waitForTimeout(300);
    if ((await page.evaluate(() => document.documentElement.lang)) !== 'sk') failures.push(`${tag}: the SK switch did not switch back to Slovak`);
  }

  if (errors.length) failures.push(`${tag}: console errors:\n  ${errors.join('\n  ')}`);
  await ctx.close();
}

try {
  await session({ width: 1360, height: 900, scheme: 'light', tag: 'desktop-light' });
  await session({ width: 1360, height: 900, scheme: 'dark', tag: 'desktop-dark' });
  await session({ width: 390, height: 844, scheme: 'light', tag: 'phone-light' });
  await session({ width: 1360, height: 900, scheme: 'light', tag: 'sk-desktop-light', locale: 'sk-SK' });
  await session({ width: 390, height: 844, scheme: 'dark', tag: 'sk-phone-dark', locale: 'sk-SK' });
} catch (e) {
  failures.push(String(e.stack || e));
} finally {
  await browser.close();
  server.kill();
}

if (failures.length) {
  console.error('FAIL\n' + failures.join('\n'));
  process.exit(1);
}
console.log(`OK — screenshots in ${outDir}`);
