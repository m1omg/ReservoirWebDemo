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

async function session({ width, height, scheme, tag }) {
  const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: scheme, deviceScaleFactor: 1 });
  await ctx.grantPermissions(['microphone'], { origin: base });
  const page = await ctx.newPage();
  const errors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(String(e)));
  const shot = (name) => page.screenshot({ path: join(outDir, `${tag}-${name}.png`), fullPage: true });

  await page.goto(base + '#intro');
  await page.waitForTimeout(300);
  if (want('intro')) await shot('intro');

  if (want('chaos')) {
    await page.goto(base + '#chaos');
    await page.waitForFunction(() => /ms|s$/.test(document.querySelector('#view-chaos .stat .value')?.textContent || ''), null, { timeout: 30000 });
    await page.waitForTimeout(4500);
    await shot('chaos');
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
    if (!/circle/i.test(verdict)) failures.push(`${tag}: gesture circle recognised as "${verdict}"`);
    await shot('gestures');
  }

  if (want('speech')) {
    await page.goto(base + '#speech');
    await page.waitForTimeout(500);
    await page.locator('#view-speech button', { hasText: 'Start microphone' }).first().click();
    await page.locator('#view-speech button', { hasText: 'Stop' }).first().waitFor({ state: 'visible', timeout: 10000 });
    // Arm recording for the first word and wait for the fake mic to say something.
    await page.locator('#view-speech .word-row button', { hasText: 'Record' }).first().click();
    try {
      await page.waitForFunction(() => document.querySelectorAll('#view-speech .dots i.on').length >= 2, null, { timeout: 15000 });
    } catch {
      failures.push(`${tag}: speech demo never detected an utterance from the fake microphone`);
    }
    await page.waitForTimeout(500);
    await shot('speech');
  }

  if (want('water')) {
    await page.goto(base + '#water');
    await page.waitForFunction(() => /%/.test(document.querySelector('#view-water .stat .value')?.textContent || ''), null, { timeout: 30000 });
    await page.waitForTimeout(3000);
    await shot('water');
  }

  if (errors.length) failures.push(`${tag}: console errors:\n  ${errors.join('\n  ')}`);
  await ctx.close();
}

try {
  await session({ width: 1360, height: 900, scheme: 'light', tag: 'desktop-light' });
  await session({ width: 1360, height: 900, scheme: 'dark', tag: 'desktop-dark' });
  await session({ width: 390, height: 844, scheme: 'light', tag: 'phone-light' });
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
