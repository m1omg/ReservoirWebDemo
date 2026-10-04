// Browser smoke test: serves the app, opens every demo in Chromium, checks
// for console errors, exercises each demo and saves screenshots.
//
//   node tests/e2e/smoke.mjs [--out dir] [--only chaos,water]
//
// Needs Playwright (a local or global install) and a Chromium build.

import { spawn, execSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
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

const launchOpts = {
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
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
    const start = page.locator('#view-speech button', { hasText: 'Start microphone' });
    if (await start.count()) await start.first().click();
    await page.waitForTimeout(2500);
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
