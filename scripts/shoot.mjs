// Faithful headless renders of out/viewer.html using the pre-installed Chromium.
// Replaces the old matplotlib 3D previews (which had no z-buffer). Produces:
//   render_brain.png            three angles of the pial surface (montage)
//   render_head.png             the head / skin surface
//   viewer_screenshot.png       brain hero shot
//   viewer_skin_screenshot.png  brain inside translucent skin
import { createServer } from 'http';
import { readFile } from 'fs/promises';
import { extname, join } from 'path';
import pw from '/opt/node22/lib/node_modules/playwright/index.js';
const { chromium } = pw;

const ROOT = '/home/user/brainmodel/out';
const MIME = { '.html':'text/html', '.js':'text/javascript', '.glb':'model/gltf-binary', '.png':'image/png' };
const server = createServer(async (req, res) => {
  try {
    const p = join(ROOT, decodeURIComponent(req.url.split('?')[0]));
    const buf = await readFile(p);
    res.writeHead(200, { 'Content-Type': MIME[extname(p)] || 'application/octet-stream' });
    res.end(buf);
  } catch { res.writeHead(404); res.end('nf'); }
});
await new Promise(r => server.listen(0, r));
const port = server.address().port;

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const page = await browser.newPage({ viewport: { width: 900, height: 900 }, deviceScaleFactor: 2 });
await page.goto(`http://localhost:${port}/viewer.html`, { waitUntil: 'networkidle' });
await page.evaluate(() => document.getElementById('spin').click());   // stop auto-rotate
await page.waitForTimeout(2500);

const set = (id, on) => page.evaluate(([i, o]) => {
  const el = document.getElementById(i);
  if (el.checked !== o) el.click();
}, [id, on]);
const orbit = (az, el) => page.evaluate(([a, e]) => window.__viewer.orbit(a, e), [az, el]);
const shot = async (name) => { await page.waitForTimeout(700);
  await page.screenshot({ path: join(ROOT, name) }); };
const panel = (show) => page.evaluate((s) => {
  for (const id of ['panel', 'hint']) document.getElementById(id).style.display = s ? '' : 'none';
}, show);

// --- clean brain previews (panel hidden) ---
await panel(false);
await set('brain', true); await set('skin', false);
await orbit(180, 8);  await shot('_b_front.png');   // frontal 3/4
await orbit(90, 6);   await shot('_b_side.png');    // lateral
await orbit(180, 82); await shot('_b_top.png');     // superior

// --- head / skin (panel hidden) ---
await set('brain', false); await set('skin', true);
await page.evaluate(() => { const s = document.getElementById('op');
  s.value = 100; s.dispatchEvent(new Event('input')); });
await orbit(150, 6);  await shot('render_head.png');

// --- viewer screenshots (panel shown, as the app looks) ---
await panel(true);
await set('brain', true); await set('skin', false);
await orbit(180, 8);  await shot('viewer_screenshot.png');

// --- brain inside translucent skin ---
await set('brain', true);
await page.evaluate(() => { const s = document.getElementById('op');
  s.value = 30; s.dispatchEvent(new Event('input')); });
await orbit(150, 6);  await shot('viewer_skin_screenshot.png');

await browser.close(); server.close();

// stitch the three brain angles into render_brain.png
try {
  const sharp = (await import('/opt/node22/lib/node_modules/sharp/lib/index.js')).default;
  const imgs = ['_b_front.png', '_b_side.png', '_b_top.png'].map(f => join(ROOT, f));
  const { width: w, height: h } = await sharp(imgs[0]).metadata();
  await sharp({ create: { width: w * 3, height: h, channels: 3, background: '#0b0d12' } })
    .composite(imgs.map((p, i) => ({ input: p, left: i * w, top: 0 })))
    .png().toFile(join(ROOT, 'render_brain.png'));
  console.log('wrote render_brain.png (montage) + head/viewer shots');
} catch (e) {
  console.log('montage skipped:', e.message, '- angle PNGs kept as _b_*.png');
}
