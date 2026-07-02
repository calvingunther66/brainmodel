// Headless screenshots of out/viewer.html using the pre-installed Chromium.
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
const page = await browser.newPage({ viewport: { width: 1000, height: 800 }, deviceScaleFactor: 2 });
await page.goto(`http://localhost:${port}/viewer.html`, { waitUntil: 'networkidle' });
await page.evaluate(() => document.getElementById('spin').click()); // stop auto-rotate for a stable frame
await page.waitForTimeout(1500);
await page.screenshot({ path: join(ROOT, 'viewer_screenshot.png') });
// show skin translucently over the brain
await page.evaluate(() => document.getElementById('skin').click());
await page.waitForTimeout(1200);
await page.screenshot({ path: join(ROOT, 'viewer_skin_screenshot.png') });
console.log('wrote viewer_screenshot.png + viewer_skin_screenshot.png');
await browser.close(); server.close();
