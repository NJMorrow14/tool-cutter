#!/usr/bin/env node
// Renders the admin's /brochure/print page to a two-page US-Letter PDF with headless Chrome, writes it to
// brand/toolfoam-brochure.pdf and (unless --no-upload) replaces the brochure in the running admin's database.
// Usage: node scripts/render-brochure.mjs [--admin http://localhost:3000] [--no-upload]
//   The admin must be running with the NEXT_PUBLIC_WEB_URL you want baked into the QR code.
import { createRequire } from 'node:module';
import { writeFileSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(root, 'apps', 'admin', 'package.json'));
const puppeteer = require('puppeteer-core');

const args = process.argv.slice(2);
const admin = args.includes('--admin') ? args[args.indexOf('--admin') + 1] : 'http://localhost:3000';
const upload = !args.includes('--no-upload');
const chrome = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const out = path.join(root, 'brand', 'toolfoam-brochure.pdf');

const browser = await puppeteer.launch({ executablePath: chrome, headless: true });
const page = await browser.newPage();
await page.goto(`${admin}/brochure/print`, { waitUntil: 'networkidle0' });
await page.evaluate(async () => { await document.fonts.ready; await Promise.all(Array.from(document.images).map((i) => i.complete ? null : new Promise((r) => { i.onload = i.onerror = r; }))); });
const pdf = await page.pdf({ format: 'Letter', printBackground: true, preferCSSPageSize: true, margin: { top: 0, right: 0, bottom: 0, left: 0 } });
await browser.close();
writeFileSync(out, pdf);
console.log(`wrote ${out} (${(pdf.length / 1024).toFixed(0)} KB)`);

if (upload) {
  const fd = new FormData();
  fd.append('file', new Blob([readFileSync(out)], { type: 'application/pdf' }), 'toolfoam-brochure.pdf');
  const res = await fetch(`${admin}/api/brochure`, { method: 'POST', body: fd, headers: process.env.ADMIN_COOKIE ? { cookie: process.env.ADMIN_COOKIE } : {} });
  console.log(res.ok ? 'uploaded to the admin brochure slot' : `upload failed: HTTP ${res.status} (set ADMIN_COOKIE if the admin is password-walled)`);
}
