import { polyToPath, ringBounds } from './geom';

export type PageSize = 'letter' | 'a4';
export const PAGES: Record<PageSize, { w: number; h: number; label: string }> = {
  letter: { w: 215.9, h: 279.4, label: 'US Letter' },
  a4: { w: 210, h: 297, label: 'A4' },
};

export interface PrintItem { name: string; polygon_mm: number[][]; thickness_mm?: number | null }
export interface PrintOpts { page: PageSize; landscape: boolean; margin_mm: number; overlap_mm: number }
export const PRINT_DEFAULTS: PrintOpts = { page: 'letter', landscape: false, margin_mm: 10, overlap_mm: 10 };

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!));
const f = (v: number) => (Math.round(v * 100) / 100).toString();

/** A 100 mm bar with 10 mm ticks. The whole point of this print is that it is to scale, and the one thing
 *  that reliably ruins that is a print dialog quietly set to "fit to page" — so every sheet carries a ruler
 *  to check against before trusting anything. */
function rulerSvg(x: number, y: number): string {
  const ticks = Array.from({ length: 11 }, (_, i) =>
    `<line x1="${f(x + i * 10)}" y1="${f(y)}" x2="${f(x + i * 10)}" y2="${f(y + (i % 5 === 0 ? 4 : 2.5))}" stroke="#000" stroke-width="0.3"/>`).join('');
  return `<g>
    <line x1="${f(x)}" y1="${f(y)}" x2="${f(x + 100)}" y2="${f(y)}" stroke="#000" stroke-width="0.4"/>${ticks}
    <text x="${f(x)}" y="${f(y + 9)}" font-size="3.2" font-family="Helvetica, Arial, sans-serif">this bar must measure exactly 100 mm — if it does not, reprint at 100% / "Actual size"</text>
  </g>`;
}

/**
 * A print-ready page set that reproduces outlines at 1:1 so they can be laid on the real tool.
 *
 * Anything bigger than a sheet is tiled, with `overlap_mm` of shared outline on each seam: cut one sheet on
 * its dashed trim line, lay it over the matching band of the next, and the outline runs straight through.
 * Every sheet says which tile it is and carries the 100 mm ruler.
 */
export function outlineSheetHtml(items: PrintItem[], opts: PrintOpts = PRINT_DEFAULTS): string {
  const base = PAGES[opts.page];
  const pw = opts.landscape ? base.h : base.w;
  const ph = opts.landscape ? base.w : base.h;
  const m = opts.margin_mm;
  const headH = 12;                       // strip at the top of every sheet for the label
  const footH = 12;                       // and at the bottom for the ruler
  const availW = pw - 2 * m;
  const availH = ph - 2 * m - headH - footH;
  const pages: string[] = [];
  const stamp = new Date().toLocaleString();

  for (const item of items) {
    const poly = item.polygon_mm;
    if (!poly || poly.length < 3) continue;
    const [x0, y0, x1, y1] = ringBounds(poly);
    const w = x1 - x0, h = y1 - y0;
    // tiles step by (available - overlap) so neighbours share a band of outline
    const stepX = Math.max(10, availW - opts.overlap_mm);
    const stepY = Math.max(10, availH - opts.overlap_mm);
    const nx = Math.max(1, Math.ceil((w - opts.overlap_mm) / stepX));
    const ny = Math.max(1, Math.ceil((h - opts.overlap_mm) / stepY));
    const local = poly.map(([x, y]) => [x - x0, y - y0]);
    const d = polyToPath(local);

    for (let ty = 0; ty < ny; ty++) {
      for (let tx = 0; tx < nx; tx++) {
        const ox = tx * stepX, oy = ty * stepY;
        const tile = nx * ny > 1 ? ` · sheet ${ty * nx + tx + 1} of ${nx * ny} (column ${tx + 1}, row ${ty + 1})` : '';
        const clipId = `c${ty}_${tx}_${pages.length}`;
        // seams: dashed where another sheet continues, so it is obvious which edge to cut
        const seam = (x: number, y: number, x2: number, y2: number) =>
          `<line x1="${f(x)}" y1="${f(y)}" x2="${f(x2)}" y2="${f(y2)}" stroke="#888" stroke-width="0.3" stroke-dasharray="3 2"/>`;
        pages.push(`<div class="page">
  <svg width="${f(pw)}mm" height="${f(ph)}mm" viewBox="0 0 ${f(pw)} ${f(ph)}" xmlns="http://www.w3.org/2000/svg">
    <text x="${f(m)}" y="${f(m + 5)}" font-size="4.2" font-weight="bold" font-family="Helvetica, Arial, sans-serif">${esc(item.name)}</text>
    <text x="${f(m)}" y="${f(m + 10)}" font-size="3.2" font-family="Helvetica, Arial, sans-serif">${f(w)} × ${f(h)} mm${
      item.thickness_mm ? ` · ${f(item.thickness_mm)} mm thick` : ''} · 1:1${tile} · ${esc(stamp)}</text>
    <clipPath id="${clipId}"><rect x="${f(m)}" y="${f(m + headH)}" width="${f(availW)}" height="${f(availH)}"/></clipPath>
    <g clip-path="url(#${clipId})">
      <g transform="translate(${f(m - ox)}, ${f(m + headH - oy)})">
        <path d="${d}" fill="none" stroke="#000" stroke-width="0.5"/>
      </g>
    </g>
    ${tx < nx - 1 ? seam(m + availW, m + headH, m + availW, m + headH + availH) : ''}
    ${ty < ny - 1 ? seam(m, m + headH + availH, m + availW, m + headH + availH) : ''}
    ${rulerSvg(m, ph - m - 6)}
  </svg>
</div>`);
      }
    }
  }
  if (!pages.length) return '';
  return `<!doctype html><html><head><meta charset="utf-8"><title>Outline 1:1</title><style>
  @page { size: ${f(pw)}mm ${f(ph)}mm; margin: 0; }
  html, body { margin: 0; padding: 0; background: #fff; }
  .page { width: ${f(pw)}mm; height: ${f(ph)}mm; page-break-after: always; overflow: hidden; }
  .page:last-child { page-break-after: auto; }
  @media screen { body { background: #4b5563; padding: 12px; } .page { background: #fff; margin: 0 auto 12px; box-shadow: 0 2px 10px rgba(0,0,0,.4); } }
</style></head><body onload="window.focus();window.print()">${pages.join('\n')}</body></html>`;
}

/** Open the sheets in a new tab and raise the print dialog. Returns false if a popup blocker stopped it. */
export function printOutlines(items: PrintItem[], opts: PrintOpts = PRINT_DEFAULTS): boolean {
  const html = outlineSheetHtml(items, opts);
  if (!html) return false;
  const win = window.open('', '_blank');
  if (!win) return false;
  win.document.write(html);
  win.document.close();
  return true;
}
