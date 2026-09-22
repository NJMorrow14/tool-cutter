import { TOOL_COLORS, type ShapeKind, type ShapeSpec, type SourceKind, type Tool, type ToolResult } from './types';

let counter = 0;
export function uid(prefix = 't'): string {
  counter += 1;
  return `${prefix}${Date.now().toString(36)}${counter.toString(36)}`;
}

export function toolColor(index: number): string {
  return TOOL_COLORS[index % TOOL_COLORS.length];
}

/** Order 4 points as TL, TR, BR, BL (y down). Mirrors backend calibration.order_corners. */
export function orderCorners(pts: number[][]): number[][] {
  if (pts.length !== 4) return pts;
  const cx = pts.reduce((s, p) => s + p[0], 0) / 4;
  const cy = pts.reduce((s, p) => s + p[1], 0) / 4;
  const ring = [...pts].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
  let start = 0;
  let best = Infinity;
  ring.forEach((p, i) => {
    const s = p[0] + p[1];
    if (s < best) {
      best = s;
      start = i;
    }
  });
  return [...ring.slice(start), ...ring.slice(0, start)];
}

export function edgeLengths(ordered: number[][]): { horiz: number; vert: number } {
  const d = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const [tl, tr, br, bl] = ordered;
  return { horiz: (d(tl, tr) + d(bl, br)) / 2, vert: (d(tl, bl) + d(tr, br)) / 2 };
}

export function polygonCentroid(poly: number[][]): { x: number; y: number } {
  let area = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % poly.length];
    const f = x0 * y1 - x1 * y0;
    area += f;
    cx += (x0 + x1) * f;
    cy += (y0 + y1) * f;
  }
  if (Math.abs(area) < 1e-9) {
    const n = poly.length || 1;
    return { x: poly.reduce((s, p) => s + p[0], 0) / n, y: poly.reduce((s, p) => s + p[1], 0) / n };
  }
  area *= 0.5;
  return { x: cx / (6 * area), y: cy / (6 * area) };
}

export function ringsToPath(rings: number[][][]): string {
  return rings
    .filter((r) => r.length >= 3)
    .map((r) => `M${r.map((p) => `${fmt(p[0])} ${fmt(p[1])}`).join(' L')} Z`)
    .join(' ');
}

export function polyToPath(poly: number[][]): string {
  if (poly.length < 3) return '';
  return `M${poly.map((p) => `${fmt(p[0])} ${fmt(p[1])}`).join(' L')} Z`;
}

export function fmt(v: number, digits = 2): string {
  return Number.isFinite(v) ? String(Math.round(v * 10 ** digits) / 10 ** digits) : '0';
}

export function fmtMm(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return `${v.toFixed(digits)} mm`;
}

/** Same palette + assignment rule as backend exporters.depth_color_map. */
const DEPTH_PALETTE = [
  '#000000', '#0000ff', '#ff0000', '#00b000', '#ff8000', '#a000c0',
  '#00a0a0', '#c08000', '#ff00ff', '#806040', '#4060ff', '#c00060',
];

export function depthColorMap(depths: (number | null)[]): Map<number | null, string> {
  const uniq = Array.from(new Set(depths.filter((d): d is number => d !== null).map((d) => Math.round(d * 100) / 100))).sort(
    (a, b) => a - b,
  );
  const map = new Map<number | null, string>();
  uniq.forEach((d, i) => map.set(d, uniq.length > 1 ? DEPTH_PALETTE[(i + 1) % DEPTH_PALETTE.length] : '#ff0000'));
  map.set(null, '#ff0000');
  return map;
}

export function depthKey(d: number | null): number | null {
  return d === null ? null : Math.round(d * 100) / 100;
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}


/** Build a UI Tool from a backend ToolResult. */
export function toolFromResult(r: ToolResult, index: number, source: SourceKind, name?: string): Tool {
  return {
    id: r.id,
    session_id: r.session_id,
    source,
    name: name ?? r.name ?? `Tool ${index + 1}`,
    color: toolColor(index),
    points: r.points.map((p) => ({ x: p.x, y: p.y, label: p.label ? 'pos' : 'neg' })),
    box: r.box,
    polygon_px: r.polygon_px,
    polygon_mm: r.polygon_mm,
    area_mm2: r.area_mm2,
    measured_thickness_mm: r.measured_thickness_mm,
    image_url: r.image_url,
    image_source: r.image_source,
    include: true,
    clearance_mm: null,
    depth_mm: null,
    rotation_deg: 0,
    offset_mm: { x: 0, y: 0 },
    notch: null,
    pending: false,
    error: null,
    auto_polygon_px: r.polygon_px,
    edited: false,
  };
}

/** Bounding box of a polygon shifted by an offset (mm). */
export function polyBounds(poly: number[][], off = { x: 0, y: 0 }): { minX: number; minY: number; maxX: number; maxY: number } | null {
  if (!poly.length) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of poly) {
    minX = Math.min(minX, x + off.x); maxX = Math.max(maxX, x + off.x);
    minY = Math.min(minY, y + off.y); maxY = Math.max(maxY, y + off.y);
  }
  return { minX, minY, maxX, maxY };
}

/** Offset that places a freshly imported tool below everything already on the mat. */
export function placementOffset(poly: number[][], existing: Tool[]): { x: number; y: number } {
  const b = polyBounds(poly);
  if (!b) return { x: 0, y: 0 };
  let nextY = 10;
  for (const t of existing) {
    const eb = polyBounds(t.polygon_mm, t.offset_mm);
    if (eb) nextY = Math.max(nextY, eb.maxY + 10);
  }
  return { x: Math.round((10 - b.minX) * 10) / 10, y: Math.round((nextY - b.minY) * 10) / 10 };
}

// ------------------------------------------------------------------ outline editing helpers (px or mm, any unit)

/** Unsigned polygon area (shoelace). */
export function polygonArea(poly: number[][]): number {
  let a = 0;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) a += poly[j][0] * poly[i][1] - poly[i][0] * poly[j][1];
  return Math.abs(a) / 2;
}

/** Distance from p to segment ab, with the parameter t of the closest point. */
export function pointSegment(p: number[], a: number[], b: number[]): { d: number; t: number; q: number[] } {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2, 0, 1) : 0;
  const q = [a[0] + t * dx, a[1] + t * dy];
  return { d: Math.hypot(p[0] - q[0], p[1] - q[1]), t, q };
}

/** Closed ring re-sampled to roughly `step` spacing along its perimeter (keeps the shape, evens the vertices). */
export function resampleRing(poly: number[][], step: number): number[][] {
  if (poly.length < 3 || step <= 0) return poly;
  const n = poly.length;
  let per = 0;
  for (let i = 0; i < n; i++) per += Math.hypot(poly[(i + 1) % n][0] - poly[i][0], poly[(i + 1) % n][1] - poly[i][1]);
  const count = Math.max(8, Math.round(per / step));
  const target = per / count;
  const out: number[][] = [];
  let acc = 0;
  let i = 0;
  let a = poly[0];
  let b = poly[1 % n];
  let segLen = Math.hypot(b[0] - a[0], b[1] - a[1]);
  let pos = 0; // distance travelled along the current segment
  out.push([a[0], a[1]]);
  while (out.length < count) {
    const need = target - acc;
    if (pos + need <= segLen) {
      pos += need;
      acc = 0;
      const t = segLen > 0 ? pos / segLen : 0;
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    } else {
      acc += segLen - pos;
      i += 1;
      if (i >= n) break;
      a = poly[i % n];
      b = poly[(i + 1) % n];
      segLen = Math.hypot(b[0] - a[0], b[1] - a[1]);
      pos = 0;
    }
  }
  return out;
}

/** Ramer–Douglas–Peucker on a closed ring. */
export function simplifyRing(poly: number[][], eps: number): number[][] {
  if (poly.length < 4) return poly;
  // open the ring at the two points farthest apart so both halves get simplified properly
  let iA = 0, iB = 0, best = -1;
  for (let i = 0; i < poly.length; i += Math.max(1, Math.floor(poly.length / 64))) {
    for (let j = i + 1; j < poly.length; j += Math.max(1, Math.floor(poly.length / 64))) {
      const d = Math.hypot(poly[i][0] - poly[j][0], poly[i][1] - poly[j][1]);
      if (d > best) { best = d; iA = i; iB = j; }
    }
  }
  const chain1 = poly.slice(iA, iB + 1);
  const chain2 = [...poly.slice(iB), ...poly.slice(0, iA + 1)];
  const s1 = rdp(chain1, eps), s2 = rdp(chain2, eps);
  const out = [...s1.slice(0, -1), ...s2.slice(0, -1)];
  return out.length >= 3 ? out : poly;
}

function rdp(pts: number[][], eps: number): number[][] {
  if (pts.length <= 2) return pts;
  let maxD = -1, idx = 0;
  const a = pts[0], b = pts[pts.length - 1];
  for (let i = 1; i < pts.length - 1; i++) {
    const { d } = pointSegment(pts[i], a, b);
    if (d > maxD) { maxD = d; idx = i; }
  }
  if (maxD <= eps) return [a, b];
  return [...rdp(pts.slice(0, idx + 1), eps).slice(0, -1), ...rdp(pts.slice(idx), eps)];
}

/** Gaussian smoothing along a closed ring; `sigma` in vertices. Optional weight per vertex (0..1) limits the effect. */
export function smoothRing(poly: number[][], sigma: number, weight?: (i: number) => number): number[][] {
  const n = poly.length;
  if (n < 4 || sigma <= 0) return poly;
  const k = Math.min(Math.floor(n / 2) - 1, Math.ceil(sigma * 3));
  const w: number[] = [];
  for (let d = -k; d <= k; d++) w.push(Math.exp(-(d * d) / (2 * sigma * sigma)));
  const ws = w.reduce((s, v) => s + v, 0);
  return poly.map((p, i) => {
    let x = 0, y = 0;
    for (let d = -k; d <= k; d++) {
      const q = poly[(i + d + n) % n];
      x += q[0] * w[d + k]; y += q[1] * w[d + k];
    }
    const f = weight ? weight(i) : 1;
    return [p[0] + (x / ws - p[0]) * f, p[1] + (y / ws - p[1]) * f];
  });
}

/** Bounding box [minX, minY, maxX, maxY]. */
export function ringBounds(poly: number[][]): number[] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of poly) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); }
  return [minX, minY, maxX, maxY];
}


// ------------------------------------------------------------------ drawn primitives

/** Outline (mm, y down, bbox at the origin) of a simple shape. */
export function shapePolygon(spec: ShapeSpec): number[][] {
  if (spec.kind === 'poly') return spec.points && spec.points.length >= 3 ? spec.points : [[0, 0], [spec.w_mm, 0], [spec.w_mm, spec.h_mm], [0, spec.h_mm]];
  const w = Math.max(1, spec.w_mm), h = Math.max(1, spec.h_mm);
  const pts: number[][] = [];
  const arc = (cx: number, cy: number, r: number, a0: number, a1: number, n: number) => {
    for (let i = 0; i <= n; i++) { const a = a0 + ((a1 - a0) * i) / n; pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); }
  };
  if (spec.kind === 'circle') {
    const r = w / 2, n = Math.max(48, Math.round(Math.PI * w / 1.0));
    for (let i = 0; i < n; i++) { const a = (2 * Math.PI * i) / n; pts.push([r + r * Math.cos(a), r + r * Math.sin(a)]); }
    return pts;
  }
  if (spec.kind === 'hex') {
    // regular hexagon, w = flat-to-flat width (socket / nut style)
    const R = w / Math.sqrt(3), cx = w / 2, cy = R;
    for (let i = 0; i < 6; i++) { const a = Math.PI / 6 + (Math.PI / 3) * i; pts.push([cx + R * Math.cos(a), cy + R * Math.sin(a)]); }
    return pts;
  }
  const r = spec.kind === 'slot' ? Math.min(w, h) / 2 : Math.min(Math.max(0, spec.r_mm), w / 2, h / 2);
  if (r <= 0.05) return [[0, 0], [w, 0], [w, h], [0, h]];
  const n = Math.max(6, Math.round((Math.PI / 2) * r / 1.0));
  arc(w - r, r, r, -Math.PI / 2, 0, n);
  arc(w - r, h - r, r, 0, Math.PI / 2, n);
  arc(r, h - r, r, Math.PI / 2, Math.PI, n);
  arc(r, r, r, Math.PI, 1.5 * Math.PI, n);
  return pts;
}

export function shapeName(spec: ShapeSpec): string {
  const f = (v: number) => (Math.round(v * 10) / 10).toString();
  if (spec.kind === 'circle') return `Circle ⌀${f(spec.w_mm)}`;
  if (spec.kind === 'hex') return `Hex ${f(spec.w_mm)}`;
  if (spec.kind === 'slot') return `Slot ${f(spec.w_mm)}×${f(spec.h_mm)}`;
  if (spec.kind === 'poly') return `Polygon ${f(spec.w_mm)}×${f(spec.h_mm)}`;
  return `Rect ${f(spec.w_mm)}×${f(spec.h_mm)}${spec.r_mm > 0 ? ` r${f(spec.r_mm)}` : ''}`;
}

/** A new shape tool at a given mat position (mm, its bbox top-left); placed below everything when `at` is omitted. */
export function shapeTool(spec: ShapeSpec, thickness_mm: number, existing: Tool[], at?: { x: number; y: number }): Tool {
  const poly = shapePolygon(spec);
  const t: Tool = {
    id: uid('shape'), session_id: '', source: 'shape', name: shapeName(spec), color: toolColor(existing.length),
    points: [], box: null, polygon_px: [], polygon_mm: poly, area_mm2: polygonArea(poly), measured_thickness_mm: thickness_mm,
    include: true, clearance_mm: null, depth_mm: null, rotation_deg: 0, offset_mm: { x: 0, y: 0 }, notch: null, shape: spec,
  };
  t.offset_mm = at ? { x: Math.round(at.x * 10) / 10, y: Math.round(at.y * 10) / 10 } : placementOffset(poly, existing);
  return t;
}

/** Shape spec from a drag on the sheet (mm). Rect/slot: corner to corner. Circle/hex: centre to edge (Alt = corner to
 *  corner). Shift constrains rect/slot to a square. Returns the spec and where its bbox top-left lands. */
export function shapeFromDrag(kind: Exclude<ShapeKind, 'poly'>, a: { x: number; y: number }, b: { x: number; y: number }, mods: { shift: boolean; alt: boolean }, radius_mm = 4): { spec: ShapeSpec; at: { x: number; y: number } } | null {
  let w = Math.abs(b.x - a.x), h = Math.abs(b.y - a.y);
  if (kind === 'circle' || kind === 'hex') {
    if (mods.alt) {
      const d = Math.max(w, h);
      if (d < 2) return null;
      return { spec: { kind, w_mm: d, h_mm: d, r_mm: 0 }, at: { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y) } };
    }
    const r = Math.hypot(b.x - a.x, b.y - a.y);
    if (r < 1) return null;
    const d = 2 * r;
    // hex height = 2R where across-flats w = R√3
    const hh = kind === 'hex' ? (2 * d) / Math.sqrt(3) : d;
    return { spec: { kind, w_mm: d, h_mm: hh, r_mm: 0 }, at: { x: a.x - d / 2, y: a.y - hh / 2 } };
  }
  if (mods.shift) { w = h = Math.max(w, h); }
  if (w < 2 || h < 2) return null;
  const x = mods.shift ? (b.x >= a.x ? a.x : a.x - w) : Math.min(a.x, b.x);
  const y = mods.shift ? (b.y >= a.y ? a.y : a.y - h) : Math.min(a.y, b.y);
  return { spec: { kind, w_mm: w, h_mm: h, r_mm: kind === 'rect' ? radius_mm : 0 }, at: { x, y } };
}

/** Polygon spec from clicked points (mm on the mat): normalised to its bbox. */
export function shapeFromPoints(pts: { x: number; y: number }[]): { spec: ShapeSpec; at: { x: number; y: number } } | null {
  if (pts.length < 3) return null;
  const minX = Math.min(...pts.map((p) => p.x)), minY = Math.min(...pts.map((p) => p.y));
  const maxX = Math.max(...pts.map((p) => p.x)), maxY = Math.max(...pts.map((p) => p.y));
  if (maxX - minX < 2 || maxY - minY < 2) return null;
  return { spec: { kind: 'poly', w_mm: maxX - minX, h_mm: maxY - minY, r_mm: 0, points: pts.map((p) => [p.x - minX, p.y - minY]) }, at: { x: minX, y: minY } };
}


// ------------------------------------------------------------------ fitting a primitive to a traced outline

/** Convex hull (monotone chain), counter-clockwise in a y-down frame. */
export function convexHull(pts: number[][]): number[][] {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (src: number[][]) => {
    const out: number[][] = [];
    for (const q of src) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
      out.push(q);
    }
    out.pop();
    return out;
  };
  return [...half(p), ...half(p.slice().reverse())];
}

/** Smallest-area enclosing rectangle (rotating calipers over the hull). Angle in radians. */
export function minAreaRect(poly: number[][]): { cx: number; cy: number; w: number; h: number; angle: number } {
  const hull = convexHull(poly);
  if (hull.length < 3) {
    const [x0, y0, x1, y1] = ringBounds(poly);
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0, angle: 0 };
  }
  let best = { cx: 0, cy: 0, w: Infinity, h: Infinity, angle: 0, area: Infinity };
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const c = Math.cos(-ang), s = Math.sin(-ang);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const q of hull) {
      const x = q[0] * c - q[1] * s, y = q[0] * s + q[1] * c;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    const area = (x1 - x0) * (y1 - y0);
    if (area < best.area) {
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      best = { cx: mx * c + my * s, cy: -mx * s + my * c, w: x1 - x0, h: y1 - y0, angle: ang, area };
    }
  }
  return { cx: best.cx, cy: best.cy, w: best.w, h: best.h, angle: best.angle };
}

function pointIn(p: number[], poly: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Fill a polygon into an n x n bitmask over `box` = [x0, y0, x1, y1], by scanline.
 *  Testing every grid point against every edge instead costs O(n^2 * V) and made fitting a 340-vertex
 *  outline take ~900 ms; this is O(n * V + area) and does the same job in single-digit milliseconds. */
function rasterMask(poly: number[][], box: number[], n: number): Uint8Array {
  const [x0, y0, x1, y1] = box;
  const dx = (x1 - x0) / n, dy = (y1 - y0) / n;
  const m = new Uint8Array(n * n);
  if (!(dx > 0 && dy > 0) || poly.length < 3) return m;
  const xs: number[] = [];
  for (let i = 0; i < n; i++) {
    const y = y0 + (i + 0.5) * dy;
    xs.length = 0;
    for (let j = 0, k = poly.length - 1; j < poly.length; k = j++) {
      const [ax, ay] = poly[k], [bx, by] = poly[j];
      if (ay > y !== by > y) xs.push(ax + ((y - ay) / (by - ay)) * (bx - ax));
    }
    if (xs.length < 2) continue;
    xs.sort((a, b) => a - b);
    for (let s = 0; s + 1 < xs.length; s += 2) {
      let ca = Math.ceil((xs[s] - x0) / dx - 0.5), cb = Math.floor((xs[s + 1] - x0) / dx - 0.5);
      if (ca < 0) ca = 0;
      if (cb > n - 1) cb = n - 1;
      for (let c = ca; c <= cb; c++) m[i * n + c] = 1;
    }
  }
  return m;
}

function maskIoU(a: Uint8Array, b: Uint8Array): number {
  let both = 0, either = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] & b[i]) both++;
    if (a[i] | b[i]) either++;
  }
  return either ? both / either : 0;
}

function unionBox(polys: number[][][]): number[] {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of polys) {
    const b = ringBounds(p);
    x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]);
  }
  return [x0, y0, x1, y1];
}

/** Overlap of two polygons, 0..1. */
export function polyIoU(a: number[][], b: number[][], n = 200): number {
  if (a.length < 3 || b.length < 3) return 0;
  const box = unionBox([a, b]);
  return maskIoU(rasterMask(a, box, n), rasterMask(b, box, n));
}

function placeRing(ring: number[][], cx: number, cy: number, angle: number): number[][] {
  const c = Math.cos(angle), s = Math.sin(angle);
  const [x0, y0, x1, y1] = ringBounds(ring);
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  return ring.map(([x, y]) => {
    const dx = x - mx, dy = y - my;
    return [cx + dx * c - dy * s, cy + dx * s + dy * c];
  });
}

export interface ShapeFit { kind: ShapeKind; label: string; polygon: number[][]; iou: number; w: number; h: number; r: number; angle: number }

/** Replace a traced outline with the simple primitive that covers it best.
 *
 *  Foam is cut, not printed: a scanned wobble of a millimetre or two is noise, and a pocket made of one clean
 *  rectangle, capsule, circle or hexagon looks better, cuts faster and is far easier to nudge afterwards. Every
 *  candidate is built around the tool's own minimum-area rectangle (so it follows the tool's angle, not the
 *  drawer's) and scored by overlap against the traced outline; the best one wins.
 */
export function fitShapes(poly: number[][]): ShapeFit[] {
  if (poly.length < 3) return [];
  const { cx, cy, w, h, angle } = minAreaRect(poly);
  const short = Math.min(w, h), long = Math.max(w, h);
  const raw: { kind: ShapeKind; label: string; ring: number[][]; w: number; h: number; r: number; angle: number }[] = [];
  const add = (kind: ShapeKind, label: string, spec: ShapeSpec, a = angle, at?: { x: number; y: number }) =>
    raw.push({ kind, label, ring: placeRing(shapePolygon(spec), at ? at.x : cx, at ? at.y : cy, a), w: spec.w_mm, h: spec.h_mm, r: spec.r_mm, angle: a });
  add('rect', 'Rectangle', { kind: 'rect', w_mm: w, h_mm: h, r_mm: 0 });
  for (const f of [0.15, 0.3, 0.5]) add('rect', 'Rounded rectangle', { kind: 'rect', w_mm: w, h_mm: h, r_mm: short * f });
  add('slot', 'Capsule', { kind: 'slot', w_mm: w, h_mm: h, r_mm: 0 });
  // a circle is only sensible for a roughly square box; size it by area so it neither swallows nor starves the tool
  if (long / short < 1.35) {
    const d = 2 * Math.sqrt(Math.abs(polygonArea(poly)) / Math.PI);
    const ctr = polygonCentroid(poly);
    add('circle', 'Circle', { kind: 'circle', w_mm: d, h_mm: d, r_mm: 0 }, 0, ctr);
    for (const a of [angle, angle + Math.PI / 6]) add('hex', 'Hexagon', { kind: 'hex', w_mm: short, h_mm: short, r_mm: 0 }, a);
  }
  // one grid for the traced outline and every candidate, so the traced mask is built once
  const N = 200;
  const box = unionBox([poly, ...raw.map((c) => c.ring)]);
  const base = rasterMask(poly, box, N);
  const cands: ShapeFit[] = raw.map((c) => ({ kind: c.kind, label: c.label, polygon: c.ring, iou: maskIoU(base, rasterMask(c.ring, box, N)), w: c.w, h: c.h, r: c.r, angle: c.angle }));
  // near-ties go to the more canonical shape: a capsule and a rounded rectangle can score the same on a
  // rounded bar, and "Capsule 160 x 40" is the one worth reading on the tool list
  const rank = (f: ShapeFit) => (f.kind === 'circle' ? 0 : f.kind === 'hex' ? 1 : f.kind === 'slot' ? 2 : f.r <= 0.05 ? 3 : 4);
  return cands.sort((a, b) => (Math.abs(a.iou - b.iou) < 0.015 ? rank(a) - rank(b) : b.iou - a.iou));
}

/** The best primitive for this outline, or null when nothing fits well enough to be worth the loss of detail. */
export function bestShape(poly: number[][], minIou = 0.82): ShapeFit | null {
  const best = fitShapes(poly)[0];
  return best && best.iou >= minIou ? best : null;
}


// ------------------------------------------------------------------ proportional ("soft") dragging

/** Cumulative distance along a closed ring, plus its perimeter. */
export function arcLengths(poly: number[][]): { arc: number[]; per: number } {
  const n = poly.length, arc = new Array<number>(n);
  let acc = 0;
  arc[0] = 0;
  for (let i = 1; i < n; i++) { acc += Math.hypot(poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1]); arc[i] = acc; }
  return { arc, per: acc + Math.hypot(poly[0][0] - poly[n - 1][0], poly[0][1] - poly[n - 1][1]) };
}

/** `poly0` with vertex `anchor` moved by (dx, dy) and its neighbours carried along, easing to nothing at
 *  `radius` measured ALONG the outline. Arc length, not straight-line distance: pulling the left edge of a
 *  narrow ruler must not drag the right edge with it. radius <= 0 moves the single vertex.
 *  Always rebuilt from `poly0`, so a caller may change the radius mid-drag without the shape creeping. */
export function softDragRing(poly0: number[][], anchor: number, arc: number[], per: number, dx: number, dy: number, radius: number): number[][] {
  if (radius <= 0) return poly0.map((v, i) => (i === anchor ? [v[0] + dx, v[1] + dy] : v));
  const r = Math.min(radius, per / 2);          // past half the perimeter the whole ring just translates
  const a0 = arc[anchor];
  return poly0.map((v, i) => {
    const raw = Math.abs(arc[i] - a0);
    const d = Math.min(raw, per - raw);
    if (d >= r) return v;
    const w = 0.5 * (1 + Math.cos((Math.PI * d) / r));
    return [v[0] + dx * w, v[1] + dy * w];
  });
}
