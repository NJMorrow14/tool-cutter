import { TOOL_COLORS, type SourceKind, type Tool, type ToolResult } from './types';

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
    include: true,
    clearance_mm: null,
    depth_mm: null,
    rotation_deg: 0,
    offset_mm: { x: 0, y: 0 },
    notch: null,
    pending: false,
    error: null,
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
