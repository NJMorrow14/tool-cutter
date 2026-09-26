'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ui from './ui.module.css';
import st from './stage.module.css';
import ed from './editor.module.css';
import ScanViewer from './ScanViewer';
import AddShape from './AddShape';
import { API_BASE_URL, autoDetect, imageUrl, mergeTools, segment, snapToBase, splitTool } from '../lib/api';
import { arcLengths, bestShape, shapeFromDrag, shapeTool, fitShapes, softDragRing, type ShapeFit, fmtMm, pointSegment, polyToPath, polygonArea, resampleRing, ringBounds, simplifyRing, smoothRing, toolColor, toolFromResult, uid } from '../lib/geom';
import { PAGES, PRINT_DEFAULTS, printOutlines, type PageSize } from '../lib/print';
import type { PromptPoint, SessionInfo, ShapeKind, Tool, ToolResult } from '../lib/types';

type View = { x: number; y: number; w: number; h: number };
type Drag =
  | { kind: 'vertices'; anchor: number; poly0: number[][]; arc: number[]; per: number; start: number[]; rigid: number[] | null; moved: boolean }
  | { kind: 'pan'; startClient: { x: number; y: number }; view0: View; from: number[]; moved: boolean }
  | { kind: 'brush'; last: number[]; start: number[]; moved: boolean }
  | { kind: 'marquee'; start: number[]; cur: number[]; alt: boolean }
  | { kind: 'cut'; start: number[]; cur: number[] };

interface Props {
  session: SessionInfo;
  tools: Tool[];
  setTools: React.Dispatch<React.SetStateAction<Tool[]>>;
  modelAvailable: boolean;
  onContinue: () => void;
}

const HISTORY_MAX = 60;
const CLICK_PX = 4;          // a press that moves less than this is a click, not a drag

function newTool(session: SessionInfo, index: number, points: PromptPoint[], box: number[] | null): Tool {
  return {
    id: uid(), session_id: session.id, source: session.source_kind, name: `Tool ${index + 1}`, color: toolColor(index),
    points, box, polygon_px: [], polygon_mm: [], area_mm2: 0,
    measured_thickness_mm: null, include: true, clearance_mm: null, depth_mm: null, rotation_deg: 0,
    offset_mm: { x: 0, y: 0 }, notch: null, pending: true, error: null,
  };
}

function applyResult(t: Tool, r: ToolResult): Tool {
  return { ...t, polygon_px: r.polygon_px, polygon_mm: r.polygon_mm, area_mm2: r.area_mm2, measured_thickness_mm: r.measured_thickness_mm,
    auto_polygon_px: r.polygon_px, edited: false, pending: false, error: r.polygon_px.length ? null : 'No region found — add a point inside the tool' };
}

/** Find the tools AND fix their outlines, on one canvas.
 *
 *  Detecting and refining used to be two steps that drew the same picture, listed the same tools and offered the
 *  same toggles; only the right-hand panel differed. They are one step now (Nolan, 2026-09-20: "a lot of overlap").
 *  Still no modes — what a gesture does follows from what is under the cursor and whether a tool is selected:
 *    nothing selected : click the mat outlines a new tool there · Alt+drag box-prompts one
 *    a tool selected  : its handles are live — drag one to move it, click the line to add one, Shift pushes,
 *                       Alt+drag grabs a group, Shift/Alt+CLICK adds an include/exclude hint and re-outlines
 *    always           : drag empty space pans, wheel zooms, click a tool selects it, click the mat deselects
 */
export default function OutlineStep({ session, tools, setTools, modelAvailable, onContinue }: Props) {
  const rect = session.rectified!;
  const W = rect.width;
  const H = rect.height;
  const mpp = rect.mm_per_px;

  const mine = useMemo(() => tools.filter((t) => t.session_id === session.id), [tools, session.id]);
  const drawn = useMemo(() => mine.filter((t) => t.polygon_px.length >= 3), [mine]);
  // drawn primitives belong to no session and are positioned on the mat, not on the scan, so they are listed
  // here (this is where you add them) but are moved and sized in Layout
  const shapes = useMemo(() => tools.filter((t) => t.source === 'shape'), [tools]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Tools ticked for combining. Kept apart from selectedId: one tool is being EDITED, several are being GATHERED.
  const [combineIds, setCombineIds] = useState<string[]>([]);
  const [combining, setCombining] = useState(false);
  const selected = mine.find((t) => t.id === selectedId) ?? null;
  const editable = selected && selected.polygon_px.length >= 3 ? selected : null;

  const [softMm, setSoftMm] = useState(25);   // how far along the outline a dragged point carries its neighbours
  const [view, setView] = useState<View>({ x: 0, y: 0, w: W, h: H });
  const [drag, setDrag] = useState<Drag | null>(null);
  const [sel, setSel] = useState<Set<number>>(new Set());
  const [splitArm, setSplitArm] = useState(false);
  const [autoBusy, setAutoBusy] = useState(false);
  const [autoOpen, setAutoOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [recovery, setRecovery] = useState<Tool[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // refine_with_sam is only ever the no-height fallback now; it is never offered as a choice.
  const [autoOpts, setAutoOpts] = useState({ mode: 'auto' as 'auto' | 'color' | 'height', min_area_mm2: 200, height_threshold_mm: 2, refine_with_sam: !rect.has_height });
  const [shapeMode, setShapeMode] = useState(false);     // outline everything as a simple primitive
  const [shapeMinIou, setShapeMinIou] = useState(0.82);  // how close a primitive must be before it is accepted
  const [pageSize, setPageSize] = useState<PageSize>(PRINT_DEFAULTS.page);
  const [addingShape, setAddingShape] = useState(false);
  const [drawKind, setDrawKind] = useState<Exclude<ShapeKind, 'poly'> | null>(null);
  const [snapBusy, setSnapBusy] = useState(false);
  const [, bump] = useState(0);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const history = useRef<Map<string, { past: number[][][]; future: number[][][] }>>(new Map());
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const reqIds = useRef<Map<string, number>>(new Map());
  const [upp, setUpp] = useState(1);
  const dragRef = useRef<Drag | null>(null);
  const lastDelta = useRef<number[]>([0, 0]);
  useEffect(() => { dragRef.current = drag; }, [drag]);

  // Borders always come from the scan topography. Nolan, 2026-09-22: "I dont want to ever follow visible tool
  // edges" — photo edges chase printed labels, shadows and specular highlights, and every measurement of that
  // path in this repo lost to the topographic one. "photo" survives ONLY as the fallback for a capture with no
  // height at all (a plain photo upload), where there is no topography to trace; it is never a user choice.
  const edgeSource: 'topo' | 'photo' = rect.has_height ? 'topo' : 'photo';
  const canClick = rect.has_height || modelAvailable;

  // ------------------------------------------------------------------ view maths
  const metrics = useCallback(() => {
    const el = svgRef.current;
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const s = Math.min(r.width / view.w, r.height / view.h);
    return { r, s, ox: (r.width - view.w * s) / 2, oy: (r.height - view.h * s) / 2 };
  }, [view]);

  useEffect(() => { const m = metrics(); if (m) setUpp(1 / m.s); }, [metrics, view]);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => { const m = metrics(); if (m) setUpp(1 / m.s); });
    ro.observe(el);
    return () => ro.disconnect();
  }, [metrics]);

  const toImage = useCallback((clientX: number, clientY: number): number[] => {
    const m = metrics();
    if (!m) return [0, 0];
    return [view.x + (clientX - m.r.left - m.ox) / m.s, view.y + (clientY - m.r.top - m.oy) / m.s];
  }, [metrics, view]);

  const clampView = (v: View): View => {
    const w = Math.min(Math.max(v.w, 40), W * 3);
    const h = Math.min(Math.max(v.h, 40), H * 3);
    return { x: Math.min(Math.max(v.x, -w / 2), W - w / 2), y: Math.min(Math.max(v.y, -h / 2), H - h / 2), w, h };
  };
  const zoomAt = useCallback((factor: number, at?: number[]) => {
    setView((v) => {
      const cx = at ? at[0] : v.x + v.w / 2, cy = at ? at[1] : v.y + v.h / 2;
      return clampView({ x: cx - (cx - v.x) / factor, y: cy - (cy - v.y) / factor, w: v.w / factor, h: v.h / factor });
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const fitTo = useCallback((poly: number[][] | null) => {
    const el = svgRef.current;
    const aspect = el ? el.getBoundingClientRect().width / Math.max(1, el.getBoundingClientRect().height) : W / H;
    let [x0, y0, x1, y1] = poly && poly.length ? ringBounds(poly) : [0, 0, W, H];
    const mw = (x1 - x0) * 0.2 + 20, mh = (y1 - y0) * 0.2 + 20;
    x0 -= mw; x1 += mw; y0 -= mh; y1 += mh;
    let w = x1 - x0, h = y1 - y0;
    if (w / h > aspect) h = w / aspect; else w = h * aspect;
    setView(clampView({ x: (x0 + x1) / 2 - w / 2, y: (y0 + y1) / 2 - h / 2, w, h }));
  }, [W, H]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Ctrl/Cmd- or Shift-click gathers tools instead of selecting one, the same modifier pair the canvas uses. */
  const toggleCombine = useCallback((id: string) => {
    setCombineIds((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);
  }, []);

  /** Combine the gathered tools into one outline (server-side: their masks are unioned and the seam healed). */
  const combineSelected = useCallback(async () => {
    if (combineIds.length < 2 || combining) return;
    setCombining(true);
    const previous = mine;
    const first = mine.find((t) => t.id === combineIds[0]);
    setError(null);
    try {
      const res = await mergeTools(session.id, combineIds);
      const idx = Math.max(0, tools.findIndex((t) => t.id === combineIds[0]));
      const merged = res.tools.filter((r) => r.polygon_px.length >= 3)
        .map((r, i) => {
          const nt = toolFromResult(r, tools.length + i, session.source_kind, first?.name);
          if (first) nt.color = first.color;        // the combined tool keeps the first one's identity
          return nt;
        });
      if (!merged.length) { setError('Combine produced no outline; the tools were left alone.'); return; }
      setRecovery(previous);                        // same undo affordance as "Clear list"
      setTools((prev) => {
        const rest = prev.filter((t) => !res.removed.includes(t.id));
        const at = Math.max(0, Math.min(idx, rest.length));
        return [...rest.slice(0, at), ...merged, ...rest.slice(at)];
      });
      setCombineIds([]);
      setSelectedId(merged[0].id);
      if (res.bridged_mm > 0) setError(`Combined ${res.removed.length} tools — they were ${res.bridged_mm} mm apart, so a bridge was drawn to join them.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Combine failed');
    } finally {
      setCombining(false);
    }
  }, [combineIds, combining, mine, tools, session.id, session.source_kind]);

  /** Selecting from the list zooms to the tool; clicking it on the canvas does not — you are already looking at it. */
  const selectTool = useCallback((id: string | null, fit = false) => {
    setSelectedId(id);
    setSel(new Set());
    if (fit && id) {
      const t = tools.find((x) => x.id === id);
      if (t && t.polygon_px.length >= 3) fitTo(t.polygon_px);
    }
  }, [tools, fitTo]);


  // ------------------------------------------------------------------ outlining (server)
  const runSegment = useCallback(async (tool: Tool) => {
    const id = (reqIds.current.get(tool.id) ?? 0) + 1;
    reqIds.current.set(tool.id, id);
    try {
      const res = await segment(session.id, [{ id: tool.id, points: tool.points, box: tool.box }], edgeSource);
      if (reqIds.current.get(tool.id) !== id) return;
      setTools((prev) => prev.map((t) => (t.id === tool.id ? applyResult(t, res.tools[0]) : t)));
    } catch (err) {
      if (reqIds.current.get(tool.id) !== id) return;
      const msg = err instanceof Error ? err.message : 'Outlining failed';
      setTools((prev) => prev.map((t) => (t.id === tool.id ? { ...t, pending: false, error: msg } : t)));
    }
  }, [session.id, setTools, edgeSource]);

  const schedule = useCallback((tool: Tool) => {
    const prev = timers.current.get(tool.id);
    if (prev) clearTimeout(prev);
    timers.current.set(tool.id, setTimeout(() => void runSegment(tool), 180));
  }, [runSegment]);

  useEffect(() => () => timers.current.forEach((t) => clearTimeout(t)), []);

  const updateTool = useCallback((id: string, fn: (t: Tool) => Tool, resegment = true) => {
    setTools((prev) => prev.map((t) => {
      if (t.id !== id) return t;
      const next = fn(t);
      if (resegment) { next.pending = true; schedule(next); }
      return next;
    }));
  }, [schedule, setTools]);

  const addTool = useCallback((points: PromptPoint[], box: number[] | null) => {
    const t = newTool(session, tools.length, points, box);
    setTools((prev) => [...prev, t]);
    setSelectedId(t.id);
    setSel(new Set());
    schedule(t);
  }, [schedule, setTools, tools.length, session]);

  const removeTool = (id: string) => {
    setRecovery(mine);
    setTools((prev) => prev.filter((t) => t.id !== id));
    if (selectedId === id) setSelectedId(null);
  };

  const doSplit = async (tool: Tool, line: [number[], number[]]) => {
    setError(null);
    setTools((prev) => prev.map((t) => (t.id === tool.id ? { ...t, pending: true } : t)));
    try {
      const res = await splitTool(session.id, tool.id, line, edgeSource);
      const idx = tools.findIndex((t) => t.id === tool.id);
      const parts = res.tools.filter((r) => r.polygon_px.length >= 3).map((r, i) => {
        const nt = toolFromResult(r, tools.length + i, session.source_kind, `${tool.name} ${String.fromCharCode(97 + i)}`);
        if (i === 0) nt.color = tool.color;
        return nt;
      });
      setTools((prev) => {
        const rest = prev.filter((t) => t.id !== tool.id);
        const at = Math.max(0, Math.min(idx, rest.length));
        return [...rest.slice(0, at), ...parts, ...rest.slice(at)];
      });
      selectTool(parts[0]?.id ?? null);
    } catch (err) {
      setTools((prev) => prev.map((t) => (t.id === tool.id ? { ...t, pending: false } : t)));
      setError(err instanceof Error ? err.message : 'Split failed');
    }
  };

  const runAuto = async (replace: boolean) => {
    setAutoBusy(true);
    setError(null);
    try {
      const res = await autoDetect(session.id, { ...autoOpts, edge_source: edgeSource, id_prefix: `${uid('a')}_` });
      const others = tools.filter((t) => t.session_id !== session.id);
      const base = replace ? others.length : tools.length;
      let found: Tool[] = res.tools.filter((r) => r.polygon_px.length >= 3).map((r, i) => toolFromResult(r, base + i, session.source_kind));
      if (shapeMode) {
        found = found.map((t) => {
          const fit = bestShape(t.polygon_px, shapeMinIou);
          if (!fit) return t;
          const ring = simplifyRing(fit.polygon, 0.25 / mpp);
          return { ...t, polygon_px: ring, polygon_mm: ring.map((p) => [p[0] * mpp, p[1] * mpp]), area_mm2: polygonArea(ring) * mpp * mpp, auto_polygon_px: t.polygon_px, edited: true };
        });
      }
      if (!found.length) { setError("No tools found. Your existing outlines have been kept. Try adjusting detection settings or click a tool on the canvas."); return; }
      setRecovery(mine.length ? mine : null);
      setTools((prev) => (replace ? [...prev.filter((t) => t.session_id !== session.id), ...found] : [...prev, ...found]));
      setSelectedId(null);
      setAutoOpen(false);
      if (res.sam_error) setError(`Detected by ${res.mode}; photo refinement unavailable (${res.sam_error}).`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Auto-detect failed');
    } finally {
      setAutoBusy(false);
    }
  };

  // ------------------------------------------------------------------ hand edits + history
  const commit = useCallback((id: string, poly: number[][], record = true) => {
    setTools((prev) => prev.map((t) => {
      if (t.id !== id) return t;
      if (record) {
        const h = history.current.get(id) ?? { past: [], future: [] };
        h.past.push(t.polygon_px);
        if (h.past.length > HISTORY_MAX) h.past.shift();
        h.future = [];
        history.current.set(id, h);
      }
      return { ...t, polygon_px: poly, polygon_mm: poly.map((p) => [p[0] * mpp, p[1] * mpp]), area_mm2: polygonArea(poly) * mpp * mpp, auto_polygon_px: t.auto_polygon_px ?? t.polygon_px, edited: true };
    }));
    bump((v) => v + 1);
  }, [mpp, setTools]);
  const record = (t: Tool) => {
    const h = history.current.get(t.id) ?? { past: [], future: [] };
    h.past.push(t.polygon_px); if (h.past.length > HISTORY_MAX) h.past.shift(); h.future = []; history.current.set(t.id, h);
  };
  const undo = useCallback(() => {
    if (!editable) return;
    const h = history.current.get(editable.id);
    if (!h?.past.length) return;
    const prev = h.past.pop()!;
    h.future.push(editable.polygon_px);
    commit(editable.id, prev, false);
    setSel(new Set());
  }, [editable, commit]);
  const redo = useCallback(() => {
    if (!editable) return;
    const h = history.current.get(editable.id);
    if (!h?.future.length) return;
    const next = h.future.pop()!;
    h.past.push(editable.polygon_px);
    commit(editable.id, next, false);
    setSel(new Set());
  }, [editable, commit]);
  const resetAuto = () => {
    if (!editable?.auto_polygon_px) return;
    commit(editable.id, editable.auto_polygon_px);
    setTools((prev) => prev.map((t) => (t.id === editable.id ? { ...t, edited: false } : t)));
    setSel(new Set());
  };

  const stepPx = Math.max(1, 1.0 / mpp);
  const doSmoothAll = (sigmaMm: number) => {
    if (!editable) return;
    commit(editable.id, simplifyRing(smoothRing(resampleRing(editable.polygon_px, stepPx), sigmaMm / mpp / stepPx), 0.15 / mpp));
    setSel(new Set());
  };
  const doSmoothSelection = (sigmaMm: number) => {
    if (!editable || !sel.size) return;
    commit(editable.id, smoothRing(editable.polygon_px, sigmaMm / mpp, (i) => (sel.has(i) ? 1 : 0)));
  };
  /** Walk every vertex out to where the tool meets the mat. */
  const doSnapBase = async (t: Tool) => {
    setSnapBusy(true);
    setError(null);
    try {
      const poly = await snapToBase(session.id, t.polygon_px);
      if (poly.length >= 3) { record(t); commit(t.id, poly, false); setSel(new Set()); }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Snap to base failed');
    } finally {
      setSnapBusy(false);
    }
  };

  const doSimplify = () => { if (editable) { commit(editable.id, simplifyRing(editable.polygon_px, 0.3 / mpp)); setSel(new Set()); } };
  const doResample = () => { if (editable) { commit(editable.id, resampleRing(editable.polygon_px, stepPx)); setSel(new Set()); } };

  // ------------------------------------------------------------------ fit a simple primitive
  /** Replace a tool's traced line with the primitive that covers it best. Reversible: the traced outline is
   *  kept in `auto_polygon_px`, so "Reset to detected" brings it back. */
  /** A fitted primitive comes out of shapePolygon at ~1 mm arc resolution — hundreds of points for a capsule,
   *  which hides the handles and defeats the point of asking for a simple shape. Collapse the straight runs. */
  const shapeRing = (ring: number[][]) => simplifyRing(ring, 0.25 / mpp);

  const simplifyToShape = (t: Tool, track = false): boolean => {
    const fit = t.polygon_px.length >= 3 ? bestShape(t.polygon_px, shapeMinIou) : null;
    if (!fit) return false;
    commit(t.id, shapeRing(fit.polygon));
    // re-score against the line it just became, so the readout names what is actually there now
    if (track) { setShapeFits(fitShapes(fit.polygon)); setShowAlts(false); }
    return true;
  };
  const simplifyAll = () => {
    const targets = drawn;
    const n = targets.filter((t) => simplifyToShape(t)).length;
    setShapeFits(null); setShowAlts(false);
    setSel(new Set());
    setError(n === targets.length ? null
      : `${n} of ${targets.length} became a simple shape — the rest are too irregular to fit one and kept their traced outline. Lower "fit at least" to force them.`);
  };
  /** Candidate primitives for the selected tool — computed ONLY when asked for.
   *  Scoring them is a 140 x 140 sampling of eight candidates against a several-hundred-vertex ring; doing that
   *  in the render path made every pointermove of a vertex drag re-score the lot, and dragging crawled. */
  const [shapeFits, setShapeFits] = useState<ShapeFit[] | null>(null);
  const [showAlts, setShowAlts] = useState(false);
  useEffect(() => { setShapeFits(null); setShowAlts(false); }, [selectedId]);
  const shapeLabel = useMemo(() => {
    const f = shapeFits?.[0];
    if (!f || f.iou < 0.985) return null;             // only call it a shape when the line really IS one
    const d = (v: number) => Math.round(v * mpp);
    return f.kind === 'circle' ? `Circle ⌀${d(f.w)} mm` : f.kind === 'hex' ? `Hexagon ${d(f.w)} mm` : `${f.label} ${d(f.w)} × ${d(f.h)} mm`;
  }, [shapeFits, mpp]);

  const selCentroid = (poly: number[][]) => {
    const pts = Array.from(sel).map((i) => poly[i]).filter(Boolean);
    return [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];
  };
  const transformSel = (fn: (p: number[], c: number[]) => number[], recordIt = true) => {
    if (!editable || !sel.size) return;
    const c = selCentroid(editable.polygon_px);
    commit(editable.id, editable.polygon_px.map((p, i) => (sel.has(i) ? fn(p, c) : p)), recordIt);
  };
  const rotateSel = (deg: number) => {
    const a = (deg * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
    transformSel((p, c) => [c[0] + (p[0] - c[0]) * ca - (p[1] - c[1]) * sa, c[1] + (p[0] - c[0]) * sa + (p[1] - c[1]) * ca]);
  };
  const nudgeSel = (dxMm: number, dyMm: number) => transformSel((p) => [p[0] + dxMm / mpp, p[1] + dyMm / mpp]);
  const deleteSel = () => {
    if (!editable || !sel.size || editable.polygon_px.length - sel.size < 3) return;
    commit(editable.id, editable.polygon_px.filter((_, i) => !sel.has(i)));
    setSel(new Set());
  };

  // ------------------------------------------------------------------ soft (proportional) dragging
  /** The ring as it should look with the grabbed point moved by (dx, dy). Rebuilt from the ring as it was
   *  when the drag started, so the radius can change mid-drag (wheel) without the shape creeping.
   *  Shared with the 3D view via geom.softDragRing. */
  const softRing = (d: Extract<Drag, { kind: 'vertices' }>, dx: number, dy: number): number[][] => {
    if (d.rigid) {
      const set = new Set(d.rigid);
      return d.poly0.map((v, i) => (set.has(i) ? [v[0] + dx, v[1] + dy] : v));
    }
    return softDragRing(d.poly0, d.anchor, d.arc, d.per, dx, dy, softMm / mpp);
  };

  useEffect(() => {
    const d = dragRef.current;
    if (d?.kind === 'vertices' && d.moved && !d.rigid && editable) commit(editable.id, softRing(d, lastDelta.current[0], lastDelta.current[1]), false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [softMm]);

  // ------------------------------------------------------------------ print 1:1
  const sheetsFor = useCallback((t: Tool | null) => {
    if (!t || t.polygon_mm.length < 3) return 0;
    const [x0, y0, x1, y1] = ringBounds(t.polygon_mm);
    const pg = PAGES[pageSize];
    const availW = pg.w - 2 * PRINT_DEFAULTS.margin_mm, availH = pg.h - 2 * PRINT_DEFAULTS.margin_mm - 24;
    const nx = Math.max(1, Math.ceil((x1 - x0 - PRINT_DEFAULTS.overlap_mm) / Math.max(10, availW - PRINT_DEFAULTS.overlap_mm)));
    const ny = Math.max(1, Math.ceil((y1 - y0 - PRINT_DEFAULTS.overlap_mm) / Math.max(10, availH - PRINT_DEFAULTS.overlap_mm)));
    return nx * ny;
  }, [pageSize]);
  const doPrint = (items: Tool[]) => {
    const ok = printOutlines(items.filter((t) => t.polygon_mm.length >= 3).map((t) => ({ name: t.name, polygon_mm: t.polygon_mm, thickness_mm: t.measured_thickness_mm })),
      { ...PRINT_DEFAULTS, page: pageSize });
    if (!ok) setError('The browser blocked the print window — allow pop-ups for localhost and try again.');
  };

  // ------------------------------------------------------------------ keyboard
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input, textarea, select, [contenteditable=true]')) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
      if (e.key.toLowerCase() === 'v' && !e.metaKey && !e.ctrlKey) { setDrawKind(null); setSplitArm(false); return; }
      if (e.key === 'Escape') { setDrawKind(null); setSplitArm(false); }
      // Escape must still clear combine ticks when no tool is being edited, so the guard allows that case
      // through; every branch below that needs a tool checks for one.
      if (!selected && !combineIds.length) return;
      const stepMm = e.shiftKey ? 2 : 0.5;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (sel.size) deleteSel(); else if (selected) removeTool(selected.id);
        e.preventDefault();
      } else if (e.key === 'Escape') { if (combineIds.length) setCombineIds([]); else if (sel.size) setSel(new Set()); else setSelectedId(null); }
      else if (!editable) return;
      else if (e.key === 'ArrowLeft') { nudgeSel(-stepMm, 0); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { nudgeSel(stepMm, 0); e.preventDefault(); }
      else if (e.key === 'ArrowUp') { nudgeSel(0, -stepMm); e.preventDefault(); }
      else if (e.key === 'ArrowDown') { nudgeSel(0, stepMm); e.preventDefault(); }
      else if (e.key.toLowerCase() === 'r') rotateSel(e.shiftKey ? -5 : 5);
      else if (e.key === '[') rotateSel(-1);
      else if (e.key === ']') rotateSel(1);
      else if (e.key.toLowerCase() === 'a' && (e.metaKey || e.ctrlKey)) { setSel(new Set(editable.polygon_px.map((_, i) => i))); e.preventDefault(); }

    };
    window.addEventListener('keydown', down);
    return () => window.removeEventListener('keydown', down);
  }); // re-bind every render: handlers close over the latest selection

  const hist = editable ? history.current.get(editable.id) : undefined;
  const ready = tools.filter((t) => t.polygon_mm.length >= 3).length;
  const imported = tools.length - mine.length;

  return (
    <div className={ed.modelingGrid}>
      <aside className={`${ui.panel} ${ed.inspector} ${ed.outliner}`} aria-label="Scene objects">
        <div className={ed.dockTitle}>Scene <span>{mine.length + shapes.length} objects</span></div>
        <div className={ed.inspectorBody}>
        <div className={ui.section}>
          <div className={ui.rowBetween}>
            <h2 className={ui.panelTitle}>Scan actions</h2>
            <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnGhost}`} onClick={() => setAutoOpen(!autoOpen)}
              aria-expanded={autoOpen} title={autoOpen ? 'Hide detection settings' : 'Show detection settings'}>{autoOpen ? 'Hide settings' : 'Settings'}</button>
          </div>
          {autoOpen && (
            <>
              <div className={ui.grid2}>
                <label className={ui.field}>
                  <span className={ui.label}>Method</span>
                  <select className={ui.select} value={autoOpts.mode} onChange={(e) => setAutoOpts({ ...autoOpts, mode: e.target.value as typeof autoOpts.mode })}>
                    <option value="auto">Auto (height)</option>
                    <option value="height">Height above mat</option>
                    <option value="color">Color vs. mat (textured scans)</option>
                  </select>
                </label>
                <label className={ui.field}>
                  <span className={ui.label}>Min area (mm²)</span>
                  <input className={ui.input} type="number" min={10} step={50} value={autoOpts.min_area_mm2} onChange={(e) => setAutoOpts({ ...autoOpts, min_area_mm2: Number(e.target.value) })} />
                </label>
              </div>
              {rect.has_height && (
                <label className={ui.field}>
                  <span className={ui.label}>Height threshold (mm)</span>
                  <input className={ui.input} type="number" min={0.5} step={0.5} value={autoOpts.height_threshold_mm} onChange={(e) => setAutoOpts({ ...autoOpts, height_threshold_mm: Number(e.target.value) })} />
                </label>
              )}
              <label className={ui.checkbox} title="Replace each traced outline with the rectangle, capsule, circle or hexagon that covers it best, turned to the tool's own angle. Anything too irregular keeps its traced line.">
                <input type="checkbox" checked={shapeMode} onChange={(e) => setShapeMode(e.target.checked)} />
                Outline as simple shapes
              </label>
              {shapeMode && (
                <label className={ed.slider} title="How closely a primitive must cover the traced outline before it is used. Lower = more tools become shapes, at the cost of fit.">
                  <span className={ui.label}>Fit at least</span>
                  <input type="range" min={60} max={95} step={1} value={Math.round(shapeMinIou * 100)} onChange={(e) => setShapeMinIou(Number(e.target.value) / 100)} />
                  <span className={ui.unit}>{Math.round(shapeMinIou * 100)}%</span>
                </label>
              )}
            </>
          )}
          <div className={ui.row}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={autoBusy} onClick={() => runAuto(true)}>
              {autoBusy ? <span className={ui.spinner} /> : null} {autoBusy ? 'Finding tools…' : mine.length ? 'Re-detect tools' : 'Detect tools'}
            </button>
            <button type="button" className={ui.btn} disabled={autoBusy || !mine.length} onClick={() => runAuto(false)}>Find more</button>
          </div>
        </div>

        <div className={ui.section}>
          <div className={`${ui.rowBetween} ${ed.sceneHeading}`}>
            <h2 className={ui.panelTitle}>Tools ({mine.length})</h2>
            <span className={ui.row} style={{ gap: 6 }}>
              {drawn.length > 0 && <button type="button" className={`${ui.btn} ${ui.btnSm}`} title="Replace every traced outline with the simple shape that fits it best" onClick={simplifyAll}>◻ Shapes</button>}
              {drawn.length > 0 && <button type="button" className={`${ui.btn} ${ui.btnSm}`} title="Print every outline at 1:1, one tool after another" onClick={() => doPrint(drawn)}>🖨 All</button>}
              <button type="button" className={`${ui.btn} ${ui.btnSm}`} title="Add a plain rectangle, slot, circle or hexagon — for something the scan cannot see" onClick={() => setAddingShape(!addingShape)}>+ Shape</button>
              {mine.length > 0 && <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnDanger}`} onClick={() => { setRecovery(mine); setTools((prev) => prev.filter((t) => t.session_id !== session.id)); setSelectedId(null); setCombineIds([]); }}>Clear list</button>}
              {combineIds.length > 0 && (
                <>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={combineIds.length < 2 || combining}
                          title={combineIds.length < 2 ? 'Ctrl/Cmd-click another tool to combine it with this one' : 'Combine the ticked tools into one outline'}
                          onClick={combineSelected}>
                    {combining ? 'Combining…' : `⊕ Combine ${combineIds.length}`}
                  </button>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => setCombineIds([])}>Clear ticks</button>
                </>
              )}
            </span>
          </div>
          {addingShape && (
            <AddShape onClose={() => setAddingShape(false)} onAdd={(spec, thick) => {
              const t = shapeTool(spec, thick, tools);
              setTools((prev) => [...prev, t]);
              setAddingShape(false);
              setError(null);
            }} />
          )}
          <input className={ui.input} type="search" placeholder="Search tools…" aria-label="Search tools" value={query} onChange={e => setQuery(e.target.value)} />
          <div className={ui.list}>
            {mine.filter(t => t.name.toLowerCase().includes(query.toLowerCase())).map((t) => (
              <div key={t.id} className={`${ui.toolRow} ${ed.sceneRow} ${t.id === selectedId ? ui.toolRowActive : ''}`}
                   style={combineIds.includes(t.id) ? { outline: '2px solid var(--accent, #f26a1b)', outlineOffset: '-2px' } : undefined}
                   onClick={(e) => (e.metaKey || e.ctrlKey || e.shiftKey) ? toggleCombine(t.id) : selectTool(t.id, true)}>
                <span className={ui.swatch} style={{ background: t.color }} />
                <button type="button" aria-label={combineIds.includes(t.id) ? `Remove ${t.name} from the combine selection` : `Edit ${t.name}`}
                        aria-pressed={combineIds.includes(t.id)} className={ui.toolName}
                        onClick={(e) => { e.stopPropagation();          // the row handles it too; a TOGGLE must not fire twice
                                          if (e.metaKey || e.ctrlKey || e.shiftKey) toggleCombine(t.id); else selectTool(t.id, true); }}>
                  {combineIds.includes(t.id) ? '☑ ' : ''}{t.name}</button>
                <span className={ui.toolMeta}>
                  {t.pending ? <span className={ui.spinner} /> : t.error ? '⚠' : `${(t.area_mm2 / 100).toFixed(1)} cm²`}
                  {t.measured_thickness_mm !== null && !t.pending ? ` · ${fmtMm(t.measured_thickness_mm)}` : ''}
                </span>
                
                <button type="button" className={ui.iconBtn} aria-label={`Remove ${t.name}`} title="Remove" onClick={(e) => { e.stopPropagation(); removeTool(t.id); }}>×</button>
              </div>
            ))}
            {!mine.length && <div className={ui.emptyState}><strong>Start with an outline</strong>Detect all tools at once, or click a tool on the canvas to add it individually.</div>}
            {!!mine.length && !mine.some(t => t.name.toLowerCase().includes(query.toLowerCase())) && <p className={ui.hint}>No tools match “{query}”.</p>}
            {mine.length > 1 && !combineIds.length && <p className={ui.hint}>Cmd/Ctrl-click tools — here or on the scan — to tick several, then Combine them into one outline.</p>}
            {imported > 0 && <p className={ui.hint}>{imported} imported tool model{imported > 1 ? 's' : ''} will join these in the layout.</p>}
          </div>
          {shapes.length > 0 && (
            <div className={ui.list} style={{ marginTop: 6 }}>
              <p className={ui.hint}>Added shapes — these sit on the foam, not on the scan, so you place and size them in Layout &amp; export.</p>
              {shapes.map((t) => (
                <div key={t.id} className={ui.toolRow} style={{ gridTemplateColumns: '12px 1fr auto auto' }}>
                  <span className={ui.swatch} style={{ background: t.color }} />
                  <span className={ui.toolName}>{t.name}</span>
                  <span className={ui.toolMeta}>{fmtMm(t.measured_thickness_mm)} thick</span>
                  <button type="button" className={ui.iconBtn} title="Remove" onClick={() => setTools((prev) => prev.filter((x) => x.id !== t.id))}>×</button>
                </div>
              ))}
            </div>
          )}
        </div>

        {session.scan?.photo_coverage && <details className={ui.disclosure}>
          <summary>Photo coverage · {Math.round(session.scan.photo_coverage.covered_fraction * 100)}% of drawer · {session.scan.photo_coverage.photo_count} photos</summary>
          <p className={ui.hint}>Green: overlapping views. Amber: one view or partial overlap. Red: missing coverage. Add overlapping photos over red or amber areas before cutting.</p>
          <div role="img" aria-label="Drawer photo coverage, from top left to bottom right" style={{ display: 'grid', gridTemplateColumns: `repeat(${session.scan.photo_coverage.cols}, 1fr)`, gap: 3, gridTemplateRows: `repeat(${session.scan.photo_coverage.rows}, 1fr)`, width: Math.min(600, 220 * (session.mat_mm?.width || 1) / (session.mat_mm?.height || 1)), maxWidth: '100%', aspectRatio: `${session.mat_mm?.width || 1} / ${session.mat_mm?.height || 1}` }}>
            {session.scan.photo_coverage.covered.flatMap((row, y) => row.map((coverage, x) => <span key={`${x}-${y}`} title={`Row ${y + 1}, column ${x + 1}: ${Math.round(coverage * 100)}% covered`} style={{ minHeight: 0, borderRadius: 3, background: coverage < .95 ? '#dc735e' : session.scan!.photo_coverage!.overlap[y][x] < .5 ? '#e0b551' : '#5b966b' }} />))}
          </div>
          <p className={ui.hint}>Coverage confirms where photos exist; it does not certify focus or dimensional accuracy.</p>
        </details>}
        </div>
      </aside>
      <div className={`${st.stageWrap} ${ed.modelingViewport}`}>
        <div className={st.toolbar}>
          <div className={ui.segmented} role="group" aria-label="Drawing tools" title="Drag on the mat to draw a shape (Shift = square, Alt = corner-to-corner for circles)">
            {([[null, '↖'], ['rect', '▭'], ['slot', '⬭'], ['circle', '○'], ['hex', '⬡']] as [Exclude<ShapeKind, 'poly'> | null, string][]).map(([k, icon]) => (
              <button key={icon} type="button" className={drawKind === k ? ui.segActive : ''} onClick={() => setDrawKind(k)}
                aria-pressed={drawKind === k} title={k ? `Draw a ${k}` : 'Select / edit (V)'}>{icon} {k === null ? 'Select' : k === 'rect' ? 'Rectangle' : k === 'circle' ? 'Circle' : k === 'slot' ? 'Slot' : 'Hex'}</button>
            ))}
          </div>
          <span className={st.toolbarSpacer} />
          <button type="button" className={ui.btn} disabled={!hist?.past.length} onClick={undo} aria-label="Undo outline edit" title="Undo outline edit (⌘/Ctrl Z)">↶</button>
          <button type="button" className={ui.btn} disabled={!hist?.future.length} onClick={redo} aria-label="Redo outline edit" title="Redo outline edit (⌘/Ctrl Shift Z)">↷</button>
        </div>
        <div className={ed.guidance} aria-live="polite">
          <div><strong>{drawKind ? `Draw a ${drawKind === 'rect' ? 'rectangle' : drawKind}` : selected ? `Editing ${selected.name}` : drawn.length ? 'Choose an outline to refine' : 'Your scan is ready'}</strong>
          {drawKind ? 'Drag across the mat. Choose Select when you are finished.' : selected ? 'Drag an edge handle to reshape. Changes can be undone.' : drawn.length ? 'Select a tool on the canvas or in the tool list.' : 'Start with Detect tools, or click a tool on the scan to outline it.'}</div>
          <span>Drag to orbit · scroll to zoom · right-drag to pan</span>
        </div>
        {recovery && <div className={ed.notice} role="status"><span>Previous tool list available.</span><button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => { setTools(prev => [...prev.filter(t => t.session_id !== session.id), ...recovery]); setRecovery(null); setSelectedId(null); }}>Undo list change</button><button className={ui.iconBtn} aria-label="Dismiss undo message" onClick={() => setRecovery(null)}>×</button></div>}
        <ScanViewer session={session} tools={[...mine, ...shapes]} selectedId={selectedId} onSelect={(id) => selectTool(id)} softMm={softMm}
          tickedIds={combineIds} onToggleSelect={toggleCombine}
          edit={{
            onEditStart: (id) => { const t = tools.find((x) => x.id === id); if (t) record(t); },
            onEdit: (id, poly) => commit(id, poly, false),
            onCreate: (x, y) => {
              if (!canClick) { setError('Click-to-outline needs scan height data or the HQ-SAM checkpoint. Use Auto-detect instead.'); return; }
              addTool([{ x: Math.round(x), y: Math.round(y), label: 'pos' }], null);
            },
            onHint: (id, x, y, label) => updateTool(id, (t) => ({ ...t, points: [...t.points, { x: Math.round(x), y: Math.round(y), label }] })),
            onSplit: (id, line) => { const t = tools.find((x) => x.id === id); if (t) void doSplit(t, line); },
            onDrawShape: (a, bpt, mods) => {
              if (!drawKind) return;
              const made = shapeFromDrag(drawKind, a, bpt, mods);
              if (!made) return;
              setTools((prev) => [...prev, shapeTool(made.spec, 20, tools, made.at)]);   // stays armed: draw several in a row
            },
          }} drawArmed={drawKind !== null} splitArmed={splitArm} />
        {error && <div role="alert" className={ui.error}>{error}</div>}
      </div>

      <aside className={`${ui.panel} ${ed.inspector} ${ed.properties}`} aria-label="Object properties">
        <div className={ed.dockTitle}>Properties <span>{selected ? 'Outline' : 'No selection'}</span></div>
        <div className={ed.inspectorBody}>
          <label className={ed.slider} title="How far along the outline a dragged point carries its neighbours. 0 = move that one point only. The wheel changes it while you drag.">
            <span className={ui.label}>Reshape radius</span>
            <input type="range" min={0} max={120} step={1} value={softMm} disabled={!editable} onChange={(e) => setSoftMm(Number(e.target.value))} />
            <span className={ui.unit}>{softMm ? `${softMm} mm` : 'off'}</span>
          </label>
        {selected ? (

          <div className={ui.section}>
            <h2 className={ui.panelTitle}>Edit outline</h2>
            <label className={ui.label}>Tool name<input className={ui.input} value={selected.name} onChange={e => updateTool(selected.id, t => ({ ...t, name: e.target.value }), false)} /></label>
            {selected.image_url && <details className={ui.disclosure}>
              <summary>Tool overhead image · {selected.image_source === 'single_photo' ? 'one complete photo' : 'stitched views'}</summary>
              <a href={`${API_BASE_URL}${selected.image_url}`} target="_blank" rel="noreferrer">
                {/* Native image endpoint is local and session-specific. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`${API_BASE_URL}${selected.image_url}`} alt={`Overhead source photo of ${selected.name} with its detected outline`} style={{ width: '100%', maxHeight: 300, objectFit: 'contain', borderRadius: 8 }} />
              </a>
              <p className={ui.hint}>Source photo and original detected outline. Click to inspect it at full size.</p>
            </details>}
            {editable ? (
              <>
                <div className={ed.stat}><span>Handles</span><span>{editable.polygon_px.length}{sel.size ? ` · ${sel.size} selected` : ''}</span></div>
                <div className={ed.stat}><span>Area</span><span>{(editable.area_mm2 / 100).toFixed(2)} cm²</span></div>
                {shapeLabel && <div className={ed.stat}><span>Shape</span><span>{shapeLabel}</span></div>}
                {editable.auto_polygon_px && (
                  <div className={ed.stat}><span>vs. detected</span><span>{((editable.area_mm2 / Math.max(1e-6, polygonArea(editable.auto_polygon_px) * mpp * mpp) - 1) * 100).toFixed(1)} %</span></div>
                )}
                <div className={ui.row} style={{ marginTop: 8 }}>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={!hist?.past.length} onClick={undo} title="⌘Z">Undo</button>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={!hist?.future.length} onClick={redo} title="⇧⌘Z">Redo</button>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={!editable.edited} onClick={resetAuto}>Reset to detected</button>
                </div>
                {sel.size > 0 && (
                  <div className={ui.row}>
                    <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => rotateSel(-5)} title="Shift+R">↺ 5°</button>
                    <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => rotateSel(5)} title="R">↻ 5°</button>
                    <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => doSmoothSelection(1.5)}>Smooth selected</button>
                    <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnDanger}`} disabled={editable.polygon_px.length - sel.size < 3} onClick={deleteSel} title="Delete">Delete {sel.size}</button>
                  </div>
                )}
                <div className={ui.row}>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => doSmoothAll(1.0)} title="Gaussian smoothing, 1 mm, whole outline">Smooth 1 mm</button>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => doSmoothAll(2.5)} title="Gaussian smoothing, 2.5 mm, whole outline">Smooth 2.5 mm</button>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={snapBusy} title="Slide every point outward to where the tool actually meets the mat — the detected edge sits half-way up the wall"
                    onClick={() => void doSnapBase(editable)}>{snapBusy ? <span className={ui.spinner} /> : '⤢'} Snap to base</button>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={doSimplify} title="Fewer handles (0.3 mm tolerance)">Fewer handles</button>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={doResample} title="One handle per millimetre">More handles</button>
                </div>
                <details className={ui.disclosure}><summary>Shape fitting &amp; advanced editing</summary><div>
                <div className={ui.row}>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} title="Swap this line for the rectangle, capsule, circle or hexagon that covers it best"
                    onClick={() => { if (!simplifyToShape(editable, true)) setError(`${editable.name} is too irregular for a simple shape — lower "fit at least" under Auto-detect to force one.`); }}>
                    ◻ Simple shape
                  </button>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} title="Score every primitive against this outline and pick one yourself"
                    onClick={() => { setShapeFits(fitShapes(editable.polygon_px)); setShowAlts(true); }}>Other shapes…</button>
                </div>
                {shapeFits && showAlts && (
                  <div className={ui.row}>
                    {shapeFits.slice(0, 4).map((f, i) => (
                      <button key={i} type="button" className={`${ui.btn} ${ui.btnSm}`} title={`covers ${Math.round(f.iou * 100)}% of the traced outline`}
                        onClick={() => { commit(editable.id, shapeRing(f.polygon)); setShapeFits(fitShapes(f.polygon)); setShowAlts(false); }}>
                        {f.label.split(' ')[0]} {Math.round(f.iou * 100)}%
                      </button>
                    ))}
                  </div>
                )}
                <div className={ui.row}>
                  <button type="button" className={`${ui.btn} ${ui.btnSm}`} title="Print this outline at 1:1 so you can lay the paper on the real tool"
                    onClick={() => doPrint([editable])}>🖨 Print 1:1{sheetsFor(editable) > 1 ? ` · ${sheetsFor(editable)} sheets` : ''}</button>
                  <select className={ui.select} style={{ width: 'auto' }} value={pageSize} onChange={(e) => setPageSize(e.target.value as PageSize)} title="Paper size">
                    {(Object.keys(PAGES) as PageSize[]).map((k) => <option key={k} value={k}>{PAGES[k].label}</option>)}
                  </select>
                </div>
                <div className={ui.row}>
                  <button type="button" className={`${ui.btn} ${ui.btnSm} ${splitArm ? ui.btnActive : ''}`} disabled={selected.pending}
                    title="Two tools came out as one? Click, then drag a line across the join (or ⌘/Ctrl+drag)" onClick={() => setSplitArm(!splitArm)}>
                    {splitArm ? 'Drag a line across the join…' : '✂ Split in two'}
                  </button>
                  {selected.box && <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => updateTool(selected.id, (t) => ({ ...t, box: null }))}>Clear box</button>}
                  {selected.points.length > 0 && <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => updateTool(selected.id, (t) => ({ ...t, points: t.points.slice(0, -1) }))}>Undo hint</button>}
                </div>
                </div></details>
                <p className={ui.hint}>The dashed amber line is what the scanner found. Clearance and export smoothing are added later — edit the line where the tool actually ends.</p>
              </>
            ) : (
              <>
                <p className={ui.hint}>{selected.pending ? 'Outlining…' : selected.error ?? 'No outline yet.'} Shift+click inside the tool to add a hint, Alt+click on anything that should be left out.</p>
                <div className={ui.row}>
                  {selected.points.length > 0 && <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => updateTool(selected.id, (t) => ({ ...t, points: t.points.slice(0, -1) }))}>Undo hint</button>}
                  {selected.box && <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => updateTool(selected.id, (t) => ({ ...t, box: null }))}>Clear box</button>}
                  <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnDanger}`} onClick={() => removeTool(selected.id)}>Remove</button>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className={ui.section}>
            <p className={ui.hint}>Click a tool on the image to fix its outline by hand, or click bare mat to outline something the detector missed.</p>
          </div>
        )}
        </div>
        <div className={ed.footer}>
          <p>{ready ? `${ready} outline${ready === 1 ? '' : 's'} ready for the foam layout` : 'Add at least one outline to continue'}</p>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${ui.btnBlock}`} disabled={!ready || autoBusy || mine.some(t => t.pending)} onClick={onContinue}>
            Arrange foam layout →
          </button>
        </div>
      </aside>
    </div>
  );
}
