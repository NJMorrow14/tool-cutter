'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ui from './ui.module.css';
import st from './stage.module.css';
import ed from './editor.module.css';
import ThreeViewer from './ThreeViewer';
import Layout3D from './Layout3D';
import { API_BASE_URL, autoLayout, buildLayoutBody, computeLayout, createSession, downloadBlob, exportLayout, exportRelief, getReliefGrid, resolveDepth, type ExportFormat, type ReliefFormat, type ReliefGrid } from '../lib/api';
import { arcLengths, depthColorMap, depthKey, fmt, fmtMm, polyToPath, polygonArea, ringsToPath, shapeFromDrag, shapeFromPoints, shapeName, shapePolygon, shapeTool, softDragRing } from '../lib/geom';
import AddShape from './AddShape';
import type { LayoutResponse, LayoutSettings, SessionInfo, ShapeKind, ShapeSpec, Tool } from '../lib/types';

const MESH_ACCEPT = '.ply,.obj,.stl,.glb,.gltf,.off,.xyz';

import { splitOutlineTool, verticesInBox, editToolOutline, resizeShapeTool } from '../lib/outline-edit';
import { zoomViewport, frameViewport, MIN_ZOOM, MAX_ZOOM, type Viewport } from '../lib/viewport';
import type { ToolChange } from '../lib/workspace-history';

interface Props {
  historyRevision: number;
  captureTools: () => ToolChange;
  selectedId: string | null;
  setSelectedId: (id: string | null) => void;
  panel: 'foam' | 'tools' | 'export';
  setPanel: (panel: 'foam' | 'tools' | 'export') => void;
  /** re-run the height-map tool finder on the current scan */
  onRedetect?: () => void;
  finding?: string | null;
  hasHeight: boolean;
  onObjectUploaded: (info: SessionInfo) => void;
  tools: Tool[];
  setTools: React.Dispatch<React.SetStateAction<Tool[]>>;
  settings: LayoutSettings;
  setSettings: (s: LayoutSettings) => void;
  matSize: { width_mm: number; height_mm: number };
  setMatSize: (m: { width_mm: number; height_mm: number }) => void;
}

type DragState = { id: string; startMm: { x: number; y: number }; delta: { x: number; y: number } };

export default function LayoutStep({ hasHeight, onObjectUploaded, tools, setTools, settings, setSettings, matSize, setMatSize, selectedId, setSelectedId, panel, setPanel, onRedetect, finding = null, captureTools, historyRevision }: Props) {
  const [view, setView] = useState<Viewport | null>(null);
  const [panning, setPanning] = useState(false);
  const panRef = useRef<{ clientX:number; clientY:number; view:Viewport; inverse:DOMMatrix } | null>(null);
  useEffect(() => { setView(null); }, [matSize.width_mm, matSize.height_mm]);
  const [layout, setLayout] = useState<LayoutResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [vertices, setVertices] = useState<Set<number>>(new Set());
  const [gesture, setGesture] = useState<{ kind: 'split' | 'box'; id: string; a: {x:number;y:number}; b: {x:number;y:number} } | null>(null);
  useEffect(() => { setVertices(new Set()); }, [selectedId, historyRevision]);
  useEffect(() => { setGesture(null); }, [historyRevision]);
  const [query, setQuery] = useState('');
  // Tools ticked for a group move / delete. Kept apart from selectedId: one tool is being EDITED
  // (its panel is open), several are being GATHERED. Ctrl/Cmd/Shift-click a row or a tool on the
  // canvas toggles a tick; a plain click still single-selects.
  const [pickedIds, setPickedIds] = useState<string[]>([]);
  const [notchMode, setNotchMode] = useState(false);
  const [drag, setDrag] = useState<DragState | null>(null);
  // Node editing on the sheet (Nolan, 2026-10-02: "make the options for adding shapes and editing nodes/vectors
  // available in the Layout and export view as well"). The sheet shows the SERVER's placed rings (with clearance),
  // so handles are computed client-side from the tool's own outline with the /api/layout convention — rotate about
  // the area centroid, then translate by offset_mm — and a drag is rotated back into the tool's frame before the
  // same soft falloff Outlines uses is applied.
  const [vdrag, setVdrag] = useState<{ id: string; idx: number; startMm: { x: number; y: number }; poly0: number[][]; arc: number[]; per: number; theta: number; selected: Set<number> } | null>(null);
  const [softMm, setSoftMm] = useState(25);
  const [addingShape, setAddingShape] = useState(false);
  const [exporting, setExporting] = useState<ExportFormat | ReliefFormat | null>(null);
  const [gcodeStats, setGcodeStats] = useState<Record<string, number> | null>(null);
  // form-fit (3D) pockets: the carved foam surface for the 3D preview, refreshed as the layout changes
  const [reliefGrid, setReliefGrid] = useState<ReliefGrid | null>(null);
  const [reliefBusy, setReliefBusy] = useState(false);
  const reliefReq = useRef(0);
  const [show3d, setShow3d] = useState(false);
  const [view3d, setView3d] = useState(settings.pocket_style === 'relief');   // a carved block is best judged in 3D
  const selectTool = (id: string | null) => { setSelectedId(id); if (id) setPanel('tools'); };
  const [gapMm, setGapMm] = useState(8);
  const [packDir, setPackDir] = useState<'columns' | 'rows'>('columns');
  const [packing, setPacking] = useState(false);
  const [unplaced, setUnplaced] = useState<string[]>([]);
  // canvas drawing: pick a tool, drag (or click points for a polygon) on the sheet
  const [drawTool, setDrawTool] = useState<ShapeKind | null>(null);
  const [drawDrag, setDrawDrag] = useState<{ a: { x: number; y: number }; b: { x: number; y: number }; shift: boolean; alt: boolean } | null>(null);
  const [polyPts, setPolyPts] = useState<{ x: number; y: number }[]>([]);
  const [polyHover, setPolyHover] = useState<{ x: number; y: number } | null>(null);
  const [drawThick, setDrawThick] = useState(20);
  const [drawRadius, setDrawRadius] = useState(4);
  const [importing, setImporting] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const reqRef = useRef(0);
  useEffect(() => {
    setDrag(null); setVdrag(null); setDrawDrag(null); setPolyPts([]); setPickedIds([]); setPacking(false);
  }, [historyRevision]);
  const exportRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (panel === 'export') {
      exportRef.current?.focus({ preventScroll: true });
      exportRef.current?.scrollIntoView({ block: 'nearest' });
    }
  }, [panel]);

  const mat = { width_mm: matSize.width_mm, height_mm: matSize.height_mm };
  const body = useMemo(() => buildLayoutBody(tools, mat, settings), [tools, mat.width_mm, mat.height_mm, settings]); // eslint-disable-line react-hooks/exhaustive-deps

  const reliefOn = settings.pocket_style === 'relief' || tools.some((t) => t.pocket_style === 'relief');
  // the carved surface for the 3D view (a 2 mm preview grid; exports use the full resolution setting)
  useEffect(() => {
    if (!reliefOn) { setReliefGrid(null); return; }
    if (drag) return;
    const id = ++reliefReq.current;
    const timer = setTimeout(async () => {
      setReliefBusy(true);
      try {
        // finer preview on a small insert; a 2 mm grid read as "quantized" even where the depth map itself was smooth
        const area = mat.width_mm * mat.height_mm;
        const g = await getReliefGrid(body, area <= 150000 ? 1 : area <= 300000 ? 1.5 : 2);
        if (reliefReq.current === id) setReliefGrid(g);
      } catch (err) {
        if (reliefReq.current === id) setError(err instanceof Error ? err.message : 'Relief preview failed');
      } finally {
        if (reliefReq.current === id) setReliefBusy(false);
      }
    }, 400);
    return () => clearTimeout(timer);
  }, [body, drag, reliefOn]);

  // recompute processed outlines whenever inputs change (debounced)
  useEffect(() => {
    if (drag) return; // wait until the drop
    const id = ++reqRef.current;
    const timer = setTimeout(async () => {
      setBusy(true);
      try {
        const res = await computeLayout(body);
        if (reqRef.current === id) {
          setLayout(res);
          setError(null);
        }
      } catch (err) {
        if (reqRef.current === id) setError(err instanceof Error ? err.message : 'Layout failed');
      } finally {
        if (reqRef.current === id) setBusy(false);
      }
    }, 160);
    return () => clearTimeout(timer);
  }, [body, drag]);

  const selected = tools.find((t) => t.id === selectedId) ?? null;
  const updateTool = (id: string, fn: (t: Tool) => Tool) => setTools((prev) => prev.map((t) => (t.id === id ? fn(t) : t)));
  const updateShape = (t: Tool, patch: Partial<ShapeSpec>) => {
    const spec = { ...(t.shape as ShapeSpec), ...patch };
    const poly = shapePolygon(spec);
    const autoNamed = t.shape ? t.name === shapeName(t.shape) : false;
    updateTool(t.id, (x) => ({ ...x, shape: spec, polygon_mm: poly, area_mm2: polygonArea(poly), name: autoNamed ? shapeName(spec) : x.name }));
  };

  /** Area centroid (shoelace) — the pivot /api/layout rotates about. */
  const centroidOf = (poly: number[][]) => {
    let a = 0, cx = 0, cy = 0;
    for (let i = 0; i < poly.length; i++) {
      const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
      const w = x0 * y1 - x1 * y0; a += w; cx += (x0 + x1) * w; cy += (y0 + y1) * w;
    }
    if (Math.abs(a) < 1e-9) { const n = poly.length; return { x: poly.reduce((s, p) => s + p[0], 0) / n, y: poly.reduce((s, p) => s + p[1], 0) / n }; }
    return { x: cx / (3 * a), y: cy / (3 * a) };
  };
  /** The tool's own outline placed as the sheet shows it (no clearance). */
  const placedRing = (t: Tool): number[][] => {
    const c = centroidOf(t.polygon_mm); const th = (t.rotation_deg * Math.PI) / 180; const cs = Math.cos(th), sn = Math.sin(th);
    return t.polygon_mm.map(([x, y]) => [c.x + (x - c.x) * cs - (y - c.y) * sn + t.offset_mm.x, c.y + (x - c.x) * sn + (y - c.y) * cs + t.offset_mm.y]);
  };
  const onVertexDown = (t: Tool, idx: number, e: React.PointerEvent) => {
    if (beginGesture(t.id, e)) return;
    if (e.button !== 0 || drawTool || notchMode) return;
    e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    const { arc, per } = arcLengths(t.polygon_mm);
    setVdrag({ id: t.id, idx, startMm: toMm(e), poly0: t.polygon_mm, arc, per, theta: (t.rotation_deg * Math.PI) / 180, selected: vertices.has(idx) ? new Set(vertices) : new Set() });
    if (!vertices.has(idx)) setVertices(new Set());
  };
  const applyVertexDrag = (e: React.PointerEvent) => {
    if (!vdrag) return;
    const p = toMm(e);
    // the drag happened in sheet space; the outline lives in the tool's frame, rotated by theta
    const dx = p.x - vdrag.startMm.x, dy = p.y - vdrag.startMm.y;
    const cs = Math.cos(-vdrag.theta), sn = Math.sin(-vdrag.theta);
    const ldx = dx * cs - dy * sn, ldy = dx * sn + dy * cs;
    const poly = vdrag.selected.size ? vdrag.poly0.map((p,i) => vdrag.selected.has(i) ? [p[0]+ldx,p[1]+ldy] : p) : softDragRing(vdrag.poly0, vdrag.idx, vdrag.arc, vdrag.per, ldx, ldy, softMm);
    // a parametric shape (circle, rect, slot, hex) resizes from the drag and stays a shape; free outlines reshape
    updateTool(vdrag.id, t => resizeShapeTool({ ...t, polygon_mm: vdrag.poly0 }, poly) ?? editToolOutline(t, poly));
  };

  // ------------------------------------------------------------------ multi-select (ticks)
  const isPicked = (id: string) => pickedIds.includes(id);
  const togglePick = (id: string) => setPickedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  /** true when the event carries the "add to the tick set" modifier (Cmd on macOS — Chrome turns Ctrl+click into a right-click there) */
  const pickMod = (e: { metaKey: boolean; ctrlKey: boolean; shiftKey: boolean }) => e.metaKey || e.ctrlKey || e.shiftKey;
  const removeTools = (ids: string[]) => {
    setTools((prev) => prev.filter((t) => !ids.includes(t.id)));
    setPickedIds((prev) => prev.filter((id) => !ids.includes(id)));
    if (selectedId && ids.includes(selectedId)) selectTool(null);
  };
  /** the tools a keyboard move/rotate applies to: the ticked set wins, otherwise the single selection */
  const activeIds = pickedIds.length ? pickedIds : selected ? [selected.id] : [];
  const nudgeIds = (ids: string[], dx: number, dy: number) =>
    setTools((prev) => prev.map((t) => (ids.includes(t.id) ? { ...t, offset_mm: { x: t.offset_mm.x + dx, y: t.offset_mm.y + dy } } : t)));
  const rotateIds = (ids: string[], deg: number) =>
    setTools((prev) => prev.map((t) => (ids.includes(t.id) ? { ...t, rotation_deg: ((t.rotation_deg + deg) % 360 + 360) % 360 } : t)));
  // a tool can disappear (auto layout never removes one, but a new capture replaces the list) — drop dead ticks
  const toolIdKey = tools.map((t) => t.id).join('|');
  useEffect(() => {
    setPickedIds((prev) => {
      const live = prev.filter((id) => tools.some((t) => t.id === id));
      return live.length === prev.length ? prev : live;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toolIdKey]);

  // ------------------------------------------------------------------ mm-space pointer math
  const margin = 6;
  const fitView = { x: -margin, y: -margin, w: Math.max(1, mat.width_mm) + 2 * margin, h: Math.max(1, mat.height_mm) + 2 * margin };
  const vb = view ?? fitView;
  const zoom = fitView.w / vb.w;
  const zoomBy = (factor: number) => setView(zoomViewport(vb, {x:vb.x+vb.w/2,y:vb.y+vb.h/2}, factor, fitView.w));
  const frameSelected = () => setView(selected ? frameViewport(placedRing(selected), fitView) : null);
  const wheelRef = useRef<(e:WheelEvent)=>void>(()=>{});
  wheelRef.current = e => {
    e.preventDefault();
    if (drag || vdrag || gesture || drawDrag || panRef.current) return;
    const matrix = svgRef.current?.getScreenCTM();
    if (!matrix) return;
    const anchor = new DOMPoint(e.clientX,e.clientY).matrixTransform(matrix.inverse());
    const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 300 : 1);
    const u=(anchor.x-vb.x)/vb.w, v=(anchor.y-vb.y)/vb.h;
    setView(previous=>{const current=previous??fitView;return zoomViewport(current,{x:current.x+u*current.w,y:current.y+v*current.h},Math.exp(-Math.max(-500,Math.min(500,delta))*0.002),fitView.w);});
  };
  useEffect(() => {
    const svg=svgRef.current;
    if(!svg)return;
    const wheel=(e:WheelEvent)=>wheelRef.current(e);
    svg.addEventListener('wheel',wheel,{passive:false});
    return ()=>svg.removeEventListener('wheel',wheel);
  }, []);
  const beginPan = (e:React.PointerEvent<SVGSVGElement>) => {
    if((e.button !== 1 && e.button !== 2) || e.ctrlKey || e.metaKey || e.shiftKey) return;
    const matrix=e.currentTarget.getScreenCTM();
    if(!matrix)return;
    e.preventDefault(); e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    panRef.current={clientX:e.clientX,clientY:e.clientY,view:vb,inverse:matrix.inverse()};setPanning(true);
  };
  const toMm = (e: React.PointerEvent): { x: number; y: number } => {
    const point = new DOMPoint(e.clientX, e.clientY).matrixTransform(svgRef.current!.getScreenCTM()!.inverse());
    return { x: point.x, y: point.y };
  };

  const beginGesture = (id: string | null, e: React.PointerEvent) => {
    if (!id || drawTool || notchMode || (!e.ctrlKey && !e.metaKey && !e.shiftKey) || (e.button !== 0 && !(e.ctrlKey && e.button === 2))) return false;
    e.preventDefault(); e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    selectTool(id);
    const p = toMm(e);
    setGesture({ kind: e.ctrlKey || e.metaKey ? 'split' : 'box', id, a:p, b:p });
    return true;
  };

  const finishPolygon = (pts: { x: number; y: number }[]) => {
    const made = shapeFromPoints(pts);
    setPolyPts([]);
    if (!made) return;
    const t = shapeTool(made.spec, drawThick, tools, made.at);
    setTools((prev) => [...prev, t]);
    selectTool(t.id);
  };
  const onSheetDown = (e: React.PointerEvent) => {
    if (beginGesture(selectedId, e)) return;
    if (drawTool && e.button === 0) {
      const p = toMm(e);
      if (drawTool === 'poly') {
        // click to add a point; click the first point (or double-click / Enter) to close
        if (polyPts.length >= 3 && Math.hypot(p.x - polyPts[0].x, p.y - polyPts[0].y) < 4) { finishPolygon(polyPts); return; }
        setPolyPts([...polyPts, p]);
        return;
      }
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
      setDrawDrag({ a: p, b: p, shift: e.shiftKey, alt: e.altKey });
      return;
    }
    if (notchMode && selectedId) {
      // notch anywhere near the selected tool: the backend snaps it onto the outline
      const p = toMm(e);
      updateTool(selectedId, (t) => ({ ...t, notch: { x_mm: p.x, y_mm: p.y, diameter_mm: settings.notch_diameter_mm } }));
      setNotchMode(false);
      return;
    }
    selectTool(null);
  };
  const onToolDown = (id: string, e: React.PointerEvent) => {
    if (beginGesture(id, e)) return;
    if (e.button !== 0) return;
    if (drawTool) { onSheetDown(e); return; }          // drawing over an existing tool is allowed
    e.stopPropagation();
    if (pickMod(e)) { togglePick(id); return; }        // Cmd/Ctrl/Shift-click ticks the tool instead of grabbing it
    selectTool(id);
    const p = toMm(e);
    if (notchMode) {
      updateTool(id, (t) => ({ ...t, notch: { x_mm: p.x, y_mm: p.y, diameter_mm: settings.notch_diameter_mm } }));
      setNotchMode(false);
      return;
    }
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    setDrag({ id, startMm: p, delta: { x: 0, y: 0 } });
  };
  const onMove = (e: React.PointerEvent) => {
    const pan=panRef.current;
    if(pan){const dx=e.clientX-pan.clientX,dy=e.clientY-pan.clientY;setView({...pan.view,x:pan.view.x-dx*pan.inverse.a-dy*pan.inverse.c,y:pan.view.y-dx*pan.inverse.b-dy*pan.inverse.d});return;}
    if (gesture) { setGesture({ ...gesture, b:toMm(e) }); return; }
    if (vdrag) { applyVertexDrag(e); return; }
    if (drawDrag) { setDrawDrag({ ...drawDrag, b: toMm(e), shift: e.shiftKey, alt: e.altKey }); return; }
    if (drawTool === 'poly') { setPolyHover(toMm(e)); return; }
    if (!drag) return;
    const p = toMm(e);
    setDrag({ ...drag, delta: { x: p.x - drag.startMm.x, y: p.y - drag.startMm.y } });
  };
  const onUp = (e: React.PointerEvent) => {
    if(panRef.current){panRef.current=null;setPanning(false);return;}
    if (gesture) {
      const tool = tools.find(t=>t.id===gesture.id), b=toMm(e);
      if (tool) {
        if (gesture.kind === 'box') setVertices(new Set(verticesInBox(placedRing(tool),[gesture.a.x,gesture.a.y],[b.x,b.y])));
        else if (Math.hypot(b.x-gesture.a.x,b.y-gesture.a.y)>1) {
          try {
            const parts=splitOutlineTool(tool,[[gesture.a.x,gesture.a.y],[b.x,b.y]]);
            setTools(prev=>prev.flatMap(t=>t.id===tool.id?parts:[t])); selectTool(parts[0].id); setError(null);
          } catch(err) { setError(err instanceof Error?err.message:'Split failed'); }
        }
      }
      setGesture(null); return;
    }
    if (vdrag) { setVdrag(null); return; }
    if (drawDrag && drawTool && drawTool !== 'poly') {
      const made = shapeFromDrag(drawTool, drawDrag.a, drawDrag.b, { shift: drawDrag.shift, alt: drawDrag.alt }, drawRadius);
      setDrawDrag(null);
      if (made) {
        const t = shapeTool(made.spec, drawThick, tools, made.at);
        setTools((prev) => [...prev, t]);
        selectTool(t.id);
      }
      return;
    }
    if (!drag) return;
    const { id, delta } = drag;
    setDrag(null);
    if (Math.hypot(delta.x, delta.y) < 0.2) return;
    updateTool(id, (t) => ({
      ...t,
      offset_mm: { x: Math.round((t.offset_mm.x + delta.x) * 10) / 10, y: Math.round((t.offset_mm.y + delta.y) * 10) / 10 },
      notch: t.notch ? { ...t.notch, x_mm: t.notch.x_mm + delta.x, y_mm: t.notch.y_mm + delta.y } : null,
    }));
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || (e.target as HTMLElement)?.closest('input, textarea, select, [contenteditable]')) return;
      if (e.key === 'Escape' && panRef.current) { panRef.current=null;setPanning(false);e.preventDefault();return; }
      if (e.key === 'Escape' && gesture) { setGesture(null); e.preventDefault(); return; }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (!view3d && e.key.toLowerCase() === 'f') { frameSelected(); e.preventDefault(); return; }
      if (!view3d && e.key === '0') { setView(null); e.preventDefault(); return; }
      if (e.key === 'Escape' && (gesture || vertices.size)) { setGesture(null); setVertices(new Set()); e.preventDefault(); return; }
      if (e.key === 'Escape' && (drawTool || polyPts.length)) { setPolyPts([]); setDrawDrag(null); setDrawTool(null); e.preventDefault(); return; }
      if (e.key === 'Enter' && drawTool === 'poly' && polyPts.length >= 3) { finishPolygon(polyPts); e.preventDefault(); return; }
      if (e.key === 'Backspace' && drawTool === 'poly' && polyPts.length) { setPolyPts(polyPts.slice(0, -1)); e.preventDefault(); return; }
      const toolKeys: Record<string, ShapeKind | null> = { v: null, '1': 'rect', '2': 'slot', '3': 'circle', '4': 'hex', '5': 'poly' };
      if (e.key.toLowerCase() in toolKeys && !e.metaKey && !e.ctrlKey) { setDrawTool(toolKeys[e.key.toLowerCase()]); setPolyPts([]); return; }
      // Escape clears the ticks first, so it does not also drop the tool being edited
      if (e.key === 'Escape' && pickedIds.length) { setPickedIds([]); e.preventDefault(); return; }
      if ((e.key === 'Delete' || e.key === 'Backspace') && pickedIds.length) { removeTools(pickedIds); e.preventDefault(); return; }
      if (vertices.size && selected) {
        const step = e.shiftKey ? 5 : 1;
        const delta: Record<string, number[]> = { ArrowLeft:[-step,0], ArrowRight:[step,0], ArrowUp:[0,-step], ArrowDown:[0,step] };
        if (delta[e.key]) {
          const [dx,dy]=delta[e.key], angle=-selected.rotation_deg*Math.PI/180;
          updateTool(selected.id,t=>editToolOutline(t,t.polygon_mm.map((p,i)=>vertices.has(i)?[p[0]+dx*Math.cos(angle)-dy*Math.sin(angle),p[1]+dx*Math.sin(angle)+dy*Math.cos(angle)]:p)));
          e.preventDefault(); return;
        }
        if (e.key==='Delete' || e.key==='Backspace') {
          if(selected.polygon_mm.length-vertices.size>=3){updateTool(selected.id,t=>editToolOutline(t,t.polygon_mm.filter((_,i)=>!vertices.has(i))));setVertices(new Set());}
          e.preventDefault(); return;
        }
      }
      const ids = activeIds;
      if (!ids.length) return;
      const step = e.shiftKey ? 5 : 1;
      if (e.key === 'ArrowLeft') nudgeIds(ids, -step, 0);
      else if (e.key === 'ArrowRight') nudgeIds(ids, step, 0);
      else if (e.key === 'ArrowUp') nudgeIds(ids, 0, -step);
      else if (e.key === 'ArrowDown') nudgeIds(ids, 0, step);
      else if (e.key.toLowerCase() === 'r') rotateIds(ids, e.shiftKey ? -90 : 90);
      else if (e.key === '[' || e.key === ']') rotateIds(ids, e.key === ']' ? 5 : -5);
      else if (e.key === 'Escape') selectTool(null);
      else if (e.key === 'Delete' || e.key === 'Backspace') removeTools(ids);
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, drawTool, polyPts, drawThick, pickedIds, tools, gesture, vertices, view3d, matSize.width_mm, matSize.height_mm]);

  // ------------------------------------------------------------------ auto layout
  const runAutoLayout = async () => {
    const apply = captureTools();
    setPacking(true);
    setError(null);
    try {
      const res = await autoLayout(mat, tools, { gap_mm: gapMm, margin_mm: 10, allow_rotate: true, direction: packDir });
      if (!apply.isCurrent()) return;
      const by = new Map(res.placements.map((p) => [p.id, p]));
      apply((prev) => prev.map((t) => {
        const p = by.get(t.id);
        return p ? { ...t, rotation_deg: p.rotation_deg, offset_mm: p.offset_mm, notch: null } : t;
      }));
      setUnplaced(res.unplaced);
      selectTool(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Auto layout failed');
    } finally {
      setPacking(false);
    }
  };

  // ------------------------------------------------------------------ export
  const exportOpts = {
    include_mat: settings.include_mat, include_labels: settings.include_labels, fill_mode: settings.fill_mode,
    mat_thickness_mm: settings.mat_thickness_mm, filename: `foam_${fmt(mat.width_mm, 0)}x${fmt(mat.height_mm, 0)}mm`,
  };
  const doExport = async (format: ExportFormat) => {
    setExporting(format);
    setError(null);
    try {
      const { blob, filename } = await exportLayout(body, format, exportOpts);
      downloadBlob(blob, filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setExporting(null);
    }
  };
  const doReliefExport = async (format: ReliefFormat) => {
    setExporting(format);
    setError(null);
    try {
      const { blob, filename, stats } = await exportRelief(body, format, exportOpts);
      downloadBlob(blob, filename);
      if (stats) setGcodeStats(stats);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Relief export failed');
    } finally {
      setExporting(null);
    }
  };
  const fetchStl = useCallback(async () => {
    // the standalone 3D preview shows the block that will actually be cut: carved when the pocket style is form-fit
    if (reliefOn) {
      const resp = await fetch(`${API_BASE_URL}/api/relief`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...body, format: 'stl', export: exportOpts, inline: true, relief: { ...body.relief, resolution_mm: Math.max(1.5, body.relief.resolution_mm) } }) });
      if (!resp.ok) throw new Error('Relief STL failed');
      return resp.arrayBuffer();
    }
    const { blob } = await exportLayout(body, 'stl', exportOpts, true);
    return blob.arrayBuffer();
  }, [body, exportOpts.mat_thickness_mm, reliefOn]); // eslint-disable-line react-hooks/exhaustive-deps
  const fetchToolsStl = useCallback(async () => {
    const { blob } = await exportLayout(body, 'stl_tools', exportOpts, true);
    return blob.arrayBuffer();
  }, [body, exportOpts.mat_thickness_mm]); // eslint-disable-line react-hooks/exhaustive-deps

  const importObject = async (file: File | null | undefined) => {
    if (!file) return;
    setImporting(true);
    setError(null);
    try {
      const info = await createSession(file, 'auto', 'object');
      onObjectUploaded(info);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  // ------------------------------------------------------------------ render helpers
  const colorMap = useMemo(() => depthColorMap(tools.filter((t) => t.include).map((t) => resolveDepth(t, settings))), [tools, settings]);
  const issues = (layout?.tools ?? []).filter((t) => t.outside_mat || t.overlaps.length);
  const gridStep = mat.width_mm > 500 ? 100 : 50;
  const gridLines: number[] = [];
  for (let v = gridStep; v < Math.max(mat.width_mm, mat.height_mm); v += gridStep) gridLines.push(v);
  const set = (patch: Partial<LayoutSettings>) => setSettings({ ...settings, ...patch });

  return (
    <div className={`${ed.modelingGrid} ${ed.layoutWorkspace}`}>
      <aside className={`${ui.panel} ${ed.inspector} ${ed.outliner}`} aria-label="Tools">
        <div className={ed.dockTitle}>Tools <span>{tools.length} objects</span></div>
        <div className={ed.inspectorBody}>
        <div className={ui.section}>
          <div className={ui.rowBetween}>
            <h2 className={ui.panelTitle}>Add tools</h2>
            <div className={ui.row}>
              <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={importing} onClick={() => fileRef.current?.click()} title="Import a PLY/OBJ/GLB/STL of one tool; it is laid flat and placed on the mat">
                {importing ? <span className={ui.spinner} /> : '+'} 3D model
              </button>
            </div>
            <input ref={fileRef} type="file" accept={MESH_ACCEPT} hidden onChange={(e) => { void importObject(e.target.files?.[0]); e.target.value = ''; }} />
          </div>
          {pickedIds.length > 0 && (
            <div className={ui.row} style={{ outline: '2px solid var(--accent, #f26a1b)', outlineOffset: '-2px', borderRadius: 6, padding: '6px 8px' }}>
              <strong style={{ fontSize: 12 }}>{pickedIds.length} ticked</strong>
              <span className={ui.row} style={{ gap: 4 }} role="group" aria-label="Move the ticked tools">
                {([['←', -1, 0], ['↑', 0, -1], ['↓', 0, 1], ['→', 1, 0]] as [string, number, number][]).map(([icon, dx, dy]) => (
                  <button key={icon} type="button" className={`${ui.btn} ${ui.btnSm}`} aria-label={`Move the ticked tools ${icon}`}
                    title="Move every ticked tool 1 mm (Shift = 5 mm) — the arrow keys do the same"
                    onClick={(e) => nudgeIds(pickedIds, dx * (e.shiftKey ? 5 : 1), dy * (e.shiftKey ? 5 : 1))}>{icon}</button>
                ))}
              </span>
              <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => setPickedIds([])}>Clear ticks</button>
              <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnDanger}`} title="Delete / Backspace does the same" onClick={() => removeTools(pickedIds)}>✕ Remove {pickedIds.length}</button>
              <span className={ui.hint} style={{ flexBasis: '100%' }}>Arrow keys move them together (Shift = 5 mm) · R rotates · Delete removes · Escape clears the ticks.</span>
            </div>
          )}
          <button type="button" className={`${ui.btn} ${ui.btnBlock}`} onClick={() => { setAddingShape(true); setView3d(false); }}>＋ Add shape</button>
            {onRedetect && <button type="button" className={`${ui.btn} ${ui.btnBlock}`} disabled={!!finding} onClick={onRedetect} title="Find the tools again from the height map (everything standing up from the drawer floor)">{finding ? <span className={ui.spinner} /> : '↻'} {finding ?? 'Re-find tools from scan'}</button>}
          <input className={ui.input} type="search" aria-label="Search tools" placeholder="Search tools…" value={query} onChange={e => setQuery(e.target.value)} />
          {!tools.length && <div className={ui.emptyState}><strong>Your insert starts here</strong>Add a shape or import a 3D tool model, then arrange it on the sheet.</div>}
          {!!tools.length && !tools.some(t => t.name.toLowerCase().includes(query.toLowerCase())) && <p className={ui.hint}>No matching tools.</p>}
          <div className={ui.list}>
            {tools.filter((t) => t.polygon_mm.length >= 3 && t.name.toLowerCase().includes(query.toLowerCase())).map((t) => (
              <div key={t.id} className={`${ui.toolRow} ${t.id === selectedId ? ui.toolRowActive : ''}`}
                onClick={(e) => (pickMod(e) ? togglePick(t.id) : selectTool(t.id))}
                style={{ gridTemplateColumns: '10px 22px minmax(0,1fr) 22px', opacity: t.include ? 1 : 0.5, ...(isPicked(t.id) ? { outline: '2px solid var(--accent, #f26a1b)', outlineOffset: '-2px' } : null) }}>
                <span className={ui.swatch} style={{ background: t.color }} />
                <button type="button" aria-pressed={isPicked(t.id)} aria-label={isPicked(t.id) ? `Untick ${t.name}` : `Tick ${t.name}`}
                  title="Ctrl/Cmd-click to tick several tools and move or remove them together"
                  style={{ background: 'none', border: 0, padding: 0, cursor: 'pointer', font: 'inherit' }}
                  onClick={(e) => { e.stopPropagation();          // the row handles it too; a TOGGLE must not fire twice
                                    togglePick(t.id); }}>{isPicked(t.id) ? '☑ ' : '☐ '}</button>
                <button type="button" className={ui.toolName} aria-pressed={t.id === selectedId} onClick={(e) => { e.stopPropagation(); if (pickMod(e)) togglePick(t.id); else selectTool(t.id); }}>{t.name}</button>
                <span className={`${ui.toolMeta} ${ed.toolDepth}`}>{t.source === 'object' ? '3D · ' : t.source === 'shape' ? '▭ · ' : ''}{t.include ? fmtMm(resolveDepth(t, settings)) : 'skipped'}{resolveDepth(t, settings) === null && t.include ? 'through' : ''}</span>
                <input type="checkbox" checked={t.include} aria-label={`Include ${t.name} in export`} title="Include in export" onClick={(e) => e.stopPropagation()} onChange={(e) => updateTool(t.id, (x) => ({ ...x, include: e.target.checked }))} />
              </div>
            ))}
            {tools.filter((t) => t.polygon_mm.length >= 3).length > 1 && !pickedIds.length &&
              <p className={ui.hint}>Ctrl/Cmd-click tools (in the list or on the sheet) to tick several, then move or remove them together.</p>}
          </div>
        </div>

        </div>
      </aside>
      <div className={`${st.stageWrap} ${ed.modelingViewport}`}>
        <div className={st.toolbar}>
          <div className={ui.segmented} role="group" aria-label="Pocket drawing tools" title="Draw a pocket on the sheet: V select · 1 rectangle · 2 slot · 3 circle · 4 hexagon · 5 polygon">
            {([[null, '↖', 'Select / move (V)'], ['rect', '▭', 'Rectangle: drag corner to corner (Shift = square) (1)'], ['slot', '⬭', 'Slot: drag corner to corner (2)'], ['circle', '○', 'Circle: drag from centre (Alt = corner to corner) (3)'], ['hex', '⬡', 'Hexagon: drag from centre (4)'], ['poly', '⬠', 'Polygon: click points, click the first point or press Enter to close (5)']] as [ShapeKind | null, string, string][]).map(([k, icon, tip]) => (
              <button key={String(k)} type="button" className={drawTool === k ? ui.segActive : ''} aria-label={k ? `Draw ${k}` : 'Select and move'} aria-pressed={drawTool === k} title={tip} onClick={() => { setDrawTool(k); setView3d(false); setPolyPts([]); setNotchMode(false); }}>{icon} {k === null ? 'Select' : k === 'rect' ? 'Rect' : k === 'poly' ? 'Polygon' : k}</button>
            ))}

          </div>
          {drawTool && (
            <>
              <label className={ui.field} style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }} title="Thickness of the item that goes in the drawn pocket">
                <input className={`${ui.input} ${ui.inputSm}`} type="number" min={1} step={0.5} value={drawThick} onChange={(e) => setDrawThick(Number(e.target.value))} style={{ width: 58 }} />
                <span className={ui.unit}>mm thick</span>
              </label>
              {drawTool === 'rect' && (
                <label className={ui.field} style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }} title="Corner radius">
                  <input className={`${ui.input} ${ui.inputSm}`} type="number" min={0} step={0.5} value={drawRadius} onChange={(e) => setDrawRadius(Number(e.target.value))} style={{ width: 52 }} />
                  <span className={ui.unit}>r</span>
                </label>
              )}
            </>
          )}
          <button type="button" className={`${ui.btn} ${ui.btnSm} ${notchMode ? ui.btnActive : ''}`} onClick={() => { setNotchMode(!notchMode); setView3d(false); setDrawTool(null); }}>
            {notchMode ? 'Click an outline to place the notch…' : '☝ Add finger notch'}
          </button>
          <div className={ui.segmented} role="group" aria-label="Layout view" title="Cutting sheet (top view) or the foam block in 3D">
            <button type="button" aria-pressed={!view3d} className={!view3d ? ui.segActive : ''} onClick={() => setView3d(false)}>2D plan</button>
            <button type="button" aria-pressed={view3d} className={view3d ? ui.segActive : ''} onClick={() => setView3d(true)}>3D preview</button>
          </div>
          <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={packing} onClick={runAutoLayout} title="Pack all included tools onto the mat, biggest first, long sides aligned, with this much foam between pockets">
            {packing ? <span className={ui.spinner} /> : '⊞'} Auto layout
          </button>


          <span className={st.toolbarSpacer} />
          {busy && <span className={ui.spinner} />}
          <span className={ui.hint}>{fmt(mat.width_mm, 1)} × {fmt(mat.height_mm, 1)} mm</span>
        </div>
        {!view3d && <div className={ed.sheetNavigation} role="toolbar" aria-label="Insert zoom and navigation">
          <button type="button" className={`${ui.btn} ${ui.btnSm}`} aria-label="Zoom out" disabled={zoom <= MIN_ZOOM} onClick={()=>zoomBy(1/1.25)}>−</button>
          <output aria-label="Zoom level">{Math.round(zoom*100)}%</output>
          <button type="button" className={`${ui.btn} ${ui.btnSm}`} aria-label="Zoom in" disabled={zoom >= MAX_ZOOM} onClick={()=>zoomBy(1.25)}>+</button>
          <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={()=>setView(null)} title="Fit insert · 0">Fit insert</button>
          <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={!selected} onClick={frameSelected} title="Frame selected tool · F">Frame selected</button>
          <span>Scroll or pinch to zoom · Right-drag to pan</span>
        </div>}
        <div className={ed.guidance}><div><strong>{drawTool ? `Draw a ${drawTool}` : notchMode ? 'Place a finger notch' : 'Arrange your tools'}</strong>{drawTool === 'poly' ? 'Click corners, then Enter to finish. Escape cancels.' : drawTool ? 'Drag across the sheet to size the pocket. Escape returns to Select.' : notchMode ? 'Click an outline where you want to lift the tool out.' : 'Drag to move · Ctrl-drag to split · Shift-drag to select nodes.'}</div><span>{pickedIds.length > 0 ? `${pickedIds.length} ticked · arrow keys move them together · Delete removes them · Escape clears` : 'Arrow keys: nudge · R: rotate · Shift: larger steps · Ctrl/Cmd-click: tick several'}</span></div>
        {view3d && (
          <Layout3D historyRevision={historyRevision} tools={tools} layout={layout} mat={mat} settings={settings} selectedId={selectedId} onSelect={selectTool} reliefGrid={reliefGrid}
            pickedIds={pickedIds} onPick={togglePick}
            onMove={(id, dx, dy) => updateTool(id, (t) => ({
              ...t, offset_mm: { x: Math.round((t.offset_mm.x + dx) * 10) / 10, y: Math.round((t.offset_mm.y + dy) * 10) / 10 },
              notch: t.notch ? { ...t.notch, x_mm: t.notch.x_mm + dx, y_mm: t.notch.y_mm + dy } : null,
            }))} />
        )}
        {addingShape && <AddShape onClose={() => setAddingShape(false)} onAdd={(spec, thick) => {
          const t = shapeTool(spec, thick, tools); setTools((prev) => [...prev, t]); selectTool(t.id); setAddingShape(false);
        }} />}
        <svg ref={svgRef} className={st.sheet} viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`} style={{ aspectRatio: `${vb.w} / ${vb.h}`, cursor: panning ? 'grabbing' : drawTool || notchMode ? 'crosshair' : drag ? 'grabbing' : 'default', display: view3d ? 'none' : undefined }}
          onPointerDownCapture={beginPan} onPointerMove={onMove} onPointerUp={onUp} onContextMenu={e => e.preventDefault()} onPointerCancel={() => { panRef.current=null;setPanning(false);setGesture(null); setDrag(null); setVdrag(null); setDrawDrag(null); }} onPointerDown={onSheetDown} onDoubleClick={() => { if (drawTool === 'poly' && polyPts.length >= 3) finishPolygon(polyPts.slice(0, -1).length >= 3 ? polyPts.slice(0, -1) : polyPts); }}>
          <rect x={vb.x} y={vb.y} width={vb.w} height={vb.h} fill="#20242b" />
          <rect x={0} y={0} width={mat.width_mm} height={mat.height_mm} fill="#fff" stroke="#111827" strokeWidth={0.4} />
          {gridLines.map((v) => (
            <g key={v} stroke="#e5e7eb" strokeWidth={0.2}>
              {v < mat.width_mm && <line x1={v} x2={v} y1={0} y2={mat.height_mm} />}
              {v < mat.height_mm && <line x1={0} x2={mat.width_mm} y1={v} y2={v} />}
            </g>
          ))}
          {gridLines.filter((v) => v < mat.width_mm).map((v) => <text key={`tx${v}`} x={v} y={-1.5} fontSize={3} fill="#9ca3af" textAnchor="middle">{v}</text>)}
          {gridLines.filter((v) => v < mat.height_mm).map((v) => <text key={`ty${v}`} x={-1.5} y={v + 1} fontSize={3} fill="#9ca3af" textAnchor="end">{v}</text>)}

          {(layout?.tools ?? []).map((lt) => {
            const tool = tools.find((t) => t.id === lt.id);
            if (!tool) return null;
            const sel = lt.id === selectedId;
            const color = colorMap.get(depthKey(lt.depth_mm)) ?? '#ff0000';
            const bad = lt.outside_mat || lt.overlaps.length > 0;
            const d = drag && drag.id === lt.id ? drag.delta : null;
            return (
              <g key={lt.id} transform={d ? `translate(${d.x} ${d.y})` : undefined} style={{ cursor: drawTool || notchMode ? 'crosshair' : 'grab' }}
                onPointerDown={(e) => onToolDown(lt.id, e)}>
                <path d={ringsToPath(lt.rings)} fill={tool.color} fillOpacity={sel ? 0.35 : 0.18} fillRule="evenodd"
                  stroke={bad ? '#dc2626' : color} strokeWidth={sel ? 0.9 : 0.5} strokeDasharray={bad ? '2 1' : undefined} strokeLinejoin="round" />
                {isPicked(lt.id) && <path d={ringsToPath(lt.rings)} fill="none" fillRule="evenodd" stroke="#f26a1b" strokeWidth={1.3} strokeLinejoin="round" style={{ pointerEvents: 'none' }} />}
                {lt.notch && <circle cx={lt.notch.x_mm} cy={lt.notch.y_mm} r={lt.notch.diameter_mm / 2} fill="none" stroke="#f59e0b" strokeWidth={0.4} strokeDasharray="1 1" />}
                {lt.centroid_mm && settings.include_labels && (
                  <text x={lt.centroid_mm[0]} y={lt.centroid_mm[1]} fontSize={Math.max(3, Math.min(6, ((lt.bbox_mm?.[2] ?? 0) - (lt.bbox_mm?.[0] ?? 0)) / 12))}
                    fill="#374151" textAnchor="middle" dominantBaseline="middle" style={{ pointerEvents: 'none' }}>
                    {lt.name}{lt.depth_mm !== null ? ` · ${fmt(lt.depth_mm, 1)}` : ''}
                  </text>
                )}
              </g>
            );
          })}
          {(() => {
            // node handles for the one selected tool (not while drawing, not with a multi-tick)
            const t = tools.find((x) => x.id === selectedId);
            if (!t || drawTool || notchMode || pickedIds.length > 1 || t.polygon_mm.length < 3) return null;
            const ring = vdrag && vdrag.id === t.id ? placedRing(t) : placedRing(t);
            const r = Math.max(0.02, vb.w / 220);
            return (
              <g key="handles">
                <path d={polyToPath(ring)} fill="none" stroke="#ffffff" strokeWidth={r * 0.35} strokeDasharray={`${r} ${r}`} style={{ pointerEvents: 'none' }} />
                {ring.map(([x, y], i) => (
                  <circle key={i} cx={x} cy={y} r={vertices.has(i) ? r * 1.3 : r} fill={vertices.has(i) ? "#60a5fa" : "#ffffff"} stroke="#f26a1b" strokeWidth={r * 0.35} style={{ cursor: 'move' }}
                    onPointerDown={(e) => onVertexDown(t, i, e)} />
                ))}
              </g>
            );
          })()}
          {gesture && (gesture.kind === 'box' ? <rect x={Math.min(gesture.a.x,gesture.b.x)} y={Math.min(gesture.a.y,gesture.b.y)} width={Math.abs(gesture.b.x-gesture.a.x)} height={Math.abs(gesture.b.y-gesture.a.y)} fill="#60a5fa25" stroke="#60a5fa" strokeWidth={0.5} style={{pointerEvents:'none'}} /> : <line x1={gesture.a.x} y1={gesture.a.y} x2={gesture.b.x} y2={gesture.b.y} stroke="#ef4444" strokeWidth={0.7} style={{pointerEvents:'none'}} />)}
          {drawDrag && drawTool && drawTool !== 'poly' && (() => {
            const made = shapeFromDrag(drawTool, drawDrag.a, drawDrag.b, { shift: drawDrag.shift, alt: drawDrag.alt }, drawRadius);
            if (!made) return null;
            const poly = shapePolygon(made.spec).map(([x, y]) => [x + made.at.x, y + made.at.y]);
            const label = made.spec.kind === 'circle' ? `⌀${fmt(made.spec.w_mm, 1)}` : made.spec.kind === 'hex' ? `${fmt(made.spec.w_mm, 1)} AF` : `${fmt(made.spec.w_mm, 1)} × ${fmt(made.spec.h_mm, 1)}`;
            return (
              <g style={{ pointerEvents: 'none' }}>
                <path d={polyToPath(poly)} fill="rgba(242,106,27,0.18)" stroke="#f26a1b" strokeWidth={0.6} strokeDasharray="2 1.2" />
                <text x={made.at.x + made.spec.w_mm / 2} y={made.at.y - 2.5} fontSize={4} fill="#c2410c" textAnchor="middle" fontWeight={700}>{label}</text>
              </g>
            );
          })()}
          {drawTool === 'poly' && polyPts.length > 0 && (
            <g style={{ pointerEvents: 'none' }}>
              <path d={`M${[...polyPts, ...(polyHover ? [polyHover] : [])].map((p) => `${fmt(p.x)} ${fmt(p.y)}`).join(' L')}${polyPts.length >= 3 ? ' Z' : ''}`}
                fill={polyPts.length >= 3 ? 'rgba(242,106,27,0.18)' : 'none'} stroke="#f26a1b" strokeWidth={0.6} strokeDasharray="2 1.2" />
              {polyPts.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={i === 0 ? 2.2 : 1.4} fill={i === 0 ? '#f26a1b' : '#fff'} stroke="#c2410c" strokeWidth={0.5} />)}
            </g>
          )}
        </svg>
        <div className={st.legend}>
          <span>Pocket depth:</span>
          {Array.from(colorMap.entries()).filter(([k]) => k !== null).map(([k, c]) => (
            <span key={String(k)}><i className={st.legendSwatch} style={{ background: c }} />{k} mm</span>
          ))}
          {settings.depth_rule === 'through' && <span><i className={st.legendSwatch} style={{ background: '#ff0000' }} />through cut</span>}
          {settings.mirror && <span className={ui.badge}>mirrored (cut from the back)</span>}
        </div>
        {unplaced.length > 0 && (
          <div className={ui.warn}>Auto layout could not fit: {unplaced.map((id) => tools.find((t) => t.id === id)?.name ?? id).join(', ')} — try a smaller gap, a bigger mat, or skip a tool.</div>
        )}
        {issues.length > 0 && (
          <div className={ui.warn}>
            {issues.map((t) => (
              <div key={t.id}>
                <strong>{t.name}</strong>: {t.outside_mat ? 'extends past the mat edge' : ''}{t.outside_mat && t.overlaps.length ? '; ' : ''}
                {t.overlaps.length ? `overlaps ${t.overlaps.map((id) => tools.find((x) => x.id === id)?.name ?? id).join(', ')}` : ''}
              </div>
            ))}
          </div>
        )}
        {error && <div className={ui.error}>{error}</div>}
      </div>

      <aside className={`${ui.panel} ${ed.inspector} ${ed.properties}`} aria-label="Insert properties" data-history-fields>
        <div className={ed.inspectorTabs} style={{gridTemplateColumns:'repeat(3, 1fr)'}} role="tablist" aria-label="Layout settings">
          {(['foam', 'tools', 'export'] as const).map(key => <button type="button" role="tab" key={key} aria-selected={panel === key} onClick={() => setPanel(key)}>{key === 'foam' ? 'Foam & fit' : key === 'tools' ? 'Selected' : 'Export'}</button>)}
        </div>
        <div className={ed.inspectorBody}>
        {panel === 'foam' && <>
        <details className={ui.disclosure}><summary>Auto layout settings</summary><div className={ui.section}>
          <label className={ui.field} style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }} title="Foam left between neighbouring pockets">
            <input className={`${ui.input} ${ui.inputSm}`} type="number" min={2} max={40} step={1} value={gapMm} onChange={(e) => setGapMm(Number(e.target.value))} style={{ width: 58 }} />
            <span className={ui.unit}>mm gap</span>
          </label>
          <select className={`${ui.select} ${ui.inputSm}`} style={{ width: 190 }} value={packDir} onChange={(e) => setPackDir(e.target.value as 'columns' | 'rows')} title="Upright: tools stand vertical, side by side across the drawer. Lying: tools run along the drawer, stacked top to bottom.">
            <option value="columns">↕ upright, across</option>
            <option value="rows">↔ lying, stacked</option>
          </select>
        </div></details>
        <h2 className={ui.panelTitle}>Insert dimensions</h2>
        <div className={ui.section}>
          <div className={ui.grid2}>
            <label className={ui.field}><span className={ui.label}>Width (mm)</span>
              <input className={ui.input} type="number" step="0.1" onFocus={(e) => e.currentTarget.select()} value={mat.width_mm} onChange={(e) => setMatSize({ ...matSize, width_mm: Number(e.target.value) })} /></label>
            <label className={ui.field}><span className={ui.label}>Height (mm)</span>
              <input className={ui.input} type="number" step="0.1" onFocus={(e) => e.currentTarget.select()} value={mat.height_mm} onChange={(e) => setMatSize({ ...matSize, height_mm: Number(e.target.value) })} /></label>
          </div>
          <label className={ui.field}><span className={ui.label}>Foam thickness (mm)</span>
            <input className={ui.input} type="number" step="1" min="3" value={settings.mat_thickness_mm} onChange={(e) => set({ mat_thickness_mm: Number(e.target.value) })} /></label>
          <p className={ui.hint}>Changing the size here only changes the export frame; tools keep their calibrated scale and position from the top-left corner.</p>
        </div>

        <div className={ui.section}>
          <h2 className={ui.panelTitle}>Pocket fit</h2>
          <label className={ui.field}><span className={ui.label}>Pocket style</span>
            <select className={ui.select} value={settings.pocket_style} onChange={(e) => set({ pocket_style: e.target.value as LayoutSettings['pocket_style'] })}>
              <option value="flat">Flat pockets — one depth per tool (laser, router, knife)</option>
              <option value="relief">Form-fit 3D — the floor follows the scanned tool (3-axis CNC)</option>
            </select></label>
          {settings.pocket_style === 'relief' && (
            <>
              <div className={ui.grid2}>
                <label className={ui.field}><span className={ui.label}>Surface smoothing (mm)</span>
                  <input className={ui.input} type="number" step="0.5" min="0" value={settings.relief_smooth_mm} onChange={(e) => set({ relief_smooth_mm: Number(e.target.value) })} /></label>
                <label className={ui.field}><span className={ui.label}>Depth clearance (mm)</span>
                  <input className={ui.input} type="number" step="0.25" min="0" value={settings.relief_clearance_mm} onChange={(e) => set({ relief_clearance_mm: Number(e.target.value) })} /></label>
                <label className={ui.field}><span className={ui.label}>Depth-map grid (mm)</span>
                  <input className={ui.input} type="number" step="0.25" min="0.3" max="5" value={settings.relief_resolution_mm} onChange={(e) => set({ relief_resolution_mm: Number(e.target.value) })} /></label>
              </div>
              <label className={ui.checkbox}><input type="checkbox" checked={settings.relief_clean_solids} onChange={(e) => set({ relief_clean_solids: e.target.checked })} /> Clean solids — photo-guided smoothing, mirror symmetry, box / cylinder / extrusion fits where the scan supports them</label>
              {settings.relief_clean_solids && <label className={ui.checkbox}><input type="checkbox" checked={settings.relief_semantic} onChange={(e) => set({ relief_semantic: e.target.checked })} /> Let the model name each solid (box, cylinder…) to guide the fit</label>}
              <p className={ui.hint}>Each pocket is carved to the tool's own scanned shape (smoothed, plus the depth clearance) so it nests in form. Tools without a scan, and drawn shapes, stay flat at their depth. {reliefBusy ? 'Updating the 3D preview…' : reliefGrid ? `Deepest cut ${fmt(reliefGrid.max_depth_mm, 1)} mm of ${settings.mat_thickness_mm} mm foam.` : ''}
                {reliefGrid && !reliefBusy && (() => { const c: Record<string, number> = {}; for (const t of reliefGrid.tools) if (t.solid && t.solid !== 'freeform') c[t.solid] = (c[t.solid] ?? 0) + 1; const parts = Object.entries(c).map(([k, n]) => `${n} ${k.replace('_', ' ')}`); return parts.length ? ` Fitted as solids: ${parts.join(', ')}.` : ''; })()}</p>
            </>
          )}
          <div className={ui.grid2}>
            <label className={ui.field}><span className={ui.label}>Clearance (mm)</span>
              <input className={ui.input} type="number" step="0.25" min="0" value={settings.default_clearance_mm} onChange={(e) => set({ default_clearance_mm: Number(e.target.value) })} /></label>
            <label className={ui.field}><span className={ui.label}>Edge smoothing (mm)</span>
              <input className={ui.input} type="number" step="0.1" min="0" value={settings.smoothing_mm} onChange={(e) => set({ smoothing_mm: Number(e.target.value) })} /></label>
          </div>
          <label className={ui.field}><span className={ui.label}>Finger notch diameter (mm)</span>
            <input className={ui.input} type="number" step="1" min="5" value={settings.notch_diameter_mm} onChange={(e) => set({ notch_diameter_mm: Number(e.target.value) })} /></label>
          <p className={ui.hint}>Clearance grows each outline so the tool drops in; 1–2 mm is typical for foam (use the higher end for LiDAR captures). Edge smoothing removes sensor wobble and rounds corners by about that radius; straight edges stay straight.</p>
        </div>

        {settings.pocket_style !== 'relief' && <div className={ui.section}>
          <h2 className={ui.panelTitle}>Pocket depth</h2>
          <label className={ui.field}><span className={ui.label}>Rule</span>
            <select className={ui.select} value={settings.depth_rule} onChange={(e) => set({ depth_rule: e.target.value as LayoutSettings['depth_rule'] })}>
              <option value="measured_minus">Tool thickness − X (tool sits proud)</option>
              <option value="fraction">Fraction of tool thickness</option>
              <option value="measured">Full tool thickness</option>
              <option value="through">Cut all the way through</option>
            </select></label>
          {settings.depth_rule === 'measured_minus' && (
            <label className={ui.field}><span className={ui.label}>Tool exposed above foam (mm)</span>
              <input className={ui.input} type="number" step="0.5" min="0" value={settings.depth_minus_mm} onChange={(e) => set({ depth_minus_mm: Number(e.target.value) })} /></label>
          )}
          {settings.depth_rule === 'fraction' && (
            <label className={ui.field}><span className={ui.label}>Fraction (0–1)</span>
              <input className={ui.input} type="number" step="0.05" min="0.1" max="1" value={settings.depth_fraction} onChange={(e) => set({ depth_fraction: Number(e.target.value) })} /></label>
          )}
          {settings.depth_rule !== 'through' && (
            <label className={ui.field}><span className={ui.label}>Depth when no thickness is known (mm)</span>
              <input className={ui.input} type="number" step="0.5" min="0.5" value={settings.fallback_depth_mm} onChange={(e) => set({ fallback_depth_mm: Number(e.target.value) })} /></label>
          )}
          <p className={ui.hint}>
            {hasHeight
              ? 'Thickness comes from the 3D data where available (tools showing a value). Others use the fallback; override any tool below.'
              : 'No height data yet: enter each tool\'s thickness below (calipers), or use the fallback depth.'}
          </p>
        </div>}

        </>}
        {panel === 'tools' && <>
        {!selected && <div className={ui.emptyState}><strong>Select a tool</strong>Choose a tool from the list or canvas to adjust its shape, position, and pocket depth.</div>}

        {selected && (
          <div className={ui.section}>
            <div className={ui.rowBetween}>
              <h2 className={ui.panelTitle} style={{ color: selected.color }}>Tool properties</h2>
              <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnGhost}`} onClick={() => selectTool(null)}>Done</button>
            </div>
            <p className={ui.hint}>{vertices.size ? `${vertices.size} nodes selected. Drag a blue node to move the group. Escape clears selection.` : 'Shift-drag a box to select nodes. Ctrl-drag across the outline to split it.'}</p>
            <label className={ui.label} title="How far along the outline a dragged node carries its neighbours (0 = just that node)" style={{ marginLeft: 8 }}>Node influence
              <input className={ui.input} type="number" min={0} max={200} step={5} value={softMm} onChange={(e) => setSoftMm(Math.max(0, Number(e.target.value) || 0))} style={{ width: 64 }} /> mm</label>
            <label className={ui.field}><span className={ui.label}>Tool name</span><input className={ui.input} value={selected.name} onChange={e => updateTool(selected.id, t => ({ ...t, name: e.target.value }))} /></label>

            {selected.shape && selected.shape.kind !== 'poly' && (
              <div className={ui.grid2}>
                <label className={ui.field}><span className={ui.label}>{selected.shape.kind === 'circle' ? 'Diameter (mm)' : selected.shape.kind === 'hex' ? 'Across flats (mm)' : 'Width (mm)'}</span>
                  <input className={ui.input} type="number" min={2} step={0.5} value={selected.shape.w_mm}
                    onChange={(e) => updateShape(selected, { w_mm: Number(e.target.value) })} /></label>
                {(selected.shape.kind === 'rect' || selected.shape.kind === 'slot') && (
                  <label className={ui.field}><span className={ui.label}>Height (mm)</span>
                    <input className={ui.input} type="number" min={2} step={0.5} value={selected.shape.h_mm}
                      onChange={(e) => updateShape(selected, { h_mm: Number(e.target.value) })} /></label>
                )}
                {selected.shape.kind === 'rect' && (
                  <label className={ui.field}><span className={ui.label}>Corner radius (mm)</span>
                    <input className={ui.input} type="number" min={0} step={0.5} value={selected.shape.r_mm}
                      onChange={(e) => updateShape(selected, { r_mm: Number(e.target.value) })} /></label>
                )}
              </div>
            )}
            {(() => {
              const fp = layout?.tools.find((lt) => lt.id === selected.id)?.footprint;
              if (!fp || fp.kind === 'none') return null;
              const label = fp.kind === 'rectangle' ? `rectangle ${fmt(fp.w_mm ?? 0, 1)} × ${fmt(fp.h_mm ?? 0, 1)} mm`
                : fp.kind === 'rounded_rectangle' ? `rounded rectangle ${fmt(fp.w_mm ?? 0, 1)} × ${fmt(fp.h_mm ?? 0, 1)} mm, r ${fmt(fp.r_mm ?? 0, 1)}`
                : fp.kind === 'capsule' ? `capsule ${fmt(fp.w_mm ?? 0, 1)} × ${fmt(fp.h_mm ?? 0, 1)} mm`
                : fp.kind === 'circle' ? `circle ⌀ ${fmt(fp.diameter_mm ?? 0, 1)} mm`
                : `smooth outline, ${fp.vertices ?? '–'} points`;
              return <p className={ui.hint}>Footprint: {label}{fp.iou != null ? ` (fits the scan ${Math.round(fp.iou * 100)} %)` : ''}.</p>;
            })()}
            <div className={ui.grid2}>
              {selected.depth_coverage != null && selected.depth_coverage < 0.5 && (
                <p className={ui.warn} style={{ gridColumn: '1 / -1' }}>The depth sensor saw only {Math.round(selected.depth_coverage * 100)}% of this object (black or glossy surface). Its height comes from the parts it did see{selected.measured_thickness_mm != null ? ` (${selected.measured_thickness_mm} mm)` : ''}; type the real thickness below if that is wrong.</p>
              )}
              <label className={ui.field}><span className={ui.label}>Thickness (mm)</span>
                <input className={ui.input} type="number" step="0.5" min="0" placeholder={selected.measured_thickness_mm !== null ? String(selected.measured_thickness_mm) : '—'}
                  value={selected.measured_thickness_mm ?? ''} onChange={(e) => updateTool(selected.id, (t) => ({ ...t, measured_thickness_mm: e.target.value === '' ? null : Number(e.target.value) }))} /></label>
              <label className={ui.field}><span className={ui.label}>Pocket depth override</span>
                <input className={ui.input} type="number" step="0.5" min="0" placeholder={fmtMm(resolveDepth({ ...selected, depth_mm: null }, settings))}
                  value={selected.depth_mm ?? ''} onChange={(e) => updateTool(selected.id, (t) => ({ ...t, depth_mm: e.target.value === '' ? null : Number(e.target.value) }))} /></label>
              <label className={ui.field}><span className={ui.label}>Clearance override</span>
                <input className={ui.input} type="number" step="0.25" min="0" placeholder={String(settings.default_clearance_mm)}
                  value={selected.clearance_mm ?? ''} onChange={(e) => updateTool(selected.id, (t) => ({ ...t, clearance_mm: e.target.value === '' ? null : Number(e.target.value) }))} /></label>
              <label className={ui.field}><span className={ui.label}>Rotation (°)</span>
                <input className={ui.input} type="number" step="1" value={selected.rotation_deg} onChange={(e) => updateTool(selected.id, (t) => ({ ...t, rotation_deg: Number(e.target.value) }))} /></label>
              {selected.source !== 'shape' && selected.session_id && (
                <label className={ui.field}><span className={ui.label}>Pocket style</span>
                  <select className={ui.select} value={selected.pocket_style ?? ''} onChange={(e) => updateTool(selected.id, (t) => ({ ...t, pocket_style: (e.target.value || undefined) as Tool['pocket_style'] }))}>
                    <option value="">Layout default ({settings.pocket_style === 'relief' ? 'form-fit 3D' : 'flat'})</option>
                    <option value="flat">Flat</option>
                    <option value="relief">Form-fit 3D</option>
                  </select></label>
              )}
            </div>
            <div className={ui.row}>
              <span className={ui.hint}>Offset {fmt(selected.offset_mm.x, 1)}, {fmt(selected.offset_mm.y, 1)} mm</span>
              <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={!selected.offset_mm.x && !selected.offset_mm.y && !selected.rotation_deg}
                onClick={() => updateTool(selected.id, (t) => ({ ...t, offset_mm: { x: 0, y: 0 }, rotation_deg: 0 }))}>Reset position</button>
              {selected.notch
                ? <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => updateTool(selected.id, (t) => ({ ...t, notch: null }))}>Remove notch</button>
                : <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => { setNotchMode(true); setView3d(false); }}>Add notch</button>}
            </div>
            <label className={ui.checkbox}><input type="checkbox" checked={selected.include} onChange={(e) => updateTool(selected.id, (t) => ({ ...t, include: e.target.checked }))} /> Include in export</label>
            <button type="button" className={`${ui.btn} ${ui.btnDanger}`} onClick={() => removeTools([selected.id])}>Remove tool</button>
          </div>
        )}

        </>}
        {panel === 'export' && <div className={ui.section}>
          <h2 ref={exportRef} tabIndex={-1} className={ui.panelTitle}>Download cutting files</h2>
          <p className={ui.hint}>{reliefOn ? 'Pocket style is form-fit 3D: the files below carve each pocket to the tool\'s scanned shape. The flat 2D files further down are the same layout as plain pockets.' : 'Choose the format your machine or design software accepts.'}</p>
          {reliefOn && (
            <div className={ui.section} style={{ paddingLeft: 0, paddingRight: 0 }}>
              <h2 className={ui.panelTitle}>Form-fit 3D cutting files</h2>
              <p className={ui.hint}>The whole block as one depth map at {settings.relief_resolution_mm} mm. STL for 3D CAM, a 16-bit depth map for relief-capable CAM (Aspire, VCarve, Carbide Create Pro), or ready G-code for a 3-axis GRBL machine with a flat end mill.</p>
              <div className={ui.grid2}>
                <label className={ui.field}><span className={ui.label}>Cutter ⌀ (mm)</span>
                  <input className={ui.input} type="number" step="0.5" min="0.5" value={settings.cnc_cutter_mm} onChange={(e) => set({ cnc_cutter_mm: Number(e.target.value) })} /></label>
                <label className={ui.field}><span className={ui.label}>Stepover (mm)</span>
                  <input className={ui.input} type="number" step="0.25" min="0.2" value={settings.cnc_stepover_mm} onChange={(e) => set({ cnc_stepover_mm: Number(e.target.value) })} /></label>
                <label className={ui.field}><span className={ui.label}>Step-down (mm)</span>
                  <input className={ui.input} type="number" step="0.5" min="0.5" value={settings.cnc_stepdown_mm} onChange={(e) => set({ cnc_stepdown_mm: Number(e.target.value) })} /></label>
                <label className={ui.field}><span className={ui.label}>Feed (mm/min)</span>
                  <input className={ui.input} type="number" step="50" min="10" value={settings.cnc_feed_mm_min} onChange={(e) => set({ cnc_feed_mm_min: Number(e.target.value) })} /></label>
                <label className={ui.field}><span className={ui.label}>Plunge (mm/min)</span>
                  <input className={ui.input} type="number" step="50" min="10" value={settings.cnc_plunge_mm_min} onChange={(e) => set({ cnc_plunge_mm_min: Number(e.target.value) })} /></label>
                <label className={ui.field}><span className={ui.label}>Safe Z (mm)</span>
                  <input className={ui.input} type="number" step="1" min="1" value={settings.cnc_safe_z_mm} onChange={(e) => set({ cnc_safe_z_mm: Number(e.target.value) })} /></label>
                <label className={ui.field}><span className={ui.label}>Spindle (rpm)</span>
                  <input className={ui.input} type="number" step="500" min="0" value={settings.cnc_spindle_rpm} onChange={(e) => set({ cnc_spindle_rpm: Number(e.target.value) })} /></label>
              </div>
              <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${ui.btnBlock}`} disabled={!!exporting} onClick={() => doReliefExport('package')}
                title="One zip: G-code, carved STL, 16-bit depth map, flat SVG + DXF, and a job sheet with every setting and pocket depth">
                {exporting === 'package' ? <span className={ui.spinner} /> : '⬇'} Download CNC package (zip)</button>
              <div className={ui.grid2}>
                <button type="button" className={ui.btn} disabled={!!exporting} onClick={() => doReliefExport('gcode')}>{exporting === 'gcode' ? <span className={ui.spinner} /> : null} G-code only (.nc)</button>
                <button type="button" className={ui.btn} disabled={!!exporting} onClick={() => doReliefExport('stl')}>{exporting === 'stl' ? <span className={ui.spinner} /> : null} 3D STL only</button>
                <button type="button" className={ui.btn} disabled={!!exporting} onClick={() => doReliefExport('png')}>{exporting === 'png' ? <span className={ui.spinner} /> : null} Depth map only (16-bit PNG)</button>
              </div>
              {gcodeStats && <p className={ui.hint}>G-code: {gcodeStats.layers} layers, {gcodeStats.runs} passes, {fmt(gcodeStats.cut_length_mm / 1000, 1)} m of cutting, deepest {fmt(gcodeStats.deepest_mm, 1)} mm, about {gcodeStats.est_minutes} min at the set feed. Z = 0 at the foam top, origin at the near-left corner; check it in a simulator before the first cut.</p>}
            </div>
          )}
          {reliefOn && <h2 className={ui.panelTitle}>Flat 2D files</h2>}
          <label className={ui.checkbox}><input type="checkbox" checked={settings.include_mat} onChange={(e) => set({ include_mat: e.target.checked })} /> Mat outline rectangle</label>
          <label className={ui.checkbox}><input type="checkbox" checked={settings.include_labels} onChange={(e) => set({ include_labels: e.target.checked })} /> Tool names + depth legend (separate layer)</label>
          <label className={ui.checkbox}><input type="checkbox" checked={settings.mirror} onChange={(e) => set({ mirror: e.target.checked })} /> Mirror (cutting from the underside)</label>
          <label className={ui.field}><span className={ui.label}>SVG style</span>
            <select className={ui.select} value={settings.fill_mode} onChange={(e) => set({ fill_mode: e.target.value as 'none' | 'fill' })}>
              <option value="none">Hairline outlines, colored by depth (laser / CNC)</option>
              <option value="fill">Filled shapes (engrave / Cricut)</option>
            </select></label>
          <div className={ui.grid2}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={!!exporting || !tools.some(t => t.include && t.polygon_mm.length >= 3)} onClick={() => doExport('svg')}>{exporting === 'svg' ? <span className={ui.spinner} /> : null} Download SVG</button>
            <button type="button" className={ui.btn} disabled={!!exporting || !tools.some(t => t.include && t.polygon_mm.length >= 3)} onClick={() => doExport('dxf')}>{exporting === 'dxf' ? <span className={ui.spinner} /> : null} Download DXF</button>
            <button type="button" className={ui.btn} disabled={!!exporting || !tools.some(t => t.include && t.polygon_mm.length >= 3)} onClick={() => doExport('stl')}>{exporting === 'stl' ? <span className={ui.spinner} /> : null} Download STL</button>
            <button type="button" className={ui.btn} onClick={() => setShow3d(true)}>Preview 3D</button>
          </div>
          <p className={ui.hint}>SVG is exactly {fmt(mat.width_mm, 1)} × {fmt(mat.height_mm, 1)} mm with one path per tool; stroke color = pocket depth. DXF has one layer per depth. STL is the foam block with pockets for CNC / CAM. Preview 3D shows the block with or without the tools sitting in it.</p>
        </div>}
        </div>
        {panel !== 'export' && <div className={ed.footer}><p>{tools.filter(t => t.include && t.polygon_mm.length >= 3).length} pockets · {fmt(mat.width_mm, 0)} × {fmt(mat.height_mm, 0)} mm insert</p><button type="button" className={`${ui.btn} ${ui.btnPrimary} ${ui.btnBlock}`} onClick={() => setPanel('export')}>Export cutting files →</button></div>}
      </aside>
      {show3d && <ThreeViewer title={`Foam block ${fmt(mat.width_mm, 0)} × ${fmt(mat.height_mm, 0)} × ${settings.mat_thickness_mm} mm`} fetchStl={fetchStl} fetchToolsStl={fetchToolsStl} onClose={() => setShow3d(false)} />}
    </div>
  );
}
