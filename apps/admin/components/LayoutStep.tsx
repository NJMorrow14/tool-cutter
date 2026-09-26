'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ui from './ui.module.css';
import st from './stage.module.css';
import ed from './editor.module.css';
import ThreeViewer from './ThreeViewer';
import Layout3D from './Layout3D';
import { autoLayout, buildLayoutBody, computeLayout, createSession, downloadBlob, exportLayout, resolveDepth, type ExportFormat } from '../lib/api';
import { depthColorMap, depthKey, fmt, fmtMm, polyToPath, polygonArea, ringsToPath, shapeFromDrag, shapeFromPoints, shapeName, shapePolygon, shapeTool } from '../lib/geom';
import type { LayoutResponse, LayoutSettings, SessionInfo, ShapeKind, ShapeSpec, Tool } from '../lib/types';

const MESH_ACCEPT = '.ply,.obj,.stl,.glb,.gltf,.off,.xyz';

interface Props {
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

export default function LayoutStep({ hasHeight, onObjectUploaded, tools, setTools, settings, setSettings, matSize, setMatSize }: Props) {
  const [layout, setLayout] = useState<LayoutResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Tools ticked for a group move / delete. Kept apart from selectedId: one tool is being EDITED
  // (its panel is open), several are being GATHERED. Ctrl/Cmd/Shift-click a row or a tool on the
  // canvas toggles a tick; a plain click still single-selects.
  const [pickedIds, setPickedIds] = useState<string[]>([]);
  const [notchMode, setNotchMode] = useState(false);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const [show3d, setShow3d] = useState(false);
  const [view3d, setView3d] = useState(false);
  const [panel, setPanel] = useState<'foam' | 'tools' | 'export'>('foam');
  useEffect(() => { if (selectedId) setPanel('tools'); }, [selectedId]);
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

  const mat = { width_mm: matSize.width_mm, height_mm: matSize.height_mm };
  const body = useMemo(() => buildLayoutBody(tools, mat, settings), [tools, mat.width_mm, mat.height_mm, settings]); // eslint-disable-line react-hooks/exhaustive-deps

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

  // ------------------------------------------------------------------ multi-select (ticks)
  const isPicked = (id: string) => pickedIds.includes(id);
  const togglePick = (id: string) => setPickedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  /** true when the event carries the "add to the tick set" modifier (Cmd on macOS — Chrome turns Ctrl+click into a right-click there) */
  const pickMod = (e: { metaKey: boolean; ctrlKey: boolean; shiftKey: boolean }) => e.metaKey || e.ctrlKey || e.shiftKey;
  const removeTools = (ids: string[]) => {
    setTools((prev) => prev.filter((t) => !ids.includes(t.id)));
    setPickedIds((prev) => prev.filter((id) => !ids.includes(id)));
    if (selectedId && ids.includes(selectedId)) setSelectedId(null);
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
  const vb = { x: -margin, y: -margin, w: mat.width_mm + 2 * margin, h: mat.height_mm + 2 * margin };
  const toMm = (e: React.PointerEvent): { x: number; y: number } => {
    const r = svgRef.current!.getBoundingClientRect();
    return { x: vb.x + ((e.clientX - r.left) / r.width) * vb.w, y: vb.y + ((e.clientY - r.top) / r.height) * vb.h };
  };

  const finishPolygon = (pts: { x: number; y: number }[]) => {
    const made = shapeFromPoints(pts);
    setPolyPts([]);
    if (!made) return;
    const t = shapeTool(made.spec, drawThick, tools, made.at);
    setTools((prev) => [...prev, t]);
    setSelectedId(t.id);
  };
  const onSheetDown = (e: React.PointerEvent) => {
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
    setSelectedId(null);
  };
  const onToolDown = (id: string, e: React.PointerEvent) => {
    if (e.button !== 0) return;
    if (drawTool) { onSheetDown(e); return; }          // drawing over an existing tool is allowed
    e.stopPropagation();
    if (pickMod(e)) { togglePick(id); return; }        // Cmd/Ctrl/Shift-click ticks the tool instead of grabbing it
    setSelectedId(id);
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
    if (drawDrag) { setDrawDrag({ ...drawDrag, b: toMm(e), shift: e.shiftKey, alt: e.altKey }); return; }
    if (drawTool === 'poly') { setPolyHover(toMm(e)); return; }
    if (!drag) return;
    const p = toMm(e);
    setDrag({ ...drag, delta: { x: p.x - drag.startMm.x, y: p.y - drag.startMm.y } });
  };
  const onUp = () => {
    if (drawDrag && drawTool && drawTool !== 'poly') {
      const made = shapeFromDrag(drawTool, drawDrag.a, drawDrag.b, { shift: drawDrag.shift, alt: drawDrag.alt }, drawRadius);
      setDrawDrag(null);
      if (made) {
        const t = shapeTool(made.spec, drawThick, tools, made.at);
        setTools((prev) => [...prev, t]);
        setSelectedId(t.id);
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
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'SELECT') return;
      if (e.key === 'Escape' && (drawTool || polyPts.length)) { setPolyPts([]); setDrawDrag(null); setDrawTool(null); e.preventDefault(); return; }
      if (e.key === 'Enter' && drawTool === 'poly' && polyPts.length >= 3) { finishPolygon(polyPts); e.preventDefault(); return; }
      if (e.key === 'Backspace' && drawTool === 'poly' && polyPts.length) { setPolyPts(polyPts.slice(0, -1)); e.preventDefault(); return; }
      const toolKeys: Record<string, ShapeKind | null> = { v: null, '1': 'rect', '2': 'slot', '3': 'circle', '4': 'hex', '5': 'poly' };
      if (e.key.toLowerCase() in toolKeys && !e.metaKey && !e.ctrlKey) { setDrawTool(toolKeys[e.key.toLowerCase()]); setPolyPts([]); return; }
      // Escape clears the ticks first, so it does not also drop the tool being edited
      if (e.key === 'Escape' && pickedIds.length) { setPickedIds([]); e.preventDefault(); return; }
      if ((e.key === 'Delete' || e.key === 'Backspace') && pickedIds.length) { removeTools(pickedIds); e.preventDefault(); return; }
      const ids = activeIds;
      if (!ids.length) return;
      const step = e.shiftKey ? 5 : 1;
      if (e.key === 'ArrowLeft') nudgeIds(ids, -step, 0);
      else if (e.key === 'ArrowRight') nudgeIds(ids, step, 0);
      else if (e.key === 'ArrowUp') nudgeIds(ids, 0, -step);
      else if (e.key === 'ArrowDown') nudgeIds(ids, 0, step);
      else if (e.key.toLowerCase() === 'r') rotateIds(ids, e.shiftKey ? -90 : 90);
      else if (e.key === '[' || e.key === ']') rotateIds(ids, e.key === ']' ? 5 : -5);
      else if (e.key === 'Escape') setSelectedId(null);
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, drawTool, polyPts, drawThick, pickedIds, tools]);

  // ------------------------------------------------------------------ auto layout
  const runAutoLayout = async () => {
    setPacking(true);
    setError(null);
    try {
      const res = await autoLayout(mat, tools, { gap_mm: gapMm, margin_mm: 10, allow_rotate: true, direction: packDir });
      const by = new Map(res.placements.map((p) => [p.id, p]));
      setTools((prev) => prev.map((t) => {
        const p = by.get(t.id);
        return p ? { ...t, rotation_deg: p.rotation_deg, offset_mm: p.offset_mm, notch: null } : t;
      }));
      setUnplaced(res.unplaced);
      setSelectedId(null);
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
  const fetchStl = useCallback(async () => {
    const { blob } = await exportLayout(body, 'stl', exportOpts, true);
    return blob.arrayBuffer();
  }, [body, exportOpts.mat_thickness_mm]); // eslint-disable-line react-hooks/exhaustive-deps
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
    <div className={`${st.layoutGrid} ${ed.layoutWorkspace}`}>
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
          <label className={ui.field} style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }} title="Foam left between neighbouring pockets">
            <input className={`${ui.input} ${ui.inputSm}`} type="number" min={2} max={40} step={1} value={gapMm} onChange={(e) => setGapMm(Number(e.target.value))} style={{ width: 58 }} />
            <span className={ui.unit}>mm gap</span>
          </label>
          <select className={`${ui.select} ${ui.inputSm}`} style={{ width: 190 }} value={packDir} onChange={(e) => setPackDir(e.target.value as 'columns' | 'rows')} title="Upright: tools stand vertical, side by side across the drawer. Lying: tools run along the drawer, stacked top to bottom.">
            <option value="columns">↕ upright, across</option>
            <option value="rows">↔ lying, stacked</option>
          </select>

          <span className={st.toolbarSpacer} />
          {busy && <span className={ui.spinner} />}
          <span className={ui.hint}>{fmt(mat.width_mm, 1)} × {fmt(mat.height_mm, 1)} mm</span>
        </div>
        <div className={ed.guidance}><div><strong>{drawTool ? `Draw a ${drawTool}` : notchMode ? 'Place a finger notch' : 'Arrange your tools'}</strong>{drawTool === 'poly' ? 'Click corners, then Enter to finish. Escape cancels.' : drawTool ? 'Drag across the sheet to size the pocket. Escape returns to Select.' : notchMode ? 'Click an outline where you want to lift the tool out.' : 'Drag a tool to position it, or use Auto layout to get started.'}</div><span>{pickedIds.length > 0 ? `${pickedIds.length} ticked · arrow keys move them together · Delete removes them · Escape clears` : 'Arrow keys: nudge · R: rotate · Shift: larger steps · Ctrl/Cmd-click: tick several'}</span></div>
        {view3d && (
          <Layout3D tools={tools} layout={layout} mat={mat} settings={settings} selectedId={selectedId} onSelect={setSelectedId}
            pickedIds={pickedIds} onPick={togglePick}
            onMove={(id, dx, dy) => updateTool(id, (t) => ({
              ...t, offset_mm: { x: Math.round((t.offset_mm.x + dx) * 10) / 10, y: Math.round((t.offset_mm.y + dy) * 10) / 10 },
              notch: t.notch ? { ...t.notch, x_mm: t.notch.x_mm + dx, y_mm: t.notch.y_mm + dy } : null,
            }))} />
        )}
        <svg ref={svgRef} className={st.sheet} viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`} style={{ aspectRatio: `${vb.w} / ${vb.h}`, cursor: drawTool || notchMode ? 'crosshair' : drag ? 'grabbing' : 'default', display: view3d ? 'none' : undefined }}
          onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} onPointerDown={onSheetDown} onDoubleClick={() => { if (drawTool === 'poly' && polyPts.length >= 3) finishPolygon(polyPts.slice(0, -1).length >= 3 ? polyPts.slice(0, -1) : polyPts); }}>
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

      <aside className={`${ui.panel} ${ed.inspector}`} aria-label="Layout settings">
        <div className={ed.inspectorTabs} style={{gridTemplateColumns:'repeat(3, 1fr)'}} role="tablist" aria-label="Layout settings">
          {(['foam', 'tools', 'export'] as const).map(key => <button type="button" role="tab" key={key} aria-selected={panel === key} onClick={() => setPanel(key)}>{key === 'foam' ? 'Foam & fit' : key === 'tools' ? 'Tools' : 'Export'}</button>)}
        </div>
        <div className={ed.inspectorBody}>
        {panel === 'foam' && <>
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

        <div className={ui.section}>
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
        </div>

        </>}
        {panel === 'tools' && <>
        <div className={ui.section}>
          <div className={ui.rowBetween}>
            <h2 className={ui.panelTitle}>Tools</h2>
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
          <div className={ui.list}>
            {tools.filter((t) => t.polygon_mm.length >= 3).map((t) => (
              <div key={t.id} className={`${ui.toolRow} ${t.id === selectedId ? ui.toolRowActive : ''}`}
                onClick={(e) => (pickMod(e) ? togglePick(t.id) : setSelectedId(t.id))}
                style={{ gridTemplateColumns: '12px auto 1fr auto auto', opacity: t.include ? 1 : 0.5, ...(isPicked(t.id) ? { outline: '2px solid var(--accent, #f26a1b)', outlineOffset: '-2px' } : null) }}>
                <span className={ui.swatch} style={{ background: t.color }} />
                <button type="button" aria-pressed={isPicked(t.id)} aria-label={isPicked(t.id) ? `Untick ${t.name}` : `Tick ${t.name}`}
                  title="Ctrl/Cmd-click to tick several tools and move or remove them together"
                  style={{ background: 'none', border: 0, padding: 0, cursor: 'pointer', font: 'inherit' }}
                  onClick={(e) => { e.stopPropagation();          // the row handles it too; a TOGGLE must not fire twice
                                    togglePick(t.id); }}>{isPicked(t.id) ? '☑ ' : '☐ '}</button>
                <input aria-label={`Rename ${t.name}`} className={ui.toolName} value={t.name}
                  onClick={(e) => { e.stopPropagation(); if (pickMod(e)) { e.preventDefault(); togglePick(t.id); } }}
                  onChange={(e) => updateTool(t.id, (x) => ({ ...x, name: e.target.value }))} />
                <span className={ui.toolMeta}>{t.source === 'object' ? '3D · ' : t.source === 'shape' ? '▭ · ' : ''}{t.include ? fmtMm(resolveDepth(t, settings)) : 'skipped'}{resolveDepth(t, settings) === null && t.include ? 'through' : ''}</span>
                <input type="checkbox" checked={t.include} aria-label={`Include ${t.name} in export`} title="Include in export" onClick={(e) => e.stopPropagation()} onChange={(e) => updateTool(t.id, (x) => ({ ...x, include: e.target.checked }))} />
              </div>
            ))}
            {tools.filter((t) => t.polygon_mm.length >= 3).length > 1 && !pickedIds.length &&
              <p className={ui.hint}>Ctrl/Cmd-click tools (in the list or on the sheet) to tick several, then move or remove them together.</p>}
          </div>
        </div>

        {selected && (
          <div className={ui.section}>
            <div className={ui.rowBetween}>
              <h2 className={ui.panelTitle} style={{ color: selected.color }}>{selected.name}</h2>
              <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnGhost}`} onClick={() => setSelectedId(null)}>Done</button>
            </div>
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
            <div className={ui.grid2}>
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
          </div>
        )}

        </>}
        {panel === 'export' && <div className={ui.section}>
          <h2 className={ui.panelTitle}>Download cutting files</h2>
          <p className={ui.hint}>Choose the format your machine or design software accepts.</p>
          <label className={ui.checkbox}><input type="checkbox" checked={settings.include_mat} onChange={(e) => set({ include_mat: e.target.checked })} /> Mat outline rectangle</label>
          <label className={ui.checkbox}><input type="checkbox" checked={settings.include_labels} onChange={(e) => set({ include_labels: e.target.checked })} /> Tool names + depth legend (separate layer)</label>
          <label className={ui.checkbox}><input type="checkbox" checked={settings.mirror} onChange={(e) => set({ mirror: e.target.checked })} /> Mirror (cutting from the underside)</label>
          <label className={ui.field}><span className={ui.label}>SVG style</span>
            <select className={ui.select} value={settings.fill_mode} onChange={(e) => set({ fill_mode: e.target.value as 'none' | 'fill' })}>
              <option value="none">Hairline outlines, colored by depth (laser / CNC)</option>
              <option value="fill">Filled shapes (engrave / Cricut)</option>
            </select></label>
          <div className={ui.grid2}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={!!exporting} onClick={() => doExport('svg')}>{exporting === 'svg' ? <span className={ui.spinner} /> : null} Download SVG</button>
            <button type="button" className={ui.btn} disabled={!!exporting} onClick={() => doExport('dxf')}>{exporting === 'dxf' ? <span className={ui.spinner} /> : null} Download DXF</button>
            <button type="button" className={ui.btn} disabled={!!exporting} onClick={() => doExport('stl')}>{exporting === 'stl' ? <span className={ui.spinner} /> : null} Download STL</button>
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
