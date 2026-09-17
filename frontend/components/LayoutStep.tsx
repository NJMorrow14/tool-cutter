'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ui from './ui.module.css';
import st from './stage.module.css';
import ThreeViewer from './ThreeViewer';
import { buildLayoutBody, computeLayout, createSession, downloadBlob, exportLayout, resolveDepth, type ExportFormat } from '../lib/api';
import { depthColorMap, depthKey, fmt, fmtMm, ringsToPath } from '../lib/geom';
import type { LayoutResponse, LayoutSettings, SessionInfo, Tool } from '../lib/types';

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
  const [notchMode, setNotchMode] = useState(false);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const [show3d, setShow3d] = useState(false);
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

  // ------------------------------------------------------------------ mm-space pointer math
  const margin = 6;
  const vb = { x: -margin, y: -margin, w: mat.width_mm + 2 * margin, h: mat.height_mm + 2 * margin };
  const toMm = (e: React.PointerEvent): { x: number; y: number } => {
    const r = svgRef.current!.getBoundingClientRect();
    return { x: vb.x + ((e.clientX - r.left) / r.width) * vb.w, y: vb.y + ((e.clientY - r.top) / r.height) * vb.h };
  };

  const onSheetDown = (e: React.PointerEvent) => {
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
    e.stopPropagation();
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
    if (!drag) return;
    const p = toMm(e);
    setDrag({ ...drag, delta: { x: p.x - drag.startMm.x, y: p.y - drag.startMm.y } });
  };
  const onUp = () => {
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
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || !selected) return;
      const step = e.shiftKey ? 5 : 1;
      const nudge = (dx: number, dy: number) => updateTool(selected.id, (t) => ({ ...t, offset_mm: { x: t.offset_mm.x + dx, y: t.offset_mm.y + dy } }));
      if (e.key === 'ArrowLeft') nudge(-step, 0);
      else if (e.key === 'ArrowRight') nudge(step, 0);
      else if (e.key === 'ArrowUp') nudge(0, -step);
      else if (e.key === 'ArrowDown') nudge(0, step);
      else if (e.key.toLowerCase() === 'r') updateTool(selected.id, (t) => ({ ...t, rotation_deg: (t.rotation_deg + (e.shiftKey ? -5 : 5)) % 360 }));
      else if (e.key === 'Escape') setSelectedId(null);
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

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
    <div className={st.layoutGrid}>
      <div className={st.stageWrap}>
        <div className={st.toolbar}>
          <button type="button" className={`${ui.btn} ${ui.btnSm} ${notchMode ? ui.btnActive : ''}`} onClick={() => setNotchMode(!notchMode)}>
            {notchMode ? 'Click an outline to place the notch…' : '☝ Add finger notch'}
          </button>
          <span className={ui.hint}>Drag tools to move · arrows nudge (Shift = 5 mm) · <kbd>R</kbd> rotate 5° (Shift = −5°)</span>
          <span className={st.toolbarSpacer} />
          {busy && <span className={ui.spinner} />}
          <span className={ui.hint}>{fmt(mat.width_mm, 1)} × {fmt(mat.height_mm, 1)} mm</span>
        </div>
        <svg ref={svgRef} className={st.sheet} viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`} style={{ aspectRatio: `${vb.w} / ${vb.h}`, cursor: notchMode ? 'crosshair' : drag ? 'grabbing' : 'default' }}
          onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} onPointerDown={onSheetDown}>
          <rect x={vb.x} y={vb.y} width={vb.w} height={vb.h} fill="#f8fafc" />
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
              <g key={lt.id} transform={d ? `translate(${d.x} ${d.y})` : undefined} style={{ cursor: notchMode ? 'crosshair' : 'grab' }}
                onPointerDown={(e) => onToolDown(lt.id, e)}>
                <path d={ringsToPath(lt.rings)} fill={tool.color} fillOpacity={sel ? 0.35 : 0.18} fillRule="evenodd"
                  stroke={bad ? '#dc2626' : color} strokeWidth={sel ? 0.9 : 0.5} strokeDasharray={bad ? '2 1' : undefined} strokeLinejoin="round" />
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
        </svg>
        <div className={st.legend}>
          <span>Pocket depth:</span>
          {Array.from(colorMap.entries()).filter(([k]) => k !== null).map(([k, c]) => (
            <span key={String(k)}><i className={st.legendSwatch} style={{ background: c }} />{k} mm</span>
          ))}
          {settings.depth_rule === 'through' && <span><i className={st.legendSwatch} style={{ background: '#ff0000' }} />through cut</span>}
          {settings.mirror && <span className={ui.badge}>mirrored (cut from the back)</span>}
        </div>
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

      <aside className={ui.panel}>
        <h2 className={ui.panelTitle}>Mat</h2>
        <div className={ui.section}>
          <div className={ui.grid2}>
            <label className={ui.field}><span className={ui.label}>Width (mm)</span>
              <input className={ui.input} type="number" step="0.1" onFocus={(e) => e.currentTarget.select()} value={mat.width_mm} onChange={(e) => setMatSize({ ...matSize, width_mm: Number(e.target.value) })} /></label>
            <label className={ui.field}><span className={ui.label}>Height (mm)</span>
              <input className={ui.input} type="number" step="0.1" onFocus={(e) => e.currentTarget.select()} value={mat.height_mm} onChange={(e) => setMatSize({ ...matSize, height_mm: Number(e.target.value) })} /></label>
          </div>
          <label className={ui.field}><span className={ui.label}>Foam thickness (mm) — for the 3D model</span>
            <input className={ui.input} type="number" step="1" min="3" value={settings.mat_thickness_mm} onChange={(e) => set({ mat_thickness_mm: Number(e.target.value) })} /></label>
          <p className={ui.hint}>Changing the size here only changes the export frame; tools keep their calibrated scale and position from the top-left corner.</p>
        </div>

        <div className={ui.section}>
          <h2 className={ui.panelTitle}>Fit</h2>
          <div className={ui.grid2}>
            <label className={ui.field}><span className={ui.label}>Clearance (mm)</span>
              <input className={ui.input} type="number" step="0.25" min="0" value={settings.default_clearance_mm} onChange={(e) => set({ default_clearance_mm: Number(e.target.value) })} /></label>
            <label className={ui.field}><span className={ui.label}>Smoothing (mm)</span>
              <input className={ui.input} type="number" step="0.1" min="0" value={settings.smoothing_mm} onChange={(e) => set({ smoothing_mm: Number(e.target.value) })} /></label>
          </div>
          <label className={ui.field}><span className={ui.label}>Finger notch diameter (mm)</span>
            <input className={ui.input} type="number" step="1" min="5" value={settings.notch_diameter_mm} onChange={(e) => set({ notch_diameter_mm: Number(e.target.value) })} /></label>
          <p className={ui.hint}>Clearance grows each outline so the tool drops in; 0.5–1.5 mm is typical for foam. Smoothing rounds off pixel jaggies.</p>
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
            <label className={ui.field}><span className={ui.label}>X — how far the tool sticks up (mm)</span>
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

        <div className={ui.section}>
          <div className={ui.rowBetween}>
            <h2 className={ui.panelTitle}>Tools</h2>
            <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={importing} onClick={() => fileRef.current?.click()} title="Import a PLY/OBJ/GLB/STL of one tool; it is laid flat and placed on the mat">
              {importing ? <span className={ui.spinner} /> : '+'} Add tool from 3D model
            </button>
            <input ref={fileRef} type="file" accept={MESH_ACCEPT} hidden onChange={(e) => { void importObject(e.target.files?.[0]); e.target.value = ''; }} />
          </div>
          <div className={ui.list}>
            {tools.filter((t) => t.polygon_mm.length >= 3).map((t) => (
              <div key={t.id} className={`${ui.toolRow} ${t.id === selectedId ? ui.toolRowActive : ''}`} onClick={() => setSelectedId(t.id)} style={{ gridTemplateColumns: '12px 1fr auto auto', opacity: t.include ? 1 : 0.5 }}>
                <span className={ui.swatch} style={{ background: t.color }} />
                <input className={ui.toolName} value={t.name} onClick={(e) => e.stopPropagation()} onChange={(e) => updateTool(t.id, (x) => ({ ...x, name: e.target.value }))} />
                <span className={ui.toolMeta}>{t.source === 'object' ? '3D · ' : ''}{t.include ? fmtMm(resolveDepth(t, settings)) : 'skipped'}{resolveDepth(t, settings) === null && t.include ? 'through' : ''}</span>
                <input type="checkbox" checked={t.include} title="Include in export" onClick={(e) => e.stopPropagation()} onChange={(e) => updateTool(t.id, (x) => ({ ...x, include: e.target.checked }))} />
              </div>
            ))}
          </div>
        </div>

        {selected && (
          <div className={ui.section}>
            <div className={ui.rowBetween}>
              <h2 className={ui.panelTitle} style={{ color: selected.color }}>{selected.name}</h2>
              <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnGhost}`} onClick={() => setSelectedId(null)}>Done</button>
            </div>
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
                : <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => setNotchMode(true)}>Add notch</button>}
            </div>
            <label className={ui.checkbox}><input type="checkbox" checked={selected.include} onChange={(e) => updateTool(selected.id, (t) => ({ ...t, include: e.target.checked }))} /> Include in export</label>
          </div>
        )}

        <div className={ui.section}>
          <h2 className={ui.panelTitle}>Export</h2>
          <label className={ui.checkbox}><input type="checkbox" checked={settings.include_mat} onChange={(e) => set({ include_mat: e.target.checked })} /> Mat outline rectangle</label>
          <label className={ui.checkbox}><input type="checkbox" checked={settings.include_labels} onChange={(e) => set({ include_labels: e.target.checked })} /> Tool names + depth legend (separate layer)</label>
          <label className={ui.checkbox}><input type="checkbox" checked={settings.mirror} onChange={(e) => set({ mirror: e.target.checked })} /> Mirror (cutting from the underside)</label>
          <label className={ui.field}><span className={ui.label}>SVG style</span>
            <select className={ui.select} value={settings.fill_mode} onChange={(e) => set({ fill_mode: e.target.value as 'none' | 'fill' })}>
              <option value="none">Hairline outlines, colored by depth (laser / CNC)</option>
              <option value="fill">Filled shapes (engrave / Cricut)</option>
            </select></label>
          <div className={ui.grid2}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={!!exporting} onClick={() => doExport('svg')}>{exporting === 'svg' ? <span className={ui.spinner} /> : null} SVG</button>
            <button type="button" className={ui.btn} disabled={!!exporting} onClick={() => doExport('dxf')}>{exporting === 'dxf' ? <span className={ui.spinner} /> : null} DXF</button>
            <button type="button" className={ui.btn} disabled={!!exporting} onClick={() => doExport('stl')}>{exporting === 'stl' ? <span className={ui.spinner} /> : null} STL (3D)</button>
            <button type="button" className={ui.btn} onClick={() => setShow3d(true)}>Preview 3D</button>
          </div>
          <p className={ui.hint}>SVG is exactly {fmt(mat.width_mm, 1)} × {fmt(mat.height_mm, 1)} mm with one path per tool; stroke color = pocket depth. DXF has one layer per depth. STL is the foam block with pockets for CNC / CAM. Preview 3D shows the block with or without the tools sitting in it.</p>
        </div>
      </aside>
      {show3d && <ThreeViewer title={`Foam block ${fmt(mat.width_mm, 0)} × ${fmt(mat.height_mm, 0)} × ${settings.mat_thickness_mm} mm`} fetchStl={fetchStl} fetchToolsStl={fetchToolsStl} onClose={() => setShow3d(false)} />}
    </div>
  );
}
