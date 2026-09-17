'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ImageStage, { StagePoint } from './ImageStage';
import ui from './ui.module.css';
import st from './stage.module.css';
import { autoDetect, imageUrl, segment } from '../lib/api';
import { fmtMm, polyToPath, toolColor, toolFromResult, uid } from '../lib/geom';
import type { PromptPoint, SessionInfo, Tool, ToolResult } from '../lib/types';

type Mode = 'new' | 'add' | 'sub' | 'box';

interface Props {
  session: SessionInfo;
  tools: Tool[];
  setTools: React.Dispatch<React.SetStateAction<Tool[]>>;
  modelAvailable: boolean;
  onContinue: () => void;
}

function newTool(session: SessionInfo, index: number, points: PromptPoint[], box: number[] | null): Tool {
  return {
    id: uid(), session_id: session.id, source: session.source_kind, name: `Tool ${index + 1}`, color: toolColor(index),
    points, box, polygon_px: [], polygon_mm: [], area_mm2: 0,
    measured_thickness_mm: null, include: true, clearance_mm: null, depth_mm: null, rotation_deg: 0,
    offset_mm: { x: 0, y: 0 }, notch: null, pending: true, error: null,
  };
}

function applyResult(t: Tool, r: ToolResult): Tool {
  return { ...t, polygon_px: r.polygon_px, polygon_mm: r.polygon_mm, area_mm2: r.area_mm2, measured_thickness_mm: r.measured_thickness_mm, pending: false, error: r.polygon_px.length ? null : 'No region found — add a point inside the tool' };
}

export default function DetectStep({ session, tools, setTools, modelAvailable, onContinue }: Props) {
  const rect = session.rectified!;
  const W = rect.width;
  const H = rect.height;
  const [mode, setMode] = useState<Mode>('new');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showHeight, setShowHeight] = useState(false);
  const [autoBusy, setAutoBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoOpts, setAutoOpts] = useState({ mode: 'auto' as 'auto' | 'color' | 'height', min_area_mm2: 200, height_threshold_mm: 2, refine_with_sam: !rect.has_height });
  const [boxDrag, setBoxDrag] = useState<{ start: StagePoint; cur: StagePoint } | null>(null);
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const reqIds = useRef<Map<string, number>>(new Map());

  const src = useMemo(() => imageUrl(session.id, 'rectified', session.version), [session.id, session.version]);
  const heightSrc = useMemo(() => (rect.has_height ? imageUrl(session.id, 'height', session.version) : null), [session.id, session.version, rect.has_height]);

  // only tools that were outlined on this image are shown/edited here; imported tool models live in the layout
  const mine = useMemo(() => tools.filter((t) => t.session_id === session.id), [tools, session.id]);
  const selected = mine.find((t) => t.id === selectedId) ?? null;

  // -------------------------------------------------------------- segmentation scheduling
  const runSegment = useCallback(
    async (tool: Tool) => {
      const id = (reqIds.current.get(tool.id) ?? 0) + 1;
      reqIds.current.set(tool.id, id);
      try {
        const res = await segment(session.id, [{ id: tool.id, points: tool.points, box: tool.box }]);
        if (reqIds.current.get(tool.id) !== id) return;
        const r = res.tools[0];
        setTools((prev) => prev.map((t) => (t.id === tool.id ? applyResult(t, r) : t)));
      } catch (err) {
        if (reqIds.current.get(tool.id) !== id) return;
        const msg = err instanceof Error ? err.message : 'Segmentation failed';
        setTools((prev) => prev.map((t) => (t.id === tool.id ? { ...t, pending: false, error: msg } : t)));
      }
    },
    [session.id, setTools],
  );

  const schedule = useCallback(
    (tool: Tool) => {
      const prev = timers.current.get(tool.id);
      if (prev) clearTimeout(prev);
      timers.current.set(tool.id, setTimeout(() => void runSegment(tool), 180));
    },
    [runSegment],
  );

  useEffect(() => () => timers.current.forEach((t) => clearTimeout(t)), []);

  const updateTool = useCallback(
    (id: string, fn: (t: Tool) => Tool, resegment = true) => {
      setTools((prev) =>
        prev.map((t) => {
          if (t.id !== id) return t;
          const next = fn(t);
          if (resegment) {
            next.pending = true;
            schedule(next);
          }
          return next;
        }),
      );
    },
    [schedule, setTools],
  );

  const addTool = useCallback(
    (points: PromptPoint[], box: number[] | null) => {
      const t = newTool(session, tools.length, points, box);
      setTools((prev) => [...prev, t]);
      setSelectedId(t.id);
      schedule(t);
    },
    [schedule, setTools, tools.length, session],
  );

  const removeTool = (id: string) => {
    setTools((prev) => prev.filter((t) => t.id !== id));
    if (selectedId === id) setSelectedId(null);
  };

  // -------------------------------------------------------------- pointer handling
  const onDown = (p: StagePoint, e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    if (!modelAvailable && mode !== 'box') {
      setError('Click-to-segment needs the HQ-SAM checkpoint on the server. Use Auto-detect instead.');
      return;
    }
    const pt = { x: Math.round(p.x), y: Math.round(p.y) };
    const effMode: Mode = e.shiftKey ? 'add' : e.altKey ? 'sub' : mode;
    if (effMode === 'box') {
      e.currentTarget.setPointerCapture(e.pointerId);
      setBoxDrag({ start: p, cur: p });
      return;
    }
    if (effMode === 'new' || !selected) {
      // clicking inside an existing outline selects it instead of creating a duplicate
      const hit = mine.find((t) => t.polygon_px.length && pointInPolygon(pt, t.polygon_px));
      if (hit && effMode === 'new') {
        setSelectedId(hit.id);
        return;
      }
      addTool([{ ...pt, label: 'pos' }], null);
      return;
    }
    updateTool(selected.id, (t) => ({ ...t, points: [...t.points, { ...pt, label: effMode === 'sub' ? 'neg' : 'pos' }] }));
  };
  const onMove = (p: StagePoint) => {
    if (boxDrag) setBoxDrag({ ...boxDrag, cur: p });
  };
  const onUp = () => {
    if (!boxDrag) return;
    const { start, cur } = boxDrag;
    setBoxDrag(null);
    const box = [Math.min(start.x, cur.x), Math.min(start.y, cur.y), Math.max(start.x, cur.x), Math.max(start.y, cur.y)].map(Math.round);
    if (box[2] - box[0] < 6 || box[3] - box[1] < 6) return;
    if (selected) updateTool(selected.id, (t) => ({ ...t, box }));
    else addTool([], box);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      if (!selected) return;
      if (e.key === 'Delete' || e.key === 'Backspace') removeTool(selected.id);
      else if (e.key === 'Escape') setSelectedId(null);
      else if (e.key.toLowerCase() === 'z' && selected.points.length > 1) updateTool(selected.id, (t) => ({ ...t, points: t.points.slice(0, -1) }));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, updateTool]);

  // -------------------------------------------------------------- auto detect
  const runAuto = async (replace: boolean) => {
    setAutoBusy(true);
    setError(null);
    try {
      const res = await autoDetect(session.id, { ...autoOpts, id_prefix: `${uid('a')}_` });
      const others = tools.filter((t) => t.session_id !== session.id);
      const base = replace ? others.length : tools.length;
      const found: Tool[] = res.tools
        .filter((r) => r.polygon_px.length >= 3)
        .map((r, i) => toolFromResult(r, base + i, session.source_kind));
      setTools((prev) => (replace ? [...prev.filter((t) => t.session_id !== session.id), ...found] : [...prev, ...found]));
      setSelectedId(null);
      if (!found.length) setError('Nothing detected. Try a lower minimum area or click on each tool.');
      else if (res.sam_error) setError(`Detected by ${res.mode}; SAM refinement unavailable (${res.sam_error}).`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Auto-detect failed');
    } finally {
      setAutoBusy(false);
    }
  };

  const ready = tools.filter((t) => t.polygon_mm.length >= 3).length;
  const imported = tools.length - mine.length;

  return (
    <div className={st.layoutGrid}>
      <div className={st.stageWrap}>
        <div className={st.toolbar}>
          <div className={ui.segmented} role="tablist">
            <button type="button" className={mode === 'new' ? ui.segActive : ''} onClick={() => setMode('new')} title="Click a tool to outline it">+ New tool</button>
            <button type="button" className={mode === 'add' ? ui.segActive : ''} onClick={() => setMode('add')} title="Add an include point to the selected tool (Shift+click)">+ Include</button>
            <button type="button" className={mode === 'sub' ? ui.segActive : ''} onClick={() => setMode('sub')} title="Add an exclude point to the selected tool (Alt+click)">− Exclude</button>
            <button type="button" className={mode === 'box' ? ui.segActive : ''} onClick={() => setMode('box')} title="Drag a box around a tool">▭ Box</button>
          </div>
          {heightSrc && (
            <label className={ui.checkbox}><input type="checkbox" checked={showHeight} onChange={(e) => setShowHeight(e.target.checked)} /> Height map</label>
          )}
          <span className={st.toolbarSpacer} />
          <span className={ui.hint}><kbd>Shift</kbd>+click include · <kbd>Alt</kbd>+click exclude · <kbd>Z</kbd> undo point · <kbd>Del</kbd> remove</span>
        </div>
        <ImageStage src={src} width={W} height={H} overlaySrc={showHeight ? heightSrc : null} overlayOpacity={0.75}
          cursor={mode === 'box' ? 'crosshair' : 'crosshair'} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}>
          {(upp) => (
            <>
              {mine.map((t) => {
                const sel = t.id === selectedId;
                return (
                  <g key={t.id} opacity={t.include ? 1 : 0.35}>
                    {t.polygon_px.length >= 3 && (
                      <path d={polyToPath(t.polygon_px)} fill={t.color} fillOpacity={sel ? 0.35 : 0.22} stroke={t.color}
                        strokeWidth={(sel ? 3 : 1.8) * upp} strokeDasharray={t.pending ? `${6 * upp} ${4 * upp}` : undefined} />
                    )}
                    {t.box && (
                      <rect x={t.box[0]} y={t.box[1]} width={t.box[2] - t.box[0]} height={t.box[3] - t.box[1]} fill="none" stroke={t.color} strokeWidth={1.5 * upp} strokeDasharray={`${5 * upp} ${4 * upp}`} />
                    )}
                    {t.points.map((p, i) => (
                      <circle key={i} cx={p.x} cy={p.y} r={(sel ? 6 : 4.5) * upp} fill={p.label === 'pos' ? '#22c55e' : '#ef4444'} stroke="#fff" strokeWidth={1.5 * upp} />
                    ))}
                    {t.polygon_px.length >= 3 && (
                      <text x={t.polygon_px[0][0]} y={t.polygon_px[0][1] - 6 * upp} fontSize={12 * upp} fontWeight={700} fill="#fff" stroke="#0f172a" strokeWidth={3 * upp} paintOrder="stroke">{t.name}</text>
                    )}
                  </g>
                );
              })}
              {boxDrag && (
                <rect x={Math.min(boxDrag.start.x, boxDrag.cur.x)} y={Math.min(boxDrag.start.y, boxDrag.cur.y)}
                  width={Math.abs(boxDrag.cur.x - boxDrag.start.x)} height={Math.abs(boxDrag.cur.y - boxDrag.start.y)}
                  fill="rgba(37,99,235,0.15)" stroke="#2563eb" strokeWidth={1.5 * upp} strokeDasharray={`${5 * upp} ${4 * upp}`} />
              )}
            </>
          )}
        </ImageStage>
        {error && <div className={ui.error}>{error}</div>}
      </div>

      <aside className={ui.panel}>
        <h2 className={ui.panelTitle}>Auto-detect</h2>
        <div className={ui.section}>
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
          <label className={ui.checkbox}>
            <input type="checkbox" checked={autoOpts.refine_with_sam} disabled={!modelAvailable} onChange={(e) => setAutoOpts({ ...autoOpts, refine_with_sam: e.target.checked })} />
            Refine edges with HQ-SAM
          </label>
          <div className={ui.row}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={autoBusy} onClick={() => runAuto(true)}>
              {autoBusy ? <span className={ui.spinner} /> : null} Detect tools
            </button>
            <button type="button" className={ui.btn} disabled={autoBusy || !mine.length} onClick={() => runAuto(false)}>Add to list</button>
          </div>
          <p className={ui.hint}>Then fix anything odd by clicking: a click inside a tool outlines it, Shift/Alt clicks grow or shrink the selected outline.</p>
        </div>

        <div className={ui.section}>
          <div className={ui.rowBetween}>
            <h2 className={ui.panelTitle}>Tools ({mine.length})</h2>
            {mine.length > 0 && <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnDanger}`} onClick={() => { setTools((prev) => prev.filter((t) => t.session_id !== session.id)); setSelectedId(null); }}>Clear all</button>}
          </div>
          <div className={ui.list}>
            {mine.map((t) => (
              <div key={t.id} className={`${ui.toolRow} ${t.id === selectedId ? ui.toolRowActive : ''}`} onClick={() => setSelectedId(t.id)}>
                <span className={ui.swatch} style={{ background: t.color }} />
                <input className={ui.toolName} value={t.name} onClick={(e) => e.stopPropagation()} onChange={(e) => updateTool(t.id, (x) => ({ ...x, name: e.target.value }), false)} />
                <span className={ui.toolMeta}>
                  {t.pending ? <span className={ui.spinner} /> : t.error ? '⚠' : `${(t.area_mm2 / 100).toFixed(1)} cm²`}
                  {t.measured_thickness_mm !== null && !t.pending ? ` · ${fmtMm(t.measured_thickness_mm)}` : ''}
                </span>
                <button type="button" className={ui.iconBtn} title="Remove" onClick={(e) => { e.stopPropagation(); removeTool(t.id); }}>×</button>
              </div>
            ))}
            {!mine.length && <p className={ui.hint}>No tools yet. Run auto-detect or click each tool in the image.</p>}
            {imported > 0 && <p className={ui.hint}>{imported} imported tool model{imported > 1 ? 's' : ''} will join these in the layout.</p>}
          </div>
          {selected && (
            <div className={ui.row}>
              <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={selected.points.length <= 1 && !selected.box} onClick={() => updateTool(selected.id, (t) => ({ ...t, points: t.points.slice(0, -1) }))}>Undo point</button>
              <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={!selected.box} onClick={() => updateTool(selected.id, (t) => ({ ...t, box: null }))}>Clear box</button>
              {selected.error && <span className={ui.hint} style={{ color: 'var(--danger)' }}>{selected.error}</span>}
            </div>
          )}
        </div>

        <div className={ui.section}>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${ui.btnBlock}`} disabled={!ready} onClick={onContinue}>
            Continue → Layout &amp; depth ({ready})
          </button>
        </div>
      </aside>
    </div>
  );
}

function pointInPolygon(p: { x: number; y: number }, poly: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > p.y !== yj > p.y && p.x < ((xj - xi) * (p.y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
