'use client';

import { useState } from 'react';
import ui from './ui.module.css';
import { shapePolygon } from '../lib/geom';
import type { ShapeKind, ShapeSpec } from '../lib/types';

const KINDS: { kind: Exclude<ShapeKind, 'poly'>; label: string; hint: string }[] = [
  { kind: 'rect', label: 'Rectangle', hint: 'boxes, cases, bit holders' },
  { kind: 'slot', label: 'Slot', hint: 'pill for a pen, punch or drill bit' },
  { kind: 'circle', label: 'Circle', hint: 'cans, sockets on end, tape' },
  { kind: 'hex', label: 'Hexagon', hint: 'nuts, hex sockets (flat-to-flat)' },
];

/** Small form to add a drawn primitive to the mat. Sizes are the pocket's *tool* size — clearance is added later. */
export default function AddShape({ onAdd, onClose }: { onAdd: (spec: ShapeSpec, thickness_mm: number) => void; onClose: () => void }) {
  const [spec, setSpec] = useState<ShapeSpec>({ kind: 'rect', w_mm: 80, h_mm: 50, r_mm: 4 });
  const [thick, setThick] = useState(20);
  const poly = shapePolygon(spec);
  const b = poly.reduce((a, p) => [Math.min(a[0], p[0]), Math.min(a[1], p[1]), Math.max(a[2], p[0]), Math.max(a[3], p[1])], [Infinity, Infinity, -Infinity, -Infinity]);
  const sc = 120 / Math.max(b[2] - b[0], b[3] - b[1], 1);
  const set = (patch: Partial<ShapeSpec>) => setSpec({ ...spec, ...patch });
  const round = spec.kind === 'circle' || spec.kind === 'hex';
  return (
    <div className={ui.section} style={{ border: '1px solid var(--border-strong)', borderRadius: 'var(--radius)', padding: 12, background: 'var(--panel-2)' }}>
      <div className={ui.rowBetween}><h2 className={ui.panelTitle}>Add a shape</h2><button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnGhost}`} onClick={onClose}>×</button></div>
      <div className={ui.segmented} role="tablist" style={{ display: 'flex' }}>
        {KINDS.map((k) => <button key={k.kind} type="button" className={spec.kind === k.kind ? ui.segActive : ''} title={k.hint} onClick={() => set({ kind: k.kind })}>{k.label}</button>)}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '130px 1fr', gap: 12, alignItems: 'start' }}>
        <svg viewBox={`-6 -6 132 132`} width={130} height={130} style={{ background: '#fff', border: '1px solid var(--border)', borderRadius: 8 }}>
          <path d={`M${poly.map((p) => `${((p[0] - b[0]) * sc).toFixed(1)} ${((p[1] - b[1]) * sc).toFixed(1)}`).join(' L')} Z`} fill="#f26a1b" fillOpacity={0.25} stroke="#f26a1b" strokeWidth={1.5} />
        </svg>
        <div style={{ display: 'grid', gap: 8 }}>
          <div className={ui.grid2}>
            <label className={ui.field}><span className={ui.label}>{round ? (spec.kind === 'circle' ? 'Diameter (mm)' : 'Across flats (mm)') : 'Width (mm)'}</span>
              <input className={ui.input} type="number" min={2} step={0.5} value={spec.w_mm} onChange={(e) => set({ w_mm: Number(e.target.value) })} /></label>
            {!round && <label className={ui.field}><span className={ui.label}>Height (mm)</span>
              <input className={ui.input} type="number" min={2} step={0.5} value={spec.h_mm} onChange={(e) => set({ h_mm: Number(e.target.value) })} /></label>}
          </div>
          <div className={ui.grid2}>
            {spec.kind === 'rect' && <label className={ui.field}><span className={ui.label}>Corner radius (mm)</span>
              <input className={ui.input} type="number" min={0} step={0.5} value={spec.r_mm} onChange={(e) => set({ r_mm: Number(e.target.value) })} /></label>}
            <label className={ui.field}><span className={ui.label}>Item thickness (mm)</span>
              <input className={ui.input} type="number" min={1} step={0.5} value={thick} onChange={(e) => setThick(Number(e.target.value))} /></label>
          </div>
        </div>
      </div>
      <div className={ui.row}>
        <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnPrimary}`} onClick={() => onAdd(spec, thick)}>Add to mat</button>
        <span className={ui.hint}>Enter the item’s own size; clearance is added on top like every other pocket. Size stays editable after adding.</span>
      </div>
    </div>
  );
}
