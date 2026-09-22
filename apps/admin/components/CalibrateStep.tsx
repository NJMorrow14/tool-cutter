'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ImageStage, { StagePoint } from './ImageStage';
import ui from './ui.module.css';
import st from './stage.module.css';
import { calibrate, imageUrl } from '../lib/api';
import { edgeLengths, orderCorners } from '../lib/geom';
import { MAT_PRESETS, type SessionInfo } from '../lib/types';

interface Props {
  session: SessionInfo;
  corners: number[][] | null;
  setCorners: (c: number[][] | null) => void;
  matSize: { width_mm: number; height_mm: number };
  setMatSize: (m: { width_mm: number; height_mm: number }) => void;
  onCalibrated: (info: SessionInfo) => void;
  onContinue: () => void;
  turns: number;
  setTurns: (t: number) => void;
}

const LABELS = ['TL', 'TR', 'BR', 'BL'];

export default function CalibrateStep({ session, corners, setCorners, matSize, setMatSize, onCalibrated, onContinue, turns, setTurns }: Props) {
  const pts = useMemo(() => corners ?? [], [corners]);
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [hover, setHover] = useState<StagePoint | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const uppRef = useRef(1);

  const W = session.original.width;
  const H = session.original.height;
  const src = useMemo(() => imageUrl(session.id, 'original', session.version), [session.id, session.version]);

  useEffect(() => {
    const img = new Image();
    img.onload = () => setNatural({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = src;
  }, [src]);

  const ordered = useMemo(() => {
    if (pts.length !== 4) return null;
    const o = orderCorners(pts);
    // rotate clockwise by `turns` quarter turns: the corner that becomes top-left moves forward
    const k = ((turns % 4) + 4) % 4;
    return k ? [...o.slice(4 - k), ...o.slice(0, 4 - k)] : o;
  }, [pts, turns]);
  const rotate = () => {
    setTurns((turns + 1) % 4);
    setMatSize({ width_mm: matSize.height_mm, height_mm: matSize.width_mm });
  };
  const measured = useMemo(() => {
    if (!ordered || !session.scan) return null;
    const { horiz, vert } = edgeLengths(ordered);
    return { width_mm: Math.round(horiz * session.scan.mm_per_px * 2) / 2, height_mm: Math.round(vert * session.scan.mm_per_px * 2) / 2 };
  }, [ordered, session]);

  const nearest = useCallback(
    (p: StagePoint, upp: number): number | null => {
      let best = -1;
      let bd = Infinity;
      pts.forEach((c, i) => {
        const d = Math.hypot(c[0] - p.x, c[1] - p.y);
        if (d < bd) {
          bd = d;
          best = i;
        }
      });
      return best >= 0 && bd <= 22 * upp ? best : null;
    },
    [pts],
  );

  const onDown = (p: StagePoint, e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const hit = nearest(p, uppRef.current);
    if (hit !== null) {
      setDragIdx(hit);
      return;
    }
    if (pts.length < 4) {
      const next = [...pts, [p.x, p.y]];
      setCorners(next);
      setDragIdx(next.length - 1);
    }
  };
  const onMove = (p: StagePoint) => {
    setHover(p);
    if (dragIdx === null) return;
    const next = pts.map((c, i) => (i === dragIdx ? [Math.min(W, Math.max(0, p.x)), Math.min(H, Math.max(0, p.y))] : c));
    setCorners(next);
  };
  const onUp = () => setDragIdx(null);

  const doCalibrate = async () => {
    if (!ordered) return;
    setBusy(true);
    setError(null);
    try {
      const info = await calibrate(session.id, ordered, matSize.width_mm, matSize.height_mm, 0);
      onCalibrated(info);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Calibration failed');
    } finally {
      setBusy(false);
    }
  };

  const canCalibrate = !!ordered && matSize.width_mm > 0 && matSize.height_mm > 0 && !busy;
  const loupePoint = dragIdx !== null ? pts[dragIdx] : null;

  const fromMarkers = session.source_kind === 'capture' && !!session.rectified && !!session.auto_calibrated;
  return (
    <div className={st.layoutGrid}>
      <div className={st.stageWrap}>
        {fromMarkers && (
          <div className={ui.warn} style={{ background: 'var(--accent-soft)', borderColor: 'var(--accent)', color: 'var(--text)' }}>
            This drawer was calibrated from its corner markers during the arc capture: the image below is already the metric,
            top-down drawer ({session.mat_mm?.width} × {session.mat_mm?.height} mm), so the handles sit on its corners. Only re-calibrate
            here if you want to crop or re-size it by hand.
            {session.scan?.markers_reordered && session.scan.marker_corners && (
              <div style={{ marginTop: 6 }}>
                The marker sheet was not laid out in the printed order, so the corners were taken from where the markers
                actually sat: {Object.entries(session.scan.marker_corners).map(([id, corner]) => `${id} → ${corner}`).join(', ')}.
              </div>
            )}
          </div>
        )}
        <div className={st.toolbar}>
          <span className={ui.hint}>
            {pts.length < 4
              ? `Click the ${LABELS.length - pts.length === 4 ? 'first' : 'next'} corner of the mat / drawer (${pts.length}/4). Drag a handle to fine-tune.`
              : 'Drag the handles onto the exact corners. Order is detected automatically.'}
          </span>
          <span className={st.toolbarSpacer} />
          <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={rotate} disabled={pts.length !== 4} title="Choose which corner is top-left (swaps width and height)">↻ Rotate 90°</button>
          <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => setCorners(null)} disabled={pts.length === 0}>Reset corners</button>
          {session.suggested_corners && (
            <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => setCorners(session.suggested_corners)}>Use detected mat</button>
          )}
        </div>
        <div style={{ position: 'relative' }}>
          <ImageStage src={src} width={W} height={H} cursor={dragIdx !== null ? 'grabbing' : pts.length < 4 ? 'crosshair' : 'default'}
            onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}>
            {(upp) => {
              uppRef.current = upp;
              const r = 7 * upp;
              return (
                <>
                  {ordered && (
                    <polygon points={ordered.map((c) => `${c[0]},${c[1]}`).join(' ')} fill="rgba(37,99,235,0.18)" stroke="#60a5fa" strokeWidth={2 * upp} />
                  )}
                  {pts.length > 1 && pts.length < 4 && (
                    <polyline points={pts.map((c) => `${c[0]},${c[1]}`).join(' ')} fill="none" stroke="#60a5fa" strokeWidth={2 * upp} strokeDasharray={`${6 * upp} ${4 * upp}`} />
                  )}
                  {pts.map((c, i) => {
                    const label = ordered ? LABELS[ordered.findIndex((o) => o[0] === c[0] && o[1] === c[1])] : String(i + 1);
                    return (
                      <g key={i}>
                        <circle cx={c[0]} cy={c[1]} r={r * 2.2} fill="rgba(255,255,255,0.001)" />
                        <circle cx={c[0]} cy={c[1]} r={r} fill={dragIdx === i ? '#f59e0b' : '#2563eb'} stroke="#fff" strokeWidth={2 * upp} />
                        <line x1={c[0] - r * 2} x2={c[0] + r * 2} y1={c[1]} y2={c[1]} stroke="#fff" strokeWidth={1 * upp} />
                        <line y1={c[1] - r * 2} y2={c[1] + r * 2} x1={c[0]} x2={c[0]} stroke="#fff" strokeWidth={1 * upp} />
                        <text x={c[0] + r * 1.8} y={c[1] - r * 1.8} fontSize={13 * upp} fill="#fff" stroke="#0f172a" strokeWidth={3 * upp} paintOrder="stroke" fontWeight={700}>{label}</text>
                      </g>
                    );
                  })}
                </>
              );
            }}
          </ImageStage>
          {loupePoint && natural && (
            <Loupe src={src} natural={natural} full={{ w: W, h: H }} point={loupePoint} />
          )}
        </div>
        {(
          <p className={ui.hint}>
            Scan view is a top-down render of the detected mat plane (plane fit inliers: {Math.round((session.scan?.plane_inlier_fraction ?? 0) * 100)}%).
            The rectangle you mark defines the drawer / mat; anything outside is dropped.
          </p>
        )}
      </div>

      <aside className={ui.panel}>
        <h2 className={ui.panelTitle}>Mat / drawer size</h2>
        <div className={ui.section}>
          <div className={ui.field}>
            <span className={ui.label}>Preset</span>
            <select className={ui.select} value="" onChange={(e) => {
              const p = MAT_PRESETS.find((x) => x.label === e.target.value);
              if (p) setMatSize({ width_mm: p.width_mm, height_mm: p.height_mm });
            }}>
              <option value="">Choose a preset…</option>
              {MAT_PRESETS.map((p) => <option key={p.label} value={p.label}>{p.label}</option>)}
            </select>
          </div>
          <div className={ui.grid2}>
            <label className={ui.field}>
              <span className={ui.label}>Width (mm)</span>
              <input className={ui.input} type="number" step="0.1" min="1" onFocus={(e) => e.currentTarget.select()} value={matSize.width_mm} onChange={(e) => setMatSize({ ...matSize, width_mm: Number(e.target.value) })} />
            </label>
            <label className={ui.field}>
              <span className={ui.label}>Height (mm)</span>
              <input className={ui.input} type="number" step="0.1" min="1" onFocus={(e) => e.currentTarget.select()} value={matSize.height_mm} onChange={(e) => setMatSize({ ...matSize, height_mm: Number(e.target.value) })} />
            </label>
          </div>
          {measured && (
            <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => setMatSize(measured)}>
              Use scan-measured {measured.width_mm} × {measured.height_mm} mm
            </button>
          )}
          <p className={ui.hint}>
            Width is the top edge (TL→TR), height the left edge (TL→BL). Use ↻ Rotate 90° if the labels sit on the wrong corners — the SVG comes out in that orientation and exactly this size.
            {' '}The scan is metric already; type a tape-measure value if the scan’s scale drifted.
          </p>
        </div>
        <div className={ui.section}>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${ui.btnBlock}`} disabled={!canCalibrate} onClick={doCalibrate}>
            {busy ? <span className={ui.spinner} /> : null} {session.rectified ? 'Re-calibrate' : 'Calibrate'}
          </button>
          {error && <div className={ui.error}>{error}</div>}
        </div>
        {session.rectified && (
          <div className={ui.section}>
            <span className={ui.label}>Top-down result</span>
            <img src={imageUrl(session.id, 'rectified', session.version)} alt="Rectified mat" style={{ width: '100%', borderRadius: 8, border: '1px solid var(--border)' }} />
            <dl className={ui.kv}>
              <dt>Resolution</dt><dd>{session.rectified.mm_per_px.toFixed(3)} mm / px</dd>
              <dt>Size</dt><dd>{session.mat_mm?.width} × {session.mat_mm?.height} mm</dd>
            </dl>
            <p className={ui.hint}>Edges should look straight and the mat should fill the frame. If not, nudge the corners and re-calibrate.</p>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${ui.btnBlock}`} onClick={onContinue}>Continue → Detect tools</button>
          </div>
        )}
      </aside>
    </div>
  );
}

function Loupe({ src, natural, full, point }: { src: string; natural: { w: number; h: number }; full: { w: number; h: number }; point: number[] }) {
  const zoom = 3;
  const size = 170;
  const nx = (point[0] / full.w) * natural.w;
  const ny = (point[1] / full.h) * natural.h;
  const bgW = natural.w * zoom;
  const bgH = natural.h * zoom;
  const posX = size / 2 - nx * zoom;
  const posY = size / 2 - ny * zoom;
  return (
    <div style={{
      position: 'absolute', top: 10, right: 10, width: size, height: size, borderRadius: 10, border: '2px solid #fff',
      boxShadow: '0 4px 16px rgba(0,0,0,0.4)', backgroundImage: `url(${src})`, backgroundRepeat: 'no-repeat',
      backgroundSize: `${bgW}px ${bgH}px`, backgroundPosition: `${posX}px ${posY}px`, pointerEvents: 'none', overflow: 'hidden',
    }}>
      <div style={{ position: 'absolute', left: size / 2, top: 0, bottom: 0, width: 1, background: 'rgba(245,158,11,0.9)' }} />
      <div style={{ position: 'absolute', top: size / 2, left: 0, right: 0, height: 1, background: 'rgba(245,158,11,0.9)' }} />
    </div>
  );
}
