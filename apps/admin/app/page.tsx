'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './page.module.css';
import { type StepId } from '../components/Stepper';
import UploadStep from '../components/UploadStep';
import CalibrateStep from '../components/CalibrateStep';
import LayoutStep from '../components/LayoutStep';
import { autoDetect, getHealth, getSession } from '../lib/api';
import { edgeLengths, orderCorners, placementOffset, toolFromResult } from '../lib/geom';
import { DEFAULT_SETTINGS, type ModelInfo, type SessionInfo } from '../lib/types';

import { useWorkspaceHistory } from '../lib/use-workspace-history';

export default function Page() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [layoutPanel, setLayoutPanel] = useState<'foam' | 'tools' | 'export'>('foam');
  const [loadError, setLoadError] = useState<string | null>(null);
  const [finding, setFinding] = useState<string | null>(null);
  const [step, setStep] = useState<StepId>('upload');
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [corners, setCorners] = useState<number[][] | null>(null);
  const [turns, setTurns] = useState(0);
  const editing = step === 'outlines' || step === 'layout';
  const { document: { tools, settings, matSize }, setTools, setSettings, setMatSize, history, canUndo, canRedo, epoch } = useWorkspaceHistory({
    tools: [], settings: DEFAULT_SETTINGS, matSize: { width_mm: 279.4, height_mm: 215.9 },
  }, editing);

  useEffect(() => {
    getHealth()
      .then((h) => { setBackendOk(true); setModel(h.model ?? null); })
      .catch(() => setBackendOk(false));
  }, []);

  // ?session=<id> opens a session created elsewhere (the phone app posts a capture, then loads this URL)
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get('session');
    if (!id) return;
    getSession(id)
      .then((info) => handleUploaded(info))
      .catch(() => setLoadError('This capture could not be opened. Choose it from Recent captures or import it again.'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** The new flow (Nolan, 2026-10-03: "the outlining and agent steps are now unnecessary"): the tools are whatever
   *  stands up from the drawer floor in the height map. Find them, drop them on the insert, and the foam is the
   *  negative of their scanned shapes. No outline editing step. */
  const findingFor = useRef<string | null>(null);
  const findTools = useCallback(async (info: SessionInfo) => {
    if (findingFor.current === info.id) return;      // React StrictMode runs effects twice in dev
    findingFor.current = info.id;
    setFinding('Finding tools in the height map…');
    setLayoutPanel('foam');
    setStep('layout');
    try {
      // hybrid = whatever stands up in the height map, PLUS photo silhouettes the depth sensor missed (black and glossy
      // objects return no depth) when the photo-assist model is available
      // always ask for hybrid on a scan: the server falls back to height-only itself when the photo model is not
      // loaded, and reading `model` here raced the health check (it was still null when a capture opened)
      const res = await autoDetect(info.id, { mode: info.rectified?.has_height ? 'hybrid' : 'auto', min_area_mm2: 200, height_threshold_mm: 2.0,
                                               refine_with_sam: !info.rectified?.has_height, edge_source: info.rectified?.has_height ? 'topo' : 'photo', id_prefix: `t${Date.now() % 100000}_` });
      const found = (res.tools ?? []).filter((r) => r.polygon_mm.length >= 3);
      // write through a FRESH writer: the memoised setTools closure is invalidated by the history.reset() that ran
      // when the capture was opened (old async closures must not write into a reset document)
      history.writer('tools')((prev) => {
        const kept = prev.filter((t) => t.session_id !== info.id);
        return [...kept, ...found.map((r, i) => ({ ...toolFromResult(r, kept.length + i, 'scan'), pocket_style: 'relief' as const }))];
      });
      if (!found.length) setLoadError('No tools stand out from the floor in this scan. Check the capture, or add shapes by hand in Design insert.');
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Finding tools failed');
    } finally {
      setFinding(null);
      findingFor.current = null;
    }
  }, [history]);

  const handleUploaded = useCallback((info: SessionInfo) => {
    setSelectedId(null);
    setModel(info.model ?? null);
    if (info.source_kind === 'object') {
      // a single tool model: already to scale, goes straight onto the mat
      const results = info.tools ?? [];
      setTools((prev) => {
        let acc = prev;
        results.forEach((r, i) => {
          const t = toolFromResult(r, prev.length + i, 'object', r.name);
          t.offset_mm = placementOffset(t.polygon_mm, acc);
          acc = [...acc, t];
        });
        return acc;
      });
      setStep('layout');
      return;
    }
    setSession(info);
    const current = history.getSnapshot().document;
    history.reset({ ...current, tools: current.tools.filter(t => t.source === 'object' || t.source === 'shape'),
      matSize: info.mat_mm ? { width_mm: info.mat_mm.width, height_mm: info.mat_mm.height } : current.matSize });
    setCorners(info.corners ?? info.suggested_corners ?? null);
    setTurns(0);
    if (info.rectified && info.mat_mm) {
      // already calibrated (phone capture with corner markers, or a re-opened session): find the tools and design
      void findTools(info);
      return;
    }
    if (info.source_kind === 'scan' && info.suggested_corners && info.scan) {
      const { horiz, vert } = edgeLengths(orderCorners(info.suggested_corners));
      history.reset({ ...history.getSnapshot().document, matSize: { width_mm: Math.round(horiz * info.scan.mm_per_px * 2) / 2, height_mm: Math.round(vert * info.scan.mm_per_px * 2) / 2 } });
    }
    setStep('calibrate');
  }, [history, setTools]);

  const handleCalibrated = useCallback((info: SessionInfo) => {
    setSession(info);
    setModel(info.model ?? null);
    const current = history.getSnapshot().document;
    history.reset({ ...current, tools: current.tools.filter(t => t.source === 'object' || t.source === 'shape'),
      matSize: info.mat_mm ? { width_mm: info.mat_mm.width, height_mm: info.mat_mm.height } : current.matSize });
  }, [history]);

  const readyTools = tools.filter((t) => t.polygon_mm.length >= 3).length;
  const hasHeight = !!session?.rectified?.has_height || tools.some((t) => t.measured_thickness_mm !== null);

  return (
    <main className={`${styles.shell} ${styles.modelingShell}`} data-workspace={step}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <h1>Foam Studio</h1>
          <span>{session?.filename || 'Untitled workspace'}</span>
        </div>
        <div className={styles.status}>
          <span className={`${styles.dot} ${backendOk === null ? '' : backendOk ? (model?.available ? styles.dotOk : styles.dotWarn) : styles.dotBad}`} />
          {backendOk === null ? 'connecting…' : backendOk ? (model?.available ? 'Connected · photo assist ready' : 'Connected · scan tools ready') : 'Server offline'}
        </div>
      </header>
      <nav className={styles.workspaceNav} aria-label="Studio workspace">
        <div className={styles.workspaceActions}>
          <button type="button" onClick={() => setStep('upload')} aria-pressed={step === 'upload'}>＋ Import tools</button>
          {session && <button type="button" onClick={() => setStep('calibrate')} aria-pressed={step === 'calibrate'}>Calibration</button>}
        </div>
        <div className={styles.viewSwitcher} role="group" aria-label="Workspace view">
          <button type="button" aria-pressed={step === 'layout'} onClick={() => { setLayoutPanel(selectedId ? 'tools' : 'foam'); setStep('layout'); }}>Design insert</button>
        </div>
        <button type="button" className={styles.exportButton} disabled={!tools.some(t => t.include && t.polygon_mm.length >= 3)} onClick={() => { setLayoutPanel('export'); setStep('layout'); }}>Export files ↗</button>
      </nav>
      {editing && <div className={styles.workspaceContext}>
        <span><strong>Design insert</strong>Arrange the tools; the foam is the negative of their scanned shapes.</span>
        <div className={styles.historyActions} role="group" aria-label="Edit history">
          <button type="button" onClick={history.undo} disabled={!canUndo} title="Undo (Ctrl/⌘ Z)">↶ Undo</button>
          <button type="button" onClick={history.redo} disabled={!canRedo} title="Redo (Ctrl/⌘ Shift Z or Ctrl Y)">↷ Redo</button>
        </div>
        <span>{readyTools} tools · {matSize.width_mm} × {matSize.height_mm} mm</span>
      </div>}
      {loadError && <p role="alert" className={styles.status}>{loadError}</p>}
      {finding && <p className={styles.status} aria-live="polite">{finding}</p>}
      <div className={styles.workspaceHeading}>
        <div>
          <h2>{{ upload: 'Start with your tools', calibrate: 'Make every millimetre count', outlines: 'Design your foam insert', layout: 'Design your foam insert' }[step]}</h2>
          <p>{{ upload: 'Open a phone capture, upload a photo, or import a 3D model.', calibrate: 'Confirm the reference corners and real dimensions before tracing.', outlines: 'Detect your tools, select an outline, then refine only what needs attention.', layout: 'Arrange your tools, set the pocket fit, and export your cutting file.' }[step]}</p>
        </div>
        {session && <div className={styles.document}><strong>{session.filename}</strong>{session.mat_mm ? `${session.mat_mm.width} × ${session.mat_mm.height} mm` : 'Scale not yet set'}{readyTools > 0 ? ` · ${readyTools} tools` : ''}</div>}
      </div>

      {step === 'upload' && <UploadStep onUploaded={handleUploaded} model={model} backendOk={backendOk} />}
      {step === 'calibrate' && session && (
        <CalibrateStep session={session} corners={corners} setCorners={setCorners} matSize={matSize} setMatSize={setMatSize}
          onCalibrated={handleCalibrated} onContinue={() => { if (session) void findTools(session); }} turns={turns} setTurns={setTurns} />
      )}
      {step === 'layout' && (
        <LayoutStep historyRevision={epoch} captureTools={history.captureTools} selectedId={selectedId} setSelectedId={setSelectedId} panel={layoutPanel} setPanel={setLayoutPanel} onRedetect={session?.rectified ? () => void findTools(session) : undefined} finding={finding} tools={tools} setTools={setTools} settings={settings} setSettings={setSettings} matSize={matSize} setMatSize={setMatSize}
          hasHeight={hasHeight} onObjectUploaded={handleUploaded} />
      )}
    </main>
  );
}
