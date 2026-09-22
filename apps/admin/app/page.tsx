'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import styles from './page.module.css';
import Stepper, { type StepDef, type StepId } from '../components/Stepper';
import UploadStep from '../components/UploadStep';
import CalibrateStep from '../components/CalibrateStep';
import OutlineStep from '../components/OutlineStep';
import LayoutStep from '../components/LayoutStep';
import { getHealth, getSession } from '../lib/api';
import { edgeLengths, orderCorners, placementOffset, toolFromResult } from '../lib/geom';
import { DEFAULT_SETTINGS, type LayoutSettings, type ModelInfo, type SessionInfo, type Tool } from '../lib/types';

export default function Page() {
  const [loadError, setLoadError] = useState<string | null>(null);
  const [step, setStep] = useState<StepId>('upload');
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [corners, setCorners] = useState<number[][] | null>(null);
  const [turns, setTurns] = useState(0);
  const [matSize, setMatSize] = useState({ width_mm: 279.4, height_mm: 215.9 });
  const [tools, setTools] = useState<Tool[]>([]);
  const [settings, setSettings] = useState<LayoutSettings>(DEFAULT_SETTINGS);

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

  const handleUploaded = useCallback((info: SessionInfo) => {
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
    setTools((prev) => prev.filter((t) => t.source === 'object' || t.source === 'shape'));
    setCorners(info.corners ?? info.suggested_corners ?? null);
    setTurns(0);
    if (info.rectified && info.mat_mm) {
      // already calibrated (phone capture with corner markers, or a re-opened session)
      setMatSize({ width_mm: info.mat_mm.width, height_mm: info.mat_mm.height });
      setStep('outlines');
      return;
    }
    if (info.source_kind === 'scan' && info.suggested_corners && info.scan) {
      const { horiz, vert } = edgeLengths(orderCorners(info.suggested_corners));
      setMatSize({ width_mm: Math.round(horiz * info.scan.mm_per_px * 2) / 2, height_mm: Math.round(vert * info.scan.mm_per_px * 2) / 2 });
    }
    setStep('calibrate');
  }, []);

  const handleCalibrated = useCallback((info: SessionInfo) => {
    setSession(info);
    setModel(info.model ?? null);
    setTools((prev) => prev.filter((t) => t.source === 'object' || t.source === 'shape'));
    if (info.mat_mm) setMatSize({ width_mm: info.mat_mm.width, height_mm: info.mat_mm.height });
  }, []);

  const readyTools = tools.filter((t) => t.polygon_mm.length >= 3).length;
  const editedTools = tools.filter((t) => t.edited).length;
  const objectCount = tools.filter((t) => t.source === 'object').length;
  const uploadHint = session ? session.filename : objectCount ? `${objectCount} tool model${objectCount > 1 ? 's' : ''}` : 'Layout scan or tool model';
  const steps: StepDef[] = useMemo(
    () => [
      { id: 'upload', title: 'Capture', hint: uploadHint, enabled: true, done: !!session || objectCount > 0 },
      { id: 'calibrate', title: 'Set scale', hint: session?.rectified ? `${session.mat_mm?.width} × ${session.mat_mm?.height} mm` : 'Mark the mat corners + size', enabled: !!session, done: !!session?.rectified },
      { id: 'outlines', title: 'Outlines', hint: readyTools ? `${readyTools} tool${readyTools > 1 ? 's' : ''}${editedTools ? `, ${editedTools} hand-edited` : ''}` : 'Find tools and refine their edges', enabled: !!session?.rectified, done: readyTools > 0 },
      { id: 'layout', title: 'Layout & export', hint: 'Arrange pockets and download', enabled: readyTools > 0, done: false },
    ],
    [session, readyTools, editedTools, objectCount, uploadHint],
  );
  const hasHeight = !!session?.rectified?.has_height || tools.some((t) => t.measured_thickness_mm !== null);

  return (
    <main className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <h1>Foam studio</h1>
          <span>From your tools to a precisely fitted insert.</span>
        </div>
        <div className={styles.status}>
          <span className={`${styles.dot} ${backendOk === null ? '' : backendOk ? (model?.available ? styles.dotOk : styles.dotWarn) : styles.dotBad}`} />
          {backendOk === null ? 'connecting…' : backendOk ? (model?.available ? 'Connected · photo assist ready' : 'Connected · scan tools ready') : 'Server offline'}
        </div>
      </header>
      <Stepper steps={steps} current={step} onSelect={setStep} />
      {loadError && <p role="alert" className={styles.status}>{loadError}</p>}
      <div className={styles.workspaceHeading}>
        <div>
          <h2>{{ upload: 'Start with your tools', calibrate: 'Make every millimetre count', outlines: 'Make the outline fit', layout: 'Design your foam insert' }[step]}</h2>
          <p>{{ upload: 'Open a phone capture, upload a photo, or import a 3D model.', calibrate: 'Confirm the reference corners and real dimensions before tracing.', outlines: 'Detect your tools, select an outline, then refine only what needs attention.', layout: 'Arrange your tools, set the pocket fit, and export your cutting file.' }[step]}</p>
        </div>
        {session && <div className={styles.document}><strong>{session.filename}</strong>{session.mat_mm ? `${session.mat_mm.width} × ${session.mat_mm.height} mm` : 'Scale not yet set'}{readyTools > 0 ? ` · ${readyTools} tools` : ''}</div>}
      </div>

      {step === 'upload' && <UploadStep onUploaded={handleUploaded} model={model} backendOk={backendOk} />}
      {step === 'calibrate' && session && (
        <CalibrateStep session={session} corners={corners} setCorners={setCorners} matSize={matSize} setMatSize={setMatSize}
          onCalibrated={handleCalibrated} onContinue={() => setStep('outlines')} turns={turns} setTurns={setTurns} />
      )}
      {step === 'outlines' && session?.rectified && (
        <OutlineStep session={session} tools={tools} setTools={setTools} modelAvailable={!!model?.available} onContinue={() => setStep('layout')} />
      )}
      {step === 'layout' && readyTools > 0 && (
        <LayoutStep tools={tools} setTools={setTools} settings={settings} setSettings={setSettings} matSize={matSize} setMatSize={setMatSize}
          hasHeight={hasHeight} onObjectUploaded={handleUploaded} />
      )}
    </main>
  );
}
