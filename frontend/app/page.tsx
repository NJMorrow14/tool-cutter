'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import styles from './page.module.css';
import Stepper, { type StepDef, type StepId } from '../components/Stepper';
import UploadStep from '../components/UploadStep';
import CalibrateStep from '../components/CalibrateStep';
import DetectStep from '../components/DetectStep';
import LayoutStep from '../components/LayoutStep';
import { getHealth } from '../lib/api';
import { edgeLengths, orderCorners, placementOffset, toolFromResult } from '../lib/geom';
import { DEFAULT_SETTINGS, type LayoutSettings, type ModelInfo, type SessionInfo, type Tool } from '../lib/types';

export default function Page() {
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
    setTools((prev) => prev.filter((t) => t.source === 'object'));
    setCorners(info.suggested_corners ?? null);
    setTurns(0);
    if (info.source_kind === 'scan' && info.suggested_corners && info.scan) {
      const { horiz, vert } = edgeLengths(orderCorners(info.suggested_corners));
      setMatSize({ width_mm: Math.round(horiz * info.scan.mm_per_px * 2) / 2, height_mm: Math.round(vert * info.scan.mm_per_px * 2) / 2 });
    }
    setStep('calibrate');
  }, []);

  const handleCalibrated = useCallback((info: SessionInfo) => {
    setSession(info);
    setModel(info.model ?? null);
    setTools((prev) => prev.filter((t) => t.source === 'object'));
    if (info.mat_mm) setMatSize({ width_mm: info.mat_mm.width, height_mm: info.mat_mm.height });
  }, []);

  const readyTools = tools.filter((t) => t.polygon_mm.length >= 3).length;
  const objectCount = tools.filter((t) => t.source === 'object').length;
  const uploadHint = session ? session.filename : objectCount ? `${objectCount} tool model${objectCount > 1 ? 's' : ''}` : 'Layout scan or tool model';
  const steps: StepDef[] = useMemo(
    () => [
      { id: 'upload', title: 'Upload', hint: uploadHint, enabled: true, done: !!session || objectCount > 0 },
      { id: 'calibrate', title: 'Calibrate', hint: session?.rectified ? `${session.mat_mm?.width} × ${session.mat_mm?.height} mm` : 'Mark the mat corners + size', enabled: !!session, done: !!session?.rectified },
      { id: 'detect', title: 'Detect tools', hint: readyTools ? `${readyTools} outlined` : 'Auto-detect, then click to fix', enabled: !!session?.rectified, done: readyTools > 0 },
      { id: 'layout', title: 'Layout & export', hint: 'Depth, clearance, SVG / DXF / STL', enabled: readyTools > 0, done: false },
    ],
    [session, readyTools, objectCount, uploadHint],
  );
  const hasHeight = !!session?.rectified?.has_height || tools.some((t) => t.measured_thickness_mm !== null);

  return (
    <main className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <h1>ToolCutter</h1>
          <span>3D scans of laid-out tools → foam insert cutting files</span>
        </div>
        <div className={styles.status}>
          <span className={`${styles.dot} ${backendOk === null ? '' : backendOk ? (model?.available ? styles.dotOk : styles.dotWarn) : styles.dotBad}`} />
          {backendOk === null ? 'connecting…' : backendOk ? (model?.available ? `HQ-SAM ${model.model_type}${model.device ? ` on ${model.device}` : ''}` : 'backend up, no SAM checkpoint') : 'backend offline'}
        </div>
      </header>
      <Stepper steps={steps} current={step} onSelect={setStep} />

      {step === 'upload' && <UploadStep onUploaded={handleUploaded} model={model} backendOk={backendOk} />}
      {step === 'calibrate' && session && (
        <CalibrateStep session={session} corners={corners} setCorners={setCorners} matSize={matSize} setMatSize={setMatSize}
          onCalibrated={handleCalibrated} onContinue={() => setStep('detect')} turns={turns} setTurns={setTurns} />
      )}
      {step === 'detect' && session?.rectified && (
        <DetectStep session={session} tools={tools} setTools={setTools} modelAvailable={!!model?.available} onContinue={() => setStep('layout')} />
      )}
      {step === 'layout' && readyTools > 0 && (
        <LayoutStep tools={tools} setTools={setTools} settings={settings} setSettings={setSettings} matSize={matSize} setMatSize={setMatSize}
          hasHeight={hasHeight} onObjectUploaded={handleUploaded} />
      )}
    </main>
  );
}
