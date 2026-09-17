'use client';

import { useCallback, useRef, useState } from 'react';
import ui from './ui.module.css';
import st from './stage.module.css';
import { createSession, type ScanKind } from '../lib/api';
import type { ModelInfo, SessionInfo } from '../lib/types';

const ACCEPT = '.ply,.obj,.stl,.glb,.gltf,.off,.xyz';

export default function UploadStep({
  onUploaded, model, backendOk,
}: { onUploaded: (info: SessionInfo) => void; model: ModelInfo | null; backendOk: boolean | null }) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [units, setUnits] = useState('auto');
  const [scanKind, setScanKind] = useState<ScanKind>('auto');

  const handleFile = useCallback(
    async (file: File | null | undefined) => {
      if (!file) return;
      setError(null);
      if (!/\.(ply|obj|stl|glb|gltf|off|xyz)$/i.test(file.name)) {
        setError('Only 3D files are accepted (PLY, OBJ, STL, GLB/GLTF, OFF, XYZ).');
        return;
      }
      setBusy('Processing 3D model: finding the mat plane or resting pose, building the height map…');
      try {
        const info = await createSession(file, units, scanKind);
        onUploaded(info);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Upload failed');
      } finally {
        setBusy(null);
      }
    },
    [onUploaded, units, scanKind],
  );

  return (
    <div className={st.stageWrap}>
      {backendOk === false && (
        <div className={ui.error}>
          Backend not reachable. Start it with <code>cd backend && python3 app.py --host 0.0.0.0</code> (port 8000).
        </div>
      )}
      {backendOk && model && !model.available && (
        <div className={ui.warn}>
          No HQ-SAM checkpoint found on the server, so click-to-outline on layout scans is disabled. Height-based auto-detect still works.
          Put <code>sam_hq_vit_h.pth</code> (or vit_l / vit_b) in <code>backend/</code> or set <code>HQSAM_CKPT</code>.
        </div>
      )}
      <div
        className={`${st.dropzone} ${drag ? st.dropActive : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); void handleFile(e.dataTransfer.files?.[0]); }}
        role="button"
        tabIndex={0}
      >
        {busy ? (
          <>
            <span className={ui.spinner} />
            <p className={st.dropSub}>{busy}</p>
          </>
        ) : (
          <>
            <p className={st.dropTitle}>Drop a 3D scan of your laid-out tools, or a 3D model of one tool</p>
            <p className={st.dropSub}>
              PLY, OBJ, GLB/GLTF, STL, OFF or XYZ — from Polycam, Scaniverse, RealityScan, or downloaded models.
            </p>
            <div className={ui.row} onClick={(e) => e.stopPropagation()}>
              <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={() => inputRef.current?.click()}>Choose 3D file…</button>
              <label className={ui.row} style={{ gap: 6 }}>
                <span className={ui.unit}>3D file is</span>
                <select className={`${ui.select} ${ui.inputSm}`} style={{ width: 'auto' }} value={scanKind} onChange={(e) => setScanKind(e.target.value as ScanKind)}>
                  <option value="auto">auto-detect</option>
                  <option value="layout">a scan of the whole layout (mat + tools)</option>
                  <option value="object">a single tool model</option>
                </select>
              </label>
              <label className={ui.row} style={{ gap: 6 }}>
                <span className={ui.unit}>Units</span>
                <select className={`${ui.select} ${ui.inputSm}`} style={{ width: 'auto' }} value={units} onChange={(e) => setUnits(e.target.value)}>
                  <option value="auto">auto</option>
                  <option value="m">meters</option>
                  <option value="cm">centimeters</option>
                  <option value="mm">millimeters</option>
                  <option value="in">inches</option>
                </select>
              </label>
            </div>
          </>
        )}
        <input ref={inputRef} type="file" accept={ACCEPT} hidden onChange={(e) => { void handleFile(e.target.files?.[0]); e.target.value = ''; }} />
      </div>
      {error && <div className={ui.error}>{error}</div>}

      <div className={st.tips}>
        <div className={ui.panel}>
          <h3>3D model of a single tool</h3>
          <ul>
            <li>Scan one tool by itself (Polycam object mode) or download a model. It is laid flat automatically on its most stable face.</li>
            <li>You get its footprint and height to scale, and it lands straight in the layout, where you drag and rotate it on the mat.</li>
            <li>Add as many as you like from the layout page; mix them with a photo of tools already on the mat.</li>
            <li>Units are guessed from the size (meters for phone scans, millimetres for CAD); override if the tool comes out 1000× wrong.</li>
          </ul>
        </div>
        <div className={ui.panel}>
          <h3>3D scan of the whole layout (gives depth too)</h3>
          <ul>
            <li>Scan the drawer or mat with the tools in place and export a mesh or point cloud (PLY/GLB with color is ideal).</li>
            <li>The mat plane is detected automatically; each tool&apos;s thickness is measured from the height above it.</li>
            <li>Phone LiDAR outlines are blobby (~5 mm); photogrammetry / object-capture modes are much sharper. Scanning each tool separately as a single model gives the cleanest pockets.</li>
            <li>Scan scale can drift a little — you can override the measured mat size with a tape-measure value.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
