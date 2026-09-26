'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import ui from './ui.module.css';
import ed from './editor.module.css';
import { getHeightfield, imageUrl, type Heightfield } from '../lib/api';
import { disposeObject, heightfieldGeometry, makeScene, pointInPolygon, sampleHeight } from '../lib/three-util';
import { arcLengths, softDragRing } from '../lib/geom';
import type { SessionInfo, Tool } from '../lib/types';

interface Props {
  session: SessionInfo;
  tools: Tool[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  /** Tools ticked for a bulk action. Drawn in the accent colour so a gathered set reads at a glance. */
  tickedIds?: string[];
  /** Cmd/Ctrl+CLICK on a tool toggles it here. Shift and Alt are taken (hint points) and Cmd+DRAG is split,
   *  so a modifier CLICK is the one gesture still free — the same click-vs-drag split the rest of the canvas uses. */
  onToggleSelect?: (id: string) => void;
  /** Outline editing, straight on the scan. Pixel coordinates throughout, as in the rectified image. */
  edit?: EditHooks;
  softMm?: number;
  /** non-null = a shape kind is armed; a drag on bare mat draws it instead of orbiting */
  drawArmed?: boolean;
  splitArmed?: boolean;
}

export interface EditHooks {
  onEditStart: (id: string) => void;
  onEdit: (id: string, poly_px: number[][]) => void;
  /** click on bare mat */
  onCreate: (x_px: number, y_px: number) => void;
  /** Shift / Alt click: "this is part of it" / "this is not" */
  onHint: (id: string, x_px: number, y_px: number, label: 'pos' | 'neg') => void;
  /** Cmd/Ctrl-drag a line across a join */
  onSplit: (id: string, line_px: [number[], number[]]) => void;
  /** Drag on bare mat with a shape armed: corner-to-corner in mm on the mat. */
  onDrawShape?: (a: { x: number; y: number }, b: { x: number; y: number }, mods: { shift: boolean; alt: boolean }) => void;
}

/** Where a tool's outline actually sits on the mat. Scanned outlines carry their position in `polygon_mm`;
 *  drawn shapes keep a shape-local ring plus `offset_mm`, so both have to go through here or a drawn shape
 *  renders in the drawer's top-left corner instead of where it was drawn. */
function placed(t: Tool): number[][] {
  const { x, y } = t.offset_mm || { x: 0, y: 0 };
  return x || y ? t.polygon_mm.map(([px_, py]) => [px_ + x, py + y]) : t.polygon_mm;
}

/** The fused LiDAR scan of the drawer as an orbitable 3D surface with the photo draped on it and the tool
 *  outlines drawn on the surface. Click a tool to select it. */
export default function ScanViewer({ session, tools, selectedId, onSelect, tickedIds, onToggleSelect, edit, softMm = 0, drawArmed = false, splitArmed = false }: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [status, setStatus] = useState<string | null>('Loading scan…');
  const [zScale, setZScale] = useState(1);
  const [coverage, setCoverage] = useState<number | null>(null);
  const [showPhoto, setShowPhoto] = useState(true);
  const sceneRef = useRef<{ root: THREE.Group; surface: THREE.Mesh | null; outlines: THREE.Group; handles: THREE.Points | null; ring: THREE.LineLoop | null; hf: Heightfield | null; camera: THREE.PerspectiveCamera; controls: OrbitControls } | null>(null);
  const mpp = session.rectified!.mm_per_px;

  // ------------------------------------------------------------------ scene + surface
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const { renderer, scene, camera, root, dispose } = makeScene(host);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    const outlines = new THREE.Group();
    root.add(outlines);
    const state = { root, surface: null as THREE.Mesh | null, outlines, handles: null as THREE.Points | null, ring: null as THREE.LineLoop | null, hf: null as Heightfield | null, camera, controls };
    sceneRef.current = state;
    let raf = 0;
    const loop = () => { controls.update(); renderer.render(scene, camera); raf = requestAnimationFrame(loop); };
    loop();

    let disposed = false;
    const tex = new THREE.TextureLoader().load(imageUrl(session.id, 'rectified', session.version));
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    getHeightfield(session.id, 2)
      .then((hf) => {
        if (disposed) return;
        state.hf = hf;
        setCoverage(hf.valid ? hf.valid.reduce((sum, value) => sum + value, 0) / hf.valid.length : null);
        const geom = heightfieldGeometry(hf, 1);
        const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85, metalness: 0.05, side: THREE.DoubleSide });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.userData.plain = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.9, side: THREE.DoubleSide });
        root.add(mesh);
        state.surface = mesh;
        // mat plane and grid under the scan
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(hf.width_mm * 1.3, hf.height_mm * 1.3), new THREE.MeshStandardMaterial({ color: 0x303640, roughness: 1 }));
        floor.position.set(hf.width_mm / 2, hf.height_mm / 2, -0.6);
        root.add(floor);
        root.updateMatrixWorld(true);
        const center = root.localToWorld(new THREE.Vector3(hf.width_mm / 2, hf.height_mm / 2, 0));
        const dist = Math.max(hf.height_mm, hf.width_mm / camera.aspect) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.2;
        camera.position.set(center.x, center.y + dist, center.z + 0.001);
        camera.near = dist / 200;
        camera.far = dist * 30;
        camera.updateProjectionMatrix();
        controls.target.copy(center);
        camera.lookAt(center);
        controls.update();
        setStatus(null);
        rebuildOutlines();
      })
      .catch((err) => setStatus(err instanceof Error ? err.message : 'No height data'));

    // ---------------------------------------------------------------- pointer: select, create, reshape, split
    const ray = new THREE.Raycaster();
    let down: { x: number; y: number; shift: boolean; alt: boolean; meta: boolean; ctrl: boolean } | null = null;
    let hdrag: { id: string; anchor: number; poly0: number[][]; arc: number[]; per: number; from: THREE.Vector3; z0: number } | null = null;
    let cut: { id: string; from: THREE.Vector3; line: THREE.Line } | null = null;
    let draw: { from: THREE.Vector3; box: THREE.Line } | null = null;

    /** Screen point -> root-local mm. Uses the scan surface when the ray meets it, else the horizontal plane
     *  at `z0` — a drag has to keep working when the pointer leaves the mesh. */
    const localAt = (e: PointerEvent, z0: number | null): THREE.Vector3 | null => {
      const r = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      if (z0 === null) {
        const hit = state.surface ? ray.intersectObject(state.surface, false)[0] : undefined;
        return hit ? root.worldToLocal(hit.point.clone()) : null;
      }
      const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
      const lr = ray.ray.clone().applyMatrix4(inv);
      if (Math.abs(lr.direction.z) < 1e-6) return null;
      const t = (z0 - lr.origin.z) / lr.direction.z;
      return t > 0 ? lr.origin.clone().addScaledVector(lr.direction, t) : null;
    };
    const px = (v: THREE.Vector3 | number[]) => (Array.isArray(v) ? [v[0] / mpp, v[1] / mpp] : [v.x / mpp, v.y / mpp]);
    /** mm tolerance that grows with camera distance, so handles stay grabbable zoomed out */
    const tolAt = (p: THREE.Vector3) => Math.max(3, camera.position.distanceTo(root.localToWorld(p.clone())) * 0.02);

    const onDown = (e: PointerEvent) => {
      down = { x: e.clientX, y: e.clientY, shift: e.shiftKey, alt: e.altKey, meta: e.metaKey, ctrl: e.ctrlKey };
      hdrag = null;
      const hooks = editRef.current;
      const t = editableRef.current;
      if (!hooks || !state.hf) return;

      if (drawRef.current) {
        // a shape kind is armed: rubber-band on the MAT PLANE, not the scan surface. Requiring a hit on the
        // mesh meant a drag started on the grey around a narrow drawer did nothing at all, which just read
        // as "dragged shapes are not being added".
        const p0 = localAt(e, 0);
        if (!p0) return;
        const geom = new THREE.BufferGeometry().setFromPoints([p0.clone(), p0.clone(), p0.clone(), p0.clone(), p0.clone()]);
        const box = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0x2563eb, depthTest: false }));
        box.renderOrder = 4;
        state.outlines.add(box);
        draw = { from: p0.clone(), box };
        controls.enabled = false;
        return;
      }
      const p = localAt(e, null);
      if (!p) return;
      if (t && (e.metaKey || e.ctrlKey || splitRef.current)) {           // draw a line across a join, then split on release
        const geom = new THREE.BufferGeometry().setFromPoints([p.clone(), p.clone()]);
        const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0xef4444, depthTest: false }));
        line.renderOrder = 4;
        state.outlines.add(line);
        cut = { id: t.id, from: p.clone(), line };
        controls.enabled = false;
        return;
      }
      if (!t || e.shiftKey || e.altKey) return;      // modifier clicks are decided on release

      const tol = tolAt(p);
      let best = -1, bd = tol;
      t.polygon_mm.forEach((v, i) => { const d = Math.hypot(v[0] - p.x, v[1] - p.y); if (d < bd) { bd = d; best = i; } });
      if (best < 0) return;
      const { arc, per } = arcLengths(t.polygon_mm);
      const z0 = 0.8;                                  // handles sit on the mat now; drag on that plane
      const from = localAt(e, z0);
      if (!from) return;
      hdrag = { id: t.id, anchor: best, poly0: t.polygon_mm, arc, per, from, z0 };
      controls.enabled = false;                      // capture phase: OrbitControls bails on enabled === false
      hooks.onEditStart(t.id);
    };

    const onMove = (e: PointerEvent) => {
      if (draw) {
        const p = localAt(e, draw.from.z);
        if (p) {
          const a = draw.from, z = a.z;
          draw.box.geometry.setFromPoints([new THREE.Vector3(a.x, a.y, z), new THREE.Vector3(p.x, a.y, z),
            new THREE.Vector3(p.x, p.y, z), new THREE.Vector3(a.x, p.y, z), new THREE.Vector3(a.x, a.y, z)]);
        }
        down = null;
        return;
      }
      if (cut) {
        const p = localAt(e, cut.from.z);
        if (p) cut.line.geometry.setFromPoints([cut.from, p]);
        down = null;
        return;
      }
      if (!hdrag) return;
      const p = localAt(e, hdrag.z0);
      if (!p) return;
      const mm = softDragRing(hdrag.poly0, hdrag.anchor, hdrag.arc, hdrag.per, p.x - hdrag.from.x, p.y - hdrag.from.y, softRef.current);
      editRef.current?.onEdit(hdrag.id, mm.map(px));
      down = null;                                   // a handle drag is never also a click
    };

    const onUp = (e: PointerEvent) => {
      if (draw) {
        const p = localAt(e, draw.from.z);
        state.outlines.remove(draw.box);
        draw.box.geometry.dispose();
        const a = draw.from;
        if (p && Math.hypot(p.x - a.x, p.y - a.y) > 3) editRef.current?.onDrawShape?.({ x: a.x, y: a.y }, { x: p.x, y: p.y }, { shift: e.shiftKey, alt: e.altKey });
        draw = null; controls.enabled = true; down = null;
        return;
      }
      if (cut) {
        const p = localAt(e, cut.from.z);
        const far = p && Math.hypot(p.x - cut.from.x, p.y - cut.from.y) > 4;
        state.outlines.remove(cut.line);
        cut.line.geometry.dispose();
        if (far && p) editRef.current?.onSplit(cut.id, [px(cut.from), px(p)]);
        cut = null; controls.enabled = true; down = null;
        return;
      }
      if (hdrag) { hdrag = null; controls.enabled = true; down = null; return; }
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) { down = null; return; }
      const mods = down;
      down = null;
      const p = localAt(e, null);
      if (!p) { onSelectRef.current(null); return; }
      const hooks = editRef.current;
      const t = editableRef.current;
      if (hooks && t && (mods.shift || mods.alt)) { hooks.onHint(t.id, p.x / mpp, p.y / mpp, mods.alt ? 'neg' : 'pos'); return; }
      const found = toolsRef.current.find((x) => x.polygon_mm.length >= 3 && pointInPolygon(p.x, p.y, placed(x)));
      if (found && (mods.meta || mods.ctrl) && onToggleRef.current) { onToggleRef.current(found.id); return; }
      if (found) { onSelectRef.current(found.id); return; }
      if (selectedRef.current) { onSelectRef.current(null); return; }   // first click on the mat lets go
      hooks?.onCreate(p.x / mpp, p.y / mpp);                            // then a click outlines something new
    };

    renderer.domElement.addEventListener('pointerdown', onDown, true);
    renderer.domElement.addEventListener('pointermove', onMove);
    renderer.domElement.addEventListener('pointerup', onUp);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener('pointerdown', onDown, true);
      renderer.domElement.removeEventListener('pointermove', onMove);
      renderer.domElement.removeEventListener('pointerup', onUp);
      controls.dispose();
      tex.dispose();
      dispose();
      sceneRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.id, session.version]);

  // keep latest props reachable from the imperative handlers
  const toolsRef = useRef(tools);
  toolsRef.current = tools;
  const onSelectRef = useRef(onSelect);
  const onToggleRef = useRef(onToggleSelect);
  onToggleRef.current = onToggleSelect;
  const tickedRef = useRef<string[]>(tickedIds ?? []);
  onSelectRef.current = onSelect;
  const zRef = useRef(zScale);
  zRef.current = zScale;
  const editable = tools.find((t) => t.id === selectedId && t.source !== 'shape' && t.polygon_mm.length >= 3) ?? null;
  const editableRef = useRef(editable);
  editableRef.current = editable;
  const editRef = useRef(edit);
  editRef.current = edit;
  const softRef = useRef(softMm);
  softRef.current = softMm;
  const splitRef = useRef(splitArmed);
  splitRef.current = splitArmed;
  const drawRef = useRef(drawArmed);
  drawRef.current = drawArmed;

  // ------------------------------------------------------------------ outlines
  /** Snap the camera straight down over the middle of the scan. */
  const overhead = () => {
    const s = sceneRef.current;
    if (!s || !s.hf) return;
    s.root.updateMatrixWorld(true);
    const centre = s.root.localToWorld(new THREE.Vector3(s.hf.width_mm / 2, s.hf.height_mm / 2, 0));
    const dist = Math.max(s.hf.height_mm, s.hf.width_mm / s.camera.aspect) / (2 * Math.tan(THREE.MathUtils.degToRad(s.camera.fov / 2))) * 1.2;
    s.camera.position.set(centre.x, centre.y + dist, centre.z + 0.001);   // a hair off vertical keeps orbit sane
    s.camera.near = dist / 200;
    s.camera.far = dist * 30;
    s.camera.updateProjectionMatrix();
    s.controls.target.copy(centre);
    s.camera.lookAt(centre);
    s.controls.update();
  };

  const perspective = () => {
    const s = sceneRef.current;
    if (!s?.hf) return;
    overhead();
    const distance = s.camera.position.distanceTo(s.controls.target);
    s.camera.position.copy(s.controls.target).add(new THREE.Vector3(.55, .85, .7).normalize().multiplyScalar(distance));
    s.camera.lookAt(s.controls.target);
    s.controls.update();
  };
  const frameSelection = () => {
    const s = sceneRef.current;
    const tool = tools.find(t => t.id === selectedId);
    if (!s?.hf || !tool || !tool.polygon_mm.length) { overhead(); return; }
    const points = placed(tool);
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const centre = s.root.localToWorld(new THREE.Vector3((x0+x1)/2, (y0+y1)/2, sampleHeight(s.hf, (x0+x1)/2, (y0+y1)/2)));
    const span = Math.max(30, y1-y0, (x1-x0)/s.camera.aspect);
    const distance = span / (2 * Math.tan(THREE.MathUtils.degToRad(s.camera.fov / 2))) * 1.6;
    const direction = s.camera.position.clone().sub(s.controls.target).normalize();
    s.camera.position.copy(centre).addScaledVector(direction, distance);
    s.controls.target.copy(centre);
    s.controls.update();
  };
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || (event.target as HTMLElement)?.closest('input, textarea, select, [contenteditable=true]')) return;
      if (event.key.toLowerCase() === 'f') { event.preventDefault(); frameSelection(); }
      if (event.key === '1') overhead();
      if (event.key === '3') perspective();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const rebuildOutlines = () => {
    const s = sceneRef.current;
    if (!s || !s.hf) return;
    disposeObject(s.outlines);
    s.outlines.clear();
    for (const t of toolsRef.current) {
      if (t.polygon_mm.length < 3) continue;
      const sel = t.id === selectedRef.current;
      const ticked = tickedRef.current.includes(t.id);
      const poly = placed(t);
      // The outline is the FOOTPRINT — where the cutter goes — so it is drawn ON THE MAT, not draped up the tool.
      // Draping it on the surface put the line on a near-vertical wall, where the 2 mm height grid turns a 0.3 mm
      // lateral wobble into centimetres of vertical zigzag (Nolan, 2026-09-23: "still very rocky" after the 2D shape
      // was fixed). Measured on the tape measure: bilinear drape 2.2 mm step-to-step, floor-side sampling 1.95 mm
      // — no sampling scheme follows a cliff on that grid. A planar line has zero roughness by construction and reads
      // as what it is. depthTest off so the far side is not swallowed by the tool body it sits against.
      const LIFT = 0.8;
      const pts = poly.map(([x, y]) => new THREE.Vector3(x, y, LIFT));
      const geom = new THREE.BufferGeometry().setFromPoints(pts);
      const flat = (color: string, opacity = 1) => new THREE.LineBasicMaterial({ color: new THREE.Color(color), linewidth: 1, transparent: opacity < 1, opacity, depthTest: false });
      const line = new THREE.LineLoop(geom, flat(sel ? '#ffffff' : ticked ? '#f26a1b' : t.color));
      line.renderOrder = 2;
      s.outlines.add(line);
      if (ticked && !sel) {          // a gathered tool gets the same double-line emphasis as the selected one
        const mark = new THREE.LineLoop(geom, flat('#f26a1b', 0.9));
        mark.position.z = 0.8; mark.renderOrder = 2;
        s.outlines.add(mark);
      }
      if (sel) {
        const glow = new THREE.LineLoop(geom, flat(t.color, 0.9));
        glow.position.z = 0.8; glow.renderOrder = 2;
        s.outlines.add(glow);
        if (editRef.current) {
          // one handle per vertex, drawn as screen-sized points so they stay grabbable at any zoom
          const hp = poly.map(([x, y]) => new THREE.Vector3(x, y, LIFT + 1.4));
          const pts2 = new THREE.Points(new THREE.BufferGeometry().setFromPoints(hp),
            new THREE.PointsMaterial({ color: 0xffffff, size: 7, sizeAttenuation: false, depthTest: false }));
          pts2.renderOrder = 3;
          s.outlines.add(pts2);
          s.handles = pts2;
        }
      }
    }
  };
  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;
  tickedRef.current = tickedIds ?? [];

  // Test hook, same idea as Layout3D's __layout3d: headless tests need a tool's SCREEN position to click it,
  // and clicking the middle of the canvas hides exactly the bugs worth catching.
  useEffect(() => {
    (window as unknown as { __scan3d?: unknown }).__scan3d = {
      ids: () => toolsRef.current.filter((t) => t.polygon_mm.length >= 3).map((t) => t.id),
      screenPos: (id: string) => {
        const s = sceneRef.current, el = hostRef.current;
        const t = toolsRef.current.find((x) => x.id === id);
        if (!s || !el || !t || t.polygon_mm.length < 3) return null;
        const poly = placed(t);
        let cx = 0, cy = 0;
        for (const q of poly) { cx += q[0]; cy += q[1]; }
        cx /= poly.length; cy /= poly.length;
        const v = new THREE.Vector3(cx, cy, (s.hf ? sampleHeight(s.hf, cx, cy) : 0) * zRef.current + 1);
        s.root.localToWorld(v);
        v.project(s.camera);
        const r = el.getBoundingClientRect();
        return { x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height };
      },
    };
  });
  useEffect(() => { rebuildOutlines(); }, [tools, selectedId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ------------------------------------------------------------------ height exaggeration / photo toggle
  useEffect(() => {
    const s = sceneRef.current;
    if (!s || !s.surface || !s.hf) return;
    const geom = heightfieldGeometry(s.hf, zScale);
    s.surface.geometry.dispose();
    s.surface.geometry = geom;
    rebuildOutlines();
  }, [zScale]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const s = sceneRef.current;
    if (!s?.surface) return;
    const photoMat = s.surface.material as THREE.Material;
    const plain = s.surface.userData.plain as THREE.Material;
    if (!s.surface.userData.photo) s.surface.userData.photo = photoMat;
    s.surface.material = showPhoto ? (s.surface.userData.photo as THREE.Material) : plain;
  }, [showPhoto]);

  return (
    <div className={ed.canvasWrap} ref={hostRef} style={{ position: 'relative' }}>
      <div className={ed.viewControls} role="toolbar" aria-label="Viewport navigation">
        <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={overhead} title="Top view · 1">Top <kbd>1</kbd></button>
        <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={perspective} title="Perspective view · 3">Perspective <kbd>3</kbd></button>
        <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={frameSelection} title="Frame selected tool, or fit the drawer · F">{selectedId ? 'Frame selected' : 'Fit all'} <kbd>F</kbd></button>
        <label className={ui.checkbox}><input type="checkbox" checked={showPhoto} onChange={(e) => setShowPhoto(e.target.checked)} /> Texture</label>
        <label className={ed.slider} title="Exaggerate heights for inspection"><span>Relief</span>
          <input type="range" min={1} max={4} step={0.5} value={zScale} onChange={(e) => setZScale(Number(e.target.value))} style={{ width: 56 }} /><span>{zScale}×</span></label>
      </div>
      <div className={ed.viewportStatus}>
        {status && <span role="status">{status}</span>}
        <span>mm · 2 mm display grid{coverage !== null && ` · ${Math.round(coverage * 100)}% depth coverage`}</span>
        <span className={ed.viewHelp}>Drag orbit · Right-drag pan · Scroll zoom</span>
        {coverage !== null && coverage < .98 && <span style={{ color: '#fcd34d' }}>Unmeasured areas shown as gaps</span>}
      </div>
    </div>
  );
}
