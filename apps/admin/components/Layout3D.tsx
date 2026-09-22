'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import ui from './ui.module.css';
import ed from './editor.module.css';
import { getToolHeightfield, resolveDepth, type Heightfield } from '../lib/api';
import { polygonCentroid } from '../lib/geom';
import { disposeObject, frameBox, heightfieldGeometry, makeScene, polygonShape, ringsShape } from '../lib/three-util';
import type { LayoutResponse, LayoutSettings, Tool } from '../lib/types';

interface Props {
  tools: Tool[];
  layout: LayoutResponse | null;
  mat: { width_mm: number; height_mm: number };
  settings: LayoutSettings;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  /** committed after a drag: move the tool by (dx, dy) mm */
  onMove: (id: string, dx: number, dy: number) => void;
}

type ToolNode = { group: THREE.Group; body: THREE.Mesh; key: string };

/** The foam block with the scanned tool bodies sitting in their pockets. Drag a tool to move it on the mat;
 *  the same keyboard nudges/rotation as the 2D sheet apply. Pockets come from the computed layout. */
export default function Layout3D({ tools, layout, mat, settings, selectedId, onSelect, onMove }: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [showTools, setShowTools] = useState(true);
  const S = useRef<{
    root: THREE.Group; camera: THREE.PerspectiveCamera; controls: OrbitControls; renderer: THREE.WebGLRenderer;
    foam: THREE.Mesh | null; pockets: THREE.Group; toolsGroup: THREE.Group; nodes: Map<string, ToolNode>; hfs: Map<string, Heightfield | null>;
    drag: { id: string; start: THREE.Vector3; delta: THREE.Vector3; moved: boolean } | null;
  } | null>(null);
  const propsRef = useRef({ tools, layout, mat, settings, selectedId, onSelect, onMove });
  propsRef.current = { tools, layout, mat, settings, selectedId, onSelect, onMove };

  // ------------------------------------------------------------------ scene
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const { renderer, scene, camera, root, dispose } = makeScene(host);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    const pockets = new THREE.Group(); root.add(pockets);
    const toolsGroup = new THREE.Group(); root.add(toolsGroup);
    const state = { root, camera, controls, renderer, foam: null as THREE.Mesh | null, pockets, toolsGroup, nodes: new Map<string, ToolNode>(), hfs: new Map<string, Heightfield | null>(), drag: null as null | { id: string; start: THREE.Vector3; delta: THREE.Vector3; moved: boolean } };
    S.current = state;
    let raf = 0;
    const loop = () => { controls.update(); renderer.render(scene, camera); raf = requestAnimationFrame(loop); };
    loop();

    // -------------------------------------------------------------- pointer: drag tools on the mat plane
    const ray = new THREE.Raycaster();
    const matPlane = new THREE.Plane(); // set per frame from root
    const hitMat = (e: PointerEvent): THREE.Vector3 | null => {
      const r = renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
      root.updateMatrixWorld(true);
      const n = new THREE.Vector3(0, 0, 1).transformDirection(root.matrixWorld);
      const o = root.localToWorld(new THREE.Vector3(0, 0, 0));
      matPlane.setFromNormalAndCoplanarPoint(n, o);
      const hit = new THREE.Vector3();
      return ray.ray.intersectPlane(matPlane, hit) ? root.worldToLocal(hit) : null;
    };
    const pickTool = (e: PointerEvent): string | null => {
      const r = renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
      const hits = ray.intersectObjects(Array.from(state.nodes.values()).map((n) => n.body), false);
      return hits.length ? (hits[0].object.userData.toolId as string) : null;
    };
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      const id = pickTool(e);
      if (!id) return;
      const start = hitMat(e);
      if (!start) return;
      controls.enabled = false;
      state.drag = { id, start, delta: new THREE.Vector3(), moved: false };
      propsRef.current.onSelect(id);
      renderer.domElement.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      const d = state.drag;
      if (!d) return;
      const p = hitMat(e);
      if (!p) return;
      d.delta.set(p.x - d.start.x, p.y - d.start.y, 0);
      if (d.delta.length() > 0.3) d.moved = true;
      const node = state.nodes.get(d.id);
      if (node) { node.group.position.x = (node.group.userData.baseX as number) + d.delta.x; node.group.position.y = (node.group.userData.baseY as number) + d.delta.y; }
    };
    const onUp = (e: PointerEvent) => {
      const d = state.drag;
      controls.enabled = true;
      if (!d) { if (e.button === 0 && !pickTool(e)) propsRef.current.onSelect(null); return; }
      state.drag = null;
      if (d.moved) propsRef.current.onMove(d.id, Math.round(d.delta.x * 10) / 10, Math.round(d.delta.y * 10) / 10);
    };
    const el = renderer.domElement;
    el.addEventListener('pointerdown', onDown); el.addEventListener('pointermove', onMove); el.addEventListener('pointerup', onUp);
    // test hook: screen position (CSS px, viewport) of a tool body's centre
    (window as unknown as { __layout3d?: unknown }).__layout3d = {
      ids: () => Array.from(state.nodes.keys()),
      bounds: (id: string) => {
        const node = state.nodes.get(id);
        if (!node) return null;
        node.group.updateMatrixWorld(true);
        const b = new THREE.Box3().setFromObject(node.body);
        const lo = root.worldToLocal(b.min.clone()), hi = root.worldToLocal(b.max.clone());
        return { x: [Math.min(lo.x, hi.x), Math.max(lo.x, hi.x)], y: [Math.min(lo.y, hi.y), Math.max(lo.y, hi.y)], z: [Math.min(lo.z, hi.z), Math.max(lo.z, hi.z)], kind: node.body.geometry.type };
      },
      pockets: () => state.pockets.children.map((o) => { const b = new THREE.Box3().setFromObject(o); const lo = root.worldToLocal(b.min.clone()), hi = root.worldToLocal(b.max.clone()); return { type: o.type, geom: (o as THREE.Mesh).geometry?.type, x: [Math.round(Math.min(lo.x, hi.x)), Math.round(Math.max(lo.x, hi.x))], y: [Math.round(Math.min(lo.y, hi.y)), Math.round(Math.max(lo.y, hi.y))], color: ((o as THREE.Mesh).material as THREE.MeshStandardMaterial)?.color?.getHexString?.() }; }),
      screenPos: (id: string) => {
        const node = state.nodes.get(id);
        if (!node) return null;
        node.group.updateMatrixWorld(true);
        const p = new THREE.Vector3(node.group.position.x, node.group.position.y, node.group.position.z + 5);
        root.localToWorld(p).project(camera);
        const r = renderer.domElement.getBoundingClientRect();
        return { x: r.left + ((p.x + 1) / 2) * r.width, y: r.top + ((1 - p.y) / 2) * r.height };
      },
    };

    return () => {
      cancelAnimationFrame(raf);
      el.removeEventListener('pointerdown', onDown); el.removeEventListener('pointermove', onMove); el.removeEventListener('pointerup', onUp);
      controls.dispose();
      dispose();
      S.current = null;
    };
  }, []);

  // ------------------------------------------------------------------ foam block
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    if (s.foam) { s.root.remove(s.foam); disposeObject(s.foam); }
    const T = settings.mat_thickness_mm;
    const geom = new THREE.BoxGeometry(mat.width_mm, mat.height_mm, T);
    const foam = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: 0x3b4252, roughness: 0.95 }));
    foam.position.set(mat.width_mm / 2, mat.height_mm / 2, -T / 2);
    s.root.add(foam);
    s.foam = foam;
    s.root.updateMatrixWorld(true);
    const center = s.root.localToWorld(new THREE.Vector3(mat.width_mm / 2, mat.height_mm / 2, 0));
    if (!s.camera.userData.framed) { frameBox(s.camera, center, mat.width_mm, mat.height_mm, s.controls.target); s.camera.userData.framed = true; }
  }, [mat.width_mm, mat.height_mm, settings.mat_thickness_mm]);

  // ------------------------------------------------------------------ pockets (from the computed layout)
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    disposeObject(s.pockets); s.pockets.clear();
    if (!layout) return;
    const T = settings.mat_thickness_mm;
    for (const lt of layout.tools) {
      const shape = ringsShape(lt.rings);
      if (!shape) continue;
      const depth = lt.depth_mm === null || lt.depth_mm >= T - 0.25 ? T : Math.min(Math.max(lt.depth_mm, 0.5), T - 2);
      const through = depth >= T;
      const bad = lt.outside_mat || lt.overlaps.length > 0;
      // pocket = a hole: draw the pocket floor at -depth and dark walls as an extrusion with the shape's inside
      const floorGeom = new THREE.ShapeGeometry(shape);
      const floor = new THREE.Mesh(floorGeom, new THREE.MeshStandardMaterial({ color: bad ? 0xdc2626 : through ? 0xf26a1b : 0x1b1f22, roughness: 0.95, side: THREE.DoubleSide }));
      floor.position.z = -depth + 0.05;
      s.pockets.add(floor);
      // cap: remove the foam top inside the pocket by painting it with the pocket colour slightly above the surface
      const cap = new THREE.Mesh(floorGeom.clone(), new THREE.MeshStandardMaterial({ color: bad ? 0xdc2626 : through ? 0xf26a1b : 0x161a1d, roughness: 0.95, side: THREE.DoubleSide, transparent: true, opacity: 0.92 }));
      cap.position.z = 0.15;
      s.pockets.add(cap);
      const edges = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(lt.rings[0].map(([x, y]) => new THREE.Vector3(x, y, 0.3))), new THREE.LineBasicMaterial({ color: bad ? 0xdc2626 : 0x9ca3af }));
      s.pockets.add(edges);
      if (lt.notch) {
        const ring = new THREE.Mesh(new THREE.RingGeometry(lt.notch.diameter_mm / 2 - 0.6, lt.notch.diameter_mm / 2, 32), new THREE.MeshBasicMaterial({ color: 0xf59e0b, side: THREE.DoubleSide }));
        ring.position.set(lt.notch.x_mm, lt.notch.y_mm, 0.4);
        s.pockets.add(ring);
      }
    }
  }, [layout, settings.mat_thickness_mm]);

  // ------------------------------------------------------------------ tool bodies
  const buildBody = async (t: Tool, s: NonNullable<typeof S.current>): Promise<THREE.Mesh> => {
    let hf = s.hfs.get(t.id);
    if (hf === undefined && t.source !== 'object' && t.source !== 'shape' && !t.edited) {
      try { hf = await getToolHeightfield(t.session_id, t.id, 1); } catch { hf = null; }
      s.hfs.set(t.id, hf);
    }
    const thick = t.measured_thickness_mm ?? resolveDepth(t, propsRef.current.settings) ?? 10;
    const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(t.color).lerp(new THREE.Color(0xbfc5cc), 0.55), roughness: 0.5, metalness: 0.45 });
    let geom: THREE.BufferGeometry;
    if (hf) {
      geom = heightfieldGeometry(hf, 1, 0, 0.2);   // heights relative to the drawer floor; flat margin cells dropped
    } else {
      geom = new THREE.ExtrudeGeometry(polygonShape(t.polygon_mm), { depth: thick, bevelEnabled: false });
    }
    const mesh = new THREE.Mesh(geom, material);
    mesh.userData.toolId = t.id;
    return mesh;
  };

  useEffect(() => {
    const s = S.current;
    if (!s) return;
    let cancelled = false;
    const want = new Map(tools.filter((t) => t.include && t.polygon_mm.length >= 3).map((t) => [t.id, t] as const));
    // drop stale
    for (const [id, node] of Array.from(s.nodes.entries())) {
      const t = want.get(id);
      const key = t ? `${t.polygon_mm.length}:${t.measured_thickness_mm}:${t.edited ? 1 : 0}:${t.color}` : '';
      if (!t || node.key !== key) { s.toolsGroup.remove(node.group); disposeObject(node.group); s.nodes.delete(id); }
    }
    // build missing
    const missing = Array.from(want.values()).filter((t) => !s.nodes.has(t.id));
    if (missing.length) setStatus(`Building ${missing.length} tool bod${missing.length > 1 ? 'ies' : 'y'}…`);
    void Promise.all(missing.map(async (t) => {
      const body = await buildBody(t, s);
      if (cancelled || !S.current) { disposeObject(body); return; }
      const group = new THREE.Group();
      group.add(body);
      const node: ToolNode = { group, body, key: `${t.polygon_mm.length}:${t.measured_thickness_mm}:${t.edited ? 1 : 0}:${t.color}` };
      s.nodes.set(t.id, node);
      s.toolsGroup.add(group);
      placeAll();
    })).then(() => { if (!cancelled) setStatus(null); });
    placeAll();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tools.map((t) => `${t.id}:${t.include}:${t.polygon_mm.length}:${t.measured_thickness_mm}:${t.edited}:${t.color}`).join('|')]);

  // ------------------------------------------------------------------ placement: rotate about the source centroid, then offset, then drop into the pocket
  const placeAll = () => {
    const s = S.current;
    if (!s) return;
    const { tools: ts, layout: lay, settings: st, selectedId: sel } = propsRef.current;
    const T = st.mat_thickness_mm;
    for (const t of ts) {
      const node = s.nodes.get(t.id);
      if (!node) continue;
      const c = polygonCentroid(t.polygon_mm);
      const lt = lay?.tools.find((x) => x.id === t.id);
      const d = lt ? (lt.depth_mm === null || lt.depth_mm >= T - 0.25 ? T : Math.min(Math.max(lt.depth_mm, 0.5), T - 2)) : (resolveDepth(t, st) ?? T);
      node.body.position.set(-c.x, -c.y, 0);
      node.group.rotation.z = (t.rotation_deg * Math.PI) / 180;
      node.group.userData.baseX = c.x + t.offset_mm.x;
      node.group.userData.baseY = c.y + t.offset_mm.y;
      if (!(s.drag && s.drag.id === t.id)) node.group.position.set(c.x + t.offset_mm.x, c.y + t.offset_mm.y, -d);
      else node.group.position.z = -d;
      const m = node.body.material as THREE.MeshStandardMaterial;
      m.emissive.set(t.id === sel ? 0x3b82f6 : 0x000000);
      m.emissiveIntensity = t.id === sel ? 0.35 : 0;
    }
  };
  useEffect(() => { placeAll(); }, [tools, layout, settings, selectedId]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { const s = S.current; if (s) s.toolsGroup.visible = showTools; }, [showTools]);

  return (
    <div className={ed.canvasWrap} ref={hostRef} style={{ position: 'relative' }}>
      <div className={ed.readout} style={{ top: 10, bottom: 'auto', pointerEvents: 'auto', display: 'flex', gap: 14, alignItems: 'center' }}>
        <span>{status ?? 'drag a tool to move it · left-drag empty space to orbit · wheel zoom · right-drag pan · R rotates the selected tool 90° · [ ] fine-tune 5°'}</span>
        <label className={ui.checkbox} style={{ color: '#e5e7eb' }}><input type="checkbox" checked={showTools} onChange={(e) => setShowTools(e.target.checked)} /> tools</label>
      </div>
    </div>
  );
}
