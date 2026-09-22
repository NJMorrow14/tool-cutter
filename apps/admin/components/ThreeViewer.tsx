'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import ui from './ui.module.css';
import st from './stage.module.css';

interface Props {
  title: string;
  fetchStl: () => Promise<ArrayBuffer>;
  fetchToolsStl: () => Promise<ArrayBuffer>;
  onClose: () => void;
}

/** Orbitable preview of the foam block (with pockets) and the tools sitting in them; each can be toggled. */
export default function ThreeViewer({ title, fetchStl, fetchToolsStl, onClose }: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [status, setStatus] = useState<string | null>('Building 3D model…');
  const [showFoam, setShowFoam] = useState(true);
  const [showTools, setShowTools] = useState(true);
  const foamRef = useRef<THREE.Group | null>(null);
  const toolsRef = useRef<THREE.Group | null>(null);

  useEffect(() => { if (foamRef.current) foamRef.current.visible = showFoam; }, [showFoam]);
  useEffect(() => { if (toolsRef.current) toolsRef.current.visible = showTools; }, [showTools]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let disposed = false;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe5e7eb);
    const camera = new THREE.PerspectiveCamera(40, 1, 1, 10000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x777788, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 1.4);
    dir.position.set(1, 2, 1.5);
    scene.add(dir);

    const resize = () => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(host);

    let raf = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    // model z (up) -> three y (up); model y -> three -z
    const root = new THREE.Group();
    root.rotation.x = -Math.PI / 2;
    scene.add(root);
    const addPart = (buf: ArrayBuffer, color: number, edgeColor: number | null, roughness: number, metalness: number) => {
      const geom = new STLLoader().parse(buf);
      geom.computeVertexNormals();
      const group = new THREE.Group();
      if (geom.attributes.position && geom.attributes.position.count > 0) {
        const mat = new THREE.MeshStandardMaterial({ color, roughness, metalness, side: THREE.DoubleSide });
        group.add(new THREE.Mesh(geom, mat));
        if (edgeColor !== null) group.add(new THREE.LineSegments(new THREE.EdgesGeometry(geom, 25), new THREE.LineBasicMaterial({ color: edgeColor })));
      }
      root.add(group);
      return group;
    };

    Promise.all([fetchStl(), fetchToolsStl().catch((err) => { console.warn('tools STL failed', err); return null; })])
      .then(([foamBuf, toolsBuf]) => {
        if (disposed) return;
        foamRef.current = addPart(foamBuf, 0x3b4252, 0xd8dee9, 0.9, 0.0);
        foamRef.current.visible = showFoam;
        if (toolsBuf) {
          toolsRef.current = addPart(toolsBuf, 0xc9a26b, null, 0.45, 0.6);
          toolsRef.current.visible = showTools;
        }
        root.updateMatrixWorld(true);
        const box = new THREE.Box3().setFromObject(foamRef.current);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const grid = new THREE.GridHelper(Math.max(size.x, size.z) * 1.6, 20, 0x9ca3af, 0xcbd5e1);
        grid.position.set(center.x, box.min.y - 0.5, center.z);
        scene.add(grid);
        const dist = Math.max(size.x, size.z) * 1.3;
        camera.position.set(center.x + dist * 0.6, center.y + dist * 0.8, center.z + dist * 0.9);
        camera.near = dist / 100;
        camera.far = dist * 20;
        camera.updateProjectionMatrix();
        controls.target.copy(center);
        controls.update();
        setStatus(null);
      })
      .catch((err) => setStatus(err instanceof Error ? err.message : 'Failed to build STL'));
    // eslint-disable-next-line react-hooks/exhaustive-deps

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
        const mm = m.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mm)) mm.forEach((x) => x.dispose());
        else mm?.dispose();
      });
      host.removeChild(renderer.domElement);
    };
  }, [fetchStl, fetchToolsStl]);

  return (
    <div className={st.modal} onClick={onClose}>
      <div className={st.modalBody} onClick={(e) => e.stopPropagation()}>
        <div className={st.modalHead}>
          <strong>{title}</strong>
          <div className={ui.row}>
            {status && <span className={ui.hint}>{status}</span>}
            <label className={ui.checkbox}><input type="checkbox" checked={showFoam} onChange={(e) => setShowFoam(e.target.checked)} /> Foam</label>
            <label className={ui.checkbox}><input type="checkbox" checked={showTools} onChange={(e) => setShowTools(e.target.checked)} /> Tools</label>
            <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={onClose}>Close</button>
          </div>
        </div>
        <div ref={hostRef} className={st.viewer} />
      </div>
    </div>
  );
}
