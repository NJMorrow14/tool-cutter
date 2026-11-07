import * as THREE from 'three';
import type { Heightfield } from './api';

/** Bilinear sample of a height field at mat coordinates (mm). 0 outside the grid. */
export function sampleHeight(hf: Heightfield, xMm: number, yMm: number): number {
  const fx = (xMm - hf.x0_mm) / hf.step_mm - 0.5;
  const fy = (yMm - hf.y0_mm) / hf.step_y_mm - 0.5;
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  if (x0 < -1 || y0 < -1 || x0 >= hf.cols || y0 >= hf.rows) return 0;
  const tx = fx - x0, ty = fy - y0;
  const get = (i: number, j: number) => (i < 0 || j < 0 || i >= hf.rows || j >= hf.cols ? 0 : hf.heights[i * hf.cols + j]);
  return (get(y0, x0) * (1 - tx) + get(y0, x0 + 1) * tx) * (1 - ty) + (get(y0 + 1, x0) * (1 - tx) + get(y0 + 1, x0 + 1) * tx) * ty;
}

/** Height-field surface as a mesh in the "model" frame (x right, y = image y, z up). Cells are centred on the grid. */
export function heightfieldGeometry(hf: Heightfield, zScale = 1, floorZ = 0, dropBelowMm: number | null = null): THREE.BufferGeometry {
  const { cols, rows } = hf;
  let geom: THREE.BufferGeometry = new THREE.PlaneGeometry(hf.width_mm, hf.height_mm, cols - 1, rows - 1);
  const pos = geom.attributes.position as THREE.BufferAttribute;
  // PlaneGeometry: row 0 at +h/2 (top), x from -w/2. Move to x0..x0+w, y0..y0+h with row 0 at y0 (image top).
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const k = i * cols + j;
      pos.setXYZ(k, hf.x0_mm + (j + 0.5) * hf.step_mm, hf.y0_mm + (i + 0.5) * hf.step_y_mm, floorZ + hf.heights[k] * zScale);
    }
  }
  // flipping y reversed the winding: reverse the index order so the front face points up (+z)
  const idx = geom.index!;
  const arr = idx.array as Uint32Array | Uint16Array;
  for (let t = 0; t < arr.length; t += 3) { const tmp = arr[t + 1]; arr[t + 1] = arr[t + 2]; arr[t + 2] = tmp; }
  idx.needsUpdate = true;
  pos.needsUpdate = true;
  if (dropBelowMm !== null || hf.valid) {
    // a tool body: drop triangles that lie entirely on the floor (the crop rectangle's flat margin) so nothing pokes
    // out of the pocket when the tool is rotated or sits near the mat edge
    const kept: number[] = [];
    for (let t = 0; t < arr.length; t += 3) {
      const a = arr[t], b = arr[t + 1], c = arr[t + 2];
      if (hf.valid && (!hf.valid[a] || !hf.valid[b] || !hf.valid[c])) continue;
      if (dropBelowMm === null || hf.heights[a] > dropBelowMm || hf.heights[b] > dropBelowMm || hf.heights[c] > dropBelowMm) kept.push(a, b, c);
    }
    geom.setIndex(kept);
    geom = geom.toNonIndexed();
  }
  geom.computeVertexNormals();
  return geom;
}

/** Polygon (mm, y down) -> THREE.Shape in the model frame. */
export function polygonShape(poly: number[][], offset: [number, number] = [0, 0]): THREE.Shape {
  const s = new THREE.Shape();
  poly.forEach(([x, y], i) => (i === 0 ? s.moveTo(x + offset[0], y + offset[1]) : s.lineTo(x + offset[0], y + offset[1])));
  s.closePath();
  return s;
}

/** Rings (outer + holes) -> Shape with holes. */
export function ringsShape(rings: number[][][]): THREE.Shape | null {
  if (!rings.length || rings[0].length < 3) return null;
  const s = polygonShape(rings[0]);
  for (const hole of rings.slice(1)) {
    if (hole.length < 3) continue;
    const p = new THREE.Path();
    hole.forEach(([x, y], i) => (i === 0 ? p.moveTo(x, y) : p.lineTo(x, y)));
    p.closePath();
    s.holes.push(p);
  }
  return s;
}

export function disposeObject(o: THREE.Object3D) {
  o.traverse((c) => {
    const m = c as THREE.Mesh;
    if (m.geometry) m.geometry.dispose();
    const mm = m.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mm)) mm.forEach((x) => x.dispose());
    else mm?.dispose();
  });
}

export function pointInPolygon(x: number, y: number, poly: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Standard scene: renderer, camera, orbit controls, lights, a root group mapping model (x, y-down, z-up) to three. */
export function makeScene(host: HTMLElement) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  // Keep CSS dimensions independent of the high-DPI drawing buffer.
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  renderer.domElement.style.display = "block";
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x20242b);
  const camera = new THREE.PerspectiveCamera(40, 1, 1, 10000);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6b7280, 1.0));
  const dir = new THREE.DirectionalLight(0xffffff, 1.6);
  dir.position.set(-0.6, 1.4, 1.0);
  scene.add(dir);
  const root = new THREE.Group();
  // The model frame is the image frame (x right, y DOWN the drawer, z up) — a left-handed triple — so a pure rotation
  // would show the drawer mirrored. Rotate z up, then reflect so image-down runs toward the viewer (+z): the 3D view
  // then has the same orientation as the 2D photo. three.js flips face winding for negative-determinant transforms.
  // three applies scale in the group's local axes first, then the rotation: flip local y (image-down), then rotate.
  //   S: (x, y, z) -> (x, -y, z);  R(-90° about x): (x, y', z) -> (x, z, -y')  =>  (x, z, y)
  // so model z (height) -> world +y (up) and model y (image down) -> world +z (toward the camera).
  root.rotation.x = -Math.PI / 2;
  root.scale.set(1, -1, 1);
  scene.add(root);
  const resize = () => {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(host);
  return { renderer, scene, camera, root, dispose: () => { ro.disconnect(); renderer.dispose(); disposeObject(scene); if (renderer.domElement.parentElement === host) host.removeChild(renderer.domElement); } };
}

/** Frame a model-space box (mm) with the camera looking from the front-top-right. */
export function frameBox(camera: THREE.PerspectiveCamera, target: THREE.Vector3, w: number, h: number, ctrlTarget?: THREE.Vector3) {
  const dist = Math.max(w, h) * 1.05;
  camera.position.set(target.x + dist * 0.15, target.y + dist * 0.9, target.z + dist * 0.75);
  camera.near = dist / 200;
  camera.far = dist * 30;
  camera.updateProjectionMatrix();
  camera.lookAt(target);
  ctrlTarget?.copy(target);
}
