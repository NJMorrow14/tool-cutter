export type Viewport = { x: number; y: number; w: number; h: number };
export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 32;

/** Zoom around an anchor so the point beneath the cursor stays in place. */
export function zoomViewport(view: Viewport, anchor: { x: number; y: number }, factor: number, fitWidth: number): Viewport {
  const w = Math.max(fitWidth / MAX_ZOOM, Math.min(fitWidth / MIN_ZOOM, view.w / factor));
  const ratio = w / view.w;
  return { x: anchor.x - (anchor.x - view.x) * ratio, y: anchor.y - (anchor.y - view.y) * ratio, w, h: view.h * ratio };
}

export function frameViewport(points: number[][], fit: Viewport): Viewport {
  if (!points.length) return fit;
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const w = Math.max((x1 - x0) * 1.3 + 4, ((y1 - y0) * 1.3 + 4) * fit.w / fit.h, fit.w / MAX_ZOOM);
  const h = w * fit.h / fit.w;
  return { x: (x0 + x1 - w) / 2, y: (y0 + y1 - h) / 2, w, h };
}
