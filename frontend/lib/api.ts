import type { LayoutResponse, LayoutSettings, PromptPoint, SessionInfo, Tool, ToolResult } from './types';

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:8000';

async function parseError(resp: Response, fallback: string): Promise<string> {
  try {
    const data = await resp.json();
    if (data && typeof data.error === 'string') return data.error;
  } catch {
    /* ignore */
  }
  return `${fallback} (HTTP ${resp.status})`;
}

async function postJson<T>(path: string, body: unknown, fallback: string): Promise<T> {
  const resp = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(await parseError(resp, fallback));
  return (await resp.json()) as T;
}

export async function getHealth(): Promise<{ ok: boolean; model: SessionInfo['model'] }> {
  const resp = await fetch(`${API_BASE_URL}/health`);
  if (!resp.ok) throw new Error('Backend not reachable');
  return resp.json();
}

export type ScanKind = 'auto' | 'layout' | 'object';

export async function createSession(file: File, units = 'auto', scanKind: ScanKind = 'auto'): Promise<SessionInfo> {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('units', units);
  fd.append('scan_kind', scanKind);
  const resp = await fetch(`${API_BASE_URL}/api/sessions`, { method: 'POST', body: fd });
  if (!resp.ok) throw new Error(await parseError(resp, 'Upload failed'));
  return resp.json();
}

export function imageUrl(sessionId: string, stage: 'original' | 'rectified' | 'height' | 'height_original', version: number) {
  return `${API_BASE_URL}/api/sessions/${sessionId}/image/${stage}?v=${version}`;
}

export function calibrate(sessionId: string, corners: number[][], width_mm?: number, height_mm?: number, rotateQuarterTurns = 0) {
  return postJson<SessionInfo>(
    `/api/sessions/${sessionId}/calibrate`,
    { corners, width_mm, height_mm, rotate_quarter_turns: rotateQuarterTurns },
    'Calibration failed',
  );
}

export interface AutoDetectOptions {
  mode: 'auto' | 'color' | 'height';
  min_area_mm2: number;
  height_threshold_mm: number;
  refine_with_sam: boolean;
  id_prefix: string;
}

export function autoDetect(sessionId: string, opts: AutoDetectOptions) {
  return postJson<{ tools: ToolResult[]; mode: string; sam_used: boolean; sam_error: string | null }>(
    `/api/sessions/${sessionId}/auto_detect`,
    opts,
    'Auto-detect failed',
  );
}

export function segment(sessionId: string, tools: { id: string; points: PromptPoint[]; box: number[] | null }[]) {
  return postJson<{ tools: ToolResult[] }>(`/api/sessions/${sessionId}/segment`, { tools }, 'Segmentation failed');
}

export function resolveDepth(tool: Tool, s: LayoutSettings): number | null {
  if (tool.depth_mm !== null && tool.depth_mm !== undefined) return tool.depth_mm;
  if (s.depth_rule === 'through') return null;
  const t = tool.measured_thickness_mm;
  if (t === null || t === undefined) return s.fallback_depth_mm;
  if (s.depth_rule === 'measured') return round1(t);
  if (s.depth_rule === 'fraction') return round1(Math.max(1, t * s.depth_fraction));
  return round1(Math.max(1, t - s.depth_minus_mm));
}

function round1(v: number) {
  return Math.round(v * 10) / 10;
}

export function buildLayoutBody(
  tools: Tool[],
  mat: { width_mm: number; height_mm: number },
  s: LayoutSettings,
) {
  return {
    mat,
    smoothing_mm: s.smoothing_mm,
    default_clearance_mm: s.default_clearance_mm,
    mirror: s.mirror,
    tools: tools
      .filter((t) => t.polygon_mm.length >= 3)
      .map((t) => ({
        id: t.id,
        session_id: t.session_id,
        thickness_mm: t.measured_thickness_mm,
        name: t.name,
        polygon_mm: t.polygon_mm,
        include: t.include,
        clearance_mm: t.clearance_mm,
        depth_mm: resolveDepth(t, s),
        rotation_deg: t.rotation_deg,
        offset_mm: t.offset_mm,
        notch: t.notch,
      })),
  };
}

export function computeLayout(body: ReturnType<typeof buildLayoutBody>) {
  return postJson<LayoutResponse>('/api/layout', body, 'Layout failed');
}

export type ExportFormat = 'svg' | 'dxf' | 'stl' | 'stl_tools';

export async function exportLayout(
  body: ReturnType<typeof buildLayoutBody>,
  format: ExportFormat,
  exportOpts: Record<string, unknown>,
  inline = false,
): Promise<{ blob: Blob; filename: string }> {
  const resp = await fetch(`${API_BASE_URL}/api/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, format, export: exportOpts, inline }),
  });
  if (!resp.ok) throw new Error(await parseError(resp, 'Export failed'));
  const cd = resp.headers.get('Content-Disposition') || '';
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
  return { blob: await resp.blob(), filename: m ? decodeURIComponent(m[1]) : `layout.${format}` };
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
