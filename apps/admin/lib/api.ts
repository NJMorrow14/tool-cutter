import type { LayoutResponse, LayoutSettings, PromptPoint, SessionInfo, Tool, ToolResult } from './types';

/** Where the foam API lives. The configured URL is used as-is, except when it says "localhost" while this page was
 *  opened from another device (the iPhone app's review screen, a tablet on the LAN): there "localhost" is the device
 *  itself, so the API is looked up on the host that served this page instead, same port. */
function resolveApiBase(): string {
  const env = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:8000';
  if (typeof window === 'undefined') return env;
  const pageHost = window.location.hostname;
  const pageIsLocal = pageHost === 'localhost' || pageHost === '127.0.0.1' || pageHost === '::1';
  try {
    const u = new URL(env);
    const envIsLocal = u.hostname === 'localhost' || u.hostname === '127.0.0.1';
    if (envIsLocal && !pageIsLocal) return `${u.protocol}//${pageHost}:${u.port || '8000'}`;
  } catch { /* keep env */ }
  return env;
}

export const API_BASE_URL = resolveApiBase();

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

export async function getSession(sessionId: string): Promise<SessionInfo> {
  const resp = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`);
  if (!resp.ok) throw new Error(await parseError(resp, 'Session not found'));
  return resp.json();
}

/** Phone-style capture: a photo of the drawer with ArUco corner markers (optionally with a depth map). */
export async function createCapture(file: File, opts: { marker_size_mm: number; inset_mm: number }): Promise<SessionInfo> {
  const fd = new FormData();
  fd.append('image', file);
  fd.append('marker_size_mm', String(opts.marker_size_mm));
  fd.append('inset_mm', String(opts.inset_mm));
  const resp = await fetch(`${API_BASE_URL}/api/captures`, { method: 'POST', body: fd });
  if (!resp.ok) throw new Error(await parseError(resp, 'Capture failed'));
  return resp.json();
}

export const MARKER_SHEET_URL = `${API_BASE_URL}/api/marker_sheet.svg?marker_mm=25`;   // must match the app's AppSettings.markerSizeMm (25 since 2026-10-02)

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
  mode: 'auto' | 'color' | 'height' | 'hybrid';
  min_area_mm2: number;
  height_threshold_mm: number;
  refine_with_sam: boolean;
  edge_source?: 'topo' | 'photo';
  id_prefix: string;
}

export function autoDetect(sessionId: string, opts: AutoDetectOptions) {
  return postJson<{ tools: ToolResult[]; mode: string; edge_source?: string; sam_used: boolean; sam_error: string | null }>(
    `/api/sessions/${sessionId}/auto_detect`,
    opts,
    'Auto-detect failed',
  );
}

export function segment(sessionId: string, tools: { id: string; points: PromptPoint[]; box: number[] | null }[], edgeSource: 'topo' | 'photo' = 'topo') {
  return postJson<{ tools: ToolResult[] }>(`/api/sessions/${sessionId}/segment`, { tools, edge_source: edgeSource }, 'Segmentation failed');
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
        pocket_style: t.pocket_style ?? s.pocket_style,
        depth_override_mm: t.depth_mm,          // relief pockets honour only an explicit per-tool depth, not the rule
        source: t.source,                       // drawn shapes are processed EXACTLY (no smoothing / straightening)
        shape: t.shape ? { kind: t.shape.kind } : undefined,
      })),
    relief: {
      default_style: s.pocket_style, resolution_mm: s.relief_resolution_mm, smooth_mm: s.relief_smooth_mm, z_clearance_mm: s.relief_clearance_mm,
      clean_solids: s.relief_clean_solids, semantic: s.relief_semantic,
      mat_thickness_mm: s.mat_thickness_mm, cutter_mm: s.cnc_cutter_mm, stepover_mm: s.cnc_stepover_mm, stepdown_mm: s.cnc_stepdown_mm,
      feed_mm_min: s.cnc_feed_mm_min, plunge_mm_min: s.cnc_plunge_mm_min, safe_z_mm: s.cnc_safe_z_mm, spindle_rpm: s.cnc_spindle_rpm,
    },
  };
}

export type ReliefFormat = 'stl' | 'png' | 'gcode' | 'package';
export interface ReliefGrid extends Heightfield { thickness_mm: number; max_depth_mm: number; tools: { id: string; style: string; max_depth_mm: number; min_depth_mm: number; solid?: string; solid_residual_mm?: number; clean_steps?: string[] }[] }
/** The carved foam top surface for the 3D preview (z <= 0 relative to the foam top) plus per-tool depths. */
export async function getReliefGrid(body: ReturnType<typeof buildLayoutBody>, resolutionMm = 2): Promise<ReliefGrid> {
  const resp = await fetch(`${API_BASE_URL}/api/relief`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, format: 'grid', relief: { ...body.relief, resolution_mm: resolutionMm } }),
  });
  if (!resp.ok) throw new Error(await parseError(resp, 'Relief preview failed'));
  const raw = (await resp.json()) as Record<string, unknown>;
  const hf = decodeHeightfield(raw);
  // no `valid` mask: with one, heightfieldGeometry drops triangles and goes non-indexed, which gives FLAT shading —
  // every grid cell its own facet, the "quantized" look. The carved block is one continuous indexed surface.
  hf.valid = undefined;
  return { ...hf, thickness_mm: raw.thickness_mm as number, max_depth_mm: raw.max_depth_mm as number, tools: raw.tools as ReliefGrid['tools'] };
}
/** Download the form-fit block: STL (carved mesh), PNG (16-bit depth map + JSON, zipped) or G-code (GRBL raster finishing). */
export async function exportRelief(body: ReturnType<typeof buildLayoutBody>, format: ReliefFormat, exportOpts: Record<string, unknown>): Promise<{ blob: Blob; filename: string; stats?: Record<string, number> }> {
  const resp = await fetch(`${API_BASE_URL}/api/relief`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, format, export: exportOpts }),
  });
  if (!resp.ok) throw new Error(await parseError(resp, 'Relief export failed'));
  const cd = resp.headers.get('Content-Disposition') || '';
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
  const statsRaw = resp.headers.get('X-Gcode-Stats');
  return { blob: await resp.blob(), filename: m ? decodeURIComponent(m[1]) : `relief.${format}`, stats: statsRaw ? JSON.parse(statsRaw) : undefined };
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

// ------------------------------------------------------------------ 3D (scan height fields)

export interface Heightfield {
  cols: number; rows: number; step_mm: number; step_y_mm: number; x0_mm: number; y0_mm: number;
  width_mm: number; height_mm: number; max_mm: number; median_mm?: number; heights: Float32Array; valid?: Uint8Array;
}

function decodeHeightfield(raw: Record<string, unknown>): Heightfield {
  const bin = atob(raw.heights_b64 as string);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const { heights_b64: _b, ...rest } = raw as Record<string, unknown> & { heights_b64: string };
  void _b;
  const valid = typeof raw.valid_b64 === 'string' ? Uint8Array.from(atob(raw.valid_b64), c => c.charCodeAt(0)) : undefined;
  return { ...(rest as Omit<Heightfield, 'heights'>), heights: new Float32Array(bytes.buffer), valid };
}

/** The fused drawer scan as a coarse grid (mm). */
export async function getHeightfield(sessionId: string, stepMm = 2): Promise<Heightfield> {
  const resp = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/heightfield?step_mm=${stepMm}`);
  if (!resp.ok) throw new Error(await parseError(resp, 'No height data'));
  return decodeHeightfield(await resp.json());
}

/** One tool's scanned top surface, cropped, in mat-mm coordinates. */
export async function getToolHeightfield(sessionId: string, toolId: string, stepMm = 1): Promise<Heightfield> {
  const resp = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/tools/${encodeURIComponent(toolId)}/heightfield?step_mm=${stepMm}`);
  if (!resp.ok) throw new Error(await parseError(resp, 'No scan for this tool'));
  return decodeHeightfield(await resp.json());
}

// ------------------------------------------------------------------ split a merged tool / auto layout

/** Cut one detected tool into two along a line (rectified px). Returns the two new tools; the old one is gone server-side. */
export function splitTool(sessionId: string, toolId: string, line: [number[], number[]], edgeSource: 'topo' | 'photo' = 'topo') {
  return postJson<{ tools: ToolResult[]; removed: string }>(`/api/sessions/${sessionId}/split`, { tool_id: toolId, line, edge_source: edgeSource }, 'Split failed');
}

/** Combine several tools of ONE scan into a single outline. The originals are gone server-side.
 *  `bridged_mm` > 0 means the parts were not touching and a bridge of that width was drawn to join them. */
export function mergeTools(sessionId: string, toolIds: string[], bridgeMm?: number) {
  return postJson<{ tools: ToolResult[]; removed: string[]; bridged_mm: number }>(
    `/api/sessions/${sessionId}/merge`, { tool_ids: toolIds, ...(bridgeMm === undefined ? {} : { bridge_mm: bridgeMm }) }, 'Combine failed');
}

export interface Placement { id: string; rotation_deg: number; offset_mm: { x: number; y: number } }

/** Pack the included tools onto the mat (biggest first, long side horizontal or vertical, `gap` mm between pockets). */
export function autoLayout(mat: { width_mm: number; height_mm: number }, tools: Tool[], opts: { gap_mm: number; margin_mm: number; allow_rotate: boolean; direction: 'columns' | 'rows' }) {
  return postJson<{ placements: Placement[]; unplaced: string[] }>('/api/autolayout', {
    mat, ...opts,
    tools: tools.filter((t) => t.include && t.polygon_mm.length >= 3).map((t) => ({ id: t.id, polygon_mm: t.polygon_mm, include: t.include })),
  }, 'Auto layout failed');
}


// ------------------------------------------------------------------ recent uploads

export interface SessionSummary {
  id: string; created: number | null; source_kind: string; filename: string | null;
  mat_mm: { width: number; height: number } | null; frames: number | null; in_memory: boolean; saved: boolean;
}

/** Push an outline out to where the tool meets the mat (the topographic edge sits half-way up the wall). */
export async function snapToBase(sessionId: string, polygon_px: number[][], opts: { max_mm?: number; floor_mm?: number } = {}): Promise<number[][]> {
  const resp = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/snap_base`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ polygon_px, ...opts }),
  });
  if (!resp.ok) throw new Error(await parseError(resp, 'Could not snap to the base'));
  return ((await resp.json()) as { polygon_px: number[][] }).polygon_px;
}

// ------------------------------------------------------------------ knowledge-driven cleanup

export interface CleanupEdit { from_mark: number; to_mark: number; kind: string; note: string }
export interface CleanupRecognition {
  tool_name: string; description: string; shape_class: string; symmetric_axis: 'long' | 'short' | 'none';
  straight_edges: boolean; right_angles: boolean; round_shaft: boolean; trace_quality: 'good' | 'minor_issues' | 'poor';
  edits: CleanupEdit[]; confidence: number; notes: string; model: string;
}
export interface CleanupVerification { follows_edge: boolean; issues: CleanupEdit[]; better_than_orange: boolean; notes: string; second_round?: boolean }
export interface CleanupStep { type: string; [k: string]: unknown }
export interface CleanupProposal {
  id: string; name?: string | null; error?: string;
  recognition: CleanupRecognition | null; verification: CleanupVerification | null; hints: Record<string, unknown>;
  /** the proposal (green) over the photo crop, measured trace faded, numbered marks — base64 PNG */
  preview_png?: string; n_marks?: number;
  polygon_px: number[][]; polygon_mm: number[][];
  applied: CleanupStep[]; refused: CleanupStep[];
  max_move_mm: number; area_before_mm2: number; area_after_mm2: number; area_change_pct: number;
}
/** Ask for cleaned-up outlines. With credentials on the server the model inspects each tool's photo + height crops
 *  (full resolution, numbered marks around the trace), says what it is and where the trace is wrong, the geometry
 *  applies what the data supports, and the model checks the result. Proposals only — commit on Accept. */
export function cleanupOutlines(sessionId: string, tools: { id: string; polygon_px: number[][]; name?: string }[], recognize = true) {
  return postJson<{ proposals: CleanupProposal[]; recognition_available: boolean; recognition_reason: string | null }>(
    `/api/sessions/${sessionId}/cleanup`, { tools, recognize }, 'Clean up failed');
}

// ------------------------------------------------------------------ drawer agent

export interface AgentLogEntry { step: number; tool: string; input?: Record<string, unknown>; summary: string; error?: boolean; t: number }
export interface AgentToolResult extends ToolResult { name?: string; status?: string; changed?: boolean; new?: boolean; preview_png?: string; edited?: boolean }
export interface AgentResult {
  tools: AgentToolResult[]; removed: string[]; changed: string[]; summary: string; stopped: string; calls: number; seconds: number;
  usage: { input_tokens: number; output_tokens: number }; log: AgentLogEntry[];
}
export interface AgentLiveTool { id: string; name: string | null; status: string | null; polygon_mm: number[][]; area_mm2: number | null; changed: boolean; new: boolean }
export interface AgentLive { tools: AgentLiveTool[]; removed: string[]; changed: string[] }
export interface AgentJob { id: string; status: 'running' | 'done' | 'error'; error: string | null; log: AgentLogEntry[]; log_total: number; seconds: number; result?: AgentResult; live?: AgentLive;
  dismissed?: boolean; started?: number; summary?: string | null; changed?: number; removed?: number }
export interface AgentJobSummary { id: string; session: string; status: 'running' | 'done' | 'error'; error: string | null; started: number; seconds: number; dismissed: boolean; log_total: number; summary: string | null; changed: number; removed: number }
/** This drawer's agent runs, newest first — a running one is resumed, a finished undismissed one is shown again. */
export async function listAgentJobs(sessionId: string): Promise<AgentJobSummary[]> {
  const resp = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/agent`);
  if (!resp.ok) throw new Error(await parseError(resp, 'Could not list agent runs'));
  return ((await resp.json()) as { jobs: AgentJobSummary[] }).jobs;
}
export function dismissAgentJob(jobId: string) {
  return postJson<{ ok: boolean }>(`/api/agent/${jobId}/dismiss`, {}, 'Could not dismiss the agent run');
}

/** Start the drawer agent on this scan's tools: the model looks, splits/merges/re-detects/edits, checks, and returns a
 *  proposal for the whole tool list. Poll `getAgentJob` for the live log. */
export function startAgent(sessionId: string, tools: { id: string; name: string; polygon_px: number[][] }[], instructions = '') {
  return postJson<{ job_id: string }>(`/api/sessions/${sessionId}/agent`, { tools, instructions }, 'Could not start the agent');
}
export async function getAgentJob(jobId: string, since = 0): Promise<AgentJob> {
  const resp = await fetch(`${API_BASE_URL}/api/agent/${jobId}?since=${since}`);
  if (!resp.ok) throw new Error(await parseError(resp, 'Agent status failed'));
  return (await resp.json()) as AgentJob;
}

export async function listSessions(): Promise<SessionSummary[]> {
  const resp = await fetch(`${API_BASE_URL}/api/sessions`);
  if (!resp.ok) return [];                    // older backend without the endpoint
  return ((await resp.json()) as { sessions: SessionSummary[] }).sessions;
}

/** Forget a session AND delete its saved raw frames. There is no undo: for a phone scan those frames are
 *  the only copy once the drawer has been put back. */
export async function deleteSession(id: string): Promise<{ frames_removed: number }> {
  const resp = await fetch(`${API_BASE_URL}/api/sessions/${id}`, { method: 'DELETE' });
  if (!resp.ok) throw new Error(await parseError(resp, 'Could not delete that scan'));
  return (await resp.json()) as { frames_removed: number };
}
