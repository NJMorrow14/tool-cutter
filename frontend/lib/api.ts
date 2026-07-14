export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

export interface PreviewStats {
  contour_count: number;
  total_area_px2: number;
  quarter_found?: boolean;
  mm_per_px?: number | null;
  image_rotation?: number;
}

export interface SamPoint {
  x: number;
  y: number;
  label: "pos" | "neg";
}

export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface QuarterManual {
  cx: number;
  cy: number;
  MA: number;
  ma: number;
  angle: number;
}

export interface PreviewPayload {
  use_quarter: boolean;
  quarter_diameter_mm: number;
  sam_points: SamPoint[];
  crop_rect?: Rect | null;
  quarter_roi?: Rect | null;
  quarter_manual?: QuarterManual | null;
  sam_auto: boolean;
  sam_multimask: boolean;
  sam_union: boolean;
  compute_mask: boolean;
  image_rotation: number;
}

export interface PreviewResponse {
  overlay_png: string;
  mask_png: string;
  svg_data_uri?: string;
  stats: PreviewStats;
  scale_down?: number;
}

export async function uploadImage(file: File) {
  const fd = new FormData();
  fd.append("image", file);
  const resp = await fetch(`${API_BASE_URL}/api/upload_image`, {
    method: "POST",
    body: fd,
  });
  if (!resp.ok) {
    const err = await safeParseError(resp);
    throw new Error(err ?? "Upload failed");
  }
  return resp.json() as Promise<{
    ok: boolean;
    shape: [number, number];
    converted_to?: string;
    auto_rotation_deg?: number;
  }>;
}

export async function requestPreview(payload: PreviewPayload): Promise<PreviewResponse> {
  const resp = await fetch(`${API_BASE_URL}/api/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data?.error ?? "Preview failed");
  }
  if (data?.error) {
    throw new Error(data.error);
  }
  return data;
}

export async function exportSvg(payload: PreviewPayload): Promise<Blob> {
  const resp = await fetch(`${API_BASE_URL}/api/export_svg`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const err = await safeParseError(resp);
    throw new Error(err ?? "Export failed");
  }
  return resp.blob();
}

async function safeParseError(resp: Response): Promise<string | null> {
  try {
    const data = await resp.json();
    if (data && typeof data.error === "string") {
      return data.error;
    }
  } catch (err) {
    // ignore
  }
  return null;
}
