export type SourceKind = 'scan' | 'object' | 'capture' | 'shape';

export interface ModelInfo {
  available: boolean;
  loaded: boolean;
  checkpoint: string | null;
  model_type: string | null;
  device: string | null;
  error: string | null;
}

export interface SessionInfo {
  id: string;
  source_kind: SourceKind;
  filename: string;
  version: number;
  original: { width: number; height: number };
  suggested_corners: number[][] | null;
  scan: {
    mm_per_px: number;
    has_height: boolean;
    plane_inlier_fraction?: number;
    unit_scale?: number;
    coverage?: number;
    photo_coverage?: { covered_fraction: number; overlap_fraction: number; rows: number; cols: number; covered: number[][]; overlap: number[][]; photo_count: number };
    /** the server reads each marker's corner off its position, so a sheet laid out in the wrong order still works */
    markers_reordered?: boolean;
    marker_corners?: Record<string, string> | null;
  } | null;
  rectified?: { width: number; height: number; mm_per_px: number; has_height: boolean };
  mat_mm?: { width: number; height: number } | null;
  corners?: number[][];
  model?: ModelInfo;
  object?: { footprint_w_mm: number; footprint_h_mm: number; thickness_mm: number } | null;
  tools?: ToolResult[];
  auto_calibrated?: boolean;
}

export type PointLabel = 'pos' | 'neg';

export interface PromptPoint {
  x: number;
  y: number;
  label: PointLabel;
}

export interface HeightStats {
  p95_mm: number;
  median_mm: number;
  max_mm: number;
}

export interface ToolResult {
  id: string;
  session_id: string;
  name?: string;
  points: { x: number; y: number; label: number }[];
  box: number[] | null;
  polygon_px: number[][];
  polygon_mm: number[][];
  area_mm2: number;
  bbox_px?: number[];
  image_url?: string;
  image_source?: 'single_photo' | 'stitched';
  measured_thickness_mm: number | null;
  height_stats: HeightStats | null;
}

export interface Notch {
  x_mm: number;
  y_mm: number;
  diameter_mm: number;
}

/** A tool as the UI tracks it: segmentation prompts + result + layout edits. */
export interface Tool {
  id: string;
  session_id: string;
  source: SourceKind;
  name: string;
  color: string;
  points: PromptPoint[];
  box: number[] | null;
  polygon_px: number[][];
  polygon_mm: number[][];
  area_mm2: number;
  image_url?: string;
  image_source?: 'single_photo' | 'stitched';
  measured_thickness_mm: number | null;
  include: boolean;
  clearance_mm: number | null; // null = use default
  depth_mm: number | null; // null = derive from rule / through cut
  rotation_deg: number;
  offset_mm: { x: number; y: number };
  notch: Notch | null;
  pending?: boolean;
  error?: string | null;
  /** outline as detected (px), kept so a hand-edited outline can be reset */
  auto_polygon_px?: number[][];
  edited?: boolean;
  /** a drawn primitive (source 'shape'): its parameters, so the size stays editable */
  shape?: ShapeSpec;
}

export type ShapeKind = 'rect' | 'circle' | 'slot' | 'hex' | 'poly';
/** `points` (mm, relative to the shape's own top-left) only for kind 'poly'. */
export interface ShapeSpec { kind: ShapeKind; w_mm: number; h_mm: number; r_mm: number; points?: number[][] }

export interface LayoutTool {
  id: string;
  name: string;
  depth_mm: number | null;
  rings: number[][][];
  area_mm2: number;
  bbox_mm: number[] | null;
  centroid_mm: number[] | null;
  notch: Notch | null;
  outside_mat: boolean;
  overlaps: string[];
}

export interface LayoutResponse {
  mat: { width_mm: number; height_mm: number };
  mirror: boolean;
  tools: LayoutTool[];
}

export type DepthRule = 'measured' | 'measured_minus' | 'fraction' | 'through';

export interface LayoutSettings {
  default_clearance_mm: number;
  smoothing_mm: number;
  mirror: boolean;
  depth_rule: DepthRule;
  depth_minus_mm: number; // for measured_minus: pocket = thickness - X (tool sits proud)
  depth_fraction: number; // for fraction: pocket = thickness * f
  fallback_depth_mm: number; // when no measurement exists
  include_mat: boolean;
  include_labels: boolean;
  fill_mode: 'none' | 'fill';
  mat_thickness_mm: number;
  notch_diameter_mm: number;
}

export const DEFAULT_SETTINGS: LayoutSettings = {
  default_clearance_mm: 1.0,
  smoothing_mm: 1.5,
  mirror: false,
  depth_rule: 'measured_minus',
  depth_minus_mm: 3,
  depth_fraction: 0.7,
  fallback_depth_mm: 15,
  include_mat: true,
  include_labels: true,
  fill_mode: 'none',
  mat_thickness_mm: 30,
  notch_diameter_mm: 20,
};

export const TOOL_COLORS = [
  '#e11d48', '#2563eb', '#16a34a', '#d97706', '#7c3aed', '#0891b2',
  '#db2777', '#65a30d', '#ea580c', '#4f46e5', '#0d9488', '#b45309',
];

export interface MatPreset {
  label: string;
  width_mm: number;
  height_mm: number;
}

export const MAT_PRESETS: MatPreset[] = [
  { label: 'US Letter (279.4 × 215.9)', width_mm: 279.4, height_mm: 215.9 },
  { label: 'A4 (297 × 210)', width_mm: 297, height_mm: 210 },
  { label: 'A3 (420 × 297)', width_mm: 420, height_mm: 297 },
  { label: 'Tabloid (431.8 × 279.4)', width_mm: 431.8, height_mm: 279.4 },
  { label: 'Kaizen sheet 24 × 12 in', width_mm: 609.6, height_mm: 304.8 },
  { label: 'Kaizen sheet 24 × 24 in', width_mm: 609.6, height_mm: 609.6 },
];
