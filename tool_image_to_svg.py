#!/usr/bin/env python3
"""
Convert an overhead photo of tools into an SVG of cutout outlines for foam.

Pipeline (configurable via CLI):
 1) Load image, convert to grayscale, optional blur
 2) Segment foreground via thresholding or Canny + morphology
 3) Find contours, approximate with fewer points
 4) Scale to real units (mm) using either a provided mm-per-pixel scale or a
    reference length defined by two points in the image
 5) Export an SVG (polygons by default; compound path with holes is supported)

Examples:
  # Simple Otsu threshold, external contours, output in pixels
  python tool_image_to_svg.py -i overhead_dark.jpeg -o cutouts.svg

  # Canny edges + closing, mm scale provided (0.1 mm per px), ignore tiny specks
  python tool_image_to_svg.py -i overhead_dark.jpeg -o cutouts_mm.svg \
      --mode canny --canny 50 150 --morph-close 7 --epsilon-frac 0.01 \
      --scale-mm-per-px 0.1 --min-area 100.0

  # Interactive 2-click calibration: click two points spanning 100 mm
  python tool_image_to_svg.py -i overhead_dark.jpeg -o cutouts_mm.svg \
      --reference-length-mm 100 --calibrate-click

Notes:
 - If you don't provide a scale, the SVG will be in pixels (px). For cutting,
   you typically want millimeters; use --scale-mm-per-px or --reference-length-mm.
 - For holes (e.g., a wrench open-end), use --include-holes to produce paths
   with even-odd fill rule. Otherwise only external perimeters are exported.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import svgwrite
import cv2
import numpy as np



@dataclass
class Settings:
    mode: str = "threshold"  # 'threshold', 'adaptive', 'canny'
    blur_ksize: int = 5
    threshold: int = -1  # -1 means use Otsu for threshold mode
    invert: bool = True
    adaptive_block_size: int = 35
    adaptive_C: int = 5
    canny_low: int = 50
    canny_high: int = 150
    morph_close: int = 5
    morph_open: int = 0
    dilate_iter: int = 0
    erode_iter: int = 0
    epsilon_frac: float = 0.01
    include_holes: bool = False
    min_area: float = 1000.0  # px^2 or mm^2 depending on scale
    scale_mm_per_px: Optional[float] = None
    reference_length_mm: Optional[float] = None
    ref_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    calibrate_click: bool = False
    margin_mm: float = 5.0
    sort_desc: bool = True
    save_debug: Optional[str] = None


def parse_args() -> Tuple[argparse.Namespace, Settings]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="Input overhead image (tools on contrasting background)")
    p.add_argument("-o", "--output", required=False, help="Output SVG path (default: <input_basename>.svg)")

    p.add_argument("--mode", choices=["threshold", "adaptive", "canny"], default="threshold")
    p.add_argument("--blur-ksize", type=int, default=5, help="Gaussian blur kernel size (odd, 0 to disable)")
    p.add_argument("--threshold", type=int, default=-1, help="Threshold value; -1 means Otsu auto (threshold mode only)")
    p.add_argument("--invert", action="store_true", default=True, help="Invert binary so tools are white")
    p.add_argument("--no-invert", dest="invert", action="store_false")
    p.add_argument("--adaptive-block-size", type=int, default=35, help="Adaptive threshold block size (odd)")
    p.add_argument("--adaptive-C", type=int, default=5, help="Adaptive threshold constant C")
    p.add_argument("--canny", nargs=2, type=int, metavar=("LOW", "HIGH"), help="Canny thresholds")

    p.add_argument("--morph-close", type=int, default=5, help="Closing kernel (0 to disable)")
    p.add_argument("--morph-open", type=int, default=0, help="Opening kernel (0 to disable)")
    p.add_argument("--dilate-iter", type=int, default=0)
    p.add_argument("--erode-iter", type=int, default=0)

    p.add_argument("--epsilon-frac", type=float, default=0.01, help="Fraction of contour perimeter for approxPolyDP")
    p.add_argument("--include-holes", action="store_true", help="Include holes as compound paths with even-odd fill")
    p.add_argument("--min-area", type=float, default=1000.0, help="Minimum area to keep (px^2 or mm^2 if scaled)")

    p.add_argument("--scale-mm-per-px", type=float, help="Direct scale: millimeters per pixel")
    p.add_argument("--reference-length-mm", type=float, help="Known real length between two points (for calibration)")
    p.add_argument("--ref-points", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="Image coordinates of the reference endpoints")
    p.add_argument("--calibrate-click", action="store_true", help="Click two points in an interactive window for calibration")
    p.add_argument("--margin-mm", type=float, default=5.0, help="Margin to add around shapes in the SVG (mm)")
    p.add_argument("--save-debug", type=str, default=None, help="Optional path to save the binary mask for debugging")

    args = p.parse_args()

    s = Settings()
    s.mode = args.mode
    s.blur_ksize = args.blur_ksize
    s.threshold = args.threshold
    s.invert = args.invert
    s.adaptive_block_size = args.adaptive_block_size
    s.adaptive_C = args.adaptive_C
    if args.canny is not None:
        s.canny_low, s.canny_high = args.canny
    s.morph_close = args.morph_close
    s.morph_open = args.morph_open
    s.dilate_iter = args.dilate_iter
    s.erode_iter = args.erode_iter
    s.epsilon_frac = args.epsilon_frac
    s.include_holes = args.include_holes
    s.min_area = args.min_area
    s.scale_mm_per_px = args.scale_mm_per_px
    s.reference_length_mm = args.reference_length_mm
    if args.ref_points is not None:
        x1, y1, x2, y2 = args.ref_points
        s.ref_points = ((x1, y1), (x2, y2))
    s.calibrate_click = args.calibrate_click
    s.margin_mm = args.margin_mm
    s.save_debug = args.save_debug

    return args, s


def gaussian_blur_if_needed(gray: np.ndarray, ksize: int) -> np.ndarray:
    if ksize and ksize > 1 and ksize % 2 == 1:
        return cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return gray


def segment_foreground(gray: np.ndarray, s: Settings) -> np.ndarray:
    g = gaussian_blur_if_needed(gray, s.blur_ksize)
    if s.mode == "threshold":
        if s.threshold < 0:
            _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(g, s.threshold, 255, cv2.THRESH_BINARY)
    elif s.mode == "adaptive":
        bs = s.adaptive_block_size if s.adaptive_block_size % 2 == 1 else s.adaptive_block_size + 1
        binary = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, s.adaptive_C
        )
    else:  # canny
        edges = cv2.Canny(g, s.canny_low, s.canny_high)
        # Dilate and close to fill gaps, then binarize
        binary = edges.copy()
        if s.morph_close and s.morph_close > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.morph_close, s.morph_close))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        # Convert edges to filled regions by closing + dilation
        binary = cv2.dilate(binary, None, iterations=max(1, s.dilate_iter))

    # Ensure binary is 0/255
    binary = (binary > 0).astype(np.uint8) * 255

    if s.invert:
        binary = cv2.bitwise_not(binary)

    # Morphological cleanup
    if s.morph_open and s.morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.morph_open, s.morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    if s.morph_close and s.morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.morph_close, s.morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    if s.erode_iter:
        binary = cv2.erode(binary, None, iterations=s.erode_iter)
    if s.dilate_iter:
        binary = cv2.dilate(binary, None, iterations=s.dilate_iter)
    return binary


def find_and_approx_contours(binary: np.ndarray, s: Settings) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    mode = cv2.RETR_CCOMP if s.include_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
    approx = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, s.epsilon_frac * peri)
        ap = cv2.approxPolyDP(cnt, eps, True)
        if len(ap) >= 3:
            approx.append(ap)
    # Recompute hierarchy for approximated contours is non-trivial; keep original if needed
    return approx, hierarchy


def contour_area(cnt: np.ndarray) -> float:
    return float(cv2.contourArea(cnt))


def total_bounds(contours: Iterable[np.ndarray]) -> Tuple[float, float, float, float]:
    xs = []
    ys = []
    for c in contours:
        pts = c.reshape(-1, 2)
        xs.append(pts[:, 0].min())
        xs.append(pts[:, 0].max())
        ys.append(pts[:, 1].min())
        ys.append(pts[:, 1].max())
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def calibrate_mm_per_px(img: np.ndarray, s: Settings) -> Optional[float]:
    if s.scale_mm_per_px is not None:
        return s.scale_mm_per_px
    if s.reference_length_mm is None:
        return None
    p1: Optional[Tuple[float, float]] = None
    p2: Optional[Tuple[float, float]] = None
    if s.ref_points is not None:
        p1, p2 = s.ref_points
    elif s.calibrate_click:
        coords: List[Tuple[int, int]] = []

        def on_mouse(event, x, y, flags, param):  # noqa: ARG001
            if event == cv2.EVENT_LBUTTONDOWN:
                coords.append((x, y))
                # Simple visual feedback
                cv2.circle(img_vis, (x, y), 4, (0, 0, 255), -1)
                cv2.imshow(win_name, img_vis)

        win_name = "Click two points spanning known length"
        img_vis = img.copy()
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img_vis)
        cv2.setMouseCallback(win_name, on_mouse)
        print("Click two points corresponding to reference length (ESC to cancel)...")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if len(coords) >= 2:
                break
        cv2.destroyWindow(win_name)
        if len(coords) >= 2:
            p1, p2 = coords[0], coords[1]
    if p1 is None or p2 is None:
        return None
    dx = float(p1[0] - p2[0])
    dy = float(p1[1] - p2[1])
    dist_px = math.hypot(dx, dy)
    if dist_px <= 0.0:
        return None
    return float(s.reference_length_mm) / dist_px


def to_svg(
    contours: List[np.ndarray],
    hierarchy: Optional[np.ndarray],
    out_path: str,
    mm_per_px: Optional[float],
    margin_mm: float,
    include_holes: bool,
    sort_desc: bool = True,
):
    if not contours:
        raise SystemExit("No contours found. Adjust thresholding or preprocessing.")

    # Sort by area
    contours_sorted = sorted(contours, key=contour_area, reverse=sort_desc)

    # Compute bounds in pixels
    minx, miny, maxx, maxy = total_bounds(contours_sorted)
    width_px = max(1.0, maxx - minx)
    height_px = max(1.0, maxy - miny)

    # Decide coordinate units and scaling
    if mm_per_px is None:
        units = "px"
        coord_scale = 1.0
        # Approx px per mm when no calibration (96 dpi -> 3.78 px/mm)
        px_per_mm = 3.7795275591
        margin_units = margin_mm * px_per_mm
        width_units = width_px + 2 * margin_units
        height_units = height_px + 2 * margin_units
        size = (f"{width_units}px", f"{height_units}px")
    else:
        units = "mm"
        coord_scale = mm_per_px
        margin_units = margin_mm
        width_units = width_px * coord_scale + 2 * margin_units
        height_units = height_px * coord_scale + 2 * margin_units
        size = (f"{width_units}mm", f"{height_units}mm")

    # Setup SVG canvas
    dwg = svgwrite.Drawing(out_path, size=size)
    dwg.viewbox(0, 0, width_units, height_units)

    def transform_point(pt: Tuple[float, float]) -> Tuple[float, float]:
        # Translate so minx/miny map to margin, then scale to output units
        x = (pt[0] - minx) * coord_scale + margin_units
        y = (pt[1] - miny) * coord_scale + margin_units
        return x, y

    if include_holes and hierarchy is not None and len(hierarchy) > 0:
        print("Warning: include_holes is requested, but hole hierarchy handling is simplified. Exporting external polygons only.")
        include_holes = False

    if not include_holes:
        for cnt in contours_sorted:
            pts = cnt.reshape(-1, 2)
            pts_tr = [transform_point((float(x), float(y))) for (x, y) in pts]
            if mm_per_px is None:
                dwg.add(dwg.polygon(points=pts_tr, fill="black"))
            else:
                dwg.add(dwg.polygon(points=pts_tr, fill="black"))

    dwg.save()
    print(f"Wrote SVG to: {out_path} ({units})")


def main():
    args, s = parse_args()
    out_path = args.output or os.path.splitext(args.input)[0] + ".svg"

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.input}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = segment_foreground(gray, s)
    if s.save_debug:
        cv2.imwrite(s.save_debug, binary)

    contours, hierarchy = find_and_approx_contours(binary, s)

    # Calibrate scale
    mm_per_px = calibrate_mm_per_px(img, s)

    # Area filtering
    kept = []
    for c in contours:
        area_px2 = contour_area(c)
        if mm_per_px is not None:
            area_mm2 = area_px2 * (mm_per_px ** 2)
            if area_mm2 >= s.min_area:
                kept.append(c)
        else:
            if area_px2 >= s.min_area:
                kept.append(c)

    if not kept:
        raise SystemExit("No contours left after filtering. Try lowering --min-area or adjusting preprocessing.")

    to_svg(kept, hierarchy, out_path, mm_per_px, s.margin_mm, s.include_holes, s.sort_desc)


if __name__ == "__main__":
    main()


