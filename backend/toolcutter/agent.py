"""The drawer agent — one model session per drawer, with tools.

Nolan (2026-10-02): "What if we took a more agentic/overall approach?" — after the per-tool cleanup showed the model
diagnosing merged detections perfectly ("this is the hammer AND the tape measure", "two screwdriver handles") and
having no way to act on it. Here the model LOOKS (drawer, tool, height profile), ACTS (split, merge, re-detect,
edit by marks, fit a shape, rename, remove) and CHECKS (look again), until it calls `finish` or the budget runs out.

The contract is unchanged from `cleanup.py` / `recognize.py`: the model never produces a coordinate. Every tool
takes mark numbers or tool ids; the geometry measures. Everything the agent does happens on a WORKING COPY of the
tool list that is returned as a proposal — the UI applies it on Accept. Every call is logged with its inputs and a
one-line result so the run can be read back like a lab notebook.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import time
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import cleanup, recognize

log = logging.getLogger(__name__)

MODEL = os.environ.get("TC_AGENT_MODEL", recognize.MODEL)
EFFORT = os.environ.get("TC_AGENT_EFFORT", "high")
MAX_CALLS = int(os.environ.get("TC_AGENT_MAX_CALLS", "80"))
MAX_SECONDS = float(os.environ.get("TC_AGENT_MAX_S", "900"))
IMG_SIDE = 1000     # longest side of any picture handed back through a tool result

SYSTEM = (
    "You are cleaning up the tool outlines of ONE drawer scanned from above with a depth camera, so that foam pockets "
    "can be CNC-cut for the tools. The scanner has already detected tools and traced each one; some traces are wrong: "
    "two tools merged into one outline, a tool cut in two, a shadow or a neighbour included, an edge a few millimetres "
    "off, a thin part missed, a drawer wall or a paper marker detected as a tool. You have tools to LOOK (the whole "
    "drawer, one tool in photo and height map with numbered marks around its trace, a height profile between two "
    "marks), to ACT (split a tool along the line through two marks, merge tools, re-detect a tool with a different "
    "height level, edit runs of its outline by mark range, fit a simple shape, rename, remove) and to CHECK by looking "
    "again. Work through the drawer systematically: look at the whole drawer first, list the tools, then inspect each "
    "one that looks suspicious, act, and look at the result. Prefer fixing the DETECTION (split / merge / re-detect) "
    "over editing an outline that is wrong because it covers the wrong object. Be conservative with edits: a wrong "
    "edit distorts a measured outline, a missing one leaves it as measured. Marks are only valid for the version of a "
    "tool you last viewed; after you change a tool, view it again before referring to marks on it. Give every tool a "
    "short real name (e.g. 'claw hammer', 'tape measure', '10 mm socket') — name them all in ONE rename_tool call near the end. You never give sizes or positions — every "
    "measurement is made by the scanner. "
    "USE COMMON SENSE ABOUT WHAT THE TOOL IS: once you know it is a hammer, its outline must look like a hammer — a "
    "straight, mirror-symmetric handle, a head with a flat face and a claw; a steel rule is a rectangle; a socket is a "
    "circle; a screwdriver handle is a symmetric taper. Where the trace disagrees with what the real object must be "
    "(a lump on a straight handle, a ragged edge on a machined part, a shadow spur), fix it with edit_outline (shape "
    "hints + runs by marks), fit_shape, or smooth_outline. A whole LOBE or FORK that the real tool does not have (a "
    "screwdriver with a second prong, a handle with a side bulge) is a piece of a neighbour or a shadow: split it off "
    "along the valley and then merge that piece into the neighbour it belongs to, or remove it. Do not leave a tool "
    "with a shape no such tool has and merely mention it. Never cut INTO the tool: a pocket may be a little "
    "generous, never tight, and a thin real part (a blade, a tang) must stay inside. "
    "THE GOAL IS A PROFESSIONAL-LOOKING TOOLBOARD: every outline smooth, free of jitter and stray teeth, true to the "
    "object's shape, with the fewest vertices that still hold it. After the detections are right, do a POLISH PASS: "
    "smooth_outline on every tool you have not already reshaped (batch several ids per call), and symmetry / straight "
    "edge hints on the tools that obviously have them. "
    "When the drawer is in good shape, call finish with a short summary of what you changed and what you left as "
    "measured, and why."
)

EDIT_ITEM = {
    "type": "object",
    "properties": {
        "from_mark": {"type": "integer"}, "to_mark": {"type": "integer"},
        "kind": {"type": "string", "enum": recognize.EDIT_KINDS},
        "note": {"type": "string"},
    },
    "required": ["from_mark", "to_mark", "kind", "note"],
}

TOOLS: List[Dict] = [
    {"name": "list_tools", "description": "List every tool in the drawer: id, name, footprint size (mm), area, height, how it was last changed.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "view_drawer", "description": "A picture of the whole drawer (photo) with every tool's outline and id drawn on it.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "view_tool", "description": "Close-up of one tool: the photo crop and the height-map crop with its trace in orange and NUMBERED MARKS around it (marks are what split_tool, height_profile and edit_outline refer to). Everything outside the tool is dimmed.",
     "input_schema": {"type": "object", "properties": {"tool_id": {"type": "string"}}, "required": ["tool_id"]}},
    {"name": "height_profile", "description": "Heights (mm) sampled along the straight line between two marks of a tool, as numbers and a small plot. Use it to confirm a valley between two merged objects or to see how tall a thin part really is.",
     "input_schema": {"type": "object", "properties": {"tool_id": {"type": "string"}, "from_mark": {"type": "integer"}, "to_mark": {"type": "integer"}},
                      "required": ["tool_id", "from_mark", "to_mark"]}},
    {"name": "split_tool", "description": "Cut a tool in two between two of its marks. By default the cut FOLLOWS THE VALLEY: the lowest path through the height map from one mark to the other, so it bends around the objects instead of clipping them (set follow_valley false for a straight line). Each side is re-traced from the topography. Returns the new tool ids AND their pictures with fresh marks, so you can refer to those marks straight away.",
     "input_schema": {"type": "object", "properties": {"tool_id": {"type": "string"}, "from_mark": {"type": "integer"}, "to_mark": {"type": "integer"}, "follow_valley": {"type": "boolean"}},
                      "required": ["tool_id", "from_mark", "to_mark"]}},
    {"name": "merge_tools", "description": "Combine two or more tools into one outline (a tool the scanner cut in two). Returns the merged tool's picture with fresh marks.",
     "input_schema": {"type": "object", "properties": {"tool_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["tool_ids"]}},
    {"name": "redetect_tool", "description": "Re-run the detection of one tool from scratch inside its own area, with a chosen height threshold in mm (the default was 2.0; lower finds thin flat things, higher drops shadows and ramps). Replaces the tool's outline and returns its new picture with fresh marks.",
     "input_schema": {"type": "object", "properties": {"tool_id": {"type": "string"}, "height_threshold_mm": {"type": "number"}}, "required": ["tool_id", "height_threshold_mm"]}},
    {"name": "edit_outline", "description": "Correct runs of a tool's outline by mark range: straight, arc, spur (sticks out — cut off), notch (dips in — bridge), too_tight (move out to the photo edge), too_loose (move in to the photo edge). Optional shape hints (rectangle, capsule, circle, L_shape, ...; symmetric_axis long/short/none; straight_edges; right_angles) are applied as global constraints when the trace supports them. Every edit is measured and refused when the data does not bear it out.",
     "input_schema": {"type": "object", "properties": {
         "tool_id": {"type": "string"},
         "edits": {"type": "array", "items": EDIT_ITEM},
         "hints": {"type": "object", "properties": {
             "shape_class": {"type": "string", "enum": ["rectangle", "rounded_rectangle", "capsule", "circle", "L_shape", "T_shape", "irregular"]},
             "symmetric_axis": {"type": "string", "enum": ["long", "short", "none"]},
             "straight_edges": {"type": "boolean"}, "right_angles": {"type": "boolean"}}}},
      "required": ["tool_id", "edits"]}},
    {"name": "fit_shape", "description": "Replace a tool's outline with the simple shape that fits it (rectangle or circle), only if the trace really is one. For a steel rule, a box, a round can.",
     "input_schema": {"type": "object", "properties": {"tool_id": {"type": "string"}, "shape_class": {"type": "string", "enum": ["rectangle", "circle"]}}, "required": ["tool_id", "shape_class"]}},
    {"name": "smooth_outline", "description": "Make outlines smooth and clean: rounds off sensor jitter and small spurs, then re-expresses the line as straight runs, arcs and gentle curves with few vertices. The result always ENCLOSES the trace (a pocket may be slightly generous, never tight). strength_mm is the smoothing radius: 1.5 for small detailed tools, 2-3 for handles and bodies, 4 for big smooth things. Several tool ids per call.",
     "input_schema": {"type": "object", "properties": {"tool_ids": {"type": "array", "items": {"type": "string"}}, "strength_mm": {"type": "number"}}, "required": ["tool_ids"]}},
    {"name": "rename_tool", "description": "Give tools short real names. Pass MANY at once in `renames` (one call for the whole drawer); `tool_id` + `name` still works for a single tool.",
     "input_schema": {"type": "object", "properties": {"tool_id": {"type": "string"}, "name": {"type": "string"},
                                                       "renames": {"type": "array", "items": {"type": "object", "properties": {"tool_id": {"type": "string"}, "name": {"type": "string"}}, "required": ["tool_id", "name"]}}}}},
    {"name": "remove_tool", "description": "Remove a detection that is not a tool at all (drawer wall, paper marker, shadow, debris).",
     "input_schema": {"type": "object", "properties": {"tool_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["tool_id", "reason"]}},
    {"name": "undo_tool", "description": "Restore one tool to the way it was before your last change to it.",
     "input_schema": {"type": "object", "properties": {"tool_id": {"type": "string"}}, "required": ["tool_id"]}},
    {"name": "finish", "description": "Stop. Summarise what you changed, what you left as measured, and anything the person should check by hand.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]


def _img_block(img: np.ndarray, side: int = IMG_SIDE) -> Dict:
    h, w = img.shape[:2]
    sc = min(1.0, side / max(h, w))
    if sc < 1.0:
        img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(buf.tobytes()).decode("ascii")}}


def _png_b64(img: np.ndarray, side: int = 900) -> str:
    return _img_block(img, side)["source"]["data"]


def valley_path(height: np.ndarray, allowed: np.ndarray, a_px: np.ndarray, b_px: np.ndarray, mpp: float) -> Optional[np.ndarray]:
    """The lowest path through the height map from a to b, kept inside `allowed` (the tool's mask, dilated), as an
    (N, 2) array of px points — the natural cut between two objects that happen to touch. Dijkstra on an 8-connected
    grid with cost = height (+ a little per step, so the path does not wander for free); the grid is worked at 2 mm
    cells for speed and the path up-sampled. None when a and b are not connected inside `allowed`."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    ys, xs = np.nonzero(allowed)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    step = max(1, int(round(2.0 / mpp)))
    h = np.nan_to_num(height[y0:y1:step, x0:x1:step]).astype(np.float64)
    ok = allowed[y0:y1:step, x0:x1:step]
    H, W = h.shape
    if H < 2 or W < 2:
        return None
    cost = np.clip(h, 0, None) + 0.05 * (step * mpp)      # mm of height per cell, plus a small distance term
    cost[~ok] = np.inf
    idx = np.arange(H * W).reshape(H, W)
    rows, cols, vals = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        src = idx[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        dst = idx[max(0, dy):H - max(0, -dy) if dy else H, max(0, dx):W - max(0, -dx) if dx else W]
        c = 0.5 * (cost.ravel()[src.ravel()] + cost.ravel()[dst.ravel()]) * (math.sqrt(2) if dx and dy else 1.0)
        keep = np.isfinite(c)
        rows.append(src.ravel()[keep]); cols.append(dst.ravel()[keep]); vals.append(c[keep])
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    g = coo_matrix((np.concatenate([vals, vals]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(H * W, H * W)).tocsr()

    def node(p):
        gy = int(np.clip(round((p[1] - y0) / step), 0, H - 1)); gx = int(np.clip(round((p[0] - x0) / step), 0, W - 1))
        if not ok[gy, gx]:   # the marks sit ON the outline; snap to the nearest allowed cell
            oy, ox = np.nonzero(ok); j = int(np.argmin((oy - gy) ** 2 + (ox - gx) ** 2)); gy, gx = oy[j], ox[j]
        return gy * W + gx
    sa, sb = node(a_px), node(b_px)
    dist, pred = dijkstra(g, directed=False, indices=sa, return_predecessors=True)
    if not np.isfinite(dist[sb]):
        return None
    path = [sb]
    while path[-1] != sa and pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    path = np.array(path[::-1])
    pts = np.column_stack([(path % W) * step + x0, (path // W) * step + y0]).astype(np.float64)
    # start and end exactly at the marks (the grid snapped them inward)
    return np.vstack([a_px[None, :], pts, b_px[None, :]])


class DrawerTools:
    """The geometry side of the agent: a working copy of the drawer's tools plus the operations the model may call.

    `helpers` is supplied by app.py (so this module does not import the Flask app): split_core(s, tid, line_px,
    prefix) -> [tool results]; merge_core(s, ids, bridge_mm, nid) -> tool result; segment_topo(s, body) -> [tool
    results]; depth_cell_mm(s) -> float. A tool result is the dict the detect/segment endpoints return (polygon_px,
    polygon_mm, area_mm2, measured_thickness_mm, ...); `s` is the server session (rectified, rect_height,
    mm_per_px, masks).
    """

    def __init__(self, s, tools: List[Dict], helpers: Dict[str, Callable]):
        self.s = s
        self.h = helpers
        self.mpp = float(s.mm_per_px)
        self.tools: Dict[str, Dict] = {}
        self.order: List[str] = []
        for t in tools:
            if len(t.get("polygon_px") or []) >= 3:
                tt = dict(t); tt.setdefault("name", t.get("id")); tt["status"] = "as detected"
                px = np.asarray(tt["polygon_px"], dtype=np.float64)
                if len(tt.get("polygon_mm") or []) != len(px):        # the UI sends polygon_px only
                    tt["polygon_mm"] = (px * self.mpp).tolist()
                if not tt.get("area_mm2"):
                    tt["area_mm2"] = float(abs(cv2.contourArea(px.astype(np.float32))) * self.mpp * self.mpp)
                tt.setdefault("session_id", getattr(s, "id", "")); tt.setdefault("points", []); tt.setdefault("box", None)
                tt.setdefault("measured_thickness_mm", None); tt.setdefault("height_stats", None)
                self.tools[str(t["id"])] = tt; self.order.append(str(t["id"]))
        self.original_ids = set(self.order)
        self.history: Dict[str, List[Dict]] = {}
        self.views: Dict[str, Dict] = {}          # tool id -> {ring, marks, bbox} of the version last viewed
        self.changed: set = set()
        self.removed: List[str] = []
        self.seq = 0
        self.previews: Dict[str, str] = {}        # tool id -> last before/after preview (base64 png)
        self.derived: Dict[str, List[str]] = {}   # split parent -> the children it produced (undo must drop them)
        self.merged_from: Dict[str, List[str]] = {}   # merged tool -> the originals it replaced (undo restores them)
        self.budget_note = ""                     # appended to every tool result by the loop: calls/time left
        self.lock = threading.Lock()              # held around every tool call; `live()` snapshots under it

    # ---------------------------------------------------------------- helpers
    def _tool(self, tid: str) -> Dict:
        t = self.tools.get(str(tid))
        if t is None:
            raise ValueError(f"no tool '{tid}' (use list_tools for the current ids)")
        return t

    def _poly_px(self, t: Dict) -> np.ndarray:
        return np.asarray(t["polygon_px"], dtype=np.float64)

    def _new_id(self, prefix: str) -> str:
        """Ids unique within the run: a merge and a split in the SAME SECOND both produced 'a58388_1' on the first
        real run, the merged tool was overwritten and result() crashed after 11 minutes of good work."""
        self.seq += 1
        return f"{prefix}{self.seq}x{int(time.time()) % 100000}"

    @staticmethod
    def _base_name(name: str) -> str:
        """'Tool 1 (1) (2)' -> 'Tool 1', so split children do not accumulate parentheses."""
        import re
        return re.sub(r"(\s\(\d+\))+$", "", name or "").strip() or name

    def _remember(self, tid: str):
        t = self.tools.get(tid)
        if t is not None:
            self.history.setdefault(tid, []).append(json.loads(json.dumps(t)))

    def _replace(self, tid: str, result: Dict, status: str, keep_name: bool = True):
        old = self.tools.get(tid, {})
        new = dict(result); new["id"] = tid
        if keep_name and old.get("name"):
            new["name"] = old["name"]
        new.setdefault("name", tid)
        new["status"] = status
        self.tools[tid] = new
        self.changed.add(tid)
        self.views.pop(tid, None)

    def _add(self, result: Dict, name: str, status: str) -> str:
        tid = str(result["id"])
        if tid in self.tools:                       # belt and braces: never overwrite a live tool
            result = dict(result); tid = result["id"] = self._new_id("x")
        new = dict(result); new["name"] = name; new["status"] = status
        self.tools[tid] = new; self.order.append(tid); self.changed.add(tid)
        return tid

    def _drop(self, tid: str):
        self.tools.pop(tid, None)
        self.views.pop(tid, None)
        if tid in self.order:
            self.order.remove(tid)
        if tid in self.original_ids:
            self.removed.append(tid)
        self.changed.discard(tid)

    def _size(self, t: Dict) -> Tuple[float, float]:
        (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(t["polygon_mm"], np.float32))
        return (max(w, h), min(w, h))

    def _height(self, t: Dict) -> Optional[float]:
        if self.s.rect_height is None:
            return None
        px = self._poly_px(t)
        m = np.zeros(self.s.rect_height.shape, np.uint8)
        cv2.fillPoly(m, [np.round(px).astype(np.int32)], 1)
        v = np.nan_to_num(self.s.rect_height[m.astype(bool)])
        return float(np.percentile(v, 90)) if v.size else None

    def _view(self, tid: str) -> Dict:
        """The ring + marks the model last saw for this tool (compute if missing)."""
        if tid not in self.views:
            t = self._tool(tid)
            mm = self._poly_px(t) * self.mpp
            ring = cleanup.prepare_ring(mm)
            marks = cleanup.mark_indices(ring)
            self.views[tid] = {"ring": ring, "marks": marks}
        return self.views[tid]

    def _mark_px(self, tid: str, k: int) -> np.ndarray:
        v = self._view(tid)
        if not (0 <= int(k) < len(v["marks"])):
            raise ValueError(f"mark {k} does not exist on {tid} (it has marks 0..{len(v['marks']) - 1})")
        return v["ring"][v["marks"][int(k)]] / self.mpp

    def _crops(self, tid: str):
        t = self._tool(tid)
        px = self._poly_px(t)
        margin = int(round(10.0 / self.mpp))
        photo, hcrop, bbox = recognize.make_crops(self.s.rectified, self.s.rect_height, px, margin)
        return px, photo, hcrop, bbox

    def _preview(self, tid: str, before_px: np.ndarray, after_px: np.ndarray) -> Optional[np.ndarray]:
        t = self.tools.get(tid)
        if t is None:
            return None
        allpx = np.vstack([before_px, after_px])
        margin = int(round(10.0 / self.mpp))
        _, _, (x0, y0, x1, y1) = recognize.make_crops(self.s.rectified, None, allpx, margin)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        raw = self.s.rectified[y0:y1, x0:x1]
        v = self.views.get(tid)
        ring_local = (v["ring"] / self.mpp - [x0, y0]) if v else (before_px - [x0, y0])
        marks = v["marks"] if v else []
        return recognize.render_preview(raw, ring_local if v else before_px - [x0, y0], after_px - [x0, y0], marks, self.mpp)

    # ---------------------------------------------------------------- tools the model calls
    def list_tools(self) -> Tuple[str, List[np.ndarray]]:
        lines = ["id | name | long x short mm | area cm2 | height mm | status"]
        for tid in self.order:
            t = self.tools[tid]
            L, S = self._size(t); hgt = self._height(t)
            lines.append(f"{tid} | {t.get('name')} | {L:.0f} x {S:.0f} | {t.get('area_mm2', 0) / 100:.1f} | "
                         f"{'-' if hgt is None else f'{hgt:.0f}'} | {t.get('status')}")
        return "\n".join(lines), []

    def view_drawer(self) -> Tuple[str, List[np.ndarray]]:
        img = self.s.rectified.copy()
        H, W = img.shape[:2]
        sc = 1.0
        for tid in self.order:
            t = self.tools[tid]
            px = np.round(self._poly_px(t)).astype(np.int32)
            cv2.polylines(img, [px.reshape(-1, 1, 2)], True, recognize.ORANGE if tid not in self.changed else recognize.GREEN, 2, cv2.LINE_AA)
            c = px.mean(axis=0).astype(int)
            label = tid.split("_")[-1] if "_" in tid else tid
            cv2.putText(img, label, (int(c[0]) - 8, int(c[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, label, (int(c[0]) - 8, int(c[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        txt = (f"Drawer {W * self.mpp:.0f} x {H * self.mpp:.0f} mm, {len(self.order)} tools. Each outline is labelled with the "
               f"LAST PART of its id (after the underscore); orange = as detected, green = changed by you. Full ids: "
               + ", ".join(f"{tid.split('_')[-1]}={tid}" for tid in self.order))
        return txt, [img]

    def view_tool(self, tool_id: str) -> Tuple[str, List[np.ndarray]]:
        t = self._tool(tool_id)
        px, photo, hcrop, (x0, y0, x1, y1) = self._crops(tool_id)
        if photo is None:
            return f"{tool_id} is too small to crop", []
        self.views.pop(tool_id, None)
        v = self._view(tool_id)
        ring_local = v["ring"] / self.mpp - [x0, y0]
        pm, hm = recognize.annotate(photo, hcrop, ring_local, v["marks"], self.mpp)
        L, S = self._size(t); hgt = self._height(t)
        txt = (f"{tool_id} '{t.get('name')}': footprint {L:.0f} x {S:.0f} mm, height {'-' if hgt is None else f'{hgt:.0f} mm'}, "
               f"{len(v['marks'])} marks (0..{len(v['marks']) - 1}) around the trace, 1 px = {self.mpp:.2f} mm. "
               "First picture: photo. Second: height map (dark = floor, brighter = taller).")
        return txt, [pm] + ([hm] if hm is not None else [])

    def height_profile(self, tool_id: str, from_mark: int, to_mark: int) -> Tuple[str, List[np.ndarray]]:
        if self.s.rect_height is None:
            return "this capture has no height map", []
        a = self._mark_px(tool_id, from_mark); b = self._mark_px(tool_id, to_mark)
        n = max(20, int(np.linalg.norm(b - a)))
        xs = np.linspace(a[0], b[0], n); ys = np.linspace(a[1], b[1], n)
        h = cv2.remap(np.nan_to_num(self.s.rect_height).astype(np.float32), xs.astype(np.float32).reshape(1, -1),
                      ys.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        d = np.linspace(0, np.linalg.norm(b - a) * self.mpp, n)
        floor = h < 1.0
        runs = []
        i = 0
        while i < n:
            if floor[i]:
                j = i
                while j < n and floor[j]:
                    j += 1
                runs.append((d[i], d[j - 1]))
                i = j
            else:
                i += 1
        # plot
        W, Hh = 600, 220
        img = np.full((Hh, W, 3), 30, np.uint8)
        top = max(2.0, float(h.max()))
        pts = np.column_stack([20 + (W - 40) * d / max(d[-1], 1e-6), Hh - 30 - (Hh - 60) * np.clip(h, 0, top) / top]).astype(np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (85, 197, 34), 2, cv2.LINE_AA)
        cv2.line(img, (20, Hh - 30), (W - 20, Hh - 30), (200, 200, 200), 1)
        cv2.putText(img, f"mark {from_mark} -> mark {to_mark}: {d[-1]:.0f} mm, 0..{top:.0f} mm tall", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        step = max(1, n // 24)
        samples = ", ".join(f"{d[i]:.0f}mm:{h[i]:.1f}" for i in range(0, n, step))
        txt = (f"Heights along the line from mark {from_mark} to mark {to_mark} ({d[-1]:.0f} mm long): min {h.min():.1f}, "
               f"max {h.max():.1f} mm. Samples (distance:height) {samples}. "
               + (f"Drops to the floor (< 1 mm) at: " + "; ".join(f"{p:.0f}-{q:.0f} mm" for p, q in runs) if runs else "Never reaches the floor along this line."))
        return txt, [img]

    def _ensure_mask(self, tool_id: str):
        if self.s.masks.get(tool_id) is None:
            # a tool without a stored mask (hand-edited, from another capture): rasterise its polygon
            m = np.zeros(self.s.rectified.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(self._poly_px(self._tool(tool_id))).astype(np.int32)], 1)
            self.s.masks[tool_id] = m.astype(bool)

    def _line_max_height(self, a: np.ndarray, b: np.ndarray) -> float:
        n = max(8, int(np.linalg.norm(b - a)))
        xs = np.linspace(a[0], b[0], n).astype(np.float32); ys = np.linspace(a[1], b[1], n).astype(np.float32)
        h = cv2.remap(np.nan_to_num(self.s.rect_height).astype(np.float32), xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR).ravel()
        return float(h[2:-2].max()) if len(h) > 4 else float(h.max())

    def _views_of(self, ids: List[str]) -> Tuple[str, List[np.ndarray]]:
        """Fresh annotated pictures (photo + height, with marks) of several tools — returned straight from a split,
        merge or re-detect so the model does not spend a call per child asking for them."""
        txts, imgs = [], []
        for tid in ids:
            tx, im = self.view_tool(tid)
            txts.append(tx); imgs += im
        return " ".join(txts), imgs

    def split_tool(self, tool_id: str, from_mark: int, to_mark: int, follow_valley: bool = True) -> Tuple[str, List[np.ndarray]]:
        t = self._tool(tool_id)
        a = self._mark_px(tool_id, from_mark); b = self._mark_px(tool_id, to_mark)
        if np.linalg.norm(b - a) < 2:
            raise ValueError("those two marks are at the same place")
        v = self._view(tool_id); n_marks = len(v["marks"])
        apart = min((int(to_mark) - int(from_mark)) % n_marks, (int(from_mark) - int(to_mark)) % n_marks)
        if apart <= max(1, n_marks // 8):
            # Sonnet spent 16 split calls on the drill, half of them "does not divide the tool": it kept picking two
            # marks a few steps apart on the SAME edge, so the cut ran along the outline instead of across the tool.
            raise ValueError(f"marks {from_mark} and {to_mark} are only {apart} marks apart along the SAME edge; a cut must enter on one "
                             f"side of the tool and leave on the other — pick the mark where the valley meets the outline on one side "
                             f"and the mark where it meets it on the opposite side (roughly {n_marks // 2} marks away)")
        self._ensure_mask(tool_id)
        how = "straight line"
        path = None
        if follow_valley and self.s.rect_height is not None and self.h.get("split_core_path"):
            k = max(3, int(round(4.0 / self.mpp)) | 1)
            allowed = cv2.dilate(self.s.masks[tool_id].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
            path = valley_path(self.s.rect_height, allowed, a, b, self.mpp)
            if path is not None:
                # only worth it when the valley really is lower than the straight cut
                hp = cv2.remap(np.nan_to_num(self.s.rect_height).astype(np.float32), path[:, 0].astype(np.float32).reshape(1, -1),
                               path[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
                h_path = float(np.percentile(hp[2:-2], 95)) if len(hp) > 4 else float(hp.max())
                h_line = self._line_max_height(a, b)
                if h_path <= h_line - 0.5:
                    how = f"valley (cut runs through ground <= {h_path:.1f} mm; a straight line would have crossed {h_line:.1f} mm)"
                else:
                    path = None
                    how = f"straight line (the valley path was no lower: {h_path:.1f} vs {h_line:.1f} mm)"
        if path is not None and self.h.get("split_core_path"):
            parts = self.h["split_core_path"](self.s, tool_id, path, self._new_id("s") + "_")
        else:
            parts = self.h["split_core"](self.s, tool_id, [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]], self._new_id("s") + "_")
        self._remember(tool_id)
        name = self._base_name(t.get("name") or tool_id)
        self._drop(tool_id)
        ids = []
        for i, r in enumerate(parts):
            ids.append(self._add(r, f"{name} ({i + 1})", f"split from {tool_id}"))
        self.derived[tool_id] = ids
        sizes = "; ".join(f"{tid}: {self._size(self.tools[tid])[0]:.0f} x {self._size(self.tools[tid])[1]:.0f} mm" for tid in ids)
        vtxt, vimgs = self._views_of(ids)
        return f"Split {tool_id} along a {how} into {len(ids)} tools: {sizes}. Their pictures follow, marks valid. {vtxt}", vimgs

    def merge_tools(self, tool_ids: List[str]) -> Tuple[str, List[np.ndarray]]:
        ids = [str(i) for i in tool_ids]
        for tid in ids:
            self._ensure_mask(tid)
        nid = self._new_id("m")
        res, bridged = self.h["merge_core"](self.s, ids, 2.0, nid)
        names = [self.tools[i].get("name") or i for i in ids]
        for tid in ids:
            self._remember(tid); self._drop(tid)
        new = self._add(res, names[0], f"merged from {', '.join(ids)}")
        self.merged_from[new] = ids
        L, S = self._size(self.tools[new])
        vtxt, vimgs = self._views_of([new])
        return f"Merged {', '.join(ids)} into {new} ({L:.0f} x {S:.0f} mm{f', bridged a {bridged:.1f} mm gap' if bridged > 0 else ''}). Its picture follows, marks valid. {vtxt}", vimgs

    def redetect_tool(self, tool_id: str, height_threshold_mm: float) -> Tuple[str, List[np.ndarray]]:
        t = self._tool(tool_id)
        px = self._poly_px(t)
        m = int(round(6.0 / self.mpp))
        box = [float(px[:, 0].min() - m), float(px[:, 1].min() - m), float(px[:, 0].max() + m), float(px[:, 1].max() + m)]
        mask = np.zeros(self.s.rectified.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        py, pxx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        res = self.h["segment_topo"](self.s, {"height_threshold_mm": float(height_threshold_mm),
                                              "tools": [{"id": tool_id, "points": [{"x": float(pxx), "y": float(py), "label": 1}], "box": box}]})
        r = res[0] if res else None
        if not r or len(r.get("polygon_px") or []) < 3:
            return f"Re-detection at {height_threshold_mm} mm found nothing inside {tool_id}'s area; the tool is unchanged.", []
        self._remember(tool_id)
        before = self._poly_px(t)
        self._replace(tool_id, r, f"re-detected at {height_threshold_mm} mm")
        after = self._poly_px(self.tools[tool_id])
        pv = self._preview(tool_id, before, after)
        if pv is not None:
            self.previews[tool_id] = _png_b64(pv)
        L, S = self._size(self.tools[tool_id])
        vtxt, vimgs = self._views_of([tool_id])
        return (f"Re-detected {tool_id} at {height_threshold_mm} mm: now {L:.0f} x {S:.0f} mm, area {t.get('area_mm2', 0) / 100:.1f} -> "
                f"{self.tools[tool_id].get('area_mm2', 0) / 100:.1f} cm2. First picture: green = new outline, faded orange = before. "
                f"Then its new pictures with fresh marks. {vtxt}"), ([pv] if pv is not None else []) + vimgs

    def edit_outline(self, tool_id: str, edits: List[Dict], hints: Optional[Dict] = None) -> Tuple[str, List[np.ndarray]]:
        t = self._tool(tool_id)
        v = self._view(tool_id)
        edits = recognize._clean_edits(edits, len(v["marks"]))
        if not edits and not hints:
            return "no valid edits (marks out of range or unknown kinds)", []
        _, photo, _, (x0, y0, x1, y1) = self._crops(tool_id)
        raw = self.s.rectified[y0:y1, x0:x1] if photo is not None else None
        mm = self._poly_px(t) * self.mpp
        prop = cleanup.propose(mm, hints or {}, ring=v["ring"], marks=v["marks"], local=edits, photo=raw,
                               origin_mm=np.array([x0, y0], dtype=np.float64) * self.mpp, mpp=self.mpp)
        applied = [a for a in prop["applied"] if not a.get("advice")]
        if not applied:
            return "Nothing applied. Refused: " + "; ".join(f"{r['type']}: {r.get('refused')}" for r in prop["refused"]), []
        self._remember(tool_id)
        before = self._poly_px(t)
        out_mm = np.asarray(prop["polygon_mm"], dtype=np.float64)
        new = dict(t); new["polygon_mm"] = out_mm.tolist(); new["polygon_px"] = (out_mm / self.mpp).tolist()
        new["area_mm2"] = float(prop["area_after_mm2"]); new["edited"] = True
        self.tools[tool_id] = new; self.changed.add(tool_id)
        new["status"] = "outline edited"
        pv = self._preview(tool_id, before, out_mm / self.mpp)
        self.views.pop(tool_id, None)
        if pv is not None:
            self.previews[tool_id] = _png_b64(pv)
        txt = ("Applied: " + "; ".join((f"{a['type']} marks {a.get('from_mark')}-{a.get('to_mark')}" if a.get('from_mark') is not None else f"{a['type']} (whole outline)")
                                        + (f" moved {a['moved_mm']} mm" if 'moved_mm' in a else "") for a in applied)
               + (". Refused: " + "; ".join(f"{r['type']}: {r.get('refused')}" for r in prop["refused"]) if prop["refused"] else "")
               + f". Largest move {prop['max_move_mm']} mm, area {prop['area_change_pct']:+.1f} %. Green = new, faded orange = before. Marks are now stale; view the tool again before more edits.")
        return txt, [pv] if pv is not None else []

    def fit_shape(self, tool_id: str, shape_class: str) -> Tuple[str, List[np.ndarray]]:
        t = self._tool(tool_id)
        mm = self._poly_px(t) * self.mpp
        prop = cleanup.propose(mm, {"shape_class": shape_class, "straight_edges": True, "right_angles": shape_class == "rectangle"})
        if not any(a["type"] == shape_class for a in prop["applied"]):
            return f"The trace is not a {shape_class}: " + "; ".join(f"{r['type']}: {r.get('refused')}" for r in prop["refused"]), []
        self._remember(tool_id)
        before = self._poly_px(t)
        out_mm = np.asarray(prop["polygon_mm"], dtype=np.float64)
        new = dict(t); new["polygon_mm"] = out_mm.tolist(); new["polygon_px"] = (out_mm / self.mpp).tolist()
        new["area_mm2"] = float(prop["area_after_mm2"]); new["edited"] = True; new["status"] = f"fitted {shape_class}"
        self.tools[tool_id] = new; self.changed.add(tool_id); self.views.pop(tool_id, None)
        pv = self._preview(tool_id, before, out_mm / self.mpp)
        if pv is not None:
            self.previews[tool_id] = _png_b64(pv)
        a = next(a for a in prop["applied"] if a["type"] == shape_class)
        return f"Fitted a {shape_class}: {a}. Area {prop['area_change_pct']:+.1f} %.", [pv] if pv is not None else []

    def smooth_outline(self, tool_ids: List[str], strength_mm: float = 2.0) -> Tuple[str, List[np.ndarray]]:
        strength = float(min(6.0, max(0.5, strength_mm or 2.0)))
        lines, imgs = [], []
        for tid in [str(i) for i in tool_ids]:
            t = self._tool(tid)
            mm = self._poly_px(t) * self.mpp
            res = cleanup.smooth_enclosing(mm, sigma_mm=strength)
            out_mm = np.asarray(res["polygon_mm"], dtype=np.float64)
            if len(out_mm) < 4:
                lines.append(f"{tid}: could not smooth"); continue
            self._remember(tid)
            before = self._poly_px(t)
            new = dict(t); new["polygon_mm"] = out_mm.tolist(); new["polygon_px"] = (out_mm / self.mpp).tolist()
            new["area_mm2"] = float(abs(cv2.contourArea(out_mm.astype(np.float32)))); new["edited"] = True
            new["status"] = f"smoothed {strength:g} mm" if t.get("status") in ("as detected", "renamed") else f"{t.get('status')}, smoothed {strength:g} mm"
            self.tools[tid] = new; self.changed.add(tid); self.views.pop(tid, None)
            pv = self._preview(tid, before, out_mm / self.mpp)
            if pv is not None:
                self.previews[tid] = _png_b64(pv); imgs.append(pv)
            lines.append(f"{tid} '{t.get('name')}': {res['vertices_before']} -> {res['vertices_after']} vertices, area {res['area_change_pct']:+.1f} %, "
                         f"deepest point still inside the trace {res['max_inset_mm']} mm")
        return "Smoothed (green = new, faded orange = before): " + "; ".join(lines) + ". Marks on these tools are now stale.", imgs

    def rename_tool(self, tool_id: Optional[str] = None, name: Optional[str] = None, renames: Optional[List[Dict]] = None) -> Tuple[str, List[np.ndarray]]:
        """One call names the whole drawer: the fourth real run spent 27 of its 80 calls on single renames."""
        items = list(renames or [])
        if tool_id and name:
            items.append({"tool_id": tool_id, "name": name})
        done, bad = [], []
        for it in items:
            try:
                t = self._tool(str(it.get("tool_id")))
            except ValueError as exc:
                bad.append(str(exc)); continue
            t["name"] = str(it.get("name") or t.get("name"))[:60]; self.changed.add(t["id"])
            if t.get("status") == "as detected":
                t["status"] = "renamed"        # so the UI can list name-only changes compactly, apart from geometry changes
            done.append(f"{t['id']} -> '{t['name']}'")
        return ("Named: " + "; ".join(done) if done else "nothing renamed") + (". Unknown: " + "; ".join(bad) if bad else ""), []

    def remove_tool(self, tool_id: str, reason: str) -> Tuple[str, List[np.ndarray]]:
        self._tool(tool_id); self._remember(tool_id); self._drop(tool_id)
        return f"Removed {tool_id} ({reason})", []

    def undo_tool(self, tool_id: str) -> Tuple[str, List[np.ndarray]]:
        """Undo the last change to a tool. Undoing a SPLIT parent also removes the children it produced, and undoing a
        MERGED tool removes it and restores the originals — the second real run restored a split parent and kept
        both children, so the drill was covered twice."""
        if tool_id in self.merged_from and tool_id in self.tools:
            originals = self.merged_from.pop(tool_id)
            self._drop(tool_id)
            back = []
            for oid in originals:
                hist = self.history.get(oid)
                if hist:
                    self.tools[oid] = hist.pop(); self.order.append(oid); self.changed.add(oid); back.append(oid)
                    if oid in self.removed:
                        self.removed.remove(oid)
            return f"Un-merged {tool_id}; restored {', '.join(back)}", []
        hist = self.history.get(tool_id)
        if not hist:
            return f"nothing to undo on {tool_id}", []
        prev = hist.pop()
        self.previews.pop(tool_id, None)       # the before/after picture belonged to the change being undone
        dropped = []
        for cid in self.derived.pop(tool_id, []):
            if cid in self.tools:
                self._drop(cid); dropped.append(cid)
        self.tools[tool_id] = prev
        if tool_id not in self.order:
            self.order.append(tool_id)
        if tool_id in self.removed:
            self.removed.remove(tool_id)
        self.views.pop(tool_id, None); self.changed.add(tool_id)
        return f"{tool_id} restored to: {prev.get('status')}" + (f"; removed its split children {', '.join(dropped)}" if dropped else ""), []

    def suspects(self) -> str:
        """Cheap geometry-side hints for the intro, so the model starts where the trouble is instead of touring all 24
        tools: a watershed on each tool's own height map that yields >= 2 plateaus separated by a saddle means two
        objects (`geometry.split_at_saddles`); very small blobs are usually debris or a marker corner."""
        if self.s.rect_height is None:
            return ""
        merged, tiny = [], []
        h = np.nan_to_num(self.s.rect_height)
        min_px = int(300.0 / self.mpp ** 2)
        for tid in self.order:
            t = self.tools[tid]
            m = self.s.masks.get(tid)
            if m is None:
                m8 = np.zeros(h.shape, np.uint8); cv2.fillPoly(m8, [np.round(self._poly_px(t)).astype(np.int32)], 1); m = m8.astype(bool)
            vals = h[m]
            if vals.size < 10:
                continue
            top = float(np.percentile(vals, 90))
            if top < 3.0:
                continue
            # at half the tool's own height a single object is one plateau; two touching objects are two, with the
            # low ground between them cut away
            high = (m & (h >= 0.5 * top)).astype(np.uint8)
            high = cv2.morphologyEx(high, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(high, connectivity=8)
            big = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_px]
            if len(big) >= 2:
                merged.append(f"{tid} ({len(big)} plateaus)")
            if (t.get("area_mm2") or 0) < 300:
                tiny.append(tid)
        out = []
        if merged:
            out.append("Likely MERGED detections (the height map shows separate plateaus with a saddle between): " + ", ".join(merged) + ".")
        if tiny:
            out.append("Very small detections (< 3 cm2, debris or a marker corner?): " + ", ".join(tiny) + ".")
        return " ".join(out)

    # ---------------------------------------------------------------- result
    def live(self) -> Dict:
        """What the drawer looks like RIGHT NOW, for the canvas while the agent works (Nolan: "I want to see the
        changes it makes to the outlines as it's making them"). Polygons only — no previews, no masks."""
        with self.lock:
            tools = []
            for tid in self.order:
                t = self.tools.get(tid)
                if t is None:
                    continue
                tools.append({"id": tid, "name": t.get("name"), "status": t.get("status"), "polygon_mm": t.get("polygon_mm") or [],
                              "area_mm2": t.get("area_mm2"), "changed": tid in self.changed, "new": tid not in self.original_ids})
            return {"tools": tools, "removed": list(self.removed), "changed": sorted(self.changed)}

    def result(self) -> Dict:
        tools = []
        for tid in self.order:
            if tid not in self.tools:
                continue
            t = dict(self.tools[tid])
            t["changed"] = tid in self.changed
            t["new"] = tid not in self.original_ids
            if tid in self.previews:
                t["preview_png"] = self.previews[tid]
            tools.append(t)
        return {"tools": tools, "removed": list(self.removed), "changed": sorted(self.changed)}


def run_drawer_agent(dt: DrawerTools, *, instructions: str = "", log_cb: Optional[Callable[[Dict], None]] = None,
                     client=None) -> Dict:
    """The loop. `client` lets tests inject a scripted stand-in for anthropic.Anthropic()."""
    import anthropic
    client = client or anthropic.Anthropic(timeout=recognize.TIMEOUT_S, max_retries=2)
    t0 = time.time()
    steps: List[Dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0}

    def note(entry: Dict):
        entry["t"] = round(time.time() - t0, 1)
        steps.append(entry)
        if log_cb:
            try:
                log_cb(entry)
            except Exception:  # noqa: BLE001
                pass

    intro_txt, intro_imgs = dt.view_drawer()
    listing, _ = dt.list_tools()
    suspects = dt.suspects()
    content: List[Dict] = [{"type": "text", "text": "Here is the drawer. " + intro_txt}] + [_img_block(i) for i in intro_imgs]
    content.append({"type": "text", "text": "Current tools:\n" + listing + ("\n\n" + suspects if suspects else "")
                    + ("\n\nInstructions from the person: " + instructions if instructions else "")
                    + "\n\nInspect what looks wrong (start with the suspects), fix the detections, then the outlines, and call finish. "
                      "You may call several tools in one turn (e.g. view three suspects at once)."})
    messages: List[Dict] = [{"role": "user", "content": content}]
    note({"step": 0, "tool": "start", "summary": f"{len(dt.order)} tools"})
    summary = ""
    calls = 0
    stopped = "finish"
    system_blocks = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    tools_cached = [dict(t) for t in TOOLS]
    tools_cached[-1] = {**tools_cached[-1], "cache_control": {"type": "ephemeral"}}
    while True:
        if calls >= MAX_CALLS:
            stopped = "call budget"; break
        if time.time() - t0 > MAX_SECONDS:
            stopped = "time budget"; break
        if calls and calls % PRUNE_EVERY == 0:
            _prune_images(messages, keep_last=KEEP_IMAGE_TURNS)
        _mark_cache(messages)
        try:
            resp = client.messages.create(model=MODEL, max_tokens=16000, system=system_blocks, tools=tools_cached, messages=messages,
                                          output_config={"effort": EFFORT})
        except Exception as exc:  # noqa: BLE001
            note({"step": calls, "tool": "error", "summary": f"{type(exc).__name__}: {str(exc)[:300]}"})
            stopped = f"api error: {str(exc)[:200]}"; break
        u = getattr(resp, "usage", None)
        if u is not None:
            usage["input_tokens"] += int(getattr(u, "input_tokens", 0) or 0); usage["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)
            usage["cache_read_tokens"] = usage.get("cache_read_tokens", 0) + int(getattr(u, "cache_read_input_tokens", 0) or 0)
            usage["cache_write_tokens"] = usage.get("cache_write_tokens", 0) + int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        texts = [b.text for b in resp.content if b.type == "text" and b.text.strip()]
        for tx in texts:
            note({"step": calls, "tool": "says", "summary": tx[:400]})
        if resp.stop_reason != "tool_use" or not uses:
            stopped = f"stop_reason {resp.stop_reason}"
            summary = summary or " ".join(texts)
            break
        results: List[Dict] = []
        done = False
        for b in uses:
            calls += 1
            name = b.name; inp = dict(b.input or {})
            if name == "finish":
                summary = str(inp.get("summary") or "")
                note({"step": calls, "tool": "finish", "input": inp, "summary": summary[:400]})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": "Finished."})
                done = True
                continue
            try:
                fn = getattr(dt, name)
                with dt.lock:
                    txt, imgs = fn(**inp)
                left = MAX_CALLS - calls; secs_left = int(MAX_SECONDS - (time.time() - t0))
                txt += f"\n[budget: {left} tool calls and {secs_left} s left" + (
                    ". WRAP UP NOW: name the tools you changed and call finish with your summary.]" if left <= 10 or secs_left < 120 else ".]")
                blocks: List[Dict] = [{"type": "text", "text": txt}] + [_img_block(i) for i in imgs]
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": blocks})
                note({"step": calls, "tool": name, "input": inp, "summary": txt[:400]})
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                log.warning("agent tool %s failed: %s\n%s", name, msg, traceback.format_exc())
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": f"Error: {msg}", "is_error": True})
                note({"step": calls, "tool": name, "input": inp, "summary": "ERROR " + msg[:300], "error": True})
        messages.append({"role": "user", "content": results})
        if done:
            break
    if not summary and stopped in ("call budget", "time budget"):
        # one last, tool-less turn for the summary the person will read — the second real run ended mid-inspection
        try:
            messages.append({"role": "user", "content": [{"type": "text", "text": "The budget is used up. In a few sentences: what did you change, what did you leave as measured, and what should the person check by hand?"}]})
            resp = client.messages.create(model=MODEL, max_tokens=2000, system=system_blocks, messages=messages, output_config={"effort": "low"})
            summary = " ".join(b.text for b in resp.content if b.type == "text").strip()
            note({"step": calls, "tool": "finish", "summary": summary[:400]})
        except Exception as exc:  # noqa: BLE001
            log.info("agent closing summary failed: %s", exc)
    try:
        out = dt.result()
    except Exception as exc:  # noqa: BLE001  — never lose the run over a bookkeeping bug; hand back what there is
        log.exception("agent result assembly failed")
        out = {"tools": [dict(t) for t in dt.tools.values()], "removed": list(dt.removed), "changed": sorted(dt.changed)}
        stopped += f" (result assembly error: {type(exc).__name__})"
    out.update({"summary": summary, "stopped": stopped, "calls": calls, "seconds": round(time.time() - t0, 1), "usage": usage, "log": steps})
    return out


KEEP_IMAGE_TURNS = int(os.environ.get("TC_AGENT_KEEP_IMAGES", "4"))
PRUNE_EVERY = int(os.environ.get("TC_AGENT_PRUNE_EVERY", "8"))   # prune in batches: every prune invalidates the cached prefix from that point


def _prune_images(messages: List[Dict], keep_last: int) -> None:
    """Replace pictures in all but the last `keep_last` tool-result turns with a note. Every turn re-sends the whole
    history, so a 60-call run that keeps 100 pictures in context is paying for ~100k picture tokens per call — the
    first real run spent its account's credit in 11 minutes. The model has already looked at those pictures and its
    own words about them stay in the transcript."""
    user_turns = [m for m in messages if m["role"] == "user" and isinstance(m.get("content"), list)]
    for m in user_turns[:-keep_last] if keep_last > 0 else user_turns:
        for blk in m["content"]:
            if isinstance(blk, dict) and blk.get("type") == "tool_result" and isinstance(blk.get("content"), list):
                pruned = []
                for c in blk["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        pruned.append({"type": "text", "text": "[picture shown earlier; call the tool again to see it]"})
                    else:
                        pruned.append(c)
                blk["content"] = pruned
            elif isinstance(blk, dict) and blk.get("type") == "image":
                blk.clear(); blk.update({"type": "text", "text": "[picture shown earlier]"})


def _mark_cache(messages: List[Dict]) -> None:
    """Cache breakpoint on the last user turn so the next call reads the whole prefix from cache."""
    for m in messages:
        if m["role"] == "user" and isinstance(m.get("content"), list):
            for blk in m["content"]:
                if isinstance(blk, dict):
                    blk.pop("cache_control", None)
    last = messages[-1]
    if last["role"] == "user" and isinstance(last.get("content"), list) and last["content"]:
        blk = last["content"][-1]
        if isinstance(blk, dict):
            blk["cache_control"] = {"type": "ephemeral"}
