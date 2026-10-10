"""Tool inspection — the KNOWLEDGE half of outline cleanup.

Shows Claude ONE tool at full resolution — the photo crop, the height map with a scale bar, the whole drawer for
context — with the traced outline drawn on and NUMBERED MARKS around it, and asks what the tool is, which
constraints its footprint obeys, and WHERE the trace is wrong (a shadow spur between marks 12 and 14, a straight
edge from 3 to 9, an edge that sits inside the real one from 20 to 24). The answer is enums, booleans and mark
numbers — never a coordinate. `cleanup.propose` turns each verdict into a measurement on the trace or the photo and
refuses whatever the data does not bear out. A second pass shows the model its own proposal drawn over the photo
and asks whether the line now follows the tool everywhere; its objections are applied as more local edits.

Degrades, never blocks: with no credentials (ANTHROPIC_API_KEY, or an `ant auth login` profile) or any API
failure, `analyze` / `verify` return None and the caller runs the constraint engine with no hints.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL = os.environ.get("TC_RECOGNIZE_MODEL", "claude-opus-5")
EFFORT = os.environ.get("TC_RECOGNIZE_EFFORT", "high")        # low | medium | high | xhigh | max — it may take longer
TIMEOUT_S = float(os.environ.get("TC_RECOGNIZE_TIMEOUT", "180"))
MAX_SIDE = 1400          # longest side of a crop sent to the model (full rectified resolution for most tools)
MIN_SIDE = 700           # small tools are UPSCALED to at least this before marks are drawn, so the marks and numbers stay legible
ORANGE = (27, 106, 242)  # BGR of the brand orange
GREEN = (85, 197, 34)

EDIT_KINDS = ["straight", "arc", "spur", "notch", "too_tight", "too_loose", "merged_neighbour", "missing_part"]

_EDIT = {
    "type": "object",
    "properties": {
        "from_mark": {"type": "integer", "description": "First mark of the run, following the numbering forward"},
        "to_mark": {"type": "integer", "description": "Last mark of the run"},
        "kind": {"type": "string", "enum": EDIT_KINDS},
        "note": {"type": "string", "description": "What you see there, one short sentence"},
    },
    "required": ["from_mark", "to_mark", "kind", "note"],
    "additionalProperties": False,
}

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string", "description": "Short name, e.g. 'steel rule', 'claw hammer', 'tape measure', 'combination square', 'unknown'"},
        "description": {"type": "string", "description": "One or two sentences: what the object is and what its footprint looks like from above"},
        "shape_class": {"type": "string", "enum": ["rectangle", "rounded_rectangle", "capsule", "circle", "L_shape", "T_shape", "irregular"]},
        "symmetric_axis": {"type": "string", "enum": ["long", "short", "none"], "description": "Mirror symmetry of the FOOTPRINT about its long or short axis"},
        "straight_edges": {"type": "boolean", "description": "The footprint has straight sides (rule, square blade, box)"},
        "right_angles": {"type": "boolean", "description": "Its straight sides meet at 90 degrees"},
        "round_shaft": {"type": "boolean", "description": "Has a cylindrical shaft whose top-down width is its diameter"},
        "trace_quality": {"type": "string", "enum": ["good", "minor_issues", "poor"], "description": "How well the orange trace follows the real edge of the tool overall"},
        "edits": {"type": "array", "items": _EDIT, "description": "Place-by-place verdicts on the orange trace, by mark range. Empty if the trace is right everywhere."},
        "confidence": {"type": "number", "description": "0 to 1 — structured output rejects minimum/maximum on numbers, so the range lives here"},
        "notes": {"type": "string"},
    },
    "required": ["tool_name", "description", "shape_class", "symmetric_axis", "straight_edges", "right_angles", "round_shaft",
                 "trace_quality", "edits", "confidence", "notes"],
    "additionalProperties": False,
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "follows_edge": {"type": "boolean", "description": "The GREEN line follows the real edge of the tool everywhere, to within about a millimetre"},
        "issues": {"type": "array", "items": _EDIT, "description": "Where the green line is still wrong, by the numbered marks (which sit on the ORIGINAL orange trace)"},
        "better_than_orange": {"type": "boolean", "description": "The green proposal is at least as good as the orange trace"},
        "notes": {"type": "string"},
    },
    "required": ["follows_edge", "issues", "better_than_orange", "notes"],
    "additionalProperties": False,
}

SYSTEM = (
    "You inspect hand tools lying in a drawer, scanned from above with a depth camera, to help cut foam pockets for "
    "them. For each tool you are shown: (1) a full-resolution top-down photo crop with the scanner's traced outline in "
    "ORANGE and small numbered white marks around it, (2) the same crop as a height map (dark = drawer floor, brighter "
    "= taller, scale bar in mm) with the same trace and marks, and (3) the whole drawer with this tool highlighted. "
    "Say what the tool is, which geometric constraints its FOOTPRINT (its outline on the drawer floor) obeys, and WHERE "
    "the trace is wrong, naming runs by their mark numbers in increasing order along the trace (wrapping past the last "
    "mark back to 0 is fine). Kinds: 'straight' = that run is a straight edge of the tool; 'arc' = a circular arc; "
    "'spur' = the trace sticks OUT past the tool there (a shadow, a bit of a neighbour, depth noise); 'notch' = the "
    "trace dips INTO the tool there (a dark patch, a printed label, a reflection); 'too_tight' = the orange line sits "
    "inside the tool's real edge along that run; 'too_loose' = it sits outside the real edge; 'merged_neighbour' = the "
    "trace includes a second object; 'missing_part' = part of the tool (a thin blade, a transparent part) is outside "
    "the trace entirely. Judge the real edge from the PHOTO and the height map together: depth blurs edges outward by a "
    "millimetre or two and misses thin flat things; the photo shows the true silhouette but also shadows and prints. "
    "Be specific and conservative: only report a run when you can see the problem; a wrong edit distorts a measured "
    "outline, a missing one merely leaves it as measured. You never give sizes or positions — everything is measured "
    "by the scanner from your mark numbers."
)


# ------------------------------------------------------------------ pictures

def _png_b64(img_bgr: np.ndarray, max_side: int = MAX_SIDE) -> str:
    h, w = img_bgr.shape[:2]
    sc = min(1.0, max_side / max(h, w))
    if sc < 1.0:
        img_bgr = cv2.resize(img_bgr, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def height_to_image(h_mm: np.ndarray, max_mm: Optional[float] = None) -> np.ndarray:
    """Height crop as a colour image (floor dark, tall bright) so the model can read relief."""
    h = np.nan_to_num(h_mm, nan=0.0)
    top = float(max_mm or max(1.0, np.percentile(h, 99)))
    v = (255 * np.clip(h / top, 0, 1)).astype(np.uint8)
    return cv2.applyColorMap(v, cv2.COLORMAP_TURBO)


def make_crops(photo_bgr: np.ndarray, height_mm: Optional[np.ndarray], poly_px: np.ndarray, margin_px: int):
    """Crop ONE tool out of the rectified drawer: everything outside its outline is dimmed and the height crop is
    masked to the outline (dilated by the margin). A plain bounding-box crop of a crowded drawer showed the hammer
    together with three screwdrivers and two pliers — the model could not know which one was being asked about.
    Returns (photo_crop, height_crop_or_None, (x0, y0, x1, y1)). The outline itself is drawn by `annotate`."""
    H, W = photo_bgr.shape[:2]
    px = np.asarray(poly_px, dtype=np.float64)
    x0, y0 = max(0, int(px[:, 0].min()) - margin_px), max(0, int(px[:, 1].min()) - margin_px)
    x1, y1 = min(W, int(px[:, 0].max()) + margin_px + 1), min(H, int(px[:, 1].max()) + margin_px + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None, None, (x0, y0, x1, y1)
    local = np.round(px - [x0, y0]).astype(np.int32)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [local], 1)
    k = max(3, (margin_px // 2) | 1)
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    photo = photo_bgr[y0:y1, x0:x1].copy()
    photo[~near] = (photo[~near] * 0.4).astype(np.uint8)
    hcrop = None
    if height_mm is not None:
        hcrop = np.where(near, np.nan_to_num(height_mm[y0:y1, x0:x1]), 0.0).astype(np.float32)
    return photo, hcrop, (x0, y0, x1, y1)


def _draw_ring(img: np.ndarray, ring_px: np.ndarray, color, thickness: int, closed: bool = True):
    pts = np.round(ring_px).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], closed, color, thickness, cv2.LINE_AA)


def _draw_marks(img: np.ndarray, ring_px: np.ndarray, marks: List[int], scale: float):
    """Numbered white discs on the ring, pushed slightly outward so they do not hide the line under them."""
    rad = max(7, int(9 * scale)); font = cv2.FONT_HERSHEY_SIMPLEX; fs = 0.34 * max(1.0, scale); th = 1
    cx, cy = ring_px.mean(axis=0)
    for k, i in enumerate(marks):
        p = ring_px[i]
        v = p - [cx, cy]; v = v / max(1e-6, np.linalg.norm(v))
        q = p + v * (rad * 1.6)
        c = (int(round(q[0])), int(round(q[1])))
        cv2.line(img, c, (int(round(p[0])), int(round(p[1]))), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, c, rad, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, rad, (0, 0, 0), 1, cv2.LINE_AA)
        txt = str(k); (tw, tht), _ = cv2.getTextSize(txt, font, fs, th)
        cv2.putText(img, txt, (c[0] - tw // 2, c[1] + tht // 2), font, fs, (0, 0, 0), th, cv2.LINE_AA)


def _scale_bar(img: np.ndarray, mpp: float, label: str):
    """A 20 mm bar bottom-left, so the model can relate what it sees to millimetres."""
    L = int(round(20.0 / mpp)); H = img.shape[0]
    x, y = 12, H - 14
    cv2.rectangle(img, (x - 4, y - 22), (x + max(L, 120) + 4, y + 8), (0, 0, 0), -1)
    cv2.line(img, (x, y), (x + L, y), (255, 255, 255), 2)
    cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def _upscale(img: np.ndarray, ring_px: np.ndarray, mpp: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Small crops are enlarged (cubic) so the marks, numbers and the line itself are legible: a 70 x 30 mm tool on
    the 0.45 mm/px mosaic is a 168 x 344 px picture, and 32 marks on it overlapped each other and the outline.
    Returns the image, the ring in its pixels and the effective mm/px. Capped so the sent picture stays <= MAX_SIDE."""
    h, w = img.shape[:2]
    up = max(1.0, MIN_SIDE / max(h, w))
    up = min(up, MAX_SIDE / max(h, w))
    if up <= 1.01:
        return img.copy(), np.asarray(ring_px, dtype=np.float64), mpp
    out = cv2.resize(img, (int(round(w * up)), int(round(h * up))), interpolation=cv2.INTER_CUBIC)
    return out, np.asarray(ring_px, dtype=np.float64) * up, mpp / up


def annotate(photo_crop: np.ndarray, height_crop: Optional[np.ndarray], ring_px_local: np.ndarray, marks: List[int],
             mpp: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The two pictures the model inspects: the photo crop and the height crop, both with the trace in orange and the
    numbered marks. Small crops are upscaled first (`_upscale`); `_png_b64` caps the sent size at MAX_SIDE."""
    photo, ring, eff = _upscale(photo_crop, ring_px_local, mpp)
    scale = max(0.6, min(2.0, 0.19 / eff))        # marks sized for a 0.19 mm/px picture
    _draw_ring(photo, ring, ORANGE, max(1, int(round(1.5 * scale))))
    _draw_marks(photo, ring, marks, scale)
    _scale_bar(photo, eff, "20 mm")
    himg = None
    if height_crop is not None:
        top = float(max(1.0, np.percentile(height_crop[height_crop > 0], 99))) if (height_crop > 0).any() else 1.0
        himg, _, _ = _upscale(height_to_image(height_crop, top), ring_px_local, mpp)
        _draw_ring(himg, ring, (255, 255, 255), max(1, int(round(1.5 * scale))))
        _draw_marks(himg, ring, marks, scale)
        _scale_bar(himg, eff, f"20 mm - colour tops out at {top:.0f} mm tall")
    return photo, himg


def context_image(rectified: np.ndarray, poly_px: np.ndarray, max_side: int = 1000) -> np.ndarray:
    """The whole drawer, small, with this tool's outline highlighted — so a socket reads as one of a set, a blade as
    the mate of a handle lying next to it."""
    h, w = rectified.shape[:2]
    sc = min(1.0, max_side / max(h, w))
    img = cv2.resize(rectified, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    _draw_ring(img, np.asarray(poly_px) * sc, ORANGE, 2)
    return img


def render_preview(photo_crop: np.ndarray, measured_px_local: np.ndarray, proposed_px_local: np.ndarray, marks: List[int],
                   mpp: float) -> np.ndarray:
    """Measured trace faded orange, proposal solid green, the ORIGINAL marks (the verify pass names runs by them)."""
    img, measured, eff = _upscale(photo_crop, measured_px_local, mpp)
    up = eff and (mpp / eff)
    proposed = np.asarray(proposed_px_local, dtype=np.float64) * up
    scale = max(0.6, min(2.0, 0.19 / eff))
    faded = img.copy()
    _draw_ring(faded, measured, ORANGE, max(1, int(round(1.5 * scale))))
    img = cv2.addWeighted(img, 0.5, faded, 0.5, 0)
    _draw_ring(img, proposed, GREEN, max(2, int(round(2 * scale))))
    _draw_marks(img, measured, marks, scale)
    _scale_bar(img, eff, "20 mm")
    return img


# ------------------------------------------------------------------ the model

_UNAVAILABLE: Optional[str] = None   # set once when credentials/SDK are missing, so 24 tools do not log 24 warnings


def unavailable_reason() -> Optional[str]:
    return _UNAVAILABLE


def _call(content: List[Dict], schema: Dict, max_tokens: int = 16000) -> Optional[Dict]:
    global _UNAVAILABLE
    if _UNAVAILABLE:
        return None
    try:
        import anthropic
    except Exception as exc:  # noqa: BLE001
        _UNAVAILABLE = f"anthropic SDK not importable ({exc})"
        log.info("recognize: %s", _UNAVAILABLE)
        return None
    try:
        client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=2)
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            log.info("recognize: model declined (%s)", getattr(response.stop_details, "category", None))
            return None
        texts = [b.text for b in response.content if b.type == "text"]
        if not texts:
            # a StopIteration here printed as an EMPTY warning on the merged-screwdrivers tool (2026-10-02): the model
            # had spent the whole budget thinking and never wrote the JSON
            log.warning("recognize: no text in the reply (stop_reason=%s, usage=%s)", response.stop_reason, getattr(response, "usage", None))
            return None
        out = json.loads(texts[0])
        out["model"] = MODEL
        u = getattr(response, "usage", None)
        if u is not None:
            out["usage"] = {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None)}
        return out
    except anthropic.AuthenticationError:
        _UNAVAILABLE = "Anthropic API key rejected — check ANTHROPIC_API_KEY"
        log.warning("recognize: %s", _UNAVAILABLE)
        return None
    except anthropic.APIConnectionError as exc:
        log.info("recognize: could not reach the API (%s)", exc)
        return None
    except anthropic.APIStatusError as exc:
        log.warning("recognize: API error %s: %s", exc.status_code, exc.message)
        return None
    except anthropic.AnthropicError as exc:
        if "authentication" in str(exc).lower() or "api_key" in str(exc).lower():
            _UNAVAILABLE = "no Anthropic credentials on the server — set ANTHROPIC_API_KEY (or run `ant auth login`) before starting the backend"
            log.info("recognize: %s", _UNAVAILABLE)
            return None
        log.warning("recognize: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        # the SDK raises a plain TypeError for "Could not resolve authentication method" — not an AnthropicError —
        # which is the no-key case on a fresh machine; say it once, not once per tool
        if "authentication method" in str(exc).lower():
            _UNAVAILABLE = "no Anthropic credentials on the server — set ANTHROPIC_API_KEY (or run `ant auth login`) before starting the backend"
            log.info("recognize: %s", _UNAVAILABLE)
            return None
        log.warning("recognize: %s: %s", type(exc).__name__, exc)
        return None


def _facts(name_hint: str, size_mm: Optional[tuple], thickness_mm: Optional[float], n_marks: int, mpp: float) -> str:
    facts = [f"The trace has {n_marks} numbered marks (0 to {n_marks - 1}) running around it in order."]
    if name_hint:
        facts.append(f"The scanner's working name for it is '{name_hint}'.")
    if size_mm:
        facts.append(f"Measured footprint about {size_mm[0]:.0f} x {size_mm[1]:.0f} mm.")
    if thickness_mm:
        facts.append(f"Measured height about {thickness_mm:.0f} mm.")
    facts.append(f"One photo pixel is {mpp:.2f} mm.")
    return " ".join(facts)


def analyze(photo_marked: np.ndarray, height_marked: Optional[np.ndarray], context: Optional[np.ndarray], *, name_hint: str = "",
            size_mm: Optional[tuple] = None, thickness_mm: Optional[float] = None, n_marks: int, mpp: float) -> Optional[Dict]:
    """First pass: what is it, what constraints hold, where is the trace wrong. None when the model is unavailable."""
    content: List[Dict] = [{"type": "text", "text": "Top-down photo crop of one object in a tool drawer, with the scanner's trace in orange and numbered marks:"},
                           _image_block(photo_marked)]
    if height_marked is not None:
        content += [{"type": "text", "text": "The same crop as a height map (dark = drawer floor, brighter = taller), same trace and marks:"},
                    _image_block(height_marked)]
    if context is not None:
        content += [{"type": "text", "text": "The whole drawer, with this tool outlined in orange, for context:"}, _image_block(context)]
    content.append({"type": "text", "text": _facts(name_hint, size_mm, thickness_mm, n_marks, mpp) +
                    " What is it, which constraints does its footprint obey, and where exactly is the orange trace wrong?"})
    out = _call(content, ANALYSIS_SCHEMA)
    if out is not None:
        out["edits"] = _clean_edits(out.get("edits"), n_marks)
    return out


def verify(preview: np.ndarray, *, name_hint: str, n_marks: int, mpp: float) -> Optional[Dict]:
    """Second pass: the proposal (green) over the photo, the measured trace faded orange, the original marks."""
    content: List[Dict] = [
        {"type": "text", "text": f"Here is the proposed cleaned outline for the {name_hint or 'tool'} in GREEN over the photo; the original "
                                 f"scanner trace is the faded orange line and the {n_marks} numbered marks sit on that original trace. "
                                 f"One photo pixel is {mpp:.2f} mm. Does the green line follow the real edge of the tool everywhere? "
                                 "Name any run where it does not, by the mark numbers, with what is wrong there."},
        _image_block(preview),
    ]
    out = _call(content, VERIFY_SCHEMA, max_tokens=8000)
    if out is not None:
        out["issues"] = _clean_edits(out.get("issues"), n_marks)
    return out


def _clean_edits(edits, n_marks: int) -> List[Dict]:
    """Keep only well-formed edits whose marks exist (the schema types them, but a mark past the end is still possible)."""
    out = []
    for e in edits or []:
        try:
            a, b = int(e["from_mark"]), int(e["to_mark"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= a < n_marks and 0 <= b < n_marks) or str(e.get("kind")) not in EDIT_KINDS:
            continue
        out.append({"from_mark": a, "to_mark": b, "kind": str(e["kind"]), "note": str(e.get("note") or "")})
    return out


SOLID_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string"},
        "solid_class": {"type": "string", "enum": ["box", "cylinder_lying", "cylinder_upright", "extrusion", "freeform"],
                        "description": "The 3D form of the object as it lies: box = flat top (a drive, a case, a cube); cylinder_lying = round body on its side (bottle, pen, handle); cylinder_upright = round can standing (flat top); extrusion = one cross-section profile along its length (a bar, a rule, a wrench shaft); freeform = anything else (a hammer head + handle, pliers)"},
        "symmetric": {"type": "boolean", "description": "Mirror-symmetric about its long axis in 3D (both halves the same height)"},
        "confidence": {"type": "number", "description": "0 to 1"},
    },
    "required": ["tool_name", "solid_class", "symmetric", "confidence"],
    "additionalProperties": False,
}


def solid_class(photo_crop: np.ndarray, height_crop: Optional[np.ndarray], *, name_hint: str = "", size_mm: Optional[tuple] = None,
                thickness_mm: Optional[float] = None) -> Optional[Dict]:
    """One light call: what SOLID is this object (box / lying cylinder / upright cylinder / extrusion / freeform) and is
    it symmetric about its long axis — the semantic hint for `solids.clean_relief`, which then fits that primitive to
    the scan and keeps it only if the fit holds. No marks, no edits: a cheap, cacheable question."""
    content: List[Dict] = [{"type": "text", "text": "Top-down photo crop of one object lying in a drawer:"}, _image_block(photo_crop, side=700)]
    if height_crop is not None:
        content += [{"type": "text", "text": "Its height map (dark = floor, brighter = taller):"}, _image_block(height_to_image(height_crop), side=700)]
    facts = []
    if name_hint:
        facts.append(f"Working name: '{name_hint}'.")
    if size_mm:
        facts.append(f"Footprint about {size_mm[0]:.0f} x {size_mm[1]:.0f} mm.")
    if thickness_mm:
        facts.append(f"Height about {thickness_mm:.0f} mm.")
    content.append({"type": "text", "text": " ".join(facts) + " What solid is it, and is it mirror-symmetric about its long axis?"})
    return _call(content, SOLID_SCHEMA, max_tokens=2000)


def _image_block(img: np.ndarray, side: int = MAX_SIDE) -> Dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(img, max_side=side)}}


def hints_from(recognition: Optional[Dict]) -> Dict:
    """Constraint hints for cleanup.propose from an analysis (empty dict = auto-detect, strict)."""
    if not recognition or float(recognition.get("confidence", 0)) < 0.5:
        return {}
    return {
        "shape_class": recognition.get("shape_class"),
        "symmetric_axis": recognition.get("symmetric_axis"),
        "straight_edges": bool(recognition.get("straight_edges", True)),
        "right_angles": bool(recognition.get("right_angles", True)),
        "round_shaft": bool(recognition.get("round_shaft", False)),
    }
