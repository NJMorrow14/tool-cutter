"""Tool library: single tools photographed one by one on the phone, kept per job / customer.

Contract: docs/tool-library.md. Storage (under `root`, i.e. `TC_LIBRARY_DIR` or backend/library):

    <job_id>/job.json
    <job_id>/<tool_id>/tool.json, photo.jpg, thumb.jpg[, mask.png]

Ids are 12 lowercase hex. The phone generates tool ids so a retried upload replaces instead of duplicating.
All writes are atomic (temp file + os.replace) and every mutation of a job's directory runs under that job's lock.
Errors are raised as `LibraryError(message, status)`; the Flask layer turns them into `{"error": ...}` responses.
"""
from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

ID_RE = re.compile(r"[0-9a-f]{12}")
NAME_MAX = 80
NOTES_MAX = 2000
MIN_AREA_MM2 = 25.0
MAX_THICKNESS_MM = 300.0
THUMB_SIDE = 320
JPEG_QUALITY = 85
SOURCE = "phone_photo"


class LibraryError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ----------------------------------------------------------------------------- ids, locks, files

def new_id() -> str:
    return secrets.token_hex(6)


def check_id(value: Any, what: str = "id") -> str:
    """A 12-hex id or a 400. Also the path-traversal guard: nothing else ever reaches the filesystem."""
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise LibraryError(f"Not a {what} (12 lowercase hex characters)")
    return value


_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def job_lock(job_id: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(job_id, threading.RLock())


def _write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    _write_bytes(path, json.dumps(obj, indent=1).encode())


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _job_dir(root: Path, job_id: str) -> Path:
    d = root / Path(check_id(job_id, "job id")).name
    if d.parent != root:
        raise LibraryError("Not a job id")
    return d


# ----------------------------------------------------------------------------- validation

def _clean_name(value: Any, required: bool) -> Optional[str]:
    if value is None:
        if required:
            raise LibraryError("name is required")
        return None
    if not isinstance(value, str):
        raise LibraryError("name must be a string")
    name = value.strip()
    if not name:
        if required:
            raise LibraryError("name is required")
        return None
    if len(name) > NAME_MAX:
        raise LibraryError(f"name must be at most {NAME_MAX} characters")
    return name


def _clean_thickness(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LibraryError("thickness_mm must be a number or null")
    t = float(value)
    if not math.isfinite(t) or t <= 0 or t > MAX_THICKNESS_MM:
        raise LibraryError(f"thickness_mm must be > 0 and <= {MAX_THICKNESS_MM:g}")
    return t


def _clean_notes(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise LibraryError("notes must be a string")
    if len(value) > NOTES_MAX:
        raise LibraryError(f"notes must be at most {NOTES_MAX} characters")
    return value


def normalise_polygon(value: Any) -> Tuple[List[List[float]], float, List[float]]:
    """-> (polygon_mm, area_mm2, bbox_mm). >= 3 finite points; buffer(0) repairs self-intersections and the
    largest part is kept; translated so min x = min y = 0; no repeated closing point."""
    if not isinstance(value, (list, tuple)):
        raise LibraryError("polygon_mm must be a list of [x, y] points")
    pts: List[Tuple[float, float]] = []
    for p in value:
        if not isinstance(p, (list, tuple)) or len(p) != 2 or any(isinstance(c, bool) or not isinstance(c, (int, float)) for c in p):
            raise LibraryError("polygon_mm points must be [x, y] numbers")
        x, y = float(p[0]), float(p[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise LibraryError("polygon_mm points must be finite")
        pts.append((x, y))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise LibraryError("polygon_mm needs at least 3 points")
    try:
        geom = Polygon(pts).buffer(0)
    except Exception as exc:  # noqa: BLE001
        raise LibraryError(f"polygon_mm is not a valid polygon: {exc}")
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.is_empty or not isinstance(geom, Polygon):
        raise LibraryError("polygon_mm is not a valid polygon")
    ring = list(geom.exterior.coords)[:-1]
    if len(ring) < 3:
        raise LibraryError("polygon_mm is not a valid polygon")
    minx = min(x for x, _ in ring)
    miny = min(y for _, y in ring)
    out = [[round(x - minx, 3), round(y - miny, 3)] for x, y in ring]
    # rounding can make neighbours equal; drop consecutive duplicates (and a closing repeat)
    dedup: List[List[float]] = []
    for p in out:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup.pop()
    stored = Polygon(dedup)
    area = float(stored.area)
    if len(dedup) < 3 or not math.isfinite(area) or area < MIN_AREA_MM2:
        raise LibraryError(f"polygon_mm area must be at least {MIN_AREA_MM2:g} mm2")
    w = max(x for x, _ in dedup)
    h = max(y for _, y in dedup)
    return dedup, round(area, 3), [round(w, 3), round(h, 3)]


def _decode(data: bytes, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    if not data:
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    return img


def make_thumb(img: np.ndarray) -> bytes:
    h, w = img.shape[:2]
    s = THUMB_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise LibraryError("Could not encode the thumbnail", 500)
    return buf.tobytes()


# ----------------------------------------------------------------------------- records

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tool_view(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    out["photo_url"] = f"/api/library/tools/{rec['id']}/photo"
    out["thumb_url"] = f"/api/library/tools/{rec['id']}/thumb"
    return out


def _tool_dirs(job_dir: Path) -> List[Path]:
    if not job_dir.is_dir():
        return []
    return [d for d in job_dir.iterdir() if d.is_dir() and ID_RE.fullmatch(d.name) and (d / "tool.json").is_file()]


def _job_view(job_dir: Path, job: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(job)
    out["tool_count"] = len(_tool_dirs(job_dir))
    return out


def _load_job(root: Path, job_id: str) -> Tuple[Path, Dict[str, Any]]:
    d = _job_dir(root, job_id)
    job = _read_json(d / "job.json") if d.is_dir() else None
    if job is None:
        raise LibraryError(f"No job {job_id}", 404)
    return d, job


def _touch_job(job_dir: Path, job: Dict[str, Any]) -> None:
    job["updated"] = time.time()
    _write_json(job_dir / "job.json", job)


def find_tool(root: Path, tool_id: str) -> Tuple[Path, Dict[str, Any]]:
    """-> (tool dir, record). Searches every job; tool ids are unique across the library."""
    check_id(tool_id, "tool id")
    if root.is_dir():
        for jd in root.iterdir():
            if not (jd.is_dir() and ID_RE.fullmatch(jd.name)):
                continue
            td = jd / tool_id
            if td.parent.parent == root and (td / "tool.json").is_file():
                rec = _read_json(td / "tool.json")
                if rec is not None:
                    return td, rec
    raise LibraryError(f"No tool {tool_id}", 404)


# ----------------------------------------------------------------------------- jobs

def list_jobs(root: Path) -> List[Dict[str, Any]]:
    jobs = []
    if root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and ID_RE.fullmatch(d.name):
                job = _read_json(d / "job.json")
                if job is not None:
                    jobs.append(_job_view(d, job))
    jobs.sort(key=lambda j: (j.get("created", 0), j.get("id", "")), reverse=True)
    return jobs


def create_job(root: Path, body: Dict[str, Any]) -> Dict[str, Any]:
    name = _clean_name(body.get("name"), required=True)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        job_id = new_id()
        d = root / job_id
        try:
            d.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise LibraryError("Could not allocate a job id", 500)
    now = time.time()
    job = {"id": job_id, "name": name, "created": now, "updated": now}
    with job_lock(job_id):
        _write_json(d / "job.json", job)
    return _job_view(d, job)


def get_job(root: Path, job_id: str) -> Dict[str, Any]:
    d, job = _load_job(root, job_id)
    return _job_view(d, job)


def update_job(root: Path, job_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    with job_lock(check_id(job_id, "job id")):
        d, job = _load_job(root, job_id)
        if "name" in body:
            job["name"] = _clean_name(body.get("name"), required=True)
        _touch_job(d, job)
        return _job_view(d, job)


def delete_job(root: Path, job_id: str) -> None:
    with job_lock(check_id(job_id, "job id")):
        d, _ = _load_job(root, job_id)
        if d.parent != root:
            raise LibraryError("Not a job id")
        shutil.rmtree(d)


# ----------------------------------------------------------------------------- tools

def list_tools(root: Path, job_id: str) -> List[Dict[str, Any]]:
    d, _ = _load_job(root, job_id)
    tools = []
    for td in _tool_dirs(d):
        rec = _read_json(td / "tool.json")
        if rec is not None:
            tools.append(tool_view(rec))
    tools.sort(key=lambda t: (t.get("created", 0), t.get("id", "")), reverse=True)
    return tools


def upload_tool(root: Path, job_id: str, tool_json: Optional[str], photo: Optional[bytes],
                mask: Optional[bytes] = None) -> Tuple[Dict[str, Any], bool]:
    """Multipart upload -> (ToolRecord, created). An existing id in this job is replaced; an id that lives in
    another job is a 409."""
    check_id(job_id, "job id")
    if not tool_json:
        raise LibraryError('Send the tool as multipart field "tool" (a JSON string)')
    try:
        meta = json.loads(tool_json)
    except ValueError as exc:
        raise LibraryError(f"tool is not valid JSON: {exc}")
    if not isinstance(meta, dict):
        raise LibraryError("tool must be a JSON object")
    if not photo:
        raise LibraryError('Send the photo as multipart field "photo" (JPEG)')
    img = _decode(photo)
    if img is None:
        raise LibraryError("photo is not a readable image")
    if photo[:3] != b"\xff\xd8\xff":                 # not a JPEG: store a JPEG anyway, the routes serve image/jpeg
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        photo = buf.tobytes()
    if mask:
        if _decode(mask, cv2.IMREAD_UNCHANGED) is None:
            raise LibraryError("mask is not a readable PNG")
        if mask[:8] != b"\x89PNG\r\n\x1a\n":
            raise LibraryError("mask must be a PNG")

    polygon, area, bbox = normalise_polygon(meta.get("polygon_mm"))
    thickness = _clean_thickness(meta.get("thickness_mm"))
    name = _clean_name(meta.get("name"), required=False)
    notes = _clean_notes(meta.get("notes"))
    edited = meta.get("edited", False)
    if not isinstance(edited, bool):
        raise LibraryError("edited must be true or false")
    scale = meta.get("scale")
    if scale is not None and not isinstance(scale, dict):
        raise LibraryError("scale must be an object or null")
    captured_at = meta.get("captured_at")
    if captured_at is not None and not isinstance(captured_at, str):
        raise LibraryError("captured_at must be an ISO 8601 string")
    tool_id = meta.get("id")
    if tool_id is not None:
        check_id(tool_id, "tool id")
    thumb = make_thumb(img)

    with job_lock(job_id):
        jd, job = _load_job(root, job_id)
        existing: Optional[Dict[str, Any]] = None
        if tool_id is not None:
            try:
                td_found, rec_found = find_tool(root, tool_id)
            except LibraryError:
                td_found = None
            if td_found is not None:
                if td_found.parent != jd:
                    raise LibraryError(f"Tool {tool_id} belongs to another job ({td_found.parent.name})", 409)
                existing = rec_found
        else:
            for _ in range(10):
                tool_id = new_id()
                try:
                    find_tool(root, tool_id)
                except LibraryError:
                    break
        now = time.time()
        if name is None:
            name = existing["name"] if existing else f"Tool {len(_tool_dirs(jd)) + 1}"
        rec = {
            "id": tool_id, "job_id": job_id, "name": name, "thickness_mm": thickness,
            "polygon_mm": polygon, "area_mm2": area, "bbox_mm": bbox,
            "captured_at": captured_at or (existing or {}).get("captured_at") or _now_iso(),
            "source": SOURCE, "scale": scale, "edited": edited, "notes": notes,
            "created": existing["created"] if existing else now, "updated": now,
        }
        td = jd / tool_id
        td.mkdir(exist_ok=True)
        _write_bytes(td / "photo.jpg", photo)
        _write_bytes(td / "thumb.jpg", thumb)
        if mask:
            _write_bytes(td / "mask.png", mask)
        elif (td / "mask.png").exists():
            (td / "mask.png").unlink()                # a replacement without a mask must not keep the old one
        _write_json(td / "tool.json", rec)            # last: a tool is visible only once its files are in place
        _touch_job(jd, job)
        return tool_view(rec), existing is None


def get_tool(root: Path, tool_id: str) -> Dict[str, Any]:
    return tool_view(find_tool(root, tool_id)[1])


def update_tool(root: Path, tool_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise LibraryError("Send a JSON object")
    td, rec = find_tool(root, tool_id)
    src_job = td.parent.name
    dst_job = src_job
    if body.get("job_id") is not None:
        dst_job = check_id(body["job_id"], "job id")
    locks = [job_lock(j) for j in sorted({src_job, dst_job})]
    for lk in locks:
        lk.acquire()
    try:
        td, rec = find_tool(root, tool_id)        # re-read under the lock
        if td.parent.name != src_job:
            raise LibraryError("The tool moved while this request ran; retry", 409)
        if "name" in body:
            rec["name"] = _clean_name(body.get("name"), required=True)
        if "thickness_mm" in body:
            rec["thickness_mm"] = _clean_thickness(body.get("thickness_mm"))
        if "notes" in body:
            rec["notes"] = _clean_notes(body.get("notes"))
        if "polygon_mm" in body:
            rec["polygon_mm"], rec["area_mm2"], rec["bbox_mm"] = normalise_polygon(body.get("polygon_mm"))
        src_dir, src = _load_job(root, src_job)
        if dst_job != src_job:
            dst_dir, dst = _load_job(root, dst_job)
            target = dst_dir / tool_id
            if target.exists():
                raise LibraryError(f"Job {dst_job} already has a tool {tool_id}", 409)
            rec["job_id"] = dst_job
            rec["updated"] = time.time()
            _write_json(td / "tool.json", rec)
            os.replace(td, target)
            _touch_job(dst_dir, dst)
        else:
            rec["updated"] = time.time()
            _write_json(td / "tool.json", rec)
        _touch_job(src_dir, src)
        return tool_view(rec)
    finally:
        for lk in reversed(locks):
            lk.release()


def delete_tool(root: Path, tool_id: str) -> None:
    td, _ = find_tool(root, tool_id)
    job_id = td.parent.name
    with job_lock(job_id):
        td, _ = find_tool(root, tool_id)
        if td.parent.parent != root:
            raise LibraryError("Not a tool id")
        shutil.rmtree(td)
        job = _read_json(td.parent / "job.json")
        if job is not None:
            _touch_job(td.parent, job)


def tool_file(root: Path, tool_id: str, kind: str) -> bytes:
    td, _ = find_tool(root, tool_id)
    name = {"photo": "photo.jpg", "thumb": "thumb.jpg"}[kind]
    p = td / name
    if not p.is_file():
        raise LibraryError(f"No {kind} for tool {tool_id}", 404)
    return p.read_bytes()
