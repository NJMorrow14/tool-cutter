"""In-memory session store for uploaded photos / scans and their derived data."""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class Session:
    id: str
    source_kind: str                         # "photo" | "scan" (layout on a mat) | "object" (one tool)
    filename: str
    original: np.ndarray                     # BGR
    original_height: Optional[np.ndarray] = None   # float32 mm (scans)
    original_frac: Optional[np.ndarray] = None     # float32 0..1 (scans)
    original_mm_per_px: Optional[float] = None     # scans only (metric raster)
    suggested_corners: Optional[list] = None
    scan_meta: Dict = field(default_factory=dict)
    object_meta: Dict = field(default_factory=dict)
    rectified: Optional[np.ndarray] = None
    rect_height: Optional[np.ndarray] = None
    rect_frac: Optional[np.ndarray] = None
    mm_per_px: Optional[float] = None
    mat_mm: Optional[Tuple[float, float]] = None
    corners: Optional[list] = None
    homography: Optional[np.ndarray] = None
    version: int = 0
    created: float = field(default_factory=time.time)
    masks: Dict[str, np.ndarray] = field(default_factory=dict)

    def info(self) -> Dict:
        h, w = self.original.shape[:2]
        d: Dict = {
            "id": self.id,
            "source_kind": self.source_kind,
            "filename": self.filename,
            "version": self.version,
            "original": {"width": int(w), "height": int(h)},
            "suggested_corners": self.suggested_corners,
            "scan": ({"mm_per_px": self.original_mm_per_px, "has_height": self.original_height is not None,
                      **self.scan_meta} if self.source_kind in ("scan", "object") else None),
            "object": (self.object_meta or None) if self.source_kind == "object" else None,
        }
        if self.rectified is not None:
            rh, rw = self.rectified.shape[:2]
            d["rectified"] = {"width": int(rw), "height": int(rh), "mm_per_px": self.mm_per_px,
                              "has_height": self.rect_height is not None}
            d["mat_mm"] = {"width": self.mat_mm[0], "height": self.mat_mm[1]} if self.mat_mm else None
            d["corners"] = self.corners
        return d


class SessionStore:
    def __init__(self, max_sessions: int = 6):
        self._items: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions

    def create(self, **kwargs) -> Session:
        sid = uuid.uuid4().hex[:12]
        s = Session(id=sid, **kwargs)
        with self._lock:
            self._items[sid] = s
            while len(self._items) > self.max_sessions:
                self._items.popitem(last=False)
        return s

    def get(self, sid: str) -> Optional[Session]:
        with self._lock:
            s = self._items.get(sid)
            if s is not None:
                self._items.move_to_end(sid)
            return s
