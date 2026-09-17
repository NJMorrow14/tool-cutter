"""HQ-SAM wrapper: lazy model loading, per-session image embedding cache, prompt-based prediction."""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
SAM_HQ_ROOT = BACKEND_ROOT / "sam-hq"
if SAM_HQ_ROOT.exists() and str(SAM_HQ_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_HQ_ROOT))

MODEL_PREFERENCE = ["vit_h", "vit_l", "vit_b", "vit_tiny"]


def infer_model_type(path: str) -> Optional[str]:
    name = os.path.basename(path).lower()
    for mt in MODEL_PREFERENCE:
        if mt in name:
            return mt
    if re.search(r"vit[_-]?t", name):
        return "vit_tiny"
    return None


def discover_checkpoint() -> Tuple[Optional[str], Optional[str]]:
    """Use HQSAM_CKPT if set, else the best *.pth found in backend/."""
    env = os.environ.get("HQSAM_CKPT", "").strip()
    if env:
        mt = os.environ.get("HQSAM_MODEL_TYPE", "").strip() or infer_model_type(env) or "vit_h"
        return env, mt
    candidates = []
    for p in BACKEND_ROOT.glob("*.pth"):
        mt = infer_model_type(str(p))
        if mt:
            candidates.append((MODEL_PREFERENCE.index(mt), str(p), mt))
    if not candidates:
        return None, None
    candidates.sort()
    return candidates[0][1], candidates[0][2]


class Segmenter:
    def __init__(self, checkpoint: Optional[str] = None, model_type: Optional[str] = None,
                 device: Optional[str] = None):
        if checkpoint is None:
            checkpoint, model_type = discover_checkpoint()
        self.checkpoint = checkpoint
        self.model_type = model_type or (infer_model_type(checkpoint) if checkpoint else None) or "vit_h"
        self.device_pref = device or os.environ.get("HQSAM_DEVICE") or None
        self.device: Optional[str] = None
        self._predictor = None
        self._lock = threading.Lock()
        self._image_key: Optional[str] = None
        self.load_error: Optional[str] = None

    # ------------------------------------------------------------------ status
    @property
    def available(self) -> bool:
        return bool(self.checkpoint) and os.path.exists(self.checkpoint)

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    def info(self) -> Dict:
        return {
            "available": self.available,
            "loaded": self.loaded,
            "checkpoint": os.path.basename(self.checkpoint) if self.checkpoint else None,
            "model_type": self.model_type if self.checkpoint else None,
            "device": self.device,
            "error": self.load_error,
        }

    # ----------------------------------------------------------------- loading
    def _pick_device(self) -> str:
        import torch  # type: ignore

        if self.device_pref:
            return self.device_pref
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _build(self, device: str):
        import torch  # type: ignore
        from segment_anything import SamPredictor, sam_model_registry  # HQ-SAM fork (vendored)

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        self.device = device
        self._image_key = None
        log.info("Loaded HQ-SAM %s from %s on %s", self.model_type, self.checkpoint, device)

    def ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        if not self.available:
            raise RuntimeError(
                "No HQ-SAM checkpoint found. Put sam_hq_vit_*.pth in backend/ or set HQSAM_CKPT.")
        with self._lock:
            if self._predictor is not None:
                return
            device = self._pick_device()
            try:
                self._build(device)
            except Exception as exc:  # noqa: BLE001
                if device != "cpu":
                    log.warning("Loading on %s failed (%s); retrying on CPU", device, exc)
                    self._build("cpu")
                else:
                    self.load_error = str(exc)
                    raise

    def _fallback_to_cpu(self) -> None:
        log.warning("Switching HQ-SAM to CPU after a device error")
        self._predictor = None
        self._build("cpu")

    # --------------------------------------------------------------- inference
    def set_image(self, key: str, img_bgr: np.ndarray) -> None:
        self.ensure_loaded()
        if self._image_key == key:
            return
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        import torch  # type: ignore

        with self._lock:
            try:
                with torch.inference_mode():
                    self._predictor.set_image(rgb)
            except RuntimeError as exc:
                if self.device == "cpu":
                    raise
                log.warning("set_image failed on %s: %s", self.device, exc)
                self._fallback_to_cpu()
                with torch.inference_mode():
                    self._predictor.set_image(rgb)
            self._image_key = key

    def predict(self, points: Optional[Sequence[Tuple[float, float]]], labels: Optional[Sequence[int]],
                box: Optional[Sequence[float]] = None, hq_token_only: bool = False) -> np.ndarray:
        """Return a boolean mask for one object given point prompts and/or a box."""
        if self._predictor is None or self._image_key is None:
            raise RuntimeError("set_image must be called before predict")
        import torch  # type: ignore

        pc = np.asarray(points, dtype=np.float32) if points else None
        pl = np.asarray(labels, dtype=np.int32) if labels else None
        bx = np.asarray(box, dtype=np.float32) if box is not None else None
        with self._lock:
            with torch.inference_mode():
                masks, scores, _ = self._predictor.predict(
                    point_coords=pc, point_labels=pl, box=bx, multimask_output=False,
                    hq_token_only=hq_token_only)
        return masks[0].astype(bool)


def clean_mask(mask: np.ndarray, positives: Sequence[Tuple[float, float]], min_area_px: float = 30.0,
               fill_holes: bool = True) -> np.ndarray:
    """Keep the connected component(s) containing positive prompts (else the largest); fill holes."""
    m = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return np.zeros_like(m, dtype=bool)
    keep = set()
    h, w = m.shape
    for (x, y) in positives or []:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            lab = int(labels[yi, xi])
            if lab > 0:
                keep.add(lab)
    if not keep:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep.add(int(np.argmax(areas)) + 1)
    out = np.isin(labels, list(keep))
    out &= np.isin(labels, [l for l in keep if stats[l, cv2.CC_STAT_AREA] >= min_area_px]) | out
    out_u8 = out.astype(np.uint8) * 255
    if fill_holes:
        contours, _ = cv2.findContours(out_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(out_u8)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        out_u8 = filled
    return out_u8 > 0
