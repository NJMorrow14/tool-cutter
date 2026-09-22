"""Run the macOS PhotogrammetrySession worker (mac/Photogrammetry) on a folder of frames."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = BACKEND_ROOT.parent / "mac" / "Photogrammetry" / ".build" / "release" / "photogrammetry"


def worker_binary() -> Optional[Path]:
    env = os.environ.get("TOOLCUTTER_PHOTOGRAMMETRY_BIN")
    for cand in ([Path(env)] if env else []) + [DEFAULT_BIN]:
        if cand.exists() and os.access(cand, os.X_OK):
            return cand
    return None


def available() -> bool:
    return worker_binary() is not None


def run_photogrammetry(input_dir: Path, output_path: Path, detail: str = "medium", timeout_s: int = 1800,
                       progress: Optional[Callable[[float, str], None]] = None) -> Path:
    """Blocks until the mesh is written. Raises RuntimeError with the worker's stderr on failure."""
    binary = worker_binary()
    if binary is None:
        raise RuntimeError("Photogrammetry worker not built. Run: cd mac/Photogrammetry && swift build -c release")
    cmd = [str(binary), str(input_dir), str(output_path), "--detail", detail]
    log.info("photogrammetry: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    err_lines = []
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.strip()
            if not line:
                continue
            err_lines.append(line)
            if line.startswith("progress") and progress:
                try:
                    progress(float(line.split()[1].rstrip("%")) / 100.0, line)
                except (IndexError, ValueError):
                    pass
            elif progress:
                progress(-1.0, line)
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("Photogrammetry timed out")
    if proc.returncode != 0 or not output_path.exists():
        tail = "\n".join(err_lines[-8:])
        raise RuntimeError(f"Photogrammetry failed (exit {proc.returncode}): {tail}")
    if output_path.suffix.lower() == ".obj":
        fix_obj_texture(output_path)
    return output_path


def fix_obj_texture(obj_path: Path) -> Optional[Path]:
    """Model I/O's OBJ export references textures it does not write. Pull the diffuse texture out of the
    USDZ produced alongside and point the MTL at it (dropping the maps we do not need)."""
    import re
    import zipfile

    usdz = obj_path.with_suffix(".usdz")
    mtl = obj_path.with_suffix(".mtl")
    if not usdz.exists() or not mtl.exists():
        return None
    tex_out = obj_path.with_name(obj_path.stem + "_diffuse.png")
    try:
        with zipfile.ZipFile(usdz) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".png") and "tex" in n.lower()]
            if not names:
                return None
            tex_out.write_bytes(z.read(names[0]))
    except zipfile.BadZipFile:
        return None
    lines = []
    for line in mtl.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("map_Kd"):
            lines.append(f"\tmap_Kd {tex_out.name}")
        elif stripped.startswith("map_"):
            continue
        else:
            lines.append(line)
    mtl.write_text("\n".join(lines) + "\n")
    return tex_out
