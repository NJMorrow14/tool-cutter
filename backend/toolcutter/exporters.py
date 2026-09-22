"""Write the finished layout as SVG (mat-sized, mm), DXF (depth layers) or STL (foam block with pockets)."""
from __future__ import annotations

import io
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

Ring = List[List[float]]

# Distinct, laser-software-friendly colors, assigned by ascending pocket depth.
DEPTH_PALETTE = [
    "#000000", "#0000ff", "#ff0000", "#00b000", "#ff8000", "#a000c0",
    "#00a0a0", "#c08000", "#ff00ff", "#806040", "#4060ff", "#c00060",
]
LABEL_COLOR = "#8a8a8a"
MAT_COLOR = "#000000"


def _fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def depth_color_map(depths: Sequence[Optional[float]]) -> Dict[Optional[float], str]:
    uniq = sorted({round(d, 2) for d in depths if d is not None})
    cmap: Dict[Optional[float], str] = {}
    for i, d in enumerate(uniq):
        cmap[d] = DEPTH_PALETTE[(i + 1) % len(DEPTH_PALETTE)] if len(uniq) > 1 else "#ff0000"
    cmap[None] = "#ff0000"
    return cmap


def _ring_path(ring: Ring) -> str:
    if len(ring) < 3:
        return ""
    parts = [f"M{_fmt(ring[0][0])} {_fmt(ring[0][1])}"]
    parts += [f"L{_fmt(x)} {_fmt(y)}" for x, y in ring[1:]]
    parts.append("Z")
    return " ".join(parts)


def layout_to_svg(layout: Dict, *, include_mat: bool = True, include_labels: bool = True,
                  fill_mode: str = "none", stroke_mm: float = 0.2, title: str = "Tool foam layout") -> str:
    W = float(layout["mat"]["width_mm"])
    H = float(layout["mat"]["height_mm"])
    tools = layout["tools"]
    cmap = depth_color_map([t.get("depth_mm") for t in tools])
    out = io.StringIO()
    out.write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
              f'width="{_fmt(W)}mm" height="{_fmt(H)}mm" viewBox="0 0 {_fmt(W)} {_fmt(H)}">\n')
    out.write(f"  <title>{title}</title>\n")
    out.write(f'  <desc>Mat {_fmt(W)} x {_fmt(H)} mm. Units are millimetres. Tool paths are grouped by pocket depth; '
              f'stroke color identifies the depth layer.</desc>\n')
    if include_mat:
        out.write(f'  <g id="mat" inkscape:label="mat" fill="none" stroke="{MAT_COLOR}" stroke-width="{_fmt(stroke_mm)}" '
                  f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">\n')
        out.write(f'    <rect x="0" y="0" width="{_fmt(W)}" height="{_fmt(H)}"/>\n')
        out.write("  </g>\n")
    out.write('  <g id="cutouts">\n')
    for t in tools:
        d = t.get("depth_mm")
        key = round(d, 2) if d is not None else None
        color = cmap.get(key, "#ff0000")
        path = " ".join(p for p in (_ring_path(r) for r in t["rings"]) if p)
        if not path:
            continue
        fill = color if fill_mode == "fill" else "none"
        stroke = "none" if fill_mode == "fill" else color
        depth_attr = f' data-depth-mm="{_fmt(d)}"' if d is not None else ""
        name = _xml_escape(t.get("name") or t["id"])
        out.write(f'    <path id="tool-{_xml_escape(str(t["id"]))}" data-name="{name}"{depth_attr} '
                  f'fill="{fill}" fill-rule="evenodd" stroke="{stroke}" stroke-width="{_fmt(stroke_mm)}" '
                  f'stroke-linejoin="round" d="{path}"/>\n')
    out.write("  </g>\n")
    if include_labels:
        out.write(f'  <g id="labels" fill="{LABEL_COLOR}" stroke="none" font-family="Helvetica, Arial, sans-serif">\n')
        for t in tools:
            c = t.get("centroid_mm")
            if not c or not t["rings"]:
                continue
            bbox = t.get("bbox_mm") or [0, 0, 0, 0]
            size = max(2.5, min(6.0, (bbox[2] - bbox[0]) / 12.0))
            label = _xml_escape(t.get("name") or t["id"])
            if t.get("depth_mm") is not None:
                label += f" · {_fmt(t['depth_mm'])} mm"
            out.write(f'    <text x="{_fmt(c[0])}" y="{_fmt(c[1])}" font-size="{_fmt(size)}" '
                      f'text-anchor="middle" dominant-baseline="middle">{label}</text>\n')
        # legend along the bottom edge (inside the mat)
        legend = [(d, col) for d, col in cmap.items() if d is not None]
        if legend:
            y = H - 3.0
            x = 3.0
            for d, col in legend:
                out.write(f'    <rect x="{_fmt(x)}" y="{_fmt(y - 2.2)}" width="3" height="3" fill="{col}"/>\n')
                out.write(f'    <text x="{_fmt(x + 4)}" y="{_fmt(y)}" font-size="2.6">depth {_fmt(d)} mm</text>\n')
                x += 30.0
        out.write("  </g>\n")
    out.write("</svg>\n")
    return out.getvalue()


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&apos;"))


def layout_to_dxf(layout: Dict, *, include_mat: bool = True, include_labels: bool = True) -> bytes:
    import ezdxf  # type: ignore
    from ezdxf import units

    W = float(layout["mat"]["width_mm"])
    H = float(layout["mat"]["height_mm"])
    doc = ezdxf.new("R2010", setup=True)
    doc.units = units.MM
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    aci_cycle = [1, 5, 3, 30, 6, 4, 2, 150, 210, 40]
    if include_mat:
        doc.layers.add("MAT", color=7)
        msp.add_lwpolyline([(0, 0), (W, 0), (W, H), (0, H)], close=True, dxfattribs={"layer": "MAT"})
    depths = sorted({round(t["depth_mm"], 2) for t in layout["tools"] if t.get("depth_mm") is not None})
    layer_for_depth: Dict[Optional[float], str] = {}
    for i, d in enumerate(depths):
        name = f"POCKET_{_fmt(d).replace('.', 'p')}MM"
        doc.layers.add(name, color=aci_cycle[i % len(aci_cycle)])
        layer_for_depth[d] = name
    doc.layers.add("CUTOUTS", color=1)
    if include_labels:
        doc.layers.add("LABELS", color=8)
    # DXF y axis points up: flip so the drawing matches the SVG when viewed from above
    for t in layout["tools"]:
        d = t.get("depth_mm")
        layer = layer_for_depth.get(round(d, 2)) if d is not None else "CUTOUTS"
        for ring in t["rings"]:
            pts = [(x, H - y) for x, y in ring]
            if len(pts) >= 3:
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer or "CUTOUTS"})
        if include_labels and t.get("centroid_mm"):
            cx, cy = t["centroid_mm"]
            label = t.get("name") or str(t["id"])
            if d is not None:
                label += f" {_fmt(d)}mm"
            msp.add_text(label, height=3.0, dxfattribs={"layer": "LABELS"}).set_placement((cx, H - cy), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def layout_to_stl(layout: Dict, *, mat_thickness_mm: float = 30.0, floor_min_mm: float = 2.0,
                  through_if_unknown: bool = True) -> bytes:
    """Foam block (W x H x thickness) with each tool pocket subtracted to its depth."""
    import trimesh  # type: ignore
    from shapely import affinity
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    W = float(layout["mat"]["width_mm"])
    H = float(layout["mat"]["height_mm"])
    T = float(mat_thickness_mm)
    slab = trimesh.creation.box((W, H, T))
    slab.apply_translation((W / 2.0, H / 2.0, T / 2.0))
    pockets = []
    for t in layout["tools"]:
        geom = t.get("shapely")
        if geom is None or geom.is_empty:
            continue
        d = t.get("depth_mm")
        if d is None:
            if not through_if_unknown:
                continue
            depth = T
        else:
            depth = float(d)
        through = depth >= T - 0.25
        depth = T if through else min(max(depth, 0.5), T - floor_min_mm)
        # SVG y-down -> 3D y-up
        geom = affinity.scale(geom, xfact=1.0, yfact=-1.0, origin=(0, 0))
        geom = affinity.translate(geom, yoff=H)
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            if not isinstance(poly, Polygon) or poly.area < 1.0:
                continue
            extra = 2.0
            solid = trimesh.creation.extrude_polygon(poly, depth + extra)
            solid.apply_translation((0.0, 0.0, T - depth))
            pockets.append(solid)
    if pockets:
        result = trimesh.boolean.difference([slab] + pockets, engine="manifold")
        if isinstance(result, trimesh.Scene):
            result = trimesh.util.concatenate(list(result.geometry.values()))
    else:
        result = slab
    return result.export(file_type="stl")


# ----------------------------------------------------------------------------- tools as 3D bodies

def _rotate_translate(xy: np.ndarray, centroid, rotation_deg: float, offset) -> np.ndarray:
    """Same transform as geometry.process_outline: rotate about the outline centroid, then translate."""
    th = math.radians(rotation_deg or 0.0)
    c, s = math.cos(th), math.sin(th)
    cx, cy = centroid
    x = xy[:, 0] - cx
    y = xy[:, 1] - cy
    out = np.empty_like(xy)
    out[:, 0] = cx + x * c - y * s + offset[0]
    out[:, 1] = cy + x * s + y * c + offset[1]
    return out


def heightfield_tool_mesh(mask: np.ndarray, height_mm: np.ndarray, mm_per_px: float, *, centroid_mm, rotation_deg: float,
                          offset_mm, z_bottom: float, max_cells: int = 12000, min_height: float = 0.4):
    """Watertight-ish block model of a tool from its scanned height map (top surface + walls + flat bottom)."""
    import trimesh  # type: ignore

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1]
    h = np.nan_to_num(height_mm[y0:y1, x0:x1], nan=0.0).astype(np.float64)
    k = max(1, int(math.ceil(math.sqrt(m.sum() / float(max_cells)))))
    if k > 1:
        H, W = m.shape
        ph, pw = (-H) % k, (-W) % k
        m = np.pad(m, ((0, ph), (0, pw)))
        h = np.pad(h, ((0, ph), (0, pw)))
        Hd, Wd = m.shape[0] // k, m.shape[1] // k
        m = m.reshape(Hd, k, Wd, k).mean(axis=(1, 3)) >= 0.5
        h = h.reshape(Hd, k, Wd, k).max(axis=(1, 3))
    Hd, Wd = m.shape
    if not m.any():
        return None
    h = np.where(m, np.maximum(h, min_height), 0.0)

    # corner heights = max of the adjacent in-mask cells
    hp = np.pad(h, 1)
    corner = np.maximum.reduce([hp[:-1, :-1], hp[:-1, 1:], hp[1:, :-1], hp[1:, 1:]])  # (Hd+1, Wd+1)

    # corner grid in mm (image px -> mm, y down)
    jj, ii = np.meshgrid(np.arange(Wd + 1), np.arange(Hd + 1))
    xy = np.column_stack([(x0 + jj.ravel() * k) * mm_per_px, (y0 + ii.ravel() * k) * mm_per_px])
    xy = _rotate_translate(xy, centroid_mm, rotation_deg, offset_mm)
    n_c = (Hd + 1) * (Wd + 1)
    top = np.column_stack([xy, z_bottom + corner.ravel()])
    bot = np.column_stack([xy, np.full(n_c, z_bottom)])
    verts = np.vstack([top, bot])
    cid = lambda i, j: i * (Wd + 1) + j  # noqa: E731

    faces = []
    ci, cj = np.nonzero(m)
    for i, j in zip(ci, cj):
        a, b, c, d = cid(i, j), cid(i, j + 1), cid(i + 1, j + 1), cid(i + 1, j)
        faces.append([a, c, b]); faces.append([a, d, c])                     # top (normal +z after y flip)
        faces.append([a + n_c, b + n_c, c + n_c]); faces.append([a + n_c, c + n_c, d + n_c])  # bottom
        # walls where the neighbour is empty
        if i == 0 or not m[i - 1, j]:
            faces.append([a, b, b + n_c]); faces.append([a, b + n_c, a + n_c])
        if i == Hd - 1 or not m[i + 1, j]:
            faces.append([d, d + n_c, c + n_c]); faces.append([d, c + n_c, c])
        if j == 0 or not m[i, j - 1]:
            faces.append([a, a + n_c, d + n_c]); faces.append([a, d + n_c, d])
        if j == Wd - 1 or not m[i, j + 1]:
            faces.append([b, c, c + n_c]); faces.append([b, c + n_c, b + n_c])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def extruded_tool_mesh(polygon_mm, *, centroid_mm, rotation_deg: float, offset_mm, z_bottom: float, thickness: float):
    """Fallback when no height map is available: the outline extruded to the tool thickness."""
    import trimesh  # type: ignore
    from shapely.geometry import Polygon

    pts = np.asarray(polygon_mm, dtype=np.float64)
    if len(pts) < 3:
        return None
    pts = _rotate_translate(pts, centroid_mm, rotation_deg, offset_mm)
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    solid = trimesh.creation.extrude_polygon(poly, max(thickness, 0.5))
    solid.apply_translation((0, 0, z_bottom))
    return solid


def layout_tools_to_stl(layout: Dict, *, mat_thickness_mm: float, geometry_for_tool) -> bytes:
    """All included tools as one STL, each sitting in its pocket. `geometry_for_tool(tool) -> (mask, height, mm_per_px) | None`."""
    import trimesh  # type: ignore

    W = float(layout["mat"]["width_mm"])
    H = float(layout["mat"]["height_mm"])
    T = float(mat_thickness_mm)
    parts = []
    for t in layout["tools"]:
        raw = t.get("raw") or {}
        d = t.get("depth_mm")
        depth = T if d is None or d >= T - 0.25 else min(max(float(d), 0.5), T - 2.0)
        z_bottom = T - depth
        centroid = t.get("source_centroid_mm") or (0.0, 0.0)
        rot = float(raw.get("rotation_deg") or 0.0)
        off = raw.get("offset_mm") or {}
        off = (float(off.get("x") or 0.0), float(off.get("y") or 0.0))
        geo = geometry_for_tool(raw)
        mesh = None
        if geo is not None:
            mask, height, mm_per_px = geo
            mesh = heightfield_tool_mesh(mask, height, mm_per_px, centroid_mm=centroid, rotation_deg=rot, offset_mm=off, z_bottom=z_bottom)
        if mesh is None:
            thick = raw.get("thickness_mm") or d or 10.0
            mesh = extruded_tool_mesh(raw.get("polygon_mm") or [], centroid_mm=centroid, rotation_deg=rot, offset_mm=off,
                                      z_bottom=z_bottom, thickness=float(thick))
        if mesh is None:
            continue
        v = mesh.vertices.copy()
        if layout.get("mirror"):
            v[:, 0] = W - v[:, 0]
            mesh.invert()
        v[:, 1] = H - v[:, 1]           # SVG y-down -> 3D y-up
        mesh.vertices = v
        mesh.invert()                   # the y flip mirrors winding; restore outward normals
        parts.append(mesh)
    if not parts:
        return trimesh.Trimesh().export(file_type="stl")
    return trimesh.util.concatenate(parts).export(file_type="stl")
