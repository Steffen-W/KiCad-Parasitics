"""
Calculate trace inductance from shortest-path geometry using bfieldtools.

Receives the path segments (WIRE + VIA elements) extracted from the PCB
network and computes the loop inductance of the trace path.
"""

import logging
import math
from typing import Any

import numpy as np
import trimesh
from bfieldtools.mesh_impedance import self_inductance_matrix
from bfieldtools.utils import find_mesh_boundaries
from scipy.spatial import cKDTree

try:
    from .pcb_types import VIA, WIRE, CuLayer, NetworkElement
except ImportError:
    from pcb_types import VIA, WIRE, CuLayer, NetworkElement

log = logging.getLogger(__name__)

_MESH_TARGET_FACES = 2000
_MAX_MESH_FACES = 5000

# =========================================================================
# Public API
# =========================================================================


def calc_path_inductance(
    path: list[int], network_info: list[NetworkElement], cu_stack: dict[int, CuLayer]
) -> dict[str, Any]:
    """Calculate the inductance of a PCB trace path.

    Parameters
    ----------
    path : list[int]
        Node IDs of the shortest path, e.g. [1, 3, 5, 8].
    network_info : list[dict]
        Element descriptors from extract_network(). Each dict has:
        - WIRE: type, nodes, start (x,y), end (x,y), width, length, layer, layer_name
        - VIA:  type, nodes, position (x,y), drill, length, layer1, layer2,
                layer1_name, layer2_name
    cu_stack : dict
        Copper stackup. cu_stack[layer_id] contains:
        - abs_height (m), thickness (m), name (str), ...

    Returns
    -------
    dict with:
        inductance : float or None – total path inductance in Henry
        segments   : list[dict]    – extracted geometry per segment
        message    : str           – human-readable summary
        _debug_data : dict         – mesh/psi/M data for debug plots
    """
    segments = _extract_segments(path, network_info, cu_stack)
    return _compute_inductance(segments)


# =========================================================================
# Segment extraction from path + network_info
# =========================================================================


def _extract_segments(
    path: list[int], network_info: list[NetworkElement], cu_stack: dict[int, CuLayer]
) -> list[dict[str, Any]]:
    """Extract physical geometry from path node sequence."""
    node_to_elem: dict[frozenset[int], NetworkElement] = {}
    for net_elem in network_info:
        fkey = frozenset(net_elem["nodes"])
        node_to_elem[fkey] = net_elem

    segments = []
    for i in range(len(path) - 1):
        node_a, node_b = path[i], path[i + 1]
        key = frozenset((node_a, node_b))
        elem = node_to_elem.get(key)

        if elem is None:
            log.warning("No element for nodes %s-%s", node_a, node_b)
            continue

        if elem["type"] == WIRE:
            layer_id = elem["layer"]
            z = cu_stack[layer_id]["abs_height"] if layer_id in cu_stack else 0.0
            # Store which node is at which end for direction tracking
            node_start, node_end = elem["nodes"]
            seg = {
                "type": WIRE,
                "start": elem["start"],
                "end": elem["end"],
                "node_at_start": node_start,
                "node_at_end": node_end,
                "path_from": node_a,
                "path_to": node_b,
                "width": elem["width"],
                "length": elem["length"],
                "layer": layer_id,
                "layer_name": elem.get("layer_name", "?"),
                "z": z,
                "thickness": cu_stack[layer_id]["thickness"]
                if layer_id in cu_stack
                else 35e-6,
            }
            if "mid" in elem:
                seg["mid"] = elem["mid"]
            if "radius" in elem:
                seg["radius"] = elem["radius"]
            if "angle" in elem:
                seg["angle"] = elem["angle"]
            if "midline_pts" in elem:
                seg["midline_pts"] = elem["midline_pts"]
            segments.append(seg)

        elif elem["type"] == VIA:
            l1, l2 = elem["layer1"], elem["layer2"]
            z1 = cu_stack[l1]["abs_height"] if l1 in cu_stack else 0.0
            z2 = cu_stack[l2]["abs_height"] if l2 in cu_stack else 0.0
            segments.append(
                {
                    "type": VIA,
                    "position": elem["position"],
                    "drill": elem["drill"],
                    "length": elem["length"],
                    "z_top": max(z1, z2),
                    "z_bot": min(z1, z2),
                    "layer1_name": elem.get("layer1_name", "?"),
                    "layer2_name": elem.get("layer2_name", "?"),
                }
            )

    return segments


# =========================================================================
# Inductance computation
# =========================================================================


def _compute_inductance(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute inductance from physical trace segments.

    1. Build ordered 3D centerline from segments
    2. Close the loop
    3. Build strip mesh (already 3D)
    4. Compute stream function + L via bfieldtools self_inductance_matrix
    """
    # --- Format segment info ---
    segment_lines, path_summary, n_layers = _format_segments(segments)

    # --- Build ordered 3D centerline ---
    centerline, width_per_seg = _build_centerline(segments)

    if len(centerline) < 3:
        return {
            "inductance": None,
            "segments": segments,
            "message": "ERROR: need at least 3 points for a loop",
        }

    # Close the loop – required for inductance calculation even for open coils
    start = np.array(centerline[0])
    end = np.array(centerline[-1])
    gap = np.linalg.norm(end[:2] - start[:2])
    trace_width_prelim = min(width_per_seg)
    if gap > trace_width_prelim * 10:
        log.warning(
            "Closing loop with large gap: gap=%.2f mm, trace width=%.2f mm",
            gap * 1e3,
            trace_width_prelim * 1e3,
        )
    centerline.append(centerline[0])
    width_per_seg.append(width_per_seg[-1])

    # --- Build strip mesh (already 3D) ---
    try:
        mesh = _build_strip_mesh(
            centerline,
            width_per_seg,
        )
        psi = _build_stream_function(mesh)
    except ValueError as e:
        return {
            "inductance": None,
            "segments": segments,
            "message": f"ERROR: {e}",
        }

    # --- Compute inductance ---
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    log.debug("Mesh: %d vertices, %d faces", n_verts, n_faces)

    if n_faces > _MAX_MESH_FACES:
        msg = "\n".join(
            [
                "ERROR: mesh too large",
                f"  {n_faces} faces (max {_MAX_MESH_FACES})",
                f"  {n_verts} vertices",
                "",
                "The trace path is too long or complex.",
                "Try a shorter or simpler path between the pads.",
            ]
        )
        log.warning("%s", msg)
        return {"inductance": None, "segments": segments, "message": msg}

    log.debug("Computing self-inductance matrix ...")
    try:
        M = self_inductance_matrix(mesh, quad_degree=2)
    except MemoryError:
        msg = "\n".join(
            [
                "ERROR: out of memory",
                f"  {n_faces} faces, {n_verts} vertices",
                "",
                "Not enough RAM for self-inductance matrix.",
                "Try a shorter or simpler path between the pads.",
            ]
        )
        log.error("%s", msg)
        return {"inductance": None, "segments": segments, "message": msg}
    L = float(psi @ M @ psi)
    log.info("L = %.2f nH", L * 1e9)

    # --- Build output: result first, summary, then segment details ---
    lines = [
        f"Loop inductance: {L * 1e9:.2f} nH",
        "",
        path_summary,
        "",
        "Simulation model:",
        f"  Loop closed with {gap * 1e3:.2f} mm straight return path",
        f"  Mesh: {n_faces} triangles (max {_MAX_MESH_FACES}), {n_layers} layers",
        "",
        "Segments:",
    ]
    lines.extend(segment_lines)

    result = {
        "inductance": L,
        "segments": segments,
        "message": "\n".join(lines),
        "_debug_data": {
            "centerline": centerline,
            "mesh_vertices": mesh.vertices,
            "mesh_faces": mesh.faces,
            "psi": psi,
            "M": M,
        },
    }
    return result


# =========================================================================
# Centerline builder
# =========================================================================


def _build_centerline(
    segments: list[dict[str, Any]],
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """Build ordered 3D centerline from path segments.

    VIAs produce two points at the same (x,y) with z_from and z_to,
    creating a vertical strip segment in the mesh.

    Returns
    -------
    centerline : list of (x, y, z) tuples in meters (not yet closed)
    width_per_seg : list of widths (m), one per centerline segment
    """
    centerline: list[tuple[float, float, float]] = []
    width_per_seg: list[float] = []
    last_width = 0.0
    current_z = None

    # Pre-scan: find the first wire width as fallback
    for seg in segments:
        if seg["type"] == WIRE and seg.get("width", 0.0) > 0:
            last_width = seg["width"]
            break

    for seg in segments:
        if seg["type"] == VIA:
            pos = tuple(seg["position"])
            via_width = _neighbor_width(seg, segments, last_width)

            # Determine VIA direction based on current_z
            z_top, z_bot = seg["z_top"], seg["z_bot"]
            if current_z is not None:
                if abs(current_z - z_top) < abs(current_z - z_bot):
                    z_from, z_to = z_top, z_bot
                else:
                    z_from, z_to = z_bot, z_top
            else:
                z_from, z_to = z_top, z_bot

            if centerline:
                last_xy = np.array(centerline[-1][:2])
                gap = np.linalg.norm(np.array(pos) - last_xy)
                if gap > 1e-6:
                    # Bridge to VIA position at z_from
                    centerline.append((pos[0], pos[1], z_from))
                    width_per_seg.append(via_width)
                # Add VIA endpoint at z_to
                centerline.append((pos[0], pos[1], z_to))
                width_per_seg.append(via_width)
            else:
                centerline.append((pos[0], pos[1], z_from))
                centerline.append((pos[0], pos[1], z_to))
                width_per_seg.append(via_width)

            current_z = z_to
            continue

        if seg["type"] != WIRE:
            continue

        w = seg.get("width", 0.0)
        if w <= 0:
            log.debug("Skipping WIRE with width=0 (id=%s)", seg.get("id", "?"))
            continue

        last_width = w
        z = seg["z"]
        s = np.array(seg["start"])
        e = np.array(seg["end"])

        # Determine traversal direction from path node info
        if seg["path_from"] == seg["node_at_start"]:
            entry, exit_ = s, e
        else:
            entry, exit_ = e, s

        if not centerline:
            centerline.append((entry[0], entry[1], z))
            current_z = z
        else:
            last_xy = np.array(centerline[-1][:2])
            gap = np.linalg.norm(entry - last_xy)
            if gap > 1e-6:
                log.debug(
                    "Gap of %.3f mm between centerline end (%.2f,%.2f) "
                    "and next segment entry (%.2f,%.2f) – inserting bridge (w=%.2f mm)",
                    gap * 1e3,
                    last_xy[0] * 1e3,
                    last_xy[1] * 1e3,
                    entry[0] * 1e3,
                    entry[1] * 1e3,
                    w * 1e3,
                )
                centerline.append((entry[0], entry[1], z))
                width_per_seg.append(w)

        midline = seg.get("midline_pts")
        if midline is not None and len(midline) > 2:
            # Reverse midline if traversal direction is opposite to stored order
            if seg["path_from"] != seg["node_at_start"]:
                midline = midline[::-1]
            # Use pre-computed arc midline points (skip first, it's the entry)
            for pt in midline[1:]:
                centerline.append((pt[0], pt[1], z))
            width_per_seg.extend([w] * (len(midline) - 1))
        else:
            centerline.append((exit_[0], exit_[1], z))
            width_per_seg.append(w)

        current_z = z

    return centerline, width_per_seg


def _neighbor_width(
    via_seg: dict[str, Any], segments: list[dict[str, Any]], fallback: float
) -> float:
    """Get bridge width for a VIA from its neighboring WIRE segments."""
    idx = segments.index(via_seg)
    widths = []
    # Look at previous segment
    if idx > 0 and segments[idx - 1]["type"] == WIRE:
        w = segments[idx - 1].get("width", 0.0)
        if w > 0:
            widths.append(w)
    # Look at next segment
    if idx < len(segments) - 1 and segments[idx + 1]["type"] == WIRE:
        w = segments[idx + 1].get("width", 0.0)
        if w > 0:
            widths.append(w)
    return float(min(widths)) if widths else fallback


# =========================================================================
# Strip mesh builder
# =========================================================================


def _build_strip_mesh(
    centerline: list[tuple[float, float, float]],
    width_per_seg: list[float],
    max_faces: int = _MESH_TARGET_FACES,
    n_width: int = 4,
) -> trimesh.Trimesh:
    """Build a 3D strip mesh from a closed centerline.

    The mesh is a quad-strip (split into triangles) that follows the 3D
    centerline.  Each layer segment gets its correct z immediately — no
    2D projection or z-warping needed.

    Parameters
    ----------
    centerline : (N, 3) array – closed center-line in meters (first == last).
    width_per_seg : (N-1,) array – trace width per segment in meters.
    max_faces : int – target maximum faces (subdivide long segments to reach it).
    n_width : int – number of quads across the strip width.

    Returns
    -------
    trimesh.Trimesh – 3D mesh in meters with ring topology (2 boundaries).
    """
    pts = np.array(centerline)
    n_pts = len(pts)
    if n_pts < 3:
        raise ValueError(f"Need >= 3 centerline points, got {n_pts}")

    # The centerline is closed (first == last), so we have n_pts-1 segments.
    # Remove the duplicate closing point for processing; we'll wrap around.
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n_pts = len(pts)
    widths = np.array(width_per_seg[:n_pts])
    if len(widths) < n_pts:
        widths = np.pad(widths, (0, n_pts - len(widths)), constant_values=widths[-1])

    # --- Subdivide long segments to reach target face count ---
    # Each quad = 2 faces, n_width quads across => 2*n_width faces per ring row
    # We want total faces <= max_faces => n_rows <= max_faces / (2*n_width)
    max_rows = max_faces // (2 * n_width)

    diffs = np.diff(np.vstack([pts, pts[0:1]]), axis=0)
    seg_lens_xy = np.linalg.norm(diffs[:, :2], axis=1)
    # Total length only counts xy-distance (VIA segments have zero xy-length)
    total_xy_len = seg_lens_xy.sum()
    if total_xy_len < 1e-15:
        raise ValueError("Centerline has zero xy-length")

    # Subdivide each segment proportionally (skip VIA segments with zero xy-len)
    new_pts = []
    new_widths = []
    # Reserve one row per VIA segment
    n_via_segs = np.sum(seg_lens_xy < 1e-15)
    target_rows = max(n_pts, max_rows - int(n_via_segs))
    for i in range(n_pts):
        j = (i + 1) % n_pts
        if seg_lens_xy[i] < 1e-15:
            # VIA segment — keep as single segment, don't subdivide
            new_pts.append(pts[i].copy())
            new_widths.append(widths[i])
        else:
            n_sub = max(1, int(target_rows * seg_lens_xy[i] / total_xy_len))
            for k in range(n_sub):
                t = k / n_sub
                new_pts.append(pts[i] * (1 - t) + pts[j] * t)
                new_widths.append(widths[i] * (1 - t) + widths[j] * t)

    pts = np.array(new_pts)
    widths = np.array(new_widths)
    n_pts = len(pts)

    log.debug("Strip mesh: %d centerline points, %d width quads", n_pts, n_width)

    # --- Compute per-point normals perpendicular to path in xy-plane ---
    # Use averaged incoming+outgoing tangent normals to avoid degeneracy at corners
    seg_normals = np.zeros((n_pts, 3))
    for i in range(n_pts):
        j = (i + 1) % n_pts
        d = pts[j] - pts[i]
        dxy = math.hypot(d[0], d[1])
        if dxy > 1e-15:
            seg_normals[i] = np.array([-d[1] / dxy, d[0] / dxy, 0.0])

    # Average incoming and outgoing normals at each vertex
    normals = np.zeros((n_pts, 3))
    last_valid = np.array([1.0, 0.0, 0.0])
    for i in range(n_pts):
        prev = (i - 1) % n_pts
        n_in = seg_normals[prev]
        n_out = seg_normals[i]
        has_in = np.linalg.norm(n_in) > 0.5
        has_out = np.linalg.norm(n_out) > 0.5
        if has_in and has_out:
            avg = n_in + n_out
            length = np.linalg.norm(avg)
            normals[i] = avg / length if length > 1e-15 else n_out
        elif has_out:
            normals[i] = n_out
        elif has_in:
            normals[i] = n_in
        else:
            normals[i] = last_valid
        last_valid = normals[i]

    # --- Build vertex grid: n_pts × (n_width+1) ---
    n_cols = n_width + 1
    verts = np.zeros((n_pts * n_cols, 3))
    for i in range(n_pts):
        half_w = widths[i] / 2
        for c in range(n_cols):
            offset = -half_w + c * (widths[i] / n_width)
            verts[i * n_cols + c] = pts[i] + offset * normals[i]

    # --- Build faces: 2 triangles per quad ---
    faces = []
    for i in range(n_pts):
        j = (i + 1) % n_pts
        for c in range(n_width):
            v00 = i * n_cols + c
            v01 = i * n_cols + c + 1
            v10 = j * n_cols + c
            v11 = j * n_cols + c + 1
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])

    faces_arr = np.array(faces)
    n_faces = len(faces_arr)
    log.debug("Strip mesh: %d vertices, %d faces", len(verts), n_faces)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces_arr, process=False)
    return mesh


def _build_stream_function(mesh: trimesh.Trimesh) -> "np.ndarray[Any, Any]":
    """Stream function psi for a single-hole ring mesh.

    psi = 0 on outer boundary, 1 on inner boundary, interpolated for interior.
    L = psi^T @ M @ psi  (I = 1 A).
    """
    boundaries, inner_verts = find_mesh_boundaries(mesh)
    if len(boundaries) != 2:
        raise ValueError(f"Expected ring mesh with 2 boundaries, got {len(boundaries)}")

    if len(boundaries[0]) >= len(boundaries[1]):
        outer_bnd, inner_bnd = boundaries[0], boundaries[1]
    else:
        outer_bnd, inner_bnd = boundaries[1], boundaries[0]

    psi = np.zeros(len(mesh.vertices))
    psi[inner_bnd] = 1.0

    if len(inner_verts) > 0:
        xy = mesh.vertices[:, :2]
        d_out = cKDTree(xy[outer_bnd]).query(xy[inner_verts])[0]
        d_in = cKDTree(xy[inner_bnd]).query(xy[inner_verts])[0]
        psi[inner_verts] = d_out / (d_out + d_in)

    return psi


# =========================================================================
# Formatting
# =========================================================================


def _format_segments(segments: list[dict[str, Any]]) -> tuple[list[str], str, int]:
    """Format segment info as human-readable lines."""
    lines = []
    total_wire_length = 0.0
    layers_used = set()

    for seg in segments:
        if seg["type"] == WIRE:
            sx, sy = seg["start"]
            ex, ey = seg["end"]
            l_mm = seg.get("length", 0) * 1e3
            lines.append(
                f"  WIRE  ({sx * 1e3:.1f},{sy * 1e3:.1f})->({ex * 1e3:.1f},{ey * 1e3:.1f})  "
                f"w={seg['width'] * 1e3:.2f}mm  l={l_mm:.2f}mm  {seg['layer_name']}"
            )
            total_wire_length += seg.get("length", 0)
            layers_used.add(seg["layer_name"])
        elif seg["type"] == VIA:
            px, py = seg["position"]
            lines.append(
                f"  VIA   ({px * 1e3:.1f},{py * 1e3:.1f})  "
                f"drill={seg['drill'] * 1e3:.2f}mm  "
                f"{seg['layer1_name']}->{seg['layer2_name']}"
            )
            layers_used.add(seg["layer1_name"])
            layers_used.add(seg["layer2_name"])

    summary = (
        f"Path: {len(segments)} segments, "
        f"{total_wire_length * 1e3:.2f} mm, "
        f"layers: {', '.join(sorted(layers_used))}"
    )
    return lines, summary, len(layers_used)


# =========================================================================
# Standalone test with real PCB data
# =========================================================================
