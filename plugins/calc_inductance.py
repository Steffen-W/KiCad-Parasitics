"""
Calculate trace inductance from shortest-path geometry using bfieldtools.

Receives the path segments (WIRE + VIA elements) extracted from the PCB
network and computes the loop inductance of the trace path.
"""

import sys
import numpy as np

HAS_BFIELDTOOLS = False
_IMPORT_ERROR = ""
_PYTHON_EXE = sys.executable
try:
    import trimesh
    from shapely.geometry import LineString
    import triangle as tr
    from scipy.spatial import cKDTree
    from bfieldtools.mesh_impedance import self_inductance_matrix
    from bfieldtools.utils import find_mesh_boundaries

    HAS_BFIELDTOOLS = True
except ImportError as _e:
    _IMPORT_ERROR = str(_e)


# =========================================================================
# Public API
# =========================================================================


def calc_path_inductance(path, network_info, cu_stack, debug_print=None):
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
    debug_print : callable, optional
        Print function for debug output.

    Returns
    -------
    dict with:
        inductance : float or None – total path inductance in Henry
        segments   : list[dict]    – extracted geometry per segment
        message    : str           – human-readable summary
    """
    if debug_print is None:
        debug_print = print

    segments = _extract_segments(path, network_info, cu_stack, debug_print)
    return _compute_inductance(segments, debug_print)


# =========================================================================
# Segment extraction from path + network_info
# =========================================================================


def _extract_segments(path, network_info, cu_stack, debug_print):
    """Extract physical geometry from path node sequence."""
    node_to_elem = {}
    for elem in network_info:
        key = frozenset(elem["nodes"])
        node_to_elem[key] = elem

    segments = []
    for i in range(len(path) - 1):
        node_a, node_b = path[i], path[i + 1]
        key = frozenset((node_a, node_b))
        elem = node_to_elem.get(key)

        if elem is None:
            debug_print(
                f"[calc_inductance] WARNING: no element for nodes {node_a}-{node_b}"
            )
            continue

        if elem["type"] == "WIRE":
            layer_id = elem["layer"]
            z = cu_stack[layer_id]["abs_height"] if layer_id in cu_stack else 0.0
            # Store which node is at which end for direction tracking
            node_start, node_end = elem["nodes"]
            segments.append(
                {
                    "type": "WIRE",
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
                    "thickness": cu_stack.get(layer_id, {}).get("thickness", 35e-6),
                }
            )

        elif elem["type"] == "VIA":
            z1 = cu_stack.get(elem["layer1"], {}).get("abs_height", 0.0)
            z2 = cu_stack.get(elem["layer2"], {}).get("abs_height", 0.0)
            segments.append(
                {
                    "type": "VIA",
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


def _compute_inductance(segments, debug_print=None):
    """Compute inductance from physical trace segments.

    1. Build ordered 2D centerline + z-profile from segments
    2. Close the loop
    3. Mesh the ring (shapely + triangle)
    4. Assign z per vertex from layer info
    5. Compute L via bfieldtools self_inductance_matrix
    """
    if debug_print is None:
        debug_print = print

    # --- Format segment info ---
    segment_lines, path_summary = _format_segments(segments)

    if not HAS_BFIELDTOOLS:
        lines = [
            f"ERROR: {_IMPORT_ERROR}",
            "",
            "Install missing packages with:",
            f'  {_PYTHON_EXE} -m pip install bfieldtools shapely triangle trimesh "scipy<1.14"',
            "",
            "Alternatively, use the KiCad IPC API which manages dependencies automatically:",
            "  KiCad -> Settings -> Plugins -> Enable KiCad API, then restart KiCad.",
        ]
        msg = "\n".join(lines)
        debug_print(f"[calc_inductance]\n{msg}")
        return {"inductance": None, "segments": segments, "message": msg}

    # --- Build ordered 2D centerline and z-profile ---
    centerline_xy, z_per_seg = _build_centerline(segments)

    if len(centerline_xy) < 3:
        return {
            "inductance": None,
            "segments": segments,
            "message": "ERROR: need at least 3 points for a loop",
        }

    # Close the loop
    start = np.array(centerline_xy[0])
    end = np.array(centerline_xy[-1])
    gap = np.linalg.norm(end - start)
    centerline_xy.append(centerline_xy[0])
    z_per_seg.append(z_per_seg[-1])

    # Trace width = minimum across all wire segments
    wire_widths = [s["width"] for s in segments if s["type"] == "WIRE"]
    trace_width = min(wire_widths)

    # --- Mesh the 2D ring ---
    centerline_mm = np.array(centerline_xy) * 1e3
    mesh_flat = _trace_path_to_mesh(
        centerline_mm, trace_width * 1e3, max_tri_area_mm2=0.02
    )
    psi = _build_stream_function(mesh_flat)

    # --- Check if multi-layer (needs z-warping) ---
    unique_z = sorted(set(z_per_seg))

    if len(unique_z) > 1:
        centerline_m = np.array(centerline_xy)
        t_params = _compute_arc_params(mesh_flat.vertices[:, :2], centerline_m)
        mesh_3d = _assign_z_from_segments(mesh_flat, t_params, centerline_m, z_per_seg)
    else:
        verts = mesh_flat.vertices.copy()
        verts[:, 2] = unique_z[0]
        mesh_3d = trimesh.Trimesh(vertices=verts, faces=mesh_flat.faces, process=False)

    # --- Compute inductance ---
    debug_print("[calc_inductance] Computing self-inductance matrix ...")
    M = self_inductance_matrix(mesh_3d, quad_degree=2)
    L = float(psi @ M @ psi)
    debug_print(f"[calc_inductance] L = {L * 1e9:.2f} nH")

    # --- Build output: result first, summary, then segment details ---
    lines = [
        f"Loop inductance: {L * 1e9:.2f} nH",
        "",
        path_summary,
        f"  loop closure gap: {gap * 1e3:.2f} mm",
        f"  mesh: {len(mesh_flat.vertices)} vertices, {len(mesh_flat.faces)} faces",
        "",
        "Segments:",
    ]
    lines.extend(segment_lines)

    return {"inductance": L, "segments": segments, "message": "\n".join(lines)}


# =========================================================================
# Centerline builder
# =========================================================================


def _build_centerline(segments):
    """Build ordered 2D centerline and z per segment from path segments.

    Returns
    -------
    centerline_xy : list of (x, y) tuples in meters (not yet closed)
    z_per_seg : list of z-values, one per centerline segment
    """
    centerline_xy = []
    z_per_seg = []

    for seg in segments:
        if seg["type"] == "VIA":
            # VIA doesn't add xy extent, just a z-transition
            continue

        if seg["type"] != "WIRE":
            continue

        s = np.array(seg["start"])
        e = np.array(seg["end"])

        # Determine traversal direction from path node info
        if seg["path_from"] == seg["node_at_start"]:
            entry, exit_ = s, e
        else:
            entry, exit_ = e, s

        if not centerline_xy:
            centerline_xy.append(tuple(entry))
        centerline_xy.append(tuple(exit_))
        z_per_seg.append(seg["z"])

    return centerline_xy, z_per_seg


# =========================================================================
# Mesh helpers (adapted from test/test_bfieldtools_coil.py)
# =========================================================================


def _trace_path_to_mesh(path_xy_mm, trace_width_mm, max_tri_area_mm2=0.02):
    """Triangulated 3D mesh from a closed trace center-line.

    Parameters
    ----------
    path_xy_mm : (N, 2) array – closed center-line in mm (first == last point).
    trace_width_mm : float – trace width in mm.
    max_tri_area_mm2 : float – max triangle area for meshing (mm²).

    Returns
    -------
    trimesh.Trimesh in meters (z = 0).
    """
    line = LineString(path_xy_mm)
    poly = line.buffer(trace_width_mm / 2, cap_style="flat", join_style="mitre")

    outer_coords = np.array(poly.exterior.coords[:-1])
    inner_coords = np.array(poly.interiors[0].coords[:-1])
    n_outer, n_inner = len(outer_coords), len(inner_coords)

    vertices = np.vstack([outer_coords, inner_coords])
    outer_seg = np.array([[i, (i + 1) % n_outer] for i in range(n_outer)])
    inner_seg = np.array(
        [[n_outer + i, n_outer + (i + 1) % n_inner] for i in range(n_inner)]
    )
    seg_arr = np.vstack([outer_seg, inner_seg])

    centroid = (
        np.mean(path_xy_mm[:-1], axis=0)
        if hasattr(path_xy_mm, "__len__")
        else np.mean(np.array(path_xy_mm)[:-1], axis=0)
    )
    if isinstance(centroid, np.ndarray) and centroid.ndim == 1:
        centroid = centroid.reshape(1, -1)
    else:
        centroid = np.array([centroid])

    tri_input = {"vertices": vertices, "segments": seg_arr, "holes": centroid}
    tri_result = tr.triangulate(tri_input, f"pq30a{max_tri_area_mm2}")

    verts_3d = np.column_stack(
        [
            tri_result["vertices"] * 1e-3,
            np.zeros(len(tri_result["vertices"])),
        ]
    )
    return trimesh.Trimesh(
        vertices=verts_3d, faces=tri_result["triangles"], process=False
    )


def _build_stream_function(mesh):
    """Stream function psi for a single-hole ring mesh.

    psi = 0 on outer boundary, 1 on inner boundary, interpolated for interior.
    L = psi^T @ M @ psi  (I = 1 A).
    """
    boundaries, inner_verts = find_mesh_boundaries(mesh)
    assert len(boundaries) == 2, f"Expected ring (2 boundaries), got {len(boundaries)}"

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


def _compute_arc_params(verts_xy_m, path_xy_m):
    """Map each mesh vertex to arc-length parameter t in [0, 1)."""
    path = np.array(path_xy_m)
    if np.allclose(path[0], path[-1]):
        path = path[:-1]
    n_seg = len(path)

    p0s = path
    p1s = np.roll(path, -1, axis=0)
    ds = p1s - p0s
    seg_lens = np.linalg.norm(ds, axis=1)
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    t_out = np.zeros(len(verts_xy_m))
    for vi, pt in enumerate(verts_xy_m):
        best_dist = np.inf
        best_t = 0.0
        for si in range(n_seg):
            if seg_lens[si] == 0:
                continue
            u = np.clip(np.dot(pt - p0s[si], ds[si]) / seg_lens[si] ** 2, 0, 1)
            proj = p0s[si] + u * ds[si]
            dist = np.linalg.norm(pt - proj)
            if dist < best_dist:
                best_dist = dist
                best_t = (cum_len[si] + u * seg_lens[si]) / total_len
        t_out[vi] = best_t

    return t_out


def _assign_z_from_segments(mesh_flat, t_params, centerline_m, z_per_seg):
    """Assign z-height to mesh vertices based on arc-length parameter.

    Each centerline segment [i]→[i+1] covers an arc-length range and has z_per_seg[i].
    Vertices in that range get that z-height. Via transitions are smoothed.
    """
    path = np.array(centerline_m)
    if np.allclose(path[0], path[-1]):
        path = path[:-1]
    n_seg = len(path)

    seg_lens = np.linalg.norm(np.diff(np.vstack([path, path[0:1]]), axis=0), axis=1)
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    # t-boundaries for each centerline segment
    t_bounds = cum_len / total_len  # [0, t1, t2, ..., 1.0]

    z_arr = np.zeros(len(t_params))
    via_hw = 0.01  # half-width for via transition smoothing (in t units)

    for vi, t in enumerate(t_params):
        # Find which segment this t belongs to
        seg_idx = min(np.searchsorted(t_bounds[1:], t), n_seg - 1)
        z_arr[vi] = z_per_seg[seg_idx]

        # Smooth z-transitions at interior segment boundaries
        if seg_idx > 0 and z_per_seg[seg_idx] != z_per_seg[seg_idx - 1]:
            t_boundary = t_bounds[seg_idx]
            if abs(t - t_boundary) < via_hw:
                u = np.clip((t - (t_boundary - via_hw)) / (2 * via_hw), 0, 1)
                z_arr[vi] = z_per_seg[seg_idx - 1] * (1 - u) + z_per_seg[seg_idx] * u

        # Smooth wrap-around transition (last segment ↔ first segment)
        if z_per_seg[-1] != z_per_seg[0]:
            if t > 1.0 - via_hw:
                u = np.clip((t - (1.0 - via_hw)) / via_hw, 0, 1)
                z_arr[vi] = z_per_seg[-1] * (1 - u) + z_per_seg[0] * u
            elif t < via_hw:
                u = np.clip(t / via_hw, 0, 1)
                z_arr[vi] = z_per_seg[0] * u + z_per_seg[-1] * (1 - u)

    new_verts = mesh_flat.vertices.copy()
    new_verts[:, 2] = z_arr
    return trimesh.Trimesh(vertices=new_verts, faces=mesh_flat.faces, process=False)


# =========================================================================
# Formatting
# =========================================================================


def _format_segments(segments):
    """Format segment info as human-readable lines."""
    lines = []
    total_wire_length = 0.0
    layers_used = set()

    for seg in segments:
        if seg["type"] == "WIRE":
            sx, sy = seg["start"]
            ex, ey = seg["end"]
            l_mm = seg.get("length", 0) * 1e3
            lines.append(
                f"  WIRE  ({sx * 1e3:.1f},{sy * 1e3:.1f})->({ex * 1e3:.1f},{ey * 1e3:.1f})  "
                f"w={seg['width'] * 1e3:.2f}mm  l={l_mm:.2f}mm  {seg['layer_name']}"
            )
            total_wire_length += seg.get("length", 0)
            layers_used.add(seg["layer_name"])
        elif seg["type"] == "VIA":
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
    return lines, summary


# =========================================================================
# Standalone test with real PCB data
# =========================================================================

if __name__ == "__main__":
    # Real path from KiCad PCB (see plugin output)
    # Path: (125,23) B.Cu -> (125,20) B.Cu -> (130,20) B.Cu -> VIA -> (130,25) F.Cu -> (125,25) F.Cu
    test_segments = [
        {
            "type": "WIRE",
            "start": (0.125, 0.020),
            "end": (0.125, 0.023),
            "node_at_start": 1,
            "node_at_end": 2,
            "path_from": 2,
            "path_to": 1,
            "width": 0.0002,
            "length": 0.003,
            "layer": 0,
            "layer_name": "B.Cu",
            "z": 0.001555,
            "thickness": 35e-6,
        },
        {
            "type": "WIRE",
            "start": (0.130, 0.020),
            "end": (0.125, 0.020),
            "node_at_start": 3,
            "node_at_end": 1,
            "path_from": 1,
            "path_to": 3,
            "width": 0.0002,
            "length": 0.005,
            "layer": 0,
            "layer_name": "B.Cu",
            "z": 0.001555,
            "thickness": 35e-6,
        },
        {
            "type": "VIA",
            "position": (0.130, 0.020),
            "drill": 0.0003,
            "length": 0.001545,
            "z_top": 0.001555,
            "z_bot": 0.000010,
            "layer1_name": "F.Cu",
            "layer2_name": "B.Cu",
        },
        {
            "type": "WIRE",
            "start": (0.130, 0.020),
            "end": (0.130, 0.025),
            "node_at_start": 4,
            "node_at_end": 5,
            "path_from": 4,
            "path_to": 5,
            "width": 0.0002,
            "length": 0.005,
            "layer": 1,
            "layer_name": "F.Cu",
            "z": 0.000010,
            "thickness": 35e-6,
        },
        {
            "type": "WIRE",
            "start": (0.130, 0.025),
            "end": (0.125, 0.025),
            "node_at_start": 5,
            "node_at_end": 6,
            "path_from": 5,
            "path_to": 6,
            "width": 0.0002,
            "length": 0.005,
            "layer": 1,
            "layer_name": "F.Cu",
            "z": 0.000010,
            "thickness": 35e-6,
        },
    ]

    print("=" * 60)
    print("Test 1: Real PCB trace (2-layer, F.Cu + B.Cu)")
    print("=" * 60)
    result = _compute_inductance(test_segments)
    print()
    print(result["message"])

    # ---- Test 2: 3-layer path ----
    # F.Cu (z=0.01mm) -> via -> In1.Cu (z=0.8mm) -> via -> B.Cu (z=1.555mm)
    # Rectangular path: bottom on F.Cu, right on In1.Cu, top+left on B.Cu
    def wire(s, e, ns, ne, pf, pt, w, layer_name, z):
        return {
            "type": "WIRE",
            "start": s,
            "end": e,
            "node_at_start": ns,
            "node_at_end": ne,
            "path_from": pf,
            "path_to": pt,
            "width": w,
            "length": ((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2) ** 0.5,
            "layer": 0,
            "layer_name": layer_name,
            "z": z,
            "thickness": 35e-6,
        }

    def via(pos, z_top, z_bot, l1, l2):
        return {
            "type": "VIA",
            "position": pos,
            "drill": 0.0003,
            "length": abs(z_top - z_bot),
            "z_top": z_top,
            "z_bot": z_bot,
            "layer1_name": l1,
            "layer2_name": l2,
        }

    # 10x10mm rectangle split across 3 layers:
    #   bottom edge: F.Cu
    #   right edge:  In1.Cu
    #   top + left:  B.Cu
    z_f, z_in, z_b = 0.010e-3, 0.800e-3, 1.555e-3
    w = 0.0002
    test_3layer = [
        wire((0, 0), (0.010, 0), 10, 11, 10, 11, w, "F.Cu", z_f),
        via((0.010, 0), z_in, z_f, "In1.Cu", "F.Cu"),
        wire((0.010, 0), (0.010, 0.010), 12, 13, 12, 13, w, "In1.Cu", z_in),
        via((0.010, 0.010), z_b, z_in, "B.Cu", "In1.Cu"),
        wire((0.010, 0.010), (0, 0.010), 14, 15, 14, 15, w, "B.Cu", z_b),
        wire((0, 0.010), (0, 0), 15, 16, 15, 16, w, "B.Cu", z_b),
    ]

    print()
    print("=" * 60)
    print("Test 2: 3-layer path (F.Cu + In1.Cu + B.Cu)")
    print("=" * 60)
    result3 = _compute_inductance(test_3layer)
    print()
    print(result3["message"])
