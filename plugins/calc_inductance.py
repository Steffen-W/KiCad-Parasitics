"""
Calculate trace inductance from shortest-path geometry using bfieldtools.

Receives the path segments (WIRE + VIA elements) extracted from the PCB
network and computes the loop inductance of the trace path.
"""

try:
    import matplotlib

    matplotlib.use("WXAgg", force=True)
except Exception:
    pass

import logging
import math
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from bfieldtools.mesh_impedance import self_inductance_matrix
from bfieldtools.utils import find_mesh_boundaries

try:
    from .pcb_types import WIRE, VIA
except ImportError:
    from pcb_types import WIRE, VIA

log = logging.getLogger(__name__)

_MESH_TARGET_FACES = 2000
_MAX_MESH_FACES = 5000

# =========================================================================
# Public API
# =========================================================================


def calc_path_inductance(path, network_info, cu_stack):
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


def _extract_segments(path, network_info, cu_stack):
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
                "thickness": cu_stack.get(layer_id, {}).get("thickness", 35e-6),
            }
            for key in ("mid", "radius", "angle", "_midline_pts"):
                if key in elem:
                    seg[key] = elem[key]
            segments.append(seg)

        elif elem["type"] == VIA:
            z1 = cu_stack.get(elem["layer1"], {}).get("abs_height", 0.0)
            z2 = cu_stack.get(elem["layer2"], {}).get("abs_height", 0.0)
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


def _make_plot_frame_class():
    """Create PlotFrame class using wx (imported lazily)."""
    import wx
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
    from matplotlib.backends.backend_wx import NavigationToolbar2Wx as NavToolbar

    class PlotFrame(wx.Frame):
        """wx.Frame wrapper around a matplotlib Figure."""

        def __init__(self, parent, fig, title="Plot"):
            size = fig.get_size_inches() * fig.get_dpi()
            super().__init__(
                parent,
                title=title,
                size=wx.Size(int(size[0]), int(size[1]) + 80),
            )
            panel = wx.Panel(self)
            sizer = wx.BoxSizer(wx.VERTICAL)
            self.canvas = FigureCanvas(panel, wx.ID_ANY, fig)
            self.toolbar = NavToolbar(self.canvas)
            self.toolbar.Realize()
            sizer.Add(self.toolbar, 0, wx.EXPAND)
            sizer.Add(self.canvas, 1, wx.EXPAND)
            panel.SetSizer(sizer)
            self.Bind(wx.EVT_CLOSE, lambda evt: self.Destroy())

    return PlotFrame


def _show_figure(fig, title, parent):
    """Display a Figure in a PlotFrame via wx.CallAfter (UI-thread safe)."""
    import wx

    PlotFrame = _make_plot_frame_class()

    def _create():
        frame = PlotFrame(parent, fig, title)
        frame.Show()
        frame.Raise()

    wx.CallAfter(_create)


def show_debug_plots(debug_data, segments, parent=None):
    """Show interactive debug plots for centerline and mesh.

    Parameters
    ----------
    debug_data : dict with centerline, mesh_vertices, mesh_faces
    segments : list[dict] – segment descriptors
    parent : wx.Window or None – parent for plot frames (None for standalone)
    """
    from matplotlib.figure import Figure
    import matplotlib
    import matplotlib.tri as mtri
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.cm import ScalarMappable

    layer_colors = {
        "F.Cu": "red",
        "B.Cu": "blue",
        "In1.Cu": "green",
        "In2.Cu": "orange",
    }

    centerline = debug_data["centerline"]
    cl = np.array(centerline) * 1e3
    verts_mm = debug_data["mesh_vertices"][:, :2] * 1e3
    faces = debug_data["mesh_faces"]

    # --- Figure 1: Centerline ---
    fig1 = Figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111)
    drawn_layers = set()
    for seg in segments:
        if seg["type"] == VIA:
            px, py = seg["position"]
            ax1.plot(
                px * 1e3,
                py * 1e3,
                "kx",
                ms=8,
                mew=2,
                zorder=6,
                label="VIA" if "VIA" not in drawn_layers else None,
            )
            drawn_layers.add("VIA")
            continue
        if seg["type"] != WIRE:
            continue
        layer_name = seg.get("layer_name", "?")
        color = layer_colors.get(layer_name, "gray")
        s = np.array(seg["start"]) * 1e3
        e = np.array(seg["end"]) * 1e3
        midline = seg.get("_midline_pts")
        if midline is not None and len(midline) > 2:
            pts_mm = [(p[0] * 1e3, p[1] * 1e3) for p in midline]
            xs, ys = zip(*pts_mm)
            label = layer_name if layer_name not in drawn_layers else None
            ax1.plot(xs, ys, ".-", color=color, lw=2, ms=4, label=label)
        else:
            label = layer_name if layer_name not in drawn_layers else None
            ax1.plot(
                [s[0], e[0]], [s[1], e[1]], ".-", color=color, lw=2, ms=4, label=label
            )
        drawn_layers.add(layer_name)
    ax1.plot(cl[:, 0], cl[:, 1], "m--", lw=0.8, alpha=0.7, label="centerline")
    ax1.plot(cl[0, 0], cl[0, 1], "go", ms=6, zorder=5, label="start")
    ax1.plot(cl[-1, 0], cl[-1, 1], "rs", ms=6, zorder=5, label="end")
    ax1.set_aspect("equal")
    ax1.invert_yaxis()
    ax1.legend(fontsize=7, loc="best")
    ax1.set_title(f"Centerline – {len(cl)} pts")
    ax1.set_xlabel("mm")
    ax1.set_ylabel("mm")
    _show_figure(fig1, f"Centerline – {len(cl)} pts", parent)

    psi = debug_data.get("psi")
    M = debug_data.get("M")

    # --- Figure 3: Inductance contribution along centerline ---
    if psi is not None and M is not None:
        Mpsi = M @ psi
        energy_vertex = psi * Mpsi
        mesh_verts = debug_data["mesh_vertices"]
        n_cols = int(faces[0].max())
        n_verts = len(mesh_verts)
        n_rows = n_verts // n_cols
        energy_per_row = (
            energy_vertex[: n_rows * n_cols].reshape(n_rows, n_cols).sum(axis=1)
        )
        fine_cl = (
            mesh_verts[: n_rows * n_cols].reshape(n_rows, n_cols, 3).mean(axis=1) * 1e3
        )
        fine_diffs = np.diff(fine_cl, axis=0)
        fine_seg_len = np.linalg.norm(fine_diffs, axis=1)
        arc_len = np.concatenate([[0], np.cumsum(fine_seg_len)])

        from matplotlib.gridspec import GridSpec

        fig3 = Figure(figsize=(10, 10))
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3, figure=fig3)
        ax3a = fig3.add_subplot(gs[0])
        ax3b = fig3.add_subplot(gs[1])
        ax3a.set_aspect("equal")

        energy_per_row_nH = energy_per_row * 1e9
        L_total_nH = (psi @ Mpsi) * 1e9

        norm = Normalize(vmin=energy_per_row_nH.min(), vmax=energy_per_row_nH.max())
        cmap = matplotlib.colormaps["hot"]
        for i in range(n_rows - 1):
            ax3a.plot(
                [fine_cl[i, 0], fine_cl[i + 1, 0]],
                [fine_cl[i, 1], fine_cl[i + 1, 1]],
                color=cmap(norm(energy_per_row_nH[i])),
                lw=4,
                solid_capstyle="round",
            )
        ax3a.plot(fine_cl[0, 0], fine_cl[0, 1], "go", ms=8, zorder=5, label="start")

        sm = ScalarMappable(cmap=cmap, norm=norm)
        fig3.colorbar(sm, ax=ax3a, label="Inductance contribution (nH)")
        ax3a.legend(fontsize=7, loc="best")
        ax3a.invert_yaxis()
        ax3a.set_title(f"Inductance along trace (total L = {L_total_nH:.2f} nH)")
        ax3a.set_xlabel("mm")
        ax3a.set_ylabel("mm")

        ax3b.fill_between(arc_len, energy_per_row_nH, alpha=0.4, color="crimson")
        ax3b.plot(arc_len, energy_per_row_nH, color="crimson", lw=1.5)
        ax3b.set_xlabel("Position along trace (mm)")
        ax3b.set_ylabel("Inductance contribution (nH)")
        ax3b.grid(True, alpha=0.3)

        title3 = f"Inductance along trace (L = {L_total_nH:.2f} nH)"
        _show_figure(fig3, title3, parent)

    # --- Figure 4: |B|-field + current distribution (psi) ---
    if psi is not None:
        from bfieldtools.mesh_magnetics import magnetic_field_coupling

        mesh = trimesh.Trimesh(
            vertices=debug_data["mesh_vertices"],
            faces=debug_data["mesh_faces"],
            process=False,
        )
        z_mean = mesh.vertices[:, 2].mean()
        z_offset = z_mean + 0.5e-3
        margin = 2e-3
        x_min, x_max = (
            mesh.vertices[:, 0].min() - margin,
            mesh.vertices[:, 0].max() + margin,
        )
        y_min, y_max = (
            mesh.vertices[:, 1].min() - margin,
            mesh.vertices[:, 1].max() + margin,
        )
        n_grid = 40
        x_grid = np.linspace(x_min, x_max, n_grid)
        y_grid = np.linspace(y_min, y_max, n_grid)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_offset)])

        B_coupling = magnetic_field_coupling(mesh, grid_pts, analytic=True)
        B = np.einsum("ijk,k->ij", B_coupling, psi)  # (Npts, 3)
        B_mag = np.linalg.norm(B, axis=1).reshape(X.shape)

        X_mm = X * 1e3
        Y_mm = Y * 1e3
        fig4 = Figure(figsize=(14, 10), constrained_layout=True)
        ax4 = fig4.add_subplot(111)

        # |B|-field as pcolormesh with log scale
        B_mag_pos = np.where(B_mag > 0, B_mag, np.nan)
        b_min = np.nanmin(B_mag_pos)
        b_max = np.nanmax(B_mag_pos)
        if b_min > 0 and b_max > b_min:
            pcm = ax4.pcolormesh(
                X_mm,
                Y_mm,
                B_mag_pos,
                norm=LogNorm(vmin=b_min, vmax=b_max),
                cmap="viridis",
                shading="gouraud",
            )
            fig4.colorbar(pcm, ax=ax4, label="|B| (T)", fraction=0.035, pad=0.04)

        # Overlay: current distribution (psi on mesh)
        tri = mtri.Triangulation(verts_mm[:, 0], verts_mm[:, 1], faces)
        psi_face = psi[faces].mean(axis=1)
        tc = ax4.tripcolor(
            tri, psi_face, shading="flat", cmap="inferno", edgecolors="face"
        )
        fig4.colorbar(
            tc, ax=ax4, label="Stream function psi (A)", fraction=0.035, pad=0.08
        )

        ax4.plot(cl[0, 0], cl[0, 1], "r^", ms=8, mew=2, zorder=8, label="start")
        ax4.legend(fontsize=7, loc="best")
        ax4.set_aspect("equal")
        ax4.invert_yaxis()
        ax4.set_title(f"|B| at z = {z_offset * 1e3:.2f} mm + current distribution")
        ax4.set_xlabel("mm")
        ax4.set_ylabel("mm")
        _show_figure(fig4, "|B| field + current distribution", parent)


# =========================================================================
# Inductance computation
# =========================================================================


def _compute_inductance(segments):
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


def _build_centerline(segments):
    """Build ordered 3D centerline from path segments.

    VIAs produce two points at the same (x,y) with z_from and z_to,
    creating a vertical strip segment in the mesh.

    Returns
    -------
    centerline : list of (x, y, z) tuples in meters (not yet closed)
    width_per_seg : list of widths (m), one per centerline segment
    """
    centerline = []
    width_per_seg = []
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

        midline = seg.get("_midline_pts")
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


def _neighbor_width(via_seg, segments, fallback):
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
    return min(widths) if widths else fallback


# =========================================================================
# Strip mesh builder
# =========================================================================


def _build_strip_mesh(
    centerline,
    width_per_seg,
    max_faces=_MESH_TARGET_FACES,
    n_width=4,
):
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

    faces = np.array(faces)
    n_faces = len(faces)
    log.debug("Strip mesh: %d vertices, %d faces", len(verts), n_faces)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


def _build_stream_function(mesh):
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


def _format_segments(segments):
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

if __name__ == "__main__":
    # Real path from KiCad PCB (see plugin output)
    # Path: (125,23) B.Cu -> (125,20) B.Cu -> (130,20) B.Cu -> VIA -> (130,25) F.Cu -> (125,25) F.Cu
    test_segments = [
        {
            "type": WIRE,
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
            "type": WIRE,
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
            "type": VIA,
            "position": (0.130, 0.020),
            "drill": 0.0003,
            "length": 0.001545,
            "z_top": 0.001555,
            "z_bot": 0.000010,
            "layer1_name": "F.Cu",
            "layer2_name": "B.Cu",
        },
        {
            "type": WIRE,
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
            "type": WIRE,
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
            "type": WIRE,
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
            "type": VIA,
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
