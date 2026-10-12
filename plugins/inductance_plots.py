"""Interactive debug plots for calc_inductance.py (centerline, mesh, |B|-field)."""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.tri as mtri
import numpy as np
import trimesh
import wx
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

try:
    from .pcb_types import VIA, WIRE
except ImportError:
    from pcb_types import VIA, WIRE


class PlotFrame(wx.Frame):
    """wx.Frame around a matplotlib Figure; stays above its parent."""

    def __init__(self, parent: wx.Window, fig: Figure, title: str) -> None:
        size = fig.get_size_inches() * fig.get_dpi()
        super().__init__(
            parent,
            title=title,
            size=wx.Size(int(size[0]), int(size[1]) + 80),
            style=wx.DEFAULT_FRAME_STYLE | wx.FRAME_FLOAT_ON_PARENT,
        )
        panel = wx.Panel(self)
        canvas = FigureCanvasWxAgg(panel, wx.ID_ANY, fig)
        toolbar = NavigationToolbar2Wx(canvas)
        toolbar.Realize()
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(toolbar, 0, wx.EXPAND)
        sizer.Add(canvas, 1, wx.EXPAND)
        panel.SetSizer(sizer)


def show_debug_plots(
    debug_data: dict[str, Any],
    segments: list[dict[str, Any]],
    parent: wx.Window,
) -> None:
    """Show interactive debug plots for centerline and mesh.

    Parameters
    ----------
    debug_data : dict with centerline, mesh_vertices, mesh_faces
    segments : list[dict] – segment descriptors
    parent : wx.Window – the plot frames stay above it and close with it
    """
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
        midline = seg.get("midline_pts")
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
    PlotFrame(parent, fig1, f"Centerline – {len(cl)} pts").Show()

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
        PlotFrame(parent, fig3, title3).Show()

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
        PlotFrame(parent, fig4, "|B| field + current distribution").Show()
