"""Plot debug_ItemList.json: outlines, endpoint circles, pads, vias, node IDs."""

import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from pcb_types import WIRE, VIA, PAD, ZONE

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "debug_ItemList.json"
)

LAYER_COLORS = {0: "red", 2: "blue", 4: "green", 6: "orange"}


def _color(layers: list[int]) -> str:
    for ly in layers:
        if ly in LAYER_COLORS:
            return LAYER_COLORS[ly]
    return "gray"


def _collect_nodes(d: dict, pos_mm: tuple) -> dict[int, tuple]:
    """Extract node IDs from net_start/net_end and map to positions."""
    nodes = {}
    for layer in d.get("layer", []):
        for net_key, p in [("net_start", pos_mm[0]), ("net_end", pos_mm[1])]:
            ns = d.get(net_key, {})
            n = ns.get(layer) or ns.get(str(layer))
            if n and n > 0:
                nodes[n] = p
    return nodes


def plot_items(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 14))
    node_positions = {}

    for oid, d in data.items():
        typ = d["type"]
        color = _color(d.get("layer", []))
        outline = d.get("_outline")

        if typ == WIRE:
            w = d.get("width", 0)
            r = w / 2 * 1e3

            if outline and len(outline) >= 3:
                pts = [(p[0] * 1e3, p[1] * 1e3) for p in outline]
                ax.add_patch(
                    Polygon(pts, closed=True, fc=color, ec=color, alpha=0.15, lw=0.3)
                )

            midline = d.get("_midline_pts")
            if midline and len(midline) >= 2:
                ax.plot(
                    [p[0] * 1e3 for p in midline],
                    [p[1] * 1e3 for p in midline],
                    "-",
                    color=color,
                    lw=0.5,
                    alpha=0.6,
                )

            sx, sy = d["start"][0] * 1e3, d["start"][1] * 1e3
            ex, ey = d["end"][0] * 1e3, d["end"][1] * 1e3
            ax.add_patch(Circle((sx, sy), r, fc="none", ec=color, lw=0.3, alpha=0.5))
            ax.add_patch(Circle((ex, ey), r, fc="none", ec=color, lw=0.3, alpha=0.5))

            if d.get("is_selected"):
                ax.plot([sx, ex], [sy, ey], "o-", color="lime", ms=4, lw=2, zorder=10)

            node_positions.update(_collect_nodes(d, ((sx, sy), (ex, ey))))

        elif typ == VIA:
            px, py = d["position"][0] * 1e3, d["position"][1] * 1e3
            r = d.get("drill", 0.0003) / 2 * 1e3
            ax.add_patch(
                Circle((px, py), r, fc="black", ec="black", alpha=0.4, zorder=5)
            )

            for layer in d.get("layer", []):
                ns = d.get("net_start", {})
                n = ns.get(layer) or ns.get(str(layer))
                if n and n > 0:
                    node_positions[n] = (px, py)

        elif typ == PAD:
            px, py = d["position"][0] * 1e3, d["position"][1] * 1e3

            if outline and len(outline) >= 3:
                pts = [(p[0] * 1e3, p[1] * 1e3) for p in outline]
                ax.add_patch(
                    Polygon(
                        pts,
                        closed=True,
                        fc="gold",
                        ec="darkgoldenrod",
                        alpha=0.3,
                        lw=0.5,
                    )
                )
            else:
                sx, sy = d.get("size", (0.001, 0.001))
                r = max(sx, sy) / 2 * 1e3
                ax.add_patch(
                    Circle((px, py), r, fc="gold", ec="darkgoldenrod", alpha=0.3)
                )

            ax.text(
                px,
                py,
                d.get("PadName", ""),
                fontsize=12,
                ha="center",
                va="top",
                zorder=8,
            )

            if d.get("is_selected"):
                sx, sy = d.get("size", (0.001, 0.001))
                r = max(sx, sy) / 2 * 1e3
                ax.add_patch(Circle((px, py), r, fc="none", ec="lime", lw=2, zorder=10))

            for layer in d.get("layer", []):
                ns = d.get("net_start", {})
                n = ns.get(layer) or ns.get(str(layer))
                if n and n > 0:
                    node_positions[n] = (px, py)

        elif typ == ZONE:
            if outline and len(outline) >= 3:
                pts = [(p[0] * 1e3, p[1] * 1e3) for p in outline]
                ax.add_patch(
                    Polygon(pts, closed=True, fc=color, ec=color, alpha=0.05, lw=0.3)
                )

    # Plot node labels
    for node_id, (x, y) in node_positions.items():
        ax.text(
            x,
            y,
            f"n{node_id}",
            fontsize=10,
            color="purple",
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=12,
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.grid(True, alpha=0.2)
    ax.set_title(f"debug_ItemList.json ({len(data)} elements)")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    input("Press Enter to close...")
    plt.close("all")


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"debug_ItemList.json not found at {DATA_FILE}\n"
            "Run the plugin once to generate it, or set PARASITIC_DEBUG=1."
        )
    with open(DATA_FILE) as f:
        data = json.load(f)
    data = {k: v for k, v in data.items() if isinstance(v, dict) and "type" in v}
    print(f"Loaded {len(data)} elements from {DATA_FILE}")
    plot_items(data)
