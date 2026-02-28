"""Generate coil geometries as debug_ItemList.json for Plot_PCB.py."""

from __future__ import annotations

import copy as _copy
import json
import math
import os
from dataclasses import dataclass, field
from typing import TypeAlias

from Get_PCB_Elements_IPC import _arc_geometry, _arc_points, _outline_from_midline
from pcb_types import WIRE, VIA, PAD, TOP_LAYER, BOTTOM_LAYER


Item = dict
NodeMap = dict[int, int]


Pos2D: TypeAlias = tuple[float, float]


@dataclass
class _IdContext:
    """Holds ID and node counters for one generation run.

    Pass a fresh instance to make_* functions or _CoilBuilder to get
    independent, deterministic IDs without relying on global state.
    """

    _next_id: int = field(default=0, repr=False)
    _next_node: int = field(default=1, repr=False)

    def new_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def new_node(self) -> int:
        n = self._next_node
        self._next_node += 1
        return n


# Module-level default context used when no explicit context is passed.
# Call new_context() to get a fresh, isolated context instead.
_default_ctx = _IdContext()


def new_context() -> _IdContext:
    """Return a fresh ID/node context for an independent generation run."""
    return _IdContext()


def _base_item(typ: str, layer: int, ctx: _IdContext) -> Item:
    return {
        "type": typ,
        "id": ctx.new_id(),
        "layer": [layer],
        "net_name": "",
        "net_code": "",
        "is_selected": False,
    }


def make_pad(
    pos: Pos2D,
    width: float = 0.0002,
    layer: int = 0,
    name: str = "",
    ctx: _IdContext | None = None,
) -> tuple[Item, int]:
    """Create a PAD at *pos*."""
    ctx = ctx or _default_ctx
    x, y = pos
    node = ctx.new_node()
    r = width / 2
    item = _base_item(PAD, layer, ctx)
    item.update(
        {
            "position": (x, y),
            "size": (width, width),
            "PadName": name,
            "conn_start": [],
            "net_start": {layer: node},
            "_outline": [
                (x - r, y - r),
                (x + r, y - r),
                (x + r, y + r),
                (x - r, y + r),
            ],
            "area": width * width,
        }
    )
    return item, node


def make_wire(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: float = 0.0002,
    layer: int = 0,
    node_start: int | None = None,
    node_end: int | None = None,
    ctx: _IdContext | None = None,
) -> tuple[Item, int, int]:
    """Create a straight WIRE."""
    ctx = ctx or _default_ctx
    ns = node_start if node_start is not None else ctx.new_node()
    ne = node_end if node_end is not None else ctx.new_node()
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1e-30
    wr = width / 2
    nx, ny = -dy / length * wr, dx / length * wr
    item = _base_item(WIRE, layer, ctx)
    item.update(
        {
            "start": (x1, y1),
            "end": (x2, y2),
            "width": width,
            "length": length,
            "area": width * length,
            "conn_start": [],
            "conn_end": [],
            "net_start": {layer: ns},
            "net_end": {layer: ne},
            "_midline_pts": [(x1, y1), (x2, y2)],
            "_outline": [
                (x1 + nx, y1 + ny),
                (x2 + nx, y2 + ny),
                (x2 - nx, y2 - ny),
                (x1 - nx, y1 - ny),
            ],
        }
    )
    return item, ns, ne


def make_arc(
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    x_mid: float,
    y_mid: float,
    width: float = 0.0002,
    layer: int = 0,
    node_start: int | None = None,
    node_end: int | None = None,
    ctx: _IdContext | None = None,
) -> tuple[Item, int, int]:
    """Create an arc WIRE via three points."""
    ctx = ctx or _default_ctx
    start, end, mid = (x_start, y_start), (x_end, y_end), (x_mid, y_mid)
    r_m, angle, arc_length, center, arc_sa = _arc_geometry(start, end, mid)

    ns = node_start if node_start is not None else ctx.new_node()
    ne = node_end if node_end is not None else ctx.new_node()

    midline: list[Pos2D] = [start, end]
    if r_m is not None and center is not None:
        midline = [start] + _arc_points(center, r_m, arc_sa, angle, width)

    item = _base_item(WIRE, layer, ctx)
    item.update(
        {
            "start": start,
            "end": end,
            "mid": mid,
            "radius": r_m,
            "angle": angle,
            "width": width,
            "length": arc_length,
            "area": width * arc_length,
            "conn_start": [],
            "conn_end": [],
            "net_start": {layer: ns},
            "net_end": {layer: ne},
            "_midline_pts": midline,
            "_outline": _outline_from_midline(midline, width / 2),
        }
    )
    return item, ns, ne


def make_via(
    x: float,
    y: float,
    drill: float = 0.0003,
    layers: list[int] | None = None,
    ctx: _IdContext | None = None,
) -> tuple[Item, NodeMap]:
    """Create a VIA at (*x*, *y*)."""
    ctx = ctx or _default_ctx
    if layers is None:
        layers = [TOP_LAYER, BOTTOM_LAYER]
    nodes: NodeMap = {}
    net_start: dict[int, int] = {}
    for ly in layers:
        n = ctx.new_node()
        nodes[ly] = n
        net_start[ly] = n
    item: Item = {
        "type": VIA,
        "id": ctx.new_id(),
        "position": (x, y),
        "drill": drill,
        "layer": layers,
        "net_name": "",
        "net_code": "",
        "is_selected": False,
        "net_start": net_start,
    }
    return item, nodes


def items_to_dict(items: list[Item]) -> dict[str, Item]:
    return {item["id"]: item for item in items}


def save_json(items: list[Item], path: str | None = None) -> str:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "debug_ItemList.json")
    d = items_to_dict(items)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Saved {len(d)} items to {path}")
    return path


def plot(items: list[Item]) -> None:
    from Plot_PCB import plot_items

    plot_items(items_to_dict(items))


class _CoilBuilder:
    """Accumulates coil segments with automatic node chaining."""

    def __init__(
        self,
        sx: float,
        sy: float,
        width: float,
        layer: int,
        node_start: int | None = None,
        ctx: _IdContext | None = None,
    ) -> None:
        self._ctx = ctx or _default_ctx
        self.items: list[Item] = []
        self.width = width
        self.layer = layer
        self.px, self.py = sx, sy
        self._start_pos: Pos2D = (sx, sy)
        self.prev_node = node_start if node_start is not None else self._ctx.new_node()
        self._start_node = self.prev_node

    def add(self, ex: float, ey: float, make_fn=make_wire, node_end=None, **kw):
        item, _, ne = make_fn(
            self.px,
            self.py,
            ex,
            ey,
            width=self.width,
            layer=self.layer,
            node_start=self.prev_node,
            node_end=node_end,
            ctx=self._ctx,
            **kw,
        )
        self.items.append(item)
        self.prev_node = ne
        self.px, self.py = ex, ey

    @property
    def start_pos(self) -> Pos2D:
        return self._start_pos

    @property
    def start_node(self) -> int:
        return self._start_node

    @property
    def end_node(self) -> int:
        return self.prev_node

    @property
    def end_pos(self) -> Pos2D:
        return self.px, self.py

    def reverse(self) -> _CoilBuilder:
        """Reverse winding direction in-place.

        The outline polygon is reversed in point order only (not rebuilt), which
        has no geometric effect but keeps its orientation consistent with the
        reversed midline.
        """
        for item in self.items:
            item["start"], item["end"] = item["end"], item["start"]
            item["net_start"], item["net_end"] = item["net_end"], item["net_start"]
            item["conn_start"], item["conn_end"] = item["conn_end"], item["conn_start"]
            item["_midline_pts"] = item["_midline_pts"][::-1]
            item["_outline"] = item["_outline"][::-1]
        self.items.reverse()
        self._start_pos, (self.px, self.py) = (self.px, self.py), self._start_pos
        self._start_node, self.prev_node = self.prev_node, self._start_node
        return self

    def mirror_x(self, center_y: float = 0.0) -> _CoilBuilder:
        """Mirror about a horizontal axis at *center_y* (flips CCW/CW)."""

        def _flip(pt: Pos2D) -> Pos2D:
            return (pt[0], 2 * center_y - pt[1])

        for item in self.items:
            item["start"] = _flip(item["start"])
            item["end"] = _flip(item["end"])
            if "mid" in item:
                item["mid"] = _flip(item["mid"])
            item["_midline_pts"] = [_flip(p) for p in item["_midline_pts"]]
            item["_outline"] = [_flip(p) for p in item["_outline"]]
        self._start_pos = _flip(self._start_pos)
        self.px, self.py = _flip((self.px, self.py))
        return self

    def mirror_y(self, center_x: float = 0.0) -> _CoilBuilder:
        """Mirror about a vertical axis at *center_x* (flips CCW/CW)."""

        def _flip(pt: Pos2D) -> Pos2D:
            return (2 * center_x - pt[0], pt[1])

        for item in self.items:
            item["start"] = _flip(item["start"])
            item["end"] = _flip(item["end"])
            if "mid" in item:
                item["mid"] = _flip(item["mid"])
            item["_midline_pts"] = [_flip(p) for p in item["_midline_pts"]]
            item["_outline"] = [_flip(p) for p in item["_outline"]]
        self._start_pos = _flip(self._start_pos)
        self.px, self.py = _flip((self.px, self.py))
        return self

    def move(self, delta: Pos2D = (0.0, 0.0), layer: int | None = None) -> _CoilBuilder:
        """Translate by *delta*, optionally change *layer*."""
        dx, dy = delta

        def _shift(pt: Pos2D) -> Pos2D:
            return (pt[0] + dx, pt[1] + dy)

        new_layer = layer if layer is not None else self.layer
        for item in self.items:
            item["start"] = _shift(item["start"])
            item["end"] = _shift(item["end"])
            if "mid" in item:
                item["mid"] = _shift(item["mid"])
            item["_midline_pts"] = [_shift(p) for p in item["_midline_pts"]]
            item["_outline"] = [_shift(p) for p in item["_outline"]]
            if new_layer != self.layer:
                old_ns = item["net_start"].pop(self.layer, None)
                old_ne = item["net_end"].pop(self.layer, None)
                if old_ns is not None:
                    item["net_start"][new_layer] = old_ns
                if old_ne is not None:
                    item["net_end"][new_layer] = old_ne
                item["layer"] = [new_layer]
        self._start_pos = _shift(self._start_pos)
        self.px, self.py = _shift((self.px, self.py))
        self.layer = new_layer
        return self

    def copy(self, layer: int | None = None) -> _CoilBuilder:
        """Deep-copy with new IDs/nodes, optionally on a different *layer*."""
        target_layer = layer if layer is not None else self.layer
        node_map: dict[int, int] = {}

        def _map_node(old: int) -> int:
            if old not in node_map:
                node_map[old] = self._ctx.new_node()
            return node_map[old]

        new = _CoilBuilder.__new__(_CoilBuilder)
        new._ctx = self._ctx
        new.width = self.width
        new.layer = target_layer
        new._start_pos = self._start_pos
        new.px, new.py = self.px, self.py
        new._start_node = _map_node(self._start_node)
        new.prev_node = _map_node(self.prev_node)
        new.items = []
        for item in self.items:
            it = _copy.deepcopy(item)
            it["id"] = self._ctx.new_id()
            it["layer"] = [target_layer]
            it["net_start"] = {target_layer: _map_node(item["net_start"][self.layer])}
            it["net_end"] = {target_layer: _map_node(item["net_end"][self.layer])}
            new.items.append(it)
        return new

    def result(self) -> list[Item]:
        return self.items


def generate_circular_coil(
    turns: int = 3,
    inner_radius: float = 0.001,
    spacing: float = 0.0003,
    width: float = 0.0002,
    ctx: _IdContext | None = None,
) -> _CoilBuilder:
    """Circular spiral at origin from 90-degree arcs (4 per turn)."""
    d_angle = math.pi / 2
    r0 = inner_radius
    pitch = (spacing + width) / 4

    cb = _CoilBuilder(r0, 0.0, width, 0, ctx=ctx)
    for i in range(turns * 4):
        a1 = i * d_angle
        r_mid = r0 + (i + 0.5) * pitch
        r2 = r0 + (i + 1) * pitch
        cb.add(
            r2 * math.cos(a1 + d_angle),
            r2 * math.sin(a1 + d_angle),
            make_arc,
            x_mid=r_mid * math.cos(a1 + d_angle / 2),
            y_mid=r_mid * math.sin(a1 + d_angle / 2),
        )
    return cb


def generate_square_coil(
    turns: int = 3,
    inner_size: float = 0.002,
    spacing: float = 0.0003,
    width: float = 0.0002,
    ctx: _IdContext | None = None,
) -> _CoilBuilder:
    """Square spiral (delegates to rectangular with equal sides)."""
    return generate_rectangular_coil(
        turns=turns,
        inner_width=inner_size,
        inner_height=inner_size,
        spacing=spacing,
        width=width,
        ctx=ctx,
    )


def generate_rectangular_coil(
    turns: int = 3,
    inner_width: float = 0.002,
    inner_height: float = 0.001,
    spacing: float = 0.0003,
    width: float = 0.0002,
    ctx: _IdContext | None = None,
) -> _CoilBuilder:
    """Rectangular spiral at origin, strictly horizontal/vertical edges."""
    pitch = spacing + width
    hw0, hh0 = inner_width / 2, inner_height / 2

    cb = _CoilBuilder(hw0, -hh0, width, 0, ctx=ctx)
    for t in range(turns):
        hw = hw0 + t * pitch
        hh = hh0 + t * pitch
        for ex, ey in [
            (hw, hh),
            (-hw, hh),
            (-hw, -hh - pitch),
            (hw + pitch, -hh - pitch),
        ]:
            cb.add(ex, ey)
    return cb


def generate_rounded_rectangular_coil(
    turns: int = 3,
    inner_width: float = 0.002,
    inner_height: float = 0.001,
    corner_radius: float = 0.0003,
    spacing: float = 0.0003,
    width: float = 0.0002,
    ctx: _IdContext | None = None,
) -> _CoilBuilder:
    """Rectangular spiral at origin with rounded corners (90-degree arcs)."""
    pitch = spacing + width
    hw0, hh0 = inner_width / 2, inner_height / 2
    c45 = math.cos(math.pi / 4)

    cb = _CoilBuilder(hw0, -hh0, width, 0, ctx=ctx)
    for t in range(turns):
        hw = hw0 + t * pitch
        hh = hh0 + t * pitch
        r = min(corner_radius, hw, hh)

        cx_tr, cy_tr = hw - r, hh - r
        cx_tl, cy_tl = -hw + r, hh - r
        cx_bl, cy_bl = -hw + r, -hh - pitch + r
        cx_br, cy_br = hw + pitch - r, -hh - pitch + r

        corners = [
            (
                (hw, cy_tr),
                (cx_tr + r * c45, cy_tr + r * c45),
                (cx_tr, hh),
            ),
            (
                (cx_tl, hh),
                (cx_tl - r * c45, cy_tl + r * c45),
                (-hw, cy_tl),
            ),
            (
                (-hw, cy_bl),
                (cx_bl - r * c45, cy_bl - r * c45),
                (cx_bl, -hh - pitch),
            ),
            (
                (cx_br, -hh - pitch),
                (cx_br + r * c45, cy_br - r * c45),
                (hw + pitch, cy_br),
            ),
        ]
        for (sx, sy), (mx, my), (ax, ay) in corners:
            cb.add(sx, sy)
            cb.add(ax, ay, make_arc, x_mid=mx, y_mid=my)

    return cb


def generate_polygon_coil(
    turns: int = 3,
    inner_apothem: float = 0.001,
    spacing: float = 0.0003,
    width: float = 0.0002,
    sides: int = 6,
    ctx: _IdContext | None = None,
) -> _CoilBuilder:
    """Regular polygon spiral at origin with constant edge-to-edge spacing."""
    pitch = spacing + width
    da = 2 * math.pi / sides
    cos_half = math.cos(math.pi / sides)
    a0 = da / 2 if sides % 2 == 0 else math.pi / 2 + da / 2

    def _vertex(apothem: float, k: int) -> Pos2D:
        r = apothem / cos_half
        a = a0 + k * da
        return (r * math.cos(a), r * math.sin(a))

    cb = _CoilBuilder(*_vertex(inner_apothem, 0), width, 0, ctx=ctx)
    for t in range(turns):
        ap = inner_apothem + t * pitch
        # Draw sides 1..sides-1; vertex 0 is the shared start/transition point
        for k in range(sides - 1):
            cb.add(*_vertex(ap, k + 1))
        if t < turns - 1:
            # Step outward to vertex 0 of the next turn (open spiral, not closed)
            cb.add(*_vertex(inner_apothem + (t + 1) * pitch, 0))

    return cb


def generate_elliptical_coil(
    turns: int = 3,
    inner_rx: float = 0.001,
    inner_ry: float = 0.0006,
    spacing: float = 0.0003,
    width: float = 0.0002,
    ctx: _IdContext | None = None,
) -> _CoilBuilder:
    """Elliptical spiral at origin from 90-degree arcs, constant aspect ratio."""
    d_angle = math.pi / 2
    aspect = inner_ry / inner_rx
    pitch_x = (spacing + width) / 4
    pitch_y = pitch_x * aspect

    cb = _CoilBuilder(inner_rx, 0.0, width, 0, ctx=ctx)
    for i in range(turns * 4):
        a1 = i * d_angle
        rx_mid = inner_rx + (i + 0.5) * pitch_x
        ry_mid = inner_ry + (i + 0.5) * pitch_y
        rx2 = inner_rx + (i + 1) * pitch_x
        ry2 = inner_ry + (i + 1) * pitch_y
        cb.add(
            rx2 * math.cos(a1 + d_angle),
            ry2 * math.sin(a1 + d_angle),
            make_arc,
            x_mid=rx_mid * math.cos(a1 + d_angle / 2),
            y_mid=ry_mid * math.sin(a1 + d_angle / 2),
        )
    return cb


generate_spiral_coil = generate_circular_coil


if __name__ == "__main__":
    ctx = new_context()
    t, w, s = 3, 0.0002, 0.0003
    inner_width = 0.003
    inner_height = 0.001
    items: list[Item] = []

    # Simple single-layer coils with pads
    for cb in [
        generate_circular_coil(turns=t, width=w, spacing=s, ctx=ctx),
        generate_square_coil(turns=t, width=w, spacing=s, ctx=ctx).move((0.006, 0.0)),
        generate_rectangular_coil(
            turns=t,
            width=w,
            spacing=s,
            inner_width=inner_width,
            inner_height=inner_height,
            ctx=ctx,
        ).move((0.013, 0.0)),
        generate_polygon_coil(turns=t, width=w, spacing=s, sides=6, ctx=ctx).move(
            (0.0, 0.006)
        ),
        generate_elliptical_coil(
            turns=t,
            width=w,
            spacing=s,
            inner_rx=0.0015,
            inner_ry=0.0008,
            ctx=ctx,
        ).move((0.006, 0.006)),
        generate_rounded_rectangular_coil(
            turns=t,
            width=w,
            spacing=s,
            inner_width=inner_width,
            inner_height=inner_height,
            corner_radius=0.0004,
            ctx=ctx,
        ).move((0.013, 0.006)),
    ]:
        pad1, _ = make_pad(cb.start_pos, width=w, name="1", layer=cb.layer, ctx=ctx)
        pad2, _ = make_pad(cb.end_pos, width=w, name="2", layer=cb.layer, ctx=ctx)
        items += [pad1] + cb.result() + [pad2]

    # Two-layer coil: Pad1 (outer, top) → inner → wire → via → inner (bot) → Pad2 (outer, bot)
    cx, cy = 0.020, 0.003
    top = generate_rectangular_coil(
        turns=t,
        width=w,
        spacing=s,
        inner_width=inner_width,
        inner_height=inner_height,
        ctx=ctx,
    ).move((cx, cy))
    bot = top.copy(layer=BOTTOM_LAYER).mirror_x(cy).move((0.0, -s - w))
    top.reverse()
    tx, ty = top.end_pos
    bx, by = bot.start_pos
    wire, _ns, _ne = make_wire(
        tx, ty, bx, by, width=w, layer=BOTTOM_LAYER, node_start=top.end_node, ctx=ctx
    )
    via_item, _node_map = make_via(tx, ty, layers=[TOP_LAYER, BOTTOM_LAYER], ctx=ctx)
    pad1, _ = make_pad(top.start_pos, width=w, name="1", layer=TOP_LAYER, ctx=ctx)
    pad2, _ = make_pad(bot.end_pos, width=w, name="2", layer=BOTTOM_LAYER, ctx=ctx)
    items += [pad1] + top.result() + [wire, via_item] + bot.result() + [pad2]

    save_json(items)
    plot(items)
