"""BoardSource for KiCad's IPC API (kipy, KiCad 9+)."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from kipy.board import Board
from kipy.board_types import (
    ArcTrack,
    BoardArc,
    BoardPolygon,
    BoardSegment,
    FootprintInstance,
    Pad,
    Track,
    Via,
)
from kipy.errors import ApiError
from kipy.proto.board.board_pb2 import BoardStackupLayerType
from kipy.proto.board.board_types_pb2 import ZoneType
from kipy.util.board_layer import canonical_name, is_copper_layer

try:
    from .board_model import (
        Copper,
        NetTie,
        Selected,
        StackupEntry,
        arc_midline,
        build_cu_stack,
        dielectric,
        layer_id,
        net_tie,
        pad,
        polygon_area,
        via,
        wire,
        zone,
    )
    from .pcb_types import BoardElement, CuLayer, Point
except ImportError:
    from board_model import (
        Copper,
        NetTie,
        Selected,
        StackupEntry,
        arc_midline,
        build_cu_stack,
        dielectric,
        layer_id,
        net_tie,
        pad,
        polygon_area,
        via,
        wire,
        zone,
    )
    from pcb_types import BoardElement, CuLayer, Point

log = logging.getLogger(__name__)


def _m(nm: float) -> float:
    return nm / 1e9


def _point(vector: Any) -> Point:
    return (_m(vector.x), _m(vector.y))


def _polyline(polyline: Any) -> list[Point]:
    """Points of a kipy PolyLine; arcs become several points."""
    points: list[Point] = []
    for node in polyline.nodes:
        if node.has_point:
            points.append(_point(node.point))
        elif node.has_arc:
            arc = node.arc
            points += arc_midline(_point(arc.start), _point(arc.end), _point(arc.mid))
    return points


def _polygon_area(polygon: Any) -> float:
    """Area of a kipy PolygonWithHoles."""
    holes = sum(polygon_area(_polyline(h)) for h in polygon.holes)
    return max(polygon_area(_polyline(polygon.outline)) - holes, 0.0)


def _selected(item: Any) -> Selected:
    if isinstance(item, FootprintInstance):
        return Selected(
            str(item.id.value), pads=tuple(map(_selected, item.definition.pads))
        )
    if isinstance(item, (Track, ArcTrack, Via, Pad)):
        return Selected(str(item.id.value), item.net.name)
    return Selected(str(item.id.value))


class IpcBoard:
    def __init__(self, board: Board) -> None:
        self._board = board
        self._enabled = set(board.get_enabled_layers())

    def _copper(self, layers: Iterable[Any]) -> list[Any]:
        """The enabled copper layers among kipy's BoardLayer values."""
        return [bl for bl in layers if is_copper_layer(bl) and bl in self._enabled]

    def _layers(self, layers: Iterable[Any]) -> list[int]:
        ids = (layer_id(canonical_name(bl)) for bl in self._copper(layers))
        return sorted(i for i in ids if i is not None)

    def selection(self) -> list[Selected]:
        return [_selected(item) for item in self._board.get_selection()]

    def net_ties(self) -> list[NetTie]:
        ties = []
        for fp in self._board.get_footprints():
            groups = [list(nt.pad_number) for nt in fp.definition.proto.net_ties]
            if groups:
                ref = fp.reference_field.text.value if fp.reference_field else "?"
                pad_nets = {p.number: p.net.name for p in fp.definition.pads}
                ties.append(net_tie(ref, groups, pad_nets, fp))
        return ties

    def copper(
        self, nets: AbstractSet[str], net_ties: Sequence[NetTie]
    ) -> Iterator[BoardElement]:
        board = self._board
        for track in board.get_tracks():
            if track.net.name in nets:
                mid = _point(track.mid) if isinstance(track, ArcTrack) else None
                start, end = _point(track.start), _point(track.end)
                layers = self._layers([track.layer])
                yield wire(
                    str(track.id.value),
                    track.net.name,
                    layers,
                    start,
                    end,
                    _m(track.width),
                    mid,
                )
        for v in board.get_vias():
            if v.net.name in nets:
                yield via(
                    str(v.id.value),
                    v.net.name,
                    self._layers(v.padstack.layers),
                    _point(v.position),
                    _m(v.diameter),
                    _m(v.drill_diameter),
                )
        for p in board.get_pads():
            if p.net.name in nets:
                yield self._pad(p)
        for z in board.get_zones():
            if z.type == ZoneType.ZT_COPPER and z.net.name in nets:
                area = sum(
                    _polygon_area(p)
                    for polys in z.filled_polygons.values()
                    for p in polys
                )
                yield zone(
                    str(z.id.value),
                    z.net.name,
                    self._layers(z.layers),
                    _polyline(z.outline.outline),
                    area,
                )
        for tie in net_ties:
            for shape in tie.footprint.definition.shapes:
                yield from self._shape(shape)

    def _shape(self, shape: Any) -> Iterator[BoardElement]:
        """Net Tie copper: segments and arcs as wires, polygons as zones."""
        layers = self._layers([shape.layer])
        if not layers:
            return
        args = (str(shape.id.value), "", layers)
        if isinstance(shape, BoardPolygon) and shape.polygons:
            outline = _polyline(shape.polygons[0].outline)
            yield zone(*args, outline, polygon_area(outline))
        elif isinstance(shape, (BoardSegment, BoardArc)):
            mid = _point(shape.mid) if isinstance(shape, BoardArc) else None
            start, end = _point(shape.start), _point(shape.end)
            yield wire(*args, start, end, _m(shape.attributes.stroke.width), mid)

    def _pad(self, p: Any) -> BoardElement:
        copper = self._copper(p.padstack.layers)
        outline: list[Point] = []
        if copper:
            # One request per pad: kipy's batch form drops pads without a shape,
            # so its results cannot be matched to the pads.
            try:
                shape = self._board.get_pad_shapes_as_polygons(p, layer=copper[0])
            except ApiError:
                log.debug("No shape for pad %s", p.id.value, exc_info=True)
            else:
                if shape is not None:
                    outline = _polyline(shape.outline)
        drill = _m(p.padstack.drill.diameter.x) if p.padstack.drill else 0.0
        position = _point(p.position)
        return pad(
            str(p.id.value), p.net.name, self._layers(copper), position, drill, outline
        )

    def stackup(self) -> dict[int, CuLayer]:
        entries: list[StackupEntry] = []
        for layer in self._board.get_stackup().layers:
            if layer.type == BoardStackupLayerType.BSLT_COPPER:
                name = (
                    canonical_name(layer.layer)
                    if is_copper_layer(layer.layer)
                    else layer.user_name
                )
                entries.append(Copper(name, _m(layer.thickness)))
            elif layer.type == BoardStackupLayerType.BSLT_DIELECTRIC:
                sub = layer.dielectric.layers
                if sub:
                    h = _m(sum(s.thickness for s in sub))
                    entries.append(dielectric(h, sub[0].epsilon_r, sub[0].loss_tangent))
                elif layer.thickness > 0:
                    entries.append(dielectric(_m(layer.thickness)))
        return build_cu_stack(entries)
