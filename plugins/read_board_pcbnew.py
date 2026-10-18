"""BoardSource for KiCad's SWIG API (pcbnew), which KiCad 11 is expected to drop."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

import pcbnew

try:
    from .board_model import (
        NetTie,
        Selected,
        build_cu_stack,
        net_tie,
        pad,
        polygon_area,
        via,
        wire,
        zone,
    )
    from .pcb_types import BoardElement, CuLayer, Point
    from .stackup_file import read_stackup_file
except ImportError:
    from board_model import (
        NetTie,
        Selected,
        build_cu_stack,
        net_tie,
        pad,
        polygon_area,
        via,
        wire,
        zone,
    )
    from pcb_types import BoardElement, CuLayer, Point
    from stackup_file import read_stackup_file


def _m(value: int) -> float:
    """pcbnew's internal unit (nm) → m."""
    return value / 1e9


def _point(vector: Any) -> Point:
    return (_m(vector.x), _m(vector.y))


def _id(item: Any) -> str:
    return str(item.m_Uuid.AsString())


def _outline(polygons: Any) -> list[Point]:
    """First outline of a SHAPE_POLY_SET."""
    if not polygons.OutlineCount():
        return []
    chain = polygons.Outline(0)
    return [_point(chain.CPoint(i)) for i in range(chain.PointCount())]


def _selected(item: Any) -> Selected:
    if isinstance(item, pcbnew.FOOTPRINT):
        return Selected(_id(item), pads=tuple(map(_selected, item.Pads())))
    if isinstance(item, (pcbnew.PCB_TRACK, pcbnew.PAD)):  # PCB_TRACK: also arcs, vias
        return Selected(_id(item), item.GetNetname())
    return Selected(_id(item))


class PcbnewBoard:
    def __init__(self, board: pcbnew.BOARD) -> None:
        self._board = board
        self._enabled = set(board.GetEnabledLayers().CuStack())

    def _layers(self, item: Any) -> list[int]:
        return sorted(set(item.GetLayerSet().CuStack()) & self._enabled)

    def selection(self) -> list[Selected]:
        board = self._board
        # Selecting a footprint flags its pads too; the footprint stands for them.
        pads = [p for p in board.GetPads() if not p.GetParentFootprint().IsSelected()]
        items = [
            *board.GetTracks(),
            *pads,
            *board.Zones(),
            *board.GetFootprints(),
            *board.GetDrawings(),
        ]
        return [_selected(item) for item in items if item.IsSelected()]

    def net_ties(self) -> list[NetTie]:
        return [
            net_tie(
                fp.GetReference(),
                # The pad groups themselves are not wrapped for Python.
                [[q.GetNumber() for q in fp.GetNetTiePads(p)] for p in fp.Pads()],
                {p.GetNumber(): p.GetNetname() for p in fp.Pads()},
                fp,
            )
            for fp in self._board.GetFootprints()
            if fp.IsNetTie()
        ]

    def copper(
        self, nets: AbstractSet[str], net_ties: Sequence[NetTie]
    ) -> Iterator[BoardElement]:
        board = self._board
        for track in board.GetTracks():
            if track.GetNetname() in nets:
                yield self._track(track)
        for p in board.GetPads():
            if p.GetNetname() in nets:
                yield self._pad(p)
        for z in board.Zones():
            if z.GetNetname() in nets and not (z.GetIsRuleArea() or z.IsTeardropArea()):
                yield zone(
                    _id(z),
                    z.GetNetname(),
                    self._layers(z),
                    _outline(z.Outline()),
                    z.GetFilledArea() / 1e18,  # nm² → m²
                )
        for tie in net_ties:
            for shape in tie.footprint.GraphicalItems():
                yield from self._shape(shape)

    def _track(self, track: Any) -> BoardElement:
        args = (_id(track), track.GetNetname(), self._layers(track))
        if isinstance(track, pcbnew.PCB_VIA):
            position = _point(track.GetPosition())
            return via(
                *args,
                position,
                _m(track.GetWidth(track.TopLayer())),
                _m(track.GetDrillValue()),
            )
        mid = _point(track.GetMid()) if isinstance(track, pcbnew.PCB_ARC) else None
        start, end = _point(track.GetStart()), _point(track.GetEnd())
        return wire(*args, start, end, _m(track.GetWidth()), mid)

    def _pad(self, p: Any) -> BoardElement:
        layers = self._layers(p)
        outline = _outline(p.GetEffectivePolygon(layers[0])) if layers else []
        position = _point(p.GetPosition())
        drill = _m(p.GetDrillSizeX())
        return pad(_id(p), p.GetNetname(), layers, position, drill, outline)

    def _shape(self, shape: Any) -> Iterator[BoardElement]:
        """Net Tie copper: segments and arcs as wires, polygons as zones."""
        if not isinstance(shape, pcbnew.PCB_SHAPE):
            return  # e.g. a text
        layers = self._layers(shape)
        if not layers:
            return
        kind = shape.GetShape()
        args = (_id(shape), "", layers)
        if kind == pcbnew.SHAPE_T_POLY:
            outline = _outline(shape.GetPolyShape())
            yield zone(*args, outline, polygon_area(outline))
        elif kind in (pcbnew.SHAPE_T_SEGMENT, pcbnew.SHAPE_T_ARC):
            mid = _point(shape.GetArcMid()) if kind == pcbnew.SHAPE_T_ARC else None
            start, end = _point(shape.GetStart()), _point(shape.GetEnd())
            yield wire(*args, start, end, _m(shape.GetWidth()), mid)

    def stackup(self) -> dict[int, CuLayer]:
        board_file = Path(self._board.GetFileName())
        thickness = _m(self._board.GetDesignSettings().GetBoardThickness())
        return build_cu_stack(read_stackup_file(board_file), thickness)
