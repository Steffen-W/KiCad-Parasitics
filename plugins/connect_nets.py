from __future__ import annotations

from collections import defaultdict
from typing import Any

try:
    from .pcb_types import WIRE, ZONE
except ImportError:
    from pcb_types import WIRE, ZONE

# Node IDs are > 0; 0 marks an endpoint without a node.
NOT_ASSIGNED = 0

# (element id, "net_start" | "net_end", layer)
Endpoint = tuple[Any, str, int]

# Per endpoint: (node key, connection list key, position key)
_Sides = tuple[tuple[str, str, str], ...]
_WIRE_SIDES: _Sides = (
    ("net_start", "conn_start", "start"),
    ("net_end", "conn_end", "end"),
)
_POINT_SIDES: _Sides = (("net_start", "conn_start", "position"),)


def Connect_Nets(data: dict[Any, Any]) -> dict[Any, dict[str, Any]]:
    """Assign a node ID to every element endpoint; touching endpoints share one.

    `data` holds connected board elements (see pcb_types.BoardElement), which
    gain the node IDs as net_start/net_end: {layer: node}.

    Wires have a start and an end endpoint per layer, pads and vias a single one
    ("net_end" mirrors "net_start"). Zones are connected later in get_parasitic.
    Two endpoints touch if one element lists the other in its conn_start/conn_end;
    the side of a connected wire is taken from its own lists or, if missing or
    listed at both ends, from matching coordinates. Node IDs already present are
    kept; the grouping does not depend on the element order.

    Raises:
        ValueError: touching endpoints carry different preset node IDs
    """
    parent: dict[Endpoint, Endpoint] = {}

    def find(e: Endpoint) -> Endpoint:
        parent.setdefault(e, e)
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    def union(a: Endpoint, b: Endpoint) -> None:
        parent[find(a)] = find(b)

    def touching_endpoint(
        conn: Any, uuid: Any, layer: int, pos: Any
    ) -> Endpoint | None:
        """Endpoint of element `conn` on `layer` that touches element `uuid` at `pos`."""
        c = data.get(conn)
        if c is None or c["type"] == ZONE or layer not in c.get("layer", []):
            return None
        if c["type"] != WIRE:
            return (conn, "net_start", layer)
        listed = [side for side, key, _ in _WIRE_SIDES if uuid in c.get(key, [])]
        if len(listed) == 1:
            return (conn, listed[0], layer)
        # Not listed, or listed at both ends (two wires closing a loop).
        for side, _, pos_key in _WIRE_SIDES:
            if pos is not None and pos == c.get(pos_key):
                return (conn, side, layer)
        return None

    endpoints: list[Endpoint] = []
    preassigned: dict[int, Endpoint] = {}
    for uuid, d in data.items():
        d.setdefault("layer", [])
        if d["type"] == ZONE:
            continue
        sides = _WIRE_SIDES if d["type"] == WIRE else _POINT_SIDES
        for layer in d["layer"]:
            for side, conn_key, pos_key in sides:
                own: Endpoint = (uuid, side, layer)
                endpoints.append(own)
                find(own)
                node = d.get(side, {}).get(layer, NOT_ASSIGNED)
                if node > NOT_ASSIGNED:
                    union(own, preassigned.setdefault(node, own))
                for conn in d.get(conn_key, []):
                    other = touching_endpoint(conn, uuid, layer, d.get(pos_key))
                    if other is not None:
                        union(own, other)

    node_of_root: dict[Endpoint, int] = {}
    for node, e in preassigned.items():
        root = find(e)
        if node_of_root.setdefault(root, node) != node:
            raise ValueError(
                f"Touching endpoints carry different preset nodes "
                f"{node_of_root[root]} and {node} (at {e})"
            )
    next_node = max(preassigned, default=NOT_ASSIGNED) + 1
    for uuid, side, layer in endpoints:
        root = find((uuid, side, layer))
        if root not in node_of_root:
            node_of_root[root] = next_node
            next_node += 1
    for d in data.values():
        d["net_start"] = defaultdict(lambda: NOT_ASSIGNED)
        d["net_end"] = defaultdict(lambda: NOT_ASSIGNED)
    for uuid, side, layer in endpoints:
        data[uuid][side][layer] = node_of_root[find((uuid, side, layer))]
    for d in data.values():
        if d["type"] != WIRE:
            d["net_end"].update(d["net_start"])

    return data
