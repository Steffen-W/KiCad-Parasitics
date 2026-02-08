"""Get PCB elements via kipy IPC API (KiCad 9+).

TODO:
- PolygonWithHoles: _linearize_polyline arc handling could be improved for edge cases
"""

import logging
import math

from kipy.board import Board
from kipy.board_types import ArcTrack, FootprintInstance
from kipy.util.board_layer import canonical_name, is_copper_layer

from Get_PCB_Stackup import extract_layer_from_string

log = logging.getLogger(__name__)


def _linearize_polyline(polyline, arc_steps=8):
    """Convert PolyLine nodes (points + arcs) to a list of (x, y) points in meters."""
    pts = []
    for node in polyline.nodes:
        if node.has_point:
            pts.append((_to_m(node.point.x), _to_m(node.point.y)))
        elif node.has_arc:
            arc = node.arc
            center = arc.center()
            if center is None:
                pts.append((_to_m(arc.start.x), _to_m(arc.start.y)))
                pts.append((_to_m(arc.end.x), _to_m(arc.end.y)))
                continue
            r = arc.radius()
            a_start = arc.start_angle()
            a_end = arc.end_angle()
            if a_start is None or a_end is None:
                pts.append((_to_m(arc.start.x), _to_m(arc.start.y)))
                pts.append((_to_m(arc.end.x), _to_m(arc.end.y)))
                continue
            # Determine sweep direction via mid point
            cx, cy = center.x, center.y
            sweep = a_end - a_start
            # Normalize to (-360, 360) first, then use mid to pick direction
            sweep = sweep % 360  # [0, 360)
            # Check: does the CCW sweep from a_start pass through mid?
            a_mid_actual = math.degrees(math.atan2(arc.mid.y - cy, arc.mid.x - cx))
            # Normalize mid angle relative to a_start
            mid_rel = (a_mid_actual - a_start) % 360
            if mid_rel > sweep:
                # Mid is not within the CCW sweep → use CW (negative) sweep
                sweep = sweep - 360
            cx_m, cy_m = _to_m(cx), _to_m(cy)
            r_m = _to_m(r)
            for k in range(arc_steps + 1):
                a = math.radians(a_start + sweep * k / arc_steps)
                pts.append((cx_m + r_m * math.cos(a), cy_m + r_m * math.sin(a)))
    return pts


def _shoelace_area(pts):
    """Signed area of polygon using shoelace formula (in m²)."""
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return area / 2


def _polygon_area(poly):
    """Compute area of a PolygonWithHoles in m² (outline - holes)."""
    outline_pts = _linearize_polyline(poly.outline)
    area = abs(_shoelace_area(outline_pts))
    for hole in poly.holes:
        hole_pts = _linearize_polyline(hole)
        area -= abs(_shoelace_area(hole_pts))
    return max(area, 0.0)


def _to_m(nm):
    """Nanometer (kipy) → Meter."""
    return nm / 1e9


def _pos_m(vec2):
    """Vector2 → (x, y) in meters."""
    return (_to_m(vec2.x), _to_m(vec2.y))


def _layer_v9(bl):
    """BoardLayer enum → v9 layer int (same numbering as Get_PCB_Stackup)."""
    return extract_layer_from_string(canonical_name(bl))


def _obj_id(obj):
    return obj.id.value


def _arc_geometry(start, end, mid):
    """Compute arc radius, sweep angle and arc length from three points (m).

    Returns (radius, angle, arc_length) where angle is the signed sweep
    in radians (positive = CCW, negative = CW).
    Falls back to (None, 0, chord) if the three points are collinear.
    """
    sx, sy = start
    mx, my = mid
    ex, ey = end
    D = 2 * (sx * (my - ey) + mx * (ey - sy) + ex * (sy - my))
    if abs(D) < 1e-30:
        # Collinear
        chord = math.hypot(ex - sx, ey - sy)
        return None, 0.0, chord

    ux = (
        (sx * sx + sy * sy) * (my - ey)
        + (mx * mx + my * my) * (ey - sy)
        + (ex * ex + ey * ey) * (sy - my)
    ) / D
    uy = (
        (sx * sx + sy * sy) * (ex - mx)
        + (mx * mx + my * my) * (sx - ex)
        + (ex * ex + ey * ey) * (mx - sx)
    ) / D

    r = math.hypot(sx - ux, sy - uy)

    a_start = math.atan2(sy - uy, sx - ux)
    a_mid = math.atan2(my - uy, mx - ux)
    a_end = math.atan2(ey - uy, ex - ux)

    def _norm(a, ref):
        while a < ref:
            a += 2 * math.pi
        while a >= ref + 2 * math.pi:
            a -= 2 * math.pi
        return a

    a_mid_ccw = _norm(a_mid, a_start)
    a_end_ccw = _norm(a_end, a_start)
    if a_mid_ccw < a_end_ccw:
        sweep = a_end_ccw - a_start
    else:
        a_start_cw = _norm(a_start, a_end)
        sweep = -(a_start_cw - a_end)

    arc_length = abs(sweep) * r
    return r, sweep, arc_length


def _cu_layers(layers, enabled_layers=None):
    """Filter to copper layers and convert to v9 ints."""
    result = []
    for bl in layers:
        if is_copper_layer(bl):
            if enabled_layers is not None and bl not in enabled_layers:
                continue
            v9 = _layer_v9(bl)
            if v9 is not None:
                result.append(v9)
    return sorted(result)


def _expand_nets_via_net_ties(board, initial_nets):
    """Expand a set of nets transitively through all Net Tie footprints.

    For each net, finds Net Tie footprints with pads on that net, adds the
    bridged nets, and repeats until no new nets are found.

    Returns:
        (expanded_nets, net_tie_fps): set of all reachable net names,
            list of Net Tie FootprintInstances involved.
    """
    expanded = set(initial_nets)
    net_tie_fps = []
    seen_fp_ids = set()

    # Build net-tie index once: list of (fp, group_nets)
    net_tie_index = []
    for fp in board.get_footprints():
        net_ties = fp.definition._proto.net_ties
        if len(net_ties) == 0:
            continue
        pad_net_map = {p.number: p.net.name for p in fp.definition.pads}
        for nt in net_ties:
            group_pads = []
            for entry in nt.pad_number:
                group_pads.extend(entry.split(","))
            group_nets = {pad_net_map[pn] for pn in group_pads if pn in pad_net_map}
            if group_nets:
                net_tie_index.append((fp, group_nets))

    changed = True
    while changed:
        changed = False
        for fp, group_nets in net_tie_index:
            fp_id = fp.id.value
            if fp_id in seen_fp_ids:
                continue
            if group_nets & expanded:
                if not group_nets.issubset(expanded):
                    expanded |= group_nets
                    changed = True
                seen_fp_ids.add(fp_id)
                net_tie_fps.append(fp)

    if net_tie_fps:
        refs = []
        for fp in net_tie_fps:
            ref = fp.reference_field.text.value if fp.reference_field else "?"
            refs.append(ref)
        log.info(
            "Net-tie expansion: %s -> %s (via %s)",
            initial_nets,
            expanded,
            ", ".join(refs),
        )

    return expanded, net_tie_fps


def _collect_footprint_shapes(fp, enabled_layers):
    """Collect copper shapes from a footprint as WIRE elements.

    Returns:
        dict of oid -> element dict (same format as ItemList entries)
    """
    items = {}

    for shape in fp.definition.shapes:
        if not is_copper_layer(shape.layer):
            continue
        if enabled_layers is not None and shape.layer not in enabled_layers:
            continue

        layer_v9 = _layer_v9(shape.layer)
        if layer_v9 is None:
            continue

        oid = _obj_id(shape)
        type_name = type(shape).__name__
        width = _to_m(shape.attributes.stroke.width)

        if type_name == "BoardSegment":
            start = _pos_m(shape.start)
            end = _pos_m(shape.end)
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length == 0:
                continue
            items[oid] = {
                "type": "WIRE",
                "id": oid,
                "start": start,
                "end": end,
                "width": width,
                "length": length,
                "area": width * length,
                "layer": [layer_v9],
                "net_name": "",
                "net_code": "",
                "is_selected": False,
                "conn_start": [],
                "conn_end": [],
            }
        elif type_name == "BoardArc":
            start = _pos_m(shape.start)
            end = _pos_m(shape.end)
            mid = _pos_m(shape.mid)
            r_m, angle, arc_length = _arc_geometry(start, end, mid)
            if arc_length == 0:
                continue
            item = {
                "type": "WIRE",
                "id": oid,
                "start": start,
                "end": end,
                "mid": mid,
                "angle": angle,
                "width": width,
                "length": arc_length,
                "area": width * arc_length,
                "layer": [layer_v9],
                "net_name": "",
                "net_code": "",
                "is_selected": False,
                "conn_start": [],
                "conn_end": [],
            }
            if r_m is not None:
                item["radius"] = r_m
            items[oid] = item

    return items


def Get_PCB_Elements_IPC(board: Board):
    """Collect all tracks/vias/pads/zones from the board via IPC API.

    Returns ItemList in the same dict format as
    Get_PCB_Elements.Get_PCB_Elements().
    """
    selection_ids = {item.id.value for item in board.get_selection()}
    log.info("Selection: %d items", len(selection_ids))

    # Get enabled copper layers from board
    enabled_layers = set(board.get_enabled_layers())

    # Determine target nets from selection
    selection = list(board.get_selection())

    # Fallback: if a single footprint with 2 pads is selected, use its pads
    if len(selection) == 1 and isinstance(selection[0], FootprintInstance):
        fp_pads = list(selection[0].definition.pads)
        if len(fp_pads) == 2:
            log.info("Footprint selected, using its 2 pads")
            selection = fp_pads
            selection_ids = {pad.id.value for pad in fp_pads}

    nets = {item.net.name for item in selection if hasattr(item, "net")}

    if len(selection) != 2:
        log.warning("Expected 2 selected items, got %d", len(selection))
        print(f"Error: Please select exactly 2 elements (selected: {len(selection)})")
        return {}

    # Expand nets transitively through Net Tie footprints
    target_nets, net_tie_fps = _expand_nets_via_net_ties(board, nets)

    if len(nets) > 1 and not net_tie_fps:
        # Multiple nets selected but no net-tie bridges them
        log.warning("Selected items belong to different nets: %s", nets)
        print(f"Error: Selected items must belong to the same net (found: {nets})")
        return {}

    log.info("Target nets: %s", target_nets)

    ItemList = {}

    # --- Tracks & Arcs ---
    for track in board.get_tracks():
        if track.net.name not in target_nets:
            continue

        oid = _obj_id(track)
        start = _pos_m(track.start)
        end = _pos_m(track.end)
        width = _to_m(track.width)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length == 0:
            continue

        temp = {
            "type": "WIRE",
            "id": oid,
            "start": start,
            "end": end,
            "width": width,
            "length": length,
            "area": width * length,
            "layer": [v for v in [_layer_v9(track.layer)] if v is not None],
            "net_name": track.net.name,
            "net_code": track.net.name,
            "is_selected": track.id.value in selection_ids,
            "conn_start": [],
            "conn_end": [],
        }

        if isinstance(track, ArcTrack):
            mid = _pos_m(track.mid)
            r_m, angle, arc_length = _arc_geometry(start, end, mid)
            temp["mid"] = mid
            temp["angle"] = angle
            temp["length"] = arc_length
            temp["area"] = width * arc_length
            if r_m is not None:
                temp["radius"] = r_m

        ItemList[oid] = temp

    # --- Vias ---
    for via in board.get_vias():
        if via.net.name not in target_nets:
            continue

        oid = _obj_id(via)
        layers = _cu_layers(via.padstack.layers, enabled_layers)

        ItemList[oid] = {
            "type": "VIA",
            "id": oid,
            "position": _pos_m(via.position),
            "width": _to_m(via.diameter),
            "drill": _to_m(via.drill_diameter),
            "layer": layers,
            "area": 0,
            "net_name": via.net.name,
            "net_code": via.net.name,
            "is_selected": via.id.value in selection_ids,
            "conn_start": [],
        }

    # --- Pads ---
    pad_objects = {}  # oid -> pad object, for area calculation
    for pad in board.get_pads():
        if pad.net.name not in target_nets:
            continue

        oid = _obj_id(pad)
        layers = _cu_layers(pad.padstack.layers, enabled_layers)
        drill_d = (
            _to_m(pad.padstack.drill.diameter.x) if pad.padstack.drill else 0.0
        )  # diameter is Vector2, use x for round drills
        pad_layer = (
            pad.padstack.copper_layers[0] if pad.padstack.copper_layers else None
        )
        size = (
            (_to_m(pad_layer.size.x), _to_m(pad_layer.size.y)) if pad_layer else (0, 0)
        )
        shape = pad_layer.shape if pad_layer else 0
        pad_objects[oid] = (pad, layers)

        ItemList[oid] = {
            "type": "PAD",
            "id": oid,
            "position": _pos_m(pad.position),
            "size": size,
            "orientation": pad.padstack.angle.degrees,
            "drill_size": (drill_d, drill_d),
            "drill": drill_d,
            "shape": shape,
            "layer": layers,
            "area": 0,
            "PadName": pad.number,
            "net_name": pad.net.name,
            "net_code": pad.net.name,
            "is_selected": pad.id.value in selection_ids,
            "conn_start": [],
        }

    # Compute pad areas via board.get_pad_shapes_as_polygons()
    for oid, (pad, layers) in pad_objects.items():
        if not layers:
            continue
        # Use first copper layer for polygon shape
        first_layer = [bl for bl in pad.padstack.layers if is_copper_layer(bl)]
        if not first_layer:
            continue
        try:
            poly = board.get_pad_shapes_as_polygons(pad, layer=first_layer[0])
            if poly is not None:
                ItemList[oid]["area"] = _polygon_area(poly)
        except Exception:
            log.debug("Failed to get pad polygon for %s", oid)

    # --- Zones ---
    for zone in board.get_zones():
        if zone.is_rule_area():
            continue
        if not hasattr(zone, "net") or zone.net is None:
            continue
        if zone.net.name not in target_nets:
            continue
        if "teardrop" in (zone.name or ""):
            continue

        oid = _obj_id(zone)
        layers = _cu_layers(zone.layers, enabled_layers)
        bbox = zone.bounding_box()
        center = (
            _to_m(bbox.pos.x + bbox.size.x // 2),
            _to_m(bbox.pos.y + bbox.size.y // 2),
        )
        outline_pts = []
        for node in zone.outline.outline.nodes:
            if node.has_point:
                outline_pts.append((_to_m(node.point.x), _to_m(node.point.y)))
        num_corners = len(outline_pts)

        ItemList[oid] = {
            "type": "ZONE",
            "id": oid,
            "position": center,
            "layer": layers,
            "area": sum(
                _polygon_area(p)
                for polys in zone.filled_polygons.values()
                for p in polys
            ),
            "NumCorners": num_corners,
            "ZoneName": zone.name or "",
            "net_name": zone.net.name,
            "net_code": zone.net.name,
            "is_selected": zone.id.value in selection_ids,
            "conn_start": [],
            "_outline": outline_pts,  # used by _build_connectivity, removed after
        }

    # --- Footprint copper shapes (Net Tie) ---
    for fp in net_tie_fps:
        fp_items = _collect_footprint_shapes(fp, enabled_layers)
        log.info(
            "Collected %d copper shapes from footprint %s",
            len(fp_items),
            fp.reference_field.text.value if fp.reference_field else "?",
        )
        ItemList.update(fp_items)

    log.info("Collected %d elements for nets %s", len(ItemList), target_nets)

    # --- Build connectivity from coordinates ---
    _build_connectivity(ItemList)

    # Store net-tie metadata (after _build_connectivity to avoid interfering)
    if net_tie_fps:
        refs = [
            fp.reference_field.text.value if fp.reference_field else "?"
            for fp in net_tie_fps
        ]
        ItemList["_net_tie_info"] = {
            "refs": refs,
            "nets": sorted(target_nets),
        }

    return ItemList


def _point_in_polygon(point, polygon):
    """Ray casting algorithm to check if point is inside polygon."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_intersect:
                            inside = not inside
                    elif p1x == p2x:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _point_to_segment_dist_sq(px, py, ax, ay, bx, by):
    """Squared distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        dx2, dy2 = px - ax, py - ay
        return dx2 * dx2 + dy2 * dy2
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    dx2, dy2 = px - proj_x, py - proj_y
    return dx2 * dx2 + dy2 * dy2


def _segments_dist_sq(ax, ay, bx, by, cx, cy, dx, dy):
    """Squared distance between segment (a,b) and segment (c,d)."""
    return min(
        _point_to_segment_dist_sq(ax, ay, cx, cy, dx, dy),
        _point_to_segment_dist_sq(bx, by, cx, cy, dx, dy),
        _point_to_segment_dist_sq(cx, cy, ax, ay, bx, by),
        _point_to_segment_dist_sq(dx, dy, ax, ay, bx, by),
    )


def _point_near_polygon(point, polygon, margin):
    """Check if point is inside polygon or within margin of its edges."""
    if _point_in_polygon(point, polygon):
        return True
    if margin <= 0:
        return False
    px, py = point
    margin_sq = margin * margin
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if _point_to_segment_dist_sq(px, py, ax, ay, bx, by) <= margin_sq:
            return True
    return False


def _compute_bbox(d):
    """Compute bounding box (min_x, min_y, max_x, max_y) for an element."""
    t = d["type"]
    if t == "WIRE":
        w = d.get("width", 0) / 2
        x0, y0 = d["start"]
        x1, y1 = d["end"]
        return (min(x0, x1) - w, min(y0, y1) - w, max(x0, x1) + w, max(y0, y1) + w)
    elif t in ("VIA", "PAD"):
        px, py = d["position"]
        if t == "PAD":
            sx, sy = d.get("size", (0, 0))
            # For rotated pads, bbox must enclose the rotated rectangle
            angle = math.radians(d.get("orientation", 0))
            if angle != 0:
                cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
                rx = (sx * cos_a + sy * sin_a) / 2
                ry = (sx * sin_a + sy * cos_a) / 2
            else:
                rx, ry = sx / 2, sy / 2
        else:
            rx = ry = d.get("width", d.get("drill", 0)) / 2
        return (px - rx, py - ry, px + rx, py + ry)
    elif t == "ZONE":
        outline = d.get("_outline", [])
        if outline:
            xs = [p[0] for p in outline]
            ys = [p[1] for p in outline]
            return (min(xs), min(ys), max(xs), max(ys))
    return (0, 0, 0, 0)


def _bboxes_overlap(a, b):
    """Check if two bounding boxes overlap."""
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _point_in_pad(point, pad_data, margin=0):
    """Check if point is within pad shape (circle/rect/oval) + margin.

    PadStackShape: 1=circle, 2=rect, 3=oval, 5=roundrect, 6=chamferedrect.
    Falls back to rectangle for unknown shapes.
    Handles pad rotation by rotating the test point into pad-local coordinates.
    """
    px, py = point
    cx, cy = pad_data["position"]
    sx, sy = pad_data.get("size", (0, 0))
    rx, ry = sx / 2 + margin, sy / 2 + margin
    shape = pad_data.get("shape", 0)

    # Rotate point into pad-local coordinate system
    dx, dy = px - cx, py - cy
    angle = math.radians(pad_data.get("orientation", 0))
    if angle != 0:
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        dx, dy = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
    dx, dy = abs(dx), abs(dy)

    if shape == 1:  # circle
        r = max(rx, ry)
        return dx * dx + dy * dy <= r * r
    elif shape == 3:  # oval
        if rx > ry:
            straight = rx - ry
            if dx <= straight:
                return dy <= ry
            return (dx - straight) ** 2 + dy * dy <= ry * ry
        else:
            straight = ry - rx
            if dy <= straight:
                return dx <= rx
            return dx * dx + (dy - straight) ** 2 <= rx * rx
    else:  # rectangle, roundrect, chamferedrect, unknown -> rect
        return dx <= rx and dy <= ry


def _build_connectivity(ItemList):
    """Build conn_start/conn_end by matching coordinates.

    Uses bounding boxes for fast pre-filtering, then:
    - Point-to-point: distance <= r_i + r_j (sum of radii)
    - Point-in-pad: shape-aware test (circle/rect/oval)
    - Point-in-zone: ray casting with margin
    """
    # Pre-compute bounding boxes and layer sets for all elements
    bboxes = {}
    layer_sets = {}
    for oid, d in ItemList.items():
        bboxes[oid] = _compute_bbox(d)
        layer_sets[oid] = set(d.get("layer", []))

    # Categorize elements
    wires = {oid: d for oid, d in ItemList.items() if d["type"] == "WIRE"}
    pads = {oid: d for oid, d in ItemList.items() if d["type"] == "PAD"}
    zones = {
        oid: d
        for oid, d in ItemList.items()
        if d["type"] == "ZONE" and len(d.get("_outline", [])) >= 3
    }

    # Collect connection points: (position, oid, endpoint_type, radius)
    points = []
    for oid, d in ItemList.items():
        if d["type"] == "WIRE":
            r = d.get("width", 0) / 2
            points.append((d["start"], oid, "start", r))
            points.append((d["end"], oid, "end", r))
        elif d["type"] == "VIA":
            r = d.get("width", d.get("drill", 0)) / 2
            points.append((d["position"], oid, "position", r))
        elif d["type"] == "PAD":
            sx, sy = d.get("size", (0, 0))
            r = max(sx, sy) / 2 if sx or sy else d.get("drill", 0) / 2
            points.append((d["position"], oid, "position", r))

    def _add_conn(oid_a, ep_a, oid_b, ep_b):
        """Add bidirectional connection."""
        key_a = "conn_start" if ep_a in ("start", "position") else "conn_end"
        if oid_b not in ItemList[oid_a][key_a]:
            ItemList[oid_a][key_a].append(oid_b)
        key_b = "conn_start" if ep_b in ("start", "position") else "conn_end"
        if oid_a not in ItemList[oid_b][key_b]:
            ItemList[oid_b][key_b].append(oid_a)

    # Point-to-point matching (Wire endpoints, Via/Pad positions)
    for i in range(len(points)):
        pos_i, oid_i, ep_i, r_i = points[i]
        for j in range(i + 1, len(points)):
            pos_j, oid_j, ep_j, r_j = points[j]
            if oid_i == oid_j:
                continue

            tol = r_i + r_j + 1e-9
            dx = pos_i[0] - pos_j[0]
            dy = pos_i[1] - pos_j[1]
            if abs(dx) > tol or abs(dy) > tol:
                continue
            if dx * dx + dy * dy > tol * tol:
                continue

            _add_conn(oid_i, ep_i, oid_j, ep_j)

    # Pad connectivity: check if wire endpoints lie within pad shape
    for pad_oid, pad_d in pads.items():
        pad_bbox = bboxes[pad_oid]
        pad_ls = layer_sets[pad_oid]
        for pos, pt_oid, ep, r in points:
            if pt_oid == pad_oid:
                continue
            conn_key = "conn_start" if ep in ("start", "position") else "conn_end"
            if pad_oid in ItemList[pt_oid].get(conn_key, []):
                continue
            if not pad_ls & layer_sets[pt_oid]:
                continue
            pt_bbox = (pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r)
            if not _bboxes_overlap(pad_bbox, pt_bbox):
                continue
            if _point_in_pad(pos, pad_d, margin=r):
                _add_conn(pt_oid, ep, pad_oid, "position")

    # Wire-body connectivity: T-junctions, wire-through-pad, wire-through-zone
    for wire_oid, wire_d in wires.items():
        wire_bbox = bboxes[wire_oid]
        wire_ls = layer_sets[wire_oid]
        wr = wire_d.get("width", 0) / 2
        sx, sy = wire_d["start"]
        ex, ey = wire_d["end"]

        # T-junction: points landing on this wire's body
        for pos, pt_oid, ep, r in points:
            if pt_oid == wire_oid:
                continue
            conn_key = "conn_start" if ep in ("start", "position") else "conn_end"
            if wire_oid in ItemList[pt_oid].get(conn_key, []):
                continue
            if not wire_ls & layer_sets[pt_oid]:
                continue
            pt_bbox = (pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r)
            if not _bboxes_overlap(wire_bbox, pt_bbox):
                continue
            thr = (wr + r) ** 2
            if _point_to_segment_dist_sq(pos[0], pos[1], sx, sy, ex, ey) <= thr:
                ds = (pos[0] - sx) ** 2 + (pos[1] - sy) ** 2
                de = (pos[0] - ex) ** 2 + (pos[1] - ey) ** 2
                wire_ep = "start" if ds <= de else "end"
                _add_conn(pt_oid, ep, wire_oid, wire_ep)

        # Wire-through-pad: wire body crosses pad without endpoint nearby
        for pad_oid, pad_d in pads.items():
            if pad_oid in wire_d["conn_start"] or pad_oid in wire_d["conn_end"]:
                continue
            if not wire_ls & layer_sets[pad_oid]:
                continue
            if not _bboxes_overlap(wire_bbox, bboxes[pad_oid]):
                continue
            px, py = pad_d["position"]
            pad_sx, pad_sy = pad_d.get("size", (0, 0))
            pad_r = max(pad_sx, pad_sy) / 2
            thr = (wr + pad_r) ** 2
            if _point_to_segment_dist_sq(px, py, sx, sy, ex, ey) <= thr:
                ds = (sx - px) ** 2 + (sy - py) ** 2
                de = (ex - px) ** 2 + (ey - py) ** 2
                wire_ep = "start" if ds <= de else "end"
                _add_conn(wire_oid, wire_ep, pad_oid, "position")

        # Wire-through-zone: wire body crosses zone without endpoint inside
        for zone_oid, zone_d in zones.items():
            if zone_oid in wire_d["conn_start"] or zone_oid in wire_d["conn_end"]:
                continue
            if not wire_ls & layer_sets[zone_oid]:
                continue
            if not _bboxes_overlap(wire_bbox, bboxes[zone_oid]):
                continue
            outline = zone_d["_outline"]
            n = len(outline)
            thr = wr * wr
            hit = False
            for i in range(n):
                ox, oy = outline[i]
                px, py = outline[(i + 1) % n]
                if _segments_dist_sq(sx, sy, ex, ey, ox, oy, px, py) <= thr:
                    hit = True
                    break
            if hit:
                zx = sum(p[0] for p in outline) / n
                zy = sum(p[1] for p in outline) / n
                ds = (sx - zx) ** 2 + (sy - zy) ** 2
                de = (ex - zx) ** 2 + (ey - zy) ** 2
                wire_ep = "start" if ds <= de else "end"
                conn_key = "conn_start" if wire_ep == "start" else "conn_end"
                if zone_oid not in wire_d[conn_key]:
                    wire_d[conn_key].append(zone_oid)
                if wire_oid not in zone_d["conn_start"]:
                    zone_d["conn_start"].append(wire_oid)

    # Zone connectivity: check if endpoints lie inside zone outlines
    for zone_oid, zone_d in zones.items():
        outline = zone_d["_outline"]
        zone_bbox = bboxes[zone_oid]
        zone_ls = layer_sets[zone_oid]
        for pos, pt_oid, ep, r in points:
            if not zone_ls & layer_sets[pt_oid]:
                continue
            pt_bbox = (pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r)
            if not _bboxes_overlap(zone_bbox, pt_bbox):
                continue
            if _point_near_polygon(pos, outline, r):
                conn_key = "conn_start" if ep in ("start", "position") else "conn_end"
                if zone_oid not in ItemList[pt_oid][conn_key]:
                    ItemList[pt_oid][conn_key].append(zone_oid)
                if pt_oid not in zone_d["conn_start"]:
                    zone_d["conn_start"].append(pt_oid)

    # Remove internal data
    for d in ItemList.values():
        d.pop("_outline", None)
