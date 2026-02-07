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


def Get_PCB_Elements_IPC(board: Board):
    """Collect all tracks/vias/pads/zones from the board via IPC API.

    Returns ItemList in the same dict format as
    Get_PCB_Elements.Get_PCB_Elements().
    """
    selection_ids = {item.id.value for item in board.get_selection()}
    log.info("Selection: %d items", len(selection_ids))

    # Get enabled copper layers from board
    enabled_layers = set(board.get_enabled_layers())

    # Determine target net from selection (exactly 2 items with same net required)
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

    if len(nets) != 1:
        log.warning("Selected items belong to different nets: %s", nets)
        print(f"Error: Selected items must belong to the same net (found: {nets})")
        return {}

    target_net = nets.pop()
    log.info("Target net: %s", target_net)

    ItemList = {}

    # --- Tracks & Arcs ---
    for track in board.get_tracks():
        if track.net.name != target_net:
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
            r = track.radius()
            if r is not None:
                r_m = _to_m(r)
                temp["radius"] = r_m
                # Arc length using mid point to determine if major or minor arc
                chord = length
                if chord <= 2 * r_m and r_m > 0:
                    half_angle = math.asin(chord / (2 * r_m))
                    # Check if mid point is on the minor arc side
                    mid = _pos_m(track.mid)
                    sx, sy = start
                    ex, ey = end
                    mid_chord_x = (sx + ex) / 2
                    mid_chord_y = (sy + ey) / 2
                    d_mid = math.hypot(mid[0] - mid_chord_x, mid[1] - mid_chord_y)
                    # If mid is far from chord midpoint, it's a major arc (>180°)
                    if d_mid > r_m:
                        arc_length = 2 * r_m * (math.pi - half_angle)
                    else:
                        arc_length = 2 * r_m * half_angle
                    temp["length"] = arc_length
                    temp["area"] = width * arc_length

        ItemList[oid] = temp

    # --- Vias ---
    for via in board.get_vias():
        if via.net.name != target_net:
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
        if pad.net.name != target_net:
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
        if zone.net.name != target_net:
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

    log.info("Collected %d elements for net %s", len(ItemList), target_net)

    # --- Build connectivity from coordinates ---
    _build_connectivity(ItemList)

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
    # Compute bounding boxes for all elements
    bboxes = {}
    for oid, d in ItemList.items():
        bboxes[oid] = _compute_bbox(d)

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
    pads = {oid: d for oid, d in ItemList.items() if d["type"] == "PAD"}

    for pad_oid, pad_d in pads.items():
        pad_bbox = bboxes[pad_oid]
        pad_layers = set(pad_d["layer"])
        for pos, pt_oid, ep, r in points:
            if pt_oid == pad_oid:
                continue
            if pad_oid in ItemList[pt_oid].get(
                "conn_start" if ep in ("start", "position") else "conn_end", []
            ):
                continue  # already connected
            pt_layers = set(ItemList[pt_oid].get("layer", []))
            if not pad_layers & pt_layers:
                continue
            # Quick bbox check: expand pad bbox by wire radius
            pt_bbox = (pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r)
            if not _bboxes_overlap(pad_bbox, pt_bbox):
                continue
            if _point_in_pad(pos, pad_d, margin=r):
                _add_conn(pt_oid, ep, pad_oid, "position")

    # Zone connectivity: check if endpoints lie inside zone outlines
    zones = {
        oid: d
        for oid, d in ItemList.items()
        if d["type"] == "ZONE" and len(d.get("_outline", [])) >= 3
    }

    for zone_oid, zone_d in zones.items():
        outline = zone_d["_outline"]
        zone_bbox = bboxes[zone_oid]
        zone_layers = set(zone_d["layer"])
        for pos, pt_oid, ep, r in points:
            pt_layers = set(ItemList[pt_oid].get("layer", []))
            if not zone_layers & pt_layers:
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
