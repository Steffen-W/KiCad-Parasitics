import numpy as np
import os
import math
import traceback

if __name__ == "__main__":
    from Get_Distance import (
        find_shortest_path,
        get_graph_from_edges,
        find_all_reachable_nodes,
    )
    import ngspyce
else:
    from .Get_Distance import (
        find_shortest_path,
        get_graph_from_edges,
        find_all_reachable_nodes,
    )
    from . import ngspyce


def round_n(n, decimals=0):
    if math.isinf(n):
        return n
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def RunSimulation(resistors, conn1, conn2):
    # https://github.com/ignamv/ngspyce/
    filename = os.path.join(os.path.dirname(__file__), "TempNetlist.net")

    Rshunt = 0.1
    with open(filename, "w") as f:
        f.write("* gnetlist -g spice-sdb\n")

        for i, res in enumerate(resistors):
            entry = "R{} {} {} {:.10f}\n".format(i + 1, res[0], res[1], res[2])
            f.write(entry)

        f.write("v1 {} 0 1\n".format(conn1))
        f.write("R{} 0 {} {}\n".format(len(resistors) + 1, conn2, Rshunt))
        f.write(".end")

    ngspyce.source(filename)
    ngspyce.dc("v1", 1, 1, 1)  # set v1 to 1V
    os.remove(filename)
    vout = ngspyce.vector(str(conn2))[0]

    if not vout == 0:
        R = (1 - vout) / (vout / Rshunt)
    else:
        R = -1
    return R


# in https://www.youtube.com/watch?v=hNHTwpegFBw
# rho_cu = 1/47e6  # Ohm * m # 26% more than 1.68e-8

rho_cu = 1.68e-8  # Ohm * m


def calcResWIRE(Length, Width, cu_thickness=0.035, freq=0):
    # https://learnemc.com/EXT/calculators/Resistance_Calculator/rect.html

    if freq == 0:
        return Length * rho_cu / (cu_thickness * Width) * 1000.0
    else:  # TODO
        # mu = 1
        # SkinDepth = 1 / np.sqrt(freq * np.pi * mu / rho_cu) # in m
        return Length * rho_cu / (cu_thickness * Width) * 1000.0


def calcResVIA(Drill, Length, cu_thickness=0.035):
    radius = Drill / 2
    area = np.pi * ((radius + cu_thickness) ** 2 - radius**2)
    return Length * rho_cu / area * 1000


def Get_shortest_path_RES(path, resistors):
    def get_res(x1, x2):
        x = next(x for x in resistors if {x1, x2} == set(x[0:2]))
        return x[2]

    RES = 0
    for i in range(1, len(path)):
        RES += get_res(path[i - 1], path[i])

    return RES


def find_all_paths_bfs(graph, start, end, max_depth=10):
    """Find all paths from start to end using BFS, up to max_depth."""
    from collections import deque
    if start not in graph or end not in graph:
        return []
    all_paths = []
    queue = deque([(start, [start])])
    while queue:
        node, current_path = queue.popleft()
        if len(current_path) > max_depth:
            continue
        if node == end:
            all_paths.append(current_path)
            continue
        for neighbor in graph[node]:
            if neighbor not in current_path:
                queue.append((neighbor, current_path + [neighbor]))
    return all_paths


def render_network_ascii(graph, path, network_info):
    """Render network path with parallel branches as ASCII."""
    if len(path) < 2:
        return "Path too short"

    res_lookup = {}
    for info in network_info:
        n1, n2 = info["nodes"]
        res_lookup[(min(n1, n2), max(n1, n2))] = info

    def get_res(a, b):
        key = (min(a, b), max(a, b))
        return res_lookup[key]["resistance"] if key in res_lookup else 0

    def get_type(a, b):
        key = (min(a, b), max(a, b))
        return res_lookup[key]["type"][0] if key in res_lookup else "?"

    def path_res(p):
        return sum(get_res(p[i], p[i+1]) for i in range(len(p)-1))

    def fmt(p):
        return "→".join(f"n{n}" for n in p)

    # Find all paths
    all_paths = find_all_paths_bfs(graph, path[0], path[-1], max_depth=len(path) + 4)
    all_paths.sort(key=path_res)

    # Common prefix length
    plen = 0
    for i in range(min(len(p) for p in all_paths)):
        if len(set(p[i] for p in all_paths)) == 1:
            plen = i + 1
        else:
            break

    # Common suffix length
    slen = 0
    for i in range(1, min(len(p) for p in all_paths) + 1):
        if len(set(p[-i] for p in all_paths)) == 1:
            slen = i
        else:
            break

    lines = []

    # Prefix (series)
    for i in range(plen - 1):
        a, b = all_paths[0][i], all_paths[0][i + 1]
        lines.append(f"n{a}")
        lines.append(f" │ {get_type(a, b)} {get_res(a, b)*1000:.2f}mΩ")

    # Parallel part
    lines.append(f"n{all_paths[0][plen - 1]}")

    mids = [(p[plen - 1: len(p) - slen + 1], path_res(p[plen - 1: len(p) - slen + 1]))
            for p in all_paths]
    mids.sort(key=lambda x: (x[0], x[1]))  # Sort by path then resistance

    prev = None
    for mid, r in mids:
        if prev:
            # Find common prefix with previous
            common = 0
            for i in range(min(len(mid), len(prev))):
                if mid[i] == prev[i]:
                    common = i + 1
                else:
                    break
            if common > 1:
                indent = "   " * (common - 1)
                diff = mid[common:]
                lines.append(f" │{indent}└─ {fmt(diff)} ({r*1000:.2f}mΩ)")
            else:
                lines.append(f" ├─ {fmt(mid[1:])} ({r*1000:.2f}mΩ)")
        else:
            lines.append(f" ├─ {fmt(mid[1:])} ({r*1000:.2f}mΩ)")
        prev = mid

    # Parallel resistance
    resistances = [r for _, r in mids]
    if all(r > 0 for r in resistances):
        r_par = 1 / sum(1/r for r in resistances)
        lines.append(f" └─ R_parallel = {r_par*1000:.2f}mΩ")

    lines.append(f"n{all_paths[0][-slen]}")

    # Suffix (series)
    for i in range(slen - 1):
        a, b = all_paths[0][-slen + i], all_paths[0][-slen + i + 1]
        lines.append(f" │ {get_type(a, b)} {get_res(a, b)*1000:.2f}mΩ")
        lines.append(f"n{b}")

    return "\n".join(lines)


def format_network_info(network_info, graph=None, path=None):
    """Format network info as human-readable string for debugging.

    Args:
        network_info: List of dicts with resistance element info
        graph: Optional adjacency dict for path visualization
        path: Optional list of nodes for path visualization
    """
    lines = ["RESISTANCE NETWORK", "-" * 40]

    # Normalize nodes (lower first) and sort by (n1, n2)
    sorted_info = sorted(
        network_info,
        key=lambda x: (min(x["nodes"]), max(x["nodes"]))
    )

    for item in sorted_info:
        n1, n2 = sorted(item["nodes"])  # lower node first
        t = item["type"]
        r = item["resistance"]

        if t == "WIRE":
            lines.append(
                f"WIRE  n{n1}--n{n2}  R={r*1000:.3f}mΩ  "
                f"L={item['length']:.3f}mm W={item['width']:.3f}mm {item['layer_name']}"
            )
        elif t == "VIA":
            lines.append(
                f"VIA   n{n1}--n{n2}  R={r*1000:.3f}mΩ  "
                f"D={item['drill']:.2f}mm {item['layer1_name']}<->{item['layer2_name']}"
            )
        elif t == "ZONE":
            lines.append(
                f"ZONE  n{n1}--n{n2}  R={r*1000:.3f}mΩ  {item['layer_name']}"
            )

    lines.append(f"Total: {len(network_info)} elements")

    # Add ASCII tree visualization if path provided
    if graph and path and len(path) >= 2:
        lines.append("")
        lines.append("PATH VISUALIZATION (start → end):")
        lines.append("-" * 40)
        lines.append(render_network_ascii(graph, path, network_info))

    return "\n".join(lines)


def Get_Parasitic(data, CuStack, conn1, conn2, netcode, debug=0, debug_print=None):
    """
    Calculate parasitic resistance and path between two connection points.

    Args:
        data: PCB element data dictionary
        CuStack: Copper stackup information
        conn1: First connection point (network node ID)
        conn2: Second connection point (network node ID)
        netcode: KiCad net code to filter elements
        debug: Debug level (0=off, 1=on)
        debug_print: Optional debug print function
    """
    if debug_print is None:
        debug_print = print if debug else lambda _: None

    resistors = []
    network_info = []  # Detailed info for debugging/analysis
    coordinates = {}
    Area = {layer_idx: 0 for layer_idx in range(32)}

    for uuid, d in data.items():
        if not netcode == d["net_code"]:
            continue

        if len(d["layer"]) > 1:
            # Multi-layer element (VIA or through-hole PAD)
            # Collect all valid nodes for this element
            nodes = []
            for layer in d["layer"]:
                if layer not in CuStack:
                    debug_print(f"WARNING: Layer {layer} not in CuStack, skipping")
                    continue
                if layer in d.get("net_start", {}):
                    node = d["net_start"][layer]
                    if node > 0:
                        nodes.append((layer, node))
                        coordinates[node] = (
                            d["position"][0],
                            d["position"][1],
                            CuStack[layer]["abs_height"],
                        )

            # Connect all layer nodes of this via/pad
            if len(nodes) > 1:
                first_layer = nodes[0][0]
                thickness = CuStack[first_layer]["thickness"]
                for i in range(len(nodes)):
                    Layer1, node1 = nodes[i]
                    for j in range(i + 1, len(nodes)):
                        Layer2, node2 = nodes[j]
                        if Layer2 not in CuStack or Layer1 not in CuStack:
                            debug_print(
                                f"ERROR: CuStack incomplete for layers {Layer1}, {Layer2}"
                            )
                            continue
                        if "drill" not in d:
                            continue
                        distance = abs(
                            CuStack[Layer2]["abs_height"]
                            - CuStack[Layer1]["abs_height"]
                        )
                        resistor = calcResVIA(
                            d["drill"], distance, cu_thickness=thickness
                        )
                        if resistor < 0:
                            raise ValueError("Error in resistance calculation!")
                        resistors.append([node1, node2, resistor, distance])
                        network_info.append({
                            "type": "VIA",
                            "nodes": (node1, node2),
                            "resistance": resistor,
                            "length": distance,
                            "drill": d["drill"],
                            "layer1": Layer1,
                            "layer2": Layer2,
                            "layer1_name": CuStack[Layer1]["name"],
                            "layer2_name": CuStack[Layer2]["name"],
                            "position": d["position"],
                            "cu_thickness": thickness,
                        })

        else:
            Layer = d["layer"][0]
            Area[Layer] += d["area"]

            if Layer not in CuStack:
                debug_print(f"WARNING: Layer {Layer} not in CuStack, skipping element")
                continue

            if d["type"] == "WIRE":
                netStart = d["net_start"].get(Layer, 0)
                netEnd = d["net_end"].get(Layer, 0)

                if netStart == 0 or netEnd == 0:
                    debug_print(
                        f"[DEBUG] Skipping WIRE: netStart={netStart}, netEnd={netEnd}, "
                        f"connStart={d.get('connStart', [])}, connEnd={d.get('connEnd', [])}"
                    )
                    continue

                thickness = CuStack[Layer]["thickness"]
                resistor = calcResWIRE(d["length"], d["width"], cu_thickness=thickness)
                if resistor < 0:
                    raise ValueError("Error in resistance calculation!")
                resistors.append([netStart, netEnd, resistor, d["length"]])
                network_info.append({
                    "type": "WIRE",
                    "nodes": (netStart, netEnd),
                    "resistance": resistor,
                    "length": d["length"],
                    "width": d["width"],
                    "layer": Layer,
                    "layer_name": CuStack[Layer]["name"],
                    "start": d["start"],
                    "end": d["end"],
                    "cu_thickness": thickness,
                })

                coordinates[netStart] = (
                    d["start"][0],
                    d["start"][1],
                    CuStack[Layer]["abs_height"],
                )
                coordinates[netEnd] = (
                    d["end"][0],
                    d["end"][1],
                    CuStack[Layer]["abs_height"],
                )

            elif d["type"] == "PAD":
                # Single-layer pad - just record its position
                if Layer in d.get("net_start", {}):
                    node = d["net_start"][Layer]
                    if node > 0:
                        coordinates[node] = (
                            d["position"][0],
                            d["position"][1],
                            CuStack[Layer]["abs_height"],
                        )

    Area_reduc = {
        layer_idx: Area[layer_idx] for layer_idx in Area if Area[layer_idx] > 0
    }

    # Handle ZONES: connect all elements touching the same zone
    zone_connections = {}
    for uuid, d in data.items():
        if netcode != d["net_code"] or d["type"] != "ZONE":
            continue

        zone_conns = d.get("conn_start", [])
        for layer in d.get("layer", []):
            if layer not in zone_connections:
                zone_connections[layer] = {}
            if uuid not in zone_connections[layer]:
                zone_connections[layer][uuid] = []

            for conn_uuid in zone_conns:
                if conn_uuid not in data:
                    continue
                conn_item = data[conn_uuid]
                if layer not in conn_item.get("layer", []):
                    continue

                if conn_item.get("type") == "WIRE":
                    # Only add the wire end that touches the zone
                    for conn_type in ["conn_start", "conn_end"]:
                        net_type = "net_start" if conn_type == "conn_start" else "net_end"
                        if uuid in conn_item.get(conn_type, []):
                            if layer in conn_item.get(net_type, {}):
                                node = conn_item[net_type][layer]
                                if node > 0 and node not in zone_connections[layer][uuid]:
                                    zone_connections[layer][uuid].append(node)
                else:
                    if "net_start" in conn_item and layer in conn_item["net_start"]:
                        node = conn_item["net_start"][layer]
                        if node > 0 and node not in zone_connections[layer][uuid]:
                            zone_connections[layer][uuid].append(node)

    # Connect zone nodes with low resistance
    zone_resistance = 0.001  # 1 mOhm
    for layer, zones in zone_connections.items():
        layer_name = CuStack[layer]["name"] if layer in CuStack else f"Layer_{layer}"
        for _, nodes in zones.items():
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1 :]:
                    zone_distance = 0.0
                    if node1 in coordinates and node2 in coordinates:
                        c1 = coordinates[node1]
                        c2 = coordinates[node2]
                        zone_distance = np.sqrt(
                            (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
                        )
                    resistors.append([node1, node2, zone_resistance, zone_distance])
                    network_info.append({
                        "type": "ZONE",
                        "nodes": (node1, node2),
                        "resistance": zone_resistance,
                        "length": zone_distance,
                        "layer": layer,
                        "layer_name": layer_name,
                    })

    for res in resistors:
        if res[2] <= 0:
            raise ValueError("Error in resistance calculation!")

    # Build graph and find path
    edges = [(i[0], i[1], i[3]) for i in resistors]
    graph = get_graph_from_edges(edges)

    path = []
    try:
        Distance, path = find_shortest_path(graph, conn1, conn2)
        short_path_RES = Get_shortest_path_RES(path, resistors)
    except Exception as e:
        short_path_RES = -1
        Distance = float("inf")
        debug_print(f"[DEBUG] Path finding failed: {e}")
        debug_print(f"[DEBUG] Graph: {len(graph)} nodes, {len(edges)} edges")
        if conn1 not in graph:
            debug_print(f"[DEBUG] conn1={conn1} NOT in graph")
        if conn2 not in graph:
            debug_print(f"[DEBUG] conn2={conn2} NOT in graph")
        if conn1 in graph and conn2 in graph:
            reachable = find_all_reachable_nodes(graph, conn1)
            debug_print(f"[DEBUG] conn2 reachable from conn1: {conn2 in reachable}")

    try:
        Resistance = RunSimulation(resistors, conn1, conn2)
    except Exception:
        Resistance = -1
        print(traceback.format_exc())
        print("ERROR in RunSimulation")

    return Resistance, Distance, short_path_RES, Area_reduc, network_info, graph, path
