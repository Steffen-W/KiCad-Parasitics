import numpy as np
import os
import math
import traceback


try:
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
except Exception:
    print(traceback.format_exc())


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
        f.write("R{} 0 {} {}\n".format(i + 2, conn2, Rshunt))
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
    # No-op debug print when debug is disabled
    if debug and debug_print is None:
        debug_print = print
    elif not debug:

        def debug_print(msg):
            return None

    resistors = []
    coordinates = {}
    Area = {layer_idx: 0 for layer_idx in range(32)}

    for uuid, d in data.items():
        if not netcode == d["NetCode"]:
            continue

        if len(d["Layer"]) > 1:
            # Multi-layer element (VIA or through-hole PAD)
            # Collect all valid nodes for this element
            nodes = []
            for layer in d["Layer"]:
                if layer not in CuStack:
                    debug_print(f"WARNING: Layer {layer} not in CuStack, skipping")
                    continue
                if layer in d.get("netStart", {}):
                    node = d["netStart"][layer]
                    if node > 0:
                        nodes.append((layer, node))
                        coordinates[node] = (
                            d["Position"][0],
                            d["Position"][1],
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
                        if "Drill" not in d:
                            continue
                        distance = abs(
                            CuStack[Layer2]["abs_height"]
                            - CuStack[Layer1]["abs_height"]
                        )
                        resistor = calcResVIA(
                            d["Drill"], distance, cu_thickness=thickness
                        )
                        if resistor < 0:
                            raise ValueError("Error in resistance calculation!")
                        resistors.append([node1, node2, resistor, distance])

        else:
            Layer = d["Layer"][0]
            Area[Layer] += d["Area"]

            if Layer not in CuStack:
                debug_print(f"WARNING: Layer {Layer} not in CuStack, skipping element")
                continue

            if d["type"] == "WIRE":
                netStart = d["netStart"].get(Layer, 0)
                netEnd = d["netEnd"].get(Layer, 0)

                if netStart == 0 or netEnd == 0:
                    debug_print(
                        f"[DEBUG] Skipping WIRE: netStart={netStart}, netEnd={netEnd}, "
                        f"connStart={d.get('connStart', [])}, connEnd={d.get('connEnd', [])}"
                    )
                    continue

                thickness = CuStack[Layer]["thickness"]
                resistor = calcResWIRE(d["Length"], d["Width"], cu_thickness=thickness)
                if resistor < 0:
                    raise ValueError("Error in resistance calculation!")
                resistors.append([netStart, netEnd, resistor, d["Length"]])

                coordinates[netStart] = (
                    d["Start"][0],
                    d["Start"][1],
                    CuStack[Layer]["abs_height"],
                )
                coordinates[netEnd] = (
                    d["End"][0],
                    d["End"][1],
                    CuStack[Layer]["abs_height"],
                )

            elif d["type"] == "PAD":
                # Single-layer pad - just record its position
                if Layer in d.get("netStart", {}):
                    node = d["netStart"][Layer]
                    if node > 0:
                        coordinates[node] = (
                            d["Position"][0],
                            d["Position"][1],
                            CuStack[Layer]["abs_height"],
                        )

    Area_reduc = {
        layer_idx: Area[layer_idx] for layer_idx in Area if Area[layer_idx] > 0
    }

    # Handle ZONES: connect all elements touching the same zone
    zone_connections = {}
    for uuid, d in data.items():
        if netcode != d["NetCode"] or d["type"] != "ZONE":
            continue

        zone_conns = d.get("connStart", [])
        for layer in d.get("Layer", []):
            if layer not in zone_connections:
                zone_connections[layer] = {}
            if uuid not in zone_connections[layer]:
                zone_connections[layer][uuid] = []

            for conn_uuid in zone_conns:
                if conn_uuid not in data:
                    continue
                conn_item = data[conn_uuid]
                if layer not in conn_item.get("Layer", []):
                    continue

                if conn_item.get("type") == "WIRE":
                    # Only add the wire end that touches the zone
                    for conn_type in ["connStart", "connEnd"]:
                        net_type = "netStart" if conn_type == "connStart" else "netEnd"
                        if uuid in conn_item.get(conn_type, []):
                            if layer in conn_item.get(net_type, {}):
                                node = conn_item[net_type][layer]
                                if node > 0 and node not in zone_connections[layer][uuid]:
                                    zone_connections[layer][uuid].append(node)
                else:
                    if "netStart" in conn_item and layer in conn_item["netStart"]:
                        node = conn_item["netStart"][layer]
                        if node > 0 and node not in zone_connections[layer][uuid]:
                            zone_connections[layer][uuid].append(node)

    # Connect zone nodes with low resistance
    zone_resistance = 1  # mOhm
    for layer, zones in zone_connections.items():
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

    for res in resistors:
        if res[2] <= 0:
            raise ValueError("Error in resistance calculation!")

    # Build graph and find path
    edges = [(i[0], i[1], i[3]) for i in resistors]
    graph = get_graph_from_edges(edges)

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

    return Resistance, Distance, short_path_RES, Area_reduc
