import numpy as np
import os
import math
import traceback


try:
    if __name__ == "__main__":
        from Get_Self_Inductance import calculate_self_inductance, interpolate_vertices
        from Get_Distance import find_shortest_path, get_graph_from_edges, find_all_reachable_nodes
        import ngspyce
    else:
        from .Get_Self_Inductance import calculate_self_inductance, interpolate_vertices
        from .Get_Distance import find_shortest_path, get_graph_from_edges, find_all_reachable_nodes
        from . import ngspyce
except Exception as e:
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
    if debug_print is None:
        debug_print = print  # Fallback to print if not provided
    
    resistors = []
    coordinates = {}

    Area = {l: 0 for l in range(32)}  # for all layer

    debug_print(f"[DEBUG Get_Parasitic] Starting analysis for NetCode={netcode}, conn1={conn1}, conn2={conn2}")
    debug_print(f"[DEBUG Get_Parasitic] Total elements in data: {len(data)}")
    
    # Debug: Count elements by type and check for invalid wires
    element_counts = {"VIA": 0, "WIRE": 0, "PAD": 0, "ZONE": 0}
    wires_with_invalid_net = []
    wires_added_count = 0
    for uuid, d in data.items():
        if netcode == d["NetCode"]:
            element_counts[d.get("type", "UNKNOWN")] = element_counts.get(d.get("type", "UNKNOWN"), 0) + 1
            # Check for wires with invalid netStart/netEnd
            if d.get("type") == "WIRE":
                for layer in d.get("Layer", []):
                    netStart = d.get("netStart", {}).get(layer, 0)
                    netEnd = d.get("netEnd", {}).get(layer, 0)
                    if netStart == 0 or netEnd == 0:
                        wires_with_invalid_net.append((uuid, layer, netStart, netEnd, d.get("Start"), d.get("End")))
                    else:
                        wires_added_count += 1
    debug_print(f"[DEBUG Get_Parasitic] Elements for NetCode={netcode}: {element_counts}")
    debug_print(f"[DEBUG Get_Parasitic] Wires that will be added to graph: {wires_added_count}")
    if wires_with_invalid_net:
        debug_print(f"[DEBUG Get_Parasitic] WARNING: {len(wires_with_invalid_net)} wires have invalid netStart/netEnd values and will NOT be added!")
        for uuid, layer, ns, ne, start, end in wires_with_invalid_net[:10]:  # Show first 10
            debug_print(f"[DEBUG Get_Parasitic]   WIRE {uuid} on layer {layer}: netStart={ns}, netEnd={ne}, Start={start}, End={end}")
            # Check what this wire is connected to
            if uuid in data:
                wire_data = data[uuid]
                debug_print(f"[DEBUG Get_Parasitic]     connStart={wire_data.get('connStart', [])}, connEnd={wire_data.get('connEnd', [])}")

    for uuid, d in data.items():
        if not netcode == d["NetCode"]:
            continue

        if len(d["Layer"]) > 1:
            # VIA connects multiple layers - all layers should be connected to each other
            # through the via (which is a single physical point)
            via_nodes = []  # Collect all netStart nodes for this via
            for layer in d["Layer"]:
                if layer in d.get("netStart", {}):
                    node = d["netStart"][layer]
                    if node > 0:  # Valid node (not NotYetConnected)
                        via_nodes.append((layer, node))
                        coordinates[node] = (
                            d["Position"][0],
                            d["Position"][1],
                            CuStack[layer]["abs_height"],
                        )
            
            # Connect all layers of the via to each other
            # A via is a single physical connection, so all its layer nodes should be shorted
            if len(via_nodes) > 1:
                thickness = CuStack[0]["thickness"]  # from Layer Top
                for i in range(len(via_nodes)):
                    Layer1, node1 = via_nodes[i]
                    for j in range(i + 1, len(via_nodes)):
                        Layer2, node2 = via_nodes[j]
                        if Layer2 in CuStack and Layer1 in CuStack:
                            distance = abs(
                                CuStack[Layer2]["abs_height"] - CuStack[Layer1]["abs_height"]
                            )
                        else:
                            debug_print(f"ERROR: CuStack is incomplete for layers {Layer1}, {Layer2}!")
                            continue
                        if "Drill" not in d:
                            continue
                        resistor = calcResVIA(d["Drill"], distance, cu_thickness=thickness)
                        if resistor < 0:
                            raise ValueError("Error in resistance calculation!")
                        resistors.append([node1, node2, resistor, distance])
                        # Log if this involves our connection points
                        if node1 == conn1 or node2 == conn1 or node1 == conn2 or node2 == conn2:
                            debug_print(f"[DEBUG Get_Parasitic] Added VIA edge: {node1} (L{Layer1}) <-> {node2} (L{Layer2}) (resistance={resistor:.6f}, distance={distance:.3f})")
            elif len(via_nodes) == 1:
                # Via has only one valid layer node - this shouldn't happen but handle it
                debug_print(f"[DEBUG Get_Parasitic] WARNING: VIA {uuid} has only 1 valid layer node: {via_nodes}")

        else:
            Layer = d["Layer"][0]
            Area[Layer] += d["Area"]
            if d["type"] == "WIRE":
                netStart = d["netStart"][Layer]
                netEnd = d["netEnd"][Layer]
                # Debug: Check if netStart/netEnd are valid
                if netStart == 0 or netEnd == 0:
                    debug_print(f"[DEBUG Get_Parasitic] WARNING: WIRE {uuid} has invalid netStart={netStart} or netEnd={netEnd} on layer {Layer}")
                thickness = CuStack[Layer]["thickness"]
                resistor = calcResWIRE(d["Length"], d["Width"], cu_thickness=thickness)
                if resistor < 0:
                    raise ValueError("Error in resistance calculation!")
                resistors.append([netStart, netEnd, resistor, d["Length"]])
                # Only log wires that involve our connection points
                if netStart == conn1 or netEnd == conn1 or netStart == conn2 or netEnd == conn2:
                    debug_print(f"[DEBUG Get_Parasitic] Added WIRE edge: {netStart} -> {netEnd} (resistance={resistor:.6f}, length={d['Length']:.3f}, uuid={uuid})")

                coordinates[d["netStart"][Layer]] = (
                    d["Start"][0],
                    d["Start"][1],
                    CuStack[Layer]["abs_height"],
                )
                coordinates[d["netEnd"][Layer]] = (
                    d["End"][0],
                    d["End"][1],
                    CuStack[Layer]["abs_height"],
                )
            elif d["type"] == "PAD":
                # PADs are connection points - they connect all their netStart nodes on all layers
                # Similar to vias, but pads can be on multiple layers too
                pad_nodes = []  # Collect all netStart nodes for this pad
                for layer in d.get("Layer", []):
                    if layer in d.get("netStart", {}):
                        node = d["netStart"][layer]
                        if node > 0:  # Valid node
                            pad_nodes.append((layer, node))
                            coordinates[node] = (
                                d["Position"][0],
                                d["Position"][1],
                                CuStack[layer]["abs_height"],
                            )
                
                # Connect all layers of the pad to each other (pad is a single physical point)
                if len(pad_nodes) > 1:
                    # Pad connects multiple layers - connect them with very low resistance
                    pad_resistance = 0.0001  # Very low resistance for pad layer connections
                    for i in range(len(pad_nodes)):
                        Layer1, node1 = pad_nodes[i]
                        for j in range(i + 1, len(pad_nodes)):
                            Layer2, node2 = pad_nodes[j]
                            distance = abs(CuStack[Layer2]["abs_height"] - CuStack[Layer1]["abs_height"])
                            resistors.append([node1, node2, pad_resistance, distance])
                            if node1 == conn1 or node2 == conn1 or node1 == conn2 or node2 == conn2:
                                debug_print(f"[DEBUG Get_Parasitic] Added PAD layer edge: {node1} (L{Layer1}) <-> {node2} (L{Layer2})")
                
                # PADs also connect to all wires/vias that connect to them
                # This is already handled by Connect_Nets giving them the same netStart values
                # But we need to ensure pads with the same netStart on the same layer are connected
                # Actually, pads with the same netStart are already in the same node, so they're implicitly connected
                # The issue might be that pads need to connect their netStart to connected wires/vias
                # But Connect_Nets should have already done this...

    Area_reduc = {l: Area[l] for l in Area.keys() if Area[l] > 0}

    # Handle ZONES: Zones connect all pads/vias that touch them
    # Since zones are large copper areas, they effectively short all connected pads/vias
    zone_connections = {}  # zone_uuid -> list of netStart nodes per layer
    for uuid, d in data.items():
        if not netcode == d["NetCode"]:
            continue
        if d["type"] == "ZONE":
            # Get all connections to this zone
            zone_conns = d.get("connStart", [])
            for layer in d.get("Layer", []):
                if layer not in zone_connections:
                    zone_connections[layer] = {}
                if uuid not in zone_connections[layer]:
                    zone_connections[layer][uuid] = []
                # Find all pads/vias connected to this zone on this layer
                for conn_uuid in zone_conns:
                    if conn_uuid in data:
                        conn_item = data[conn_uuid]
                        if layer in conn_item.get("Layer", []):
                            # Get the netStart node for this connection on this layer
                            if "netStart" in conn_item and layer in conn_item["netStart"]:
                                node = conn_item["netStart"][layer]
                                if node not in zone_connections[layer][uuid]:
                                    zone_connections[layer][uuid].append(node)
    
    # Add zone connections: all nodes connected to the same zone on the same layer
    # are effectively shorted (very low resistance)
    zone_resistance = 0.001  # Very low resistance for zone connections (effectively short)
    for layer, zones in zone_connections.items():
        for zone_uuid, nodes in zones.items():
            # Connect all nodes in this zone to each other (fully connected graph)
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    # Add edge with very low resistance (zone connection)
                    resistors.append([node1, node2, zone_resistance, 0.0])
                    debug_print(f"[DEBUG Get_Parasitic] Zone connection: {node1} <-> {node2} (zone {zone_uuid}, layer {layer})")

    for res in resistors:
        if res[2] <= 0:
            raise ValueError("Error in resistance calculation!")

    # edges = list( (node1, node2, distance) )
    edges = [(i[0], i[1], i[3]) for i in resistors]
    graph = get_graph_from_edges(edges)
    
    debug_print(f"[DEBUG Get_Parasitic] Found {len(resistors)} resistors/connections (before zones)")
    debug_print(f"[DEBUG Get_Parasitic] Graph nodes: {sorted(graph.keys())}")
    debug_print(f"[DEBUG Get_Parasitic] Looking for path from conn1={conn1} to conn2={conn2}")
    debug_print(f"[DEBUG Get_Parasitic] conn1 in graph: {conn1 in graph}")
    debug_print(f"[DEBUG Get_Parasitic] conn2 in graph: {conn2 in graph}")
    if conn1 in graph:
        debug_print(f"[DEBUG Get_Parasitic] conn1 neighbors: {list(graph[conn1].keys())}")
    if conn2 in graph:
        debug_print(f"[DEBUG Get_Parasitic] conn2 neighbors: {list(graph[conn2].keys())}")
    debug_print(f"[DEBUG Get_Parasitic] Total edges in graph: {len(edges)}")
    debug_print(f"[DEBUG Get_Parasitic] All edges (first 20): {edges[:20]}")
    
    # Check for edges involving conn1 or conn2
    conn1_edges = [e for e in edges if conn1 in (e[0], e[1])]
    conn2_edges = [e for e in edges if conn2 in (e[0], e[1])]
    debug_print(f"[DEBUG Get_Parasitic] Edges involving conn1={conn1}: {len(conn1_edges)} edges")
    if conn1_edges:
        debug_print(f"[DEBUG Get_Parasitic] conn1 edges: {conn1_edges[:10]}")
    debug_print(f"[DEBUG Get_Parasitic] Edges involving conn2={conn2}: {len(conn2_edges)} edges")
    if conn2_edges:
        debug_print(f"[DEBUG Get_Parasitic] conn2 edges: {conn2_edges[:10]}")
    
            
    
    try:
        Distance, path = find_shortest_path(graph, conn1, conn2)
        debug_print(f"[DEBUG Get_Parasitic] Path found! Distance={Distance}, Path={path}")
        path3d = [coordinates[p] for p in path]
        short_path_RES = Get_shortest_path_RES(path, resistors)
    except Exception as e:
        short_path_RES = -1
        Distance, path3d = float("inf"), []
        error_msg = traceback.format_exc()
        debug_print(error_msg)
        debug_print("ERROR in find_shortest_path")
        debug_print(f"[DEBUG Get_Parasitic] Path finding failed!")
        debug_print(f"[DEBUG Get_Parasitic] Exception: {e}")
        if conn1 not in graph:
            debug_print(f"[DEBUG Get_Parasitic] conn1={conn1} is NOT in the graph!")
            debug_print(f"[DEBUG Get_Parasitic] Available nodes: {sorted(graph.keys())}")
        if conn2 not in graph:
            debug_print(f"[DEBUG Get_Parasitic] conn2={conn2} is NOT in the graph!")
            debug_print(f"[DEBUG Get_Parasitic] Available nodes: {sorted(graph.keys())}")
        
        # Additional debugging: find reachable nodes
        reachable_from_conn1 = find_all_reachable_nodes(graph, conn1)
        debug_print(f"[DEBUG Get_Parasitic] Nodes reachable from conn1={conn1}: {len(reachable_from_conn1)} nodes")
        debug_print(f"[DEBUG Get_Parasitic] conn2={conn2} is {'reachable' if conn2 in reachable_from_conn1 else 'NOT reachable'} from conn1")
        if conn2 not in reachable_from_conn1:
            # Show some example reachable nodes
            sample_reachable = sorted(list(reachable_from_conn1))[:20]
            debug_print(f"[DEBUG Get_Parasitic] Sample reachable nodes from conn1: {sample_reachable}")

    inductance_nH = 0
    try:
        if len(path3d) > 2:
            vertices = interpolate_vertices(path3d, num_points=1000)
            inductance_nH = 0  # calculate_self_inductance(vertices, current=1) * 1e9
    except Exception as e:
        inductance_nH = 0
        print(traceback.format_exc())
        print("ERROR in calculate_self_inductance")

    try:
        Resistance = RunSimulation(resistors, conn1, conn2)
    except Exception as e:
        Resistance = -1
        print(traceback.format_exc())
        print("ERROR in RunSimulation")
    return Resistance, Distance, inductance_nH, short_path_RES, Area_reduc
