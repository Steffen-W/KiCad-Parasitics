import numpy as np
import os
import math
import time
import traceback

try:
    from .Get_Distance import (
        find_shortest_path,
        get_graph_from_edges,
        find_all_reachable_nodes,
    )
    from .impedance import (
        get_Via_Parasitics,
        RHO_CU,
        analyze_microstrip,
        analyze_stripline,
        analyze_coplanar,
    )
    from . import ngspyce
except ImportError:
    from Get_Distance import (
        find_shortest_path,
        get_graph_from_edges,
        find_all_reachable_nodes,
    )
    from impedance import (
        get_Via_Parasitics,
        RHO_CU,
        analyze_microstrip,
        analyze_stripline,
        analyze_coplanar,
    )
    import ngspyce


def analyze_trace(length, width, cu_layer_id, CuStack, frequency=100e6):
    """Analyze a PCB trace with frequency-dependent parameters.

    Args:
        length: trace length in m
        width: trace width in m
        cu_layer_id: copper layer ID (key into CuStack)
        CuStack: dict per layer with thickness, name, abs_height,
            die_above, die_below, model (Microstrip/Stripline/Coplanar/Coplanar_Grounded/R only)
        frequency: operating frequency in Hz (default 100 MHz)
    Returns:
        dict with r_dc, r_ac, inductance, capacitance, z0, delay,
        skin_depth, loss_conductor, loss_dielectric, epsilon_eff,
        angle_electrical
    """
    layer = CuStack[cu_layer_id]
    t = layer["thickness"]
    r_dc = RHO_CU * length / (t * width)

    dc_only = {
        "r_dc": r_dc,
        "r_ac": r_dc,
        "inductance": None,
        "capacitance": None,
        "z0": None,
        "delay": None,
        "skin_depth": None,
        "loss_conductor": None,
        "loss_dielectric": None,
        "epsilon_eff": None,
        "angle_electrical": None,
        "wavelength_ratio": None,
    }

    if frequency <= 0:
        return dc_only

    model = layer.get("model", "R only")
    die_above = layer.get("die_above")
    die_below = layer.get("die_below")

    if model == "Stripline" and die_above and die_below:
        ha, hb = die_above["h"], die_below["h"]
        h_total = ha + t + hb
        epsilon_r = (die_above["epsilon_r"] * ha + die_below["epsilon_r"] * hb) / (
            ha + hb
        )
        loss_tangent = (
            die_above["loss_tangent"] * ha + die_below["loss_tangent"] * hb
        ) / (ha + hb)

        result = analyze_stripline(
            w=width,
            h=h_total,
            t=t,
            a=hb,
            epsilon_r=epsilon_r,
            frequency=frequency,
            length=length,
            tan_d=loss_tangent,
        )

    elif model == "Microstrip" and (die_above or die_below):
        die = die_above if die_above else die_below
        result = analyze_microstrip(
            w=width,
            h=die["h"],
            t=t,
            epsilon_r=die["epsilon_r"],
            frequency=frequency,
            length=length,
            tan_d=die["loss_tangent"],
        )
    elif model in ("Coplanar", "Coplanar_Grounded") and (die_above or die_below):
        die = die_above if die_above else die_below
        gap = layer.get("gap")
        if gap is None or gap <= 0:
            return dc_only
        result = analyze_coplanar(
            w=width,
            s=gap,
            h=die["h"],
            t=t,
            epsilon_r=die["epsilon_r"],
            frequency=frequency,
            length=length,
            tan_d=die["loss_tangent"],
            with_ground=(model == "Coplanar_Grounded"),
        )
    else:
        return dc_only

    # Derive R_ac from conductor losses
    # alpha_c [dB] -> [Np] = dB * ln(10)/20;  R_ac = 2 * Z0 * alpha_c_Np
    loss_c_dB = result["loss_conductor"]
    z0 = result["z0"]
    if loss_c_dB > 0 and z0 > 0:
        alpha_c_Np = loss_c_dB * math.log(10) / 20
        r_ac = max(2 * z0 * alpha_c_Np, r_dc)
    else:
        r_ac = r_dc

    return {
        "r_dc": r_dc,
        "r_ac": r_ac,
        "inductance": result["inductance"],
        "capacitance": result["capacitance"],
        "z0": z0,
        "delay": result["delay"],
        "skin_depth": result["skin_depth"],
        "loss_conductor": result["loss_conductor"],
        "loss_dielectric": result["loss_dielectric"],
        "epsilon_eff": result["epsilon_eff"],
        "angle_electrical": result["angle_electrical"],
        "wavelength_ratio": result["angle_electrical"] / (2 * math.pi),
    }


def _run_spice(
    filename, elements, conn1, conn2, ac_freq=None, debug=0, debug_print=None
):
    """Write netlist, run simulation, return impedance.

    Args:
        elements: list of ("R"/"L"/"C", node1, node2, value)
        ac_freq: None for DC, frequency in Hz for AC
    Returns:
        For DC: real resistance in Ohm (float)
        For AC: complex impedance in Ohm (complex)
        On error: -1
    """
    debug_print = debug_print or (lambda _: None)

    Rshunt = 0.1
    with open(filename, "w") as f:
        f.write("* Parasitic Network\n")
        for idx, (kind, na, nb, val) in enumerate(elements, 1):
            f.write(f"{kind}{idx} {na} {nb} {val:.15f}\n")
        f.write(f"v1 {conn1} 0 dc 0 ac 1\n")
        f.write(f"Rshunt 0 {conn2} {Rshunt}\n")
        if ac_freq:
            f.write(f".ac lin 1 {ac_freq} {ac_freq}\n")
        f.write(".end\n")

    if debug:
        mode = f"AC @ {ac_freq:.0f} Hz" if ac_freq else "DC"
        debug_print(
            f"[{time.strftime('%H:%M:%S')}] [SPICE] {mode}: {len(elements)} elements, "
            f"conn {conn1} <-> {conn2}"
        )

    ngspyce.source(filename)
    if ac_freq:
        ngspyce.cmd("run")
    else:
        ngspyce.dc("v1", 1, 1, 1)
    os.remove(filename)

    if debug:
        debug_print(f"[{time.strftime('%H:%M:%S')}] [SPICE] done")

    vout = ngspyce.vector(str(conn2))
    if len(vout) > 0 and vout[0] != 0:
        z = (1 - vout[0]) / (vout[0] / Rshunt)
        if debug:
            if ac_freq and isinstance(z, complex):
                debug_print(
                    f"[SPICE]   vout={vout[0]:.6e}, Z={abs(z) * 1000:.3f} mOhm "
                    f"({z.real * 1000:.3f} + j{z.imag * 1000:.3f} mΩ)"
                )
            else:
                debug_print(f"[SPICE]   vout={vout[0]:.6e}, Z={abs(z) * 1000:.3f} mOhm")
        return z  # Return complex for AC, real for DC
    if debug:
        debug_print(f"[SPICE]   vout={vout}, simulation failed")
    return -1


def RunSimulation(
    resistors,
    conn1,
    conn2,
    network_info=None,
    frequencies=None,
    debug=0,
    debug_print=None,
):
    """Run DC simulation and optionally AC simulations at given frequencies.

    Returns:
        r_dc: DC resistance in Ohm
        z_ac: dict {freq: impedance} for each frequency (empty if no frequencies)
    """
    debug_print = debug_print or (lambda _: None)

    # https://github.com/ignamv/ngspyce/
    filename = os.path.join(os.path.dirname(__file__), "TempNetlist.net")
    gnd = 0

    # --- DC: R-only network ---
    dc_elements = [("R", r[0], r[1], r[2]) for r in resistors]
    r_dc = _run_spice(
        filename, dc_elements, conn1, conn2, debug=debug, debug_print=debug_print
    )

    # --- AC: RLC network per frequency ---
    z_ac = {}
    for freq in frequencies or []:
        ac = []
        idx = 0
        for elem in network_info or []:
            n1, n2 = elem["nodes"]
            idx += 1

            if elem["type"] == "WIRE":
                hf = elem.get("hf", {}).get(freq)
                R = hf["r_ac"] if hf else elem["resistance"]
                L = hf.get("inductance") if hf else None
                C = hf.get("capacitance") if hf else None

                # Segment if > lambda/20
                wr = hf.get("wavelength_ratio") if hf else None
                n_seg = max(1, int(wr / 0.05) + 1) if wr and wr > 0.05 else 1

                if L:
                    # Distributed RLGC model (n_seg=1 for lumped)
                    R_seg, L_seg = R / n_seg, L / n_seg
                    C_seg = C / n_seg if C else None
                    prev = n1
                    for s in range(n_seg):
                        nxt = n2 if s == n_seg - 1 else f"w{idx}s{s}"
                        mid = f"w{idx}s{s}m"
                        ac.append(("R", prev, mid, R_seg))
                        ac.append(("L", mid, nxt, L_seg))
                        if C_seg:
                            ac.append(("C", prev, gnd, C_seg))
                        prev = nxt
                else:
                    # R-only (no HF data)
                    ac.append(("R", n1, n2, R))

            elif elem["type"] == "VIA":
                R = elem["resistance"]
                L = elem.get("inductance")
                C = elem.get("capacitance")
                # R-L in series, C parallel to ground planes
                if L:
                    mid = f"mid{idx}"
                    ac.append(("R", n1, mid, R))
                    ac.append(("L", mid, n2, L))
                else:
                    ac.append(("R", n1, n2, R))
                if C:
                    ac.append(("C", n1, n2, C))

            else:  # ZONE
                R = elem["resistance"]
                ac.append(("R", n1, n2, R))

        z_ac[freq] = _run_spice(
            filename,
            ac,
            conn1,
            conn2,
            ac_freq=freq,
            debug=debug,
            debug_print=debug_print,
        )

    return r_dc, z_ac


def Get_shortest_path_RES(path, resistors):
    def get_res(x1, x2):
        x = next(x for x in resistors if {x1, x2} == set(x[0:2]))
        return x[2]

    RES = 0
    for i in range(1, len(path)):
        RES += get_res(path[i - 1], path[i])

    return RES


def extract_network(data, CuStack, netcode, debug=0, debug_print=None):
    """Extract electrical network from PCB data (DC resistances only).

    Args:
        data: PCB element data dictionary
        CuStack: Copper stackup information
        netcode: KiCad net code to filter elements
        debug: Debug level (0=off, 1=on)
        debug_print: Optional debug print function

    Returns:
        dict with:
            resistors: list of [node1, node2, r_dc, length]
            network_info: list of element dicts (WIRE/VIA/ZONE) with geometric data
            coordinates: dict {node: (x, y, z)}
            area: dict {layer: area}
            graph: adjacency dict from get_graph_from_edges
    """
    if debug_print is None:
        debug_print = print if debug else lambda _: None

    resistors = []
    network_info = []
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
                        via = get_Via_Parasitics(d["drill"], thickness, distance)
                        if via["resistance"] < 0:
                            raise ValueError("Error in resistance calculation!")
                        resistors.append([node1, node2, via["resistance"], distance])
                        network_info.append(
                            {
                                "type": "VIA",
                                "nodes": (node1, node2),
                                "resistance": via["resistance"],
                                "inductance": via["inductance"],
                                "capacitance": via["capacitance"],
                                "length": distance,
                                "drill": d["drill"],
                                "layer1": Layer1,
                                "layer2": Layer2,
                                "layer1_name": CuStack[Layer1]["name"],
                                "layer2_name": CuStack[Layer2]["name"],
                                "position": d["position"],
                            }
                        )

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

                trace = analyze_trace(
                    d["length"], d["width"], Layer, CuStack, frequency=0
                )
                resistors.append([netStart, netEnd, trace["r_dc"], d["length"]])

                network_info.append(
                    {
                        "type": "WIRE",
                        "nodes": (netStart, netEnd),
                        "resistance": trace["r_dc"],
                        "length": d["length"],
                        "width": d["width"],
                        "layer": Layer,
                        "layer_name": CuStack[Layer]["name"],
                        "start": d["start"],
                        "end": d["end"],
                    }
                )

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

    area = {layer_idx: Area[layer_idx] for layer_idx in Area if Area[layer_idx] > 0}

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
                        net_type = (
                            "net_start" if conn_type == "conn_start" else "net_end"
                        )
                        if uuid in conn_item.get(conn_type, []):
                            if layer in conn_item.get(net_type, {}):
                                node = conn_item[net_type][layer]
                                if (
                                    node > 0
                                    and node not in zone_connections[layer][uuid]
                                ):
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
                    network_info.append(
                        {
                            "type": "ZONE",
                            "nodes": (node1, node2),
                            "resistance": zone_resistance,
                            "length": zone_distance,
                            "layer": layer,
                            "layer_name": layer_name,
                        }
                    )

    for res in resistors:
        if res[2] <= 0:
            raise ValueError("Error in resistance calculation!")

    # Build graph
    edges = [(i[0], i[1], i[3]) for i in resistors]
    graph = get_graph_from_edges(edges)

    return {
        "resistors": resistors,
        "network_info": network_info,
        "coordinates": coordinates,
        "area": area,
        "graph": graph,
    }


def find_path(network, conn1, conn2, debug=0, debug_print=None):
    """Find shortest path between two nodes in the network.

    Args:
        network: dict from extract_network()
        conn1: First connection point (node ID)
        conn2: Second connection point (node ID)
        debug: Debug level (0=off, 1=on)
        debug_print: Optional debug print function

    Returns:
        (distance, path, short_path_resistance)
    """
    if debug_print is None:
        debug_print = print if debug else lambda _: None

    graph = network["graph"]
    resistors = network["resistors"]
    edges = [(i[0], i[1], i[3]) for i in resistors]

    try:
        distance, path = find_shortest_path(graph, conn1, conn2)
        short_path_res = Get_shortest_path_RES(path, resistors)
    except Exception as e:
        short_path_res = -1
        distance = float("inf")
        path = []
        debug_print(f"[DEBUG] Path finding failed: {e}")
        debug_print(f"[DEBUG] Graph: {len(graph)} nodes, {len(edges)} edges")
        if conn1 not in graph:
            debug_print(f"[DEBUG] conn1={conn1} NOT in graph")
        if conn2 not in graph:
            debug_print(f"[DEBUG] conn2={conn2} NOT in graph")
        if conn1 in graph and conn2 in graph:
            reachable = find_all_reachable_nodes(graph, conn1)
            debug_print(f"[DEBUG] conn2 reachable from conn1: {conn2 in reachable}")

    return distance, path, short_path_res


def simulate_network(
    network, conn1, conn2, CuStack, frequencies=None, debug=0, debug_print=None
):
    """Run DC simulation and optionally AC simulations with HF parameters.

    If frequencies are given, computes HF parameters via analyze_trace for each
    WIRE element and enriches network_info with the "hf" key before simulation.

    Args:
        network: dict from extract_network()
        conn1: First connection point (node ID)
        conn2: Second connection point (node ID)
        CuStack: Copper stackup information
        frequencies: list of frequencies in Hz for HF analysis
        debug: Debug level (0=off, 1=on)
        debug_print: Optional debug print function

    Returns:
        (resistance_dc, impedance_ac, network_info)
        network_info is enriched with "hf" key for WIRE elements if frequencies given.
    """
    if debug_print is None:
        debug_print = print if debug else lambda _: None

    resistors = network["resistors"]
    network_info = network["network_info"]

    # Compute HF parameters for WIRE elements
    for elem in network_info:
        if elem["type"] != "WIRE":
            continue
        hf = {}
        for f in frequencies or []:
            hf[f] = analyze_trace(
                elem["length"], elem["width"], elem["layer"], CuStack, frequency=f
            )
            wr = hf[f].get("wavelength_ratio")
            if wr and wr > 0.05:
                debug_print(
                    f"WARNING: WIRE {elem['layer_name']} "
                    f"L={elem['length'] * 1000:.2f}mm is {wr:.2f}λ at {f / 1e6:.0f}MHz "
                    f"(> λ/20, should be segmented)"
                )
        elem["hf"] = hf

    try:
        resistance_dc, impedance_ac = RunSimulation(
            resistors,
            conn1,
            conn2,
            network_info,
            frequencies,
            debug=debug,
            debug_print=debug_print,
        )
    except Exception:
        resistance_dc = -1
        impedance_ac = {}
        print(traceback.format_exc())
        print("ERROR in RunSimulation")

    return resistance_dc, impedance_ac, network_info


if __name__ == "__main__":
    die_core = {"h": 1.51e-3, "epsilon_r": 4.5, "loss_tangent": 0.02}
    die_pp = {"h": 0.2e-3, "epsilon_r": 4.5, "loss_tangent": 0.02}
    die_core2 = {"h": 1.0e-3, "epsilon_r": 4.3, "loss_tangent": 0.02}

    # 2-Layer: F.Cu/B.Cu → Microstrip
    cu_2L = {
        0: {
            "thickness": 35e-6,
            "name": "F.Cu",
            "abs_height": 0.0,
            "die_above": None,
            "die_below": die_core,
            "model": "Microstrip",
        },
        2: {
            "thickness": 35e-6,
            "name": "B.Cu",
            "abs_height": 1.545e-3,
            "die_above": die_core,
            "die_below": None,
            "model": "Microstrip",
        },
    }
    for lid in (0, 2):
        tr = analyze_trace(0.01, 0.2e-3, lid, cu_2L, 100e6)
        assert tr["z0"] is not None and tr["r_ac"] > tr["r_dc"]

    # 4-Layer: F.Cu/B.Cu → Microstrip, In1/In2 → Stripline
    cu_4L = {
        0: {
            "thickness": 35e-6,
            "name": "F.Cu",
            "abs_height": 0.0,
            "die_above": None,
            "die_below": die_pp,
            "model": "Microstrip",
        },
        4: {
            "thickness": 35e-6,
            "name": "In1.Cu",
            "abs_height": 0.235e-3,
            "die_above": die_pp,
            "die_below": die_core2,
            "model": "Stripline",
        },
        6: {
            "thickness": 35e-6,
            "name": "In2.Cu",
            "abs_height": 1.270e-3,
            "die_above": die_core2,
            "die_below": die_pp,
            "model": "Stripline",
        },
        2: {
            "thickness": 35e-6,
            "name": "B.Cu",
            "abs_height": 1.505e-3,
            "die_above": die_pp,
            "die_below": None,
            "model": "Microstrip",
        },
    }
    for lid in (0, 4, 6, 2):
        tr = analyze_trace(0.01, 0.2e-3, lid, cu_4L, 100e6)
        assert tr["z0"] is not None and tr["r_ac"] > tr["r_dc"]

    # No dielectric → DC-only
    cu_fb = {0: {"thickness": 35e-6, "name": "F.Cu", "abs_height": 0.0}}
    tr = analyze_trace(0.01, 0.2e-3, 0, cu_fb, 100e6)
    assert tr["r_ac"] == tr["r_dc"] and tr["z0"] is None

    print("analyze_trace() OK")
