import logging
import math
import os
import tempfile
from typing import Any, NamedTuple
import numpy as np


class Resistor(NamedTuple):
    """A resistive edge in the PCB copper network."""

    node1: int
    node2: int
    resistance: float  # DC resistance in Ohm
    length: float  # physical length in m


try:
    from .pcb_types import (
        WIRE,
        VIA,
        PAD,
        ZONE,
        NetworkElement,
        CuLayer,
        DielectricInfo,
        SpiceElement,
    )
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
    from pcb_types import (
        WIRE,
        VIA,
        PAD,
        ZONE,
        NetworkElement,
        CuLayer,
        DielectricInfo,
        SpiceElement,
    )

log = logging.getLogger(__name__)


def analyze_trace(
    length: float,
    width: float,
    cu_layer_id: int,
    CuStack: dict[int, CuLayer],
    frequency: float = 100e6,
) -> dict[str, Any]:
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
        assert die is not None
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
        assert die is not None
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
    filename: str,
    elements: list[SpiceElement],
    conn1: int,
    conn2: int,
    ac_freq: float | None = None,
    far_node: int | None = None,
    rload_far: float | None = None,
) -> float | complex:
    """Write netlist, run simulation, return impedance.

    Circuit topology
    ----------------
    The DUT (R/L/C network) is connected between conn1 and conn2.
    Capacitors inside the DUT reference GND (node 0) as the ground plane.
    Vmeas (0 V) and v1 (1 V AC) are in series and form the port excitation.
    Two large Rref resistors anchor both port nodes to GND for DC convergence.

    conn1 o──────────────────── DUT ─────────────────── o conn2
          │       [R─L series per seg,                  │
          │        C shunt to GND per seg]              │
        [Rref1]                                      [Rref2]
        [1e15Ω]                                      [1e15Ω]
          │                                             │
         GND                                           GND

    Port excitation / current sense (series branch):

    conn1 o──(+)Vmeas 0V(-)──o _p ──(+)v1 AC=1V(-)──o conn2

    Voltage relations (from the two sources in series):
        V(conn1) - V(_p)   = 0 V   → V(conn1) = V(_p)
        V(_p)   - V(conn2) = 1 V
        ──────────────────────────
        V(conn1) - V(conn2) = 1 V  (effective port voltage)

    Sign convention (verified with DUT = 50 Ω → result must be +50 Ω):
        Z = -1 / I(Vmeas)
    ngspice defines I(Vmeas) as positive when current flows from conn1
    into the (+) terminal of Vmeas; with a resistive DUT that current
    flows conn2 → v1 → _p → Vmeas → conn1 → DUT → conn2, so
    I(Vmeas) < 0 and the minus sign yields a positive Z.

    Args:
        elements: list of ("R"/"L"/"C", node1, node2, value)
        ac_freq: None for DC, frequency in Hz for AC
        far_node: when set (with rload_far), the node at the far end that gets
            Rload to GND; port return is GND instead of conn2
        rload_far: load resistance in Ohm placed from far_node to GND
    Returns:
        For DC: real resistance in Ohm (float)
        For AC: complex impedance in Ohm (complex)
        On error: -1
    """
    with open(filename, "w") as f:
        f.write("* Parasitic Network\n")
        for idx, (kind, na, nb, val) in enumerate(elements, 1):
            f.write(f"{kind}{idx} {na} {nb} {val:.15f}\n")
        # Port excitation between conn1 and conn2.
        # Vmeas (0 V) measures the port current; V1 drives conn1–conn2 = 1 V.
        # Z = V_port / I_into_port = 1 / (-I(Vmeas)).
        # Rref1/2 (1e15 Ω) provide a DC path for both port nodes so that
        # ngspice does not see a floating node. At 1e15 Ω the loading is
        # < 1e-15 A/V and has no effect on the measurement.
        f.write(f"Vmeas {conn1} _p 0\n")
        if far_node is not None and rload_far is not None:
            # Single-ended mode: port return is GND, Rload terminates far end
            f.write("v1 _p 0 dc 0 ac 1\n")
            f.write(f"Rref1 {conn1} 0 1e15\n")
            f.write(f"Rload {far_node} 0 {rload_far:.15f}\n")
        else:
            # Differential mode (default): port return is conn2
            f.write(f"v1 _p {conn2} dc 0 ac 1\n")
            f.write(f"Rref1 {conn1} 0 1e15\n")
            f.write(f"Rref2 {conn2} 0 1e15\n")
        if ac_freq:
            f.write(f".ac lin 1 {ac_freq} {ac_freq}\n")
        f.write(".end\n")

    se_tag = " SE" if far_node is not None else ""
    mode = f"AC{se_tag} @ {ac_freq:.0f} Hz" if ac_freq else "DC"
    log.info(
        "[SPICE] %s: %d elements, conn %s <-> %s", mode, len(elements), conn1, conn2
    )
    if log.isEnabledFor(logging.DEBUG):
        with open(filename) as _f:
            log.debug("[SPICE] netlist:\n%s", _f.read())

    ngspyce.source(filename)
    if ac_freq:
        ngspyce.cmd("run")
    else:
        ngspyce.dc("v1", 1, 1, 1)
    os.remove(filename)

    log.info("[SPICE] done")

    i_meas = ngspyce.vector("vmeas#branch")
    if len(i_meas) > 0 and i_meas[0] != 0:
        z: float | complex = -1.0 / i_meas[0]
        if ac_freq and isinstance(z, complex):
            # S11 with Z0=50 Ω reference — informational only, not returned
            z0_ref = 50.0
            s11 = (z - z0_ref) / (z + z0_ref)
            log.info(
                "[SPICE]   I=%s, Z=%.3f mΩ (%.3f + j%.3f mΩ)  S11=%.2f dB",
                f"{i_meas[0]:.6e}",
                abs(z) * 1000,
                z.real * 1000,
                z.imag * 1000,
                20 * math.log10(abs(s11) + 1e-15),
            )
        else:
            log.info("[SPICE]   I=%s, Z=%.3f mOhm", f"{i_meas[0]:.6e}", abs(z) * 1000)
        return z
    log.warning("[SPICE]   I=%s, simulation failed", i_meas)
    return -1


def RunSimulation(
    resistors: list[Resistor],
    conn1: int,
    conn2: int,
    network_info: list[NetworkElement] | None = None,
    frequencies: list[float] | None = None,
    rload_far: float | None = None,
) -> tuple[
    float | complex,
    dict[float, float | complex],
    dict[float, float | complex],
    dict[float, float | complex],
]:
    """Run DC simulation and optionally AC simulations at given frequencies.

    Returns:
        r_dc: DC resistance in Ohm
        z_ac: dict {freq: differential impedance} for each frequency
        z_ac_conn1: dict {freq: single-ended Zin at conn1} (empty if rload_far None)
        z_ac_conn2: dict {freq: single-ended Zin at conn2} (empty if rload_far None)
    """
    # https://github.com/ignamv/ngspyce/
    _fd, filename = tempfile.mkstemp(suffix=".net", prefix="parasitic_")
    os.close(_fd)
    gnd = 0

    # --- DC: R-only network ---
    dc_elements: list[SpiceElement] = [
        ("R", r.node1, r.node2, r.resistance) for r in resistors
    ]
    r_dc = _run_spice(filename, dc_elements, conn1, conn2)

    # --- AC: RLC network per frequency ---
    z_ac: dict[float, float | complex] = {}
    z_ac_conn1: dict[float, float | complex] = {}
    z_ac_conn2: dict[float, float | complex] = {}
    for freq in frequencies or []:
        ac: list[SpiceElement] = []
        idx = 0
        for elem in network_info or []:
            n1, n2 = elem["nodes"]
            idx += 1

            if elem["type"] == WIRE:
                hf = elem.get("hf", {}).get(freq)
                R = hf["r_ac"] if hf else elem["resistance"]
                L = hf.get("inductance") if hf else None
                C = hf.get("capacitance") if hf else None

                # Segment if > lambda/20
                wr = hf.get("wavelength_ratio") if hf else None
                n_seg = max(1, int(wr / 0.05) + 1) if wr and wr > 0.05 else 1

                if L:
                    # Distributed RLGC model (n_seg=1 for lumped)
                    # TODO: Use π-topology (C/2 at both segment ends) instead of
                    # L-topology (C at left end only). π is more accurate at high
                    # electrical lengths but requires adding C_seg/2 at n2 of last
                    # segment as well, which needs care at network junctions.
                    R_seg, L_seg = R / n_seg, L / n_seg
                    C_seg = C / n_seg if C else None
                    prev: int | str = n1
                    for s in range(n_seg):
                        nxt: int | str = n2 if s == n_seg - 1 else f"w{idx}s{s}"
                        mid = f"w{idx}s{s}m"
                        ac.append(("R", prev, mid, R_seg))
                        ac.append(("L", mid, nxt, L_seg))
                        if C_seg:
                            ac.append(("C", prev, gnd, C_seg))
                        prev = nxt
                else:
                    # R-only (no HF data)
                    ac.append(("R", n1, n2, R))

            elif elem["type"] == VIA:
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

        z_ac[freq] = _run_spice(filename, ac, conn1, conn2, ac_freq=freq)

        if rload_far is not None:
            z_ac_conn1[freq] = _run_spice(
                filename,
                ac,
                conn1,
                0,
                ac_freq=freq,
                far_node=conn2,
                rload_far=rload_far,
            )
            z_ac_conn2[freq] = _run_spice(
                filename,
                ac,
                conn2,
                0,
                ac_freq=freq,
                far_node=conn1,
                rload_far=rload_far,
            )

    return r_dc, z_ac, z_ac_conn1, z_ac_conn2


def Get_shortest_path_RES(path: list[int], resistors: list[Resistor]) -> float:
    res_map = {
        (min(r.node1, r.node2), max(r.node1, r.node2)): r.resistance for r in resistors
    }
    return sum(
        res_map[min(path[i - 1], path[i]), max(path[i - 1], path[i])]
        for i in range(1, len(path))
    )


def extract_network(
    data: dict[Any, Any], CuStack: dict[int, CuLayer], netcode: int | None = None
) -> dict[str, Any]:
    """Extract electrical network from PCB data (DC resistances only).

    # TODO: Split into separate functions for wire, via, and zone resistance
    # calculation; function is currently ~230 lines.

    Args:
        data: PCB element data dictionary
        CuStack: Copper stackup information
        netcode: KiCad net code to filter elements (None = use all elements)

    Returns:
        dict with:
            resistors: list of [node1, node2, r_dc, length]
            network_info: list of element dicts (WIRE/VIA/ZONE) with geometric data
            coordinates: dict {node: (x, y, z)}
            area: dict {layer: area}
            graph: adjacency dict from get_graph_from_edges
    """

    resistors: list[Resistor] = []
    network_info: list[NetworkElement] = []
    coordinates: dict[int, tuple[float, float, float]] = {}
    Area = {layer_idx: 0 for layer_idx in range(32)}

    for uuid, d in data.items():
        if netcode is not None and d["net_code"] != netcode:
            continue

        if len(d["layer"]) > 1:
            # Multi-layer element (VIA or through-hole PAD)
            # Collect all valid nodes for this element
            nodes = []
            for layer in d["layer"]:
                if layer not in CuStack:
                    log.warning("Layer %s not in CuStack, skipping", layer)
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
                            log.error(
                                "CuStack incomplete for layers %s, %s", Layer1, Layer2
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
                        resistors.append(
                            Resistor(node1, node2, via["resistance"], distance)
                        )
                        network_info.append(
                            {
                                "type": VIA,
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

        elif len(d["layer"]) == 1:
            Layer = d["layer"][0]
            Area[Layer] += d["area"]

            if Layer not in CuStack:
                log.warning("Layer %s not in CuStack, skipping element", Layer)
                continue

            if d["type"] == WIRE:
                netStart = d["net_start"].get(Layer, 0)
                netEnd = d["net_end"].get(Layer, 0)

                if netStart == 0 or netEnd == 0:
                    log.debug(
                        "Skipping WIRE: netStart=%s, netEnd=%s, connStart=%s, connEnd=%s",
                        netStart,
                        netEnd,
                        d.get("connStart", []),
                        d.get("connEnd", []),
                    )
                    continue

                trace = analyze_trace(
                    d["length"], d["width"], Layer, CuStack, frequency=0
                )
                if trace["r_dc"] <= 0:
                    raise ValueError(
                        f"Invalid resistance for WIRE (layer={Layer}, "
                        f"length={d['length']}, width={d['width']}): r_dc={trace['r_dc']}"
                    )
                resistors.append(Resistor(netStart, netEnd, trace["r_dc"], d["length"]))

                info: NetworkElement = {
                    "type": WIRE,
                    "nodes": (netStart, netEnd),
                    "resistance": trace["r_dc"],
                    "length": d["length"],
                    "width": d["width"],
                    "layer": Layer,
                    "layer_name": CuStack[Layer]["name"],
                    "start": d["start"],
                    "end": d["end"],
                }
                if "mid" in d:
                    info["mid"] = d["mid"]
                if "radius" in d:
                    info["radius"] = d["radius"]
                if "angle" in d:
                    info["angle"] = d["angle"]
                if "midline_pts" in d:
                    info["midline_pts"] = d["midline_pts"]
                network_info.append(info)

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

            elif d["type"] == PAD:
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
    zone_connections: dict[int, dict[Any, list[int]]] = {}
    for uuid, d in data.items():
        if (netcode is not None and d["net_code"] != netcode) or d["type"] != ZONE:
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

                if conn_item.get("type") == WIRE:
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
        for zone_nodes in zones.values():
            for i, node1 in enumerate(zone_nodes):
                for node2 in zone_nodes[i + 1 :]:
                    zone_distance = 0.0
                    if node1 in coordinates and node2 in coordinates:
                        c1 = coordinates[node1]
                        c2 = coordinates[node2]
                        zone_distance = np.sqrt(
                            (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
                        )
                    resistors.append(
                        Resistor(node1, node2, zone_resistance, zone_distance)
                    )
                    network_info.append(
                        {
                            "type": ZONE,
                            "nodes": (node1, node2),
                            "resistance": zone_resistance,
                            "length": zone_distance,
                            "layer": layer,
                            "layer_name": layer_name,
                        }
                    )

    # Build graph
    edges = [(r.node1, r.node2, r.length) for r in resistors]
    graph = get_graph_from_edges(edges)

    return {
        "resistors": resistors,
        "network_info": network_info,
        "coordinates": coordinates,
        "area": area,
        "graph": graph,
    }


def find_path(
    network: dict[str, Any], conn1: int, conn2: int
) -> tuple[float, list[int], float]:
    """Find shortest path between two nodes in the network.

    Args:
        network: dict from extract_network()
        conn1: First connection point (node ID)
        conn2: Second connection point (node ID)

    Returns:
        (distance, path, short_path_resistance)
    """
    graph = network["graph"]
    resistors = network["resistors"]
    edges = [(r.node1, r.node2, r.length) for r in resistors]

    try:
        distance, path = find_shortest_path(graph, conn1, conn2)
        short_path_res = Get_shortest_path_RES(path, resistors)
    except Exception as e:
        short_path_res = -1
        distance = float("inf")
        path = []
        log.warning("Path finding failed: %s", e)
        log.debug("Graph: %d nodes, %d edges", len(graph), len(edges))
        if conn1 not in graph:
            log.debug("conn1=%s NOT in graph", conn1)
        if conn2 not in graph:
            log.debug("conn2=%s NOT in graph", conn2)
        if conn1 in graph and conn2 in graph:
            reachable = find_all_reachable_nodes(graph, conn1)
            log.debug("conn2 reachable from conn1: %s", conn2 in reachable)

    return distance, path, short_path_res


def simulate_network(
    network: dict[str, Any],
    conn1: int,
    conn2: int,
    CuStack: dict[int, CuLayer],
    frequencies: list[float] | None = None,
    rload_far: float | None = None,
) -> tuple[
    float | complex,
    dict[float, float | complex],
    dict[float, float | complex],
    dict[float, float | complex],
    list[NetworkElement],
]:
    """Run DC simulation and optionally AC simulations with HF parameters.

    If frequencies are given, computes HF parameters via analyze_trace for each
    WIRE element and enriches network_info with the "hf" key before simulation.

    Args:
        network: dict from extract_network()
        conn1: First connection point (node ID)
        conn2: Second connection point (node ID)
        CuStack: Copper stackup information
        frequencies: list of frequencies in Hz for HF analysis
        rload_far: when set, also measure single-ended Zin at conn1 and conn2
            with this load resistance (in Ohm) at the far end

    Returns:
        (resistance_dc, impedance_ac, z_ac_conn1, z_ac_conn2, network_info)
        network_info is enriched with "hf" key for WIRE elements if frequencies given.
        z_ac_conn1/z_ac_conn2 are empty dicts when rload_far is None.
    """
    resistors = network["resistors"]
    network_info = network["network_info"]

    # Compute HF parameters for WIRE elements
    for elem in network_info:
        if elem["type"] != WIRE:
            continue
        hf = {}
        for f in frequencies or []:
            hf[f] = analyze_trace(
                elem["length"], elem["width"], elem["layer"], CuStack, frequency=f
            )
            wr = hf[f].get("wavelength_ratio")
            if wr and wr > 0.05:
                log.info(
                    "WIRE %s L=%.2fmm is %.2fλ at %.0fMHz (> λ/20, segmented into %d parts)",
                    elem["layer_name"],
                    elem["length"] * 1000,
                    wr,
                    f / 1e6,
                    max(1, int(wr / 0.05) + 1),
                )
        if hf:
            elem["hf"] = hf

    try:
        resistance_dc, impedance_ac, z_ac_conn1, z_ac_conn2 = RunSimulation(
            resistors,
            conn1,
            conn2,
            network_info,
            frequencies,
            rload_far,
        )
    except Exception:
        resistance_dc = -1
        impedance_ac = {}
        z_ac_conn1 = {}
        z_ac_conn2 = {}
        log.exception("RunSimulation failed")

    return resistance_dc, impedance_ac, z_ac_conn1, z_ac_conn2, network_info


if __name__ == "__main__":
    die_core: DielectricInfo = {"h": 1.51e-3, "epsilon_r": 4.5, "loss_tangent": 0.02}
    die_pp: DielectricInfo = {"h": 0.2e-3, "epsilon_r": 4.5, "loss_tangent": 0.02}
    die_core2: DielectricInfo = {"h": 1.0e-3, "epsilon_r": 4.3, "loss_tangent": 0.02}

    # 2-Layer: F.Cu/B.Cu → Microstrip
    cu_2L: dict[int, CuLayer] = {
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
    cu_4L: dict[int, CuLayer] = {
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
    cu_fb: dict[int, CuLayer] = {
        0: {"thickness": 35e-6, "name": "F.Cu", "abs_height": 0.0}
    }
    tr = analyze_trace(0.01, 0.2e-3, 0, cu_fb, 100e6)
    assert tr["r_ac"] == tr["r_dc"] and tr["z0"] is None

    print("analyze_trace() OK")
