from __future__ import annotations

import cmath
import logging
import math
from collections import defaultdict
from contextlib import closing
from typing import Any, Callable, NamedTuple, Union

# Resistance tying each port node to GND. It must stay far above any port
# impedance (loading error < 1e-7 at 50 Ω) yet close enough to milliohm trace
# resistances for a well-conditioned matrix; 1e15 Ω made the DC operating point
# fail with "singular matrix".
_RREF_OHM = 1e9

_GND = 0


Impedance = Union[float, complex]


class Resistor(NamedTuple):
    """A resistive edge in the PCB copper network."""

    node1: int
    node2: int
    resistance: float  # DC resistance in Ohm
    length: float  # physical length in m


try:
    from .analysis_progress import Progress, Stage, no_progress, report
    from .get_distance import (
        DisjointSets,
        find_all_reachable_nodes,
        find_shortest_path,
        get_graph_from_edges,
    )
    from .impedance import (
        RHO_CU,
        analyze_coplanar,
        analyze_microstrip,
        analyze_stripline,
        get_Via_Parasitics,
    )
    from .pcb_types import (
        VIA,
        WIRE,
        ZONE,
        CuLayer,
        NetworkElement,
        SpiceElement,
    )
    from .spice_worker import SpiceError, SpiceWorker
except ImportError:
    from analysis_progress import Progress, Stage, no_progress, report
    from get_distance import (
        DisjointSets,
        find_all_reachable_nodes,
        find_shortest_path,
        get_graph_from_edges,
    )
    from impedance import (
        RHO_CU,
        analyze_coplanar,
        analyze_microstrip,
        analyze_stripline,
        get_Via_Parasitics,
    )
    from pcb_types import (
        VIA,
        WIRE,
        ZONE,
        CuLayer,
        NetworkElement,
        SpiceElement,
    )
    from spice_worker import SpiceError, SpiceWorker

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


# Failures of a single simulation: SpiceError (including a timeout or a dead
# ngspice process), ValueError for invalid elements, and the RuntimeError or
# OSError that ngspyce and ctypes raise in the ngspice process.
_SPICE_FAILURES = (RuntimeError, ValueError, OSError)

# Failures of the parasitics of a single element; the element is skipped.
_ELEMENT_FAILURES = (ValueError, ZeroDivisionError)


def _checked_resistance(resistance: float) -> float:
    """`resistance` if it is finite and positive, else ValueError."""
    if not (math.isfinite(resistance) and resistance > 0):
        raise ValueError(f"resistance {resistance}")
    return resistance


def _run_spice(
    spice: SpiceWorker,
    elements: list[SpiceElement],
    conn1: int,
    conn2: int,
    ac_freq: float | None = None,
    far_node: int | None = None,
    rload_far: float | None = None,
) -> complex:
    """Simulate the DUT between conn1 and conn2 and return its port impedance.

    Circuit topology
    ----------------
    The DUT (R/L/C network) is connected between conn1 and conn2.
    Capacitors inside the DUT reference GND (node 0) as the ground plane.
    Vmeas (0 V) and v1 (1 V AC) are in series and form the port excitation.
    Two Rref resistors (_RREF_OHM) anchor both port nodes to GND.

    conn1 o──────────────────── DUT ─────────────────── o conn2
          │       [R─L series per seg,                  │
          │        C shunt to GND per seg]              │
        [Rref1]                                      [Rref2]
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
        Impedance in Ohm; for DC the imaginary part is zero.
    Raises:
        SpiceError: ngspice failed (see SpiceWorker.measure), no finite
            impedance, or a non-positive DC resistance
        ValueError: non-finite element value or invalid rload_far
    """
    if rload_far is not None and not (math.isfinite(rload_far) and rload_far > 0):
        raise ValueError(f"rload_far must be finite and > 0, got {rload_far}")

    lines = []
    for idx, (kind, na, nb, val) in enumerate(elements, 1):
        if not math.isfinite(val):
            raise ValueError(
                f"Non-finite value in SPICE element {kind}{idx} ({na}->{nb}): {val}"
            )
        lines.append(f"{kind}{idx} {na} {nb} {val:.15g}")
    lines.append(f"Vmeas {conn1} _p 0")
    lines.append(f"Rref1 {conn1} 0 {_RREF_OHM:g}")
    if far_node is not None and rload_far is not None:
        # Single-ended mode: port return is GND, Rload terminates the far end
        lines.append("v1 _p 0 dc 0 ac 1")
        lines.append(f"Rload {far_node} 0 {rload_far:.15g}")
    else:
        # Differential mode: port return is conn2
        lines.append(f"v1 _p {conn2} dc 0 ac 1")
        lines.append(f"Rref2 {conn2} 0 {_RREF_OHM:g}")
    if ac_freq:
        lines.append(f".ac lin 1 {ac_freq} {ac_freq}")
    else:
        lines.append(".dc v1 1 1 1")

    se_tag = " SE" if far_node is not None else ""
    mode = f"AC{se_tag} @ {ac_freq:.0f} Hz" if ac_freq else "DC"
    log.info(
        "[SPICE] %s: %d elements, conn %s <-> %s", mode, len(elements), conn1, conn2
    )
    log.debug("[SPICE] netlist:\n%s", "\n".join(lines))

    i_meas = spice.measure(lines, "vmeas#branch")
    if i_meas == 0 or not cmath.isfinite(i_meas):
        raise SpiceError(f"No finite port current (I={i_meas})")
    z = -1.0 / i_meas
    if not cmath.isfinite(z) or (not ac_freq and z.real <= 0):
        raise SpiceError(f"Implausible port impedance {z}")
    if ac_freq:
        # S11 with Z0=50 Ω reference — informational only, not returned
        z0_ref = 50.0
        s11 = (z - z0_ref) / (z + z0_ref)
        log.info(
            "[SPICE]   I=%s, Z=%.3f mΩ (%.3f + j%.3f mΩ)  S11=%.2f dB",
            f"{i_meas:.6e}",
            abs(z) * 1000,
            z.real * 1000,
            z.imag * 1000,
            20 * math.log10(abs(s11) + 1e-15),
        )
    else:
        log.info("[SPICE]   I=%s, Z=%.3f mOhm", f"{i_meas:.6e}", abs(z) * 1000)
    return z


def _ac_elements(
    network_info: list[NetworkElement],
    freq: float,
    conducts: Callable[[int, int], bool],
) -> list[SpiceElement]:
    """RLC netlist elements of the network at `freq`, skipping non-conducting ones."""
    ac: list[SpiceElement] = []
    for idx, elem in enumerate(network_info, start=1):
        n1, n2 = elem["nodes"]
        if not conducts(n1, n2):
            continue

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
                        ac.append(("C", prev, _GND, C_seg))
                    prev = nxt
            else:
                # R-only (no HF data)
                ac.append(("R", n1, n2, R))

        elif elem["type"] == VIA:
            R = elem["resistance"]
            L = elem.get("inductance")
            C = elem.get("capacitance")
            # R-L in series, C to the surrounding ground planes
            if L:
                mid = f"mid{idx}"
                ac.append(("R", n1, mid, R))
                ac.append(("L", mid, n2, L))
            else:
                ac.append(("R", n1, n2, R))
            if C:
                ac.append(("C", n1, _GND, C))

        else:  # ZONE
            R = elem["resistance"]
            ac.append(("R", n1, n2, R))
    return ac


def RunSimulation(
    resistors: list[Resistor],
    conn1: int,
    conn2: int,
    network_info: list[NetworkElement] | None = None,
    frequencies: list[float] | None = None,
    rload_far: float | None = None,
    progress: Progress = no_progress,
) -> tuple[
    float | None,
    dict[float, Impedance | None],
    dict[float, Impedance | None],
    dict[float, Impedance | None],
]:
    """Run DC simulation and optionally AC simulations at given frequencies.

    Each simulation fails independently; a failed one yields None.

    Returns:
        r_dc: DC resistance in Ohm
        z_ac: dict {freq: differential impedance} for each frequency
        z_ac_conn1: dict {freq: single-ended Zin at conn1} (empty if rload_far None)
        z_ac_conn2: dict {freq: single-ended Zin at conn2} (empty if rload_far None)
    Raises:
        SpiceUnavailableError: ngspice cannot be started
        CancelledError: `progress` asked to stop
    """
    # Simulate only elements between the ports: copper not connected to conn1
    # would float, and an element whose terminals share a node carries no current
    # through it (its shunt capacitance is neglected).
    network_info = network_info or []
    frequencies = frequencies or []
    edges = [(r.node1, r.node2, 0.0) for r in resistors]
    edges += [(*e["nodes"], 0.0) for e in network_info]
    connected = find_all_reachable_nodes(get_graph_from_edges(edges), conn1)

    def conducts(n1: int, n2: int) -> bool:
        return n1 != n2 and n1 in connected

    z_ac: dict[float, Impedance | None] = {}
    z_ac_conn1: dict[float, Impedance | None] = {}
    z_ac_conn2: dict[float, Impedance | None] = {}
    # (label, result store, port, port return, extra _run_spice arguments)
    ac_ports: list[tuple[str, dict[float, Impedance | None], int, int, dict[str, Any]]]
    ac_ports = [("AC", z_ac, conn1, conn2, {})]
    if rload_far is not None:
        far1 = {"far_node": conn2, "rload_far": rload_far}
        far2 = {"far_node": conn1, "rload_far": rload_far}
        ac_ports.append(("Zin(conn1)", z_ac_conn1, conn1, _GND, far1))
        ac_ports.append(("Zin(conn2)", z_ac_conn2, conn2, _GND, far2))
    total = 1 + len(frequencies) * len(ac_ports)
    finished = 0

    def simulate(
        label: str, elements: list[SpiceElement], port: int, ret: int, **kw: Any
    ) -> Impedance | None:
        nonlocal finished
        try:
            return _run_spice(spice, elements, port, ret, **kw)
        except _SPICE_FAILURES as e:
            log.error("[SPICE] %s failed: %s", label, e)
            return None
        finally:
            finished += 1

    dc_elements: list[SpiceElement] = [
        ("R", r.node1, r.node2, r.resistance)
        for r in resistors
        if conducts(r.node1, r.node2)
    ]

    def on_wait() -> None:
        report(progress, Stage.SPICE, finished, total)

    with closing(SpiceWorker(on_wait)) as spice:
        dc = simulate("DC", dc_elements, conn1, conn2)
        for freq in frequencies:
            ac = _ac_elements(network_info, freq, conducts)
            for label, store, port, ret, kw in ac_ports:
                store[freq] = simulate(
                    f"{label} at {freq:.0f} Hz", ac, port, ret, ac_freq=freq, **kw
                )

    r_dc = None if dc is None else dc.real
    return r_dc, z_ac, z_ac_conn1, z_ac_conn2


def Get_shortest_path_RES(path: list[int], resistors: list[Resistor]) -> float:
    res_map = {
        (min(r.node1, r.node2), max(r.node1, r.node2)): r.resistance for r in resistors
    }
    return sum(
        res_map[min(path[i - 1], path[i]), max(path[i - 1], path[i])]
        for i in range(1, len(path))
    )


# Resistance from each connection of a zone to its star node: zones are nearly
# ideal conductors, the simulator needs a finite value.
ZONE_LINK_OHM = 1e-3


def _zone_links(
    data: dict[Any, Any], CuStack: dict[int, CuLayer]
) -> list[NetworkElement]:
    """A star node per zone area and layer, linked to every node touching it.

    Zones that touch on a layer form one area. The links have no length, so a
    path crosses a zone as an ideal conductor. An area touched at a single node
    carries no current and gets no star. Star nodes follow the node IDs of
    Connect_Nets.
    """
    zones = {d["id"]: d for d in data.values() if d["type"] == ZONE}
    next_node = 1 + max(
        (
            n
            for d in data.values()
            if d["type"] != ZONE
            for side in (
                ("net_start", "net_end") if d["type"] == WIRE else ("net_start",)
            )
            for n in d[side].values()
        ),
        default=0,
    )
    links: list[NetworkElement] = []
    for layer in sorted({layer for z in zones.values() for layer in z["layer"]}):
        areas: DisjointSets[str] = DisjointSets()
        touching: defaultdict[str, set[int]] = defaultdict(set)
        for zone_id, z in zones.items():
            if layer not in z["layer"]:
                continue
            for conn in z["conn_start"]:
                c = data.get(conn)
                if c is None or layer not in c["layer"]:
                    continue
                if c["type"] == ZONE:
                    areas.union(zone_id, conn)
                elif c["type"] == WIRE:
                    touching[zone_id].update(
                        c[side][layer]
                        for side, key in (
                            ("net_start", "conn_start"),
                            ("net_end", "conn_end"),
                        )
                        if zone_id in c[key]
                    )
                else:
                    touching[zone_id].add(c["net_start"][layer])
        nodes_of: defaultdict[str, set[int]] = defaultdict(set)
        for zone_id, nodes in touching.items():
            nodes_of[areas.find(zone_id)] |= nodes
        layer_name = CuStack[layer]["name"] if layer in CuStack else f"Layer_{layer}"
        for nodes in nodes_of.values():
            if len(nodes) < 2:
                continue
            star, next_node = next_node, next_node + 1
            links += [
                {
                    "type": ZONE,
                    "nodes": (node, star),
                    "resistance": ZONE_LINK_OHM,
                    "length": 0.0,
                    "layer": layer,
                    "layer_name": layer_name,
                }
                for node in sorted(nodes)
            ]
    return links


def extract_network(
    data: dict[Any, Any],
    CuStack: dict[int, CuLayer],
    progress: Progress = no_progress,
) -> dict[str, Any]:
    """Extract electrical network from PCB data (DC resistances only).

    Args:
        data: PCB element data dictionary
        CuStack: Copper stackup information
        progress: reports each element; see Progress

    An element that cannot be modelled (layer missing from the stackup, or no
    finite positive resistance) is logged, skipped and listed in "skipped".

    Returns:
        dict with:
            resistors: list of Resistor
            network_info: list of element dicts (WIRE/VIA/ZONE) with geometric data
            area: dict {layer: area}
            graph: adjacency dict from get_graph_from_edges
            skipped: descriptions of the elements that were left out
    Raises:
        CancelledError: `progress` asked to stop
    """

    resistors: list[Resistor] = []
    network_info: list[NetworkElement] = []
    Area: defaultdict[int, float] = defaultdict(float)
    skipped: list[str] = []

    def skip(element: str) -> None:
        log.warning("Skipping %s", element)
        skipped.append(element)

    for finished, (uuid, d) in enumerate(data.items()):
        report(progress, Stage.EXTRACT, finished, len(data))

        if len(d["layer"]) > 1:
            # Multi-layer element (VIA or through-hole PAD)
            # Collect all valid nodes for this element
            nodes = []
            for layer in d["layer"]:
                if layer not in CuStack:
                    skip(f"{d['type']} on layer {layer}: layer not in the stackup")
                    continue
                if layer in d.get("net_start", {}):
                    node = d["net_start"][layer]
                    if node > 0:
                        nodes.append((layer, node))

            # Connect all layer nodes of this via/pad
            if len(nodes) > 1:
                first_layer = nodes[0][0]
                thickness = CuStack[first_layer]["thickness"]
                for i in range(len(nodes)):
                    Layer1, node1 = nodes[i]
                    for j in range(i + 1, len(nodes)):
                        Layer2, node2 = nodes[j]
                        if "drill" not in d:
                            continue
                        distance = abs(
                            CuStack[Layer2]["abs_height"]
                            - CuStack[Layer1]["abs_height"]
                        )
                        try:
                            via = get_Via_Parasitics(d["drill"], thickness, distance)
                            _checked_resistance(via["resistance"])
                        except _ELEMENT_FAILURES as e:
                            skip(
                                f"{d['type']} at {d['position']} between layers "
                                f"{Layer1} and {Layer2}: {e}"
                            )
                            continue
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
            if Layer not in CuStack:
                skip(f"{d['type']} on layer {Layer}: layer not in the stackup")
                continue
            Area[Layer] += d["area"]

            if d["type"] == WIRE:
                netStart = d["net_start"].get(Layer, 0)
                netEnd = d["net_end"].get(Layer, 0)

                if netStart == 0 or netEnd == 0:
                    log.debug(
                        "Skipping WIRE: netStart=%s, netEnd=%s, conn_start=%s, conn_end=%s",
                        netStart,
                        netEnd,
                        d.get("conn_start", []),
                        d.get("conn_end", []),
                    )
                    continue

                try:
                    trace = analyze_trace(
                        d["length"], d["width"], Layer, CuStack, frequency=0
                    )
                    _checked_resistance(trace["r_dc"])
                except _ELEMENT_FAILURES as e:
                    skip(
                        f"WIRE on layer {Layer} (length={d['length']}, "
                        f"width={d['width']}): {e}"
                    )
                    continue
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

    area = {layer: a for layer, a in Area.items() if a > 0}

    for link in _zone_links(data, CuStack):
        resistors.append(Resistor(*link["nodes"], link["resistance"], link["length"]))
        network_info.append(link)

    # Build graph
    edges = [(r.node1, r.node2, r.length) for r in resistors]
    graph = get_graph_from_edges(edges)

    return {
        "resistors": resistors,
        "network_info": network_info,
        "area": area,
        "graph": graph,
        "skipped": skipped,
    }


def find_path(
    network: dict[str, Any], conn1: int, conn2: int
) -> tuple[float, list[int], float | None]:
    """Find shortest path between two nodes in the network.

    Args:
        network: dict from extract_network()
        conn1: First connection point (node ID)
        conn2: Second connection point (node ID)

    Returns:
        (distance, path, short_path_resistance); (inf, [], None) if no path exists
    """
    graph = network["graph"]
    resistors = network["resistors"]
    edges = [(r.node1, r.node2, r.length) for r in resistors]

    try:
        distance, path = find_shortest_path(graph, conn1, conn2)
        short_path_res = Get_shortest_path_RES(path, resistors)
    except (ValueError, KeyError) as e:
        short_path_res = None
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
    progress: Progress = no_progress,
) -> tuple[
    float | None,
    dict[float, Impedance | None],
    dict[float, Impedance | None],
    dict[float, Impedance | None],
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
        progress: reports HF parameters and simulations; see Progress

    Returns:
        (resistance_dc, impedance_ac, z_ac_conn1, z_ac_conn2)
        network["network_info"] is enriched in place with an "hf" key for WIRE
        elements if frequencies are given.
        z_ac_conn1/z_ac_conn2 are empty dicts when rload_far is None.
        A failed simulation yields None (see RunSimulation).
    Raises:
        SpiceUnavailableError, CancelledError: see RunSimulation
    """
    resistors = network["resistors"]
    network_info = network["network_info"]

    # Compute HF parameters for WIRE elements
    for finished, elem in enumerate(network_info):
        report(progress, Stage.HF, finished, len(network_info))
        if elem["type"] != WIRE:
            continue
        hf = {}
        for f in frequencies or []:
            try:
                hf[f] = analyze_trace(
                    elem["length"], elem["width"], elem["layer"], CuStack, frequency=f
                )
            except _ELEMENT_FAILURES:
                log.exception(
                    "analyze_trace failed for WIRE %s at %.0f MHz "
                    "(L=%.2fmm, W=%.3fmm) -- falling back to DC resistance",
                    elem.get("layer_name"),
                    f / 1e6,
                    elem["length"] * 1000,
                    elem["width"] * 1000,
                )
                continue
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

    return RunSimulation(
        resistors, conn1, conn2, network_info, frequencies, rload_far, progress
    )
