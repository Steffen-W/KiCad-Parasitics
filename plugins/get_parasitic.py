from __future__ import annotations

import cmath
import logging
import math
import threading
from collections import defaultdict
from types import ModuleType
from typing import Any, NamedTuple, Union

import numpy as np

# Wall-clock budget for a single native ngspice call (see _run_native_spice_call).
SPICE_TIMEOUT_S = 30.0

# Resistance tying each port node to GND. It must stay far above any port
# impedance (loading error < 1e-7 at 50 Ω) yet close enough to milliohm trace
# resistances for a well-conditioned matrix; 1e15 Ω made the DC operating point
# fail with "singular matrix".
_RREF_OHM = 1e9


Impedance = Union[float, complex]


class Resistor(NamedTuple):
    """A resistive edge in the PCB copper network."""

    node1: int
    node2: int
    resistance: float  # DC resistance in Ohm
    length: float  # physical length in m


try:
    from .get_distance import (
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
        PAD,
        VIA,
        WIRE,
        ZONE,
        CuLayer,
        DielectricInfo,
        NetworkElement,
        SpiceElement,
    )
except ImportError:
    from get_distance import (
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
        PAD,
        VIA,
        WIRE,
        ZONE,
        CuLayer,
        DielectricInfo,
        NetworkElement,
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


class SpiceUnavailableError(RuntimeError):
    """Raised when libngspice cannot be loaded; the message addresses the user."""


def load_ngspyce() -> ModuleType:
    """Import ngspyce, which loads libngspice on first import."""
    try:
        try:
            from . import ngspyce
        except ImportError:
            import ngspyce
    except OSError as e:
        raise SpiceUnavailableError(
            f"The ngspice library could not be loaded ({e}).\n\n"
            "Install ngspice (macOS: brew install libngspice) or set the "
            "LIBNGSPICE environment variable to the library path."
        ) from e
    return ngspyce


class SpiceError(RuntimeError):
    """Raised when an ngspice simulation produces no usable result."""


# Failures of a single simulation: ngspice and ngspyce raise RuntimeError
# (including SpiceError and SpiceTimeoutError) or ValueError, netlist I/O OSError.
_SPICE_FAILURES = (RuntimeError, ValueError, OSError)


class SpiceTimeoutError(RuntimeError):
    """Raised when ngspice exceeds SPICE_TIMEOUT_S or is still blocked by such a call."""


# Native call that exceeded SPICE_TIMEOUT_S and may still be running.
_timed_out_call: threading.Thread | None = None


def _run_native_spice_call(fn: Any, *args: Any) -> Any:
    """Run a native ngspyce call with a wall-clock timeout.

    ngspice runs in-process via ctypes and a non-convergent circuit can block
    indefinitely. The call runs in a daemon thread so the caller (typically the
    GUI thread) stops waiting after SPICE_TIMEOUT_S. The native call itself
    cannot be killed; until it returns, further calls are rejected because
    ngspice keeps global state and is not reentrant.
    """
    global _timed_out_call
    if _timed_out_call is not None and _timed_out_call.is_alive():
        raise SpiceTimeoutError("ngspice is still blocked by a timed-out call")

    result: list[Any] = []
    error: list[BaseException] = []

    def _target() -> None:
        try:
            result.append(fn(*args))
        except BaseException as e:  # noqa: BLE001 - re-raised in caller thread
            error.append(e)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(SPICE_TIMEOUT_S)
    if thread.is_alive():
        _timed_out_call = thread
        raise SpiceTimeoutError(
            f"ngspice call did not return within {SPICE_TIMEOUT_S:.0f}s "
            "(likely non-convergent or pathological circuit)"
        )
    if error:
        raise error[0]
    return result[0]


def _release_ngspice(ngspyce: ModuleType) -> None:
    """Unload the current circuit and all result plots; failures are only logged."""
    if _timed_out_call is not None and _timed_out_call.is_alive():
        return
    try:
        _run_native_spice_call(ngspyce.cmd, "destroy all")
        _run_native_spice_call(ngspyce.cmd, "remcirc")
    except _SPICE_FAILURES:
        log.warning("[SPICE] Could not release ngspice state", exc_info=True)


def _run_spice(
    ngspyce: ModuleType,
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
        For DC: real resistance in Ohm (float)
        For AC: complex impedance in Ohm (complex)
    Raises:
        SpiceError: rejected netlist, no new result plot, no finite impedance,
            or a non-positive DC resistance
        SpiceTimeoutError: ngspice did not return in time or is still blocked
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

    se_tag = " SE" if far_node is not None else ""
    mode = f"AC{se_tag} @ {ac_freq:.0f} Hz" if ac_freq else "DC"
    log.info(
        "[SPICE] %s: %d elements, conn %s <-> %s", mode, len(elements), conn1, conn2
    )
    log.debug("[SPICE] netlist:\n%s", "\n".join(lines))

    plots_before = set(_run_native_spice_call(ngspyce.plots))
    try:
        if _run_native_spice_call(ngspyce.circ, lines) != 0:
            raise SpiceError("ngspice rejected the netlist")
        if ac_freq:
            _run_native_spice_call(ngspyce.cmd, "run")
        else:
            _run_native_spice_call(ngspyce.dc, "v1", 1, 1, 1)
        new_plots = set(_run_native_spice_call(ngspyce.plots)) - plots_before
        if len(new_plots) != 1:
            raise SpiceError(f"Expected one new result plot, got {sorted(new_plots)}")
        # Copy: the vector points into ngspice memory that _release_ngspice frees.
        i_meas = ngspyce.vector("vmeas#branch", new_plots.pop()).copy()
    finally:
        _release_ngspice(ngspyce)
    log.info("[SPICE] done")

    if len(i_meas) == 0 or i_meas[0] == 0 or not cmath.isfinite(i_meas[0]):
        raise SpiceError(f"No finite port current (I={i_meas})")
    z: float | complex = -1.0 / i_meas[0]
    if not cmath.isfinite(z) or (not ac_freq and z.real <= 0):
        raise SpiceError(f"Implausible port impedance {z}")
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


def RunSimulation(
    resistors: list[Resistor],
    conn1: int,
    conn2: int,
    network_info: list[NetworkElement] | None = None,
    frequencies: list[float] | None = None,
    rload_far: float | None = None,
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
        SpiceUnavailableError: libngspice cannot be loaded
    """
    ngspyce = load_ngspyce()

    # Simulate only elements between the ports: copper not connected to conn1
    # would float, and an element whose terminals share a node carries no current
    # through it (its shunt capacitance is neglected).
    edges = [(r.node1, r.node2, 0.0) for r in resistors]
    edges += [(*e["nodes"], 0.0) for e in network_info or []]
    connected = find_all_reachable_nodes(get_graph_from_edges(edges), conn1)

    def conducts(n1: int, n2: int) -> bool:
        return n1 != n2 and n1 in connected

    def simulate(
        label: str, elements: list[SpiceElement], port: int, ret: int, **kw: Any
    ) -> Impedance | None:
        try:
            return _run_spice(ngspyce, elements, port, ret, **kw)
        except _SPICE_FAILURES:
            log.exception("[SPICE] %s failed", label)
            return None

    gnd = 0

    # --- DC: R-only network ---
    dc_elements: list[SpiceElement] = [
        ("R", r.node1, r.node2, r.resistance)
        for r in resistors
        if conducts(r.node1, r.node2)
    ]
    dc = simulate("DC", dc_elements, conn1, conn2)
    r_dc = None if dc is None else dc.real

    # --- AC: RLC network per frequency ---
    z_ac: dict[float, Impedance | None] = {}
    z_ac_conn1: dict[float, Impedance | None] = {}
    z_ac_conn2: dict[float, Impedance | None] = {}
    for freq in frequencies or []:
        ac: list[SpiceElement] = []
        for idx, elem in enumerate(network_info or [], start=1):
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

        f_str = f"{freq:.0f} Hz"
        z_ac[freq] = simulate(f"AC at {f_str}", ac, conn1, conn2, ac_freq=freq)
        if rload_far is not None:
            for port, far, store in (
                (conn1, conn2, z_ac_conn1),
                (conn2, conn1, z_ac_conn2),
            ):
                store[freq] = simulate(
                    f"Zin({port}) at {f_str}",
                    ac,
                    port,
                    gnd,
                    ac_freq=freq,
                    far_node=far,
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
            resistors: list of Resistor
            network_info: list of element dicts (WIRE/VIA/ZONE) with geometric data
            coordinates: dict {node: (x, y, z)}
            area: dict {layer: area}
            graph: adjacency dict from get_graph_from_edges
    """

    resistors: list[Resistor] = []
    network_info: list[NetworkElement] = []
    coordinates: dict[int, tuple[float, float, float]] = {}
    Area: defaultdict[int, float] = defaultdict(float)

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
            if Layer not in CuStack:
                log.warning("Layer %s not in CuStack, skipping element", Layer)
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

    area = {layer: a for layer, a in Area.items() if a > 0}

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
                    for conn_type, net_type in (
                        ("conn_start", "net_start"),
                        ("conn_end", "net_end"),
                    ):
                        nets = conn_item.get(net_type, {})
                        if uuid in conn_item.get(conn_type, []) and layer in nets:
                            node = nets[layer]
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
) -> tuple[
    float | None,
    dict[float, Impedance | None],
    dict[float, Impedance | None],
    dict[float, Impedance | None],
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
        A failed simulation yields None (see RunSimulation).
    """
    resistors = network["resistors"]
    network_info = network["network_info"]

    # Compute HF parameters for WIRE elements
    for elem in network_info:
        if elem["type"] != WIRE:
            continue
        hf = {}
        for f in frequencies or []:
            try:
                hf[f] = analyze_trace(
                    elem["length"], elem["width"], elem["layer"], CuStack, frequency=f
                )
            except (ValueError, ZeroDivisionError):
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

    resistance_dc, impedance_ac, z_ac_conn1, z_ac_conn2 = RunSimulation(
        resistors, conn1, conn2, network_info, frequencies, rload_far
    )
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
