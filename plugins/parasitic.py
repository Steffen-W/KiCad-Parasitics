from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from typing import Any

try:
    from .connect_nets import NOT_ASSIGNED
except ImportError:
    from connect_nets import NOT_ASSIGNED
try:
    from .get_distance import find_all_reachable_nodes
except ImportError:
    from get_distance import find_all_reachable_nodes
try:
    from .get_parasitic import (
        extract_network,
        find_path,
        simulate_network,
    )
except ImportError:
    from get_parasitic import (
        extract_network,
        find_path,
        simulate_network,
    )
try:
    from .analysis_progress import (
        Progress,
        Stage,
        no_progress,
        report,
    )
    from .spice_worker import SpiceUnavailableError
except ImportError:
    from analysis_progress import (
        Progress,
        Stage,
        no_progress,
        report,
    )
    from spice_worker import SpiceUnavailableError
try:
    from .pcb_types import CuLayer
except ImportError:
    from pcb_types import CuLayer

log = logging.getLogger(__name__)

# far-end termination for single-ended Zin measurement
_ZIN_TERMINATION_OHM: float = 50.0


def _build_path_error(
    element1: dict[str, Any],
    element2: dict[str, Any],
    conn1: int,
    conn2: int,
    graph: dict[int, Any],
    cu_stack: dict[int, CuLayer],
) -> str:
    """Diagnostic message explaining why no path was found between two nodes."""

    def layer_names(elem: dict[str, Any]) -> str:
        return ", ".join(
            cu_stack[layer_id]["name"] if layer_id in cu_stack else str(layer_id)
            for layer_id in elem.get("layer", [])
        )

    def elem_line(label: str, elem: dict[str, Any], conn: int) -> str:
        in_graph = conn in graph
        status = (
            f"in graph ({len(graph.get(conn, {}))} neighbors)"
            if in_graph
            else "NOT in graph"
        )
        return f"  {label}: {elem.get('type', '?')}  net={elem.get('net', '?')!r}  layer={layer_names(elem)}  node={conn}  {status}"

    in1, in2 = conn1 in graph, conn2 in graph
    if not in1 and not in2:
        reason = "Neither element is connected to any trace in the copper network."
    elif not in1:
        reason = "Element 1 has no connected traces in the copper network."
    elif not in2:
        reason = "Element 2 has no connected traces in the copper network."
    else:
        reachable = find_all_reachable_nodes(graph, conn1)
        if conn2 in reachable:
            reason = "Nodes are connected by BFS but path finding failed — check log."
        else:
            reason = (
                f"Both nodes exist but are in separate copper islands "
                f"(El.1 can reach {len(reachable)}/{len(graph)} nodes).\n"
                f"Check for copper gaps or missing vias."
            )

    return "\n".join(
        [
            "No path found between selected elements.",
            elem_line("El.1", element1, conn1),
            elem_line("El.2", element2, conn2),
            f"  Network: {len(graph)} nodes total",
            reason,
        ]
    )


def analyze_pcb_parasitic(
    data: dict[str, Any],
    CuStack: dict[int, CuLayer],
    element1: dict[str, Any],
    element2: dict[str, Any],
    frequencies: list[float] | None = None,
    progress: Progress = no_progress,
) -> dict[str, Any]:
    """Analyze parasitic resistance/impedance between two PCB elements.

    Args:
        data: PCB element data from Connect_Nets()
        CuStack: Copper stackup (see board_model.build_cu_stack)
        element1, element2: Selected element dicts
        frequencies: List of frequencies for AC analysis (Hz)
        progress: reports each analysis stage; see Progress

    Returns:
        dict with: resistance_dc, impedance_ac, distance, short_path_resistance,
                   area, network_info, graph, path, conn1, conn2, error,
                   spice_unavailable (message when ngspice cannot be started),
                   skipped (elements left out of the network, see extract_network)
    Raises:
        CancelledError: `progress` asked to stop
    """
    result: dict[str, Any] = {
        "resistance_dc": None,
        "impedance_ac": {},
        "distance": float("inf"),
        "short_path_resistance": None,
        "area": {},
        "network_info": None,
        "graph": None,
        "path": None,
        "conn1": None,
        "conn2": None,
        "error": None,
        "spice_unavailable": None,
        "skipped": [],
    }

    network = extract_network(data, CuStack, progress=progress)
    result["skipped"] = network["skipped"]
    best_distance, best_conn = _best_connection(network, element1, element2, progress)

    if not best_conn:
        best_conn = (
            element1["net_start"].get(element1["layer"][0], NOT_ASSIGNED),
            element2["net_start"].get(element2["layer"][0], NOT_ASSIGNED),
            element1["layer"][0],
            element2["layer"][0],
        )

    conn1, conn2 = best_conn[0], best_conn[1]
    result["conn1"], result["conn2"] = conn1, conn2
    if NOT_ASSIGNED in (conn1, conn2):
        result["error"] = (
            "Select pads, vias or tracks as measurement points; "
            "zones have no single connection point."
        )
        return result
    if conn1 == conn2:
        result["error"] = (
            "The selected elements touch each other directly, "
            "so there is no copper path between them to analyze."
        )
        return result

    # Get full path info for best connection
    Distance, path, short_path_RES = find_path(network, conn1, conn2)

    log.info(
        "Best path: conn1=%s, conn2=%s, layers=%s/%s, distance=%.3fmm",
        conn1,
        conn2,
        best_conn[2],
        best_conn[3],
        best_distance * 1000,
    )

    if math.isinf(Distance):
        result["error"] = _build_path_error(
            element1, element2, conn1, conn2, network["graph"], CuStack
        )
        log.error("%s", result["error"])
        return result

    result.update(
        {
            "distance": Distance,
            "short_path_resistance": short_path_RES,
            "area": network["area"],
            "network_info": network["network_info"],
            "graph": network["graph"],
            "path": path,
        }
    )

    try:
        Resistance, Z_ac, Z_ac_conn1, Z_ac_conn2 = simulate_network(
            network,
            conn1,
            conn2,
            CuStack,
            frequencies=frequencies,
            rload_far=_ZIN_TERMINATION_OHM,
            progress=progress,
        )
    except SpiceUnavailableError as e:
        log.error("%s", e)
        result["spice_unavailable"] = str(e)
        return result

    result.update(
        {
            "resistance_dc": Resistance,
            "impedance_ac": Z_ac,
            "impedance_ac_conn1": Z_ac_conn1,
            "impedance_ac_conn2": Z_ac_conn2,
            "zin_termination_ohm": _ZIN_TERMINATION_OHM,
        }
    )
    return result


def _best_connection(
    network: dict[str, Any],
    element1: dict[str, Any],
    element2: dict[str, Any],
    progress: Progress,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Shortest path over all layer combinations of the two elements.

    Returns:
        (distance, (conn1, conn2, layer1, layer2)); (inf, None) if none connects
    """
    layer_pairs = [
        (layer1, layer2)
        for layer1 in element1.get("layer", [])
        for layer2 in element2.get("layer", [])
    ]
    best_distance = float("inf")
    best_conn = None
    for finished, (layer1, layer2) in enumerate(layer_pairs):
        report(progress, Stage.PATH, finished, len(layer_pairs))
        conn1 = element1["net_start"].get(layer1, NOT_ASSIGNED)
        conn2 = element2["net_start"].get(layer2, NOT_ASSIGNED)
        if NOT_ASSIGNED in (conn1, conn2):
            continue
        distance, _, _ = find_path(network, conn1, conn2)
        if distance < best_distance:
            best_distance = distance
            best_conn = (conn1, conn2, layer1, layer2)
    return best_distance, best_conn


def format_result_message(
    result: dict[str, Any],
    CuStack: dict[int, CuLayer],
    net_ties: Sequence[str] = (),
    nets: Sequence[str] = (),
) -> str:
    """Format analysis result as human-readable message.

    net_ties lists the Net Ties joining `nets`, the nets that were analyzed.
    """
    lines = []
    if result["skipped"]:
        lines += [
            (
                f"Warning: {len(result['skipped'])} modelling issue(s), so the result "
                "may be incomplete (see debug_parasitic.log)."
            ),
            "",
        ]

    if result["error"]:
        return "\n".join([*lines, str(result["error"])])

    if net_ties:
        lines += ["Net tie:", *(f"  {ref}" for ref in net_ties)]
        lines += ["Bridged nets:", *(f"  {net}" for net in nets), ""]

    Distance = result["distance"]
    short_path_RES = result["short_path_resistance"]
    Resistance = result["resistance_dc"]
    Z_ac = result["impedance_ac"]
    Z_ac_conn1 = result.get("impedance_ac_conn1", {})
    Z_ac_conn2 = result.get("impedance_ac_conn2", {})
    Area = result["area"]

    lines.append(f"Shortest distance ≈ {Distance * 1000:.3f} mm")

    if short_path_RES is not None:
        lines.append(f"Shortest path resistance ≈ {short_path_RES * 1000:.3f} mΩ")
        if short_path_RES == 0:
            lines.append("(Zones treated as ideal conductors)")
    else:
        log_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "debug_parasitic.log"
        )
        lines.append(f"No connection found [DEBUG: {log_file}]")

    # Build impedance table (DC + AC)
    has_valid_dc = Resistance is not None
    has_ac = bool(Z_ac)
    has_zin = bool(Z_ac_conn1) or bool(Z_ac_conn2)
    termination: float | None = result.get("zin_termination_ohm")
    term_str = format_si(termination, "Ω") if termination is not None else "?"

    if has_valid_dc or has_ac:
        lines.append("")
        if has_zin:
            zin1_hdr = f"|Zin1/{term_str}|"
            zin2_hdr = f"|Zin2/{term_str}|"
            lines.append(
                f"{'Freq':^8}  {'|Z|':^12}  {'Re':^12}  {'Im':^12}  {zin1_hdr:^14}  {zin2_hdr:^14}"
            )
            lines.append("─" * (8 + 2 + 12 + 2 + 12 + 2 + 12 + 2 + 14 + 2 + 14))
        else:
            lines.append(f"{'Freq':^8}  {'|Z|':^12}  {'Re':^12}  {'Im':^12}")
            lines.append("─" * (8 + 2 + 12 + 2 + 12 + 2 + 12))

        if has_valid_dc:
            dc_str = format_si(Resistance, "Ω")
            zin_suffix = f"  {term_str:>14}  {term_str:>14}" if has_zin else ""
            lines.append(
                f"{'DC':>8}  {dc_str:>12}  {dc_str:>12}  {'-':>12}{zin_suffix}"
            )
        else:
            lines.append(f"{'DC':>8}  {'[error]':>12}")

        for freq, z in sorted(Z_ac.items()):
            f_str = format_si(freq, "Hz", precision=0)
            if z is None:
                lines.append(f"{f_str:>8}  {'[error]':>12}")
                continue
            z = complex(z)
            row = f"{f_str:>8}  {format_si(abs(z), 'Ω'):>12}  {format_si(z.real, 'Ω'):>12}  {format_si(z.imag, 'Ω'):>12}"
            if has_zin:
                row += "".join(
                    f"  {_format_zin(zin_by_freq, freq):>14}"
                    for zin_by_freq in (Z_ac_conn1, Z_ac_conn2)
                )
            lines.append(row)
    elif result["spice_unavailable"]:
        lines.append("")
        lines.append(result["spice_unavailable"])
    else:
        lines.append("")
        lines.append("ERROR: ngspice simulation failed, see debug_parasitic.log")

    if Area:
        lines.append("")
        lines.append("Copper area (traces only):")
        for layer, area in Area.items():
            info = CuStack[layer]
            lines.append(
                f"  {info['name']}: {area * 1e6:.3f} mm² "
                f"({info['thickness'] * 1e6:.0f} μm, {info['model']} assumption)"
            )

    return "\n".join(lines)


def _format_zin(zin_by_freq: dict[float, Any], freq: float) -> str:
    """|Zin| at freq; "-" when not simulated, "[error]" when the simulation failed."""
    if freq not in zin_by_freq:
        return "-"
    zin = zin_by_freq[freq]
    return "[error]" if zin is None else format_si(abs(zin), "Ω")


_SI_PREFIXES = [
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1, " "),
    (1e-3, "m"),
    (1e-6, "µ"),
    (1e-9, "n"),
    (1e-12, "p"),
]


def format_si(value: float, unit: str, precision: int = 3) -> str:
    """Format value with SI prefix (µ, m, k, M, G, etc.).

    Args:
        value: numeric value in base unit
        unit: base unit string (e.g. "Ω", "Hz", "F", "H")
        precision: decimal places

    Returns:
        Formatted string like "1.234 mΩ" or "56.789 kHz"
    """
    abs_val = abs(value) if value != 0 else 0
    for scale, prefix in _SI_PREFIXES:
        if abs_val >= scale * 0.9999:  # small tolerance for floating point
            scaled = value / scale
            return f"{scaled:.{precision}f} {prefix}{unit}"

    # Fallback for very small values
    return f"{value:.{precision}f} {unit}"
