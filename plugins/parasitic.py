from __future__ import annotations

import logging
import math
import os
import textwrap
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any, NamedTuple

try:
    from .get_distance import DisjointSets, dijkstra, find_all_reachable_nodes
except ImportError:
    from get_distance import DisjointSets, dijkstra, find_all_reachable_nodes
try:
    from .get_parasitic import (
        ZONE_LINK_OHM,
        extract_network,
        find_path,
        simulate_network,
    )
except ImportError:
    from get_parasitic import (
        ZONE_LINK_OHM,
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
    from .pcb_types import WIRE, ZONE, CuLayer, NetworkElement, Point
except ImportError:
    from pcb_types import WIRE, ZONE, CuLayer, NetworkElement, Point

log = logging.getLogger(__name__)

# far-end termination for single-ended Zin measurement
_ZIN_TERMINATION_OHM: float = 50.0


class MeasurementPoint(NamedTuple):
    """A free end of the selection (see selection_ends)."""

    position: Point
    nodes: dict[int, int]  # layer -> node


def selection_ends(data: dict[str, Any]) -> list[MeasurementPoint]:
    """The free ends of the selected elements of `data` (from Connect_Nets).

    A selected track joins the nodes at its two ends; a pad or via, selected or
    not, is one point that covers its nodes on all layers. An end is a point of
    the selection touched by at most one selected track: a selected pad or via
    without a selected track, or the end of a run of tracks. Zones have no
    single connection point and are left out.
    """
    points: DisjointSets[int] = DisjointSets()
    position: dict[int, Point] = {}
    layer_of: dict[int, int] = {}
    tracks: Counter[int] = Counter()  # selected tracks per node
    for d in data.values():
        if d["type"] == ZONE:
            continue
        if d["type"] != WIRE:
            nodes = [d["net_start"][layer] for layer in d["layer"]]
            for node in nodes:
                points.union(node, nodes[0])
            if d["is_selected"]:
                for layer, node in zip(d["layer"], nodes):
                    layer_of[node] = layer
                    position[node] = d["position"]  # wins over the track ends there
        elif d["is_selected"]:
            for layer in d["layer"]:
                for side, pos in (("net_start", d["start"]), ("net_end", d["end"])):
                    node = d[side][layer]
                    tracks[node] += 1
                    layer_of[node] = layer
                    position.setdefault(node, pos)

    nodes_of: defaultdict[int, list[int]] = defaultdict(list)
    for node in layer_of:
        nodes_of[points.find(node)].append(node)
    return [
        MeasurementPoint(position[nodes[0]], {layer_of[n]: n for n in nodes})
        for nodes in nodes_of.values()
        if sum(tracks[n] for n in nodes) <= 1
    ]


# A measurement point and the layer it is measured on
_Probe = tuple[MeasurementPoint, int]


def _farthest_pair(
    graph: dict[int, dict[int, float]],
    ends: list[MeasurementPoint],
    progress: Progress,
) -> tuple[_Probe, _Probe] | None:
    """The two ends with the longest shortest path between them.

    For an end on several layers (pad, via) the layer with the shortest path is
    used. None for fewer than two ends, or if any two ends are not connected:
    the selection then spans separate copper islands and no single path
    stands for it.
    """
    # TODO: prefer the top layer when both ends offer it, as measured with a
    # probe from the top; a track between two vias on B.Cu then includes them.
    distances = []  # per end except the last: {layer: {node: distance}}
    for finished, end in enumerate(ends[:-1]):
        report(progress, Stage.PATH, finished, len(ends) - 1)
        distances.append(
            {layer: dijkstra(graph, node)[0] for layer, node in end.nodes.items()}
        )
    best: tuple[float, _Probe, _Probe] | None = None
    for (i, a), (_, b) in combinations(enumerate(ends), 2):
        distance, layer_a, layer_b = min(
            (distances[i][la].get(nb, math.inf), la, lb)
            for la in a.nodes
            for lb, nb in b.nodes.items()
        )
        if math.isinf(distance):
            return None
        if best is None or distance > best[0]:
            best = (distance, (a, layer_a), (b, layer_b))
    return (best[1], best[2]) if best else None


def _point_text(
    point: MeasurementPoint, layers: Iterable[int], cu_stack: dict[int, CuLayer]
) -> str:
    names = ", ".join(cu_stack[k]["name"] if k in cu_stack else str(k) for k in layers)
    x, y = point.position
    return f"({x * 1e3:.3f}, {y * 1e3:.3f}) mm on {names}"


def _path_notes(path: list[int], network_info: list[NetworkElement]) -> list[str]:
    """Simplifications that shape the result along `path`."""
    on_path = set(path)
    if not any(e["type"] == ZONE and set(e["nodes"]) <= on_path for e in network_info):
        return []
    link = format_si(ZONE_LINK_OHM, "Ω")
    return [
        f"Zones on the path count as nearly ideal conductors: {link} per "
        "connection, and crossing them adds no length."
    ]


def _islands_error(
    ends: list[MeasurementPoint],
    graph: dict[int, dict[int, float]],
    cu_stack: dict[int, CuLayer],
) -> str:
    """Message naming the free ends of each copper island of the selection."""
    islands: list[tuple[set[int], list[MeasurementPoint]]] = []
    for end in ends:
        nodes = set(end.nodes.values())
        island = next((i for i in islands if nodes & i[0]), None)
        if island is None:
            reach = nodes.union(*(find_all_reachable_nodes(graph, n) for n in nodes))
            islands.append((reach, [end]))
        else:
            island[1].append(end)
    lines = [
        "The selection is not connected by copper, so no single path stands "
        f"for it. Its free ends lie on {len(islands)} separate copper islands:"
    ]
    for i, (_, members) in enumerate(islands, 1):
        lines += [f"  Island {i}:"]
        lines += [f"    {_point_text(end, end.nodes, cu_stack)}" for end in members]
    lines.append(
        "Select copper of one island only, or check the board for a copper gap "
        "or a missing via."
    )
    return "\n".join(lines)


def analyze_pcb_parasitic(
    data: dict[str, Any],
    CuStack: dict[int, CuLayer],
    frequencies: list[float] | None = None,
    progress: Progress = no_progress,
) -> dict[str, Any]:
    """Analyze parasitic resistance/impedance across the selected elements.

    Measures between the two free ends of the selection (see selection_ends)
    with the longest shortest path between them.

    Args:
        data: PCB element data from Connect_Nets(), selected ones marked
        CuStack: Copper stackup (see board_model.build_cu_stack)
        frequencies: List of frequencies for AC analysis (Hz)
        progress: reports each analysis stage; see Progress

    Returns:
        dict with: resistance_dc, impedance_ac, distance, short_path_resistance,
                   area, network_info, graph, path, conn1, conn2,
                   measured (texts naming the two measured points),
                   free_ends (number of free ends of the selection),
                   notes (simplifications of the analysis, see Board.notes), error,
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
        "measured": [],
        "free_ends": 0,
        "notes": [],
        "error": None,
        "spice_unavailable": None,
        "skipped": [],
    }

    ends = selection_ends(data)
    if len(ends) < 2:
        result["error"] = (
            "The selection has no two free ends to measure between. Select two "
            "pads or vias, or an open run of tracks; items that touch count as "
            "one point, and a closed loop has no free end."
        )
        return result

    network = extract_network(data, CuStack, progress=progress)
    result["skipped"] = network["skipped"]
    pair = _farthest_pair(network["graph"], ends, progress)
    if pair is None:
        result["error"] = _islands_error(ends, network["graph"], CuStack)
        log.warning("%s", result["error"])
        return result

    conn1, conn2 = (end.nodes[layer] for end, layer in pair)
    result["conn1"], result["conn2"] = conn1, conn2
    result["free_ends"] = len(ends)
    result["measured"] = [_point_text(end, [layer], CuStack) for end, layer in pair]
    Distance, path, short_path_RES = find_path(network, conn1, conn2)
    log.info(
        "Measuring %s to %s (nodes %s/%s, %d free ends), distance=%.3fmm",
        *result["measured"],
        conn1,
        conn2,
        len(ends),
        Distance * 1000,
    )

    result.update(
        {
            "distance": Distance,
            "short_path_resistance": short_path_RES,
            "area": network["area"],
            "network_info": network["network_info"],
            "graph": network["graph"],
            "path": path,
            "notes": _path_notes(path, network["network_info"]),
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


def format_result_message(
    result: dict[str, Any],
    CuStack: dict[int, CuLayer],
    net_ties: Sequence[str] = (),
    nets: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> str:
    """Format analysis result as human-readable message.

    net_ties lists the Net Ties joining `nets`, the nets that were analyzed;
    notes are simplifications made while reading the board (see Board.notes),
    listed with those of the analysis.
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

    notes = [*notes, *result["notes"]]
    if notes:
        lines.append("Simplified in this model (expected, not an error):")
        lines += [
            textwrap.fill(note, 88, initial_indent="  - ", subsequent_indent="    ")
            for note in notes
        ]
        lines.append("")

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

    if result["measured"]:
        point1, point2 = result["measured"]
        lines += [f"Measured from {point1}", f"           to {point2}"]
        if result["free_ends"] > 2:
            lines.append(
                f"           (farthest apart of {result['free_ends']} free ends "
                "along the copper)"
            )
        lines.append("")
    lines.append(f"Shortest distance ≈ {Distance * 1000:.3f} mm")

    if short_path_RES is not None:
        lines.append(f"Shortest path resistance ≈ {short_path_RES * 1000:.3f} mΩ")
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
