from collections import defaultdict
from typing import Any

try:
    from .pcb_types import WIRE, ZONE
except ImportError:
    from pcb_types import WIRE, ZONE

# Network assignment states
# Valid net IDs are > 0, so 0 means "not yet assigned"
NOT_ASSIGNED = 0
ERROR = -1


def Connect_Nets(data: dict[Any, dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """
    This function connects networks in a KiCad-like data format.
    """

    # Initialization: Ensure missing keys exist to prevent KeyError
    for uuid, d in data.items():
        data[uuid].setdefault("layer", [])
        data[uuid].setdefault("net_start", defaultdict(lambda: NOT_ASSIGNED))
        data[uuid].setdefault("net_end", defaultdict(lambda: NOT_ASSIGNED))

    def getNet(conn: Any, uuid: Any, layer: int, pos: tuple = (0, 0)) -> int:
        """
        Retrieves the network ID from a connected element.

        Args:
            conn: The connected element to get the network FROM
            uuid: The element being processed (used for position matching)
            layer: The layer to get the network for
            pos: Position hint for wire-to-wire connections

        Returns:
            Network ID (>0) if found, NOT_ASSIGNED (0) if not assigned yet,
            ERROR (-1) if layer mismatch or connection data is corrupt
        """
        if conn not in data:
            return NOT_ASSIGNED

        if layer not in data[conn].get("layer", []):
            return ERROR

        conn_data = data[conn]

        if conn_data["type"] == ZONE:  # handled in Get_Parasitic
            return NOT_ASSIGNED

        if conn_data["type"] == WIRE:
            # For wires: check which end connects to us based on connection list
            if uuid in conn_data.get("conn_start", []):
                return conn_data["net_start"].get(layer, NOT_ASSIGNED)
            elif uuid in conn_data.get("conn_end", []):
                return conn_data["net_end"].get(layer, NOT_ASSIGNED)
            # Fallback: try to match by position (for incomplete connection data)
            if pos == conn_data.get("start", (0, 0)):
                return conn_data["net_start"].get(layer, NOT_ASSIGNED)
            if pos == conn_data.get("end", (0, 0)):
                return conn_data["net_end"].get(layer, NOT_ASSIGNED)
            # No match found - connection data is corrupt/incomplete
            return ERROR
        else:
            # For vias/pads: use netStart (they have single connection point per layer)
            return conn_data["net_start"].get(layer, NOT_ASSIGNED)

    def setNet(
        conn: Any, uuid: Any, layer: int, newNet: int, pos: tuple = (0, 0)
    ) -> None:
        """
        Sets the network ID on a connected element.

        Args:
            conn: The connected element to set the network ON
            uuid: The element being processed
            layer: The layer to set the network for
            newNet: The network ID to assign
            pos: Position hint for wire-to-wire connections
        """
        if conn not in data:
            return
        if layer not in data[conn].get("layer", []):
            return

        conn_data = data[conn]

        if conn_data["type"] == ZONE:  # handled in Get_Parasitic
            return

        if conn_data["type"] == WIRE:
            # For wires: set the appropriate end based on connection list
            if uuid in conn_data.get("conn_start", []):
                conn_data["net_start"][layer] = newNet
            elif uuid in conn_data.get("conn_end", []):
                conn_data["net_end"][layer] = newNet
            elif pos == conn_data.get("start", (0, 0)):
                # Fallback: use position matching (for incomplete connection data)
                conn_data["net_start"][layer] = newNet
            elif pos == conn_data.get("end", (0, 0)):
                conn_data["net_end"][layer] = newNet
            # else: no match - silently ignore (connection data incomplete)
        else:
            # For vias/pads: set netStart
            conn_data["net_start"][layer] = newNet

    # Connecting networks
    nodeCounter = 0

    # Multiple passes to handle dependency ordering.
    # In the worst case a linear chain of N elements needs N passes;
    # real PCB nets are never deeper than a few dozen elements, so 10 passes
    # is a safe upper bound while avoiding an infinite loop on corrupt data.
    _MAX_PASSES = 10
    for _ in range(_MAX_PASSES):
        changed = False

        # Process netStart connections
        for uuid, d in data.items():
            if d.get("type") == ZONE:  # handled in Get_Parasitic
                continue

            for layer in d["layer"]:
                if d["net_start"].get(layer, NOT_ASSIGNED) > NOT_ASSIGNED:
                    continue

                pos = d.get("start", d.get("position", (0, 0)))
                tempNet = NOT_ASSIGNED

                # Try to get network from connected elements
                for conn in d.get("conn_start", []):
                    if conn in data:
                        tmp = getNet(conn, uuid, layer, pos)
                        if tmp > NOT_ASSIGNED:
                            tempNet = tmp
                            break

                # If no network found, create a new one
                if tempNet == NOT_ASSIGNED:
                    nodeCounter += 1
                    tempNet = nodeCounter
                    changed = True

                # Propagate network to all connections
                for conn in d.get("conn_start", []):
                    if conn in data:
                        setNet(conn, uuid, layer, tempNet, pos)

                d["net_start"][layer] = tempNet

        # Process netEnd connections
        for uuid, d in data.items():
            for layer in d["layer"]:
                if d["net_end"].get(layer, NOT_ASSIGNED) > NOT_ASSIGNED:
                    continue

                pos = d.get("end", d.get("position", (0, 0)))
                tempNet = NOT_ASSIGNED

                # Try to get network from connected elements
                for conn in d.get("conn_end", []):
                    if conn in data:
                        tmp = getNet(conn, uuid, layer, pos)
                        if tmp > NOT_ASSIGNED:
                            tempNet = tmp
                            break

                # If no network from connections, try to use netStart (but NOT for wires)
                # Wires need separate nodes for each end to calculate resistance
                if tempNet == NOT_ASSIGNED and d.get("type") != WIRE:
                    tempNet = d["net_start"].get(layer, NOT_ASSIGNED)

                # If still no network, create a new one
                if tempNet == NOT_ASSIGNED:
                    nodeCounter += 1
                    tempNet = nodeCounter
                    changed = True

                # Propagate network to all connections
                for conn in d.get("conn_end", []):
                    if conn in data:
                        setNet(conn, uuid, layer, tempNet, pos)

                d["net_end"][layer] = tempNet

        # Stop if nothing changed (converged)
        if not changed:
            break

    return data
