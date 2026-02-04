from collections import defaultdict

# Return values for error handling
OK = 0
NotYetConnected = 0
ErrorConnection = -1


def Connect_Nets(data):
    """
    This function connects networks in a KiCad-like data format.
    """

    # Initialization: Ensure missing keys exist to prevent KeyError
    for uuid, d in data.items():
        data[uuid].setdefault("Layer", [])
        data[uuid].setdefault("netStart", defaultdict(lambda: NotYetConnected))
        data[uuid].setdefault("netEnd", defaultdict(lambda: NotYetConnected))

    def getNet(conn, uuid, layer, pos: tuple = (0, 0)):
        """
        Retrieves the network ID from a connected element.

        Args:
            conn: The connected element to get the network FROM
            uuid: The element being processed (used for position matching)
            layer: The layer to get the network for
            pos: Position hint for wire-to-wire connections

        Returns:
            Network ID if found, NotYetConnected if not assigned yet,
            ErrorConnection if layer mismatch or connection data is corrupt
        """
        if conn not in data:
            return NotYetConnected

        if layer not in data[conn].get("Layer", []):
            return ErrorConnection

        conn_data = data[conn]

        if conn_data["type"] == "ZONE":  # handled in Get_Parasitic
            return NotYetConnected

        if conn_data["type"] == "WIRE":
            # For wires: check which end connects to us based on connection list
            if uuid in conn_data.get("connStart", []):
                return conn_data["netStart"].get(layer, NotYetConnected)
            elif uuid in conn_data.get("connEnd", []):
                return conn_data["netEnd"].get(layer, NotYetConnected)
            # Fallback: try to match by position (for incomplete connection data)
            if pos == conn_data.get("Start", (0, 0)):
                return conn_data["netStart"].get(layer, NotYetConnected)
            if pos == conn_data.get("End", (0, 0)):
                return conn_data["netEnd"].get(layer, NotYetConnected)
            # No match found - connection data is corrupt/incomplete
            return ErrorConnection
        else:
            # For vias/pads: use netStart (they have single connection point per layer)
            return conn_data["netStart"].get(layer, NotYetConnected)

    def setNet(conn, uuid, layer, newNet, pos: tuple = (0, 0)):
        """
        Sets the network ID on a connected element.

        Args:
            conn: The connected element to set the network ON
            uuid: The element being processed
            layer: The layer to set the network for
            newNet: The network ID to assign
            pos: Position hint for wire-to-wire connections

        Returns:
            OK on success, ErrorConnection if layer mismatch or connection data is corrupt
        """
        if conn not in data:
            return ErrorConnection

        if layer not in data[conn].get("Layer", []):
            return ErrorConnection

        conn_data = data[conn]

        if conn_data["type"] == "ZONE":  # handled in Get_Parasitic
            return OK

        if conn_data["type"] == "WIRE":
            # For wires: set the appropriate end based on connection list
            if uuid in conn_data.get("connStart", []):
                conn_data["netStart"][layer] = newNet
            elif uuid in conn_data.get("connEnd", []):
                conn_data["netEnd"][layer] = newNet
            elif pos == conn_data.get("Start", (0, 0)):
                # Fallback: use position matching (for incomplete connection data)
                conn_data["netStart"][layer] = newNet
            elif pos == conn_data.get("End", (0, 0)):
                conn_data["netEnd"][layer] = newNet
            else:
                # No match found - connection data is corrupt/incomplete
                return ErrorConnection
        else:
            # For vias/pads: set netStart
            conn_data["netStart"][layer] = newNet

        return OK

    # Connecting networks
    nodeCounter = 0

    # Multiple passes to handle dependency ordering
    # This ensures proper network propagation even when processing order matters
    max_passes = 10
    for pass_num in range(max_passes):
        changed = False

        # Process netStart connections
        for uuid, d in data.items():
            if d.get("type") == "ZONE":  # handled in Get_Parasitic
                continue

            for layer in d["Layer"]:
                if d["netStart"].get(layer, NotYetConnected) > NotYetConnected:
                    continue

                pos = d.get("Start", d.get("Position", (0, 0)))
                tempNet = NotYetConnected

                # Try to get network from connected elements
                for conn in d.get("connStart", []):
                    if conn in data:
                        tmp = getNet(conn, uuid, layer, pos)
                        if tmp > NotYetConnected:
                            tempNet = tmp
                            break

                # If no network found, create a new one
                if tempNet == NotYetConnected:
                    nodeCounter += 1
                    tempNet = nodeCounter
                    changed = True

                # Propagate network to all connections
                for conn in d.get("connStart", []):
                    if conn in data:
                        setNet(conn, uuid, layer, tempNet, pos)

                d["netStart"][layer] = tempNet

        # Process netEnd connections
        for uuid, d in data.items():
            for layer in d["Layer"]:
                if d["netEnd"].get(layer, NotYetConnected) > NotYetConnected:
                    continue

                pos = d.get("End", d.get("Position", (0, 0)))
                tempNet = NotYetConnected

                # Try to get network from connected elements
                for conn in d.get("connEnd", []):
                    if conn in data:
                        tmp = getNet(conn, uuid, layer, pos)
                        if tmp > NotYetConnected:
                            tempNet = tmp
                            break

                # If no network from connections, try to use netStart (but NOT for wires)
                # Wires need separate nodes for each end to calculate resistance
                if tempNet == NotYetConnected and d.get("type") != "WIRE":
                    tempNet = d["netStart"].get(layer, NotYetConnected)

                # If still no network, create a new one
                if tempNet == NotYetConnected:
                    nodeCounter += 1
                    tempNet = nodeCounter
                    changed = True

                # Propagate network to all connections
                for conn in d.get("connEnd", []):
                    if conn in data:
                        setNet(conn, uuid, layer, tempNet, pos)

                d["netEnd"][layer] = tempNet

        # Stop if nothing changed (converged)
        if not changed:
            break

    return data
