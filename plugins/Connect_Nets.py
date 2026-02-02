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
    #     data[uuid].setdefault("connStart", [])
    #     data[uuid].setdefault("connEnd", [])

    def getNet(uuid, conn_uuid, layer, pos: tuple = (0, 0)):
        """
        Retrieves the network connection for uuid (the connected element) on the given layer.
        Note: The function is called as getNet(conn, uuid, ...) where:
        - uuid parameter = conn (the connected element we want to get network from)
        - conn_uuid parameter = uuid (the element being processed)
        So we get the network FROM uuid (first parameter), not from conn_uuid.
        """
        if uuid not in data:
            return NotYetConnected
        
        if layer not in data[uuid].get("Layer", []):
            return ErrorConnection  # Error if layer is missing

        temp = NotYetConnected

        # Get network from uuid (the connected element, first parameter)
        if data[uuid]["type"] == "WIRE":
            # For wires, check both netStart and netEnd
            # Determine which end of the wire connects to conn_uuid
            wire_start = data[uuid].get("Start", (0, 0))
            wire_end = data[uuid].get("End", (0, 0))
            conn_uuid_pos = data[conn_uuid].get("Position", data[conn_uuid].get("Start", (0, 0)))
            
            # Check if conn_uuid connects to wire start or end
            if conn_uuid_pos == wire_start or (conn_uuid in data[uuid].get("connStart", [])):
                temp = data[uuid]["netStart"].get(layer, NotYetConnected)
            elif conn_uuid_pos == wire_end or (conn_uuid in data[uuid].get("connEnd", [])):
                temp = data[uuid]["netEnd"].get(layer, NotYetConnected)
            else:
                # Try both
                temp = data[uuid]["netStart"].get(layer, NotYetConnected)
                if temp == NotYetConnected:
                    temp = data[uuid]["netEnd"].get(layer, NotYetConnected)
        else:
            # For vias/pads, get netStart on the layer
            temp = data[uuid]["netStart"].get(layer, NotYetConnected)
            # If netStart not set, try netEnd
            if temp == NotYetConnected:
                temp = data[uuid]["netEnd"].get(layer, NotYetConnected)

        return temp

    def setNet(target_uuid, source_uuid, layer, newNet, pos: tuple = (0, 0)):
        """
        Sets a network connection for a target UUID based on its connection to a source UUID.
        This is bidirectional - it sets the network on the target based on the source.
        
        Args:
            target_uuid: The UUID to set the network on (the connected element)
            source_uuid: The UUID that has the network (the element being processed)
            layer: The layer to set the network on
            newNet: The network number to assign
            pos: Position hint (for wire-to-wire connections)
        """
        if target_uuid not in data:
            return ErrorConnection
        
        if layer not in data[target_uuid].get("Layer", []):
            return ErrorConnection  # Error if layer is missing

        target_type = data[target_uuid].get("type")
        source_type = data[source_uuid].get("type")
        
        if target_type == "WIRE" and source_type == "WIRE":
            # Wire-to-wire connection: set based on position
            if pos == data[target_uuid].get("Start", (0, 0)):
                data[target_uuid]["netStart"][layer] = newNet
            if pos == data[target_uuid].get("End", (0, 0)):
                data[target_uuid]["netEnd"][layer] = newNet
        elif target_type == "WIRE":
            # Wire connecting to via/pad: determine which end connects
            wire_start = data[target_uuid].get("Start", (0, 0))
            wire_end = data[target_uuid].get("End", (0, 0))
            source_pos = data[source_uuid].get("Position", (0, 0))
            
            # Check which end of the wire connects to the source
            if source_uuid in data[target_uuid].get("connStart", []):
                # Connection is at wire start
                if wire_start == source_pos or wire_start == data[source_uuid].get("Start", (0, 0)):
                    data[target_uuid]["netStart"][layer] = newNet
                else:
                    # Default to start if in connStart list
                    data[target_uuid]["netStart"][layer] = newNet
            elif source_uuid in data[target_uuid].get("connEnd", []):
                # Connection is at wire end
                if wire_end == source_pos or wire_end == data[source_uuid].get("Start", (0, 0)):
                    data[target_uuid]["netEnd"][layer] = newNet
                else:
                    # Default to end if in connEnd list
                    data[target_uuid]["netEnd"][layer] = newNet
        else:
            # Via/pad connecting to wire: set netStart on the via/pad
            # The source (wire) already has the network, propagate it to target (via/pad)
            if source_uuid in data[target_uuid].get("connStart", []):
                data[target_uuid]["netStart"][layer] = newNet
            if source_uuid in data[target_uuid].get("connEnd", []):
                # Also set netEnd if not already set
                if data[target_uuid]["netStart"].get(layer, NotYetConnected) == NotYetConnected:
                    data[target_uuid]["netEnd"][layer] = newNet

        return OK

    # Connecting networks
    nodeCounter = 0

    # Process in multiple passes to handle dependencies
    # This ensures that when a wire connects to a via/pad, the via/pad's network
    # is propagated to the wire correctly
    max_passes = 10  # Prevent infinite loops
    for pass_num in range(max_passes):
        changed = False
        
        # First pass: Start network connections
        for uuid, d in data.items():
            for layer in data[uuid]["Layer"]:
                if d["netStart"].get(layer, NotYetConnected) > NotYetConnected:
                    continue

                pos = d.get("Start", (0, 0))
                tempNet = NotYetConnected

                for conn in d.get("connStart", []):
                    if conn in data:
                        tmp = getNet(conn, uuid, layer, pos)
                        if tmp > NotYetConnected:
                            tempNet = tmp
                            break  # Use first valid connection

                if tempNet == NotYetConnected:
                    nodeCounter += 1
                    tempNet = nodeCounter
                    changed = True

                # Set network on all connections (bidirectional)
                for conn in d.get("connStart", []):
                    if conn in data:
                        setNet(conn, uuid, layer, tempNet, pos)

                data[uuid]["netStart"][layer] = tempNet

        # Second pass: End network connections
        for uuid, d in data.items():
            for layer in data[uuid]["Layer"]:
                if d["netEnd"].get(layer, NotYetConnected) > NotYetConnected:
                    continue

                pos = d.get("End", (0, 0))
                tempNet = NotYetConnected
                
                for conn in d.get("connEnd", []):
                    if conn in data:
                        tmp = getNet(conn, uuid, layer, pos)
                        if tmp > NotYetConnected:
                            tempNet = tmp
                            break  # Use first valid connection
                
                if tempNet == NotYetConnected:
                    # Try to use netStart if available (for wires that connect at both ends)
                    tempNet = d["netStart"].get(layer, NotYetConnected)
                    if tempNet == NotYetConnected:
                        nodeCounter += 1
                        tempNet = nodeCounter
                        changed = True

                # Set network on all connections (bidirectional)
                for conn in d.get("connEnd", []):
                    if conn in data:
                        setNet(conn, uuid, layer, tempNet, pos)
                
                data[uuid]["netEnd"][layer] = tempNet
        
        # If nothing changed, we're done
        if not changed:
            break

    return data
