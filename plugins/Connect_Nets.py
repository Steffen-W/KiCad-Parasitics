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
        Retrieves the network connection for a given UUID and layer.
        """
        if layer not in data[uuid].get("Layer", []):
            return ErrorConnection  # Error if layer is missing

        temp = NotYetConnected

        if data[uuid]["type"] == "WIRE" and data[conn_uuid]["type"] == "WIRE":
            if pos == data[uuid].get("Start", (0, 0)):
                temp = data[uuid]["netStart"].get(layer, NotYetConnected)
            if temp > NotYetConnected:
                return temp
            if pos == data[uuid].get("End", (0, 0)):
                temp = data[uuid]["netEnd"].get(layer, NotYetConnected)
        else:
            if conn_uuid in data[uuid].get("connStart", []):
                temp = data[uuid]["netStart"].get(layer, NotYetConnected)
            if temp > NotYetConnected:
                return temp
            if conn_uuid in data[uuid].get("connEnd", []):
                return data[uuid]["netEnd"].get(layer, NotYetConnected)

        return temp

    def setNet(uuid, conn_uuid, layer, newNet, pos: tuple = (0, 0)):
        """
        Sets a network connection for a given UUID.
        """
        if layer not in data[uuid].get("Layer", []):
            return ErrorConnection  # Error if layer is missing

        if data[uuid]["type"] == "WIRE" and data[conn_uuid]["type"] == "WIRE":
            if pos == data[uuid].get("Start", (0, 0)):
                data[uuid]["netStart"][layer] = newNet
            if pos == data[uuid].get("End", (0, 0)):
                data[uuid]["netEnd"][layer] = newNet
        else:
            if conn_uuid in data[uuid].get("connStart", []):
                data[uuid]["netStart"][layer] = newNet
            if conn_uuid in data[uuid].get("connEnd", []):
                data[uuid]["netEnd"][layer] = newNet

        return OK

    # Connecting networks
    nodeCounter = 0

    # First pass: Start network connections
    for uuid, d in data.items():
        for layer in data[uuid]["Layer"]:
            if d["netStart"].get(layer, NotYetConnected) > NotYetConnected:
                continue

            pos = d.get("Start", (0, 0))
            tempNet = NotYetConnected

            for conn in d.get("connStart", []):
                tmp = getNet(conn, uuid, layer, pos)
                if tmp > NotYetConnected:
                    tempNet = tmp

            if tempNet == NotYetConnected:
                nodeCounter += 1
                tempNet = nodeCounter

            for conn in d.get("connStart", []):
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
                tmp = getNet(conn, uuid, layer, pos)
                if tmp > NotYetConnected:
                    tempNet = tmp
            if tempNet == NotYetConnected:
                nodeCounter += 1
                tempNet = nodeCounter

            for conn in d.get("connEnd", []):
                setNet(conn, uuid, layer, tempNet, pos)
            data[uuid]["netEnd"][layer] = tempNet

    return data
