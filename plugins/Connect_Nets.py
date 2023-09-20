import numpy as np


OK = 0
NotYetConnected = 0
ErrorConnection = -1


def Connect_Nets(data):
    for uuid, d in list(data.items()):
        layer = {x: NotYetConnected for x in d["Layer"]}  # connection with start

        if "connStart" in d:
            data[uuid]["netStart"] = dict(layer)
        if "connEnd" in d:
            data[uuid]["netEnd"] = dict(layer)

    def getNet(uuid, conn_uuid, layer, pos: (0, 0)):
        if layer not in data[uuid]["Layer"]:
            return ErrorConnection

        temp = NotYetConnected

        if data[uuid]["type"] == "WIRE" and data[conn_uuid]["type"] == "WIRE":
            if pos == data[uuid]["Start"]:
                temp = data[uuid]["netStart"][layer]
            if temp > NotYetConnected:
                return temp
            if pos == data[uuid]["End"]:
                temp = data[uuid]["netEnd"][layer]
        else:
            if "netStart" in data[uuid] and conn_uuid in data[uuid]["connStart"]:
                temp = data[uuid]["netStart"][layer]
            if temp > NotYetConnected:
                return temp
            if "netEnd" in data[uuid] and conn_uuid in data[uuid]["connEnd"]:
                return data[uuid]["netEnd"][layer]
        return temp

    def setNet(uuid, conn_uuid, layer, newNet: NotYetConnected, pos: (0, 0)):
        if layer not in data[uuid]["Layer"]:
            return ErrorConnection

        if data[uuid]["type"] == "WIRE" and data[conn_uuid]["type"] == "WIRE":
            if pos == data[uuid]["Start"]:
                data[uuid]["netStart"][layer] = newNet
            if pos == data[uuid]["End"]:
                data[uuid]["netEnd"][layer] = newNet
        else:
            if "netStart" in data[uuid] and conn_uuid in data[uuid]["connStart"]:
                data[uuid]["netStart"][layer] = newNet

            if "netEnd" in data[uuid] and conn_uuid in data[uuid]["connEnd"]:
                data[uuid]["netEnd"][layer] = newNet
        return OK

    nodeCounter = 0
    for uuid, d in list(data.items()):
        for layer in data[uuid]["Layer"]:
            if "netStart" not in data[uuid]:
                continue
            if d["netStart"][layer] > NotYetConnected:
                continue

            if "Start" in d:
                pos = d["Start"]
            else:
                pos = (0, 0)

            tempNet = NotYetConnected
            for conn in d["connStart"]:
                tmp = getNet(conn, uuid, layer, pos)
                if tmp > NotYetConnected:
                    tempNet = tmp
                    continue
            if tempNet == NotYetConnected:
                nodeCounter += 1
                tempNet = nodeCounter

            for conn in d["connStart"]:
                setNet(conn, uuid, layer, newNet=tempNet, pos=pos)
            data[uuid]["netStart"][layer] = tempNet

    for uuid, d in list(data.items()):
        for layer in data[uuid]["Layer"]:
            if "netEnd" not in data[uuid]:
                continue
            if d["netEnd"][layer] > NotYetConnected:
                continue

            if "End" in d:
                pos = d["End"]
            else:
                pos = (0, 0)

            tempNet = NotYetConnected
            for conn in d["connEnd"]:
                tmp = getNet(conn, uuid, layer, d["End"])
                if tmp > 0:
                    tempNet = tmp
            if tempNet == NotYetConnected:
                nodeCounter += 1
                tempNet = nodeCounter

            for conn in d["connEnd"]:
                setNet(conn, uuid, layer, newNet=tempNet, pos=d["End"])
            data[uuid]["netEnd"][layer] = tempNet

    return data
