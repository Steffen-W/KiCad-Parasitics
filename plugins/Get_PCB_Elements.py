import numpy as np
import pcbnew
from typing import Any

ToMM = pcbnew.ToMM


def SaveDictToFile(dict_name, filename):
    with open(filename, "w") as f:
        f.write("data = {\n")
        for uuid, d in list(dict_name.items()):
            f.write(str(uuid))
            f.write(":")
            f.write(str(d))
            f.write(",\n")
        f.write("}")


# Überprüfe, ob der Punkt sich innerhalb des Polygons befindet
def IsPointInPolygon(point_, polygon_):
    point = np.array(point_)
    polygon = np.array(polygon_)

    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if point[1] > min(p1y, p2y):
            if point[1] <= max(p1y, p2y):
                if point[0] <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersect = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point[0] <= x_intersect:
                            inside = not inside
                    elif p1x == p2x:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def getHash(obj: pcbnew.EDA_ITEM):
    return obj.m_Uuid.Hash()


def getHashList(objlist):
    return [getHash(obj) for obj in objlist]


def getPolygon(obj: pcbnew.PAD):
    try:
        poly_obj = obj.GetEffectivePolygon()
    except Exception:
        poly_obj = obj.GetEffectivePolygon(aLayer=0)  # TODO correct layer
    Polygon = [ToMM(poly_obj.CVertex(p)) for p in range(poly_obj.FullPointCount())]
    return Polygon


def getLayer(obj: pcbnew.BOARD_ITEM, PossibleLayer=set([0, 31])):
    return sorted(set(obj.GetLayerSet().CuStack()) & PossibleLayer)


def getConnections(track: pcbnew.PCB_TRACK, connect: pcbnew.CONNECTIVITY_DATA):
    def getVectorLen(vector):
        return np.sqrt(vector.dot(vector))

    def getDistance(point1, point2):
        return getVectorLen(np.array(point2) - np.array(point1))

    def MoveToObjCenter(wirePos, width, objPos):
        objPos = np.array(objPos)
        wirePos = np.array(wirePos)

        diffVector = objPos - wirePos

        x = np.sign(diffVector[0]) * min([abs(diffVector[0]), width / 2])
        y = np.sign(diffVector[1]) * min([abs(diffVector[1]), width / 2])
        return wirePos + np.array([x, y])

    ConnStart = []
    ConnEnd = []

    Start = ToMM(track.GetStart())
    End = ToMM(track.GetEnd())

    for con in connect.GetConnectedTracks(track):
        if type(con) is pcbnew.PCB_VIA:
            print(ToMM(con.GetWidth()))
            print(ToMM(con.GetPosition()))
        elif type(con) is pcbnew.PCB_TRACK:
            conStart = ToMM(con.GetStart())
            conEnd = ToMM(con.GetEnd())
            if Start == conStart:
                ConnStart.append(getHash(con))
            if Start == conEnd:
                ConnStart.append(getHash(con))
            if End == conStart:
                ConnEnd.append(getHash(con))
            if End == conEnd:
                ConnEnd.append(getHash(con))

            if getHash(con) not in ConnStart + ConnEnd:
                distance = [
                    getDistance(Start, conStart),
                    getDistance(Start, conEnd),
                    getDistance(End, conStart),
                    getDistance(End, conEnd),
                ]
                minDis = min(distance)

                if distance[0] == minDis or distance[1] == minDis:
                    ConnStart.append(getHash(con))
                else:
                    ConnEnd.append(getHash(con))

    for con in connect.GetConnectedPads(track):
        Polygon = getPolygon(con)
        Start_ = MoveToObjCenter(Start, ToMM(track.GetWidth()), ToMM(con.GetPosition()))
        End_ = MoveToObjCenter(End, ToMM(track.GetWidth()), ToMM(con.GetPosition()))

        if IsPointInPolygon(Start_, Polygon):
            ConnStart.append(getHash(con))
        if IsPointInPolygon(End_, Polygon):
            ConnEnd.append(getHash(con))

    return ConnStart, ConnEnd


def Get_PCB_Elements(board: pcbnew.BOARD, connect: pcbnew.CONNECTIVITY_DATA):
    DesignSettings: pcbnew.BOARD_DESIGN_SETTINGS = board.GetDesignSettings()
    PossibleLayer = set(DesignSettings.GetEnabledLayers().CuStack())
    BoardThickness = ToMM(DesignSettings.GetBoardThickness())

    # print(f"BoardThickness {BoardThickness}mm")
    # print("GetTracks", len(board.GetTracks()))
    # print("GetAreaCount", board.GetAreaCount())
    # print("GetPads", len(board.GetPads()))
    # print("AllConnectedItems", len(board.AllConnectedItems()))
    # print("GetFootprints", len(board.GetFootprints()))
    # print("GetDrawings", len(board.GetDrawings()))
    # print("GetAllNetClasses", len(board.GetAllNetClasses()))

    ItemList = {}

    for track in board.GetTracks():
        temp: dict[str, Any] = {"layer": getLayer(track, PossibleLayer)}
        if type(track) is pcbnew.PCB_VIA:
            temp["type"] = "VIA"
            temp["position"] = ToMM(track.GetStart())
            temp["drill"] = ToMM(track.GetDrill())
            temp["width"] = ToMM(track.GetWidth())
            temp["conn_start"] = sorted(
                getHashList(connect.GetConnectedPads(track))
                + getHashList(connect.GetConnectedTracks(track))
            )
            temp["area"] = 0
        elif type(track) is pcbnew.PCB_TRACK:
            temp["type"] = "WIRE"
            temp["start"] = ToMM(track.GetStart())
            temp["end"] = ToMM(track.GetEnd())
            temp["width"] = ToMM(track.GetWidth())
            temp["length"] = ToMM(track.GetLength())
            temp["area"] = temp["width"] * temp["length"]
            if track.GetLength() == 0:
                continue
            temp["layer"] = [track.GetLayer()]
            temp["conn_start"], temp["conn_end"] = getConnections(track, connect)
        elif type(track) is pcbnew.PCB_ARC:
            temp["type"] = "WIRE"
            temp["start"] = ToMM(track.GetStart())
            temp["end"] = ToMM(track.GetEnd())
            temp["radius"] = ToMM(track.GetRadius())
            temp["width"] = ToMM(track.GetWidth())
            temp["length"] = ToMM(track.GetLength())
            temp["area"] = temp["width"] * temp["length"]
            if track.GetLength() == 0:
                continue
            temp["layer"] = [track.GetLayer()]
            temp["conn_start"], temp["conn_end"] = getConnections(track, connect)
        else:
            print("type", type(track), "is not considered!")
            continue

        temp["net_name"] = track.GetNetname()
        temp["net_code"] = track.GetNetCode()
        temp["id"] = getHash(track)
        temp["is_selected"] = track.IsSelected()
        ItemList[temp["id"]] = temp

    for item in board.AllConnectedItems():
        temp = {"layer": getLayer(item, PossibleLayer)}
        if type(item) is pcbnew.PAD:
            temp["type"] = "PAD"
            temp["shape"] = item.GetShape()
            # temp["PadAttr"] = Pad.ShowPadAttr()
            # temp["IsFlipped"] = Pad.IsFlipped()
            temp["position"] = ToMM(item.GetPosition())
            temp["size"] = ToMM(item.GetSize())
            temp["orientation"] = item.GetOrientation().AsDegrees()
            temp["drill_size"] = ToMM(item.GetDrillSize())
            temp["drill"] = temp["drill_size"][0]
            Layers = temp.get("layer", [])

            if len(Layers):
                try:
                    poly_obj = item.GetEffectivePolygon()
                except Exception:
                    poly_obj = item.GetEffectivePolygon(aLayer=Layers[0])

                temp["area"] = ToMM(ToMM(poly_obj.Area()))
            else:
                temp["area"] = 0

            temp["PadName"] = item.GetPadName()
            # temp["FootprintUUID"] = getHash(Pad.GetParent())
            # if Pad.GetParent():
            #     temp["FootprintReference"] = Pad.GetParent().GetReference()

        elif type(item) is pcbnew.ZONE:
            if "teardrop" in item.GetZoneName():  # skip teardrop zones
                continue
            temp["type"] = "ZONE"
            temp["position"] = ToMM(item.GetPosition())
            temp["area"] = ToMM(ToMM(item.GetFilledArea()))
            temp["NumCorners"] = item.GetNumCorners()
            temp["ZoneName"] = item.GetZoneName()
        elif type(item) is pcbnew.PCB_TRACK:
            continue  # already in board.GetTracks()
        elif type(item) is pcbnew.BOARD_CONNECTED_ITEM:
            if item.GetNetCode() == 0:
                continue
            print("type", type(item), "is not considered!")
            continue
        else:
            print("type", type(item), "is not considered!")
            continue

        temp["net_name"] = item.GetNetname()
        temp["net_code"] = item.GetNetCode()
        temp["id"] = getHash(item)
        temp["is_selected"] = item.IsSelected()
        temp["conn_start"] = sorted(
            getHashList(connect.GetConnectedPads(item))
            + getHashList(connect.GetConnectedTracks(item))
        )
        ItemList[temp["id"]] = temp

    for uuid, d in list(ItemList.items()):
        if d["type"] == "ZONE":
            zone_pos = d.get("position", (0, 0))
            for item in d["conn_start"]:
                if item not in ItemList:
                    continue
                item_data = ItemList[item]
                if item_data.get("type") == "WIRE":
                    # Add zone to the wire end that's closer
                    start = item_data.get("start", (0, 0))
                    end = item_data.get("end", (0, 0))
                    dist_start = (start[0] - zone_pos[0]) ** 2 + (start[1] - zone_pos[1]) ** 2
                    dist_end = (end[0] - zone_pos[0]) ** 2 + (end[1] - zone_pos[1]) ** 2
                    conn_key = "conn_start" if dist_start <= dist_end else "conn_end"
                    if uuid not in item_data.get(conn_key, []):
                        item_data.setdefault(conn_key, []).append(uuid)
                else:
                    if uuid not in item_data.get("conn_start", []):
                        item_data.setdefault("conn_start", []).append(uuid)

    return ItemList, BoardThickness
