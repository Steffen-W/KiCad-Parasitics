import numpy as np
import pcbnew


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
        p1x, p1y = p2x, p2y

    return inside


def getHash(obj: pcbnew.EDA_ITEM):
    return obj.m_Uuid.Hash()


def getHashList(objlist):
    return [getHash(obj) for obj in objlist]


def getPolygon(obj: pcbnew.PAD):
    try:
        poly_obj = obj.GetEffectivePolygon()
    except:
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
        temp = {"Layer": getLayer(track, PossibleLayer)}
        if type(track) is pcbnew.PCB_VIA:
            temp["type"] = "VIA"
            temp["Position"] = ToMM(track.GetStart())
            temp["Drill"] = ToMM(track.GetDrill())
            temp["Width"] = ToMM(track.GetWidth())
            temp["connStart"] = sorted(
                getHashList(connect.GetConnectedPads(track))
                + getHashList(connect.GetConnectedTracks(track))
            )
            temp["Area"] = 0
        elif type(track) is pcbnew.PCB_TRACK:
            temp["type"] = "WIRE"
            temp["Start"] = ToMM(track.GetStart())
            temp["End"] = ToMM(track.GetEnd())
            temp["Width"] = ToMM(track.GetWidth())
            temp["Length"] = ToMM(track.GetLength())
            temp["Area"] = temp["Width"] * temp["Length"]
            if track.GetLength() == 0:
                continue
            temp["Layer"] = [track.GetLayer()]
            temp["connStart"], temp["connEnd"] = getConnections(track, connect)
        elif type(track) is pcbnew.PCB_ARC:
            temp["type"] = "WIRE"
            temp["Start"] = ToMM(track.GetStart())
            temp["End"] = ToMM(track.GetEnd())
            temp["Radius"] = ToMM(track.GetRadius())
            temp["Width"] = ToMM(track.GetWidth())
            temp["Length"] = ToMM(track.GetLength())
            temp["Area"] = temp["Width"] * temp["Length"]
            if track.GetLength() == 0:
                continue
            temp["Layer"] = [track.GetLayer()]
            temp["connStart"], temp["connEnd"] = getConnections(track, connect)
        else:
            print("type", type(track), "is not considered!")
            continue

        temp["Netname"] = track.GetNetname()
        temp["NetCode"] = track.GetNetCode()
        temp["id"] = getHash(track)
        temp["IsSelected"] = track.IsSelected()
        ItemList[temp["id"]] = temp

    for item in board.AllConnectedItems():
        temp = {"Layer": getLayer(item, PossibleLayer)}
        if type(item) is pcbnew.PAD:
            temp["type"] = "PAD"
            temp["Shape"] = item.GetShape()
            # temp["PadAttr"] = Pad.ShowPadAttr()
            # temp["IsFlipped"] = Pad.IsFlipped()
            temp["Position"] = ToMM(item.GetPosition())
            temp["Size"] = ToMM(item.GetSize())
            temp["Orientation"] = item.GetOrientation().AsDegrees()
            temp["DrillSize"] = ToMM(item.GetDrillSize())
            temp["Drill"] = temp["DrillSize"][0]
            Layers = temp.get("Layer", [])

            if len(Layers):
                try:
                    poly_obj = item.GetEffectivePolygon()
                except:
                    poly_obj = item.GetEffectivePolygon(aLayer=Layers[0])

                temp["Area"] = ToMM(ToMM(poly_obj.Area()))
            else:
                temp["Area"] = 0

            temp["PadName"] = item.GetPadName()
            # temp["FootprintUUID"] = getHash(Pad.GetParent())
            # if Pad.GetParent():
            #     temp["FootprintReference"] = Pad.GetParent().GetReference()

        elif type(item) is pcbnew.ZONE:
            # pcbnew.ZONE().GetZoneName
            if "teardrop" in item.GetZoneName():
                continue
            temp["type"] = "ZONE"
            temp["Position"] = ToMM(item.GetPosition())
            temp["Area"] = ToMM(ToMM(item.GetFilledArea()))
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

        temp["Netname"] = item.GetNetname()
        temp["NetCode"] = item.GetNetCode()
        temp["id"] = getHash(item)
        temp["IsSelected"] = item.IsSelected()
        temp["connStart"] = sorted(
            getHashList(connect.GetConnectedPads(item))
            + getHashList(connect.GetConnectedTracks(item))
        )
        ItemList[temp["id"]] = temp

    for uuid, d in list(ItemList.items()):  # TODO: WIRES still need to be considered
        if d["type"] == "ZONE":
            for item in d["connStart"]:
                if not "connEND" in ItemList[item]:
                    ItemList[item]["connStart"].append(uuid)

    return ItemList
