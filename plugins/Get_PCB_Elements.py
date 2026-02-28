import logging
import numpy as np
import pcbnew
from typing import Any, Iterable, overload, Sequence, Union

try:
    from .pcb_types import WIRE, VIA, PAD, ZONE
except ImportError:
    from pcb_types import WIRE, VIA, PAD, ZONE

log = logging.getLogger(__name__)


@overload
def ToM(value: tuple) -> tuple[float, ...]: ...
@overload
def ToM(value: int | float) -> float: ...


def ToM(value: Any) -> Any:
    """Convert KiCad internal units to meters (SI base unit)."""
    mm = pcbnew.ToMM(value)
    if isinstance(mm, tuple):
        return tuple(v / 1000 for v in mm)
    return mm / 1000


def SaveDictToFile(dict_name: dict, filename: str) -> None:
    with open(filename, "w") as f:
        f.write("data = {\n")
        for uuid, d in list(dict_name.items()):
            f.write(str(uuid))
            f.write(":")
            f.write(str(d))
            f.write(",\n")
        f.write("}")


def IsPointInPolygon(
    point_: Union[Sequence[float], np.ndarray], polygon_: Sequence[Sequence[float]]
) -> bool:
    """Ray casting algorithm to check if point is inside polygon."""
    x, y = point_[0], point_[1]
    n = len(polygon_)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon_[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_intersect:
                            inside = not inside
                    elif p1x == p2x:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def getHash(obj: pcbnew.EDA_ITEM) -> int:
    return obj.m_Uuid.Hash()


def getHashList(objlist: Iterable[Any]) -> list[int]:
    return [getHash(obj) for obj in objlist]


def getPolygon(obj: pcbnew.PAD) -> list[tuple[float, ...]]:
    try:
        poly_obj = obj.GetEffectivePolygon()
    except Exception:
        poly_obj = obj.GetEffectivePolygon(aLayer=0)  # TODO correct layer
    Polygon = [ToM(poly_obj.CVertex(p)) for p in range(poly_obj.FullPointCount())]
    return Polygon


def getLayer(obj: pcbnew.BOARD_ITEM, PossibleLayer: set[int] = {0, 31}) -> list[int]:
    return sorted(set(obj.GetLayerSet().CuStack()) & PossibleLayer)


def getConnections(
    track: pcbnew.PCB_TRACK, connect: pcbnew.CONNECTIVITY_DATA
) -> tuple[list[int], list[int]]:
    def getVectorLen(vector: np.ndarray) -> float:
        return float(np.sqrt(vector.dot(vector)))

    def getDistance(point1: np.ndarray, point2: np.ndarray) -> float:
        return getVectorLen(np.array(point2) - np.array(point1))

    def MoveToObjCenter(
        wirePos: np.ndarray, width: float, objPos: np.ndarray
    ) -> np.ndarray:
        objPos = np.array(objPos)
        wirePos = np.array(wirePos)

        diffVector = objPos - wirePos

        x = np.sign(diffVector[0]) * min([abs(diffVector[0]), width / 2])
        y = np.sign(diffVector[1]) * min([abs(diffVector[1]), width / 2])
        return wirePos + np.array([x, y])

    ConnStart = []
    ConnEnd = []

    Start = ToM(track.GetStart())
    End = ToM(track.GetEnd())

    for con in connect.GetConnectedTracks(track):
        if type(con) is pcbnew.PCB_VIA:
            pass  # VIAs handled separately
        elif type(con) is pcbnew.PCB_TRACK:
            conStart = ToM(con.GetStart())
            conEnd = ToM(con.GetEnd())
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
        Start_ = MoveToObjCenter(Start, ToM(track.GetWidth()), ToM(con.GetPosition()))
        End_ = MoveToObjCenter(End, ToM(track.GetWidth()), ToM(con.GetPosition()))

        if IsPointInPolygon(Start_, Polygon):
            ConnStart.append(getHash(con))
        if IsPointInPolygon(End_, Polygon):
            ConnEnd.append(getHash(con))

    return ConnStart, ConnEnd


def Get_PCB_Elements(
    board: pcbnew.BOARD, connect: pcbnew.CONNECTIVITY_DATA
) -> tuple[dict[int, dict[str, Any]], float]:
    DesignSettings: pcbnew.BOARD_DESIGN_SETTINGS = board.GetDesignSettings()
    PossibleLayer = set(DesignSettings.GetEnabledLayers().CuStack())
    BoardThickness = ToM(DesignSettings.GetBoardThickness())

    ItemList = {}

    for track in board.GetTracks():
        temp: dict[str, Any] = {"layer": getLayer(track, PossibleLayer)}
        if type(track) is pcbnew.PCB_VIA:
            temp["type"] = VIA
            temp["position"] = ToM(track.GetStart())
            temp["drill"] = ToM(track.GetDrill())
            temp["width"] = ToM(track.GetWidth())
            temp["conn_start"] = sorted(
                getHashList(connect.GetConnectedPads(track))
                + getHashList(connect.GetConnectedTracks(track))
            )
            temp["area"] = 0
        elif type(track) in (pcbnew.PCB_TRACK, pcbnew.PCB_ARC):
            temp["type"] = WIRE
            temp["start"] = ToM(track.GetStart())
            temp["end"] = ToM(track.GetEnd())
            temp["width"] = ToM(track.GetWidth())
            temp["length"] = ToM(track.GetLength())
            temp["area"] = temp["width"] * temp["length"]
            if track.GetLength() == 0:
                continue
            temp["layer"] = [track.GetLayer()]
            temp["conn_start"], temp["conn_end"] = getConnections(track, connect)
            if type(track) is pcbnew.PCB_ARC:
                temp["radius"] = ToM(track.GetRadius())
        else:
            log.warning("Unhandled track type: %s", type(track))
            continue

        temp["net_name"] = track.GetNetname()
        temp["net_code"] = track.GetNetCode()
        temp["id"] = getHash(track)
        temp["is_selected"] = track.IsSelected()
        ItemList[temp["id"]] = temp

    for item in board.AllConnectedItems():
        temp = {"layer": getLayer(item, PossibleLayer)}
        if type(item) is pcbnew.PAD:
            temp["type"] = PAD
            temp["shape"] = item.GetShape()
            # temp["PadAttr"] = Pad.ShowPadAttr()
            # temp["IsFlipped"] = Pad.IsFlipped()
            temp["position"] = ToM(item.GetPosition())
            temp["size"] = ToM(item.GetSize())
            temp["orientation"] = item.GetOrientation().AsDegrees()
            temp["drill_size"] = ToM(item.GetDrillSize())
            temp["drill"] = temp["drill_size"][0]
            Layers = temp.get("layer", [])

            if len(Layers):
                try:
                    poly_obj = item.GetEffectivePolygon()
                except Exception:
                    poly_obj = item.GetEffectivePolygon(aLayer=Layers[0])

                temp["area"] = ToM(ToM(poly_obj.Area()))
            else:
                temp["area"] = 0

            temp["PadName"] = item.GetPadName()
            # temp["FootprintUUID"] = getHash(Pad.GetParent())
            # if Pad.GetParent():
            #     temp["FootprintReference"] = Pad.GetParent().GetReference()

        elif type(item) is pcbnew.ZONE:
            if "teardrop" in item.GetZoneName():  # skip teardrop zones
                continue
            temp["type"] = ZONE
            temp["position"] = ToM(item.GetPosition())
            temp["area"] = ToM(ToM(item.GetFilledArea()))
            temp["NumCorners"] = item.GetNumCorners()
            temp["ZoneName"] = item.GetZoneName()
        elif type(item) is pcbnew.PCB_TRACK:
            continue  # already in board.GetTracks()
        elif type(item) is pcbnew.BOARD_CONNECTED_ITEM:
            if item.GetNetCode() == 0:
                continue
            log.warning("Unhandled item type: %s", type(item))
            continue
        else:
            log.warning("Unhandled item type: %s", type(item))
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
        if d["type"] == ZONE:
            zone_pos = d.get("position", (0, 0))
            for item in d["conn_start"]:
                if item not in ItemList:
                    continue
                item_data = ItemList[item]
                if item_data.get("type") == WIRE:
                    # Add zone to the wire end that's closer
                    start = item_data.get("start", (0, 0))
                    end = item_data.get("end", (0, 0))
                    dist_start = (start[0] - zone_pos[0]) ** 2 + (
                        start[1] - zone_pos[1]
                    ) ** 2
                    dist_end = (end[0] - zone_pos[0]) ** 2 + (end[1] - zone_pos[1]) ** 2
                    conn_key = "conn_start" if dist_start <= dist_end else "conn_end"
                    conn = item_data.setdefault(conn_key, [])
                    if uuid not in conn:
                        conn.append(uuid)
                else:
                    if uuid not in item_data.get("conn_start", []):
                        item_data.setdefault("conn_start", []).append(uuid)

    return ItemList, BoardThickness
