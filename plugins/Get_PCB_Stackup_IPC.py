"""Get PCB stackup via kipy IPC API (KiCad 9+)."""

from typing import Any

from kipy.util.board_layer import canonical_name, is_copper_layer

from Get_PCB_Stackup import extract_layer_from_string, Get_PCB_Stackup_fun


def Get_PCB_Stackup_IPC(board: Any) -> dict[int, dict]:
    """Get CuStack via kipy IPC API.

    Returns the same dict format as Get_PCB_Stackup_fun().
    """
    stackup = board.get_stackup()

    # Build flat list: each entry is either a copper or dielectric layer
    flat = []
    for sl in stackup.layers:
        if sl.type == 1:  # BSLT_COPPER
            name = (
                canonical_name(sl.layer) if is_copper_layer(sl.layer) else sl.user_name
            )
            flat.append(
                {
                    "kind": "copper",
                    "layer_id": extract_layer_from_string(name),
                    "name": name,
                    "thickness": sl.thickness / 1e9,  # nm → m
                }
            )
        elif sl.type == 2:  # BSLT_DIELECTRIC
            sub = sl.dielectric.layers
            if sub:
                thickness = sum(s.thickness for s in sub) / 1e9
                epsilon_r = sub[0].epsilon_r if sub[0].epsilon_r else 4.3
                loss_tangent = sub[0].loss_tangent if sub[0].loss_tangent else 0.02
            elif sl.thickness > 0:
                thickness = sl.thickness / 1e9
                epsilon_r = 4.3
                loss_tangent = 0.02
            else:
                continue
            flat.append(
                {
                    "kind": "dielectric",
                    "thickness": thickness,
                    "epsilon_r": epsilon_r,
                    "loss_tangent": loss_tangent,
                }
            )

    # Build CuStack from flat list
    CuStack = {}
    abs_height = 0.0
    for i, entry in enumerate(flat):
        if entry["kind"] == "dielectric":
            abs_height += entry["thickness"]
            continue

        def _die(idx: int) -> dict | None:
            if 0 <= idx < len(flat) and flat[idx]["kind"] == "dielectric":
                d = flat[idx]
                return {
                    "h": d["thickness"],
                    "epsilon_r": d["epsilon_r"],
                    "loss_tangent": d["loss_tangent"],
                }
            return None

        die_above = _die(i - 1)
        die_below = _die(i + 1)
        if die_above and die_below:
            model = "Stripline"
        elif die_above or die_below:
            model = "Microstrip"
        else:
            model = "R only"

        CuStack[entry["layer_id"]] = {
            "thickness": entry["thickness"],
            "name": entry["name"],
            "abs_height": abs_height,
            "die_above": die_above,
            "die_below": die_below,
            "model": model,
            "gap": None,
        }
        abs_height += entry["thickness"]

    if not CuStack:
        # Fallback: delegate to file-based parser
        board_file = board.document.board_filename
        return Get_PCB_Stackup_fun(board_file, new_v9=True)

    return CuStack


if __name__ == "__main__":
    from kipy import KiCad

    kicad = KiCad()
    board = kicad.get_board()

    print(f"Board: {board.document.board_filename}\n")

    TYPE_NAMES = {
        1: "Copper",
        2: "Dielectric",
        3: "Silkscreen",
        4: "Soldermask",
        5: "Solderpaste",
    }

    stackup = board.get_stackup()
    print("=== Raw Stackup (copper & dielectric only) ===")
    for sl in stackup.layers:
        if sl.type not in (1, 2):
            continue
        tname = TYPE_NAMES.get(sl.type, f"type={sl.type}")
        t_um = sl.thickness / 1000  # nm → µm
        if sl.type == 1:
            name = canonical_name(sl.layer) if is_copper_layer(sl.layer) else "?"
            print(
                f"  {name:8s}  {tname:12s}  {t_um:8.1f} µm  {sl.material_name or '-'}"
            )
        else:
            sub = sl.dielectric.layers
            if sub:
                eps = sub[0].epsilon_r or "?"
                tan_d = sub[0].loss_tangent or "?"
                mat = sub[0].material_name or "-"
            else:
                eps = "-"
                tan_d = "-"
                mat = "-"
            print(
                f"  {'':8s}  {tname:12s}  {t_um:8.1f} µm  "
                f"εr={eps}  tan δ={tan_d}  {mat}"
            )
    print()

    CuStack = Get_PCB_Stackup_IPC(board)

    print("=== CuStack ===")
    for layer_id, info in sorted(CuStack.items()):
        die_str = []
        for label, die in [("above", info["die_above"]), ("below", info["die_below"])]:
            if die:
                die_str.append(f"{label}: {die['h'] * 1e6:.0f}µm εr={die['epsilon_r']}")
            else:
                die_str.append(f"{label}: -")
        print(
            f"  Layer {layer_id:2d} ({info['name']:5s})  {info['model']:11s}  "
            f"{info['thickness'] * 1e6:.1f}µm Cu  {', '.join(die_str)}"
        )
