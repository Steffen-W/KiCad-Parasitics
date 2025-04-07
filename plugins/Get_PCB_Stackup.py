from os.path import exists

try:
    from .s_expression_parse import parse_sexp
except:
    from s_expression_parse import parse_sexp

import re


def extract_layer_from_string_old(input_string):  # kicad <9.0
    if input_string == "F.Cu":
        return 0
    elif input_string == "B.Cu":
        return 31
    else:
        match = re.search(r"In(\d+)\.Cu", input_string)
        if match:
            return int(match.group(1))
    return None


def extract_layer_from_string(input_string):  # kicad >=9.0
    # https://gitlab.com/kicad/code/kicad/-/commit/5e0abadb23425765e164f49ee2f893e94ddb97fc
    if input_string == "F.Cu":
        return 0
    elif input_string == "B.Cu":
        return 2
    else:
        match = re.match(r"In(\d+)\.Cu", input_string)
        if match:
            inner_index = int(match.group(1))
            return 2 * inner_index + 2  # In1_Cu = 4, In2_Cu = 6, ...
    return None


def search_recursive(line: list, entry: str, all=False):
    if type(line[0]) == str and line[0] == entry:
        if all:
            return line
        else:
            return line[1]

    for e in line:
        if type(e) == list:
            res = search_recursive(line=e, entry=entry, all=all)
            if not res == None:
                return res
    return None


def Get_PCB_Stackup_fun(ProjectPath="./test.kicad_pcb", new_v9=True):
    def readFile2var(path):
        if not exists(path):
            return None

        with open(path, "r") as file:
            data = file.read()
        return data

    PhysicalLayerStack = []
    CuStack = {}
    try:
        if exists(ProjectPath):
            txt = readFile2var(ProjectPath)
            parsed = parse_sexp(txt)

            while True:
                setup = search_recursive(parsed, "setup", all=True)
                if not setup:
                    break

                stackup = search_recursive(setup, "stackup", all=True)
                if not stackup:
                    break

                abs_height = 0.0
                for layer in stackup:
                    tmp = {}
                    tmp["layer"] = search_recursive(layer, "layer")
                    tmp["thickness"] = search_recursive(layer, "thickness")
                    tmp["epsilon_r"] = search_recursive(layer, "epsilon_r")
                    tmp["type"] = search_recursive(layer, "type")

                    if not tmp["thickness"] == None:
                        if new_v9:
                            tmp["cu_layer"] = extract_layer_from_string(tmp["layer"])
                        else:
                            tmp["cu_layer"] = extract_layer_from_string_old(
                                tmp["layer"]
                            )
                        tmp["abs_height"] = abs_height
                        abs_height += float(tmp["thickness"])
                        PhysicalLayerStack.append(tmp)
                break

            for Layer in PhysicalLayerStack:
                if not Layer["cu_layer"] == None:
                    CuStack[Layer["cu_layer"]] = {
                        "thickness": Layer["thickness"],
                        "name": Layer["layer"],
                        "abs_height": Layer["abs_height"],
                    }
                    if Layer["thickness"] <= 0:
                        raise Exception("Problematic layer thickness detected")
    except:
        print("ERROR: Reading the CuStack")

    if not CuStack:
        layers = search_recursive(parsed, "layers", all=True)
        for layer in layers:
            if type(layer) == list and "signal" in layer:
                CuStack[layer[0]] = {
                    "thickness": 0.035,
                    "name": layer[1],
                    "abs_height": float(layer[0]) / 20,  # arbitrary assumption
                }
        print("estimated CuStack", CuStack)

    return PhysicalLayerStack, CuStack
