import re

dbg = False

term_regex = r"""(?mx)
    \s*(?:
        (?P<brackl>\()|
        (?P<brackr>\))|
        (?P<num>\-?\d+\.\d+|\-?\d+)|
        (?P<sq>"[^"]*")|
        (?P<s>[^(^)\s]+)
       )"""


def parse_sexp(sexp):
    stack = []
    out = []
    if dbg:
        print("%-6s %-14s %-44s %-s" % tuple("term value out stack".split()))
    for termtypes in re.finditer(term_regex, sexp):
        term, value = [(t, v) for t, v in termtypes.groupdict().items() if v][0]
        if dbg:
            print("%-7s %-14s %-44r %-r" % (term, value, out, stack))
        if term == "brackl":
            stack.append(out)
            out = []
        elif term == "brackr":
            assert stack, "Trouble with nesting of brackets"
            tmpout, out = out, stack.pop(-1)
            out.append(tmpout)
        elif term == "num":
            v = float(value)
            if v.is_integer():
                v = int(v)
            out.append(v)
        elif term == "sq":
            out.append(value[1:-1])
        elif term == "s":
            out.append(value)
        else:
            raise NotImplementedError("Error: %r" % (term, value))
    assert not stack, "Trouble with nesting of brackets"
    return out[0]


def print_sexp(exp):
    out = ""
    if type(exp) == type([]):
        out += "(" + " ".join(print_sexp(x) for x in exp) + ")"
    elif type(exp) == type("") and re.search(r"[\s()]", exp):
        out += '"%s"' % repr(exp)[1:-1].replace('"', '"')
    else:
        out += "%s" % exp
    return out


if __name__ == "__main__":
    from pprint import pprint

    sexp = """(sym_lib_table
  (version 7)
  (lib (name "4xxx")(type "KiCad")(uri "${KICAD7_SYMBOL_DIR}/4xxx.kicad_sym")(options "")(descr "4xxx series symbols"))
  (lib (name "4xxx_IEEE")(type "KiCad")(uri "${KICAD7_SYMBOL_DIR}/4xxx_IEEE.kicad_sym")(options "")(descr "4xxx series IEEE symbols"))
  (lib (name "74xGxx")(type "KiCad")(uri "${KICAD7_SYMBOL_DIR}/74xGxx.kicad_sym")(options "")(descr "74xGxx symbols"))
  (lib (name "74xx")(type "KiCad")(uri "${KICAD7_SYMBOL_DIR}/74xx.kicad_sym")(options "")(descr "74xx symbols"))
  (lib (name "74xx_IEEE")(type "KiCad")(uri "${KICAD7_SYMBOL_DIR}/74xx_IEEE.kicad_sym")(options "")(descr "74xx series IEEE symbols"))
  )"""

    parsed = parse_sexp(sexp)
    # pprint(parsed)
    for line in parsed:
        if type(line) == list and line[0] == "lib":
            for item in line:
                print(item)
