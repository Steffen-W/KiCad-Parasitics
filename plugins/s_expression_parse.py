import logging
import re

log = logging.getLogger(__name__)

term_regex = r"""(?mx)
    \s*(?:
        (?P<brackl>\()|
        (?P<brackr>\))|
        (?P<num>\-?\d+\.\d+|\-?\d+)|
        (?P<sq>"(?:(?:\\")|[^"])*")|
        (?P<s>[^(^)\s]+)
       )"""


def parse_sexp(sexp: str) -> list:
    stack: list = []
    out: list = []
    for termtypes in re.finditer(term_regex, sexp):
        term, value = [(t, v) for t, v in termtypes.groupdict().items() if v][0]
        log.debug("%-7s %-14s %-44r %-r", term, value, out, stack)
        if term == "brackl":
            stack.append(out)
            out = []
        elif term == "brackr":
            if not stack:
                raise ValueError("Unmatched closing bracket in S-expression")
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
            raise NotImplementedError(f"Error: {term!r}, {value!r}")
    if stack:
        raise ValueError("Unclosed bracket in S-expression")
    return out[0]


def print_sexp(exp: list | str | int | float) -> str:
    out = ""
    if isinstance(exp, list):
        out += "(" + " ".join(print_sexp(x) for x in exp) + ")"
    elif isinstance(exp, str) and re.search(r"[\s()]", exp):
        out += '"%s"' % repr(exp)[1:-1].replace('"', '"')
    else:
        out += "%s" % exp
    return out


if __name__ == "__main__":
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
        if isinstance(line, list) and line[0] == "lib":
            for item in line:
                print(item)
