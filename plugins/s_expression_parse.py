import re
from typing import Any

_TERM = re.compile(
    r"""(?mx)
    \s*(?:
        (?P<brackl>\()|
        (?P<brackr>\))|
        (?P<num>\-?\d+\.\d+|\-?\d+)|
        (?P<sq>"(?:(?:\\")|[^"])*")|
        (?P<s>[^(^)\s]+)
       )"""
)


def parse_sexp(sexp: str, pos: int = 0) -> list[Any]:
    """The first complete S-expression in `sexp` from index `pos` on.

    Raises:
        ValueError: brackets do not match before the expression is complete
    """
    stack: list[list[Any]] = []
    out: list[Any] = []
    for match in _TERM.finditer(sexp, pos):
        term, value = match.lastgroup, match.group(match.lastgroup or 0)
        if term == "brackl":
            stack.append(out)
            out = []
        elif term == "brackr":
            if not stack:
                raise ValueError("Unmatched closing bracket in S-expression")
            done, out = out, stack.pop()
            if not stack:
                return done
            out.append(done)
        elif term == "num":
            v = float(value)
            out.append(int(v) if v.is_integer() else v)
        elif term == "sq":
            out.append(value[1:-1])
        else:
            out.append(value)
    raise ValueError("Incomplete S-expression")
