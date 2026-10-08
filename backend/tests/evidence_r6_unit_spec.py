"""Layer-B specification of r7 writer 2 (owner decisions 2 and 3), written independently of ``numgrammar/units.py``.

``spec_unit(tail)`` is the unit production built from the SECOND copy of its terminals (``evidence_r6_pins``): the length
of the unit glued at the start of *tail* when it ends at a boundary, else -1. ``sweep_unitprod`` and ``sweep_label`` are
the two layer-B sweeps (registered in ``evidence_r6_layer_b.SWEEPS``):

  unitprod  every code point at every boundary inside the glued unit of a few expressions ('took 150mg daily'): an outside
            character leaves the number unresolved (context); for any other character the number resolves ONLY when the
            glued text up to its first boundary is a unit of the specification and what follows the boundary in the
            word is inert glue (r7 writer 3, the neighbour specification: '3ml,E' is no reading) (soundness), with the
            clean value and the number alone as literal; for a letter it resolves EXACTLY when the specification says so
  label     every code point at every boundary inside a label word ('AUC0-24 -0.5 h'): replacing the digits the reader
            calls label digits (``labels=True``) by a letter changes no token outside the label word (decision 3: label
            digits never act on a neighbour by themselves)
"""
from __future__ import annotations

import re

from app.evidence.numgrammar import tokens
from tests import evidence_r6_neighbour_spec as N
from tests import evidence_r6_oracle as O
from tests import evidence_r6_pins as P

_LETTERS = O._ASCII_LETTERS | O._GREEK | O._ACCENTED          # every letter the alphabet lists (e E x X included)
_SUP = "".join(chr(c) for c in (0x2070, 0xB9, 0xB2, 0xB3, *range(0x2074, 0x207A)))
_SUP_SIGN = "⁺⁻"
_CLOSERS = ")],;:.'"
_OPENERS = "(["


def _alt(terms: frozenset[str]) -> str:
    return "(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + ")"


_PRE, _BASE, _PREFIXED = _alt(P.UNIT_PREFIX_PIN), _alt(P.UNIT_BASE_PIN), _alt(P.UNIT_PREFIXED_BASE_PIN)
_EXP = f"(?:[23](?![0-9])|\\^[+-]?[0-9][0-9]?(?![0-9])|[{_SUP_SIGN}]?[{_SUP}][{_SUP}]?)"
_ATOM = f"(?:{_PRE}?{_BASE}|{_PRE}{_PREFIXED}){_EXP}?"
_SPEC = re.compile(f"{_ATOM}(?:(?:/|per){_ATOM})*(?=$|[{re.escape(_CLOSERS)}](?![0-9])|[{re.escape(_OPENERS)}])")


def spec_unit(tail: str) -> int:
    m = _SPEC.match(tail)
    return -1 if m is None else m.end()


def spec_split(unit: str, nxt: str) -> bool:
    """The next word continues a unit that ends its word ('m' + 'L')."""
    m = _SPEC.match(unit + nxt)
    return m is not None and m.end() > len(unit)


def _fold(c: str) -> str:
    return chr(P.EQUIV_PIN.get(ord(c), ord(c)))


UNIT_EXPRS = (("took 150mg daily", "150"), ("BSA 1.73m2 ok", "1.73"), ("dose 2.5mg/kg/h iv", "2.5"),
              ("took 10mL ok", "10"), ("rate 5h^-1 ok", "5"), ("took 5mgperkg ok", "5"), ("vol 3ml, ok", "3"),
              ("cell (20mg) ok", "20"))


def _unit_case(text: str, number: str, b: int, cp: int) -> str:
    """'' when the variant reads as the specification says, else the violation."""
    c, kind = chr(cp), P.expected_class(cp)
    s0 = text.index(number)
    ns, we = s0 + len(number), text.index(" ", s0)
    v = text[:b] + c + text[b:]
    hit = next((t for t in tokens(v) if t.start <= s0 < t.end), None)
    if kind == "outside":
        ok = hit is not None and hit.value is None and hit.reason == "unresolved_context"
        return "" if ok else f"outside: {hit}"
    f = _fold(c)
    if kind == "gap" or f in (" ", " ") or (b == ns and f not in _LETTERS):
        return ""                                     # a gap splits the word; the number's own boundary: the golden file
    word = (text[:b] + f + text[b:])[s0:we + 1]
    tail = word[len(number):]
    n = spec_unit(tail)
    want = n >= 0 and (n < len(tail) or not spec_split(tail, text[we + 1:].split(" ")[0])) and N.glue(tail[n:].rstrip())
    got = hit is not None and hit.value is not None and (hit.start, hit.end) == (s0, ns)
    if got and (not want or v[hit.start:hit.end] != number):
        return f"resolved {v[hit.start:hit.end]!r} though the glued text {tail!r} is no unit of the specification"
    if f in _LETTERS and got != want:
        return f"letter: the number is {'resolved' if got else 'unresolved'}, the specification says {want}"
    return ""


def sweep_unitprod(cps: list[int]) -> list[str]:
    bad = []
    for text, number in UNIT_EXPRS:
        s0 = text.index(number)
        ns, we = s0 + len(number), text.index(" ", s0)
        for b in range(ns, we + 1):
            for cp in cps:
                if 0xD800 <= cp <= 0xDFFF:
                    continue
                if (msg := _unit_case(text, number, b, cp)):
                    bad.append(f"{text!r}@{b} U+{cp:04X}: {msg}")
                    if len(bad) >= 20:
                        return bad
    return bad


LABEL_EXPRS = (("AUC0-24 -0.5 h", 0, (4, 5, 7)), ("by CYP2D6 3/4 of", 3, (7, 8, 9)), ("t1/2 12 / 15 ok", 0, (2, 3, 4)),
               ("CYP2D6 [38] 20 trials", 0, (4, 5, 6)), ("C24h 4 12/15", 0, (2, 3, 4)), ("m12 3/4 tab", 0, (2, 3)))


def _outside(toks, lo: int, hi: int) -> list[tuple]:
    return [(t.start, t.end, t.value, t.shape, t.qualified, t.ranged, t.parts) for t in toks
            if not (lo <= t.start and t.end <= hi)]


def _label_case(text: str, ls: int, b: int, cp: int) -> str:
    c = chr(cp)
    if P.expected_class(cp) == "gap" or _fold(c) in (" ", " "):
        return ""                                     # a gap splits the label word: not a label digit question
    v = text[:b] + c + text[b:]
    le = v.index(" ", ls)
    labelled = [t for t in tokens(v, labels=True) if t.why == "label" and ls <= t.start and t.end <= le]
    q = list(v)
    for t in labelled:
        q[t.start:t.end] = ["q" if ch in "0123456789" else ch for ch in v[t.start:t.end]]
    a, z = _outside(tokens(v), ls, le), _outside(tokens("".join(q)), ls, le)
    return "" if a == z else f"label digits change a neighbour: {a} != {z}"


def sweep_label(cps: list[int]) -> list[str]:
    bad = []
    for text, ls, cuts in LABEL_EXPRS:
        for b in cuts:
            for cp in cps:
                if 0xD800 <= cp <= 0xDFFF:
                    continue
                if (msg := _label_case(text, ls, b, cp)):
                    bad.append(f"{text!r}@{b} U+{cp:04X}: {msg}")
                    if len(bad) >= 20:
                        return bad
    return bad
