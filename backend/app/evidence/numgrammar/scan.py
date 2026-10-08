"""Inside ONE word (a maximal run of non-gap characters): the tokens it holds, or why it holds none.

EBNF (explicit classes only; ``digit`` is ``0-9`` and nothing else)::

    token        = [sign] (bare_power | body [exponent])
    body         = digits {sep digits}                 (read_body decides the shape)
    exponent     = e_notation | multiplier_power
    e_notation   = ("e"|"E") ["-"|"+"] digit{1,4}      (glued)
    multiplier_power = MULT "10" power                 (glued; MULT = x X * × ·)
    bare_power   = "10" "^" ["-"|"+"] digit{1,4}       (caret only: "10^6" = 1e6)
    power        = "^" ["-"|"+"] digit{1,4} | ["⁺"|"⁻"] supdigit{1,4}
    suffix       = "%" | "‰" | star_run                (zero gap, not part of the literal)
    unit         = see units.py                        (r7, decision 2: '150mg', '24h', '1.73m2'; context, not literal)
    ratio        = int "/" int                         (r7: '12/15', value n/d, parts (n, d); int = 0 | [1-9]
                                                        digit{0,5}; then the word ends, a trailing closer or an
                                                        opener follows: every other slash shape is refused)

A token starts at the word start, right after ( [ = < > ≤ ≥ ~ ≈ ±, or right after a joiner (- ± x X * × ·) that
directly follows the previous token. A word led by a letter is a LABEL: digit runs not after a starter never form
tokens and any rejection inside it makes it inert (``AUC0-24``, ``t1/2``, ``CYP2B6``, ``mL/min/1.73m2``). A word led
by a non-letter inside character is NUMERIC-LED: a digit run that cannot start, or a token that fails its
post-boundary, REJECTS the whole word (``Scan.kind == "rejected"``). Everything runs in time linear in the word:
every character is consumed once and a rejection ends the scan.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from app.evidence.numgrammar.alphabet import (
    ASCII_LETTERS,
    DIGITS,
    EXP_MARKERS,
    INSIDE,
    JOINERS,
    LETTERS,
    MULTIPLIERS,
    OPENERS_BR,
    OPERATORS,
    PERCENTS,
    PLUS_MINUS,
    QUALIFIERS,
    SEPARATORS,
    SIGNS,
    STARTERS,
    SUP_CHARS,
    SUP_DIGITS,
    SUP_SIGNS,
    SUP_TO_ASCII,
    TRAILING,
    char_class,
)
from app.evidence.numgrammar.body import MAX_DIGITS, MIDDOT, exp_int, read_body, to_float
from app.evidence.numgrammar.body import mantissa as read_mantissa
from app.evidence.numgrammar.glue import glue, prefix
from app.evidence.numgrammar.tok import MAX_PARTS, REASON_SHAPE, Tok
from app.evidence.numgrammar.units import unit_end

_DIG = re.compile("[0-9]+")
_PLAIN_INT = re.compile("0|[1-9][0-9]{0,39}")
_PLAIN_DEC = re.compile("(0|[1-9][0-9]{0,38})([.,])([0-9]{1,39})")
_SUPDIG = re.compile(char_class(SUP_DIGITS, negate=False) + "+")
_CITATION = re.compile("\\[[1-9][0-9]?,[0-9]{1,2}\\]")      # '[1,2]': a pair of reference numbers, never a decimal
RATIO_INT = re.compile("0|[1-9][0-9]{0,5}")                 # one side of a ratio (r7, owner decision 6)
OK, PCT, STAR, TAIL, UNIT, BAD = 0, 1, 2, 3, 4, -1
TAIL_NONE, TAIL_MULT, TAIL_STAR = 0, 1, 2
_SUP_MINUS = "\u207b"
_DEGREE = "\u00b0"


class Scan(NamedTuple):
    """kind: ``numeric`` (>= 1 resolved token), ``rejected`` (a numeric-led word the grammar refuses: ``why``,
    ``parts`` = the resolved numbers before the failure), ``label`` (digits inside a label: no tokens)."""
    kind: str
    toks: tuple[Tok, ...] = ()
    tail: int = TAIL_NONE          # a glued multiplier (x X × ·) or a trailing star run ends the word
    why: str = ""
    parts: tuple[float, ...] = ()
    unit: str = ""                 # a glued unit that ends the word ('mg' of '150mg'): the next word may split it


LABEL = Scan("label")


def read_power(w: str, k: int, n: int) -> tuple[int | None, int] | None:
    """The power starting at w[k]: ``^[sign]digits`` or ``[⁺⁻]supdigits`` -> (exponent or None when out of range,
    end index); None when w[k:] is not a power."""
    c = w[k]
    if c == "^":
        p, neg = k + 1, False
        if p < n and w[p] in SIGNS:
            neg, p = w[p] == "-", p + 1
        m = _DIG.match(w, p)
        return None if m is None else (exp_int(m.group(), neg), m.end())
    if c in SUP_CHARS:
        p, neg = k, False
        if c in SUP_SIGNS:
            neg, p = c == _SUP_MINUS, p + 1
        m = _SUPDIG.match(w, p)
        return None if m is None else (exp_int(m.group().translate(SUP_TO_ASCII), neg), m.end())
    return None


def _scaled(sign: str, mantissa: str, e: int | None) -> tuple[float | None, str]:
    """(value, canonical) of ``sign mantissa e e`` (exact decimal rounding), value None on overflow."""
    canon = f"{sign}{mantissa}e{e}"
    return to_float(canon), canon


def _bare_power(w: str, j: int, n: int, sign: str) -> tuple | None:
    """``10^6`` -> a token; a superscript ten (``10³``) -> rejected; anything else -> None (plain ``10``)."""
    c = w[j]
    if c in SUP_CHARS:
        pw = read_power(w, j, n)
        return None if pw is None else (pw[1], None, "bare_superscript_power", "", "")
    if c != "^":
        return None
    pw = read_power(w, j, n)
    if pw is None:
        return None
    value, canon = _scaled(sign, "1", pw[0]) if pw[0] is not None else (None, "")
    return (pw[1], None, "exponent_range", "", "") if value is None else (pw[1], value, "bare_power", canon, "")


def _exponent(w: str, j: int, n: int, sign: str, canon: str) -> tuple | None:
    """What glued after the body at w[j]: e-notation or multiplier_power -> (end, value, canonical) or a rejection
    tuple ``(end, None, why)``; None when nothing of the kind follows."""
    c = w[j]
    if c in EXP_MARKERS:
        p, neg = j + 1, False
        if p < n and w[p] in SIGNS:
            neg, p = w[p] == "-", p + 1
        m = _DIG.match(w, p)
        if m is None:
            return None
        e, end = exp_int(m.group(), neg), m.end()
    elif c in MULTIPLIERS and w[j + 1:j + 3] == "10":
        pw = read_power(w, j + 3, n) if j + 3 < n else None
        if pw is None:
            ten = c == MIDDOT or w[j + 3:j + 4] != "0"       # '×10', '×103', '·100'; '×100' is a product
            return (j, None, "ten_after_multiplier") if ten else None
        e, end = pw
    else:
        return None
    if e is None:
        return end, None, "exponent_range"
    value, scaled = _scaled(sign, canon, e)
    return (end, value, scaled) if value is not None else (end, None, "exponent_range")


def read_token(w: str, s: int, n: int, style: str) -> tuple | None:
    """The token starting at w[s] (a sign or a digit): ``(end, value, shape, canonical, mark)``; ``value`` None
    means rejected and ``shape`` is then the reason; None when no token starts here."""
    i, sign = s, ""
    c = w[i]
    if c in SIGNS:
        sign = "-" if c == "-" else ""
        i += 1
        if i >= n or w[i] not in DIGITS:
            return None
    elif c not in DIGITS:
        return None
    j = _DIG.match(w, i).end()
    head = w[i:j]
    if j < n and head == "10":
        bare = _bare_power(w, j, n, sign)
        if bare is not None:
            return bare
    groups, seps, total = [head], [], j - i
    while j < n and total <= MAX_DIGITS:
        sp = w[j]
        if sp not in SEPARATORS or j + 1 >= n or w[j + 1] not in DIGITS or (sp == MIDDOT and w[j + 1:j + 3] == "10"):
            break
        k = _DIG.match(w, j + 1).end()
        total += k - j - 1
        seps.append(sp)
        groups.append(w[j + 1:k])
        j = k
    body = read_body(groups, seps, style)
    exp = None
    if j < n:                         # a mantissa (an exponent follows) reads one separator as a decimal
        mant = read_mantissa(groups, seps, style)
        if mant is not None and not isinstance(mant, str):
            exp = _exponent(w, j, n, sign, mant[0])
            body = mant if exp is not None else body
    if isinstance(body, str):
        return j, None, body, "", ""
    canon, shape, mark = body
    if exp is None and j < n:
        exp = _exponent(w, j, n, sign, canon)
    if exp is not None:
        if exp[1] is None:
            return exp[0], None, exp[2], sign + canon, mark          # the canonical mantissa rides along as a part
        return exp[0], exp[1], "sci", exp[2], mark
    return j, to_float(sign + canon), shape, sign + canon, mark


def _after_mult(w: str, e: int, n: int, p: str, x: str) -> tuple[int, int, str]:
    """A glued multiplier (x X × · *) after a token: at the word end (the operand is the next word), before digits
    that do not start a ten ('2×5', '3×250'); a ten here is a power whose exponent was lost."""
    if x == "":
        return TAIL, e + 1, ""
    if x in DIGITS:
        if w[e + 1:e + 3] == "10" and (p == MIDDOT or w[e + 3:e + 4] != "0"):
            return BAD, e, "ten_after_multiplier"
        return OK, e, ""
    return BAD, e, "glued_letters" if x in LETTERS else "glued_symbol"


def post(w: str, e: int, n: int) -> tuple[int, int, str]:
    """The post-boundary of a token ending at *e*: ``(OK|PCT|STAR|TAIL|UNIT, index after the suffix or unit, "")`` or
    ``(BAD, e, reason)``. A letter that starts no complete unit production (units.py), ``=``, ``^``, ``+``, ``:`` +
    digit, ``/`` + digit, a second sign... glued to the digits reject the word."""
    if e >= n:
        return OK, e, ""
    p = w[e]
    x = w[e + 1] if e + 1 < n else ""
    if p in TRAILING:
        return (OK, e, "") if x not in DIGITS else (BAD, e, "glued_symbol")
    if p in PERCENTS:
        if x == "" or x in TRAILING or x in LETTERS or x in OPENERS_BR or x == "-":
            return PCT, e + 1, ""
        return BAD, e, "glued_symbol"
    if p == "*":
        k = e + 1
        while k < n and w[k] == "*":
            k += 1
        y = w[k] if k < n else ""
        if y == "" or y in TRAILING:
            return STAR, k, ""
        return _after_mult(w, e, n, p, y) if k == e + 1 and y in DIGITS else (BAD, e, "glued_symbol")
    if p == _DEGREE or p in OPENERS_BR:
        return OK, e, ""
    if p == "/":
        return (OK, e, "") if x != "" and x not in DIGITS else (BAD, e, "glued_symbol")
    if p == "-":
        return (OK, e, "") if x == "" or x in DIGITS or x in LETTERS else (BAD, e, "glued_symbol")
    if p == PLUS_MINUS:
        return (OK, e, "") if x in DIGITS or x in SIGNS else (BAD, e, "glued_symbol")
    if p in MULTIPLIERS:
        return _after_mult(w, e, n, p, x)
    if p in LETTERS:
        u = unit_end(w, e)
        return (UNIT, u, "") if u > e else (BAD, e, "glued_letters")
    return BAD, e, "glued_symbol"


def ratio_parts(num: str, den: str) -> tuple[float, str, tuple[float, float]] | None:
    """(value, canonical, parts) of the ratio num/den of two ``RATIO_INT`` strings, None for a zero denominator."""
    n, d = to_float(num), to_float(den)
    return None if not d else (n / d, f"{num}/{den}", (n, d))


def _ratio(w: str, e: int, n: int, num: str) -> tuple[int, float | None, str, tuple[float, ...]]:
    """The ratio whose numerator *num* ends at w[e] == '/': (end, value, canonical, parts) or (e, None, why, parts)
    when the shape is refused (a long or zero denominator, a third part, a percent, a range, glued letters)."""
    m = _DIG.match(w, e + 1)
    den, k = m.group(), m.end()
    rp = ratio_parts(num, den) if RATIO_INT.fullmatch(den) else None
    parts = (to_float(num), to_float(den)) if len(den) <= MAX_DIGITS else (to_float(num),)
    if rp is None:
        return e, None, "fraction", parts
    x = w[k] if k < n else ""
    if x == "" or x in OPENERS_BR or (x in TRAILING and (k + 1 >= n or w[k + 1] not in DIGITS)):
        return k, rp[0], rp[1], rp[2]
    return e, None, "glued_letters" if x in LETTERS else "fraction", parts


def _qualified(w: str, i: int) -> bool:
    """A qualifier stands before the token at w[i] in its own word, with only openers, ``=``, ``:``, ``;``, ``*`` and
    signs between (``<5``, ``<=5``, ``~=5``, ``\u2264(5)``, ``<-5``)."""
    k = i
    while k > 0 and w[k - 1] in OPERATORS:
        k -= 1
    return any(c in QUALIFIERS for c in w[k:i])


def _rejected(why: str, parts: list[float]) -> Scan:
    return Scan("rejected", why=why, parts=tuple(parts[:MAX_PARTS]))


def scan_word(w: str, off: int, style: str) -> Scan:
    """Tokens of the word *w* (starting at global offset *off*): see the module docstring."""
    n = len(w)
    if _PLAIN_INT.fullmatch(w):                              # the common cases: a bare integer, a plain decimal
        return Scan("numeric", (Tok(off, off + n, to_float(w), "int", (), False, False, w, ""),))
    m = _PLAIN_DEC.fullmatch(w)
    if m is not None and not isinstance(body := read_body([m[1], m[3]], [m[2]], style), str):
        return Scan("numeric", (Tok(off, off + n, to_float(body[0]), body[1], (), False, False, body[0], body[2]),))
    if w[0] == "[" and _CITATION.fullmatch(w.rstrip(".,;:)'")) is not None:
        return _rejected("citation_list", [])
    lead_i = 1 if w[0] in STARTERS else 0
    lead = w[lead_i] if lead_i < n else ""
    numeric_led = lead != "" and lead in INSIDE and lead not in ASCII_LETTERS
    toks: list[Tok] = []
    parts: list[float] = []
    i, prev_end, tail, unit, glue_from = 0, -1, TAIL_NONE, "", 0
    while i < n:
        c = w[i]
        if prev_end == i and c in JOINERS and i + 1 < n and w[i + 1] in DIGITS:
            i += 1                                              # '12-15', '36.6±4.3', '2×5', '45%-55%'
            continue
        if c not in DIGITS and c not in SIGNS:
            i += 1
            continue
        r = read_token(w, i, n, style)
        if r is None:
            i += 1
            continue
        end, value, shape, canon, mark = r
        pre = w[i - 1] if i else ""
        joiner = i > 0 and prev_end == i - 1 and pre in JOINERS
        if not (i == 0 or pre in STARTERS or joiner):
            if numeric_led:
                return _rejected("bad_start", parts)           # '.05', '*12', 'x-20' inside a numeric word
            i = max(end, i + 1)
            continue
        if not joiner and i > glue_from and not prefix(w[glue_from:i]):     # r7 writer 3: '-(12)', '-=12', 'x(12)'
            got = parts + ([value] if value is not None else [])
            return _rejected("glued_symbol", got) if numeric_led else LABEL
        if value is None:
            if not numeric_led:
                return LABEL
            mantissa = to_float(canon) if canon else None
            return _rejected(shape, parts + ([mantissa] if mantissa is not None else []))
        if shape == "int" and not joiner and c in DIGITS and end + 1 < n and w[end] == "/" and w[end + 1] in DIGITS \
                and RATIO_INT.fullmatch(canon):                         # unsigned: '+3/4' and '-3/4' are refused
            r_end, r_value, r_canon, r_parts = _ratio(w, end, n, canon)
            if r_value is None:
                return _rejected(r_canon, parts + list(r_parts)) if numeric_led else LABEL
            toks.append(Tok(off + i, off + r_end, r_value, "ratio", r_parts, _qualified(w, i), False, r_canon, ""))
            parts.append(r_value)
            i = prev_end = glue_from = r_end
            tail = TAIL_NONE
            continue
        act, after, why = post(w, end, n)
        if act == BAD:
            return _rejected(why, parts + [value]) if numeric_led else LABEL
        ranged = joiner and pre == "-"
        qualified = _qualified(w, i)
        if ranged and toks:
            toks[-1] = toks[-1]._replace(ranged=True)
        # a scientific token remembers its mantissa (``1.2`` of ``1.2e-3``), a bare power its ten: citing only that
        # is a partial number, never the number
        part = (to_float(canon.partition("e")[0]),) if shape == "sci" else ((10.0,) if shape == "bare_power" else ())
        toks.append(Tok(off + i, off + end, value, shape, part, qualified, ranged, canon, mark))
        parts.append(value)
        i = prev_end = glue_from = end
        tail, unit = TAIL_NONE, ""
        if act == TAIL:
            tail = TAIL_MULT
        elif act in (PCT, STAR, UNIT):
            i = prev_end = glue_from = after
            tail = TAIL_STAR if act == STAR and i == n else TAIL_NONE
            unit = w[end:after] if act == UNIT and after == n else ""
    if not toks:
        return LABEL
    rest = w[glue_from:]                                    # r7 writer 3: '12-x', '12(e)', '5/x'; a lone '-' is a range
    if rest and tail != TAIL_MULT and rest != "-" and not glue(rest):
        return _rejected("glued_symbol", parts) if numeric_led else LABEL
    return Scan("numeric", tuple(toks), tail, unit=unit)


__all__ = ["LABEL", "RATIO_INT", "Scan", "post", "ratio_parts", "read_power", "read_token", "scan_word", "REASON_SHAPE"]
