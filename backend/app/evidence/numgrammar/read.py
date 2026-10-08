"""One TOKEN at a position of a word (split out of scan.py in r7 round 3): its body, its glued exponent, a bare power.

    token        = [sign] (bare_power | body [exponent])
    body         = digits {sep digits}                 (body.read_body decides the shape)
    exponent     = e_notation | multiplier_power
    e_notation   = ("e"|"E") ["-"|"+"] digit{1,4}      (glued)
    multiplier_power = MULT "10" power                 (glued; MULT = x X * × ·)
    bare_power   = "10" "^" ["-"|"+"] digit{1,4}       (caret only: "10^6" = 1e6)
    power        = "^" ["-"|"+"] digit{1,4} | ["⁺"|"⁻"] supdigit{1,4}

A multiplier glued to a ten WITHOUT a power is no exponent here: it is the product production's (product.py), judged
by scan.scan_word with both operands (r7 round 3: the ten carve-out that refused only that one operand shape is gone).
"""
from __future__ import annotations

import re

from app.evidence.numgrammar.alphabet import (
    CARET,
    DIGITS,
    EXP_MARKERS,
    HYPHEN,
    MULTIPLIERS,
    SEPARATORS,
    SIGNS,
    SUP_CHARS,
    SUP_DIGITS,
    SUP_MINUS,
    SUP_SIGNS,
    SUP_TO_ASCII,
    char_class,
)
from app.evidence.numgrammar.body import MAX_DIGITS, MIDDOT, exp_int, read_body, to_float
from app.evidence.numgrammar.body import mantissa as read_mantissa

_DIG = re.compile("[0-9]+")
_SUPDIG = re.compile(char_class(SUP_DIGITS, negate=False) + "+")


def read_power(w: str, k: int, n: int) -> tuple[int | None, int] | None:
    """The power starting at w[k]: ``^[sign]digits`` or ``[⁺⁻]supdigits`` -> (exponent or None when out of range,
    end index); None when w[k:] is not a power."""
    c = w[k]
    if c == CARET:
        p, neg = k + 1, False
        if p < n and w[p] in SIGNS:
            neg, p = w[p] in HYPHEN, p + 1
        m = _DIG.match(w, p)
        return None if m is None else (exp_int(m.group(), neg), m.end())
    if c in SUP_CHARS:
        p, neg = k, False
        if c in SUP_SIGNS:
            neg, p = c == SUP_MINUS, p + 1
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
    if c != CARET:
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
            neg, p = w[p] in HYPHEN, p + 1
        m = _DIG.match(w, p)
        if m is None:
            return None
        e, end = exp_int(m.group(), neg), m.end()
    elif c in MULTIPLIERS and w[j + 1:j + 3] == "10":
        pw = read_power(w, j + 3, n) if j + 3 < n else None
        if pw is None:
            return None                                 # no power: the product production judges it (product.py)
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
        sign = "-" if c in HYPHEN else ""
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
        elif mant is None and seps and _exponent(w, j, n, sign, "1") is not None:
            return j, None, "groups", "", ""      # r7 round 2: a thousands group never carries an exponent
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


__all__ = ["read_power", "read_token"]
