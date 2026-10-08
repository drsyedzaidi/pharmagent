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

from app.evidence.numgrammar import product as P
from app.evidence.numgrammar.alphabet import (
    ASCII_LETTERS,
    DECIMAL_MARKS,
    DEGREE,
    DIGITS,
    HYPHEN,
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
    SLASH,
    STARTERS,
    TRAILING,
    char_class,
)
from app.evidence.numgrammar.alphabet import STAR as STAR_MARK  # noqa: I001  (STAR is an action code here)
from app.evidence.numgrammar.body import MAX_DIGITS, MIDDOT, read_body, to_float
from app.evidence.numgrammar.glue import glue, prefix
from app.evidence.numgrammar.read import read_power, read_token
from app.evidence.numgrammar.tok import MAX_PARTS, REASON_SHAPE, Tok
from app.evidence.numgrammar.units import unit_end

_DIG = re.compile("[0-9]+")
_PLAIN_INT = re.compile("0|[1-9][0-9]{0,39}")
_PLAIN_DEC = re.compile("(0|[1-9][0-9]{0,38})(" + char_class(DECIMAL_MARKS, negate=False) + ")([0-9]{1,39})")
RATIO_INT = re.compile("0|[1-9][0-9]{0,5}")                 # one side of a ratio (r7, owner decision 6)
OK, PCT, STAR, TAIL, UNIT, BAD = 0, 1, 2, 3, 4, -1
TAIL_NONE, TAIL_MULT, TAIL_STAR = 0, 1, 2
_OPEN_SHAPES = frozenset({"int", "grouped", "style_dependent"})   # without a decimal mark, a separator may go on
COMPOUND_MIN = 3        # letters of the shortest word a number's dash joins in the corpus ('3-day'; units aside)


class Scan(NamedTuple):
    """kind: ``numeric`` (>= 1 resolved token), ``rejected`` (a numeric-led word the grammar refuses: ``why``,
    ``parts`` = the resolved numbers before the failure), ``label`` (digits inside a label: no tokens)."""
    kind: str
    toks: tuple[Tok, ...] = ()
    tail: int = TAIL_NONE          # a glued multiplier (x X × ·) or a trailing star run ends the word
    why: str = ""
    parts: tuple[float, ...] = ()
    unit: str = ""                 # a glued unit that ends the word ('mg' of '150mg'): the next word may split it
    dash: bool = False             # r7 writer 3: a range dash ends the word right after its last number ('5.0-')
    need_unit: bool = False        # r7 round 3: a glued product's power-of-ten amount ends the word ('2x100'): the
    #                                next word must be a unit production ('2x100 mg'), product.py


LABEL = Scan("label")


def _after_mult(w: str, e: int, n: int, p: str, x: str) -> tuple[int, int, str]:
    """A glued multiplier (x X × · *) after a token: at the word end (the operand is the next word) or before digits
    (the amount of a product: scan_word judges the two operands by the product production, product.py)."""
    if x == "":
        return TAIL, e + 1, ""
    if x in DIGITS:
        return OK, e, ""
    return BAD, e, "glued_letters" if x in LETTERS else "glued_symbol"


def _product(w: str, count: Tok, mult: str, amount: tuple[str, str, str], act: int, end: int) -> tuple[str, bool]:
    """(refusal, the next word must be a unit) of the glued ``count MULT amount`` whose amount token ends at *end*:
    the product production (product.py); a middle dot glued between digits is the separator production, so one
    before a ten ('0·100', '12·105') is no product."""
    if mult == MIDDOT:
        return P.name(amount[1]), False
    ends = act == OK and all(c in TRAILING for c in w[end:])
    why = P.fault(count, amount, act == UNIT or ends)
    return why, not why and act != UNIT and P.power_of_ten(amount[1])


def _scale(t: Tok) -> bool:
    """A power of ten standing as a token: a bare power ('10^3'), or a scientific token whose mantissa is one."""
    return t.shape == "bare_power" or (t.shape == "sci" and t.parts[:1] == (1.0,))


def post(w: str, e: int, n: int, open_number: bool = False) -> tuple[int, int, str]:
    """The post-boundary of a token ending at *e*: ``(OK|PCT|STAR|TAIL|UNIT, index after the suffix or unit, "")`` or
    ``(BAD, e, reason)``. A letter that starts no complete unit production (units.py), ``=``, ``^``, ``+``, ``:`` +
    digit, ``/`` + digit, a second sign... glued to the digits reject the word."""
    if e >= n:
        return OK, e, ""
    p = w[e]
    x = w[e + 1] if e + 1 < n else ""
    if p in TRAILING:
        if x in DIGITS:
            return BAD, e, "glued_symbol"
        if open_number and p in SEPARATORS and x in LETTERS and unit_end(w, e + 1) < 0:
            return BAD, e, "glued_letters"      # r7 round 3: '1,OOO', '0.OS', '12.l': a separator that may go on into
        return OK, e, ""                        # digits the text layer spelled with look-alikes ('0.28,and' cannot)
    if p in PERCENTS:
        if x == "" or x in TRAILING or x in LETTERS or x in OPENERS_BR or x in HYPHEN:
            return PCT, e + 1, ""
        return BAD, e, "glued_symbol"
    if p == STAR_MARK:
        k = e + 1
        while k < n and w[k] == STAR_MARK:
            k += 1
        y = w[k] if k < n else ""
        if y == "" or y in TRAILING:
            return STAR, k, ""
        return _after_mult(w, e, n, p, y) if k == e + 1 and y in DIGITS else (BAD, e, "glued_symbol")
    if p == DEGREE or p in OPENERS_BR:
        return OK, e, ""
    if p in SLASH:
        return (OK, e, "") if x != "" and x not in DIGITS else (BAD, e, "glued_symbol")
    if p in HYPHEN:
        if x == "" or x in DIGITS or (x in LETTERS and _compound(w, e + 1)):
            return OK, e, ""
        return BAD, e, "glued_letters" if x in LETTERS else "glued_symbol"
    if p == PLUS_MINUS:
        return (OK, e, "") if x in DIGITS or x in SIGNS else (BAD, e, "glued_symbol")
    if p in MULTIPLIERS:
        return _after_mult(w, e, n, p, x)
    if p in LETTERS:
        u = unit_end(w, e)
        return (UNIT, u, "") if u > e else (BAD, e, "glued_letters")
    return BAD, e, "glued_symbol"


def _compound(w: str, i: int) -> bool:
    """r7 round 3: the letters after a number's dash join a compound only as a unit production ('24-h', '5-mg') or a
    word of COMPOUND_MIN letters or more ('10-fold', '3-day', '12-week'); a shorter run may be a look-alike of digits
    ('12-lS' for 12-15, '12-l'), so the word is refused."""
    if unit_end(w, i) > i:
        return True
    k = i
    while k < len(w) and w[k] in LETTERS:
        k += 1
    return k - i >= COMPOUND_MIN


def lead_power(w: str, i: int) -> int:
    """The index of the power (its caret or first superscript) when ``w[i:]`` starts with a power of ten (``10^3``,
    ``10⁻³``) that ends the way a glued token may end (``post``: the word end, a trailing closer not before a digit, an
    opener, a unit, a denominator, a percent or star suffix) and whatever follows is inert glue: ``10^3.``, ``10^3)``,
    ``10^3/mL``; -1 otherwise (``10^3x``, ``10^3-``, ``10^3.5``, ``10^3/5``). The multiplier-power production across a
    gap reads its power word with this (r7 round 2: '1.2 x 10^3.' reads like '1.2x10^3.')."""
    n = len(w)
    if w[i:i + 2] != "10" or i + 2 >= n:
        return -1
    pw = read_power(w, i + 2, n)
    if pw is None:
        return -1
    act, after, _ = post(w, pw[1], n)
    if act in (BAD, TAIL):
        return -1
    rest = w[after:] if act in (PCT, STAR, UNIT) else w[pw[1]:]
    return i + 2 if not rest or (_DIG.search(rest) is None and glue(rest)) else -1


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
    lead_i = 1 if w[0] in STARTERS else 0
    lead = w[lead_i] if lead_i < n else ""
    numeric_led = lead != "" and lead in INSIDE and lead not in ASCII_LETTERS
    toks: list[Tok] = []
    parts: list[float] = []
    i, prev_end, tail, unit, glue_from, need_unit = 0, -1, TAIL_NONE, "", 0, False
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
        if shape == "int" and not joiner and c in DIGITS and end + 1 < n and w[end] in SLASH and w[end + 1] in DIGITS \
                and RATIO_INT.fullmatch(canon):                         # unsigned: '+3/4' and '-3/4' are refused
            r_end, r_value, r_canon, r_parts = _ratio(w, end, n, canon)
            if r_value is None:
                return _rejected(r_canon, parts + list(r_parts)) if numeric_led else LABEL
            toks.append(Tok(off + i, off + r_end, r_value, "ratio", r_parts, _qualified(w, i), False, r_canon, ""))
            parts.append(r_value)
            i = prev_end = glue_from = r_end
            tail = TAIL_NONE
            continue
        act, after, why = post(w, end, n, not mark and shape in _OPEN_SHAPES)
        if act == BAD:
            return _rejected(why, parts + [value]) if numeric_led else LABEL
        if joiner and pre in MULTIPLIERS and toks:                    # r7 round 3: the product production
            why, need_unit = _product(w, toks[-1], pre, (shape, canon, mark), act, end)
            if why:
                return _rejected(why, parts + [value]) if numeric_led else LABEL
        ranged = joiner and pre in HYPHEN
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
    if len(toks) > 1 and any(_scale(t) for t in toks):      # r7 round 3: '1.2(10^3)', '10^3-10^6': a detached scale
        return _rejected("detached_exponent", parts) if numeric_led else LABEL
    rest = w[glue_from:]                                    # r7 writer 3: '12-x', '12(e)', '5/x'; a lone '-' is a range
    if rest and tail != TAIL_MULT and rest not in HYPHEN and not glue(rest):
        return _rejected("glued_symbol", parts) if numeric_led else LABEL
    return Scan("numeric", tuple(toks), tail, unit=unit, dash=rest in HYPHEN, need_unit=need_unit)


__all__ = ["LABEL", "RATIO_INT", "Scan", "lead_power", "post", "ratio_parts", "read_power", "read_token", "scan_word",
           "REASON_SHAPE"]
