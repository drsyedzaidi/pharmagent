"""The productions of the NEIGHBOUR-SHAPE SWEEP, written again independently of the reader (r7 writer 3; r7 round
2 split them out of ``evidence_r6_neighbour_spec`` to keep each file under 400 lines; r7 round 4 moved the number
words to ``evidence_r6_neighbour_numbers``). See ``evidence_r6_neighbour_spec`` for the oracle and the harness."""

from __future__ import annotations

import re

from app.evidence.numgrammar import tokens
from tests import evidence_r6_pins as P
from tests import evidence_r6_product_spec as X
from tests.evidence_r6_neighbour_numbers import (  # noqa: F401  (re-exported: the oracle reads the spec here)
    CLOSE,
    DIG,
    LET,
    OPEN,
    QUAL,
    STARTERS,
    TRAIL,
    _bare_power,
    _has_tokens,
    _lead,
    _number_end,
    _pure,
    _raised,
    _ratio_led,
    _split_decimal,
    lead_letter,
    number,
)

PUNCT = OPEN | CLOSE | frozenset(",;:.'/=")
OPL = frozenset("eExX")
SUP = frozenset("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
MOD = frozenset("+-*×·^±%‰") | SUP | QUAL
MULT = frozenset("xX*×·")
SEPS = frozenset(".,'")
LIST = frozenset(",;")                                     # r7 round 2: a list mark between two numbers
GAPS = re.compile("[^ \t\n\r]+")


def _alt(terms: frozenset[str]) -> str:
    return "(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + ")"


# r7 round 2: a unit symbol (the unit production's atom, from the SECOND copy of its terminals), the power of ten, the
# one-letter spaced exponent and the unit exponent of a spaced unit symbol ('h-1', 'm2')
ATOM = "(?:" + _alt(P.UNIT_PREFIX_PIN) + "?" + _alt(P.UNIT_BASE_PIN) + "|" + _alt(P.UNIT_PREFIX_PIN) \
    + _alt(P.UNIT_PREFIXED_BASE_PIN) + ")"
_ATOM = re.compile(ATOM)
_UNIT_LED = re.compile(ATOM + "(?=$|[/·×⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻)\\],;:.'^+-])")
_POWER = re.compile("10(?:\\^[-+]?[0-9]{1,4}|[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,4})")
_ONE_LETTER = re.compile("(.)([-+]?[0-9]{1,4})")
_UNIT_EXP = re.compile("[-+][0-9]{1,2}|[23]")


def fold(s: str) -> str:
    return "".join(chr(P.EQUIV_PIN.get(ord(c), ord(c))) for c in s)


def _let(w: str, i: int) -> bool:
    return 0 <= i < len(w) and w[i] in LET


def _core(w: str) -> tuple[int, int]:
    lo, hi = 0, len(w)
    while lo < hi and w[lo] in OPEN:
        lo += 1
    while hi > lo and w[hi - 1] in TRAIL:
        hi -= 1
    return lo, hi


def lone_op(w: str, i: int) -> bool:
    if w[i] not in OPL or _let(w, i - 1) or _let(w, i + 1):
        return False
    if _core(w) == (i, i + 1):
        return True
    left = i > 0 and (w[i - 1] in MOD or w[i - 1] in DIG or w[i - 1] == "=")
    right = i + 1 < len(w) and (w[i + 1] in MOD or w[i + 1] in DIG or w[i + 1] in "=([")
    return left or right


def _joined(w: str, i: int) -> bool:
    c = w[i]
    if c in "+-±%‰":
        return _let(w, i - 1) or _let(w, i + 1)
    if c == "*":
        a, b = i, i + 1
        while a > 0 and w[a - 1] == "*":
            a -= 1
        while b < len(w) and w[b] == "*":
            b += 1
        return _let(w, a - 1) or _let(w, b)
    if c in "×·":
        return _let(w, i - 1) and _let(w, i + 1)
    if c in SUP:
        a = i
        while a > 0 and w[a - 1] in SUP:
            a -= 1
        return _let(w, a - 1)
    if c == "^":
        return _let(w, i - 1) and (_let(w, i + 1) or w[i + 1:i + 2] in ("+", "-"))
    return False


def prose(w: str) -> bool:
    return not any(c in DIG for c in w) and all(
        not lone_op(w, i) if c in OPL else (c not in MOD or _joined(w, i)) for i, c in enumerate(w))


def unit_powers(w: str) -> bool:
    """r7 round 2: every caret or superscript run follows a unit symbol, or a stretch (letters, digits, decimal marks)
    that follows a slash (a unit denominator)."""
    i = 0
    while i < len(w):
        if w[i] != "^" and w[i] not in SUP:
            i += 1
            continue
        a = i
        while a > 0 and (w[a - 1] in LET or w[a - 1] in DIG or w[a - 1] in ".,"):
            a -= 1
        if not (a > 0 and w[a - 1] == "/") and (a == i or not _ATOM.fullmatch(w[a:i])):
            return False
        i += 1
        while w[i:i + 1] in SUP and w[i - 1] in SUP and i < len(w):
            i += 1
    return True


def spaced_exp(w: str) -> bool:
    """r7 round 2: one letter, a sign and digits that is no unit symbol with a unit exponent ('é-3'; never 'h-1')."""
    lo, hi = _core(w)
    m = _ONE_LETTER.fullmatch(w[lo:hi])
    return m is not None and m.group(1) in LET and not (_ATOM.fullmatch(m.group(1)) and _UNIT_EXP.fullmatch(m.group(2)))


def _in_name(w: str, j: int) -> bool:
    """r7 round 4: the letter or allele star w[j] continues a name: a label digit on its left ('CYP2E1', 'CYP2D6*4'),
    or for the star a slash after such a digit ('CYP2C19*1/*17')."""
    left = w[j - 1] if j > 0 else ""
    return left in DIG or (w[j] == "*" and left == "/" and j > 1 and w[j - 2] in DIG)


def scaled_at(w: str, j: int) -> bool:
    """r7 round 4: w[j] is the SCALE production on the PRINTED text: a multiplier (x X * × ·) before a digit or a signed
    digit, no label digit on its left (the times sign and the middle dot wherever they stand), or a lone e E x X."""
    c = w[j]
    if c not in MULT and c not in OPL:
        return False
    k = j + 1 if w[j + 1:j + 2] in ("+", "-") else j
    number_after = k + 1 < len(w) and w[k + 1] in DIG
    if number_after and c in MULT:
        return c in "×·" or not _in_name(w, j)
    return c in OPL and lone_op(w, j) and not (number_after and _in_name(w, j))


def label(w: str) -> bool:
    i = _lead(w)
    return i < len(w) and w[i] in LET and not any(scaled_at(w, j) for j in range(i, len(w))) \
        and prose("".join("q" if c in DIG else c for c in w[i:]))


def glue(run: str) -> bool:
    """The text between the tokens of one word, or after its last one (a unit's boundary included), is inert:
    punctuation around one prose stretch whose digits read as letters ('', ')', '-fold', ',44.83'; never ',E', '(x)')."""
    a, b = 0, len(run)
    while a < b and run[a] in PUNCT:
        a += 1
    while b > a and run[b - 1] in PUNCT:
        b -= 1
    mid = run[a:b]
    if any(scaled_at(mid, j) for j in range(len(mid))):
        return False                                       # r7 round 4: the scale production on the printed text
    return prose("".join("q" if c in DIG else c for c in mid))


def plain_number(words: list[str], k: int) -> bool:
    return 0 <= k < len(words) and number(words[k]) and not any(
        t.shape == "ratio" for t in tokens(words[k], mode="cell")) and not _bare_power(words[k]) \
        and not X.refused_amount(words, k, prose)


def _number_ok(words: list[str], k: int, side: int, listed: bool, kind: set[str], tight: bool = True,
               adjacent: bool = True) -> bool:
    w = words[k]
    if X.power_word(words, k) and not set(kind) & {"ratio", "power"}:
        return True                                        # r7 round 3: a scientific number's power word is a number
    if not number(w):
        return False
    if adjacent and X.refused_amount(words, k, prose):
        return False                                       # r7 round 3: beside a refused product's amount
    if listed:
        return True
    if _bare_power(w) or "power" in kind or ("ratio" in kind and side < 0):
        return False
    if tight and ((side > 0 and _raised(w)) or (side < 0 and "raised" in kind and w[-1] in DIG)):
        return False                                       # r7 round 4: a raised ten after a number, a dropped multiplier
    return not (tight and side > 0 and _ratio_led(w))


def _infix_ok(words: list[str], k: int, side: int) -> bool:
    j = k + side
    mult = words[k] in MULT and len(words[k]) == 1
    if mult:                                               # r7 round 3: the product production (evidence_r6_product_spec)
        return X.product_ok(words, k - 1, j) if side > 0 else X.product_ok(words, X.count_index(words, j, prose), k + 1)
    if not 0 <= j < len(words):
        return False
    if plain_number(words, j):
        return True
    return (not (mult and side > 0) and not any(c in DIG for c in words[j]) and not all(c in PUNCT for c in words[j])
            and prose(words[j]) and plain_number(words, j + side))


def _word_ok(words: list[str], k: int, side: int, adjacent: bool, split: bool, assigned: bool, kind: set[str],
             listed: bool, tight: bool = True) -> bool:
    w = words[k]
    lo, hi = _core(w)
    if side > 0:
        if adjacent and w[:1] in MULT and len(w) > 1:
            return False                                   # '12 xenon', '12 x3/4': the operand of a multiplier
        if "ratio" in kind and (_UNIT_LED.match(w[lo:hi]) or w[lo:lo + 1] in MULT):
            return False                                   # a ratio is dimensionless
        if spaced_exp(w):
            return False
        if adjacent and w[lo:lo + 1] in ("/", ":") and not any(c in DIG for c in w) and not _UNIT_LED.match(w[lo + 1:hi]):
            return False                                   # r7 round 4: '12 /lS'; '12 /kg' is a unit
    if re.fullmatch("0|[1-9][0-9]{0,39}", w):
        return not split and _number_ok(words, k, side, listed, kind, tight, adjacent)
    if side < 0 and (assigned or w.endswith("=")) and _variable(w):
        return True
    if any(c in DIG for c in w):
        if lead_letter(w):
            if "ratio" in kind and side < 0 and _number_end(w) and not listed:
                return False
            return _has_tokens(w) or (label(w) and unit_powers(w))
        if re.fullmatch(r"\[[1-9][0-9]?,[0-9]{1,2}\]", w):
            return True                                    # a bracket list: masked, a label
        return not split and _number_ok(words, k, side, listed, kind, tight, adjacent)
    if prose(w):
        if adjacent and _power_across(words, k, side, kind):
            return False                                   # r7 round 4: '1.2 times 10^3'
        return unit_powers(w)
    if w[lo:hi] == "±":
        return True
    if hi - lo == 1 and w[lo] in "%‰" and (side > 0 or (k >= 1 and number(words[k - 1]))):
        return True
    if side < 0 and _qualifier(w):
        return True
    mult = len(w) == 1 and w in "xX*×·"
    if mult and "ratio" in kind:
        return False                                       # a ratio is never an operand
    infix = w in ("-", "~") or mult or (side > 0 and all(c in QUAL for c in w))
    return infix and adjacent and _infix_ok(words, k, side)


def _power_across(words: list[str], k: int, side: int, kind: set[str]) -> bool:
    """r7 round 4: the prose word *k* stands between a plain mantissa (T after it, or the number word before it) and a
    power of ten with no mantissa of its own."""
    if side > 0:
        return "plain" in kind and k + 1 < len(words) and _bare_power(words[k + 1])
    w = words[k - 1] if k >= 1 else ""
    if "power" not in kind or not number(w) or (w.endswith("-") and w[-2:-1] not in DIG | {"%", "‰"}):
        return False                                       # a dash ends a number word only right after its digits
    got = [t for t in tokens(words[k - 1][:-1] if words[k - 1].endswith("-") else words[k - 1], mode="cell")]
    return bool(got) and got[-1].value is not None and (got[-1].shape in X.PLAIN or (
        got[-1].shape == "style_dependent" and bool(got[-1].decimal)))


def _variable(w: str) -> bool:
    """r7 round 3: a name before '=': one letter after openers, or a label whose powers follow unit symbols."""
    i = 0
    while i < len(w) and w[i] in OPEN:
        i += 1
    return i < len(w) and w[i] in LET and (i + 1 == len(w) or (label(w) and unit_powers(w)))


def _qualifier(w: str) -> bool:
    k = len(w)
    while k > 0 and w[k - 1] in QUAL | frozenset("=([:;*+-"):
        k -= 1
    head, run = w[:k], w[k:]
    if not any(c in QUAL for c in run):
        return False
    a, b = 0, len(head)
    while a < b and head[a] in PUNCT:
        a += 1
    while b > a and head[b - 1] in PUNCT:
        b -= 1
    return prose("".join("q" if c in DIG else c for c in head[a:b]))


def target_kind(literal: str) -> set[str]:
    """r7 round 2: what the target is (its window rules differ): a ratio, a power of ten standing alone, or plain; r7
    round 4: a pure power in e-notation is a power, an unsigned int that is a raised ten is 'raised', and one int or
    decimal token is 'plain' (a mantissa a spelled-out multiplier may scale)."""
    if re.fullmatch("[0-9]+ ?/ ?[0-9]+", literal):
        return {"ratio"}
    got = tokens(literal, mode="cell")
    if _POWER.fullmatch(literal) or (len(got) == 1 and _pure(got[0])):
        return {"power"}
    out = {"raised"} if _raised(literal) else set()
    return out | ({"plain"} if len(got) == 1 and got[0].value is not None and got[0].shape in X.PLAIN else set())


_GROUP = re.compile("\\[[1-9][0-9]{0,2}(?:[,-][1-9][0-9]{0,2})*\\]")


def _masked(words: list[str]) -> list[str]:
    """The words as the reader sees them: a bracket group that is a citation (after a word that ends in a letter) or a
    list (with a comma) is masked with letters (productions.py: a non-value token, a label to its neighbours)."""
    out = list(words)
    for i, w in enumerate(words):
        for m in _GROUP.finditer(w):
            if m.end() < len(w) and w[m.end()] in DIG:
                continue
            s0, prev = m.start(), (words[i - 1].rstrip(")],;:.'") if i else "")
            glued = s0 > 0 and (w[s0 - 1] in LET or (w[s0 - 1] in TRAIL and s0 > 1 and w[s0 - 2] in LET))
            if glued or (s0 == 0 and ("," in m.group() or (prev and prev[-1] in LET and prev not in ("x", "X", "e", "E")))):
                out[i] = out[i][:s0] + "[" + "c" * (m.end() - s0 - 1) + out[i][m.end():]   # r7 round 3: the opener stays
    return out


def spec_inert(text: str, first: int, last: int, kind: set[str] = frozenset()) -> bool:
    """The window of the number over words first..last of *text* (folded) is inert."""
    words = _masked([m.group() for m in GAPS.finditer(text)])
    if _split_decimal(words, first, last):
        return False
    dash = words[last].endswith("-") and words[last][-2:-1] in DIG | {"%", "‰"}   # T's own word: '5.0-' only
    for side, k0 in ((-1, first - 1), (1, last + 1)):
        k, split, assigned, ratio_mark, listed = k0, False, False, False, False
        edge = words[first][0] if side < 0 else words[last][-1]
        while 0 <= k < len(words) and all(c in PUNCT for c in words[k]):
            split = split or (edge in DIG and all(c in SEPS for c in words[k]))
            assigned = assigned or "=" in words[k]
            ratio_mark = ratio_mark or words[k] in ("/", ":")
            listed = listed or any(c in LIST for c in words[k])
            k += side
        if not 0 <= k < len(words):
            if side > 0 and dash:
                return False
            continue
        if side > 0 and dash:                              # r7 round 4: a number or a unit production after the dash
            ok = k == k0 and (number(words[k]) or _UNIT_LED.match(words[k][_core(words[k])[0]:]) is not None)
        elif ratio_mark and any(c in DIG for c in words[k]) and not lead_letter(words[k]):
            ok = False                                     # '1 : 2': a ratio mark between two numbers
        elif ratio_mark and side > 0 and lead_letter(words[k]) and not _UNIT_LED.match(words[k][_core(words[k])[0]:]):
            ok = False                                     # r7 round 4: '12 / lS'; '12 / kg' is a unit
        else:
            facing = words[k][-1 if side < 0 else 0]
            listed = listed or (words[k] if side < 0 else words[last])[-1] in LIST
            tight = side < 0 or words[last][-1] in DIG
            ok = _word_ok(words, k, side, k == k0, split and facing in DIG, assigned, set(kind), listed, tight)
        if not ok:
            return False
    return True
