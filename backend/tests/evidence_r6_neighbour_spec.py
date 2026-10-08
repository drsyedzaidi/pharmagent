"""The NEIGHBOUR-SHAPE SWEEP of r7 writer 3: an independent statement of the neighbour-word allow-list, and its harness.

For a canonical expression and its target number T, a word W is inserted one or two words before or after T (a gap on
each side). ``spec_inert`` says, from the productions alone and written again here (never imported from the reader),
whether every word of T's window is an accepted inert production. The oracle (``check``):

    reader keeps T resolved with its literal and value   ->  the specification must call the window inert
    the specification calls the window NOT inert         ->  T is unresolved (no resolved token holds T's characters)
    any resolved token over T's characters               ->  it is T itself: never a different value or literal

A window the specification calls inert may still leave T unresolved when a multi-word FORM of the reader takes T (a mixed
number ``150 1/5``, a ten after a multiplier): that is fail closed and is listed by ``strict`` for review, never an error.

Productions (the definitions of glue.py and neighbours.py, restated): PUNCT words ``( [ ) ] , ; : . ' / =`` are skipped;
a PROSE word holds no digit and every modifier in it is joined to a letter (sign, plus-minus, percent and a star run on one
side; times sign and middle dot on both; a superscript run after a letter; a caret after a letter and before a letter or
a sign; never a qualifier) and has no lone
operator letter (e E x X with no letter beside it that touches a modifier, a digit, '=' or (on its right) an opener, or is
the whole word); a LABEL is led by a letter that is no lone operator letter and is prose with its digits read as letters;
a NUMBER is not led by a letter and reads alone as resolved numbers (a dangling dash aside); then the qualifier, dispersion,
percent, variable, infix, citation-pair and range-end productions of neighbours.py.
"""
from __future__ import annotations

import itertools
import re
from typing import NamedTuple

from app.evidence.numgrammar import tokens
from tests import evidence_r6_oracle as O
from tests import evidence_r6_pins as P

DIG = frozenset("0123456789")
LET = O._ASCII_LETTERS | O._GREEK | O._ACCENTED
OPEN, CLOSE = frozenset("(["), frozenset(")]")
PUNCT = OPEN | CLOSE | frozenset(",;:.'/=")
TRAIL = CLOSE | frozenset(",;:.'")
OPL = frozenset("eExX")
SUP = frozenset("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
QUAL = frozenset("<>≤≥~≈")
MOD = frozenset("+-*×·^±%‰") | SUP | QUAL
STARTERS = OPEN | QUAL | frozenset("=±")
SEPS = frozenset(".,'")
GAPS = re.compile("[^ \t\n\r]+")


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


def _lead(w: str) -> int:
    i = 0
    while i < len(w) and (w[i] in OPEN or w[i] in STARTERS):
        i += 1
    return i


def lead_letter(w: str) -> bool:
    return _lead(w) < len(w) and w[_lead(w)] in LET


def label(w: str) -> bool:
    i = _lead(w)
    return i < len(w) and w[i] in LET and not lone_op(w, i) and prose("".join("q" if c in DIG else c for c in w[i:]))


def glue(run: str) -> bool:
    """The text between the tokens of one word, or after its last one (a unit's boundary included), is inert:
    punctuation around one prose stretch whose digits read as letters ('', ')', '-fold', ',44.83'; never ',E', '(x)')."""
    a, b = 0, len(run)
    while a < b and run[a] in PUNCT:
        a += 1
    while b > a and run[b - 1] in PUNCT:
        b -= 1
    return prose("".join("q" if c in DIG else c for c in run[a:b]))


def number(w: str) -> bool:
    if lead_letter(w) or not any(c in DIG for c in w):
        return False
    body = w[:-1] if w.endswith("-") else w
    got = tokens(body, mode="cell")
    return bool(got) and all(t.value is not None for t in got)


def _has_tokens(w: str) -> bool:
    return any(t.value is not None for t in tokens(w[:-1] if w.endswith("-") else w, mode="cell"))


def _infix_ok(words: list[str], k: int, side: int) -> bool:
    j = k + side
    if 0 <= j < len(words) and number(words[j]):
        return True
    return (0 <= j < len(words) and not any(c in DIG for c in words[j]) and not all(c in PUNCT for c in words[j])
            and prose(words[j]) and 0 <= j + side < len(words) and number(words[j + side]))


def _word_ok(words: list[str], k: int, side: int, adjacent: bool, split: bool, assigned: bool) -> bool:
    w = words[k]
    if side < 0 and (assigned or w.endswith("=")) and _lead_open_letter(w):
        return True
    if any(c in DIG for c in w):
        if lead_letter(w):
            return _has_tokens(w) or label(w)
        return (not split and number(w)) or re.fullmatch(r"\[[1-9][0-9]?,[0-9]{1,2}\]", w) is not None
    if prose(w):
        return True
    lo, hi = _core(w)
    if w[lo:hi] == "±":
        return True
    if hi - lo == 1 and w[lo] in "%‰" and (side > 0 or (k >= 1 and number(words[k - 1]))):
        return True
    if side < 0 and _qualifier(w):
        return True
    mult = len(w) == 1 and w in "xX*×·"
    if side < 0 and mult and re.match("[+-]?10(?!0)", words[k + 1]):
        return False                                       # 'x 10^3' left of the number: a scale, never a product
    infix = w in ("-", "~") or mult or (side > 0 and all(c in QUAL for c in w))
    return infix and adjacent and _infix_ok(words, k, side)


def _lead_open_letter(w: str) -> bool:
    i = 0
    while i < len(w) and w[i] in OPEN:
        i += 1
    return i < len(w) and w[i] in LET


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


def spec_inert(text: str, first: int, last: int) -> bool:
    """The window of the number over words first..last of *text* (folded) is inert."""
    words = [m.group() for m in GAPS.finditer(text)]
    dash = words[last].endswith("-")
    for side, k0 in ((-1, first - 1), (1, last + 1)):
        k, split, assigned = k0, False, False
        edge = words[first][0] if side < 0 else words[last][-1]
        while 0 <= k < len(words) and all(c in PUNCT for c in words[k]):
            split = split or (edge in DIG and all(c in SEPS for c in words[k]))
            assigned = assigned or "=" in words[k]
            k += side
        if not 0 <= k < len(words):
            if side > 0 and dash:
                return False
            continue
        if side > 0 and dash:
            ok = k == k0 and (number(words[k]) or (lead_letter(words[k]) and prose(words[k])))
        else:
            facing = words[k][-1 if side < 0 else 0]
            ok = _word_ok(words, k, side, k == k0, split and facing in DIG, assigned)
        if not ok:
            return False
    return True


# ------------------------------------------------------------------------------------------------- the harness ---
class Target(NamedTuple):
    text: str
    literal: str
    mode: str = "page"


TARGETS = (
    Target("Cmax was 12.5 ng/mL daily", "12.5"), Target("dose 150 mg", "150"), Target("k was 1.2e-3 per h", "1.2e-3"),
    Target("count 1.2 x 10^3 cells", "1.2 x 10^3"), Target("CL 5.3 ± 0.4 L/h", "5.3"),
    Target("CL 5.3 ± 0.4 L/h", "0.4"), Target("Cmax ( 12.5 ) ng", "12.5"), Target("n = 12", "12"),
    Target("change −20 mg", "−20"), Target("range 12 – 15 mg", "12"), Target("range 12 – 15 mg", "15"),
    Target("range 5.0- 12.4 mg", "5.0"), Target("conc <0.5 ng/mL", "0.5"), Target("response 45% of", "45"),
    Target("ratio 12/15 of", "12/15"), Target("12.5 (3.1)", "12.5", "cell"),
)
ALPHABET = "+-*×·^²⁻<~±%015eExXq([)].,;:/='"
SAMPLES = ("mg", "ng/mL", "h-1", "(\u03bcL/min/10^6", "h^-1", "m²", "ng·h/mL", "%CV", "L/h", "°C", "μg", "daily", "once-daily",
           "-fold", "e.g.", "i.e.", "Cmax*", "*P", "exposure", "and", "AUC0-24", "CYP2D6", "t1/2", "Cmax,ss",
           "AUC0–∞", "C24h", "E2", "x10^3", "e-3", "E+03", "(x10^3)", "×10³", "[1,2]", "[46]", "p<", "<=",
           "n=12", "2x", "12.5", "(3.1)", "-15", "3/4", "10^3", "1.2e3", "x10", "E-3.", "e3,")


def vocabulary(max_len: int = 3) -> list[str]:
    words = ["".join(t) for n in range(1, max_len + 1) for t in itertools.product(ALPHABET, repeat=n)]
    return words + [s for s in SAMPLES if s not in words]


def positions(t: Target) -> list[tuple[int, int, int]]:
    """(insertion offset, word index of T's first word after the insertion, side) for one and two words before and
    after T. The inserted word gets a gap on each side."""
    text = t.text
    s = text.index(t.literal)
    e = s + len(t.literal)
    spans = [(m.start(), m.end()) for m in GAPS.finditer(text)]
    fi = next(i for i, (a, b) in enumerate(spans) if a <= s < b)
    li = next(i for i, (a, b) in enumerate(spans) if a < e <= b)
    out = [(spans[fi][0], fi + 1, -1), (spans[li][1], fi, 1)]
    if fi >= 1:
        out.append((spans[fi - 1][0], fi + 1, -2))
    if li + 1 < len(spans):
        out.append((spans[li + 1][1], fi, 2))
    return out


def variant(t: Target, at: int, word: str) -> tuple[str, int]:
    """The text with *word* inserted at offset *at* (a gap on each side) and the new start of T's literal."""
    s = t.text.index(t.literal)
    v = t.text[:at] + " " + word + " " + t.text[at:]
    return v, s + (len(word) + 2 if at <= s else 0)


_MULT_POWER = re.compile("[xX*×·]10(?:\\^([-+]?[0-9]{1,4})|([⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,4}))")
_SUP_ASCII = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def sci_join(t: Target, at: int, word: str) -> float | None:
    """The value of the multiplier-power production when *word* ('x10^3', '×10³') is inserted right after a plain number
    T that ends its word: the power is parsed INTO the number (the one accepted way a word beside a number changes its
    value); None otherwise."""
    m = _MULT_POWER.fullmatch(fold(word))
    s = t.text.index(t.literal)
    e = s + len(t.literal)
    if m is None or at != e or re.fullmatch("[−-]?[0-9]+(?:\\.[0-9]+)?", t.literal) is None:
        return None
    return float(fold(t.literal)) * 10.0 ** int((m.group(1) or m.group(2)).translate(_SUP_ASCII))


def check(t: Target, at: int, fi: int, word: str) -> tuple[str, str]:
    """('', '') when the variant reads as the oracle says, ('error', why) on a violation, ('strict', why) when an inert
    window still leaves T unresolved (a form took it: fail closed, listed for review)."""
    v, s = variant(t, at, word)
    e = s + len(t.literal)
    want = next(x for x in tokens(t.text, mode=t.mode) if x.start == t.text.index(t.literal))
    got = [x for x in tokens(v, mode=t.mode) if x.start < e and s < x.end]
    resolved = [x for x in got if x.value is not None]
    joined = sci_join(t, at, word)
    if joined is not None:                     # T and the power are ONE token, or T is unresolved: never T alone
        ok = all((x.start, x.end) == (s, e + 1 + len(word)) and abs(x.value - joined) <= 1e-9 * abs(joined)
                 for x in resolved)
        return ("", "") if ok else ("error", f"a power not parsed into T: {[(v[x.start:x.end], x.value) for x in resolved]}")
    if any((x.start, x.end, x.value) != (s, e, want.value) for x in resolved):
        return "error", f"a different value or literal: {[(v[x.start:x.end], x.value) for x in resolved]}"
    words = [m for m in GAPS.finditer(v)]
    first = next(i for i, m in enumerate(words) if m.start() <= s < m.end())
    last = next(i for i, m in enumerate(words) if m.start() < e <= m.end())
    inert = spec_inert(fold(v), first, last)
    if resolved and not inert:
        return "error", "T keeps its value beside a word the specification refuses"
    if not resolved and inert:
        return "strict", "an inert window, T unresolved"
    return "", ""


def sweep(words: list[str], targets: tuple[Target, ...] = TARGETS) -> tuple[list[str], list[str]]:
    """(errors, strict cases) of every word at every position of every target."""
    errors: list[str] = []
    strict: list[str] = []
    for t in targets:
        for at, fi, pos in positions(t):
            for w in words:
                kind, why = check(t, at, fi, w)
                if kind == "error" and len(errors) < 50:
                    errors.append(f"{t.text!r}/{t.literal!r} {pos:+d} {w!r}: {why}")
                elif kind == "strict" and len(strict) < 2000:
                    strict.append(f"{t.text!r}/{t.literal!r} {pos:+d} {w!r}")
    return errors, strict


def _task(args: tuple[int, list[str]]) -> tuple[int, list[str], list[str]]:
    i, chunk = args
    errors, strict = sweep(chunk)
    return len(chunk), errors, strict


def full_run(workers: int, chunk: int = 1500) -> dict:
    """Every word of the vocabulary at every position of every target, on a spawn pool."""
    import multiprocessing as mp
    import time
    words = vocabulary()
    tasks = [(i, words[i:i + chunk]) for i in range(0, len(words), chunk)]
    t0, n, errors, strict = time.perf_counter(), 0, [], []
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, bad, st in pool.imap_unordered(_task, tasks):
            n += k
            errors += bad
            strict += st
    state = O.repo_state()
    variants = n * sum(len(positions(t)) for t in TARGETS)
    return {"words": n, "variants": variants, "errors": len(errors), "failures": errors[:50], "strict": len(strict),
            "wall_seconds": round(time.perf_counter() - t0, 1), "head": state["head"], "dirty": state["dirty"]}
