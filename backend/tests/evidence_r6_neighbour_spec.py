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

r7 round 2 (the productions live in ``evidence_r6_neighbour_rules``): a number neighbour is never a bare power, a ten behind
a lone multiplier, a ratio part led by '/' or ':', or a fraction right after an integer that ends its word unless a list mark
separates them; a lone ':' or '/' between two numbers is no production; a word led by a multiplier right after the number,
a one-letter spaced exponent (unless a unit symbol with a unit exponent) and a word with a power that follows no unit symbol
are refused; a ratio target refuses a unit or multiplier after it, a multiplier before it and a number before it without a
list mark; a bracket group with a comma, or after a word ending in a letter, is masked (it is a label to its neighbours).
"""
from __future__ import annotations

import itertools
import re
from typing import NamedTuple

from app.evidence.numgrammar import tokens
from tests import evidence_r6_oracle as O
from tests.evidence_r6_neighbour_rules import (  # noqa: F401  (re-exported: the unit specification reads glue here)
    _POWER,
    GAPS,
    fold,
    glue,
    spec_inert,
    target_kind,
)


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
    Target("count 10^6 cells/mL", "10^6"), Target("ka 0.12 h-1 rate", "0.12"),       # r7 round 2: a power, a unit
    Target("dose 2 x 100 mg daily", "2"), Target("dose 2 x 100 mg daily", "100"),    # r7 round 3: the product
    Target("count 3.1 x 7 cells", "7"),
)
ALPHABET = "+-*×·^²⁻<~±%015eExXq([)].,;:/='"
SAMPLES = ("mg", "ng/mL", "h-1", "(\u03bcL/min/10^6", "h^-1", "m²", "ng·h/mL", "%CV", "L/h", "°C", "μg", "daily", "once-daily",
           "-fold", "e.g.", "i.e.", "Cmax*", "*P", "exposure", "and", "AUC0-24", "CYP2D6", "t1/2", "Cmax,ss",
           "AUC0–∞", "C24h", "E2", "x10^3", "e-3", "E+03", "(x10^3)", "×10³", "[1,2]", "[46]", "p<", "<=",
           "n=12", "2x", "12.5", "(3.1)", "-15", "3/4", "10^3", "1.2e3", "x10", "E-3.", "e3,",
           # r7 round 2: look-alikes of an exponent or a power, unit exponents, scale words, a label that continues
           "é-3", "è+03", "ε-03", "l0³", "lO³", "h⁻¹", "s⁻¹", "day⁻¹", "L-1", "M", "million", "AUC0-", "t1/", "mg/",
           "[12,13]", "(10^3)", "1/2,", "Cmax:12",
           # r7 round 3: product amounts and scales, look-alike groups, split decimals, scale words glued to a label
           "1e6", "10000", "1,000,000", "100", "1000", "x1000", "(E3)", "Count(E3)", "0.", "12.", "1,OOO",
           "OOO", "lS", ".OS", "mg[3]", "h[12]", "2.5", "-3",
           # r7 round 3 follow-up: a times sign or middle dot glued before digits in a label word (no label), unit glue
           "AUC×1000", "Count·5", "ng×h", "CYP2D6*4")


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


_MANTISSA_MULT = re.compile("([(\\[<>≤≥~≈=±]*)([-+]?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?)[xX*×·]")   # starters, mantissa


def sci_join(t: Target, at: int, word: str) -> tuple[float, int, int] | None:
    """(value, start, end) of the multiplier-power production when the inserted *word* joins T: a power word ('x10^3',
    '×10³') right after a plain number T that ends its word, or (r7 round 2) a plain number with a glued multiplier
    ('5x', '-5x', '2×') right before a power of ten T: the power and the number are ONE token (the one accepted way a
    word beside a number changes its value); None otherwise."""
    s = t.text.index(t.literal)
    e = s + len(t.literal)
    m = _MULT_POWER.fullmatch(fold(word))
    if m is not None and at == e and re.fullmatch("[−-]?[0-9]+(?:\\.[0-9]+)?", t.literal) is not None:
        return float(fold(t.literal)) * 10.0 ** int((m.group(1) or m.group(2)).translate(_SUP_ASCII)), s, e + 1 + len(word)
    p = _POWER.fullmatch(t.literal)
    b = _MANTISSA_MULT.fullmatch(fold(word))
    if p is not None and b is not None and at == s:
        exp = t.literal[2:].lstrip("^").translate(_SUP_ASCII)
        return float(b.group(2)) * 10.0 ** int(exp), s + 1 + len(b.group(1)), e + len(word) + 2
    # r7 round 3: T, then a lone multiplier word, then the inserted power ('2 x' + '10^3'): one scientific number
    after = t.text[e:]
    q = _POWER.fullmatch(fold(word).rstrip(")],;:.'"))
    if q is not None and re.match(" [xX*×·] ", after) and at == e + 2 and re.fullmatch("[0-9]+(?:\\.[0-9]+)?", t.literal):
        exp = fold(word).rstrip(")],;:.'")[2:].lstrip("^").translate(_SUP_ASCII)
        return float(t.literal) * 10.0 ** int(exp), s, e + 3 + len(q.group())
    return None


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
        value, js, je = joined
        ok = all((x.start, x.end) == (js, je) and abs(x.value - value) <= 1e-9 * max(abs(value), 1.0) for x in resolved)
        return ("", "") if ok else ("error", f"a power not parsed into T: {[(v[x.start:x.end], x.value) for x in resolved]}")
    if any((x.start, x.end, x.value) != (s, e, want.value) for x in resolved):
        return "error", f"a different value or literal: {[(v[x.start:x.end], x.value) for x in resolved]}"
    words = [m for m in GAPS.finditer(v)]
    first = next(i for i, m in enumerate(words) if m.start() <= s < m.end())
    last = next(i for i, m in enumerate(words) if m.start() < e <= m.end())
    inert = spec_inert(fold(v), first, last, target_kind(t.literal))
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
