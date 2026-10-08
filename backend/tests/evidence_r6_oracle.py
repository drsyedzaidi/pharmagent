"""Oracle and harness of the r6 exhaustive tests: every code point at every boundary of every canonical expression.

For one expression E (a sentence, a page fragment or a table cell), a boundary b and one code point c, the variant is
``E[:b] + c + E[b:]``. The oracle never re-implements the grammar; it states what the one-separator rule and the five
classes of the alphabet promise, from the class of c alone (r7: four classes; the LINE list is deleted and the line
rule is DERIVED: a character of no list is outside):

outside   every token on c's line, and every token whose word is c's word or one word before / after it, is
          ``unresolved_context``; every other token equals the clean token (offsets shifted); no resolved token
          spans c. (All ~1.1 M outside code points must behave identically: the sweep compares each one with the
          reference outside character, and the 35 outside characters of the deleted LINE list among them.)
gap       identical to inserting U+0020.
equiv     identical to inserting the equivalent's target.
inside /  structurally identical (token spans, shapes, flags: not values) to the class representative of c
context   (``rep_of``), pinned in the golden file; a letter, bracket or ``; : / = deg`` never makes a number the
          clean text did not hold.

Page-mode text is batched: the variants of one pair are joined with ``SEP`` so each sits on its own line and the
tokenizer scans ~40 000 of them in one call; the per-variant result is read back by offset.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from bisect import bisect_right
from itertools import groupby
from pathlib import Path
from typing import NamedTuple

from app.evidence.numgrammar import Tok, tokens
from tests import evidence_r6_pins as P
from tests.evidence_r6_attest import ATTESTATION
from tests.evidence_r6_expressions import (  # noqa: F401  (re-exported: the tests read everything through ``O``)
    CID_PANEL,
    EXPRESSIONS,
    PAIRS,
    REF_FORMER_LINE,
    REF_OUTSIDE,
    SEP,
    Expr,
    full,
    golden_text,
    run,
    sig,
    variants,
)
from tests.evidence_r6_sampling import (  # noqa: F401
    ALWAYS,
    categories,
    non_outside_cps,
    outside_cps,
    stratified_outside,
)

GOLDEN = Path(__file__).with_name("evidence_r6_golden.json")
_WORDS = re.compile("[^ \t\n\r]+")
_MULT = frozenset("xX*×·")
_ASCII_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_GREEK = frozenset(chr(c) for c in (0x3BC, 0x3C9, 0x3B2, 0x3B1, 0x3C4, 0x3BB, 0x3B7, 0x3B8, 0x3B5, 0x3B3, 0x3C3,
                                    0x3C1, 0x3C0, 0x394, 0x3A9, 0x3A3, 0x3BA))
# r7: the letters and marks folded into CONTEXT from the deleted LINE list
_ACCENTED = frozenset(chr(c) for c in (0xE9, 0xE8, 0xEB, 0xE2, 0xE1, 0xC1, 0xE3, 0xE6, 0xC5, 0xE4, 0xF6, 0xD6, 0xFC,
                                       0xED, 0xEF, 0xF3, 0xF8, 0xD8, 0xFA, 0x171, 0xF1, 0x144, 0xE7, 0xC7, 0x107,
                                       0x10D, 0x11F, 0x142, 0x15B))
_MARKS = frozenset(chr(c) for c in (0x26, 0x40, 0x201C, 0x201D, 0xA9, 0xAE, 0x2122, 0x221E, 0xA7, 0xB6, 0x2020,
                                    0x2021))


def repo_state() -> dict:
    """{"head": the commit the code under test is at, "dirty": the files that differ from it (git status --porcelain)}.
    A log of a run on a dirty tree is evidence of the tree, not of the commit: the run records both, and the pytest
    wrapper refuses a dirty tree unless R6_ALLOW_DIRTY=1. The attestation file the runs write is not counted."""
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
    own = f"tests/{ATTESTATION.name}"           # the record of a passing run is written by the run itself: not a change
    return {"head": git("rev-parse", "HEAD").strip(),
            "dirty": [line for line in git("status", "--porcelain").splitlines() if not line.endswith(own)]}


# ------------------------------------------------------------------ class machinery ---
def class_of(cp: int) -> str:
    return P.expected_class(cp)


def rep_of(ch: str) -> str:
    """The class representative of an inside or context character (the 29 behaviours the grammar can tell apart; a
    mark folded into CONTEXT by r7 has no role in the grammar: ``&`` stands for all twelve)."""
    if ch == "0":
        return "0"
    if ch in "123456789":
        return "5"
    if ch in "eE":
        return "e"
    if ch in "xX":
        return "x"
    if ch in _ASCII_LETTERS or ch in _GREEK or ch in _ACCENTED:
        return "m"
    if ch in _MARKS:
        return "&"
    if ch in "⁰¹²³⁴⁵⁶⁷⁸⁹":
        return "²"
    if ch in "⁺⁻":
        return "⁻"
    if ch in "%‰":
        return "%"
    if ch in "<>":
        return "<"                                   # '<=' and '>=' are digraph qualifiers
    if ch in "≤≥~≈":
        return "~"
    if ch in "([":
        return "("
    if ch in ")]":
        return ")"
    return ch


# characters that can never be part of a resolved literal ('(' '[' '=' start a token and ')' ']' ';' ':' trail one, so
# next to them a number may split in two; a letter in a thin-space group may leave '900' of '18 900': values are not
# compared, only that no resolved literal contains such a character)
NON_NUMERIC = frozenset(_ASCII_LETTERS - set("eExX")) | _GREEK | _ACCENTED | _MARKS | frozenset("/°")


# ------------------------------------------------------------- the independent oracle ---
def _words(s: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORDS.finditer(s)]


def check_taint(e: Expr, b: int, ins: str, got: list[Tok]) -> list[str]:
    """What the tiers promise for an outside insertion (word, window and line: r7 has no LINE class); [] when it
    holds."""
    s, ln = variants(e, b, ins), len(ins)
    words = _words(s)
    starts = [w[0] for w in words]

    def widx(pos: int) -> int:
        k = bisect_right(starts, pos) - 1
        return k if k >= 0 and pos < words[k][1] else -1

    cws = {i for p in range(b, b + ln) if (i := widx(p)) >= 0}
    region: set[int] = set()
    for cw in cws:
        region |= {cw - 1, cw, cw + 1}
        if cw >= 1 and set(s[words[cw - 1][0]:words[cw - 1][1]]) <= _MULT:
            region.add(cw - 2)                                   # the right operand behind a multiplier word
    region = {i for i in region if 0 <= i < len(words)}
    line_of = (lambda p: 0) if e.mode == "cell" else (lambda p: s.count("\n", 0, p))
    cline = line_of(b)

    def affected(a: int, z: int) -> bool:
        return bool(set(range(widx(a), widx(z - 1) + 1)) & region) or cline in (line_of(a), line_of(z - 1))

    def shift(a: int, z: int) -> tuple[int, int]:
        return a + (ln if a >= b else 0), z + (ln if z > b else 0)

    clean = run(e, e.text)
    want = {shift(t.start, t.end): t for t in clean}
    errors: list[str] = []
    seen = set()
    for g in got:
        if g.start < b + ln and g.end > b and g.value is not None:
            errors.append(f"resolved token {g.value} spans the inserted character")
        if affected(g.start, g.end):
            if g.value is not None or g.shape != "unresolved_context":
                errors.append(f"affected token {(g.start, g.end)} is {g.shape} / {g.value}")
            continue
        c = want.get((g.start, g.end))
        if c is None:
            errors.append(f"new token {(g.start, g.end, g.shape, g.value)} in an unaffected place")
        elif full([c])[0][2:] != full([g])[0][2:]:
            errors.append(f"unaffected token {(g.start, g.end)} changed: {c.shape}/{c.value} -> {g.shape}/{g.value}")
        seen.add((g.start, g.end))
    for (a, z), c in want.items():
        if not affected(a, z) and (a, z) not in seen:
            errors.append(f"unaffected clean token {(a, z, c.shape, c.value)} disappeared")
    return errors


def check_not_inside(b: int, got: list[Tok], ch: str = "") -> list[str]:
    """A letter, a Greek letter, a mark, ``/`` or the degree sign can never be part of a resolved number's literal,
    except the slash of a ratio (r7, owner decision 6: ``12/15`` is one ratio token)."""
    return [f"resolved token {g.value} spans the inserted character" for g in got
            if g.value is not None and g.start <= b < g.end and not (ch == "/" and g.shape == "ratio")]


# --------------------------------------------------------------------- class sweeps ---
def check_class(e: Expr, b: int, cp: int, golden: dict[str, str] | None) -> list[str]:
    """The oracle for one non-outside code point (inside, context, gap or an equivalent's source)."""
    c = chr(cp)
    got = run(e, variants(e, b, c))
    kind = class_of(cp)
    if cp in P.EQUIV_PIN:
        want = run(e, variants(e, b, chr(P.EQUIV_PIN[cp])))
        return [] if full(got) == full(want) and [t.detail for t in got] == [t.detail for t in want] else [
            f"U+{cp:04X}: not identical to its equivalent U+{P.EQUIV_PIN[cp]:04X}"]
    if kind == "gap":
        if c == "\n" and _has_taint(e):
            return []
        want = run(e, variants(e, b, " "))
        same = full(got) == full(want) and [t.detail for t in got] == [t.detail for t in want]
        return [] if same else [f"U+{cp:04X}: a gap character reads unlike U+0020"]
    rep = rep_of(c)
    want = run(e, variants(e, b, rep))
    errors = [] if sig(got) == sig(want) else [f"U+{cp:04X}: structure differs from its representative {rep!r}"]
    if c in NON_NUMERIC:
        errors += [f"U+{cp:04X}: {m}" for m in check_not_inside(b, got, c)]
    if golden is not None and c == rep and golden.get(f"{e.name}:{b}:{rep}") != golden_text(got):
        errors.append(f"U+{cp:04X}: golden differs for {e.name}:{b}:{rep!r}")
    return errors


def _has_taint(e: Expr) -> bool:
    return any(class_of(ord(ch)) == "outside" for ch in e.text) or "(cid:" in e.text


def reference_errors(e: Expr, b: int) -> list[str]:
    """The independent oracle on the reference outside character, a character of the deleted LINE list (r7: it is
    outside now and must taint its whole line) and the cid panel."""
    errors = [f"outside: {m}" for m in check_taint(e, b, REF_OUTSIDE, run(e, variants(e, b, REF_OUTSIDE)))]
    errors += [f"former line: {m}" for m in check_taint(e, b, REF_FORMER_LINE,
                                                        run(e, variants(e, b, REF_FORMER_LINE)))]
    for marker in CID_PANEL:
        errors += [f"{marker}: {m}" for m in check_taint(e, b, marker, run(e, variants(e, b, marker)))]
    return errors


# ------------------------------------------------------------------ the outside batch ---
def _ref_items(e: Expr, b: int) -> list[tuple]:
    return [(t.start, t.end, t.value, t.shape, t.parts, t.qualified, t.ranged, t.canonical, t.decimal,
             t.detail.replace("U+FFFD", "U+{cp}")) for t in run(e, variants(e, b, REF_OUTSIDE))]


def batch_errors(e: Expr, b: int, cps: list[int]) -> list[str]:
    """Every outside code point in *cps* at boundary b must read exactly like the reference outside character."""
    ref = _ref_items(e, b)
    one = len(e.text) + 1
    stride = one + len(SEP)
    joined = SEP.join(variants(e, b, chr(cp)) for cp in cps)
    got = tokens(joined, e.style, mode="page")
    errors: list[str] = []
    by_variant = {k: list(g) for k, g in groupby(got, key=lambda t: t.start // stride)}
    for k, cp in enumerate(cps):
        base = k * stride
        items = by_variant.get(k, [])
        rel = [(t.start - base, t.end - base, t.value, t.shape, t.parts, t.qualified, t.ranged, t.canonical, t.decimal,
                t.detail) for t in items]
        want = [r[:-1] + (r[-1].replace("{cp}", f"{cp:04X}"),) for r in ref]
        if rel != want:
            errors.append(f"U+{cp:04X}: {rel[:2]} != reference {want[:2]}")
            if len(errors) >= 20:
                break
        if any(t.end > base + one for t in items):
            errors.append(f"U+{cp:04X}: a token leaks past its own line")
    return errors


class PairResult(NamedTuple):
    pair: tuple[int, int]
    variants: int
    errors: list[str]
    seconds: float


def sweep_pair(pair: tuple[int, int], outside: list[int], golden: dict[str, str] | None, chunk: int = 40_000,
               direct: list[int] | None = None) -> PairResult:
    """Everything for one (expression, boundary): reference checks, every non-outside code point directly, every
    outside code point in *outside* through the batch."""
    t0 = time.perf_counter()
    e, b = EXPRESSIONS[pair[0]], pair[1]
    errors = [f"{e.name}@{b}: {m}" for m in reference_errors(e, b)]
    n = 0
    for cp in direct if direct is not None else non_outside_cps():
        errors += [f"{e.name}@{b}: {m}" for m in check_class(e, b, cp, golden)]
        n += 1
    for i in range(0, len(outside), chunk):
        part = outside[i:i + chunk]
        errors += [f"{e.name}@{b}: {m}" for m in batch_errors(e, b, part)]
        n += len(part)
    return PairResult(pair, n, errors[:50], time.perf_counter() - t0)


def load_golden() -> dict[str, str] | None:
    """The golden file as ``{"expression:boundary:representative": reading}``; the file holds one line per pair."""
    if not GOLDEN.exists():
        return None
    nested = json.loads(GOLDEN.read_text("utf-8"))
    return {f"{pair}:{rep}": reading for pair, reps in nested.items() for rep, reading in reps.items()}


def write_golden(flat: dict[str, str]) -> None:
    """One line per (expression, boundary): a diff names the pair whose reading changed."""
    grouped: dict[str, dict[str, str]] = {}
    for key, reading in sorted(flat.items()):
        name, boundary, rep = key.split(":", 2)                     # the representative may be ':' itself
        grouped.setdefault(f"{name}:{boundary}", {})[rep] = reading
    body = ",\n".join(f"{json.dumps(pair)}: {json.dumps(reps, sort_keys=True, ensure_ascii=True)}"
                      for pair, reps in grouped.items())
    GOLDEN.write_text("{\n" + body + "\n}\n", "utf-8")


def build_golden() -> dict[str, str]:
    """The pin of the 29 class representatives at every boundary (regenerate with R6_REGEN_GOLDEN=1)."""
    reps = sorted({rep_of(chr(cp)) for cp in P.INSIDE_SET | P.CONTEXT_SET})
    out: dict[str, str] = {}
    for e in EXPRESSIONS:
        for b in e.cuts:
            for rep in reps:
                out[f"{e.name}:{b}:{rep}"] = golden_text(run(e, variants(e, b, rep)))
    return out


# ------------------------------------------------------------- the multiprocessing run ---
def _task(args: tuple[tuple[int, int], int]) -> PairResult:
    pair, stratified = args
    golden = load_golden()
    out = stratified_outside(stratified) if stratified else outside_cps()
    return sweep_pair(pair, out, golden)


def full_run(workers: int, log: Path | None, stratified: int = 0, pairs: list[tuple[int, int]] | None = None) -> dict:
    """Run every pair on a pool (spawned processes); *stratified* limits the outside code points (0 = all of them)."""
    import multiprocessing as mp
    pairs = pairs if pairs is not None else PAIRS
    t0, done, variants_n, failures = time.perf_counter(), 0, 0, []
    state = repo_state()
    if log is not None:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"RUN head={state['head']} dirty={len(state['dirty'])} files={state['dirty'][:20]}\n")
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for res in pool.imap_unordered(_task, [(p, stratified) for p in pairs]):
            done += 1
            variants_n += res.variants
            failures += res.errors
            name = EXPRESSIONS[res.pair[0]].name
            line = (f"[{done}/{len(pairs)}] {name}@{res.pair[1]}: {res.variants} variants, "
                    f"{len(res.errors)} errors, {res.seconds:.1f}s, elapsed {time.perf_counter() - t0:.0f}s")
            print(line, flush=True)
            if log is not None:
                with log.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n" + "".join(f"    {m}\n" for m in res.errors[:5]))
    summary = {"pairs": len(pairs), "variants": variants_n, "errors": len(failures),
               "wall_seconds": round(time.perf_counter() - t0, 1), "workers": workers, "head": state["head"],
               "dirty": state["dirty"]}
    if log is not None:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"SUMMARY {json.dumps(summary)}\n")
    return {**summary, "failures": failures[:50]}


if __name__ == "__main__":                                # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="r6 exhaustive tokenizer sweep")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--stratified", type=int, default=0, help="limit the outside code points (0 = every one)")
    ap.add_argument("--only", type=str, default="", help="expression name prefix")
    ap.add_argument("--regen-golden", action="store_true")
    ns = ap.parse_args()
    if ns.regen_golden:
        write_golden(build_golden())
        print(f"golden written: {GOLDEN}")
        raise SystemExit(0)
    chosen = [p for p in PAIRS if EXPRESSIONS[p[0]].name.startswith(ns.only)]
    res = full_run(ns.workers, ns.log, ns.stratified, chosen)
    print(json.dumps(res, indent=1))
    raise SystemExit(1 if res["errors"] else 0)
