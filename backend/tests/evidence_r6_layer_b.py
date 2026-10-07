"""Layer B of the r6 exhaustive tests beyond ``tokens()``: every code point at every boundary of the inputs of the
paths that sit around the tokenizer (round-1 audit: the tokenizer sweep ran 323 million variants with 0 errors while the
parse placeholder, the label reader, the style evidence and the text-field locator each had a hole).

``sweep_*`` return the list of violations for a list of code points; the tests run the stratified sample, the
``exhaustive`` test runs all 1 112 064 code points on a process pool.

  scale     a label that scales the printed number keeps raising the flag when ANY non-ASCII character is inserted
            anywhere in it (a look-alike, an invisible mark or a stray symbol never hides the scale)
  style     a line holding a character outside the alphabet (or a LINE character glued to its numbers) casts no vote on
            the document's decimal style
  thin      the parse placeholder restores exactly the approved spaces: every other character, a private-use one
            included, comes out of ``ThinSpaces.restore`` as it went in
  text      a text-field record whose page or snippet holds ANY inserted character inside a number never passes against
            the clean one (no normalisation equates a character with another or drops it)
  locator   (round 2) ``normalise`` rewrites a character exactly as the specification says (``evidence_r6_round2_spec``):
            the approved equivalents, seven ligatures, the lower case of the listed capitals, the GAP spaces collapsed;
            every other code point comes out as itself, so no compatibility form is ever another character
  unit      (round 2) a unit holding a character outside INSIDE / CONTEXT / GAP is unresolved: it equals no unit and is
            recognised as none
  key       (round 2) a header key drops exactly the spaces and ``_ - , .`` and keeps every other character
"""
from __future__ import annotations

from app.evidence.extract_rules import norm_key
from app.evidence.normalize import decimal_evidence, normalise
from app.evidence.numgrammar import scale_shapes
from app.evidence.numgrammar.equivalents import EQUIV, THIN
from app.evidence.parse_guard import THIN_SPACES, ParseLimitError, protect_thin_spaces
from app.evidence.record_checks import _UNIT_TOKEN_RE, recognised_unit
from app.evidence.unit_norm import canonical_unit, same_unit, unit_resolved
from tests import evidence_r6_oracle as O
from tests import evidence_r6_pins as P
from tests import evidence_r6_round2_spec as S
from tests.evidence_r6_round1_support import run

SCALE_LABELS = ("AUC (x1000)", "AUC (×10^3)", "AUC × 10³", "Cmax (thousands)", "AUC x 1000", "Dose (millions)",
                "CL / 1000", "Cmax E+03")
STYLE_BASE = "AUC was 12.345 mg.h/L in healthy volunteers.\nCmax 3.4 ng/mL\n"
STYLE_LINES = ("Werte 3,4 und 5,6 und 7,8", "3,4 5,6", "x 3,4")
TEXT_PAGE, TEXT_SNIPPET = "Inoculum 100 cells/mL were infused.", "Inoculum 100 cells/mL"


def non_ascii(cps: list[int]) -> list[int]:
    """The characters the reader cannot read (LINE or OUTSIDE class) that stand for no ASCII character: not an approved
    equivalent of one (U+2010 is ``-``, U+00A0 a gap: inserting those is inserting an ASCII character, which may
    legitimately change a label). A readable character (``±``, ``²``, ``μ``) changes what the label SAYS; an
    unreadable one must never change what it MEANS."""
    ascii_equivalents = {ord(k) for k, v in EQUIV.items() if v.isascii() or v == THIN}
    return [cp for cp in cps if cp >= 0x80 and not 0xD800 <= cp <= 0xDFFF and cp not in ascii_equivalents
            and P.expected_class(cp) in ("line", "outside")]


def sweep_scale(cps: list[int]) -> list[str]:
    bad = []
    for label in SCALE_LABELS:
        if scale_shapes(label) is not True:
            bad.append(f"clean label {label!r} is not read as a scale")
            continue
        for b in range(len(label) + 1):
            for cp in non_ascii(cps):
                variant = label[:b] + chr(cp) + label[b:]
                if scale_shapes(variant) is not True:
                    bad.append(f"{label!r}@{b}: U+{cp:04X} hides the scale")
                    if len(bad) >= 20:
                        return bad
    return bad


def sweep_style(cps: list[int]) -> list[str]:
    """A character outside the alphabet anywhere in a line taints every number of it, so the line casts no vote; a LINE
    character does the same to the words it touches and their neighbours, so a line made of one glued word does too.
    The lines are independent (the reader works line by line): one appended line is checked against the base text, the
    others on their own."""
    bad, base = [], decimal_evidence(STYLE_BASE)
    for cp in cps:
        if 0xD800 <= cp <= 0xDFFF or P.expected_class(cp) not in ("outside", "line"):
            continue
        c = chr(cp)
        for line in STYLE_LINES:
            for b in range(len(line) + 1):
                variant = line[:b] + c + line[b:]
                if P.expected_class(cp) == "outside" and decimal_evidence(variant) != (0, 0):
                    bad.append(f"{line!r}@{b}: U+{cp:04X} votes {decimal_evidence(variant)}")
        glued = f"3,4{c}5,6"
        if decimal_evidence(STYLE_BASE + (glued + "\n") * 6) != base:
            bad.append(f"U+{cp:04X}: a glued word '3,4{c}5,6' changes the document's evidence")
        if len(bad) >= 20:
            break
    return bad


class _Page:
    page_number = 1

    def __init__(self, texts: list[str]) -> None:
        self.chars = [{"text": t} for t in texts]


def sweep_thin(cps: list[int]) -> list[str]:
    bad = []
    for cp in cps:
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        texts = ["1", c, "2", "3", "4", " ", "5"]
        page = _Page(list(texts))
        try:
            thin = protect_thin_spaces(page)
        except ParseLimitError:
            bad.append(f"U+{cp:04X}: refused a page of 7 glyphs")
            continue
        out = thin.restore("".join(ch["text"] for ch in page.chars))
        want = "".join(THIN if t in THIN_SPACES else t for t in texts)
        if out != want:
            bad.append(f"U+{cp:04X}: restored {out!r} != {want!r}")
        if thin.mark and any(t == thin.mark for t in texts if t not in THIN_SPACES):
            bad.append(f"U+{cp:04X}: the placeholder is a glyph of the page")
        if len(bad) >= 20:
            break
    return bad


def sweep_text(cps: list[int]) -> list[str]:
    """The number's digits are 'Inoculum 1|0|0 cells'. A character inserted between two of them, or put in the place of
    one of them, in the page or in the snippet, makes the other one not match, whatever the character is (a
    superscript, a fullwidth or a subscript digit is never the digit it looks like)."""
    bad = []
    for cp in cps:
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        variants = [(f"insert@{b}", TEXT_PAGE[:b] + c + TEXT_PAGE[b:], TEXT_SNIPPET[:b] + c + TEXT_SNIPPET[b:])
                    for b in (10, 11)]
        variants += [(f"replace@{b}", TEXT_PAGE[:b] + c + TEXT_PAGE[b + 1:], TEXT_SNIPPET[:b] + c + TEXT_SNIPPET[b + 1:])
                     for b in (9, 10, 11) if c != TEXT_PAGE[b]]
        for what, page, snippet in variants:
            for where, p, s in (("page", page, TEXT_SNIPPET), ("snippet", TEXT_PAGE, snippet)):
                if run(p, s, s, field="Cell dose").status == "pass":
                    bad.append(f"U+{cp:04X} ({what}, in the {where}) still passes: {p!r} / {s!r}")
        if len(bad) >= 20:
            break
    return bad


def sweep_locator(cps: list[int]) -> list[str]:
    """``normalise('a' + c + 'b')`` is the specification's text for c, and the index map points every output character of c
    at c's own offset."""
    bad = []
    for cp in cps:
        if 0xD800 <= cp <= 0xDFFF:
            continue
        norm, m = normalise("a" + chr(cp) + "b")
        want = "a" + S.spec_locator(chr(cp)) + "b"
        if norm != want:
            bad.append(f"U+{cp:04X}: normalise gives {norm!r}, the specification {want!r}")
        elif not (len(m) == len(norm) and m[0] == 0 and m[-1] == 2 and set(m[1:-1]) == {1}):
            bad.append(f"U+{cp:04X}: the index map {m} does not point at the character")
        if len(bad) >= 20:
            break
    return bad


def sweep_unit(cps: list[int]) -> list[str]:
    """A unit ``mg`` + c is resolved exactly when c is an alphabet character (INSIDE, CONTEXT, GAP or an equivalent's
    source); with any other c it equals no unit, ``mg`` included, and no unit is recognised in it. The printed-unit
    reader starts a unit with a percent sign of the alphabet or a letter of the alphabet and with nothing else (round
    1 accepted U+2031, which the alphabet puts outside)."""
    bad = []
    for cp in cps:
        if 0xD800 <= cp <= 0xDFFF:
            continue
        if (_UNIT_TOKEN_RE.match(chr(cp)) is not None) != S.spec_unit_token(chr(cp)):
            bad.append(f"U+{cp:04X}: the printed-unit reader starts a unit with it: {S.spec_unit_token(chr(cp))} expected")
        unit, resolved = "mg" + chr(cp), S.spec_unit_resolved(chr(cp))
        if unit_resolved(unit) != resolved:
            bad.append(f"U+{cp:04X}: the unit is {'resolved' if not resolved else 'unresolved'}")
        elif not resolved and (same_unit(unit, "mg") or canonical_unit(unit) == "mg" or recognised_unit(unit) is not None):
            bad.append(f"U+{cp:04X}: a unit with this character equals or is read as mg")
        if len(bad) >= 20:
            break
    return bad


def sweep_key(cps: list[int]) -> list[str]:
    """The header key of ``C`` + c + ``max`` is ``cmax`` exactly when c is a space or one of ``_ - , .`` (or a hyphen's
    equivalent); every other character stays in the key, so such a header matches no field."""
    bad = []
    for cp in cps:
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if (norm_key("C" + c + "max") == "cmax") != (c in S.KEY_STRIP_PIN):
            bad.append(f"U+{cp:04X}: the key is {norm_key('C' + c + 'max')!r}")
        if len(bad) >= 20:
            break
    return bad


SWEEPS = {"scale": sweep_scale, "style": sweep_style, "thin": sweep_thin, "text": sweep_text,
          "locator": sweep_locator, "unit": sweep_unit, "key": sweep_key}


# ------------------------------------------------------------------------ the process pool ---
def _task(args: tuple[str, list[int]]) -> tuple[str, int, list[str]]:
    name, chunk = args
    return name, len(chunk), SWEEPS[name](chunk)


def full_run(workers: int, chunk: int = 20_000) -> dict:
    """Every sweep over every code point (surrogates excluded) on a spawn pool."""
    import multiprocessing as mp
    import time
    cps = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    tasks = [(name, cps[i:i + chunk]) for name in SWEEPS for i in range(0, len(cps), chunk)]
    t0, failures, done = time.perf_counter(), [], {name: 0 for name in SWEEPS}
    with mp.get_context("spawn").Pool(workers) as pool:
        for name, n, bad in pool.imap_unordered(_task, tasks):
            done[name] += n
            failures += bad
    state = O.repo_state()
    return {"code_points": done, "errors": len(failures), "failures": failures[:50],
            "wall_seconds": round(time.perf_counter() - t0, 1), "head": state["head"], "dirty": state["dirty"]}
