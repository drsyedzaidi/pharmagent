"""censored_value (error): a verified number that is a bound, an approximation, a spelled-out sign or one end of a
printed range (split out of value_checks.py in r7 round 3; the range half lives in value_ranges.py since r7 round 4).

The words around the token are read (a table cell with the cells one separator away: ``value_context.cell_context``,
whose walk goes on through the cells of a bound or range phrase: r7 round 4):

* a qualifier before it: an operator run ('<0.5', '<= 5', '< ( 5'), the MEDLINE spelling of <= and >= ('< or = 0.5',
  '> or = 18': r7 round 3), a qualifier word (``word_lists.QUALIFIER_WORDS``: 'below', 'approximately', 'around',
  'NMT', r7 round 4 'not exceeding', 'maximum', 'beyond', 'GE' ...), a bound look-back ('less than', 'at least', 'up
  to', 'in excess of'; r7 round 3: any comparative before 'than', 'fewer than', 'lower than') and 'less than or equal
  to'. A word broken at a line-end hyphen ('approxi-' line break 'mately', a hyphen, a U+2010 or a soft hyphen) is read
  whole (r7 round 3);
* a SIGN WORD before it (r7 round 4, ``word_lists.SIGN_WORDS``: 'minus 12%', 'negative 1.5'): the printed digits are
  not the value;
* a bound AFTER it (r7 round 3): 'or' or 'and' then a word of ``word_lists.AFTER_BOUNDS`` ('0.5 or less', '18 or older',
  '12 and above'), r7 round 4 also with one unit word between ('18 years or older', '0.5 ng/mL or less');
* a range (value_ranges.py), unless the field names an interval.

Every word list is matched with ``word_match.loose_in``: a look-alike spelling (a Greek letter, a mark, a digit inside
the word, an OCR letter shape: 'bel0w', 'beIow', 'up t0') flags like the plain one. r7 round 4: a bracketed reference
group ('[3]', '[3, 4]': the citation production's group) never hides a word: the words are read twice, as printed and
with every such group that does not hold the token taken out, and either reading raises the flag.
"""
from __future__ import annotations

from typing import Any

from app.evidence import models as M
from app.evidence.numgrammar import GAP, Tok, words_after
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import EQUALS, LETTERS, QUALIFIERS, SPACE
from app.evidence.numgrammar.equivalents import fold
from app.evidence.numgrammar.productions import bracket_groups
from app.evidence.qc_flags import is_interval_field
from app.evidence.value_context import context, flag, verified_token, wrapped_words
from app.evidence.value_ranges import TRAIL as _TRAIL
from app.evidence.value_ranges import ranged, unit_word
from app.evidence.word_lists import (
    AFTER_BOUNDS,
    AND,
    BOUND_LEADS,
    BOUND_WORDS,
    COMPARATIVE_END,
    EQUAL,
    OR,
    QUALIFIER_WORDS,
    SIGN_WORDS,
    THAN,
    TO,
)
from app.evidence.word_match import loose_in

LOOK_BACK = 4                      # words before the token the qualifier look-back reads ('less than or equal to')
LOOK_AHEAD = 3                     # words after it: 'or less', 'mg to 40', 'years or older'


def _word_spans(t: str, start: int, n: int) -> list[tuple[int, int]]:
    """[a, b) of up to *n* whole words before the word that holds *start*, nearest last."""
    a = start
    while a > 0 and t[a - 1] not in GAP:
        a -= 1
    out: list[tuple[int, int]] = []
    while len(out) < n:
        e = a
        while e > 0 and t[e - 1] in GAP:
            e -= 1
        if e == 0:
            break
        s = e
        while s > 0 and t[s - 1] not in GAP:
            s -= 1
        out.insert(0, (s, e))
        a = s
    return out


def words_before_joined(text: str, start: int, n: int) -> list[str]:
    """The *n* whole words before the token's word, in reading order, with a word broken at a line end rejoined: a
    word that ends in a letter and a hyphen (or a character the reader cannot read, a soft hyphen) and the word on the
    next line that starts with a letter are one word ('approxi-' + 'mately' is 'approximately'; r7 round 3)."""
    t = fold(text)
    return wrapped_words(t, _word_spans(t, start, n + 2))[-n:]


def _own_prefix(text: str, a: int) -> str:
    """The characters of the token's own word before the token ('<' of '<0.5')."""
    k = a
    while k > 0 and text[k - 1] not in GAP:
        k -= 1
    return text[k:a]


def _comparative(word: str) -> bool:
    """A bound lead before 'than': less, more, greater ... or any letter word ending in -er ('fewer', 'lower')."""
    core = fold(word).rstrip(_TRAIL)
    return loose_in(core, BOUND_LEADS) or (all(c in LETTERS for c in core) and core.endswith(COMPARATIVE_END))


def _qualifier(tok: Tok, text: str, a: int, prev: list[str]) -> str:
    """What stands before the token and makes it a bound or an approximation, '' when nothing does. *prev* holds up
    to four whole words before the token's word, in reading order."""
    w = [x.rstrip(_TRAIL) for x in prev]
    w1, w2, w3 = (w[-1] if w else ""), (w[-2] if len(w) > 1 else ""), (w[-3] if len(w) > 2 else "")
    if tok.qualified:
        return _own_prefix(text, a) or next((x for x in reversed(prev) if not QUALIFIERS.isdisjoint(x)),
                                            prev[-1] if prev else "")
    if W.qualifies(prev):
        return prev[-1]
    if w1 and all(c in EQUALS for c in w1) and loose_in(w2, OR) and W.qualifies([w3]):
        return " ".join(prev[-3:])                     # '< or =', '> or =': the MEDLINE spelling of <= and >=
    if (loose_in(w1, BOUND_WORDS) and loose_in(w2, BOUND_LEADS)) or (loose_in(w1, THAN) and _comparative(w2)):
        return " ".join(prev[-2:])
    if loose_in(w1, TO) and loose_in(w2, EQUAL) and loose_in(w3, OR):
        return " ".join(prev[-3:])
    return prev[-1] if loose_in(w1, QUALIFIER_WORDS) else ""


def _after(nxt: list[str]) -> str:
    """A bound printed after the number: 'or less', 'or older', 'and above' (r7 round 3), with one unit word between
    ('18 years or older', '0.5 ng/mL or less': r7 round 4), '' otherwise."""
    for k in (0, 1):
        if k and not (nxt and unit_word(nxt[0])):
            break
        joined = len(nxt) > k + 1 and (loose_in(nxt[k], OR) or loose_in(nxt[k], AND))
        if joined and loose_in(nxt[k + 1].rstrip(_TRAIL), AFTER_BOUNDS):
            return " ".join(nxt[:k + 2])
    return ""


def _sign(prev: list[str]) -> str:
    """A sign spelled out before the number ('minus 12%', 'negative 1.5', 'plus 3': r7 round 4), '' otherwise."""
    return prev[-1] if prev and loose_in(prev[-1].rstrip(_TRAIL), SIGN_WORDS) else ""


def _message(tok: Tok, ctx: str, a: int, b: int, style: str, field: Any, literal: Any) -> str:
    """The censored_value message for one reading of the words around the token at [a, b) of *ctx*, '' when none."""
    prev, nxt = words_before_joined(ctx, a, LOOK_BACK), words_after(ctx, b, LOOK_AHEAD)
    if q := _qualifier(tok, ctx, a, prev):
        return f"{field}: the page prints {q!r} before {literal!r}; a bound or approximation is not an exact value"
    if s := _sign(prev):
        return (f"{field}: the page prints the sign word {s!r} before {literal!r}; the printed digits are not the "
                "value: read it on the PDF")
    if after := _after(nxt):
        return f"{field}: the page prints {after!r} after {literal!r}; a bound is not an exact value"
    if is_interval_field(str(field or "")) or not ranged(tok, ctx, a, prev, nxt, style):
        return ""
    return (f"{field}: {literal!r} is one end of a printed range; a range bound is not an exact value (use an "
            "interval field for it)")


def without_groups(text: str, a: int, b: int) -> str | None:
    """*text* with every bracketed reference group that does not overlap [a, b) blanked 1:1 with spaces (offsets stay),
    None when there is none (r7 round 4: '[3]' between 'less than' and the number never hides the bound)."""
    t = fold(text)
    spans = [(s, e) for s, e in bracket_groups(t) if e <= a or s >= b]
    if not spans:
        return None
    out = list(text)
    for s, e in spans:
        out[s:e] = [SPACE] * (e - s)
    return "".join(out)


def censored_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    loc = p.get("locator") or {}
    got = verified_token(loc, page)
    if got is None:
        return []
    text, a, b, tok = got
    literal, field = loc.get("literal"), p["field"]
    style = loc.get("decimal_style") or "."
    if tok is None or tok.value is None:         # the stored token is not one readable number any more
        return [flag(M.FLAG_CENSORED_VALUE, "error", f"{field}: {literal!r} is not one readable number next to "
                     "its neighbours, so a qualifier or a range bound cannot be ruled out; read the value on the "
                     "PDF", p)]
    ctx, off = context(loc, page, text)      # a table cell's marks are its own: a cell boundary is structure
    for reading in (ctx, without_groups(ctx, a + off, b + off)):
        if reading is not None and (msg := _message(tok, reading, a + off, b + off, style, field, literal)):
            return [flag(M.FLAG_CENSORED_VALUE, "error", msg, p)]
    return []


__all__ = ["LOOK_BACK", "censored_flags", "without_groups", "words_before_joined"]
