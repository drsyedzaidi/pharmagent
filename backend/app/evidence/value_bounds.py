"""censored_value (error): a verified number that is a bound, an approximation or one end of a printed range (split out
of value_checks.py in r7 round 3).

The words around the token are read (a table cell with the cells one separator away: ``value_context.cell_context``):

* a qualifier before it: an operator run ('<0.5', '<= 5', '< ( 5'), the MEDLINE spelling of <= and >= ('< or = 0.5',
  '> or = 18': r7 round 3), a qualifier word (``word_lists.QUALIFIER_WORDS``: 'below', 'approximately', 'around',
  'NMT' ...), a bound look-back ('less than', 'at least', 'up to'; r7 round 3: any comparative before 'than', 'fewer
  than', 'lower than') and 'less than or equal to'. A word broken at a line-end hyphen ('approxi-' line break 'mately',
  a hyphen, a U+2010 or a soft hyphen) is read whole (r7 round 3);
* a bound AFTER it (r7 round 3): 'or' or 'and' then a word of ``word_lists.AFTER_BOUNDS`` ('0.5 or less', '18 or older',
  '12 and above');
* a range: the reader's range mark (``Tok.ranged``, a unit or percent between the number and the range word included:
  '20 mg - 40 mg'), a lone '-' or '~' word, 'to' on either side ('12 to 15', '20 mg to 40 mg': r7 round 3) and
  'between 12 and 24' (r7 round 3), unless the field names an interval.

Every word list is matched with ``loose_in``: a look-alike spelling (a Greek letter, a mark, a digit inside the word:
'bel0w') flags like the plain one.
"""
from __future__ import annotations

from typing import Any

from app.evidence import models as M
from app.evidence.numgrammar import GAP, Tok, loose_in, tokens, words_after
from app.evidence.numgrammar import glue as G
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    CLOSERS_BR,
    DIGITS,
    EQUALS,
    GROUP_ONLY,
    HYPHEN,
    LETTERS,
    LF,
    OUTSIDE_RE,
    PERCENTS,
    QUALIFIERS,
    RANGE_WORDS,
    TRAILING,
)
from app.evidence.numgrammar.equivalents import fold
from app.evidence.qc_flags import is_interval_field
from app.evidence.value_context import context, flag, verified_token
from app.evidence.word_lists import (
    AFTER_BOUNDS,
    AND,
    BETWEEN,
    BOUND_LEADS,
    BOUND_WORDS,
    COMPARATIVE_END,
    EQUAL,
    OR,
    QUALIFIER_WORDS,
    THAN,
    TO,
)

_TRAIL = "".join(sorted(TRAILING - CLOSERS_BR - GROUP_ONLY))
_CLOSE = "".join(sorted(CLOSERS_BR))
LOOK_BACK = 4                      # words before the token the qualifier look-back reads ('less than or equal to')
LOOK_AHEAD = 3                     # words after it: 'or less', 'mg to 40'


def _numeric_word(word: str, style: str) -> bool:
    """A word that starts with a resolved number ('15', '15.4', '(15'): the other end of 'to'."""
    toks = tokens(word.rstrip(_TRAIL + _CLOSE), style, mode="cell")
    return bool(toks) and toks[0].start <= 1 and toks[0].value is not None


def _unit_word(word: str) -> bool:
    """A word without a digit between a number and a range word: a unit or a noun ('mg', 'h', 'patients') or a lone
    percent sign."""
    core = word.rstrip(_TRAIL + _CLOSE)
    return bool(core) and not any(c in DIGITS for c in core) and (G.prose(core) or all(c in PERCENTS for c in core))


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
    spans = _word_spans(t, start, n + 2)
    words: list[str] = []
    i = 0
    while i < len(spans):
        s, e = spans[i]
        w = t[s:e]
        if i + 1 < len(spans) and len(w) > 1 and w[-2] in LETTERS and (w[-1] in HYPHEN or OUTSIDE_RE.match(w[-1])) \
                and LF in t[e:spans[i + 1][0]] and t[spans[i + 1][0]] in LETTERS:
            words.append(w[:-1] + t[spans[i + 1][0]:spans[i + 1][1]])
            i += 2
            continue
        words.append(w)
        i += 1
    return words[-n:]


def _own_word(text: str, a: int) -> str:
    """The whole word of *text* that holds the offset *a*."""
    k, e = a, a
    while k > 0 and text[k - 1] not in GAP:
        k -= 1
    while e < len(text) and text[e] not in GAP:
        e += 1
    return text[k:e]


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
    """A bound printed after the number: 'or less', 'or older', 'and above' (r7 round 3), '' otherwise."""
    joined = len(nxt) > 1 and (loose_in(nxt[0], OR) or loose_in(nxt[0], AND))
    if joined and loose_in(nxt[1].rstrip(_TRAIL), AFTER_BOUNDS):
        return " ".join(nxt[:2])
    return ""


def _ranged(tok: Tok, ctx: str, a: int, prev: list[str], nxt: list[str], style: str) -> bool:
    """One end of a printed range (see the module docstring)."""
    def is_to(word: str) -> bool:
        return loose_in(word.rstrip(_TRAIL), TO)
    to_before = len(prev) > 1 and is_to(prev[-1]) and (_numeric_word(prev[-2], style) or (
        len(prev) > 2 and _unit_word(prev[-2]) and _numeric_word(prev[-3], style)))
    to_after = len(nxt) > 1 and ((is_to(nxt[0]) and _numeric_word(nxt[1], style)) or (
        len(nxt) > 2 and _unit_word(nxt[0]) and is_to(nxt[1]) and _numeric_word(nxt[2], style)))
    between = (bool(prev) and loose_in(prev[-1], BETWEEN) and len(nxt) > 1 and loose_in(nxt[0], AND)
               and _numeric_word(nxt[1], style)) or (len(prev) > 2 and loose_in(prev[-1], AND)
                                                    and _numeric_word(prev[-2], style) and loose_in(prev[-3], BETWEEN))
    # the tokenizer marks a dash glued to the number, a lone '-' or '~' word before or after it (a unit between them
    # too); a lone one in the neighbouring table cell is read here, and so is a dash glued to the end of the previous
    # cell after its number (r7 writer 3: the cells ['5.0-', '12.4'] are one range)
    glued = bool(prev) and prev[-1][-1] in HYPHEN and W.ends_with_number(prev[-1][:-1]) \
        and W.digit_led(_own_word(ctx, a))
    lone = bool(prev) and prev[-1] in RANGE_WORDS or bool(nxt) and nxt[0] in RANGE_WORDS
    return tok.ranged or glued or lone or to_before or to_after or between


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
    prev, nxt = words_before_joined(ctx, a + off, LOOK_BACK), words_after(ctx, b + off, LOOK_AHEAD)
    q = _qualifier(tok, ctx, a + off, prev)
    if q:
        return [flag(M.FLAG_CENSORED_VALUE, "error", f"{field}: the page prints {q!r} before {literal!r}; a bound "
                     "or approximation is not an exact value", p)]
    if (after := _after(nxt)):
        return [flag(M.FLAG_CENSORED_VALUE, "error", f"{field}: the page prints {after!r} after {literal!r}; a bound "
                     "is not an exact value", p)]
    if is_interval_field(str(field or "")) or not _ranged(tok, ctx, a + off, prev, nxt, style):
        return []
    return [flag(M.FLAG_CENSORED_VALUE, "error", f"{field}: {literal!r} is one end of a printed range; a "
                 "range bound is not an exact value (use an interval field for it)", p)]


__all__ = ["LOOK_BACK", "censored_flags", "words_before_joined"]
