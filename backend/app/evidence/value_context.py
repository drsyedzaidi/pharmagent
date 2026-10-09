"""The verified token and the words around it: the shared reading of the value checks (value_bounds, value_scale,
value_checks) and of the verifier's cell window (r7 round 3: split out of value_checks.py, which grew past 400 lines).

A table cell is one line, and the cells around it are the words one separator away: ``cell_context`` joins the cited
cell with the cells on each side up to the first one that holds a word (an empty, blank or punctuation-only cell, such
as a spanned cell or a lone bracket, is walked past and read too), so the neighbour-word allow-list, the qualifier and
range look-back and the readability walk all read the same words.
"""
from __future__ import annotations

from typing import Any

from app.evidence import gaps
from app.evidence.numgrammar import GAP, Tok, is_data_cell
from app.evidence.numgrammar.alphabet import HYPHEN, LETTERS, LF, OUTSIDE_RE, RANGE_WORDS
from app.evidence.numgrammar.equivalents import fold
from app.evidence.numgrammar.neighbours import punct_only
from app.evidence.read_cache import read
from app.evidence.word_lists import AND, BETWEEN, BOUND_WORDS, EQUAL, OR, THAN, TO
from app.evidence.word_match import loose_in

CONNECTORS = TO | OR | AND | THAN | BETWEEN | EQUAL | BOUND_WORDS      # the words of a bound or range phrase


def flag(code: str, severity: str, message: str, p: dict[str, Any]) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "field": p["field"],
            "group": p.get("group_label") or "", "origin": "script"}


def token_context(loc: dict[str, Any], page: dict[str, Any]) -> tuple[str, int, int] | None:
    """(text, start, end) of the verified token: the cited cell for a table locator, else the page
    text or the column the span indexes; None when the locator has no usable token."""
    token = loc.get("token")
    if not (isinstance(token, list) and len(token) == 2 and all(isinstance(t, int) for t in token)):
        return None
    if loc.get("kind") == "table":
        cell = loc.get("cell") or {}
        try:
            text = str((page.get("tables") or [])[cell["table"]]["rows"][cell["row"]][cell["col"]] or "")
        except (KeyError, IndexError, TypeError):
            return None
    else:
        column, columns = loc.get("column"), page.get("columns") or []
        text = str(columns[column] if isinstance(column, int) and 0 <= column < len(columns)
                   else page.get("text") or "")
    a, b = token
    return (text, a, b) if 0 <= a < b <= len(text) else None


def mode(loc: dict[str, Any]) -> str:
    """How the verified token's text is read: a table cell is one line, page text splits at line feeds."""
    return "cell" if loc.get("kind") == "table" else "page"


def verified_token(loc: dict[str, Any], page: dict[str, Any]) -> tuple[str, int, int, Tok | None] | None:
    """(text, start, end, the token the verifier matched there or None) from the locator's stored token."""
    ctx = token_context(loc, page)
    if ctx is None:
        return None
    text, a, b = ctx
    style = loc.get("decimal_style") or "."
    return text, a, b, next((t for t in read(text, style, mode(loc)).toks if (t.start, t.end) == (a, b)), None)


def cell_row(loc: dict[str, Any], page: dict[str, Any]) -> tuple[list[Any], int] | None:
    """(the row of the cited table cell, its column), None for a text locator or an unusable cell locator."""
    if loc.get("kind") != "table":
        return None
    cell = loc.get("cell") or {}
    try:
        return (page.get("tables") or [])[cell["table"]]["rows"][cell["row"]], cell["col"]
    except (KeyError, IndexError, TypeError):
        return None


def _connector(text: str) -> bool:
    """A cell that is one word of a bound or range phrase ('to', 'or', 'and', 'than', 'between', 'least', 'equal',
    ``word_lists``; look-alike spellings too): the bound checks read on through it (r7 round 4)."""
    words = gaps.split(text)
    return len(words) == 1 and loose_in(words[0], CONNECTORS)


def _walk(row: list[Any], col: int, side: int, window: bool) -> list[str]:
    """The cells from *col* + *side* outwards up to and including the first one that holds a word. With *window*, a
    cell that is one lone range word ('-', '~') ends the walk unread: it is a MARKER cell (a sign or a range dash a
    ruling split off), flagged censored_value by the bound checks, never a word of the number's window. Without it (the
    bound checks, r7 round 4) the walk goes on through a CONNECTOR cell and through a number cell right after one, so
    a phrase split into cells is read whole: ['12', 'to', '15'], ['less', 'than', '12'], ['between', '12', 'and',
    '15']."""
    out, k, joined = [], col + side, False
    while 0 <= k < len(row):
        text = str(row[k] or "")
        if window and fold(gaps.strip(text)) in RANGE_WORDS:
            break
        out.append(text)
        if not (gaps.blank(text) or punct_only(gaps.strip(text))):
            linked = not window and (_connector(text) or (joined and is_data_cell(text)))
            if not linked:
                break
            joined = _connector(text)
        k += side
    return out


def cell_context(row: list[Any], col: int, *, window: bool = False) -> tuple[str, int]:
    """(the cited cell joined with the cells one separator away on each side, the offset of the cited cell in it);
    *window*: the reading of the neighbour-word allow-list (a lone range-word cell is a marker, see ``_walk``)."""
    before = " ".join(reversed(_walk(row, col, -1, window)))
    after = " ".join(_walk(row, col, 1, window))
    cell = str(row[col] or "") if 0 <= col < len(row) else ""
    return f"{before} {cell} {after}", len(before) + 1


def context(loc: dict[str, Any], page: dict[str, Any], text: str) -> tuple[str, int]:
    """(the text the words around the token are read in, the offset of *text* in it): the page text or column itself,
    or for a table cell its ``cell_context``."""
    found = cell_row(loc, page)
    return (text, 0) if found is None else cell_context(*found)


# ------------------------------------------------------------------ words wrapped at a line end (r7 round 3, 4) ---
def _wraps(t: str, w: str, gap_end: int, nxt: int) -> bool:
    """The word *w* ends in a letter and a hyphen (or a character the reader cannot read, a soft hyphen), a line break
    stands in the gap before position *nxt*, and the word there starts with a letter: one word the line end broke."""
    return len(w) > 1 and w[-2] in LETTERS and (w[-1] in HYPHEN or OUTSIDE_RE.match(w[-1]) is not None) \
        and LF in t[gap_end:nxt] and t[nxt] in LETTERS


def wrapped_words(t: str, spans: list[tuple[int, int]]) -> list[str]:
    """The words of *spans* (in reading order, over the folded text *t*) with every word broken at a line end rejoined
    ('approxi-' + 'mately', 'mil-' + 'lion', 'thou' U+2010 + 'sand')."""
    words: list[str] = []
    i = 0
    while i < len(spans):
        s, e = spans[i]
        w = t[s:e]
        if i + 1 < len(spans) and _wraps(t, w, e, spans[i + 1][0]):
            words.append(w[:-1] + t[spans[i + 1][0]:spans[i + 1][1]])
            i += 2
            continue
        words.append(w)
        i += 1
    return words


def spans_after(t: str, end: int, n: int) -> list[tuple[int, int]]:
    """[a, b) of up to *n* whole words after the word that holds the offset *end* - 1."""
    i = max(end - 1, 0)
    while i < len(t) and t[i] not in GAP:
        i += 1
    out: list[tuple[int, int]] = []
    while len(out) < n:
        while i < len(t) and t[i] in GAP:
            i += 1
        if i >= len(t):
            break
        j = i
        while j < len(t) and t[j] not in GAP:
            j += 1
        out.append((i, j))
        i = j
    return out


def words_after_joined(text: str, end: int, n: int) -> list[str]:
    """The next *n* whole words after the token's word, a word broken at a line end rejoined (r7 round 4)."""
    t = fold(text)
    return wrapped_words(t, spans_after(t, end, n + 1))[:n]


__all__ = ["cell_context", "cell_row", "context", "flag", "mode", "spans_after", "token_context", "verified_token",
           "words_after_joined", "wrapped_words"]
