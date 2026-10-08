"""The table path of the verifier (split out of verify.py in r7 round 3): the cited cell, the cells around it, and the
page-text span of the cited cell's row.

* readability: a character the reader cannot read in a cell one separator away fails the record (``_neighbour_offence``,
  the walk past empty, blank and punctuation-only cells of r7 round 2);
* the WINDOW (r7 round 3, the round-3 audit): the cited number is read again over the cell and the cells one separator
  away (``value_context.cell_context``) and must be the same resolved token there, so a sign or an opener in the
  previous cell ('(-' | '12)'), a decimal tail ('12' | '.5', '0.' | '5'), a power ('12' | '^3'), a slash ('12' | '/15')
  or an operator letter ('1.2 e' | '-3', 'x' | '10^3') refuses it like it does in running text; a cell boundary stays
  structure for a list of signed numbers (owner decision 9: '-11.8' | '-14.2' are two values);
* the row span: the row's cells are matched to a page-text line as WHOLE normalised words, left to right in their order
  (the first line of a wrapped cell), so '50 mg' never matches '50 mg/kg' and a duplicate value lands on the cited
  column; the line that aligns the most other cells wins, a tie is no span ('row not located'), never the first match.
  Each line is read once per record (r7 round 3: the cost was quadratic in a long line).
"""
from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any

from app.evidence import gaps
from app.evidence import models as M
from app.evidence.normalize import COLLAPSED, normalise
from app.evidence.numgrammar import digits_of, tokens, unreadable
from app.evidence.numgrammar.alphabet import LF
from app.evidence.numgrammar.neighbours import punct_only
from app.evidence.read_cache import read
from app.evidence.value_context import cell_context
from app.evidence.verify_messages import context_sentence, tok_sentence

Fail = tuple[str, str]                                  # (reason, reviewer sentence)
_LF_RE = re.compile(re.escape(LF))


def cell_text(cell: Any, page: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """(cell text, table) for a well-formed in-range cell locator, else None."""
    if not isinstance(cell, dict):
        return None
    idx = [cell.get(k) for k in ("table", "row", "col")]
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in idx):
        return None
    t, r, c = idx
    tables = page.get("tables") or []
    if t >= len(tables):
        return None
    rows = tables[t].get("rows") or []
    if r >= len(rows) or c >= len(rows[r]):
        return None
    return str(rows[r][c] or ""), tables[t]


def neighbour_offence(row: list[Any], col: int, literal: str) -> str:
    """The reviewer sentence when a cell one separator away holds a character outside the allowed alphabet or a
    ``(cid:N)`` marker, else ''. The walk on each side goes past the cells that hold no word the allow-list judges."""
    for side, where in ((-1, "in the previous cell"), (1, "in the next cell")):
        k = col + side
        while 0 <= k < len(row):
            text = str(row[k] or "")
            if (bad := unreadable(text)) is not None:
                return context_sentence(literal, bad[1], where)
            if not (gaps.blank(text) or punct_only(gaps.strip(text))):
                break
            k += side
    return ""


def window_fault(row: list[Any], col: int, token: list[int], style: str) -> Fail | None:
    """The cited token read again over the cells one separator away: None when it is the same resolved token there,
    else (reason, sentence) of what the reading makes of it."""
    ctx, off = cell_context(row, col, window=True)
    a, b = token[0] + off, token[1] + off
    reading = read(ctx, style, "cell")
    k = bisect_right(reading.starts, a) - 1
    tok = reading.toks[k] if k >= 0 and reading.toks[k].end > a else None
    if tok is not None and tok.value is not None and (tok.start, tok.end) == (a, b):
        return None
    if tok is not None and tok.value is None:
        return tok.reason, tok_sentence(tok, ctx)
    return M.REASON_PARTIAL_NUMBER, ""


# ------------------------------------------------------------------------------- the row span ---
def _words(line: str) -> list[tuple[str, int, int]]:
    """(normalised word, raw start, raw end) of every word of *line*: the line is normalised ONCE and split at its
    collapsed spaces; the index map gives each word's raw offsets."""
    norm, m = normalise(line)
    out, at = [], 0
    for w in norm.split(COLLAPSED):
        if w:
            out.append((w, m[at], m[at + len(w) - 1] + 1))
        at += len(w) + 1
    return out


def _cell_words(cell: Any) -> list[str]:
    """The normalised words of a cell's first non-blank line (a wrapped cell prints that line on the row's line)."""
    first = next((ln for ln in str(cell or "").split(LF) if not gaps.blank(ln)), "")
    return [w for w in normalise(first)[0].split(COLLAPSED) if w] if first else []


def _align(words: list[str], cells: list[list[str]]) -> list[int | None]:
    """The word index each cell starts at, matched as whole words left to right in the row's order (None: not found
    after the cell before)."""
    out: list[int | None] = []
    pos = 0
    for cw in cells:
        hit = next((j for j in range(pos, len(words) - len(cw) + 1) if cw and words[j:j + len(cw)] == cw), None)
        out.append(hit)
        pos = hit + len(cw) if hit is not None else pos
    return out


def row_span(spans: list[list[int]], text: str, row: list[Any], col: int) -> list[int] | None:
    """Page-text span of the cited cell among the snippet's occurrences *spans* (see the module docstring)."""
    if not spans or not 0 <= col < len(row):
        return None
    nl, cells = [m.start() for m in _LF_RE.finditer(text)], [_cell_words(c) for c in row]
    lines: dict[int, tuple[int, list[tuple[str, int, int]], int | None]] = {}
    found: list[tuple[int, list[int]]] = []
    for span in spans:
        k = bisect_right(nl, span[0] - 1)
        ls, le = (nl[k - 1] + 1 if k else 0), (nl[k] if k < len(nl) else len(text))
        if ls not in lines:                                 # each line is read once per record
            words = _words(text[ls:le])
            at = _align([w for w, _, _ in words], cells)
            lines[ls] = (sum(1 for i, j in enumerate(at) if i != col and j is not None and cells[i]), words, at[col])
        score, words, j = lines[ls]
        if score and j is not None and words[j][1] + ls <= span[0] \
                and span[1] <= words[j + len(cells[col]) - 1][2] + ls:
            found.append((score, span))
    top = max((score for score, _ in found), default=0)
    best = {tuple(span) for score, span in found if score == top}
    return list(best.pop()) if top and len(best) == 1 else None


def row_fragment(literal: str, text: str, row: list[Any], col: int, style: str) -> bool:
    """True when a page-text line that aligns another cell of the row as whole words prints the cell's literal only as
    PART of a longer number ('2' of a row line reading '281'): a ruling split the number."""
    want, cells = normalise(literal)[0], [_cell_words(c) for c in row]
    for line in text.split(LF):
        at = _align([w for w, _, _ in _words(line)], cells)
        if not any(j is not None and i != col and cells[i] for i, j in enumerate(at)):
            continue
        toks = [normalise(line[t.start:t.end])[0] for t in tokens(line, style, mode="page")]
        if want not in toks and any(want in t and digits_of(want) != digits_of(t) for t in toks):
            return True
    return False


__all__ = ["cell_text", "neighbour_offence", "row_fragment", "row_span", "window_fault"]
