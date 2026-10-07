"""pdfplumber side of the ink check: each pdfplumber char takes the verdict of the pdfium glyph at
its origin (parse_ink module docstring, step 4).

A pdfplumber whitespace char at whose exact origin pdfium draws a VISIBLE number glyph that no
other char takes (a minus drawn from WinAnsi byte 0xAD that pdfminer decodes as a space) is
respelled with pdfium's text, so '-20' never reads as '20'.
"""
from __future__ import annotations

import unicodedata
from typing import Any

import numpy as np

from app.evidence.numgrammar.alphabet import CID_MARKER
from app.evidence.numgrammar.alphabet import NUMBER_CHARS as OVERLAP_CRITICAL
from app.evidence.numgrammar.equivalents import fold

MATCH_PT = 0.5
EXACT_PT = 0.05                    # origins of one glyph agree to ~1e-4 pt between the two libraries
MAX_UNMATCHED = 0.2
DRIFT_PT = 3.0                     # a run's origins may drift this far apart (Standard-14 widths)
MISMATCH, INK_MISMATCH = "glyph_text_mismatch", "glyph_ink_mismatch"


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _pdfminer_ctm(page: Any) -> np.ndarray:
    """pdfminer's page CTM (pdf user space -> its device space) as a 3x3 matrix."""
    x0, y0, x1, y1 = page.page_obj.mediabox
    a, b, c, d, e, f = {90: (0, -1, 1, 0, -y0, x1), 180: (-1, 0, 0, -1, x1, y1),
                        270: (0, 1, -1, 0, y1, -x0)}.get(page.page_obj.rotate, (1, 0, 0, 1, -x0, -y0))
    return np.array([[a, c, e], [b, d, f], [0, 0, 1]], dtype=float)


def _decoded_apart(text: str, exact: list[int], texts: list[str]) -> set[str]:
    """Number characters of a char and the glyphs at its exact origin that decode to OTHER text
    (pdfium follows /ActualText, pdfminer reads the drawn code): the text layer is in doubt. A side
    that could not decode the glyph at all ('(cid:1)') is no evidence; a decode outside the alphabet (a control, a
    private-use code, U+FFFD) that differs from the text layer is NOT: it is a glyph in doubt. Texts are compared
    through the approved equivalents table, never NFKC or a dash list."""
    if CID_MARKER.match(text):
        return set()
    decoded = [t for t in (fold(texts[k]) for k in exact) if t]
    out = {ch for t in decoded if t != fold(text) for ch in t + fold(text)}
    return out & OVERLAP_CRITICAL


def _drifted(prev: int | None, ux: float, uy: float, text: str, origins: np.ndarray, norm: list[str]) -> list:
    """[(0, k)] for the glyph right after *prev* in pdfium's order when it has this char's text on the
    same baseline within DRIFT_PT: the two libraries' advances disagree (a Standard-14 width) and
    every later origin of the run is shifted."""
    for k in range(prev + 1, min(prev + 4, len(norm))) if prev is not None else ():
        if (norm[k] and (norm[k] in text or text in norm[k]) and abs(origins[k, 1] - uy) <= EXACT_PT
                and abs(origins[k, 0] - ux) <= DRIFT_PT):
            return [(0.0, k)]
    return []


def _union(boxes: np.ndarray, ks: list[int]) -> tuple[float, float, float, float]:
    sel = boxes[ks]
    return (float(sel[:, 0].min()), float(sel[:, 1].min()), float(sel[:, 2].max()), float(sel[:, 3].max()))


def _transfer(page: Any, chars: list[dict[str, Any]], texts: list[str], origins: np.ndarray,
              visible: np.ndarray, odd: np.ndarray, glyph_boxes: np.ndarray | None = None
              ) -> tuple[set[int], dict[str, set[str]], dict[int, str], dict[int, tuple[float, ...]]] | None:
    """(ids of pdfplumber chars whose matching pdfium glyph is hidden (or that have none),
    {glyph_check code: number characters in doubt}, {char id: pdfium text} for whitespace chars that
    draw a visible number glyph, {char id: [l, b, r, t] union of the visible matched glyphs' tight boxes
    in user space} (empty without *glyph_boxes*)); None when too many chars have no glyph."""
    grid: dict[tuple[int, int], list[int]] = {}
    for k, (x, y) in enumerate(origins):
        grid.setdefault((int(x // 2), int(y // 2)), []).append(k)
    inv = np.linalg.inv(_pdfminer_ctm(page))
    norm = [_norm(t) for t in texts]
    hidden, unmatched, judged, prev = set(), 0, 0, None
    flags: dict[str, set[str]] = {MISMATCH: set(), INK_MISMATCH: set()}
    spaces: dict[int, list[int]] = {}             # whitespace char -> visible number glyphs at its origin
    claimed: set[int] = set()
    found: dict[int, tuple[float, ...]] = {}
    for c in chars:
        raw = str(c.get("text", ""))          # identity is judged on the raw text; the NFKC text only pairs glyphs
        text = _norm(raw)
        ux, uy, _ = inv @ np.array([float(c["matrix"][4]), float(c["matrix"][5]), 1.0])
        near = [k for i in (-1, 0, 1) for j in (-1, 0, 1) for k in grid.get((int(ux // 2) + i, int(uy // 2) + j), [])]
        near = [(float(np.hypot(origins[k, 0] - ux, origins[k, 1] - uy)), k) for k in near]
        if not text.strip():
            spaces[id(c)] = [k for d, k in sorted(near)
                             if d <= EXACT_PT and visible[k] and fold(texts[k]) in OVERLAP_CRITICAL]
            continue
        judged += 1
        same = sorted((d, k) for d, k in near if d <= MATCH_PT and norm[k] and (norm[k] in text or text in norm[k]))
        same = same or _drifted(prev, ux, uy, text, origins, norm)
        if same:
            used = [k for d, k in same if d <= same[0][0] + EXACT_PT]
            ok = any(visible[k] for k in used)
        else:                      # same glyph, decoded differently ('(cid:1)', U+2010 vs '-'): all must show
            used = [k for d, k in near if d <= EXACT_PT]
            unmatched += not used
            ok = bool(used) and all(visible[k] for k in used)
            flags[MISMATCH] |= _decoded_apart(raw, used, texts)
        prev = used[0] if used else None
        claimed.update(used)
        shown = [k for k in used if visible[k]]
        if glyph_boxes is not None and ok and shown:
            found[id(c)] = _union(glyph_boxes, shown)
        if any(odd[k] for k in used):
            flags[INK_MISMATCH] |= set(fold(raw)) & OVERLAP_CRITICAL
        if not ok:
            hidden.add(id(c))
    if judged and unmatched > MAX_UNMATCHED * judged:
        return None
    free = {cid: [k for k in ks if k not in claimed] for cid, ks in spaces.items()}
    respell = {cid: fold(texts[ks[0]]) for cid, ks in free.items() if ks}
    if glyph_boxes is not None:
        found.update({cid: _union(glyph_boxes, [ks[0]]) for cid, ks in free.items() if ks})
    return hidden, {code: chars_ for code, chars_ in flags.items() if chars_}, respell, found
