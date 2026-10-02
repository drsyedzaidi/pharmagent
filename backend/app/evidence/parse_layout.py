"""Page layout sidecar: where each word of the parsed page text sits on the PDF page.

``<sha>.layout.json`` (written beside ``<sha>.pages.json``, never inside it) holds, per page:

    {n, status, text_sha256, col_sha256, words, col_words, tables}

- ``words``: ``[start, end, l, b, r, t]`` per word of the page text (code-point offsets into
  ``page.text``), the box in PDF USER SPACE points (pdfium's frame, the one the ink check measures:
  rotation, CropBox and MediaBox origin are the renderer's business, never pdfplumber's displayed
  frame). A word with a character that has no matched, visible pdfium glyph is ``[start, end]`` only.
- ``col_words``: the same per column (offsets into ``page.columns[i]``), or None without columns.
- ``tables``: ``[{index, cells: {"row:col": [l, b, r, t]}}]``; a cell has an entry only when its words
  read exactly the cell's text.
- ``status``: ``ok``; ``no_boxes`` (ink check unverified: the glyph geometry is not trusted);
  ``unmapped`` (the words do not follow the page text); ``skipped`` (word caps).

``text_sha256`` / ``col_sha256`` bind the offsets to the exact texts they index. Nothing here changes
the parse's text: the module only reads it. Every doubt leaves a box out (the image route then
shows the whole page and says so): a wrong box is worse than none.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

from app.evidence.parse_guard import _PLACEHOLDER, restore

MAX_PAGE_WORDS = 6000
MAX_DOC_WORDS = 300_000
LAYOUT_VERSION = 1
Box = tuple[float, float, float, float]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _round_out(box: Box) -> list[float]:
    """[l, b, r, t] to 0.1 pt, rounded outwards so the stored box still contains the glyphs."""
    l, b, r, t = box
    return [math.floor(l * 10) / 10, math.floor(b * 10) / 10, math.ceil(r * 10) / 10, math.ceil(t * 10) / 10]


def word_box(word: dict[str, Any], char_boxes: dict[int, Box]) -> Box | None:
    """Union of the glyph boxes of the word's characters; None when a non-space char has none."""
    boxes: list[Box] = []
    for c in word.get("chars") or ():
        if not str(c.get("text", "")).strip() or c.get("text") == _PLACEHOLDER:
            continue                          # spacing (a protected thin space draws nothing)
        box = char_boxes.get(id(c))
        if box is None:
            return None
        boxes.append(box)
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _word_text(word: dict[str, Any]) -> str:
    return restore(str(word.get("text", "")))


def map_words(text: str, words: list[dict[str, Any]]) -> list[tuple[int, int]] | None:
    """[start, end) of each word in *text*, matching in order and skipping whitespace only; None when a
    word does not start where the cursor stands or non-space text is left over."""
    out: list[tuple[int, int]] = []
    cur = 0
    for w in words:
        wt = _word_text(w)
        if not wt:
            return None
        while cur < len(text) and text[cur].isspace() and not text.startswith(wt, cur):
            cur += 1
        if not text.startswith(wt, cur):
            return None
        out.append((cur, cur + len(wt)))
        cur += len(wt)
    if any(not ch.isspace() for ch in text[cur:]):
        return None
    return out


def _entries(offsets: list[tuple[int, int]], words: list[dict[str, Any]], char_boxes: dict[int, Box]
             ) -> list[list[float]]:
    out: list[list[float]] = []
    for (s, e), w in zip(offsets, words, strict=True):
        box = word_box(w, char_boxes)
        out.append([s, e] if box is None else [s, e, *_round_out(box)])
    return out


def column_offsets(col_text: str, lines: list[list[dict[str, Any]]]) -> list[tuple[int, int]] | None:
    """Offsets of the words of one column string built as '\\n'.join(' '.join(words of a line)); None
    when that rebuild does not give *col_text* back."""
    out: list[tuple[int, int]] = []
    rebuilt: list[str] = []
    cur = 0
    for line in lines:
        texts = [_word_text(w) for w in line]
        for t in texts:
            out.append((cur, cur + len(t)))
            cur += len(t) + 1
        rebuilt.append(" ".join(texts))
    return out if "\n".join(rebuilt) == col_text else None


def cell_boxes(cell_words: dict[str, list[dict[str, Any]]], cell_texts: dict[str, str],
               char_boxes: dict[int, Box]) -> dict[str, list[float]]:
    """{"row:col": [l, b, r, t]} for every cell whose words (in reading order) read exactly its text."""
    out: dict[str, list[float]] = {}
    for key, words in cell_words.items():
        if not words or " ".join(_word_text(w) for w in words) != restore(cell_texts.get(key, "")):
            continue
        boxes = [word_box(w, char_boxes) for w in words]
        if any(b is None for b in boxes):
            continue
        out[key] = _round_out((min(b[0] for b in boxes), min(b[1] for b in boxes),   # type: ignore[index]
                               max(b[2] for b in boxes), max(b[3] for b in boxes)))  # type: ignore[index]
    return out


def _table_cells(tables: list[dict[str, Any]], cell_words: list[dict[str, list[dict[str, Any]]]],
                 char_boxes: dict[int, Box]) -> list[dict[str, Any]]:
    out = []
    for pos, (table, words) in enumerate(zip(tables, cell_words, strict=False)):
        texts = {f"{r}:{c}": str(cell or "") for r, row in enumerate(table.get("rows") or [])
                 for c, cell in enumerate(row)}
        cells = cell_boxes(words, texts, char_boxes)
        if cells:
            out.append({"index": pos, "cells": cells})      # position in page.tables (a locator's cell.table)
    return out


def _empty(page_text: str, columns: list[str] | None, status: str) -> dict[str, Any]:
    return {"status": status, "text_sha256": sha256_text(page_text),
            "col_sha256": None if not columns else [sha256_text(c) for c in columns],
            "words": [], "col_words": None, "tables": []}


def build_page(*, page_text: str, columns: list[str] | None, col_lines: list[list[list[dict[str, Any]]]] | None,
               words: list[dict[str, Any]], char_boxes: dict[int, Box], cell_words: list[dict[str, Any]],
               ink_ok: bool, tables: list[dict[str, Any]], doc_words: int) -> dict[str, Any]:
    """The layout record of one page (without ``n``). *page_text*, *columns* and *tables* are the parse's
    restored page fields; *words* are pdfplumber's (with their chars), *char_boxes* the ink check's
    {char id: glyph box}. Never raises: any failure is an ``unmapped`` page."""
    try:
        if not ink_ok:
            return _empty(page_text, columns, "no_boxes")
        if len(words) > MAX_PAGE_WORDS or doc_words + len(words) > MAX_DOC_WORDS:
            return _empty(page_text, columns, "skipped")
        offsets = map_words(page_text, words)
        if offsets is None:
            return _empty(page_text, columns, "unmapped")
        out = _empty(page_text, columns, "ok")
        out["words"] = _entries(offsets, words, char_boxes)
        if columns and col_lines:
            cols = []
            for text, lines in zip(columns, col_lines, strict=True):
                flat = [w for ln in lines for w in ln]
                offs = column_offsets(text, lines)
                if offs is None:
                    cols = None
                    break
                cols.append(_entries(offs, flat, char_boxes))
            out["col_words"] = cols
        out["tables"] = _table_cells(tables, cell_words, char_boxes)
        return out
    except Exception:                      # fail closed: the page keeps its text, loses its boxes
        return _empty(page_text, columns, "unmapped")


def build_document(sha256: str, parser: str, pages: list[dict[str, Any]]) -> dict[str, Any]:
    return {"sha256": sha256, "parser": parser, "layout_version": LAYOUT_VERSION, "pages": pages}
