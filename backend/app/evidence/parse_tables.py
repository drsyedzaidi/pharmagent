"""Text lines and tables of one parsed page (pdfplumber words in, rows and cell word lists out).

Ruled tables come from pdfplumber's `lines` strategy; when it finds none, a 'Table N' caption starts
a `layout` fallback that splits each text line on large horizontal gaps and merges continuation rows
(lower confidence: records from it are flagged `table_low_confidence`). Besides the table dicts that
land in the parse JSON, every table returns {"row:col": the cell's words}, which the layout sidecar
(parse_layout) turns into cell boxes; nothing here changes the parse's text.
"""
from __future__ import annotations

import re
from typing import Any

from app.evidence import gaps
from app.evidence.numgrammar.alphabet import DIGITS, GAP, char_class
from app.evidence.parse_guard import _is_white, tables_affordable

CELL_GAP_FACTOR = 2.0           # horizontal gap (x modal size) that separates layout-table cells
LINE_Y_TOLERANCE = 3.0
_GAPS, _DIGITS = char_class(GAP, negate=False) + "+", char_class(DIGITS, negate=False) + "+"   # r7 round 3: the lists
CAPTION_RE = re.compile("^Table" + _GAPS + _DIGITS)
STOP_RE = re.compile("^(Table|Figure|Fig\\.)" + _GAPS + _DIGITS)
MAX_LAYOUT_ROWS = 80


def text_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group words into text lines by vertical position (like extract_text does)."""
    lines: list[dict[str, Any]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(w["top"] - lines[-1]["top"]) <= LINE_Y_TOLERANCE:
            lines[-1]["words"].append(w)
        else:
            lines.append({"top": w["top"], "bottom": w["bottom"], "words": [w]})
    for ln in lines:
        ln["words"].sort(key=lambda w: w["x0"])
        ln["bottom"] = max(w["bottom"] for w in ln["words"])
        ln["x0"], ln["x1"] = ln["words"][0]["x0"], max(w["x1"] for w in ln["words"])
        ln["text"] = " ".join(w["text"] for w in ln["words"])
    return lines


def _inked(o: dict[str, Any]) -> bool:
    """False for a path that puts no ink on a white page (white stroke / fill, nothing painted): it
    must not become a table ruling that splits a cell a reader sees whole."""
    if o.get("object_type") not in ("line", "rect", "curve"):
        return True
    stroke = bool(o.get("stroke")) and not _is_white(o.get("stroking_color"))
    return stroke or (bool(o.get("fill")) and not _is_white(o.get("non_stroking_color")))


def _clean_cell(cell: Any) -> str:
    return gaps.collapse(str(cell)) if cell is not None else ""


def _caption_above(lines: list[dict[str, Any]], top: float) -> str | None:
    above = [ln for ln in lines if ln["bottom"] <= top + 1 and top - ln["bottom"] < 40
             and CAPTION_RE.match(ln["text"])]
    return above[-1]["text"] if above else None


def _in_cell(w: dict[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    """A word belongs to the cell its centre lies in (pdfplumber's own rule for characters)."""
    x0, top, x1, bottom = bbox
    return x0 <= (w["x0"] + w["x1"]) / 2 < x1 and top <= (w["top"] + w["bottom"]) / 2 < bottom


def _cell_words_in(words: list[dict[str, Any]], bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    inside = [w for w in words if _in_cell(w, bbox)]
    return [w for ln in text_lines(inside) for w in ln["words"]] if inside else []


def _lines_tables(body, lines: list[dict[str, Any]], words: list[dict[str, Any]]
                  ) -> tuple[list[dict[str, Any]], list[dict[str, list[dict[str, Any]]]]]:
    """(tables, per table {"row:col": the cell's words}) from ruled lines."""
    out: list[dict[str, Any]] = []
    cells: list[dict[str, list[dict[str, Any]]]] = []
    if not tables_affordable(body):          # dense ruling grid: cell finding would explode
        return out, cells
    for i, t in enumerate(body.filter(_inked).find_tables()):
        every = [[_clean_cell(c) for c in row] for row in t.extract()]
        keep = [k for k, r in enumerate(every) if any(r)]
        rows = [every[k] for k in keep]
        if not rows:
            continue
        out.append({"index": i, "strategy": "lines", "bbox": [round(float(v), 2) for v in t.bbox],
                    "caption": _caption_above(lines, t.bbox[1]), "rows": rows})
        cells.append({f"{r}:{c}": _cell_words_in(words, bbox)
                      for r, k in enumerate(keep) for c, bbox in enumerate(t.rows[k].cells) if bbox})
    return out, cells


def _cells(line: dict[str, Any], gap: float) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for w in line["words"]:
        if cells and w["x0"] - cells[-1]["x1"] <= gap:
            cells[-1]["text"] += " " + w["text"]
            cells[-1]["x1"] = w["x1"]
            cells[-1]["words"].append(w)
        else:
            cells.append({"text": w["text"], "x0": w["x0"], "x1": w["x1"], "words": [w]})
    return cells


def _place(cells: list[dict[str, Any]], col_x: list[float]) -> list[dict[str, Any] | None]:
    """Cells under their header columns: a row with fewer cells than the header (a blank or hidden
    cell) puts each cell in the nearest column left to right, so no value shifts into its neighbour's
    (None marks a blank column)."""
    if len(cells) >= len(col_x):
        return list(cells)
    out: list[dict[str, Any] | None] = [None] * len(col_x)
    nxt = 0
    for k, c in enumerate(cells):
        last = len(col_x) - (len(cells) - k)                # leave a column for every later cell
        j = min(range(nxt, last + 1), key=lambda i: abs(col_x[i] - c["x0"]))
        out[j], nxt = c, j + 1
    return out


def _layout_table(lines: list[dict[str, Any]], start: int, gap: float
                  ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]] | None:
    """Rows below a 'Table N' caption line, split on large gaps; 1-cell rows merge upward. Returns the
    table and its {"row:col": words} cell map."""
    caption, i = lines[start]["text"], start + 1
    while i < len(lines) and i - start <= 2 and len(_cells(lines[i], gap)) < 2:
        caption += " " + lines[i]["text"]; i += 1
    rows: list[list[str]] = []
    words: list[list[list[dict[str, Any]]]] = []
    col_x: list[float] = []
    prev_bottom = lines[start]["bottom"]
    line_h = max(1.0, lines[start]["bottom"] - lines[start]["top"])
    while i < len(lines) and len(rows) < MAX_LAYOUT_ROWS:
        ln = lines[i]
        cells = _cells(ln, gap)
        if STOP_RE.match(ln["text"]) or ln["top"] - prev_bottom > 2.0 * line_h:
            break
        if len(cells) >= 2:
            if not rows:
                col_x = [c["x0"] for c in cells]
            placed = _place(cells, col_x)
            rows.append([c["text"] if c else "" for c in placed])
            words.append([list(c["words"]) if c else [] for c in placed])
        elif rows and col_x and (cells[0]["x1"] - cells[0]["x0"]) < 0.5 * (col_x[-1] - col_x[0] + 1):
            j = min(range(len(col_x)), key=lambda k: abs(col_x[k] - cells[0]["x0"]))
            while len(rows[-1]) <= j:
                rows[-1].append("")
                words[-1].append([])
            rows[-1][j] = gaps.strip(rows[-1][j] + " " + cells[0]["text"])
            words[-1][j] = words[-1][j] + cells[0]["words"]
        else:
            break
        prev_bottom = ln["bottom"]
        i += 1
    if len(rows) < 2:
        return None
    table = {"strategy": "layout", "bbox": None, "caption": gaps.strip(caption), "rows": rows}
    return table, {f"{r}:{c}": ws for r, row in enumerate(words) for c, ws in enumerate(row) if ws}


def find_tables(body, lines: list[dict[str, Any]], words: list[dict[str, Any]], modal: float
            ) -> tuple[list[dict[str, Any]], list[dict[str, list[dict[str, Any]]]]]:
    """(tables, per table {"row:col": the cell's words}); ruled lines first, else layout tables."""
    tables, cells = _lines_tables(body, lines, words)
    if tables:
        return tables, cells
    gap = CELL_GAP_FACTOR * (modal or 10.0)
    for idx, ln in enumerate(lines):
        if CAPTION_RE.match(ln["text"]):
            found = _layout_table(lines, idx, gap)
            if found:
                table, cell_words = found
                table["index"] = len(tables)
                tables.append(table)
                cells.append(cell_words)
    return tables, cells
