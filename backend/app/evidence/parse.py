"""PDF -> parse JSON (pages, body text, small glyphs, columns, tables, metadata, decimal style).

Deterministic, pdfplumber-only. Body text keeps characters whose size is at least
0.75 x the modal character size of the page, so superscript/subscript glyphs
land in `small_text` instead of corrupting numbers. Tables come from ruled lines
first; unruled tables are recovered below a 'Table N' caption by splitting each
text line on large horizontal gaps (strategy 'layout', lower confidence).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pdfplumber

PARSER_NAME = f"pdfplumber {pdfplumber.__version__}"
BODY_SIZE_RATIO = 0.75          # chars smaller than this x modal size are 'small_text'
COLUMN_GAP_FRACTION = 0.08      # minimum empty band (fraction of page width) between two columns
COLUMN_MIN_WORDS = 20
COLUMN_MIN_SPLIT_LINES = 8      # a page needs this many lines with text on both sides
COLUMN_LINE_FRACTION = 0.5      # share of text lines that must have words on both sides
COLUMN_LINE_CROSSING = 0.15     # share of text lines allowed to span the gap (title, abstract)
CELL_GAP_FACTOR = 2.0           # horizontal gap (x modal size) that separates layout-table cells
LINE_Y_TOLERANCE = 3.0
CAPTION_RE = re.compile(r"^Table\s+\d+")
STOP_RE = re.compile(r"^(Table|Figure|Fig\.)\s+\d+")
MAX_LAYOUT_ROWS = 80
_META_KEYS = ("title", "author", "subject", "creator", "producer", "creation_date")
_COMMA_DECIMAL = re.compile(r"\d,\d{1,2}(?!\d)")
_POINT_DECIMAL = re.compile(r"\d\.\d+")
_POINT_THOUSANDS = re.compile(r"\b\d{1,3}(?:\.\d{3})+,\d+")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_decimal_style(text: str) -> str:
    """',' when decimal-comma evidence outweighs decimal-point evidence, else '.'."""
    comma = len(_POINT_THOUSANDS.findall(text))
    text = _POINT_THOUSANDS.sub(" ", text)
    comma += len(_COMMA_DECIMAL.findall(text))
    point = len(_POINT_DECIMAL.findall(text))
    return "," if comma > point else "."


def _metadata(raw: dict[str, Any] | None) -> dict[str, str]:
    raw = raw or {}
    lookup = {"creation_date": "CreationDate"}
    out = {}
    for key in _META_KEYS:
        value = raw.get(lookup.get(key, key.capitalize()))
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        out[key] = ("" if value is None else str(value))[:500]
    return out


def _modal_size(chars: list[dict[str, Any]]) -> float:
    if not chars:
        return 0.0
    counts = Counter(round(float(c.get("size") or 0.0), 1) for c in chars)
    top = max(counts.values())
    return max(size for size, n in counts.items() if n == top)


def _lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _columns(words: list[dict[str, Any]], lines: list[dict[str, Any]], width: float) -> list[str] | None:
    """Two columns when a vertical band >= 8% of the page width is empty on nearly every line."""
    if len(words) < COLUMN_MIN_WORDS or not lines:
        return None
    n = int(width) + 1
    coverage = [0] * n                       # how many text lines cover each x position
    for ln in lines:
        hit = [False] * n
        for w in ln["words"]:
            for x in range(max(0, int(w["x0"])), min(n, int(w["x1"]) + 2)):
                hit[x] = True
        for x, h in enumerate(hit):
            coverage[x] += h
    allowed = max(2, int(COLUMN_LINE_CROSSING * len(lines)))
    lo, hi = int(0.25 * width), int(0.75 * width)
    best, run_start = (0, 0, 0), None
    for x in range(lo, hi + 1):
        if coverage[x] <= allowed:
            run_start = x if run_start is None else run_start
            if x - run_start > best[0]:
                best = (x - run_start, run_start, x)
        else:
            run_start = None
    length, gap0, gap1 = best
    if length < COLUMN_GAP_FRACTION * width:
        return None
    left_lines, right_lines, split, n_left, n_right = [], [], 0, 0, 0
    for ln in lines:                        # lines that span the gap (title, abstract) join neither column
        lw = [w["text"] for w in ln["words"] if w["x1"] <= gap0 + 1]
        rw = [w["text"] for w in ln["words"] if w["x0"] >= gap1 - 1]
        if len(lw) + len(rw) < len(ln["words"]):
            continue
        n_left, n_right = n_left + len(lw), n_right + len(rw)
        split += bool(lw and rw)
        if lw:
            left_lines.append(" ".join(lw))
        if rw:
            right_lines.append(" ".join(rw))
    if min(n_left, n_right) < 0.2 * len(words):
        return None
    if split < max(COLUMN_MIN_SPLIT_LINES, COLUMN_LINE_FRACTION * len(lines)):
        return None
    return ["\n".join(left_lines), "\n".join(right_lines)]


def _clean_cell(cell: Any) -> str:
    return " ".join(str(cell).split()) if cell is not None else ""


def _caption_above(lines: list[dict[str, Any]], top: float) -> str | None:
    above = [ln for ln in lines if ln["bottom"] <= top + 1 and top - ln["bottom"] < 40
             and CAPTION_RE.match(ln["text"])]
    return above[-1]["text"] if above else None


def _lines_tables(body, lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, t in enumerate(body.find_tables()):
        rows = [[_clean_cell(c) for c in row] for row in t.extract()]
        rows = [r for r in rows if any(r)]
        if not rows:
            continue
        out.append({"index": i, "strategy": "lines", "bbox": [round(float(v), 2) for v in t.bbox],
                    "caption": _caption_above(lines, t.bbox[1]), "rows": rows})
    return out


def _cells(line: dict[str, Any], gap: float) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for w in line["words"]:
        if cells and w["x0"] - cells[-1]["x1"] <= gap:
            cells[-1]["text"] += " " + w["text"]
            cells[-1]["x1"] = w["x1"]
        else:
            cells.append({"text": w["text"], "x0": w["x0"], "x1": w["x1"]})
    return cells


def _layout_table(lines: list[dict[str, Any]], start: int, gap: float) -> dict[str, Any] | None:
    """Rows below a 'Table N' caption line, split on large gaps; 1-cell rows merge upward."""
    caption, i = lines[start]["text"], start + 1
    while i < len(lines) and i - start <= 2 and len(_cells(lines[i], gap)) < 2:
        caption += " " + lines[i]["text"]; i += 1
    rows: list[list[str]] = []
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
            rows.append([c["text"] for c in cells])
        elif rows and col_x and (cells[0]["x1"] - cells[0]["x0"]) < 0.5 * (col_x[-1] - col_x[0] + 1):
            j = min(range(len(col_x)), key=lambda k: abs(col_x[k] - cells[0]["x0"]))
            while len(rows[-1]) <= j:
                rows[-1].append("")
            rows[-1][j] = (rows[-1][j] + " " + cells[0]["text"]).strip()
        else:
            break
        prev_bottom = ln["bottom"]
        i += 1
    if len(rows) < 2:
        return None
    return {"strategy": "layout", "bbox": None, "caption": caption.strip(), "rows": rows}


def _tables(body, lines: list[dict[str, Any]], modal: float) -> list[dict[str, Any]]:
    tables = _lines_tables(body, lines)
    if tables:
        return tables
    gap = CELL_GAP_FACTOR * (modal or 10.0)
    for idx, ln in enumerate(lines):
        if CAPTION_RE.match(ln["text"]):
            table = _layout_table(lines, idx, gap)
            if table:
                table["index"] = len(tables)
                tables.append(table)
    return tables


def _parse_page(page) -> dict[str, Any]:
    chars = page.chars
    modal = _modal_size(chars)
    threshold = BODY_SIZE_RATIO * modal
    body = page.filter(lambda o: o.get("object_type") != "char" or float(o.get("size") or 0) >= threshold)
    small = page.filter(lambda o: o.get("object_type") == "char" and float(o.get("size") or 0) < threshold)
    text = body.extract_text() or ""
    small_text = (small.extract_text() or "") if len(small.chars) else ""
    words = body.extract_words()
    lines = _lines(words)
    return {
        "n": page.page_number,
        "text": text,
        "small_text": small_text,
        "columns": _columns(words, lines, float(page.width)),
        "tables": _tables(body, lines, modal),
    }


def parse_pdf(path: str | Path) -> dict[str, Any]:
    """Parse a PDF file into the parse-JSON contract (see module docstring)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    sha = file_sha256(path)
    with pdfplumber.open(str(path)) as pdf:
        metadata = _metadata(pdf.metadata)
        pages = [_parse_page(pg) for pg in pdf.pages]
    full_text = "\n".join(p["text"] for p in pages)
    return {
        "sha256": sha,
        "parser": PARSER_NAME,
        "decimal_style": infer_decimal_style(full_text),
        "text_chars": sum(len(p["text"]) for p in pages),
        "metadata": metadata,
        "pages": pages,
    }
