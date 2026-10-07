"""PDF -> parse JSON (pages, body text, small glyphs, columns, tables, metadata, decimal style).

Deterministic, pdfplumber-only. Body text keeps characters whose size is at least
0.75 x the modal character size of the page, so superscript/subscript glyphs
land in `small_text` instead of corrupting numbers; so do number glyphs raised above
the baseline of a larger glyph right before them (an exponent at 80 % size). Each page's
`glyph_check` names the number characters whose drawn glyph may not be the text (parse_fonts,
parse_ink.overlapping_chars). Tables come from ruled lines
first; unruled tables are recovered below a 'Table N' caption by splitting each
text line on large horizontal gaps (strategy 'layout', lower confidence).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import pdfplumber

from app.evidence import parse_layout
from app.evidence.normalize import infer_decimal_style
from app.evidence.numgrammar.alphabet import DIGITS, SIGN_CHARS
from app.evidence.numgrammar.equivalents import EQUIV
from app.evidence.parse_budget import check_glyph_budget
from app.evidence.parse_fonts import MISMATCH, UNVERIFIED, GlyphChecker
from app.evidence.parse_guard import (
    Deadline,
    check_page_box,
    check_streams,
    close_zero_gaps,
    lay_out_visible,
    protect_thin_spaces,
    visibility_filter,
)
from app.evidence.parse_ink import InkChecker, duplicate_ids
from app.evidence.parse_marks import orientation_outliers, overlapping_chars
from app.evidence.parse_tables import find_tables, text_lines

# The suffix versions this module's output: a parse cache written by an earlier version (no
# visibility filter, thin spaces turned into word gaps, no rendered-ink check, no round-3 glyph / space
# checks, no layout sidecar) is re-parsed. Version 7 only ADDS the layout sidecar (parse_layout): every
# text-bearing field is byte-identical to version 6, because stored record spans index them.
PARSER_NAME = f"pdfplumber {pdfplumber.__version__} / evidence-parse 7"
BODY_SIZE_RATIO = 0.75          # chars smaller than this x modal size are 'small_text'
COLUMN_GAP_FRACTION = 0.08      # minimum empty band (fraction of page width) between two columns
COLUMN_MIN_WORDS = 20
COLUMN_MIN_SPLIT_LINES = 8      # a page needs this many lines with text on both sides
COLUMN_LINE_FRACTION = 0.5      # share of text lines that must have words on both sides
COLUMN_LINE_CROSSING = 0.15     # share of text lines allowed to span the gap (title, abstract)
RAISABLE = DIGITS | SIGN_CHARS | frozenset(k for k, v in EQUIV.items() if v in SIGN_CHARS)   # exponent glyphs
RAISED_SIZE_RATIO = 0.95        # an exponent is set smaller than its base ...
RAISE_FRACTION = 0.15           # ... with its baseline at least this x the base size higher
_META_KEYS = ("title", "author", "subject", "creator", "producer", "creation_date")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _line_cover(lines: list[dict[str, Any]], n: int) -> list[tuple[int, int]]:
    """Coverage change points: +1 / -1 at the integer x where each line's merged word cover starts / ends."""
    events: Counter[int] = Counter()
    for ln in lines:
        spans = sorted((max(0, int(w["x0"])), min(n, int(w["x1"]) + 2)) for w in ln["words"])
        cur: list[int] | None = None
        for a, b in spans + [(n + 1, n + 1)]:
            if cur is not None and a <= cur[1]:
                cur[1] = max(cur[1], b)
                continue
            if cur is not None and cur[0] < cur[1]:
                events[cur[0]] += 1
                events[cur[1]] -= 1
            cur = [a, b]
    return sorted(events.items())


def _widest_gap(lines: list[dict[str, Any]], width: float, allowed: int) -> tuple[int, int, int]:
    """(length, first x, last x) of the first longest run of integer x in [25%, 75%] of the width
    covered by <= *allowed* lines (interval sweep: no per-x allocation, whatever the page width)."""
    n = int(width) + 1
    lo, hi = int(0.25 * width), int(0.75 * width)
    best, run_start, cov, x = (0, 0, 0), None, 0, lo
    points = _line_cover(lines, n)
    cov = sum(d for p, d in points if p <= lo)
    for p, d in [(p, d) for p, d in points if p > lo] + [(hi + 1, 0)]:
        seg_end = min(p, hi + 1) - 1                 # coverage is `cov` on [x, seg_end]
        if seg_end >= x:
            if cov <= allowed:
                run_start = x if run_start is None else run_start
                if seg_end - run_start > best[0]:
                    best = (seg_end - run_start, run_start, seg_end)
            else:
                run_start = None
            x = seg_end + 1
        if p > hi:
            break
        cov += d
    return best


def _columns(words: list[dict[str, Any]], lines: list[dict[str, Any]], width: float
             ) -> list[list[list[dict[str, Any]]]] | None:
    """Two columns when a vertical band >= 8% of the page width is empty on nearly every line: per
    column, its lines, each a list of word dicts (_join_columns builds the column strings)."""
    if len(words) < COLUMN_MIN_WORDS or not lines:
        return None
    allowed = max(2, int(COLUMN_LINE_CROSSING * len(lines)))
    length, gap0, gap1 = _widest_gap(lines, width, allowed)
    if length < COLUMN_GAP_FRACTION * width:
        return None
    left_lines, right_lines, split, n_left, n_right = [], [], 0, 0, 0
    for ln in lines:                        # lines that span the gap (title, abstract) join neither column
        lw = [w for w in ln["words"] if w["x1"] <= gap0 + 1]
        rw = [w for w in ln["words"] if w["x0"] >= gap1 - 1]
        if len(lw) + len(rw) < len(ln["words"]):
            continue
        n_left, n_right = n_left + len(lw), n_right + len(rw)
        split += bool(lw and rw)
        if lw:
            left_lines.append(lw)
        if rw:
            right_lines.append(rw)
    if min(n_left, n_right) < 0.2 * len(words):
        return None
    if split < max(COLUMN_MIN_SPLIT_LINES, COLUMN_LINE_FRACTION * len(lines)):
        return None
    return [left_lines, right_lines]


def _join_columns(col_lines: list[list[list[dict[str, Any]]]] | None) -> list[str] | None:
    if col_lines is None:
        return None
    return ["\n".join(" ".join(w["text"] for w in ln) for ln in lines) for lines in col_lines]


def _visible_chars(page, ink: InkChecker) -> tuple[Any, int, str]:
    """(predicate, chars dropped, ink status): colour / geometry rules, duplicate strikes, then ink
    (which leaves the page's glyph doubts in ink.flags)."""
    cheap, dropped = visibility_filter(page)     # white-on-white, squashed, off-page / off-crop glyphs
    candidates = [c for c in page.chars if cheap(c)]
    drop = duplicate_ids(candidates)             # one glyph to a reader; not counted as hidden
    inkless = ink.hidden(page, [c for c in candidates if id(c) not in drop])
    drop |= inkless or set()

    def visible(c: dict[str, Any]) -> bool:
        return cheap(c) and id(c) not in drop

    return visible, dropped + len(inkless or ()), "unverified" if inkless is None else "ok"


def _raised_ids(chars: list[dict[str, Any]]) -> set[int]:
    """ids of number glyphs set smaller than, right after and above the baseline of the glyph before
    them: an exponent ('10' + raised '3') that is too large for small_text must still never join the
    body number ('103') or vanish from it ('1.2 x 10' read as plain numbers)."""
    out: set[int] = set()
    base = None
    for c in chars:                              # content order: the base glyph precedes its exponent
        if base is not None and str(c.get("text")) in RAISABLE:
            size, bsize = float(c.get("size") or 0), float(base.get("size") or 0)
            gap = float(c["x0"]) - float(base["x1"])
            if (size < RAISED_SIZE_RATIO * bsize and -0.5 <= gap < bsize
                    and float(c["bottom"]) < float(base["bottom"]) - RAISE_FRACTION * bsize):
                out.add(id(c))
                continue                         # '10' + raised '-' '3': both measured from the '0'
        base = c
    return out


def _parse_page(page, ink: InkChecker, glyphs: GlyphChecker, doc_words: int = 0
                ) -> tuple[dict[str, Any], dict[str, Any], int]:
    """(the page's parse fields, its layout record, its word count)."""
    check_page_box(page)
    hidden = lay_out_visible(page)               # render modes 3 / 7 never reach the char list
    visible, dropped, ink_status = _visible_chars(page, ink)
    thin = protect_thin_spaces(page)
    close_zero_gaps(page)
    chars = [c for c in page.chars if visible(c)]
    raised = _raised_ids(chars)
    modal = _modal_size(chars)
    threshold = BODY_SIZE_RATIO * modal

    def is_small(o: dict[str, Any]) -> bool:
        return float(o.get("size") or 0) < threshold or id(o) in raised

    def is_body(o: dict[str, Any]) -> bool:
        return o.get("object_type") != "char" or (visible(o) and not is_small(o))

    body = page.filter(is_body)
    small = page.filter(lambda o: o.get("object_type") == "char" and visible(o) and is_small(o))
    text = body.extract_text() or ""
    small_text = (small.extract_text() or "") if len(small.chars) else ""
    words = body.extract_words(return_chars=True)
    lines = text_lines(words)
    col_lines = _columns(words, lines, float(page.width))
    tables, cell_words = find_tables(body, lines, words, modal)
    parsed = thin.restore({
        "n": page.page_number,
        "text": text,
        "small_text": small_text,
        "columns": _join_columns(col_lines),
        "tables": tables,
        "hidden_chars": hidden + dropped,
        "ink_check": ink_status,
        "glyph_check": _glyph_check(page, glyphs, chars, ink.flags),
    })
    layout = parse_layout.build_page(
        page_text=parsed["text"], columns=parsed["columns"], col_lines=col_lines, words=words,
        char_boxes=ink.boxes, cell_words=cell_words, ink_ok=ink_status == "ok", tables=parsed["tables"],
        doc_words=doc_words, thin=thin)
    return parsed, {"n": page.page_number, **layout}, len(words)


def _glyph_check(page, glyphs: GlyphChecker, chars: list[dict[str, Any]],
                 ink_flags: dict[str, set[str]]) -> dict[str, str]:
    """{flag code: characters} whose text may not be the glyph a reader sees (parse_fonts, glyphs
    rotated against their neighbours, decoded apart or drawn odd per the ink check), plus number
    characters over-struck by another glyph; {} when nothing is in doubt."""
    try:
        out = {code: set(v) for code, v in glyphs.page(page).items()}
    except Exception:                            # fail closed: the fonts could not be judged
        out = {UNVERIFIED: set("0123456789")}
    extra = {**ink_flags, MISMATCH: set(ink_flags.get(MISMATCH, ())) | orientation_outliers(chars),
             "overlapping_glyphs": overlapping_chars(chars)}
    merged = {code: out.get(code, set()) | set(extra.get(code, ())) for code in {*out, *extra}}
    return {code: "".join(sorted(v)) for code, v in sorted(merged.items()) if v}


def parse_pdf(path: str | Path) -> dict[str, Any]:
    """Parse a PDF file into the parse-JSON contract (see module docstring)."""
    doc = parse_pdf_with_layout(path)
    doc.pop("layout")
    return doc


def parse_pdf_with_layout(path: str | Path) -> dict[str, Any]:
    """parse_pdf plus a ``layout`` key: the word-box sidecar document (parse_layout), which
    ingest.parse_and_cache stores beside the parse JSON and never inside it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    sha = file_sha256(path)
    deadline = Deadline()
    with pdfplumber.open(str(path)) as pdf:
        check_streams(pdf.doc)
        check_glyph_budget(pdf.pages)            # glyph floods are refused before any layout
        metadata = _metadata(pdf.metadata)
        pages, layouts, n_words = [], [], 0
        ink, glyphs = InkChecker(str(path), len(pdf.pages)), GlyphChecker()
        try:
            for pg in pdf.pages:
                deadline.check(pg.page_number)
                parsed, layout, n = _parse_page(pg, ink, glyphs, n_words)
                pages.append(parsed)
                layouts.append(layout)
                n_words += n
                pg.close()                       # drop the page's cached layout before the next one
        finally:
            ink.close()
    full_text = "\n".join(p["text"] for p in pages)
    return {
        "sha256": sha,
        "parser": PARSER_NAME,
        "decimal_style": infer_decimal_style(full_text),
        "text_chars": sum(len(p["text"]) for p in pages),
        "metadata": metadata,
        "pages": pages,
        "layout": parse_layout.build_document(sha, PARSER_NAME, layouts),
    }
