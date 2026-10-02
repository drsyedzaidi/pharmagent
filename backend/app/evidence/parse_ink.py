"""Rendered-ink visibility: a glyph is page text only if it puts visible ink on the rendered page.

Colour rules alone cannot see a glyph covered by a later white fill, drawn in 0.97 grey, clipped
away, squashed by '1 Tz', drawn in the colour of the box beneath it, or placed outside the CropBox.
So pdfium renders each page (CropBox region, annotations and form-field Widget appearances drawn: the
document's forms are initialised before any page loads) and the verdict comes from ink:

1. pdfium's text page lists every glyph with its text object, origin and tight ink box (user space).
2. Text objects are grouped so that no two objects in a group overlap (greedy colouring of their
   boxes, at most MAX_GROUPS). B is the page rendered with every text object switched to an
   invisible render mode (clip modes keep their clip); R_g shows only group g's text. Pixels where
   R_g and B differ by at least INK_DELTA (max over RGB, 0-255) are group g's VISIBLE ink: anything
   that covers, clips or matches the text is identical in both renders, and an overlapping hidden
   layer is in another group, so it can neither lend ink to nor take ink from this one.
3. A glyph is visible when at least MIN_INSIDE_FRACTION of its tight box lies on the rendered page
   (CropBox) and that box, plus a PAD_PX ring no other glyph box claims, holds at least
   max(1, INK_FRACTION x region area) ink pixels. Whitespace is never judged.
4. Each pdfplumber char takes the verdict of the pdfium glyph with the same text at the same origin
   (pdfminer's text matrix mapped back through its page CTM; both libraries agree to ~1e-4 pt), the
   nearest one within MATCH_PT; a char whose text the two libraries decode differently ('(cid:1)',
   U+2010 for '-') takes the glyphs at its exact origin (EXACT_PT), all of which must be visible.
   A char with no glyph there takes the next glyph of the run in pdfium's order (same text, same
   baseline, within DRIFT_PT: a Standard-14 width disagreement), else it is dropped as hidden; when
   more than MAX_UNMATCHED of a page's chars have none, the page is not judged. Glyphs at a char's
   origin that decode to other number text (/ActualText) and number glyphs whose ink is odd
   (parse_marks.ink_odd) are reported per page in ``InkChecker.flags``.

Costs are bounded: 2 + groups renders per page at <= RENDER_SCALE (scaled down to MAX_RENDER_PIXELS),
at most RENDER_MAX_PAGES pages and RENDER_BUDGET_S seconds per document. A page that cannot be judged
(render error, budget spent, page counts disagree, too many groups or unmatched chars, scale below
MIN_RENDER_SCALE) returns None: the caller marks it 'unverified', keeps only the colour-rule text,
and records citing it carry an error flag. A glyph repeating an earlier one's text within DUPLICATE_PT
(fake-bold double strike, or a copy laid over a digit to double it) is one glyph to a reader:
``duplicate_ids`` drops the copy before judging (not counted as hidden). Any other over-strike of a
number character (``overlapping_chars``) cannot be read twice either; the parse lists it in the page's
glyph_check so records citing it carry an error flag.
"""
from __future__ import annotations

import ctypes
import time
import unicodedata
from typing import Any

import numpy as np

from app.evidence.parse_marks import CRITICAL as OVERLAP_CRITICAL
from app.evidence.parse_marks import MAX_INK_CHARS, ink_odd

RENDER_SCALE = 2.0                 # 144 dpi
MIN_RENDER_SCALE = 0.5
MAX_RENDER_PIXELS = 6_000_000      # per render (~18 MB RGB)
RENDER_MAX_PAGES = 300
RENDER_BUDGET_S = 60.0
INK_DELTA = 40                     # ~16 % of full scale: 0.84 grey on white still counts as ink
INK_FRACTION = 0.05
MIN_INSIDE_FRACTION = 0.5          # at least half the box inside the CropBox render
PAD_PX = 1
MAX_GROUPS = 8
MATCH_PT = 0.5
EXACT_PT = 0.05                    # origins of one glyph agree to ~1e-4 pt between the two libraries
MAX_UNMATCHED = 0.2
DUPLICATE_PT = 1.0
DRIFT_PT = 3.0                     # a run's origins may drift this far apart (Standard-14 widths)
MISMATCH, INK_MISMATCH = "glyph_text_mismatch", "glyph_ink_mismatch"
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−﹘﹣－"})
_UNOWNED = ("Mn", "Me", "Sk", "Lm")   # combining marks / modifiers sit on another glyph by design


def _open_document(path: str) -> Any:
    """The pdfium document with its form environment set up BEFORE any page loads, so form-field
    (Widget) appearances are drawn like a viewer draws them: a field laid over page text covers it.
    A document whose forms cannot be initialised raises, and every page is then unverified."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    try:
        doc.init_forms()
    except Exception:
        doc.close()
        raise
    return doc


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def duplicate_ids(chars: list[dict[str, Any]]) -> set[int]:
    """ids of chars repeating an earlier char's text within DUPLICATE_PT (x0 and top)."""
    seen: dict[tuple[str, int, int], list[tuple[float, float]]] = {}
    out: set[int] = set()
    for c in chars:
        text, x, top = str(c.get("text", "")), float(c["x0"]), float(c["top"])
        gx, gy = int(x // DUPLICATE_PT), int(top // DUPLICATE_PT)
        near = (seen.get((text, gx + i, gy + j), []) for i in (-1, 0, 1) for j in (-1, 0, 1))
        if any(abs(x - px) <= DUPLICATE_PT and abs(top - pt) <= DUPLICATE_PT for cell in near for px, pt in cell):
            out.add(id(c))
        else:
            seen.setdefault((text, gx, gy), []).append((x, top))
    return out


# ------------------------------------------------------------- pdfium side ---

def _glyphs(page: Any) -> tuple[list[str], np.ndarray, np.ndarray, list[int], dict[int, Any]]:
    """(texts, origins n x 2, tight boxes n x 4 [l, b, r, t], object keys, key -> object handle)."""
    import pypdfium2.raw as raw
    tp = page.get_textpage()
    texts, origins, boxes, keys, handles = [], [], [], [], {}
    x, y = ctypes.c_double(), ctypes.c_double()
    lft, rgt, bot, top = (ctypes.c_double() for _ in range(4))
    try:
        for i in range(raw.FPDFText_CountChars(tp)):
            text = chr(raw.FPDFText_GetUnicode(tp, i))
            if raw.FPDFText_IsGenerated(tp, i) == 1 or not text.strip():
                continue
            obj = raw.FPDFText_GetTextObject(tp, i)
            key = ctypes.cast(obj, ctypes.c_void_p).value
            if not key:
                continue
            raw.FPDFText_GetCharOrigin(tp, i, x, y)
            raw.FPDFText_GetCharBox(tp, i, lft, rgt, bot, top)
            texts.append(text)
            origins.append((x.value, y.value))
            boxes.append((lft.value, bot.value, rgt.value, top.value))
            keys.append(key)
            handles.setdefault(key, obj)
    finally:
        tp.close()
    return texts, np.array(origins, float).reshape(-1, 2), np.array(boxes, float).reshape(-1, 4), keys, handles


def _groups(keys: list[int], boxes: np.ndarray) -> dict[int, int] | None:
    """Text object -> group so that no two objects of a group overlap; None above MAX_GROUPS."""
    objs = sorted(set(keys))
    index = {k: i for i, k in enumerate(objs)}
    lo, hi = np.full((len(objs), 2), np.inf), np.full((len(objs), 2), -np.inf)
    for k, b in zip(keys, boxes, strict=True):
        i = index[k]
        lo[i] = np.minimum(lo[i], b[:2])
        hi[i] = np.maximum(hi[i], b[2:])
    group = np.full(len(objs), -1)
    for i in range(len(objs)):
        hit = ((lo[:, 0] < hi[i, 0] - 0.1) & (hi[:, 0] > lo[i, 0] + 0.1)
               & (lo[:, 1] < hi[i, 1] - 0.1) & (hi[:, 1] > lo[i, 1] + 0.1) & (group >= 0))
        used = set(group[hit].tolist())
        group[i] = next(g for g in range(len(objs) + 1) if g not in used)
        if group[i] >= MAX_GROUPS:
            return None
    return {k: int(group[index[k]]) for k in objs}


def _render(page: Any, scale: float, handles: dict[int, Any], modes: dict[int, int], show: set[int]) -> Any:
    """The page bitmap with only the text objects in *show* painted (others invisible, clips kept)."""
    import pypdfium2.raw as raw
    for key, obj in handles.items():
        mode = modes[key]
        if key not in show:
            mode = raw.FPDF_TEXTRENDERMODE_CLIP if mode >= raw.FPDF_TEXTRENDERMODE_FILL_CLIP \
                else raw.FPDF_TEXTRENDERMODE_INVISIBLE
        raw.FPDFTextObj_SetTextRenderMode(obj, mode)
    return page.render(scale=scale, may_draw_forms=True, draw_annots=True)


def _ink(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.maximum(a, b) - np.minimum(a, b)).max(axis=2) >= INK_DELTA


def _device_affine(posconv: Any) -> np.ndarray:
    """2x3 float map user space -> bitmap pixels. pdfium's converter returns whole pixels, so the map is
    fitted on probe points 10 000 pt apart (rounding error < 1e-4 px per pt)."""
    src = np.array([(0.0, 0.0, 1.0), (1e4, 0.0, 1.0), (0.0, 1e4, 1.0)])
    dst = np.array([posconv.to_bitmap(x, y) for x, y, _ in src], float)
    return np.linalg.solve(src, dst).T


def _pixel_boxes(boxes: np.ndarray, posconv: Any, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(clipped integer pixel boxes [x0, y0, x1, y1], unclipped areas); a box is at least one pixel."""
    affine = _device_affine(posconv)
    pts = np.stack([boxes[:, [0, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]], boxes[:, [2, 1]]], axis=1)
    dev = pts @ affine[:, :2].T + affine[:, 2]
    lo = np.floor(dev.min(axis=1))
    hi = np.maximum(np.ceil(dev.max(axis=1)), lo + 1)
    area = (hi - lo).prod(axis=1)
    h, w = shape[:2]
    px = np.stack([np.clip(lo[:, 0], 0, w), np.clip(lo[:, 1], 0, h), np.clip(hi[:, 0], 0, w),
                   np.clip(hi[:, 1], 0, h)], axis=1).astype(np.int64)
    return px, area


def _box_sums(mask: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    table = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int32)   # <= MAX_RENDER_PIXELS
    np.cumsum(mask, axis=0, dtype=np.int32, out=table[1:, 1:])
    np.cumsum(table[1:, 1:], axis=1, out=table[1:, 1:])
    x0, y0, x1, y1 = boxes.T
    return table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]


def _claims(boxes: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    acc = np.zeros((h + 1, w + 1), dtype=np.int32)
    for dy, dx, sign in ((1, 0, 1), (1, 2, -1), (3, 0, -1), (3, 2, 1)):
        np.add.at(acc, (boxes[:, dy], boxes[:, dx]), sign)
    np.cumsum(acc, axis=0, out=acc)
    np.cumsum(acc, axis=1, out=acc)
    return acc[:h, :w]


def _pad(px: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    lo = np.maximum(px[:, :2] - PAD_PX, 0)
    hi = np.minimum(px[:, 2:] + PAD_PX, [w, h])
    return np.concatenate([lo, hi], axis=1)


def _judge(texts: list[str], boxes: np.ndarray, groups: list[int], renders: dict[int, np.ndarray],
           blank: np.ndarray, posconv: Any) -> tuple[np.ndarray, np.ndarray]:
    """(visible, odd) per pdfium glyph: the visible verdict (module docstring, step 3) and
    parse_marks.ink_odd (a number glyph whose ink is not the plain glyph). A glyph's region is its tight box
    (overlaps inside one text object are normal kerning: a period under an italic 'T') plus a PAD_PX
    ring of pixels no other glyph box claims (pdfium's box can sit a pixel off a hairline's
    anti-aliased ink). Glyphs sharing one exact box (a ligature that pdfium reports as 'f' + 'i') are
    one glyph: the first claims and decides."""
    px, area = _pixel_boxes(boxes, posconv, blank.shape)
    padded = _pad(px, blank.shape)
    first: dict[tuple, int] = {}
    canon = np.array([first.setdefault((groups[i], *np.round(boxes[i], 2).tolist()), i) for i in range(len(texts))])
    marks = np.array([unicodedata.category(t[:1]) in _UNOWNED for t in texts])
    claims = (canon == np.arange(len(texts))) & ~marks
    inside = (px[:, 2] - px[:, 0]) * (px[:, 3] - px[:, 1])          # box pixels on the rendered page
    visible, odd = np.zeros(len(texts), bool), np.zeros(len(texts), bool)
    for g, img in renders.items():
        sel = np.array(groups) == g
        ink = _ink(img, blank)
        free = _claims(px[sel & claims], blank.shape) == 0
        b, pb = px[sel], padded[sel]
        own_area = inside[sel] + _box_sums(free, pb) - _box_sums(free, b)
        own_ink = _box_sums(ink, b) + _box_sums(ink & free, pb) - _box_sums(ink & free, b)
        visible[sel] = ((inside[sel] >= MIN_INSIDE_FRACTION * area[sel])
                        & (own_ink >= np.maximum(1, INK_FRACTION * own_area)))
        odd |= ink_odd(texts, boxes, px, area, ink, sel & claims & visible, blank, INK_DELTA)
    return visible[canon], odd[canon]


# ---------------------------------------------------------- pdfplumber side ---

def _pdfminer_ctm(page: Any) -> np.ndarray:
    """pdfminer's page CTM (pdf user space -> its device space) as a 3x3 matrix."""
    x0, y0, x1, y1 = page.page_obj.mediabox
    a, b, c, d, e, f = {90: (0, -1, 1, 0, -y0, x1), 180: (-1, 0, 0, -1, x1, y1),
                        270: (0, 1, -1, 0, y1, -x0)}.get(page.page_obj.rotate, (1, 0, 0, 1, -x0, -y0))
    return np.array([[a, c, e], [b, d, f], [0, 0, 1]], dtype=float)


def _fold(text: str) -> str:
    return _norm(text).translate(_DASHES)


def _decoded_apart(text: str, exact: list[int], texts: list[str]) -> set[str]:
    """Number characters of a char and the glyphs at its exact origin that decode to OTHER text
    (pdfium follows /ActualText, pdfminer reads the drawn code): the text layer is in doubt. A side
    that could not decode the glyph at all ('(cid:1)', a control or private-use code) is no evidence."""
    if text.startswith("(cid:"):
        return set()
    decoded = [t for t in (_fold(texts[k]) for k in exact) if t and t != "\ufffd"
               and not unicodedata.category(t[0]).startswith("C")]
    out = {ch for t in decoded if t != _fold(text) for ch in t + _fold(text)}
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


def _transfer(page: Any, chars: list[dict[str, Any]], texts: list[str], origins: np.ndarray,
              visible: np.ndarray, odd: np.ndarray) -> tuple[set[int], dict[str, set[str]]] | None:
    """(ids of pdfplumber chars whose matching pdfium glyph is hidden (or that have none),
    {glyph_check code: number characters in doubt}); None when too many chars have no glyph."""
    grid: dict[tuple[int, int], list[int]] = {}
    for k, (x, y) in enumerate(origins):
        grid.setdefault((int(x // 2), int(y // 2)), []).append(k)
    inv = np.linalg.inv(_pdfminer_ctm(page))
    norm = [_norm(t) for t in texts]
    hidden, unmatched, judged, prev = set(), 0, 0, None
    flags: dict[str, set[str]] = {MISMATCH: set(), INK_MISMATCH: set()}
    for c in chars:
        text = _norm(str(c.get("text", "")))
        if not text.strip():
            continue
        judged += 1
        ux, uy, _ = inv @ np.array([float(c["matrix"][4]), float(c["matrix"][5]), 1.0])
        near = [k for i in (-1, 0, 1) for j in (-1, 0, 1) for k in grid.get((int(ux // 2) + i, int(uy // 2) + j), [])]
        near = [(float(np.hypot(origins[k, 0] - ux, origins[k, 1] - uy)), k) for k in near]
        same = sorted((d, k) for d, k in near if d <= MATCH_PT and norm[k] and (norm[k] in text or text in norm[k]))
        same = same or _drifted(prev, ux, uy, text, origins, norm)
        if same:
            used = [k for d, k in same if d <= same[0][0] + EXACT_PT]
            ok = any(visible[k] for k in used)
        else:                      # same glyph, decoded differently ('(cid:1)', U+2010 vs '-'): all must show
            used = [k for d, k in near if d <= EXACT_PT]
            unmatched += not used
            ok = bool(used) and all(visible[k] for k in used)
            flags[MISMATCH] |= _decoded_apart(text, used, texts)
        prev = used[0] if used else None
        if any(odd[k] for k in used):
            flags[INK_MISMATCH] |= set(_fold(text)) & OVERLAP_CRITICAL
        if not ok:
            hidden.add(id(c))
    if judged and unmatched > MAX_UNMATCHED * judged:
        return None
    return hidden, {code: chars_ for code, chars_ in flags.items() if chars_}


class InkChecker:
    """One pdfium document per parse; ``hidden(page, chars)`` returns hidden char ids or None
    (unverified), and leaves the page's {glyph_check code: number characters in doubt} in ``flags``."""

    def __init__(self, path: str, n_pages: int) -> None:
        self.deadline = time.monotonic() + RENDER_BUDGET_S
        self.flags: dict[str, set[str]] = {}
        try:
            self.doc = _open_document(path)
            self.ok = len(self.doc) == n_pages            # pages must pair up one to one
        except Exception:
            self.doc, self.ok = None, False

    def hidden(self, page: Any, chars: list[dict[str, Any]]) -> set[int] | None:
        n, self.flags = page.page_number, {}
        if (not self.ok or n > RENDER_MAX_PAGES or time.monotonic() > self.deadline
                or len(chars) > MAX_INK_CHARS):
            return None
        pdf_page = None
        try:
            pdf_page = self.doc[n - 1]
            out = self._hidden(pdf_page, page, chars)
        except Exception:                                  # any render / mapping failure: not judged
            return None
        finally:
            if pdf_page is not None:
                pdf_page.close()                           # pages and bitmaps must not pile up
        if out is None:
            return None
        self.flags = out[1]
        return out[0]

    def _hidden(self, pdf_page: Any, page: Any, chars: list[dict[str, Any]]
                ) -> tuple[set[int], dict[str, set[str]]] | None:
        import pypdfium2.raw as raw
        w, h = pdf_page.get_size()
        scale = min(RENDER_SCALE, (MAX_RENDER_PIXELS / max(w * h, 1.0)) ** 0.5)
        texts, origins, boxes, keys, handles = _glyphs(pdf_page)
        if len(texts) > MAX_INK_CHARS:
            return None
        groups = _groups(keys, boxes) if keys else {}
        if scale < MIN_RENDER_SCALE or groups is None:
            return None
        modes = {k: raw.FPDFTextObj_GetTextRenderMode(o) for k, o in handles.items()}
        blank = _render(pdf_page, scale, handles, modes, set())
        renders = {g: _render(pdf_page, scale, handles, modes, {k for k in handles if groups[k] == g}).to_numpy()
                   for g in sorted(set(groups.values()))}
        visible, odd = (_judge(texts, boxes, [groups[k] for k in keys], renders, blank.to_numpy(),
                               blank.get_posconv(pdf_page)) if keys else (np.zeros(0, bool), np.zeros(0, bool)))
        return _transfer(page, chars, texts, origins, visible, odd)

    def close(self) -> None:
        if self.doc is not None:
            self.doc.close()
