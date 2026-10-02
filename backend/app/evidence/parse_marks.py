"""Shape checks on number glyphs: is the ink a reader sees the plain glyph the text layer names?

The ink check (parse_ink.py) proves a glyph puts SOME ink on the page. These checks catch the ways
that ink can still read as another number (each returns the CRITICAL characters involved; the parse
lists them in the page's glyph_check so records whose literal holds one carry an error flag):

- ``orientation_outliers`` (glyph_text_mismatch): a number glyph drawn rotated or mirrored against
  its neighbours in content order (a '6' under a 180-degree text matrix reads as 9). A table or page
  rotated as a whole keeps one orientation and raises nothing.
- ``ink_odd`` (glyph_ink_mismatch), from the renders: a number glyph whose visible ink stops short of
  a side of its own outline box (clipped, or covered by a later fill: an '8' cut to its right half
  reads as 3), whose box lies partly off the rendered page, or whose box -- or the gap to the next
  glyph of its line -- holds non-text ink in the text-free render (a decimal point drawn as a path
  between '12' and '4', a rule struck through a digit). Uniform backgrounds (cell shading) and marks
  in the colour of the background are not ink.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

CRITICAL = frozenset("0123456789.,+-−")
NEIGHBOURS = 3                      # content-order neighbours on each side that set the orientation
LOSS_PX = 3                         # ink missing from a side of the outline box ...
LOSS_FRACTION = 0.15                # ... and at least this share of the box's width / height
INSET_FRACTION = 0.15               # vertical inset of the box band read in the text-free render
NEXT_GAP_EM = 0.6                   # a following glyph this close (x box height) is the same word
MAX_INK_CHARS = 25_000              # a denser page is not judged (unverified): bounded cost per page


def _orientation(c: dict[str, Any]) -> tuple[int, ...]:
    a, b, cc, d = (float(v) for v in (c.get("matrix") or (1, 0, 0, 1))[:4])
    scale = max(abs(a), abs(b), abs(cc), abs(d)) or 1.0
    return tuple(0 if abs(v / scale) < 0.5 else (1 if v > 0 else -1) for v in (a, b, cc, d))


def orientation_outliers(chars: list[dict[str, Any]]) -> set[str]:
    """Number characters drawn in another orientation than most of their NEIGHBOURS on each side."""
    keys = [_orientation(c) for c in chars]
    out: set[str] = set()
    for i, c in enumerate(chars):
        text = str(c.get("text", ""))
        if text not in CRITICAL:
            continue
        near = keys[max(0, i - NEIGHBOURS):i] + keys[i + 1:i + 1 + NEIGHBOURS]
        if near and keys[i] != max(set(near), key=near.count) and near.count(keys[i]) * 2 < len(near):
            out.add(text)
    return out


def _next_left(i: int, boxes: np.ndarray, px: np.ndarray) -> int | None:
    """Pixel x of the next glyph's left edge when it continues glyph i's line (same word), else None."""
    if i + 1 >= len(boxes):
        return None
    a, b = boxes[i], boxes[i + 1]
    height = max(a[3] - a[1], 1e-6)
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    if overlap < 0.5 * height or not 0 <= b[0] - a[2] < NEXT_GAP_EM * height or b[0] < a[0]:
        return None
    return int(px[i + 1, 0])


def _short_side(sub: np.ndarray) -> bool:
    """True when the ink in a box misses one side by >= LOSS_PX and >= LOSS_FRACTION of the box."""
    for axis in (0, 1):
        line = sub.any(axis=axis)
        n = len(line)
        if not line.any() or n == 0:
            continue
        first, last = int(np.argmax(line)), n - 1 - int(np.argmax(line[::-1]))
        if max(first, n - 1 - last) >= max(LOSS_PX, LOSS_FRACTION * n):
            return True
    return False


def _marked(blank: np.ndarray, x0: int, x1: int, y0: int, y1: int, ink_delta: int) -> bool:
    """True when the text-free render is not one colour over the region (non-text ink there). The
    top is inset (a rule above the cap height), the bottom only by a pixel: a point sits on the
    baseline, an underline below it."""
    y0, y1 = y0 + int(np.ceil(INSET_FRACTION * (y1 - y0))), y1 - 1
    region = blank[y0:y1, x0:x1]
    if region.size == 0:
        return False
    flat = region.reshape(-1, region.shape[-1]).astype(np.int16)
    return bool((flat.max(axis=0) - flat.min(axis=0)).max() >= ink_delta)


def ink_odd(texts: list[str], boxes: np.ndarray, px: np.ndarray, area: np.ndarray, ink: np.ndarray,
            sel: np.ndarray, blank: np.ndarray, ink_delta: int) -> np.ndarray:
    """Per glyph of one group (*sel*, its visible-ink mask *ink*): True for a number glyph whose ink
    is not the plain glyph (module docstring). *px* are clipped pixel boxes, *area* unclipped areas."""
    odd = np.zeros(len(texts), bool)
    for i in np.flatnonzero(sel):
        if texts[i] not in CRITICAL:
            continue
        x0, y0, x1, y1 = (int(v) for v in px[i])
        if (x1 - x0) * (y1 - y0) < area[i]:
            odd[i] = True                                    # partly off the rendered page
            continue
        nxt = _next_left(i, boxes, px)
        right = max(x1 - 1, x0 + 1) if nxt is None else max(nxt, x0 + 1)
        odd[i] = _short_side(ink[y0:y1, x0:x1]) or _marked(blank, x0 + 1, right, y0, y1, ink_delta)
    return odd


OVERLAP_FRACTION = 0.3                # share of the narrower glyph's advance two glyphs may share
OVERLAP_BUDGET_S = 5.0


def overlapping_chars(chars: list[dict[str, Any]]) -> set[str]:
    """Number characters drawn over another glyph (a second '1' 1.2 pt right of the first reads as one
    bold '1', the text says '11'): boxes sharing > OVERLAP_FRACTION of the narrower advance and over
    half the lower height. Exact copies are duplicate_ids' job; whatever else overlaps cannot both be
    read, so the characters involved are returned for the records citing them to be flagged. A page
    denser than MAX_INK_CHARS, or one that takes OVERLAP_BUDGET_S, returns every number character."""
    grid: dict[tuple[int, int], list[dict[str, Any]]] = {}
    out: set[str] = set()
    end = time.monotonic() + OVERLAP_BUDGET_S
    if len(chars) > MAX_INK_CHARS:
        return set(CRITICAL)
    for n, c in enumerate(chars):
        if n % 256 == 0 and time.monotonic() > end:
            return set(CRITICAL)
        text = str(c.get("text", ""))
        if not text.strip() or float(c["x1"]) <= float(c["x0"]):
            continue
        gx, gy = int(float(c["x0"]) // 10), int(float(c["top"]) // 10)
        for o in (o for i in range(-3, 4) for j in (-1, 0, 1) for o in grid.get((gx + i, gy + j), [])):
            if not ({text, str(o["text"])} & CRITICAL):
                continue
            xo = min(float(c["x1"]), float(o["x1"])) - max(float(c["x0"]), float(o["x0"]))
            yo = min(float(c["bottom"]), float(o["bottom"])) - max(float(c["top"]), float(o["top"]))
            narrow = min(float(c["x1"]) - float(c["x0"]), float(o["x1"]) - float(o["x0"]))
            low = min(float(c["bottom"]) - float(c["top"]), float(o["bottom"]) - float(o["top"]))
            if xo > OVERLAP_FRACTION * narrow and yo > 0.5 * low:
                out |= {ch for ch in text + str(o["text"]) if ch in CRITICAL}
        grid.setdefault((gx, gy), []).append(c)
    return out
