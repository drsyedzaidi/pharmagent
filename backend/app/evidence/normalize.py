"""Index-mapped text normaliser and ASCII-only number parsing for the literal-match verifier.

Every transform returns (text, idx_map) with idx_map[i] = offset of text[i] in the
ORIGINAL string, so a match found in normalised text maps back to raw page offsets.
Pipeline: per-char NFKC -> dash/space folding -> casefold -> de-hyphenate line
wraps -> collapse whitespace. Numbers are matched with ASCII digits only; any other
Unicode digit is reported by has_non_ascii_digits() and must be rejected upstream.
"""
from __future__ import annotations

import re
import unicodedata

IdxMap = list[int]

_DASHES = "‐‑‒–—―−﹘﹣－"
_SPACES = "              　"
_FOLD = str.maketrans({**{c: "-" for c in _DASHES}, **{c: " " for c in _SPACES}, "µ": "μ"})
_DROP = {"­", "​", "‌", "‍", "﻿"}        # soft hyphen, zero-width marks
_ASCII_DIGITS = set("0123456789")

# ASCII-only number token: optional sign (only when not glued to a word), digits with optional
# 3-digit groups separated by space/comma/point/apostrophe, optional decimal part. Lookbehinds
# stop fragments ('24' inside '245.6') and alphanumeric prefixes ('AUC0', 't1/2') from matching.
NUM_RE = re.compile(
    r"(?:(?<![\w.)\]])[-+])?(?<![\w.])\d+(?:[ ,.']\d{3}(?!\d))*(?:[.,]\d+)?(?!\d)", re.ASCII)
_COMMA_DECIMAL = re.compile(r"\d,\d{1,2}(?!\d)")
_POINT_DECIMAL = re.compile(r"\d\.\d+")
_POINT_THOUSANDS = re.compile(r"\b\d{1,3}(?:\.\d{3})+,\d+")
_DIGITS_ONLY = re.compile(r"[^0-9]")
_WORD_RE = re.compile(r"\w", re.ASCII)
_ASCII_GAPS = str.maketrans({c: "\x00" for c in " \t\n\r\f\v"})   # 1:1, matches no NUM_RE separator


def fold_chars(text: str) -> str:
    """1:1 character folding: Unicode dashes -> '-', exotic spaces -> ' ', micro sign -> Greek mu."""
    return text.translate(_FOLD)


def has_non_ascii_digits(text: str) -> bool:
    """True when any digit-like character (category Nd or No) is not ASCII 0-9."""
    return any(ch not in _ASCII_DIGITS and unicodedata.category(ch) in ("Nd", "No") for ch in text)


def infer_decimal_style(text: str) -> str:
    """',' when decimal-comma evidence outweighs decimal-point evidence, else '.'."""
    comma = len(_POINT_THOUSANDS.findall(text))
    text = _POINT_THOUSANDS.sub(" ", text)
    comma += len(_COMMA_DECIMAL.findall(text))
    point = len(_POINT_DECIMAL.findall(text))
    return "," if comma > point else "."


# ----------------------------------------------------------- numbers ---

def _to_float(token: str, decimal_style: str) -> float | None:
    """Token -> value. A separator followed by a non-3-digit group is always decimal ('0,5'); a
    separator repeated is always thousands ('1.250.000'); a single 3-digit tail follows the style."""
    sign = -1.0 if token.startswith("-") else 1.0
    body = token.lstrip("+-").replace(" ", "").replace("'", "")
    seps = [c for c in body if c in ".,"]
    if seps:
        last = seps[-1]
        tail = body[body.rfind(last) + 1:]
        style_dec = "," if decimal_style == "," else "."
        is_decimal = seps.count(last) == 1 and (len(tail) != 3 or last == style_dec)
        body = body.replace("." if last == "," else ",", "")          # the other separator: thousands
        body = body.replace(last, ".") if is_decimal else body.replace(last, "")
    try:
        return sign * float(body)
    except ValueError:
        return None


def number_spans(text: str, decimal_style: str = ".", *, strict: bool = False) -> list[tuple[int, int, float]]:
    """(start, end, value) for every ASCII number token in *text* (offsets into *text*).

    *strict* is for raw page / cell text: an ASCII space, tab or line break is a word or column
    gap and never a thousands separator ('12 145' is two numbers); only a thin / no-break space
    (U+2009, U+202F, U+00A0, ...) groups digits. Lenient mode (snippets, proposed values, which a
    model writes with plain spaces) accepts a single ASCII space too."""
    folded = fold_chars(text.translate(_ASCII_GAPS) if strict else text)
    out: list[tuple[int, int, float]] = []
    last_end = -1
    for m in NUM_RE.finditer(folded):
        start = m.start()
        # '24' in 'AUC0-24' or '2' in 't1/2' belongs to a label: a range/ratio is only accepted
        # when the number before the separator was itself accepted ('0-24', '1.95-2.74').
        if start >= 2 and folded[start - 1] in "-/" and _WORD_RE.match(folded[start - 2]) and last_end != start - 1:
            continue
        value = _to_float(m.group(0), decimal_style)
        if value is not None:
            out.append((start, m.end(), value))
            last_end = m.end()
    return out


def numbers_in(text: str, decimal_style: str = ".") -> list[tuple[str, float]]:
    """(literal as written, value) for every number token in *text*."""
    return [(text[s:e], v) for s, e, v in number_spans(text, decimal_style)]


def parse_number(token: str, decimal_style: str = ".") -> float | None:
    """Value of the first number token in *token* ('3,412' -> 3412, '18,2' with ',' -> 18.2), else None."""
    spans = number_spans(token, decimal_style)
    return spans[0][2] if spans else None


def digits_only(text: str) -> str:
    return _DIGITS_ONLY.sub("", text)


# ------------------------------------------------- mapped transforms ---

def _identity(text: str, idx_map: IdxMap | None) -> IdxMap:
    return list(range(len(text))) if idx_map is None else idx_map


def _expand_with_map(text: str, idx_map: IdxMap | None, fn) -> tuple[str, IdxMap]:
    """Apply a per-character expansion fn(ch) -> str; every output char maps to the input char."""
    src_map = _identity(text, idx_map)
    out: list[str] = []
    out_map: IdxMap = []
    for ch, raw in zip(text, src_map):
        rep = fn(ch)
        out.append(rep)
        out_map.extend([raw] * len(rep))
    return "".join(out), out_map


def nfkc_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    return _expand_with_map(text, idx_map, lambda ch: "" if ch in _DROP else unicodedata.normalize("NFKC", ch))


def casefold_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    return _expand_with_map(text, idx_map, str.casefold)


def dehyphenate_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """Join words wrapped at a hyphen: letter '-' newline [ws] letter -> the hyphen and break are dropped."""
    src_map = _identity(text, idx_map)
    out: list[str] = []
    out_map: IdxMap = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "-" and i > 0 and text[i - 1].isalpha() and i + 1 < n and text[i + 1] == "\n":
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].isalpha():
                i = j
                continue
        out.append(ch)
        out_map.append(src_map[i])
        i += 1
    return "".join(out), out_map


def strip_ws_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """Collapse whitespace runs to one space and trim both ends; the space maps to the run's first char."""
    src_map = _identity(text, idx_map)
    out: list[str] = []
    out_map: IdxMap = []
    pending: int | None = None
    for ch, raw in zip(text, src_map):
        if ch.isspace():
            pending = raw if pending is None else pending
            continue
        if pending is not None and out:
            out.append(" ")
            out_map.append(pending)
        pending = None
        out.append(ch)
        out_map.append(raw)
    return "".join(out), out_map


def normalise(text: str) -> tuple[str, IdxMap]:
    """Full pipeline; returns (normalised text, map of each char to its raw offset)."""
    norm, m = nfkc_with_map(text)
    norm, m = _expand_with_map(norm, m, lambda ch: fold_chars(ch))
    norm, m = casefold_with_map(norm, m)
    norm, m = dehyphenate_with_map(norm, m)
    return strip_ws_with_map(norm, m)
