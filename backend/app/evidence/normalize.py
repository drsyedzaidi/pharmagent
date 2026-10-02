"""Index-mapped text normaliser and ASCII-only number parsing for the literal-match verifier.

Every transform returns (text, idx_map) with idx_map[i] = offset of text[i] in the
ORIGINAL string, so a match found in normalised text maps back to raw page offsets.
Pipeline: per-char NFKC -> dash/space folding -> casefold -> de-hyphenate line
wraps -> collapse whitespace. Numbers are matched with ASCII digits only; any other
Unicode digit is reported by has_non_ascii_digits() and must be rejected upstream.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import NamedTuple

from app.evidence.number_glue import glued

IdxMap = list[int]

_DASHES = "‐‑‒–—―−﹘﹣－˗➖⁃⸺⸻"        # look-alike minus signs fold to '-'
_PLUSES = "﹢＋"
_SPACES = "              　"
_APOSTROPHES = "’ʼ′"          # typographic apostrophe / modifier letter / prime: Swiss digit grouping too
_FOLD = str.maketrans({**{c: "-" for c in _DASHES}, **{c: " " for c in _SPACES}, **{c: "'" for c in _APOSTROPHES},
                       **{c: "+" for c in _PLUSES},
                       "µ": "μ"})
_DROP = {"­", "​", "‌", "‍", "﻿"}        # soft hyphen, zero-width marks
_ASCII_DIGITS = set("0123456789")

# ASCII-only number token: optional sign (only when not glued to a word), digits with optional
# 3-digit groups separated by space/comma/point/apostrophe, optional decimal part. A raised / mid
# dot between digits is a decimal point (Lancet '12·4') unless it multiplies a power of ten
# ('2.5·10^4'). Lookbehinds stop fragments ('24' inside '245.6', '4' after '12·') and alphanumeric
# prefixes ('AUC0', 't1/2') from matching.
MIDDOTS = "·‧˙٫"
_POWER_AHEAD = r"10(?:\^|[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻])"
NUM_RE = re.compile(
    r"(?:(?<![\w.)\]])[-+])?(?<![\w." + MIDDOTS + r"])\d+(?:[ ,.']\d{3}(?!\d))*"
    r"(?:[.,]\d+|[" + MIDDOTS + r"](?!" + _POWER_AHEAD + r")\d+)?(?!\d)", re.ASCII)
_COMMA_DECIMAL = re.compile(r"\d,\d{1,2}(?!\d)")
_POINT_DECIMAL = re.compile(r"\d\.\d+")
_POINT_THOUSANDS = re.compile(r"\b\d{1,3}(?:\.\d{3})+,\d+")
_DIGITS_ONLY = re.compile(r"[^0-9]")
_WORD_RE = re.compile(r"\w", re.ASCII)
_ASCII_GAPS = str.maketrans({c: "\x00" for c in " \t\n\r\f\v"})   # 1:1, matches no NUM_RE separator
# Scientific notation belongs to the number it follows: '1.2e-3', '1.2E+03', '1.2 x 10^3', '1.2 × 10⁻³'.
# A '10' after the multiplier with no readable exponent ('4.5 x 10' whose superscript went to small_text,
# or '10-3') makes the token UNRESOLVED (value None): it can never pass as its mantissa or as 10.
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_GAP = r"[ \x00]{0,2}"
_EXP = r"(?:\^[ \x00]?(?P<pow>[+-]?\d{1,4})(?!\d)|(?P<sup>[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,4})|(?P<odd>[+-]\d{1,4}(?!\d)))"
# Multiplier: x, *, or ANY non-ASCII character other than a comparison or plus-minus sign ('×', '·',
# look-alikes '⋅ ∙ • ⨯ ✕ ✖ х'): a symbol between a number and '10' is never read as a token boundary.
_MULTIPLIER = r"(?:[xX*]|[^\x00-\x7f±∓≤≥≦≧≈≃≅≠∼<>=])"
# 'N x 10' followed by digits and no caret / superscript ('1.2 × 103', '1.2 x 10 3') is a power of ten
# whose exponent was set at full size but raised: the text layer cannot say 10^3 from 103, so the token
# is unresolved (group 'raw'). A tail starting with 0 ('2 x 100 mg', '3 x 1000 mg') is a plain
# multiplication and stays plain numbers.
_RAW_EXP = r"10(?P<raw>[1-9]\d{0,2}|[ \x00][1-9]\d?)(?!\d|[.,]\d)"
_SCI_RE = re.compile(r"[eE](?P<e>[+-]?\d{1,4})(?!\d)|[ \x00]{1,2}[eE](?P<eg>[+-]?\d{1,4})(?![\w.,]\d|\w)|"
                     + _GAP + _MULTIPLIER + _GAP + "(?:" + _RAW_EXP + r"|10(?!\d|[.,]\d)" + _EXP + "?)")
_POW_RE = re.compile(_EXP)
MAX_EXPONENT = 300


class NumToken(NamedTuple):
    """One number token: offsets, value (None when an exponent is unreadable) and, for scientific
    notation, the mantissa's value (so a mantissa-only citation can be named a partial number)."""
    start: int
    end: int
    value: float | None
    mantissa: float | None = None


def fold_chars(text: str) -> str:
    """1:1 character folding: Unicode dashes -> '-', exotic spaces -> ' ', micro sign -> Greek mu."""
    return text.translate(_FOLD)


def has_non_ascii_digits(text: str) -> bool:
    """True when any digit-like character (category Nd or No) is not ASCII 0-9."""
    return any(ch not in _ASCII_DIGITS and unicodedata.category(ch) in ("Nd", "No") for ch in text)


_CITATIONS = re.compile(r"\[\s*\d+(?:\s*[,;\-–]\s*\d+)+\s*\]|\(\s*\d+(?:\s*[,;]\s*\d+){2,}\s*\)")


def decimal_evidence(text: str) -> tuple[int, int]:
    """(decimal-comma, decimal-point) evidence counts in *text*. Citation lists ('[1,2]', '[3-5]',
    '(1,4,7)') are not numbers and are removed first."""
    text = _CITATIONS.sub(" ", text)
    comma = len(_POINT_THOUSANDS.findall(text))
    text = _POINT_THOUSANDS.sub(" ", text)
    comma += len(_COMMA_DECIMAL.findall(text))
    point = len(_POINT_DECIMAL.findall(text))
    return comma, point


def infer_decimal_style(text: str) -> str:
    """',' when decimal-comma evidence outweighs decimal-point evidence, else '.'."""
    comma, point = decimal_evidence(text)
    return "," if comma > point else "."


# ----------------------------------------------------------- numbers ---

def _to_float(token: str, decimal_style: str) -> float | None:
    """Token -> value. A separator followed by a non-3-digit group is always decimal ('0,5'); a
    separator repeated is always thousands ('1.250.000'); a single 3-digit tail follows the style,
    except after a lone '0' ('0,412' groups no thousands: it is 0.412 whatever the style)."""
    sign = -1.0 if token.startswith("-") else 1.0
    body = token.lstrip("+-").replace(" ", "").replace("'", "")
    dot = next((c for c in body if c in MIDDOTS), None)
    if dot is not None:                                   # Lancet decimal: '12·4', '1,234·5'
        whole, frac = body.split(dot, 1)
        body = whole.replace(",", "").replace(".", "") + "." + frac
        try:
            return sign * float(body)
        except ValueError:
            return None
    seps = [c for c in body if c in ".,"]
    if seps:
        last = seps[-1]
        tail = body[body.rfind(last) + 1:]
        style_dec = "," if decimal_style == "," else "."
        whole = body[:body.rfind(last)]
        is_decimal = seps.count(last) == 1 and (len(tail) != 3 or last == style_dec or whole in ("0", ""))
        body = body.replace("." if last == "," else ",", "")          # the other separator: thousands
        body = body.replace(last, ".") if is_decimal else body.replace(last, "")
    try:
        return sign * float(body)
    except ValueError:
        return None


def _scaled(mantissa: float, exponent: str | None) -> float | None:
    """mantissa x 10^exponent (exact decimal rounding), None when unreadable or out of float range."""
    if exponent is None:
        return None
    exp = int(exponent.translate(_SUPERSCRIPT))
    value = float(f"{mantissa!r}e{exp}") if abs(exp) <= MAX_EXPONENT else math.inf
    return value if math.isfinite(value) else None


def _scientific(folded: str, end: int, mantissa: float, bare_ten: bool) -> tuple[int, float | None] | None:
    """(token end, value) when scientific notation follows the number ending at *end*, else None."""
    m = _SCI_RE.match(folded, end)
    if m is not None and (m.group("e") or m.group("eg")) is not None:     # '3.4E-03', SAS '3.4 E-03'
        return m.end(), _scaled(mantissa, m.group("e") or m.group("eg"))
    if m is not None and m.group("raw"):                                   # '1.2 × 103': exponent unreadable
        return m.end(), None
    if m is not None and (m.group("pow") or m.group("sup") or m.group("odd") or bare_ten):
        return m.end(), _scaled(mantissa, m.group("pow") or m.group("sup"))
    p = _POW_RE.match(folded, end) if mantissa == 10 and folded[end - 2:end] == "10" else None
    if p is not None:                                                      # a bare power: '10^6'
        return p.end(), _scaled(1.0, p.group("pow") or p.group("sup"))
    return None


def _glued_digit(text: str, start: int, end: int) -> bool:
    """True when a non-ASCII digit (fullwidth, Arabic-Indic, ...) touches the token on either side."""
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return (bool(before) and before not in _ASCII_DIGITS and unicodedata.category(before) in ("Nd", "No")) or \
        (bool(after) and after not in _ASCII_DIGITS and unicodedata.category(after) == "Nd")


def _dot_ten(token: str) -> bool:
    """'2·10': a Lancet 2.10, or 2 x 10 whose exponent went to small_text -- undecidable."""
    return any(token.endswith(d + "10") for d in MIDDOTS)


def number_tokens(text: str, decimal_style: str = ".", *, strict: bool = False,
                  bare_ten: bool = True) -> list[NumToken]:
    """Every ASCII number token in *text*, scientific notation included (offsets into *text*).

    *strict* is for raw page / cell text: an ASCII space, tab or line break is a word or column
    gap and never a thousands separator ('12 145' is two numbers); only a thin / no-break space
    (U+2009, U+202F, U+00A0, ...) groups digits. Lenient mode (snippets, proposed values, which a
    model writes with plain spaces) accepts a single ASCII space too. *bare_ten* reads 'N x 10'
    with no exponent as notation whose exponent was lost (unresolved, fail closed); pass False only
    when the page has no small glyphs that could have held it."""
    folded = fold_chars(text.translate(_ASCII_GAPS) if strict else text)
    out: list[NumToken] = []
    last_end = -1
    for m in NUM_RE.finditer(folded):
        start = m.start()
        if start < last_end:                                    # inside a scientific token
            continue
        # '24' in 'AUC0-24' or '2' in 't1/2' belongs to a label: a range/ratio is only accepted
        # when the number before the separator was itself accepted ('0-24', '1.95-2.74').
        if start >= 2 and folded[start - 1] in "-/" and _WORD_RE.match(folded[start - 2]) and last_end != start - 1:
            continue
        value = _to_float(m.group(0), decimal_style)
        if value is None:
            continue
        # '１2.5' (a digit we cannot read is part of it) or '2·10' that may have lost its exponent
        if _glued_digit(folded, start, m.end()) or (bare_ten and _dot_ten(m.group(0))):
            out.append(NumToken(start, m.end(), None))
            last_end = m.end()
            continue
        sci = _scientific(folded, m.end(), value, bare_ten)
        tok = NumToken(start, m.end(), value) if sci is None else NumToken(start, sci[0], sci[1], value)
        if glued(folded, start, tok.end, strict=strict):          # '12З', '12․4', '˗20', '12½': unreadable
            tok = NumToken(start, tok.end, None, tok.mantissa)
        out.append(tok)
        last_end = out[-1].end
    return out


def number_spans(text: str, decimal_style: str = ".", *, strict: bool = False,
                 bare_ten: bool = True) -> list[tuple[int, int, float]]:
    """(start, end, value) for every number token with a readable value (see number_tokens)."""
    return [(t.start, t.end, t.value) for t in number_tokens(text, decimal_style, strict=strict, bare_ten=bare_ten)
            if t.value is not None]


def numbers_in(text: str, decimal_style: str = ".") -> list[tuple[str, float]]:
    """(literal as written, value) for every number token in *text*."""
    return [(text[s:e], v) for s, e, v in number_spans(text, decimal_style)]


def parse_number(token: str, decimal_style: str = ".") -> float | None:
    """Value of the first number token in *token* ('3,412' -> 3412, '18,2' with ',' -> 18.2, '1.2e-3' ->
    0.0012), else None (also when that token's exponent is unreadable: '4.5 x 10')."""
    tokens = number_tokens(token, decimal_style)
    return tokens[0].value if tokens else None


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
