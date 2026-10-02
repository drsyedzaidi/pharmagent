"""Glyph identity: is the character the text layer reports the glyph a reader sees?

The ink check proves that a glyph puts ink on the page, not that the ink is the character the text
layer names. A font can say otherwise in two ways: its /ToUnicode CMap maps a code to a different
character than the glyph drawn for that code (Helvetica's 'three' reported as '4'), or its glyph
program is arbitrary (a Type3 procedure named 'three' that paints a 4). Numbers are what the
verifier trusts, so every font a page draws with is checked for the characters numbers are made of
(CRITICAL: digits, sign, decimal / grouping marks, exponent marks):

- Type3 font: its glyphs are drawing procedures, so the identity of any CRITICAL character it can
  produce is unknown -> ``glyph_text_mismatch`` (error).
- Simple font (Type1 / TrueType) with /ToUnicode: each code's ToUnicode text is compared with the
  character its encoding names (base encoding + /Differences, Adobe Glyph List). A disagreement on a
  CRITICAL character -> ``glyph_text_mismatch``; no usable glyph name (symbolic embedded font
  without /Encoding, unknown glyph name) -> ``glyph_unverified`` (warn).
- Type0 font with /ToUnicode: for an embedded TrueType (CIDFontType2) the glyph a CID selects
  (CIDToGIDMap) is looked up in the font's own cmap / glyph names (fontTools); a disagreement ->
  ``glyph_text_mismatch``, no identity (CFF CIDs, no cmap entry, unreadable program) ->
  ``glyph_unverified``.

The result per page is {code: CRITICAL characters affected}; records whose literal contains one of
them carry the flag (record_checks.glyph_flags). Limit: an embedded outline that draws a different
shape under a correct glyph name and cmap entry is not detected (that needs OCR).
"""
from __future__ import annotations

import io
import unicodedata
from typing import Any

from pdfminer.encodingdb import name2unicode
from pdfminer.pdffont import PDFCIDFont, PDFType3Font
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdftypes import PDFObjRef, PDFStream, resolve1

MISMATCH, UNVERIFIED = "glyph_text_mismatch", "glyph_unverified"
CRITICAL = frozenset("0123456789.,+-eE×^·'")
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−﹘﹣－"})
MAX_FORM_DEPTH = 4
MAX_FONTS = 400


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(_DASHES)


def _critical(text: str | None) -> set[str]:
    return {c for c in _fold(text or "") if c in CRITICAL}


def _name(obj: Any) -> str:
    obj = resolve1(obj)
    return str(getattr(obj, "name", obj or ""))


def _font_specs(resources: Any, depth: int = 0, seen: set[int] | None = None) -> list[tuple[Any, dict]]:
    """(cache key, font dict) for every font of *resources* and of the Form XObjects it draws."""
    seen = set() if seen is None else seen
    resources = resolve1(resources)
    if not isinstance(resources, dict) or depth > MAX_FORM_DEPTH:
        return []
    out: list[tuple[Any, dict]] = []
    for ref in (resolve1(resources.get("Font")) or {}).values():
        spec = resolve1(ref)
        if isinstance(spec, dict):
            out.append((ref.objid if isinstance(ref, PDFObjRef) else id(spec), spec))
    for ref in (resolve1(resources.get("XObject")) or {}).values():
        key = ref.objid if isinstance(ref, PDFObjRef) else id(ref)
        xobj = resolve1(ref)
        if key in seen or not isinstance(xobj, PDFStream) or _name(xobj.attrs.get("Subtype")) != "Form":
            continue
        seen.add(key)
        out.extend(_font_specs(xobj.attrs.get("Resources"), depth + 1, seen))
    return out[:MAX_FONTS]


def _descriptor(spec: dict) -> dict:
    fd = resolve1(spec.get("FontDescriptor"))
    return fd if isinstance(fd, dict) else {}


def _tounicode(font: Any) -> dict[int, str]:
    umap = getattr(font, "unicode_map", None)
    return dict(getattr(umap, "cid2unichr", {}) or {}) if umap is not None else {}


def _cff_encoding(fd: dict) -> dict[int, str] | None:
    """code -> character from an embedded Type1C (CFF) program's own encoding, else None."""
    stream = resolve1(fd.get("FontFile3"))
    if not isinstance(stream, PDFStream) or _name(stream.attrs.get("Subtype")) != "Type1C":
        return None
    try:
        from fontTools.cffLib import CFFFontSet
        cff = CFFFontSet()
        cff.decompile(io.BytesIO(stream.get_data()), None)
        enc = cff[cff.fontNames[0]].Encoding
    except Exception:                                        # unreadable program: identity unknown
        return None
    if isinstance(enc, str):
        return None
    out = {}
    for code, glyph in enumerate(enc or []):
        try:
            out[code] = name2unicode(glyph)
        except Exception:
            continue
    return out


def _encoding(font: Any, spec: dict) -> dict[int, str] | None:
    """code -> character the font's glyph names give, or None when the font names no glyphs: an
    embedded symbolic / TrueType font without /Encoding draws by its own built-in table (a Type1
    program's is parsed by pdfminer, a Type1C one by fontTools; TrueType is not resolved)."""
    if "Encoding" in spec:
        return font.cid2unicode
    fd = _descriptor(spec)
    if "FontFile" in fd:
        return font.cid2unicode                              # pdfminer read the Type1 built-in encoding
    if "FontFile3" in fd:
        return _cff_encoding(fd)
    if "FontFile2" in fd or _name(spec.get("Subtype")) != "Type1":
        return None
    return font.cid2unicode                                  # a standard-14 font: StandardEncoding


def _simple(font: Any, spec: dict) -> tuple[set[str], set[str]]:
    """(mismatch, unverified) CRITICAL characters of a Type1 / TrueType font with a ToUnicode map."""
    encoding = _encoding(font, spec)
    bad, unknown = set(), set()
    for code, text in _tounicode(font).items():
        crit = _critical(text)
        enc = None if encoding is None else encoding.get(code)
        if not crit and not _critical(enc):
            continue
        if enc is None:
            unknown |= crit
        elif _fold(text) != _fold(enc):
            bad |= crit
    return bad, unknown


def _cid_to_gid(desc: dict) -> Any:
    raw = resolve1(desc.get("CIDToGIDMap"))
    if isinstance(raw, PDFStream):
        data = raw.get_data()
        return lambda cid: int.from_bytes(data[2 * cid:2 * cid + 2], "big") if 2 * cid + 2 <= len(data) else None
    return lambda cid: cid                                   # /Identity (or absent)


def _truetype_names(desc: dict) -> tuple[list[str], dict[str, set[str]]] | None:
    """(glyph order, glyph name -> characters its cmap gives it) of an embedded TrueType, else None."""
    stream = resolve1(_descriptor(desc).get("FontFile2"))
    if not isinstance(stream, PDFStream):
        return None
    try:
        from fontTools.ttLib import TTFont
        font = TTFont(io.BytesIO(stream.get_data()), lazy=True)
        order = list(font.getGlyphOrder())
        names: dict[str, set[str]] = {}
        for uni, glyph in (font.getBestCmap() or {}).items():
            names.setdefault(glyph, set()).add(chr(uni))
        return order, names
    except Exception:                                        # unreadable program: identity unknown
        return None


def _glyph_chars(order: list[str], names: dict[str, set[str]], gid: int | None) -> set[str]:
    if gid is None or not 0 <= gid < len(order):
        return set()
    name = order[gid]
    if name in names:
        return names[name]
    try:
        return {name2unicode(name)}
    except Exception:
        return set()


def _cid(font: Any, spec: dict) -> tuple[set[str], set[str]]:
    """(mismatch, unverified) CRITICAL characters of a Type0 font with a ToUnicode map."""
    descendants = resolve1(spec.get("DescendantFonts"))
    desc = resolve1(descendants[0]) if isinstance(descendants, list) and descendants else None
    desc = desc if isinstance(desc, dict) else {}
    program = _truetype_names(desc) if _name(desc.get("Subtype")) == "CIDFontType2" else None
    gid_of = _cid_to_gid(desc)
    bad, unknown = set(), set()
    for cid, text in _tounicode(font).items():
        crit = _critical(text)
        if not crit:
            continue
        glyph = _glyph_chars(*program, gid_of(cid)) if program else set()
        if not glyph:
            unknown |= crit
        elif _fold(text) not in {_fold(g) for g in glyph}:
            bad |= crit
    return bad, unknown


def _type3(font: Any, spec: dict) -> set[str]:
    """Every CRITICAL character a Type3 font can produce (its codes FirstChar..LastChar, or ToUnicode)."""
    first, last = int(resolve1(spec.get("FirstChar")) or 0), int(resolve1(spec.get("LastChar")) or 255)
    codes = range(max(0, first), min(255, last) + 1)
    return set().union(set(), *(_critical(font.cid2unicode.get(c)) for c in codes),
                       *map(_critical, _tounicode(font).values()))


class GlyphChecker:
    """One per parse: fonts are judged once per document (cached by object id)."""

    def __init__(self) -> None:
        self.rsrc = PDFResourceManager(caching=False)
        self.cache: dict[Any, tuple[str, set[str], set[str]]] = {}

    def _judge(self, key: Any, spec: dict) -> tuple[str, set[str], set[str]]:
        if key not in self.cache:
            try:
                font = self.rsrc.get_font(None, spec)
                if isinstance(font, PDFType3Font):
                    verdict = (font.fontname, _type3(font, spec), set())
                elif isinstance(font, PDFCIDFont):
                    verdict = (font.fontname, *_cid(font, spec))
                else:
                    verdict = (font.fontname, *_simple(font, spec))
            except Exception:                                # unreadable font dict: identity unknown
                verdict = ("", set(), set("0123456789"))
            self.cache[key] = verdict
        return self.cache[key]

    def page(self, page: Any) -> dict[str, str]:
        """{flag code: CRITICAL characters} for the fonts the page's chars use; {} when all agree."""
        used = {str(c.get("fontname")) for c in page.chars}
        try:
            specs = _font_specs(page.page_obj.resources)
        except Exception:
            return {UNVERIFIED: "0123456789"}
        bad, unknown = set(), set()
        for key, spec in specs:
            name, b, u = self._judge(key, spec)
            if name and name not in used:
                continue
            bad |= b
            unknown |= u
        out = {MISMATCH: "".join(sorted(bad)), UNVERIFIED: "".join(sorted(unknown - bad))}
        return {k: v for k, v in out.items() if v}
