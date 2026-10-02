"""Round-3 attack regressions on the parse: what a reader sees next to a number is part of it.

- a /ToUnicode CMap that maps a drawn digit to a space, and a Type3 glyph decoded as a space whose
  procedure paints a digit (reader 231, text '23') leave no flag-free passing '23';
- a digit drawn as a path right after / before a number (reader 231 / 1455) is never verified as
  23 / 455 without an error flag;
- text stroked with a wide line over a number (a black bar) leaves no flag-free '231';
- a letter that looks like a digit glued after a number (Arial '12З', reader 123) never passes 12;
- an explicit space that advances 0 pt between digits ('(n = 12' + ' 0)', reader 120) is not a gap;
- a minus drawn from WinAnsi byte 0xAD (pdfminer decodes a space) never lets '20' pass for '-20'.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import rule_extract, value_number
from app.evidence.extract_tools import check_proposals
from app.evidence.parse import parse_pdf
from tests import evidence_trickpdf as T

TEMPLATE = {"fields": [{"name": "cmax", "type": "number", "unit": "ng/mL", "aliases": ["Cmax"]},
                       {"name": "auc", "type": "number", "aliases": ["AUC"]},
                       {"name": "n", "type": "integer", "aliases": ["N"]},
                       {"name": "change", "type": "number", "unit": "%"}]}
ARIAL = Path("/System/Library/Fonts/Supplemental/Arial.ttf")


def _doc(tmp_path, data: bytes, name: str = "x.pdf") -> dict:
    path = tmp_path / name
    path.write_bytes(data)
    return parse_pdf(path)


def _keyless(doc: dict) -> list[dict]:
    return check_proposals(rule_extract(doc, TEMPLATE, "src_x")["proposals"], doc, TEMPLATE)


def _clean_pass(rec: dict) -> bool:
    return rec["verify_status"] == "pass" and not any(f["severity"] == "error" for f in rec["flags"])


def _no_clean(records: list[dict], field: str, value: float) -> None:
    for r in records:
        if r["field"] == field and r.get("value_num") == value:
            assert not _clean_pass(r), (r["value"], r["verify_status"], r["flags"])


def _prop(field: str, value: str, snippet: str, *, unit: str | None = None) -> dict:
    loc = M.Locator(page=1, kind="text", snippet=snippet, decimal_style=".").to_json()
    return {"source_id": "src_x", "field": field, "group_label": "", "value": value,
            "value_num": value_number(value, "."), "unit": unit, "locator": loc, "extractor": "llm"}


# ------------------------------------------------------- glyph identity ---

def test_tounicode_space_for_a_drawn_digit_flags_the_page_numbers(tmp_path):
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding /ToUnicode 7 0 R >>"
    data = T.build(T.table(T.line("F2", 200, 660, "231"), ruled=True).encode(), fonts="/F1 3 0 R /F2 6 0 R",
                   extra={6: font, 7: T.stream(T._cmap({0x31: " "}))})
    doc = _doc(tmp_path, data)
    assert set("0123456789") <= set(doc["pages"][0]["glyph_check"].get(M.FLAG_GLYPH_MISMATCH, ""))
    _no_clean(_keyless(doc), "cmax", 23.0)


def test_type3_glyph_decoded_as_space_flags_the_page_numbers(tmp_path):
    proc = b"556 0 d0 BT /H 1000 Tf 0 0 Td (1) Tj ET"
    extra = {7: T.stream(proc),
             6: (b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 600 1000] /FontMatrix [0.001 0 0 0.001 0 0] "
                 b"/CharProcs << /space 7 0 R >> /Encoding << /Type /Encoding /Differences [32 /space] >> "
                 b"/FirstChar 32 /LastChar 32 /Widths [556] /Resources << /Font << /H 3 0 R >> >> >>")}
    cmax = "BT /F1 11 Tf 1 0 0 1 200 660 Tm (23) Tj /F3 11 Tf ( ) Tj ET\n"
    doc = _doc(tmp_path, T.build(T.table(cmax, ruled=True).encode(), fonts="/F1 3 0 R /F3 6 0 R", extra=extra))
    _no_clean(_keyless(doc), "cmax", 23.0)


# ------------------------------------------------------------ path digits ---

def _path_one(x: float, y: float, s: float = 11.0) -> str:
    k = s / 1000
    pts = [(259, 0), (359, 0), (359, 703), (300, 703), (300, 640), (140, 590), (140, 520), (259, 560)]
    p = " ".join(f"{x + a * k:.3f} {y + b * k:.3f} {'m' if i == 0 else 'l'}" for i, (a, b) in enumerate(pts))
    return f"0 g {p} h f\n"


def test_path_digit_after_or_before_a_number_is_flagged(tmp_path):
    cmax = "BT /F1 11 Tf 1 0 0 1 200 660 Tm (23) Tj ET\n" + _path_one(200 + 2 * 6.116, 660)
    c = T.table(cmax, ruled=True)
    assert "1 0 0 1 200 646 Tm (455)" in c
    c = c.replace("1 0 0 1 200 646 Tm (455)", "1 0 0 1 206.116 646 Tm (455)") + _path_one(200, 646)
    recs = _keyless(_doc(tmp_path, T.build(c.encode())))
    _no_clean(recs, "cmax", 23.0)
    _no_clean(recs, "cmax", 455.0)


def test_plain_ruled_table_numbers_stay_clean(tmp_path):
    recs = _keyless(_doc(tmp_path, T.build(T.table(T.line("F1", 200, 660, "231"), ruled=True).encode())))
    cmax = {r["value_num"]: r for r in recs if r["field"] == "cmax"}
    assert _clean_pass(cmax[231.0]) and _clean_pass(cmax[455.0]), cmax


# ------------------------------------------------------------ stroke cover ---

def test_wide_stroked_text_over_a_number_leaves_no_clean_pass(tmp_path):
    bar = "".join(f"q BT /F1 11 Tf 2 Tr 16 w 0 j 0 0 0 RG 0 g 1 0 0 1 {x} 665.5 Tm (-) Tj ET Q\n" for x in (203, 210))
    doc = _doc(tmp_path, T.build(T.table(T.line("F1", 200, 660, "231") + bar, ruled=True).encode()))
    _no_clean(_keyless(doc), "cmax", 231.0)


# ---------------------------------------------------- homoglyph letters ---

@pytest.mark.skipif(not ARIAL.is_file(), reason="needs the Arial TTF")
def test_cyrillic_lookalike_glued_after_a_number_never_passes(tmp_path):
    from fpdf import FPDF
    pdf = FPDF(unit="pt", format="letter")
    pdf.add_page()
    pdf.add_font("A", "", str(ARIAL))
    pdf.set_font("A", size=11)
    y = 60
    for t in ["Results. Healthy adult volunteers received a single oral dose of bupropion.",
              "Pharmacokinetic parameters are summarised in Table 1 below.", "Table 1 Pharmacokinetic parameters"]:
        pdf.text(50, y, t)
        y += 14
    rows = [("Group", "Cmax (ng/mL)", "AUC"), ("150 mg", "12З", "1890"), ("300 mg", "455", "1O5")]
    top = y - 10
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            pdf.text(50 + 150 * c, y + 14 * r, cell)
    for k in range(len(rows) + 1):
        pdf.line(45, top + 14 * k, 445, top + 14 * k)
    for x in (45, 195, 345, 445):
        pdf.line(x, top, x, top + 14 * len(rows))
    recs = _keyless(_doc(tmp_path, bytes(pdf.output())))
    _no_clean(recs, "cmax", 12.0)
    _no_clean(recs, "auc", 1.0)


# ------------------------------------------------- zero-advance space ---

def test_zero_advance_space_between_digits_is_not_a_gap(tmp_path):
    c = T.line("F1", 50, 750, "Results. Healthy adult volunteers received a single oral dose of bupropion.")
    c += ("BT /F1 11 Tf 1 0 0 1 50 736 Tm (Cmax and AUC were estimated by NCA \\(n = 12) Tj "
          "-3.058 Tw ( 0\\).) Tj 0 Tw ET\n")
    doc = _doc(tmp_path, T.build(c.encode()))
    assert "(n = 120)." in doc["pages"][0]["text"], doc["pages"][0]["text"]
    _no_clean(_keyless(doc), "n", 12.0)


def test_ordinary_space_between_numbers_stays_a_gap(tmp_path):
    c = T.line("F1", 50, 750, "Results. Samples were taken at 1 2 4 and 8 h after the dose (n = 12).")
    doc = _doc(tmp_path, T.build(c.encode()))
    assert "1 2 4 and 8" in doc["pages"][0]["text"], doc["pages"][0]["text"]


# ------------------------------------------------- WinAnsi 0xAD minus ---

def test_winansi_soft_hyphen_minus_is_not_dropped(tmp_path):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 8, "Clearance 12.4 L/h, volume 45.6 L, half-life 3.5 h.", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "The mean change from baseline was ­20% at week 12.", new_x="LMARGIN", new_y="NEXT")
    doc = _doc(tmp_path, bytes(pdf.output()))
    props = [_prop("change", "20", "change from baseline was 20%", unit="%"),
             _prop("change", "20", "change from baseline was ­20%", unit="%")]
    for rec in check_proposals(props, doc, TEMPLATE):
        assert not _clean_pass(rec), (doc["pages"][0]["text"], rec["verify_reason"], rec["flags"])


# ------------------------------------------------------- parse cache ---

def test_round2_parse_cache_is_not_reused(tmp_path):
    """A cache written before these checks (evidence-parse 5) is parsed again."""
    import json

    from app.evidence import ingest
    from app.evidence.parse import PARSER_NAME, file_sha256
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(T.build(T.table(T.line("F1", 200, 660, "231"), ruled=True).encode()))
    sha = file_sha256(pdf)
    stale = PARSER_NAME.rsplit("evidence-parse", 1)[0] + "evidence-parse 5"
    assert stale != PARSER_NAME
    ev = ingest.evidence_dir(tmp_path)
    (ev / f"{sha}.pages.json").write_text(json.dumps({"sha256": sha, "parser": stale, "pages": []}), "utf-8")
    doc, _ = ingest.parse_and_cache(pdf, tmp_path)
    assert doc["parser"] == PARSER_NAME and doc["pages"]
