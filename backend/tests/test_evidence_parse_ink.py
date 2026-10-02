"""Rendered-ink visibility: a glyph reaches the page text only when it puts visible ink on the rendered
page (CropBox respected). Text covered by a later fill, drawn in near-background grey, clipped away,
squashed to no width, drawn in the colour of the box under it, or placed outside the CropBox is
dropped and counted in hidden_chars, so the verifier can never cite it. A page whose render fails or
runs out of budget is marked unverified and every record citing it carries an error flag."""
from __future__ import annotations

import pytest

from app.evidence import parse_ink
from app.evidence.extract_tools import check_proposals
from app.evidence.parse import parse_pdf
from app.evidence.record_checks import hidden_text_flags
from tests import evidence_fixtures as fx
from tests import evidence_rawpdf as raw

BOX = [0, 0, 612, 792]
LINE = "Cmax was 9999 ng/mL in all patients"
BODY = raw.text(raw.BODY * 3)
HIDDEN = {
    "white_rect_over_black_text": BODY + raw.text([LINE], y0=400) + b"1 1 1 rg 40 380 500 40 re f\n",
    "near_white_grey_0_97": BODY + raw.text([LINE], y0=400, prefix="0.97 0.97 0.97 rg"),
    "outside_clip_path": BODY + b"q 0 0 10 10 re W n\n" + raw.text([LINE], y0=400) + b"Q\n",
    "horizontal_squash_1_Tz": BODY + raw.text([LINE], y0=400, prefix="1 Tz"),
    "same_colour_on_same_colour_box": BODY + b"1 0 0 rg 40 380 500 40 re f\n"
                                      + raw.text([LINE], y0=400, prefix="1 0 0 rg"),
}


def _page(tmp_path, content, extras=None, name="x.pdf"):
    path = tmp_path / name
    path.write_bytes(raw.build([(BOX, content, extras or {})]))
    return parse_pdf(path)["pages"][0]


@pytest.mark.parametrize("case", sorted(HIDDEN))
def test_hidden_variants_are_removed_and_counted(tmp_path, case):
    page = _page(tmp_path, HIDDEN[case])
    assert "9999" not in page["text"] and "9999" not in page["small_text"], case
    assert page["hidden_chars"] >= len(LINE.replace(" ", "")), case
    assert "rat hepatocytes" in page["text"] and page["ink_check"] == "ok"


def test_text_outside_the_cropbox_is_removed(tmp_path):
    page = _page(tmp_path, BODY + raw.text([LINE], y0=400), {"CropBox": [0, 500, 612, 792]})
    assert "9999" not in page["text"] and page["hidden_chars"] >= len(LINE.replace(" ", ""))
    assert "rat hepatocytes" in page["text"]


@pytest.mark.parametrize("prefix", ["", "0.6 g", "0 0 1 rg"])
def test_visible_control_is_kept(tmp_path, prefix):
    page = _page(tmp_path, BODY + raw.text([LINE], y0=400, prefix=prefix))
    assert LINE in page["text"] and page["hidden_chars"] == 0 and page["ink_check"] == "ok"


def test_hidden_value_does_not_verify_and_visible_one_does(tmp_path):
    prop = {"field": "Cmax", "value": "9999", "unit": "ng/mL", "value_num": 9999.0, "group_label": "",
            "source_id": "s", "locator": {"page": 1, "kind": "text", "snippet": "Cmax was 9999 ng/mL",
                                          "decimal_style": "."}}
    tmpl = {"fields": [{"name": "Cmax", "type": "number", "unit": "ng/mL"}]}
    for content, want in ((HIDDEN["white_rect_over_black_text"], "fail"), (BODY + raw.text([LINE], y0=400), "pass")):
        path = tmp_path / f"{want}.pdf"
        path.write_bytes(raw.build([(BOX, content)]))
        assert check_proposals([prop], parse_pdf(path), tmpl)[0]["verify_status"] == want


def test_hidden_line_superimposed_on_visible_text_is_not_trusted(tmp_path):
    """Its glyph boxes hold another line's ink, so that ink proves nothing about it."""
    visible = raw.text(["Cmax was 1234 ng/mL in all patients"], y0=400)
    for dy in (0, 4):
        hidden = raw.text([LINE], y0=400 + dy, prefix="0.97 0.97 0.97 rg")
        page = _page(tmp_path, BODY + visible + hidden, name=f"s{dy}.pdf")
        assert "9999" not in page["text"] and page["hidden_chars"] > 0, dy


@pytest.mark.parametrize("rotate", [90, 180, 270])
def test_rotated_pages_keep_visible_text_and_drop_covered_text(tmp_path, rotate):
    """pdfplumber reads rotated pages in display order (lines may split or reverse); the ink check must
    keep every visible glyph and still drop the covered ones."""
    page = _page(tmp_path, BODY + raw.text([LINE], y0=400), {"Rotate": rotate}, name=f"v{rotate}.pdf")
    assert "9999" in page["text"] and page["hidden_chars"] == 0 and page["ink_check"] == "ok"
    page = _page(tmp_path, HIDDEN["white_rect_over_black_text"], {"Rotate": rotate}, name=f"h{rotate}.pdf")
    assert "9999" not in page["text"] and page["hidden_chars"] >= len(LINE.replace(" ", ""))
    assert page["ink_check"] == "ok"


def test_offset_media_and_crop_boxes_keep_visible_text(tmp_path):
    box = [100, 200, 712, 992]
    content = raw.text(raw.BODY, x=150, y0=900) + raw.text([LINE], x=150, y0=600)
    path = tmp_path / "off.pdf"
    path.write_bytes(raw.build([(box, content, {"CropBox": [120, 300, 700, 980]})]))
    page = parse_pdf(path)["pages"][0]
    assert LINE in page["text"] and page["hidden_chars"] == 0


def test_text_inside_a_form_xobject_is_judged_like_page_text(tmp_path):
    form = raw.text([LINE], y0=400)
    page = _page(tmp_path, BODY + b"/Fm1 Do\n", {"XObject": {"Fm1": form}})
    assert LINE in page["text"] and page["hidden_chars"] == 0
    covered = BODY + b"/Fm1 Do\n1 1 1 rg 40 380 500 40 re f\n"
    page = _page(tmp_path, covered, {"XObject": {"Fm1": form}}, name="c.pdf")
    assert "9999" not in page["text"] and page["hidden_chars"] > 0


def test_render_failure_marks_the_page_unverified_and_flags_records(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("pdfium failed")

    monkeypatch.setattr(parse_ink, "_render", boom)
    page = _page(tmp_path, BODY + raw.text([LINE], y0=400))
    assert page["ink_check"] == "unverified"
    flags = hidden_text_flags({"field": "Cmax", "group_label": ""}, page)
    assert flags and flags[0]["code"] == "hidden_text" and flags[0]["severity"] == "error"


def test_render_budget_exhaustion_marks_pages_unverified(tmp_path, monkeypatch):
    monkeypatch.setattr(parse_ink, "RENDER_BUDGET_S", -1.0)
    assert _page(tmp_path, BODY + raw.text([LINE], y0=400))["ink_check"] == "unverified"


def test_unopenable_render_marks_every_page_unverified(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("cannot open")

    monkeypatch.setattr(parse_ink, "_open_document", boom)
    assert _page(tmp_path, BODY)["ink_check"] == "unverified"


def test_fixture_papers_lose_no_visible_text(tmp_path):
    for name, path in fx.build_all(tmp_path).items():
        doc = parse_pdf(path)
        assert all(p["hidden_chars"] == 0 and p["ink_check"] == "ok" for p in doc["pages"]), name


def test_kerned_period_and_hairline_hyphen_stay_visible(tmp_path):
    """Real-paper regressions: a period kerned under the previous glyph, and a 6.5 pt hyphen whose ink
    sits a pixel outside pdfium's tight box, are visible text."""
    kerned = b"BT /F1 11 Tf 1 0 0 1 50 400 Tm [(R.F.T) 150 (.) ( Received)] TJ ET\n"
    hyphen = b"BT /F1 6.5 Tf 1 0 0 1 50 380 Tm (drug-related co-medication) Tj ET\n"
    page = _page(tmp_path, BODY + kerned + hyphen)
    assert "T.Received" in page["text"].replace(" ", "")
    assert "drug-related co-medication" in page["small_text"]      # 6.5 pt is below the body-size cut
    assert page["hidden_chars"] == 0


def test_a_copy_drawn_over_visible_digits_does_not_double_them(tmp_path):
    line = raw.text(["Cmax was 99 ng/mL"], y0=400)
    page = _page(tmp_path, BODY + line + line)
    assert "Cmax was 99 ng/mL" in page["text"] and "9999" not in page["text"]
