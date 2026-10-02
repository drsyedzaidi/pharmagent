"""The page image against hostile PDFs and hostile locators.

The image is the reviewer's one check on what the text layer claims, so it must show the DRAWN glyphs
(never the text layer), carry the flags that say the two may disagree, and never crop around a guess."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from tests import evidence_layout_fixtures as F
from tests import evidence_trickpdf as T
from tests.evidence_image_support import (
    add_record,
    build_papers,
    image,
    ink,
    make_client,
    meta,
    pixels,
    record_of,
    records,
    session_with,
    source_id,
)


@pytest.fixture(scope="module")
def papers(tmp_path_factory):
    return build_papers(tmp_path_factory)


@pytest.fixture
def client(tmp_path, monkeypatch):
    yield from make_client(tmp_path, monkeypatch)


def _control(text: str) -> bytes:
    return T.build(T.table(T.line("F1", 200, 660, text), ruled=True).encode())


def _crop_of(client, sid, name, value):
    rec = record_of(client, sid, name, "cmax", value)
    m = meta(client, sid, rec["id"])
    assert m.status_code == 200, m.text
    body = m.json()
    png = image(client, sid, rec["id"], body["image_sha256"]).content
    return body, pixels(png)


def test_tounicode_image_shows_the_drawn_digit_and_the_record_carries_the_flag(client):
    """Text layer says 241, the page draws 231: the image equals a page that really prints 231."""
    sid = session_with(client, {"poc": F.T.tounicode_pdf(), "ctl231": _control("231"), "ctl241": _control("241")})
    m, shown = _crop_of(client, sid, "poc", "241")
    drawn_m, drawn = _crop_of(client, sid, "ctl231", "231")
    _, intended = _crop_of(client, sid, "ctl241", "241")
    assert m["extracted"]["value"] == "241" and drawn_m["extracted"]["value"] == "231"
    assert "glyph_text_mismatch" in m["flags"] and m["compare_urgent"] is True
    assert any(n["code"] == "text_layer_may_disagree" and n["severity"] == "error"
               and n["text"].startswith("Text layer and page may disagree") for n in m["notices"])
    assert m["page_checks"]["glyph_check"] == ["glyph_text_mismatch"]
    assert shown.shape == drawn.shape and np.array_equal(shown, drawn)          # the DRAWN digit
    assert not np.array_equal(shown, intended)                                  # not the text layer's
    assert drawn_m["flags"] == [] and drawn_m["compare_urgent"] is False        # a clean page stays calm


def test_the_value_outline_sits_on_the_drawn_digits_of_the_poc(client):
    sid = session_with(client, {"poc": F.T.tounicode_pdf()})
    rec = record_of(client, sid, "poc", "cmax", "241")
    m = meta(client, sid, rec["id"]).json()
    png = image(client, sid, rec["id"], m["image_sha256"]).content
    assert m["precision"] == "cell" and ink(png, m["highlight"][0]) > 0.05


EXPECTED = {                                     # PoC -> (cmax value, flag the record must carry)
    "tounicode": ("241", "glyph_text_mismatch"),
    "poc1_space": ("23", "glyph_text_mismatch"),
    "type3": ("231", "glyph_ink_mismatch"),
    "actualtext": ("281", "glyph_text_mismatch"),
    "rotated_glyph": ("236", "glyph_text_mismatch"),
    "vector_point": ("124", "glyph_ink_mismatch"),
    "clipped_eight": ("281", "glyph_ink_mismatch"),
    "covered_half": ("281", "glyph_ink_mismatch"),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_poc_answers_with_its_flag_and_an_urgent_comparison(client, name):
    value, flag = EXPECTED[name]
    sid = session_with(client, {name: F.adversarial()[name]})
    rec = record_of(client, sid, name, "cmax", value)
    r = meta(client, sid, rec["id"])
    assert r.status_code == 200, (name, r.text)
    m = r.json()
    assert flag in m["flags"] and m["compare_urgent"] is True, name
    assert flag in m["page_checks"]["glyph_check"]
    img = image(client, sid, rec["id"], m["image_sha256"])
    assert img.status_code == 200 and img.content.startswith(b"\x89PNG")


@pytest.mark.parametrize("name", ["widget", "white_ruling", "overstrike", "plain_cell"])
def test_other_pocs_never_500(client, name):
    sid = session_with(client, {name: F.adversarial()[name]})
    for rec in records(client, sid):
        r = meta(client, sid, rec["id"])
        assert r.status_code in (200, 422), (name, r.status_code, r.text)
        if r.status_code == 200:
            body = r.json()
            assert image(client, sid, rec["id"], body["image_sha256"]).status_code == 200
            if "hidden_text" in body["flags"]:
                assert body["compare_urgent"] is True


@pytest.mark.parametrize("name", sorted(F.poc_extras()))
def test_round_two_pocs_answer_and_never_500(client, name):
    sid = session_with(client, {name: F.poc_extras()[name]})
    recs = records(client, sid)
    assert recs, name
    for rec in recs:
        r = meta(client, sid, rec["id"])
        assert r.status_code in (200, 422), (name, r.status_code, r.text)
        if r.status_code == 200:
            body = r.json()
            assert image(client, sid, rec["id"], body["image_sha256"]).status_code == 200
            if body["flags"] or body["page_checks"]["glyph_check"]:
                assert body["compare_urgent"] or not body["flags"]


def test_widget_record_is_flagged_hidden_text_and_urgent(client):
    sid = session_with(client, {"widget": F.adversarial()["widget"]})
    rec = next(r for r in records(client, sid) if r["field"] == "cmax")
    m = meta(client, sid, rec["id"]).json()
    assert "hidden_text" in m["flags"] and m["compare_urgent"] is True
    assert m["page_checks"]["hidden_chars"] > 0


def _dense_page_pdf() -> bytes:
    """A ruled PK table plus 27 000 tiny digits: past the ink check's glyph cap the page is 'unverified'
    (the colour-rule text is kept and every record citing it is flagged)."""
    filler = "".join(f"BT /F1 3 Tf 1 0 0 1 4 {300 - 2.5 * i:.1f} Tm ({'0123456789' * 36}) Tj ET\n" for i in range(75))
    return T.build((T.table(T.line("F1", 200, 660, "3412"), ruled=True) + filler).encode())


def test_unverified_ink_page_is_urgent_and_says_so(client):
    sid = session_with(client, {"dense": _dense_page_pdf()})
    rec = next(r for r in records(client, sid) if r["field"] == "cmax")
    m = meta(client, sid, rec["id"])
    assert m.status_code == 200, m.text
    m = m.json()
    assert m["page_checks"]["ink_check"] == "unverified" and m["compare_urgent"] is True
    assert m["precision"] == "page" and m["reason"] == "no_boxes"           # no trusted glyph geometry
    codes = {n["code"] for n in m["notices"]}
    assert {"page_unchecked", "value_not_located"} <= codes
    assert image(client, sid, rec["id"], m["image_sha256"]).status_code in (200, 409)


# -------------------------------------------------------- hostile locators ---

@pytest.fixture
def seeded(client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()})
    return sid, source_id(client, sid, "paper_a")


BASE = {"page": 1, "kind": "text", "snippet": "Cmax", "span": [0, 4], "literal": "x", "token": [0, 1], "column": None}
TABLE = {"page": 1, "kind": "table", "snippet": "3,412", "span": None, "literal": "3,412", "token": [0, 5],
         "cell": {"table": 0, "row": 1, "col": 2, "header": "Cmax (ng/mL)", "row_label": "150 mg"}}
FUZZ = [
    {**BASE, "span": [10**9, 10**9 + 5]}, {**BASE, "span": [-5, 4]}, {**BASE, "span": [True, 4]},
    {**BASE, "span": [4, 0]}, {**BASE, "span": "0:4"}, {**BASE, "span": [0.5, 3.5]}, {**BASE, "span": [10**30, 10**31]},
    {**BASE, "token": [10**9, 10**9 + 1]}, {**BASE, "token": [-1, 2]}, {**BASE, "token": {"a": 1}},
    {**BASE, "literal": 5}, {**BASE, "literal": None, "token": [0, 1]}, {**BASE, "column": 99},
    {**BASE, "column": True}, {**BASE, "column": -1}, {**BASE, "column": "0"}, {**BASE, "kind": "figure"},
    {**BASE, "kind": ["table"]}, {**BASE, "snippet": "Z" * 5000}, {**BASE, "snippet": None},
    {**TABLE, "cell": {"table": 10**9, "row": 1, "col": 2}}, {**TABLE, "cell": {"table": 0, "row": 10**9, "col": 0}},
    {**TABLE, "cell": {"table": 0, "row": 1, "col": 10**9}}, {**TABLE, "cell": {"table": -1, "row": 0, "col": 0}},
    {**TABLE, "cell": {"table": True, "row": 1, "col": 2}}, {**TABLE, "cell": "0:1:2"}, {**TABLE, "cell": None},
    {**TABLE, "cell": {"table": 0, "row": 1.5, "col": 2}}, {**TABLE, "token": [0, 99]}, {**TABLE, "literal": "9,999"},
]


@pytest.mark.parametrize("i", range(len(FUZZ)))
def test_fuzzed_locators_never_500_and_never_crop(client, seeded, i):
    sid, src = seeded
    rid = add_record(client, sid, src, {**FUZZ[i], "page": 1}, value=f"v{i}", verify_status="fail")
    r = meta(client, sid, rid)
    assert r.status_code in (200, 422), (i, r.status_code, r.text)
    assert str(settings.data_dir) not in r.text
    if r.status_code == 200:
        m = r.json()
        assert m["precision"] == "page" and m["resolved"] is False and m["highlight"] == [] and m["compare_urgent"]
        assert m["dpi"] == 90.0 and m["snippet_box"] is None


def test_no_server_path_appears_in_any_error_body(client, seeded):
    sid, src = seeded
    rid = add_record(client, sid, src, {"page": 9, "kind": "text", "snippet": "x"}, verify_status="fail")
    bodies = [meta(client, sid, rid).text, meta(client, sid, "nope").text, meta(client, sid, rid, zoom="x").text,
              image(client, sid, rid, "0" * 64).text]
    for b in bodies:
        assert str(settings.data_dir) not in b and ".pdf" not in b and "Traceback" not in b
