"""Page layout sidecar (evidence-parse 7): word boxes in PDF user space, offsets into the page text.

The text-bearing fields of the parse (text, small_text, columns, tables, hidden_chars, ink_check,
glyph_check) must stay byte-identical to evidence-parse 6 (golden hashes recorded on that commit):
stored record spans index them. The boxes come from pdfium's glyph boxes (the ink check's own
geometry), never from pdfplumber's displayed-frame coordinates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.evidence import ingest, parse_ink
from app.evidence import parse as P
from app.evidence import parse_layout as L
from app.evidence.parse import parse_pdf, parse_pdf_with_layout
from app.evidence.parse_ink import _device_affine, _open_document
from tests import evidence_layout_fixtures as F

GOLDEN = json.loads((Path(__file__).parent / "evidence_layout_golden.json").read_text("utf-8"))
KEYS = ("text", "small_text", "columns", "tables", "hidden_chars", "ink_check", "glyph_check")
QUIRK = {"raw_rot180_crop", "raw_rot270_crop", "raw_mbo_rot270_crop"}   # pdfplumber drops lines here (v6 too)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> dict[str, Path]:
    return F.corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="module")
def docs(corpus) -> dict[str, dict]:
    return {name: parse_pdf_with_layout(p) for name, p in corpus.items()}


def _h(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def test_parser_version_is_bumped():
    assert P.PARSER_NAME.endswith("evidence-parse 7")


def test_parse_pdf_contract_has_no_layout_key(corpus):
    assert "layout" not in parse_pdf(corpus["paper_a"])


def test_text_fields_are_byte_identical_to_v6(docs):
    for name, doc in docs.items():
        gold = GOLDEN[name]
        assert doc["decimal_style"] == gold["decimal_style"] and doc["text_chars"] == gold["text_chars"], name
        for pg, want in zip(doc["pages"], gold["pages"], strict=True):
            assert {k: _h(pg[k]) for k in KEYS} == want, name
            assert "layout" not in pg and "words" not in pg


def _layout_page(doc: dict, n: int = 1) -> dict:
    return doc["layout"]["pages"][n - 1]


def test_layout_is_bound_to_the_pdf_and_parser(docs):
    for name, doc in docs.items():
        lay = doc["layout"]
        assert lay["sha256"] == doc["sha256"] and lay["parser"] == doc["parser"], name
        assert [p["n"] for p in lay["pages"]] == [p["n"] for p in doc["pages"]], name


def _covered(text: str, words: list[list]) -> bool:
    """Every non-space char of *text* lies in exactly one word range; ranges are ordered, disjoint."""
    last, seen = 0, set()
    for w in words:
        s, e = w[0], w[1]
        if not (last <= s < e <= len(text)):
            return False
        seen.update(range(s, e))
        last = e
    return all(i in seen for i, ch in enumerate(text) if not ch.isspace() or ch == " ")


def test_every_word_is_mapped_into_the_page_text(docs):
    for name, doc in docs.items():
        for pg, lp in zip(doc["pages"], doc["layout"]["pages"], strict=True):
            assert lp["text_sha256"] == hashlib.sha256(pg["text"].encode()).hexdigest()
            if lp["status"] != "ok":
                assert lp["status"] in ("no_boxes", "skipped", "unmapped"), (name, lp["status"])
                continue
            assert _covered(pg["text"], lp["words"]), name
            assert all(len(w) in (2, 6) for w in lp["words"])


def test_ordinary_pages_are_fully_boxed(docs):
    for name in ("paper_a", "paper_b", "paper_c", "paper_d", "raw_plain", "raw_rot90", "raw_mbo", "raw_crop"):
        for lp in docs[name]["layout"]["pages"]:
            assert lp["status"] == "ok", (name, lp["status"])
            assert lp["words"] and all(len(w) == 6 for w in lp["words"]), name


def _ink_fraction(path: Path, page: int, box: list[float], scale: float = 2.0) -> float:
    doc = _open_document(str(path))
    try:
        pg = doc[page - 1]
        bmp = pg.render(scale=scale, may_draw_forms=True, draw_annots=True)
        arr = bmp.to_numpy()[:, :, :3].min(axis=2) < 140
        aff = _device_affine(bmp.get_posconv(pg))
        l, b, r, t = box
        pts = np.array([[l, b], [r, t], [l, t], [r, b]], float) @ aff[:, :2].T + aff[:, 2]
        x0, y0 = np.floor(pts.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
        h, w = arr.shape
        reg = arr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        return float(reg.mean()) if reg.size else 0.0
    finally:
        doc.close()


@pytest.mark.parametrize("name", list(F.BOXES))
def test_word_boxes_sit_on_ink_for_every_rotation_crop_and_origin(name, corpus, docs):
    key = f"raw_{name}"
    lp = _layout_page(docs[key])
    if key in QUIRK:                                   # fewer lines survive, those that do must still align
        assert lp["status"] in ("ok", "no_boxes")
    else:
        assert lp["status"] == "ok" and len(lp["words"]) >= 12      # 3 lines x 4 words (rotated: per glyph)
    for w in lp["words"]:
        if len(w) == 6:
            assert _ink_fraction(corpus[key], 1, w[2:]) > 0.05, (key, w)


def test_poc_pages_keep_their_flags_and_are_mapped(docs):
    for name in ("tounicode", "poc1_space", "vector_point", "rotated_glyph"):
        doc = docs[name]
        assert doc["pages"][0]["glyph_check"], name
        assert _layout_page(doc)["status"] == "ok", name
        assert any(len(w) == 6 for w in _layout_page(doc)["words"])


def test_unverified_ink_page_has_no_boxes(corpus, monkeypatch):
    monkeypatch.setattr(parse_ink, "MAX_INK_CHARS", 5)
    doc = parse_pdf_with_layout(corpus["paper_a"])
    assert doc["pages"][0]["ink_check"] == "unverified"
    lp = _layout_page(doc)
    assert lp["status"] == "no_boxes" and not lp["words"]


def test_word_cap_skips_the_page(corpus, monkeypatch):
    monkeypatch.setattr(L, "MAX_PAGE_WORDS", 10)
    lp = _layout_page(parse_pdf_with_layout(corpus["paper_a"]))
    assert lp["status"] == "skipped" and not lp["words"]


def test_document_word_cap_skips_later_pages(corpus, docs, monkeypatch):
    monkeypatch.setattr(L, "MAX_DOC_WORDS", len(_layout_page(docs["paper_c"])["words"]) + 1)
    doc = parse_pdf_with_layout(corpus["paper_c"])
    assert [p["status"] for p in doc["layout"]["pages"]] == ["ok", "skipped"]


def test_unmapped_page_when_words_do_not_follow_the_text():
    words = [{"text": "zzz", "chars": []}]
    lp = L.build_page(page_text="Cmax was 3", columns=None, col_lines=None, words=words, char_boxes={},
                      cell_words=[], ink_ok=True, tables=[], doc_words=0)
    assert lp["status"] == "unmapped" and not lp["words"]


# ----------------------------------------------------------------- columns ---

def test_two_column_offsets_index_the_column_strings(docs):
    pg, lp = docs["paper_c"]["pages"][0], _layout_page(docs["paper_c"])
    assert pg["columns"] and lp["col_words"] and lp["col_sha256"] == [
        hashlib.sha256(c.encode()).hexdigest() for c in pg["columns"]]
    for text, words in zip(pg["columns"], lp["col_words"], strict=True):
        assert words and _covered(text, words)


def test_single_column_page_has_no_column_words(docs):
    assert _layout_page(docs["paper_a"])["col_words"] is None


# ------------------------------------------------------------------ tables ---

def _cell(doc: dict, table: int, text: str, page: int = 1) -> tuple[str, list[float]]:
    pg = doc["pages"][page - 1]
    for r, row in enumerate(pg["tables"][table]["rows"]):
        for c, cell in enumerate(row):
            if cell == text:
                key = f"{r}:{c}"
                return key, next(t for t in doc["layout"]["pages"][page - 1]["tables"]
                                 if t["index"] == table)["cells"][key]
    raise AssertionError(text)


def test_ruled_table_cells_have_boxes_on_ink(corpus, docs):
    key, box = _cell(docs["paper_a"], 0, "3,412 (812)")
    assert key == "1:2" and _ink_fraction(corpus["paper_a"], 1, box) > 0.05
    page = docs["paper_a"]["pages"][0]
    cells = next(t for t in _layout_page(docs["paper_a"])["tables"] if t["index"] == 0)["cells"]
    non_empty = sum(1 for row in page["tables"][0]["rows"] for c in row if c)
    assert len(cells) == non_empty


def test_layout_table_cells_have_boxes_and_continuation_rows_merge(corpus, docs):
    _, box = _cell(docs["paper_c"], 0, "18,2 ± 3,4", page=2)
    assert _ink_fraction(corpus["paper_c"], 2, box) > 0.05
    _, label = _cell(docs["paper_c"], 0, "CLint (µL/min/10^6 cells)", page=2)
    assert label[3] - label[1] > 12                    # two text lines tall: the wrapped label is one cell


def test_cell_with_mismatching_words_gets_no_entry():
    ch = {"text": "a"}
    word = {"text": "abc", "chars": [ch]}
    boxes = {id(ch): (1.0, 2.0, 3.0, 4.0)}
    assert L.cell_boxes({"0:0": [word]}, {"0:0": "xyz"}, boxes) == {}
    assert L.cell_boxes({"0:0": [word]}, {"0:0": "abc"}, boxes) == {"0:0": [1.0, 2.0, 3.0, 4.0]}
    assert L.cell_boxes({"0:0": [word]}, {"0:0": "abc"}, {}) == {}          # a glyph without a box: no cell box


# ------------------------------------------------------------ parse_and_cache ---

def test_parse_and_cache_writes_the_sidecar_before_the_pages_file(corpus, tmp_path, monkeypatch):
    order: list[str] = []
    real = ingest._atomic_write
    monkeypatch.setattr(ingest, "_atomic_write", lambda p, d: (order.append(Path(p).name), real(p, d)))
    doc, parse_path = ingest.parse_and_cache(corpus["paper_a"], tmp_path)
    sha = doc["sha256"]
    assert order == [f"{sha}.layout.json", f"{sha}.pages.json"]
    assert "layout" not in json.loads(parse_path.read_text("utf-8"))
    assert json.loads((tmp_path / "evidence" / f"{sha}.layout.json").read_text("utf-8"))["sha256"] == sha


def test_cache_without_a_bound_sidecar_is_parsed_again(corpus, tmp_path):
    doc, parse_path = ingest.parse_and_cache(corpus["paper_a"], tmp_path)
    side = parse_path.with_name(f"{doc['sha256']}.layout.json")
    side.unlink()
    again, _ = ingest.parse_and_cache(corpus["paper_a"], tmp_path)
    assert side.is_file() and again["sha256"] == doc["sha256"]
    bad = json.loads(side.read_text("utf-8"))
    side.write_text(json.dumps({**bad, "sha256": "0" * 64}), "utf-8")
    ingest.parse_and_cache(corpus["paper_a"], tmp_path)
    assert json.loads(side.read_text("utf-8"))["sha256"] == doc["sha256"]


def test_cache_hit_when_pages_and_sidecar_agree(corpus, tmp_path, monkeypatch):
    ingest.parse_and_cache(corpus["paper_a"], tmp_path)
    monkeypatch.setattr(ingest.parse_sandbox, "parse_isolated",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-parsed")))
    ingest.parse_and_cache(corpus["paper_a"], tmp_path)
