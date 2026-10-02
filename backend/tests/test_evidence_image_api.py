"""GET /records/{id}/image/meta and /image over HTTP: authorisation, geometry, cache, limits, headers."""
from __future__ import annotations

import hashlib
import struct
import threading
import time

import pytest

from app.config import settings
from app.evidence import ingest, render, render_cache, render_limits
from app.evidence import store as evstore
from tests.evidence_image_support import (
    EV,
    add_record,
    build_papers,
    drop_layout,
    ink,
    make_client,
    meta,
    record_of,
    records,
    session_with,
    source_id,
    verified_text_record,
)
from tests.evidence_image_support import image as fetch_image


@pytest.fixture(scope="module")
def papers(tmp_path_factory):
    return build_papers(tmp_path_factory)


@pytest.fixture
def client(tmp_path, monkeypatch):
    yield from make_client(tmp_path, monkeypatch)


def _one(client, papers, names=("paper_a",)):
    sid = session_with(client, {n: papers[n].read_bytes() for n in names})
    return sid


def test_table_cell_record_meta_and_image(client, papers):
    sid = _one(client, papers)
    rec = record_of(client, sid, "paper_a", "cmax", "3,412")
    r = meta(client, sid, rec["id"])
    assert r.status_code == 200, r.text
    m = r.json()
    assert (m["precision"], m["resolved"], m["reason"], m["zoom"], m["dpi"]) == ("cell", True, None, 1, 144.0)
    assert m["record_id"] == rec["id"] and m["page"] == 1 and m["pages"] == 1 and m["flags"] == []
    assert m["extracted"] == {"value": "3,412", "unit": "ng/mL", "literal": "3,412", "kind": "table"}
    assert m["compare_urgent"] is False and m["notices"] == []
    assert m["page_checks"] == {"ink_check": "ok", "hidden_chars": 0, "glyph_check": []}
    assert 1 <= len(m["highlight"]) <= 12 and m["snippet_box"] and len(m["crop_pt"]) == 4
    img = fetch_image(client, sid, rec["id"], m["image_sha256"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert struct.unpack(">II", img.content[16:24]) == (m["width"], m["height"])
    assert hashlib.sha256(img.content).hexdigest() == m["image_sha256"] == img.headers["x-image-sha256"]
    assert len(img.content) == m["png_bytes"]
    assert ink(img.content, m["highlight"][0]) > 0.05          # glyph ink under the outline
    for r_ in m["highlight"]:
        assert r_["x"] >= 0 and r_["y"] >= 0 and r_["x"] + r_["w"] <= m["width"] and r_["y"] + r_["h"] <= m["height"]


def test_image_headers_are_hardened(client, papers):
    sid = _one(client, papers)
    rec = record_of(client, sid, "paper_a", "cmax", "3,412")
    m = meta(client, sid, rec["id"])
    assert m.headers["cache-control"] == "private, no-store" and "authorization" in m.headers["vary"].lower()
    img = fetch_image(client, sid, rec["id"], m.json()["image_sha256"])
    h = img.headers
    assert h["x-content-type-options"] == "nosniff" and h["cache-control"] == "private, no-store"
    assert "authorization" in h["vary"].lower() and h["content-disposition"] == 'inline; filename="record.png"'
    assert h["content-security-policy"] == "default-src 'none'; sandbox"
    assert h["cross-origin-resource-policy"] == "same-origin" and int(h["content-length"]) == len(img.content)


def test_text_record_is_outlined_at_the_value(client, papers):
    sid = _one(client, papers, ("paper_d",))
    rid = verified_text_record(client, sid, "paper_d", field="cl", value="2.31", snippet="was 2.31 L/h")
    m = meta(client, sid, rid).json()
    assert (m["precision"], m["resolved"]) == ("value", True)
    img = fetch_image(client, sid, rid, m["image_sha256"])
    assert ink(img.content, m["highlight"][0]) > 0.05 and m["snippet_box"]["w"] > m["highlight"][0]["w"]


def test_two_column_text_record_uses_the_column_words(client, papers):
    sid = _one(client, papers, ("paper_c",))
    rid = verified_text_record(client, sid, "paper_c", field="species", value="Sprague-Dawley",
                               snippet="Sprague-Dawley rats", field_type="text", column=0)
    stored = evstore.get_store().get_record(rid)
    assert stored.locator["column"] == 0
    m = meta(client, sid, rid).json()
    assert m["resolved"] is True and m["precision"] == "line"
    img = fetch_image(client, sid, rid, m["image_sha256"])
    assert ink(img.content, m["highlight"][0]) > 0.03


def test_layout_table_record_is_a_cell(client, papers):
    sid = _one(client, papers, ("paper_c",))
    rec = record_of(client, sid, "paper_c", "clint")
    m = meta(client, sid, rec["id"]).json()
    assert m["precision"] == "cell" and m["resolved"]
    assert "table_low_confidence" in {f["code"] for f in rec["flags"]}      # QC flag unchanged by the image


def test_zoom_two_is_sharper(client, papers):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    one, two = meta(client, sid, rid, zoom=1).json(), meta(client, sid, rid, zoom=2).json()
    assert two["dpi"] == 216.0 and two["image_sha256"] != one["image_sha256"]
    assert abs(two["width"] / 3 - one["width"] / 2) < 2


def test_unlocated_value_is_the_whole_page_at_90_dpi_and_urgent(client, papers):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    drop_layout(client, sid, "paper_a")
    m = meta(client, sid, rid).json()
    assert (m["precision"], m["resolved"], m["reason"], m["dpi"]) == ("page", False, "layout_unavailable", 90.0)
    assert m["highlight"] == [] and m["snippet_box"] is None and m["compare_urgent"] is True
    note = next(n for n in m["notices"] if n["code"] == "value_not_located")
    assert note["text"].startswith("Could not locate the cited value on the page. Showing the whole page")
    assert "parsed before page positions were stored" in note["text"]
    img = fetch_image(client, sid, rid, m["image_sha256"])
    assert (m["width"], m["height"]) == (round(595.28 * 1.25), round(841.89 * 1.25)) or m["width"] > 700
    assert img.status_code == 200


def test_old_parser_cache_is_unavailable_not_an_error(client, papers):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    src = evstore.get_store().get_source(source_id(client, sid, "paper_a"))
    side = ingest.Path(src.parse_path).with_name(f"{src.sha256}.layout.json")
    import json
    doc = json.loads(side.read_text())
    side.write_text(json.dumps({**doc, "parser": "pdfplumber 0.11.10 / evidence-parse 6"}))
    m = meta(client, sid, rid).json()
    assert m["precision"] == "page" and m["reason"] == "layout_unavailable"


def test_verify_failed_record_has_no_span_and_shows_the_page(client, papers):
    sid = _one(client, papers)
    sid_src = source_id(client, sid, "paper_a")
    rid = add_record(client, sid, sid_src, {"page": 1, "kind": "text", "snippet": "not on the page", "span": None,
                                            "literal": None, "token": None, "column": None},
                     verify_status="fail")
    m = meta(client, sid, rid).json()
    assert (m["precision"], m["resolved"], m["reason"]) == ("page", False, "no_span") and m["compare_urgent"]


# ------------------------------------------------------------ authorisation ---

def test_other_session_unknown_record_and_foreign_source_are_404(client, papers):
    s1 = _one(client, papers)
    s2 = _one(client, papers, ("paper_b",))
    rid = record_of(client, s1, "paper_a", "cmax", "3,412")["id"]
    for sid, record in ((s2, rid), (s1, "rec_missing"), (s1, " "), (s1, "rec_" + "a" * 300)):
        assert meta(client, sid, record).status_code == 404, (sid, record)
        assert fetch_image(client, sid, record, "0" * 64).status_code == 404
    # a record of session 2 that points at a source of session 1
    foreign = add_record(client, s2, source_id(client, s1, "paper_a"),
                         {"page": 1, "kind": "text", "snippet": "x", "span": None, "literal": None, "token": None})
    r = meta(client, s2, foreign)
    assert r.status_code == 404 and "paper_a" not in r.text and "evidence" not in r.json()["error"]["message"].lower()


def test_unknown_session_is_404(client):
    assert meta(client, "no-such-session", "x").status_code == 404


def test_token_is_required_and_the_owner_is_enforced(client, papers):
    settings.api_token = "secret-token"
    h = {"Authorization": "Bearer secret-token"}
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, h=h)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412", h=h)["id"]
    assert meta(client, sid, rid).status_code == 401
    assert fetch_image(client, sid, rid, "0" * 64).status_code == 401
    ok = meta(client, sid, rid, h=h)
    assert ok.status_code == 200
    assert fetch_image(client, sid, rid, ok.json()["image_sha256"], h=h).status_code == 200
    settings.api_token = "another-token"
    assert meta(client, sid, rid, h={"Authorization": "Bearer another-token"}).status_code == 403


# ------------------------------------------------------------------ inputs ---

def test_zoom_must_be_one_or_two(client, papers):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    for bad in ("3", "0", "-1", "abc", "1.5", "", "1 "):
        r = meta(client, sid, rid, zoom=bad)
        assert r.status_code == 400 and r.json()["error"]["message"] == "zoom must be 1 or 2", bad
        assert fetch_image(client, sid, rid, "0" * 64, zoom=bad).status_code == 400


def test_other_query_parameters_cannot_change_the_output(client, papers):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    base = meta(client, sid, rid).json()
    noisy = meta(client, sid, rid, dpi=600, x0=0, y0=0, x1=9999, y1=9999, box="1,2,3,4", scale=9, page=7).json()
    assert noisy == base
    a = fetch_image(client, sid, rid, base["image_sha256"])
    b = fetch_image(client, sid, rid, base["image_sha256"], dpi=600, x0=0, width=99999, page=3)
    assert a.content == b.content


def test_sha_is_required_and_must_match(client, papers):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    good = meta(client, sid, rid).json()["image_sha256"]
    for bad in (None, "", "0" * 64, good[:-1], good.upper() + "x"):
        r = fetch_image(client, sid, rid, bad)
        assert r.status_code == 409 and r.json()["error"]["message"] == "page image changed; reload", bad
    assert fetch_image(client, sid, rid, good).status_code == 200


def test_record_citing_a_missing_page_is_422(client, papers):
    sid = _one(client, papers)
    src = source_id(client, sid, "paper_a")
    for page in (99, 0, -1, True, "1", None, 1.5):
        rid = add_record(client, sid, src, {"page": page, "kind": "text", "snippet": f"s{page}", "span": None,
                                            "literal": None, "token": None}, verify_status="fail")
        r = meta(client, sid, rid)
        assert r.status_code == 422 and "page that does not exist" in r.json()["error"]["message"], page


def test_pdf_changed_on_disk_is_409(client, papers):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    src = evstore.get_store().get_source(source_id(client, sid, "paper_a"))
    with open(src.pdf_path, "ab") as fh:
        fh.write(b"\n%tampered")
    r = meta(client, sid, rid)
    assert r.status_code == 409 and "changed on disk" in r.json()["error"]["message"]
    assert src.pdf_path not in r.text


# ------------------------------------------------------------ cache + limits ---

def _count_renders(monkeypatch) -> list[int]:
    calls = [0]
    real = render.render_page

    def counting(*a, **k):
        calls[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(render, "render_page", counting)
    return calls


def test_second_call_is_a_cache_hit_without_a_child(client, papers, monkeypatch):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    calls = _count_renders(monkeypatch)
    first = meta(client, sid, rid).json()
    assert calls[0] == 1
    assert meta(client, sid, rid).json() == first and fetch_image(client, sid, rid, first["image_sha256"]).status_code == 200
    assert calls[0] == 1


def test_cache_survives_a_second_session_with_the_same_pdf(client, papers, monkeypatch):
    s1, s2 = _one(client, papers), _one(client, papers)
    r1, r2 = (record_of(client, s, "paper_a", "cmax", "3,412")["id"] for s in (s1, s2))
    calls = _count_renders(monkeypatch)
    a, b = meta(client, s1, r1).json(), meta(client, s2, r2).json()
    assert calls[0] == 1 and a["image_sha256"] == b["image_sha256"]


def test_evicted_entry_is_rendered_again_on_the_image_call(client, papers, monkeypatch):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    sha = meta(client, sid, rid).json()["image_sha256"]
    for p in render_cache.render_dir(settings.data_dir).iterdir():
        p.unlink()
    calls = _count_renders(monkeypatch)
    assert fetch_image(client, sid, rid, sha).status_code == 200 and calls[0] == 1


def test_sixty_first_uncached_render_is_429_with_retry_after_and_hits_are_free(client, papers, monkeypatch):
    monkeypatch.setattr(render_limits, "RENDERS_PER_WINDOW", 2)
    sid = _one(client, papers)
    ids = [r["id"] for r in records(client, sid) if r["field"] in ("cmax", "auc", "n")][:3]
    assert len(ids) == 3
    assert meta(client, sid, ids[0]).status_code == 200 and meta(client, sid, ids[1]).status_code == 200
    r = meta(client, sid, ids[2])
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert meta(client, sid, ids[0]).status_code == 200                     # a hit costs nothing
    render_limits.reset()
    assert meta(client, sid, ids[2]).status_code == 200


def test_busy_renderer_is_503_with_retry_after(client, papers, monkeypatch):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    monkeypatch.setattr(render, "RENDER_QUEUE_WAIT_S", 0.05)
    held = [render._render_slots.acquire() for _ in range(render.RENDER_SLOTS)]
    try:
        r = meta(client, sid, rid)
    finally:
        for _ in held:
            render._render_slots.release()
    assert r.status_code == 503 and int(r.headers["retry-after"]) >= 1
    assert meta(client, sid, rid).status_code == 200


def test_render_failure_is_a_generic_422(client, papers, monkeypatch):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    monkeypatch.setattr(render, "RENDER_TARGET", "tests.evidence_sandbox_targets:sleep_with_args")
    monkeypatch.setattr(render, "RENDER_WALL_S", 0.5)
    r = meta(client, sid, rid)
    assert r.status_code == 422 and r.json()["error"]["message"] == "the page could not be rendered"
    assert str(settings.data_dir) not in r.text


def test_only_two_children_run_at_once_for_four_parallel_clients(client, papers, monkeypatch):
    sid = _one(client, papers)
    ids = [r["id"] for r in records(client, sid)][:4]
    live, peak, lock = [0], [0], threading.Lock()
    real = render.run_child

    def counting(*a, **k):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        try:
            time.sleep(0.25)
            return real(*a, **k)
        finally:
            with lock:
                live[0] -= 1
    monkeypatch.setattr(render, "run_child", counting)
    out: list[int] = []
    threads = [threading.Thread(target=lambda i=i: out.append(meta(client, sid, i).status_code)) for i in ids]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(out) == [200] * len(ids) and peak[0] <= render.RENDER_SLOTS


def test_no_audit_entry_and_no_state_change_from_viewing(client, papers):
    sid = _one(client, papers)
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    before = client.get(f"/api/sessions/{sid}/audit").json()["entries"]
    snap = client.get(EV.format(sid=sid)).json()["records"]
    meta(client, sid, rid)
    assert client.get(f"/api/sessions/{sid}/audit").json()["entries"] == before
    assert client.get(EV.format(sid=sid)).json()["records"] == snap


def _big_page_pdf(side: int) -> bytes:
    from tests import evidence_rawpdf as raw
    ops = "".join(f"BT /F1 11 Tf 1 0 0 1 {x} {y} Tm ({t}) Tj ET\n" for x, y, t in (
        (50, side - 90, "Table 1 Pharmacokinetic parameters"), (50, side - 110, "Group"),
        (250, side - 110, "Cmax (ng/mL)"), (450, side - 110, "AUC"), (50, side - 124, "150 mg"),
        (250, side - 124, "3412"), (450, side - 124, "1890"), (50, side - 138, "300 mg"),
        (250, side - 138, "455"), (450, side - 138, "3720")))
    return raw.build([([0, 0, side, side], ops.encode())])


def test_page_too_large_for_a_readable_render_is_a_generic_422(client):
    sid = session_with(client, {"big": _big_page_pdf(3000)})
    rec = next(r for r in records(client, sid) if r["field"] == "cmax")
    r = meta(client, sid, rec["id"])
    assert r.status_code == 422 and r.json()["error"]["message"] == "the page is too large to render at a readable resolution"
    assert str(settings.data_dir) not in r.text


def test_big_but_readable_page_is_rendered_at_a_lower_dpi(client):
    sid = session_with(client, {"big": _big_page_pdf(2000)})
    rec = next(r for r in records(client, sid) if r["field"] == "cmax")
    m = meta(client, sid, rec["id"])
    assert m.status_code == 200, m.text
    assert m.json()["dpi"] < 144 and m.json()["width"] * m.json()["height"] <= 4_000_000
