"""The page-image renderer: the child target (crop policy, geometry, caps) and the sandboxed parent
(limits, slots, environment). Pixels are compared with an independent pdfium render of the same page."""
from __future__ import annotations

import base64
import io
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.evidence import ingest, parse_sandbox, regions, render, render_child
from app.evidence import parse_layout as L  # noqa: F401  (documents the box source)
from app.evidence.parse import parse_pdf_with_layout
from app.evidence.parse_ink import _device_affine, _open_document
from tests import evidence_layout_fixtures as F
from tests import evidence_rawpdf as raw

TARGETS = "tests.evidence_sandbox_targets"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> dict[str, Path]:
    return F.corpus(tmp_path_factory.mktemp("render"))


@pytest.fixture(scope="module")
def docs(corpus) -> dict[str, dict]:
    return {n: parse_pdf_with_layout(corpus[n]) for n in corpus}


def _value_box(doc: dict, word: str = "3,412", page: int = 1, nth: int = 0) -> list[float]:
    text, lp = doc["pages"][page - 1]["text"], doc["layout"]["pages"][page - 1]
    hits = [w for w in lp["words"] if len(w) == 6 and text[w[0]:w[1]] == word]
    if not hits:                          # pdfplumber reads rotated pages glyph by glyph: take a digit
        hits = [w for w in lp["words"] if len(w) == 6 and text[w[0]:w[1]].isdigit()]
    return hits[nth][2:]


def _decode(reply: dict) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(reply["png_b64"]))).convert("RGB"))


def _reference(path: Path, scale: float, page: int = 1) -> tuple[np.ndarray, np.ndarray]:
    doc = _open_document(str(path))
    try:
        pg = doc[page - 1]
        bmp = pg.render(scale=scale, may_draw_forms=True, draw_annots=True)
        return bmp.to_numpy()[:, :, :3][:, :, ::-1], _device_affine(bmp.get_posconv(pg))
    finally:
        doc.close()


def _child(path: Path, **kw) -> dict:
    return render_child.render_region(str(path), **{"page": 1, "scale": 2.0, **kw})


# ------------------------------------------------------------------ geometry ---

def test_png_matches_the_reply_and_the_pixels_are_the_inks_own_raster(corpus, docs):
    box = _value_box(docs["raw_plain"])
    reply = _child(corpus["raw_plain"], value_boxes=[box], snippet_box=box)
    png = base64.b64decode(reply["png_b64"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", png[16:24]) == (reply["width"], reply["height"])
    assert (reply["scale"], reply["dpi"]) == (2.0, 144.0)
    ref, _ = _reference(corpus["raw_plain"], 2.0)
    x0, y0, x1, y1 = reply["crop_px"]
    assert np.array_equal(_decode(reply), ref[y0:y1, x0:x1])           # byte for byte, no annotation


def test_rects_sit_on_the_value_inside_the_image(corpus, docs):
    box = _value_box(docs["raw_plain"])
    reply = _child(corpus["raw_plain"], value_boxes=[box], snippet_box=box)
    img = _decode(reply)
    (x, y, w, h), = reply["rects_px"]
    assert 0 <= x and 0 <= y and x + w <= reply["width"] and y + h <= reply["height"]
    region = img[int(y):int(y + h) + 1, int(x):int(x + w) + 1].min(axis=2) < 140
    assert region.mean() > 0.05                                        # glyph ink under the outline


def test_crop_has_padding_minimum_size_and_stays_inside_the_page(corpus, docs):
    box = _value_box(docs["raw_plain"])
    reply = _child(corpus["raw_plain"], value_boxes=[box], snippet_box=box)
    w_pt, h_pt = reply["width"] / 2.0, reply["height"] / 2.0
    assert w_pt >= render_child.MIN_W - 1 and h_pt >= render_child.MIN_H - 1
    assert w_pt <= 612 and h_pt <= 792
    l, b, r, t = reply["crop_pt"]
    assert l <= box[0] and box[2] <= r and b <= box[1] and box[3] <= t
    assert l >= -1 and b >= -1 and r <= 613 and t <= 793


def test_oversized_snippet_falls_back_to_the_value_and_oversized_context_to_the_whole_page(corpus, docs):
    reply = _child(corpus["raw_plain"], value_boxes=[[100, 100, 120, 110]], snippet_box=[20, 40, 590, 750])
    assert reply["crop_px"] != [0, 0, 1224, 1584] and reply["width"] < 1224      # the value's own crop
    wide = _child(corpus["raw_plain"], value_boxes=[[100, 100, 120, 110]], context_boxes=[[10, 10, 600, 780]])
    assert wide["crop_px"] == [0, 0, 1224, 1584]


def test_whole_page_flag_renders_the_page_at_the_lower_scale(corpus):
    reply = _child(corpus["raw_plain"], scale=1.25, whole_page=True)
    assert (reply["width"], reply["height"]) == (765, 990) and reply["rects_px"] == []


@pytest.mark.parametrize("name", ["raw_rot90", "raw_rot270", "raw_crop", "raw_mbo", "raw_rot90_crop",
                                  "raw_mbo_rot90_crop", "raw_mbo_rot270_crop"])
def test_crop_follows_rotation_cropbox_and_mediabox_origin(name, corpus, docs):
    box = _value_box(docs[name])
    reply = _child(corpus[name], value_boxes=[box], snippet_box=box)
    img = _decode(reply)
    (x, y, w, h), = reply["rects_px"]
    assert 0 <= x and 0 <= y and x + w <= reply["width"] and y + h <= reply["height"], name
    assert (img[int(y):int(y + h) + 1, int(x):int(x + w) + 1].min(axis=2) < 140).mean() > 0.05, name


def test_zoom_two_is_sharper_not_wider(corpus, docs):
    box = _value_box(docs["raw_plain"])
    one = _child(corpus["raw_plain"], value_boxes=[box], snippet_box=box, scale=2.0)
    two = _child(corpus["raw_plain"], value_boxes=[box], snippet_box=box, scale=3.0)
    assert two["dpi"] == 216.0 and abs(two["width"] / 3.0 - one["width"] / 2.0) < 2


def test_widget_appearance_is_painted_like_the_ink_check_sees_it(corpus):
    reply = _child(corpus["widget"], scale=2.0, whole_page=True)
    ref, _ = _reference(corpus["widget"], 2.0)
    assert np.array_equal(_decode(reply), ref)


# --------------------------------------------------------------------- caps ---

def _page_pdf(tmp_path: Path, side: int) -> Path:
    p = tmp_path / f"big_{side}.pdf"
    p.write_bytes(raw.build([([0, 0, side, side], raw.text(["Cmax was 3,412 ng/mL"] * 3, x=170, y0=500))]))
    return p


def test_big_page_lowers_the_scale_to_the_pixel_budget(tmp_path):
    reply = _child(_page_pdf(tmp_path, 2000), value_boxes=[[170, 500, 200, 510]], snippet_box=[170, 500, 300, 510])
    assert 1.0 <= reply["scale"] < 1.3 and reply["width"] * reply["height"] <= render_child.MAX_OUT_PIXELS


def test_whole_page_over_the_output_cap_is_refused_below_72_dpi(tmp_path):
    reply = _child(_page_pdf(tmp_path, 2000), whole_page=True, scale=2.0)
    assert reply == {"error": "page_too_large"}


def test_page_over_the_full_render_budget_is_refused(tmp_path):
    assert _child(_page_pdf(tmp_path, 3000), value_boxes=[[170, 500, 200, 510]]) == {"error": "page_too_large"}


def test_png_over_the_byte_cap_is_retried_then_refused(corpus, monkeypatch):
    monkeypatch.setattr(render_child, "PNG_MAX_BYTES", 500)
    assert _child(corpus["raw_plain"], whole_page=True, scale=2.0) == {"error": "png_too_large"}


def test_too_many_boxes_are_refused(corpus):
    boxes = [[1.0, 1.0, 2.0, 2.0]] * (render_child.MAX_BOXES + 1)
    assert _child(corpus["raw_plain"], value_boxes=boxes) == {"error": "too_many_boxes"}


# ------------------------------------------------------------------ sandbox ---

def _region(doc: dict, word: str = "3,412") -> regions.Region:
    box = tuple(_value_box(doc, word))
    return regions.Region("value", True, None, (box,), box)


def test_render_page_runs_in_a_child_and_returns_a_checked_png(corpus, docs):
    res = render.render_page(corpus["raw_plain"], 1, 1, _region(docs["raw_plain"]))
    assert res.png.startswith(b"\x89PNG") and res.meta["png_bytes"] == len(res.png)
    assert struct.unpack(">II", res.png[16:24]) == (res.meta["width"], res.meta["height"])
    assert res.meta["rects_px"] and res.meta["dpi"] == 144.0


def test_unlocated_region_renders_the_whole_page_at_90_dpi(corpus):
    res = render.render_page(corpus["raw_plain"], 1, 1, regions.Region("page", False, "layout_unavailable"))
    assert (res.meta["width"], res.meta["height"], res.meta["dpi"]) == (765, 990, 90.0)
    assert res.meta["rects_px"] == []


def test_zoom_must_be_one_or_two(corpus, docs):
    with pytest.raises(ValueError, match="zoom"):
        render.render_page(corpus["raw_plain"], 1, 3, _region(docs["raw_plain"]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1e12])
def test_non_finite_boxes_never_reach_the_child(corpus, bad):
    with pytest.raises(render.RenderFailedError):
        render.render_args(1, 1, regions.Region("value", True, None, ((0.0, 0.0, bad, 1.0),), None))


def test_wall_limit_kills_the_child(corpus, docs, monkeypatch):
    monkeypatch.setattr(render, "RENDER_TARGET", f"{TARGETS}:sleep_with_args")
    monkeypatch.setattr(render, "RENDER_WALL_S", 0.6)
    t0 = time.monotonic()
    with pytest.raises(render.RenderFailedError) as exc:
        render.render_page(corpus["raw_plain"], 1, 1, _region(docs["raw_plain"]))
    assert time.monotonic() - t0 < 6 and "killed" not in str(exc.value) and "/" not in str(exc.value)


def test_memory_cap_kills_the_child(corpus, docs, monkeypatch):
    monkeypatch.setattr(render, "RENDER_TARGET", f"{TARGETS}:hog_memory_with_args")
    monkeypatch.setattr(render, "RENDER_MAX_RSS", 120 * 2**20)
    with pytest.raises(render.RenderFailedError):
        render.render_page(corpus["raw_plain"], 1, 1, _region(docs["raw_plain"]))


def test_child_crash_on_a_broken_pdf_is_a_generic_failure(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\nnot a pdf")
    with pytest.raises(render.RenderFailedError) as exc:
        render.render_page(bad, 1, 1, regions.Region("page", False, "x"))
    assert str(bad) not in str(exc.value)


def test_page_beyond_the_document_fails_generically(corpus):
    with pytest.raises(render.RenderFailedError):
        render.render_page(corpus["raw_plain"], 7, 1, regions.Region("page", False, "x"))


def test_child_with_kwargs_is_scrubbed_of_secrets(corpus, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("PHARMAGENT_API_TOKEN", "tok")
    kept = parse_sandbox.run_child(corpus["raw_plain"], target=f"{TARGETS}:report_env", wall_s=20,
                                   max_rss=512 * 2**20, args={"a": 1})
    scrubbed = parse_sandbox.run_child(corpus["raw_plain"], target=f"{TARGETS}:report_env", wall_s=20,
                                       max_rss=512 * 2**20, args={"a": 1}, scrub_env=True)
    assert kept["kwargs"] == {"a": 1} and "ANTHROPIC_API_KEY" in kept["env"]
    assert "ANTHROPIC_API_KEY" not in scrubbed["env"] and "PHARMAGENT_API_TOKEN" not in scrubbed["env"]
    assert "PYTHONPATH" in scrubbed["env"] and "PATH" in scrubbed["env"]


def test_old_targets_still_get_only_the_path(corpus):
    out = parse_sandbox.run_child(corpus["raw_plain"], target=f"{TARGETS}:report_pid", wall_s=20,
                                  max_rss=512 * 2**20)
    assert "pid" in out
    with pytest.raises(parse_sandbox.ParseFailedError):          # args reach the target only when given
        parse_sandbox.run_child(corpus["raw_plain"], target=f"{TARGETS}:report_pid", wall_s=20,
                                max_rss=512 * 2**20, args={"x": 1})


def test_render_slots_are_separate_from_parse_slots(corpus, docs):
    parse_sandbox._slots.acquire("holder-1", wait_s=1)
    parse_sandbox._slots.acquire("holder-2", wait_s=1)
    try:
        res = render.render_page(corpus["raw_plain"], 1, 1, _region(docs["raw_plain"]))
        assert res.png
    finally:
        parse_sandbox._slots.release("holder-1")
        parse_sandbox._slots.release("holder-2")


def test_busy_renderer_refuses_without_taking_an_upload_slot(corpus, docs, monkeypatch):
    monkeypatch.setattr(render, "RENDER_QUEUE_WAIT_S", 0.1)
    held = [render._render_slots.acquire() for _ in range(render.RENDER_SLOTS)]
    try:
        with pytest.raises(render.RenderBusyError) as exc:
            render.render_page(corpus["raw_plain"], 1, 1, _region(docs["raw_plain"]))
        assert isinstance(exc.value, parse_sandbox.ParseBusyError) and exc.value.retry_after == 3
        assert ingest.validate_pdf_file(corpus["raw_plain"], "uploader") == 1      # uploads still validate
    finally:
        for _ in held:
            render._render_slots.release()


def test_at_most_the_slot_count_of_children_run_at_once(corpus, docs, monkeypatch):
    live, peak, lock = [0], [0], threading.Lock()
    real = render.run_child

    def counting(*a, **k):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        try:
            time.sleep(0.2)
            return real(*a, **k)
        finally:
            with lock:
                live[0] -= 1
    monkeypatch.setattr(render, "run_child", counting)
    errs: list[Exception] = []

    def go():
        try:
            render.render_page(corpus["raw_plain"], 1, 1, _region(docs["raw_plain"]))
        except Exception as exc:                               # noqa: BLE001
            errs.append(exc)
    threads = [threading.Thread(target=go) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errs and peak[0] <= render.RENDER_SLOTS
