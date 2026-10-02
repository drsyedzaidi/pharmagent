"""Render cache: content-addressed keys, torn-write safety, LRU eviction, single flight, admission."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import pytest
from fastapi import HTTPException

from app.evidence import render, render_cache, render_limits
from app.evidence.regions import Region

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 40


def _meta(png: bytes = PNG) -> dict:
    return {"width": 10, "height": 5, "scale": 2.0, "dpi": 144.0, "crop_px": [0, 0, 10, 5],
            "crop_pt": [0.0, 0.0, 5.0, 2.5], "rects_px": [[1, 1, 2, 2]], "snippet_px": None, "page_px": [10, 5],
            "png_sha256": hashlib.sha256(png).hexdigest(), "png_bytes": len(png)}


def _region(x: float = 100.0) -> Region:
    return Region("value", True, None, ((x, 500.0, x + 30, 510.0),), (x, 500.0, x + 90, 510.0))


def _key(**kw) -> str:
    base = {"sha": "a" * 64, "page": 1, "zoom": 1, "region": _region(), "version": render.render_version()}
    base.update(kw)
    args = render.render_args(base["page"], base["zoom"], base["region"])
    return render_cache.cache_key(base["sha"], args, base["version"])


def test_key_changes_with_pdf_page_zoom_boxes_and_version():
    base = _key()
    assert base == _key() and len(base) == 64
    assert len({base, _key(sha="b" * 64), _key(page=2), _key(zoom=2), _key(region=_region(101.0)),
                _key(version="other"), _key(region=Region("page", False, "layout_unavailable"))}) == 7


def test_key_ignores_sub_tenth_point_noise_and_the_reason_text():
    assert _key(region=_region(100.0)) == _key(region=_region(100.04))
    assert _key(region=Region("page", False, "a")) == _key(region=Region("page", False, "b"))


def test_put_then_get_round_trips_and_touches_recency(tmp_path):
    render_cache.put("k1", PNG, _meta(), tmp_path)
    old = time.time() - 1000
    for ext in ("png", "json"):
        os.utime(render_cache.render_dir(tmp_path) / f"k1.{ext}", (old, old))
    png, meta = render_cache.get("k1", tmp_path)
    assert png == PNG and meta["width"] == 10 and meta["rects_px"] == [[1, 1, 2, 2]]
    assert (render_cache.render_dir(tmp_path) / "k1.png").stat().st_mtime > old + 500


def test_missing_entry_is_a_miss(tmp_path):
    assert render_cache.get("nope", tmp_path) is None


@pytest.mark.parametrize("damage", ["png_only", "json_only", "png_changed", "json_garbage", "size_lie"])
def test_torn_or_damaged_entries_are_misses_and_are_deleted(tmp_path, damage):
    render_cache.put("k", PNG, _meta(), tmp_path)
    d = render_cache.render_dir(tmp_path)
    if damage == "png_only":
        (d / "k.json").unlink()
    elif damage == "json_only":
        (d / "k.png").unlink()
    elif damage == "png_changed":
        (d / "k.png").write_bytes(PNG[:-1] + b"y")
    elif damage == "json_garbage":
        (d / "k.json").write_text("{not json")
    else:
        (d / "k.json").write_text(json.dumps({**_meta(), "png_bytes": 1}))
    assert render_cache.get("k", tmp_path) is None
    assert not (d / "k.png").exists() and not (d / "k.json").exists()


def _fill(tmp_path, n: int, size: int = 100, *, start: float = 0.0) -> list[str]:
    keys = []
    for i in range(n):
        png = PNG + bytes(size)
        render_cache.put(f"key{i:04d}", png, _meta(png), tmp_path)
        d = render_cache.render_dir(tmp_path)
        t = start + i
        for ext in ("png", "json"):
            os.utime(d / f"key{i:04d}.{ext}", (t, t))
        keys.append(f"key{i:04d}")
    return keys


def test_entry_cap_evicts_least_recently_used_down_to_eighty_percent(tmp_path, monkeypatch):
    monkeypatch.setattr(render_cache, "MAX_ENTRIES", 10)
    monkeypatch.setattr(render_cache, "MAX_BYTES", 10**9)
    keys = _fill(tmp_path, 10, start=1000.0)
    render_cache.get(keys[0], tmp_path)                       # the oldest is used: it is now the newest
    png = PNG + bytes(100)
    render_cache.put("newest", png, _meta(png), tmp_path)     # 11 entries: over the cap
    left = {p.stem for p in render_cache.render_dir(tmp_path).glob("*.png")}
    assert len(left) <= 8 and keys[0] in left and "newest" in left
    assert keys[1] not in left and keys[2] not in left


def test_byte_cap_evicts_down_to_eighty_percent(tmp_path, monkeypatch):
    one = len(PNG) + 5000 + len(json.dumps({k: v for k, v in _meta(PNG + bytes(5000)).items()}))
    monkeypatch.setattr(render_cache, "MAX_BYTES", one * 5)
    monkeypatch.setattr(render_cache, "MAX_ENTRIES", 10**6)
    _fill(tmp_path, 8, size=5000, start=1000.0)
    total = sum(p.stat().st_size for p in render_cache.render_dir(tmp_path).iterdir())
    assert total <= one * 5 * 1.01


def test_eviction_deletes_orphans_too(tmp_path, monkeypatch):
    monkeypatch.setattr(render_cache, "MAX_ENTRIES", 3)
    d = render_cache.render_dir(tmp_path)
    (d / "orphan.png").write_bytes(PNG)
    os.utime(d / "orphan.png", (1.0, 1.0))
    _fill(tmp_path, 4, start=1000.0)
    assert not (d / "orphan.png").exists()


def test_single_flight_runs_one_render_for_sixteen_threads(tmp_path):
    calls = [0]

    def work():
        with render_cache.single_flight("same"):
            if render_cache.get("same", tmp_path) is None:
                calls[0] += 1
                time.sleep(0.05)
                render_cache.put("same", PNG, _meta(), tmp_path)
    threads = [threading.Thread(target=work) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert calls[0] == 1 and not render_cache._flights


def test_single_flight_keys_do_not_block_each_other():
    order = []

    def a():
        with render_cache.single_flight("a"):
            order.append("a-in")
            time.sleep(0.15)

    t = threading.Thread(target=a)
    t.start()
    time.sleep(0.03)
    with render_cache.single_flight("b"):
        order.append("b-in")
    t.join()
    assert order == ["a-in", "b-in"]


# ---------------------------------------------------------------- admission ---

def test_sixty_first_uncached_render_in_a_window_is_429_with_retry_after():
    render_limits.reset()
    for _ in range(render_limits.RENDERS_PER_WINDOW):
        render_limits.admit("s")
    with pytest.raises(HTTPException) as exc:
        render_limits.admit("s")
    assert exc.value.status_code == 429 and int(exc.value.headers["Retry-After"]) >= 1
    render_limits.admit("other-session")                       # per session
    render_limits.reset()
    render_limits.admit("s")


def test_in_flight_cap_refuses_at_once_and_releases():
    render_limits.reset()
    held = []
    ctxs = [render_limits.in_flight() for _ in range(render_limits.MAX_RENDERS_IN_FLIGHT)]
    for c in ctxs:
        c.__enter__()
        held.append(c)
    with pytest.raises(HTTPException) as exc:
        with render_limits.in_flight():
            pass
    assert exc.value.status_code == 503 and exc.value.headers["Retry-After"]
    held.pop().__exit__(None, None, None)
    with render_limits.in_flight():
        pass
    for c in held:
        c.__exit__(None, None, None)
    render_limits.reset()
