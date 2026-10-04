"""parse_sandbox, parent side: scratch cleanup, kill on failure, slot release, and real child processes."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.evidence import parse_sandbox as PS
from app.evidence.parse_guard import ParseLimitError
from app.evidence.parse_queue import FairSlots, ParseBusyError
from tests.evidence_fakeproc import FakeProc

TARGETS = "tests.evidence_sandbox_targets"


# ---------------------------------------------- parent: slots, scratch, kill ---

def test_run_child_always_removes_its_scratch_directory(monkeypatch, tmp_path):
    made = []
    real = PS.tempfile.mkdtemp
    monkeypatch.setattr(PS.tempfile, "mkdtemp", lambda **kw: made.append(real(dir=tmp_path, **kw)) or made[-1])
    PS.run_child(tmp_path / "a.pdf", target=f"{TARGETS}:report_pid", wall_s=20, max_rss=2**30)
    with pytest.raises(PS.ParseFailedError):
        PS.run_child(tmp_path / "a.pdf", target=f"{TARGETS}:raise_runtime", wall_s=20, max_rss=2**30)
    assert len(made) == 2 and not any(Path(p).exists() for p in made)


def test_a_watch_that_blows_up_still_kills_the_child_and_cleans_up(monkeypatch, tmp_path):
    proc = FakeProc()
    monkeypatch.setattr(PS, "_start", lambda argv, scrub=False: proc)

    def explode(*_a):
        raise KeyboardInterrupt

    monkeypatch.setattr(PS, "_watch", explode)
    with pytest.raises(KeyboardInterrupt):
        PS.run_child(tmp_path / "a.pdf", target="x:y", wall_s=1, max_rss=1)
    assert proc.killed


def test_run_child_passes_args_and_scrub_flag_to_the_child(monkeypatch, tmp_path):
    seen = {}

    def fake_start(argv, scrub=False):
        seen.update(argv=argv, scrub=scrub)
        Path(argv[2]).write_text(json.dumps({"ok": 1}), "utf-8")
        return FakeProc(lifetime=0.0)

    monkeypatch.setattr(PS, "_start", fake_start)
    monkeypatch.setattr(PS, "_rss_bytes", lambda pid: 1)
    assert PS.run_child(tmp_path / "a.pdf", target="x:y", wall_s=5, max_rss=77, args={"zoom": 2}, scrub_env=True) == 1
    assert seen["scrub"] is True and seen["argv"][0] == "x:y" and seen["argv"][3] == 77
    assert json.loads(seen["argv"][4]) == {"zoom": 2}
    PS.run_child(tmp_path / "a.pdf", target="x:y", wall_s=5, max_rss=77)
    assert seen["scrub"] is False and len(seen["argv"]) == 4


def _one_slot(monkeypatch) -> FairSlots:
    slots = FairSlots(capacity=1, max_queue=2, max_queued_per_session=1)
    monkeypatch.setattr(PS, "_slots", slots)
    monkeypatch.setattr(PS, "QUEUE_WAIT_S", 0.3)
    return slots


@pytest.mark.parametrize("target,error", [
    ("raise_runtime", PS.ParseFailedError), ("raise_limit", ParseLimitError), ("die_without_answer", PS.ParseAbortedError),
])
def test_the_parse_slot_is_released_when_the_child_fails(monkeypatch, tmp_path, target, error):
    slots = _one_slot(monkeypatch)
    with pytest.raises(error):
        PS.run_isolated(tmp_path / "a.pdf", target=f"{TARGETS}:{target}", wall_s=20, max_rss=2**30, session_id="s1")
    slots.acquire("other", wait_s=0.2)                                  # would raise ParseBusyError if still held
    slots.release("other")
    assert slots.queued == 0


def test_the_slot_is_released_when_the_child_is_killed_at_the_wall_clock(monkeypatch, tmp_path):
    slots = _one_slot(monkeypatch)
    with pytest.raises(PS.ParseAbortedError, match="time limit"):
        PS.run_isolated(tmp_path / "a.pdf", target=f"{TARGETS}:sleep_forever", wall_s=0.5, max_rss=2**30,
                        session_id="s1")
    slots.acquire("other", wait_s=0.2)
    slots.release("other")


def test_a_second_session_waits_then_is_refused_busy_while_the_slot_is_held(monkeypatch, tmp_path):
    slots = _one_slot(monkeypatch)
    slots.acquire("holder", wait_s=1)
    try:
        with pytest.raises(ParseBusyError):
            PS.run_isolated(tmp_path / "a.pdf", target=f"{TARGETS}:report_pid", wall_s=20, max_rss=2**30,
                            session_id="s2")
    finally:
        slots.release("holder")
    assert PS.run_isolated(tmp_path / "a.pdf", target=f"{TARGETS}:report_pid", wall_s=20, max_rss=2**30,
                           session_id="s2")["pid"] != os.getpid()


# --------------------------------------------------------- real child process ---

def test_the_killed_child_process_is_really_gone(tmp_path):
    started = []
    real_start = PS._start
    PS._start = lambda argv, scrub=False: started.append(real_start(argv, scrub)) or started[-1]
    try:
        with pytest.raises(PS.ParseAbortedError, match="time limit"):
            PS.run_child(tmp_path / "a.pdf", target=f"{TARGETS}:sleep_with_args", wall_s=0.5, max_rss=2**30,
                         args={"k": 1})
    finally:
        PS._start = real_start
    assert started and not started[0].is_alive() and started[0].exitcode is not None


def test_real_child_environment_is_scrubbed_on_request_and_not_otherwise(monkeypatch, tmp_path):
    monkeypatch.setenv("EVIDENCE_R5_SECRET", "hunter2")
    kw = {"target": f"{TARGETS}:report_env", "wall_s": 20, "max_rss": 2**30, "args": {"note": "x"}}
    full = PS.run_child(tmp_path / "a.pdf", **kw)
    scrubbed = PS.run_child(tmp_path / "a.pdf", scrub_env=True, **kw)
    assert "EVIDENCE_R5_SECRET" in full["env"] and full["kwargs"] == {"note": "x"}
    assert "EVIDENCE_R5_SECRET" not in scrubbed["env"]


def test_real_child_reports_a_missing_pdf_as_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        PS.parse_isolated(tmp_path / "does-not-exist.pdf")


def test_real_child_that_dies_silently_is_an_abort(tmp_path):
    with pytest.raises(PS.ParseAbortedError, match=r"died without an answer \(exit code 3\)"):
        PS.run_child(tmp_path / "a.pdf", target=f"{TARGETS}:die_without_answer", wall_s=20, max_rss=2**30)


def test_real_child_memory_cap_is_enforced_by_the_parent_when_the_child_watchdog_is_slow(monkeypatch, tmp_path):
    monkeypatch.setattr(PS, "_rss_bytes", lambda pid: 10 * 2**30)       # the parent's reading says: 10 GiB
    with pytest.raises(PS.ParseAbortedError, match="memory above 64 MB"):
        PS.run_child(tmp_path / "a.pdf", target=f"{TARGETS}:sleep_forever", wall_s=30, max_rss=64 * 2**20)
