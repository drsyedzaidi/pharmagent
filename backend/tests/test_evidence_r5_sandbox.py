"""parse_sandbox: the child process, its limits, its answer, and what the parent releases afterwards.

The child half (_child, _self_watchdog, _child_main) runs in another interpreter in production, where a
line tracer cannot see it, so it is also driven in-process here with its seams faked (the watchdog
thread, os._exit, sys.argv). The parent half is driven with a fake process object so a hung child, a
memory hog or a vanished process is simulated without sleeping for the real limits; the integration tests
at the end use real children.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evidence import parse_sandbox as PS
from app.evidence.parse_guard import ParseLimitError
from tests.evidence_fakeproc import FakeProc

TARGETS = "tests.evidence_sandbox_targets"


# ---------------------------------------------------------------- child side ---

def test_own_peak_rss_is_bytes_on_both_platforms(monkeypatch):
    import resource
    monkeypatch.setattr(resource, "getrusage", lambda _who: SimpleNamespace(ru_maxrss=2048))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert PS._own_peak_rss() == 2048                    # macOS reports bytes
    monkeypatch.setattr(sys, "platform", "linux")
    assert PS._own_peak_rss() == 2048 * 1024             # Linux reports kilobytes


def test_self_watchdog_exits_with_the_memory_code_once_over_the_cap(monkeypatch):
    readings = iter([10, 20, 500])                        # under, under, over the 100-byte cap
    monkeypatch.setattr(PS, "_own_peak_rss", lambda: next(readings))
    monkeypatch.setattr(PS, "POLL_S", 0.0)
    exits: list[int] = []

    class Stop(Exception):
        pass

    def fake_exit(code: int) -> None:
        exits.append(code)
        raise Stop

    monkeypatch.setattr(PS.os, "_exit", fake_exit)
    with pytest.raises(Stop):
        PS._self_watchdog(100)
    assert exits == [PS._MEMORY_EXIT]


def _run_child_inline(monkeypatch, tmp_path, target: str, args_json: str | None = None, path: str = "p.pdf") -> dict:
    started = []
    monkeypatch.setattr(PS.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: started.append(kw)))
    out = tmp_path / "out.json"
    PS._child(target, path, str(out), 2**30, args_json)
    assert started and started[0]["daemon"] is True and started[0]["args"] == (2**30,)   # watchdog started
    return json.loads(out.read_text("utf-8"))


def test_child_writes_the_targets_result(monkeypatch, tmp_path):
    res = _run_child_inline(monkeypatch, tmp_path, f"{TARGETS}:echo_args", '{"a": 1}', "x.pdf")
    assert res == {"ok": {"path": "x.pdf", "kwargs": {"a": 1}}}


def test_child_without_arguments_calls_the_target_with_the_path_only(monkeypatch, tmp_path):
    res = _run_child_inline(monkeypatch, tmp_path, f"{TARGETS}:report_pid")
    assert res == {"ok": {"pid": os.getpid()}}


@pytest.mark.parametrize("target,expected", [
    ("raise_limit", {"type": "ParseLimitError", "limit": True, "missing": False}),
    ("raise_missing", {"type": "FileNotFoundError", "limit": False, "missing": True}),
    ("raise_runtime", {"type": "RuntimeError", "limit": False, "missing": False}),
    ("raise_system_exit", {"type": "SystemExit", "limit": False, "missing": False}),
])
def test_child_reports_every_exception_type_instead_of_losing_it(monkeypatch, tmp_path, target, expected):
    err = _run_child_inline(monkeypatch, tmp_path, f"{TARGETS}:{target}")["error"]
    assert {k: err[k] for k in expected} == expected


def test_child_message_is_cut_to_500_characters(monkeypatch, tmp_path):
    assert len(_run_child_inline(monkeypatch, tmp_path, f"{TARGETS}:raise_runtime")["error"]["message"]) == 500


def test_child_reports_an_unimportable_target(monkeypatch, tmp_path):
    err = _run_child_inline(monkeypatch, tmp_path, "tests.no_such_module_r5:f")["error"]
    assert err["type"] == "ModuleNotFoundError"


def test_child_main_reads_its_arguments_from_argv(monkeypatch):
    calls = []
    monkeypatch.setattr(PS, "_child", lambda *a: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["-c", "mod:fn", "/p.pdf", "/out.json", "12345"])
    PS._child_main()
    monkeypatch.setattr(sys, "argv", ["-c", "mod:fn", "/p.pdf", "/out.json", "12345", '{"zoom": 2}'])
    PS._child_main()
    assert calls == [("mod:fn", "/p.pdf", "/out.json", 12345, None), ("mod:fn", "/p.pdf", "/out.json", 12345, '{"zoom": 2}')]


# --------------------------------------------------------------- starting it ---

def test_frozen_build_starts_a_spawn_process_not_a_python_dash_c(monkeypatch):
    made = {}

    class Ctx:
        def Process(self, **kw):
            made.update(kw)
            return SimpleNamespace(start=lambda: made.setdefault("started", True))

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(PS.multiprocessing, "get_context", lambda how: (made.setdefault("how", how), Ctx())[1])
    proc = PS._start(("t:f", "/p", "/o", 1), True)
    assert made["how"] == "spawn" and made["target"] is PS._child and made["daemon"] is True
    assert made["args"] == ("t:f", "/p", "/o", 1) and made["started"] and proc is not None


def test_interpreter_child_environment_is_scrubbed_only_on_request(monkeypatch):
    seen = []
    monkeypatch.setattr(PS.subprocess, "Popen", lambda cmd, env, stdin: (seen.append(env), SimpleNamespace(pid=7))[1])
    monkeypatch.setenv("EVIDENCE_R5_SECRET", "hunter2")
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    PS._Interpreter(("t:f", "/p", "/o", 1), scrub_env=True)
    PS._Interpreter(("t:f", "/p", "/o", 1))
    scrubbed, full = seen
    assert "EVIDENCE_R5_SECRET" not in scrubbed and full["EVIDENCE_R5_SECRET"] == "hunter2"
    assert set(scrubbed) <= PS._SAFE_ENV | {"PYTHONPATH"}
    backend_root = str(Path(PS.__file__).resolve().parents[2])
    assert scrubbed["PYTHONPATH"].split(os.pathsep) == [backend_root, "/elsewhere"]


# ------------------------------------------------------------- reading RSS ---

def _fake_psutil(monkeypatch, *, rss: int | None = None, error: Exception | None = None) -> None:
    class Proc:
        def __init__(self, pid):
            if error:
                raise error

        def memory_info(self):
            return SimpleNamespace(rss=rss)

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=Proc))


def test_rss_comes_from_psutil_when_installed(monkeypatch):
    _fake_psutil(monkeypatch, rss=123456)
    assert PS._rss_bytes(1) == 123456


def test_rss_is_none_when_psutil_cannot_read_the_process(monkeypatch):
    _fake_psutil(monkeypatch, error=RuntimeError("no such process"))
    assert PS._rss_bytes(1) is None


def test_rss_falls_back_to_ps_without_psutil_and_reads_the_real_process(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)                    # import psutil -> ImportError
    if Path("/proc").is_dir():
        pytest.skip("Linux reads /proc/<pid>/statm instead; covered by the statm test")
    assert PS._rss_bytes(os.getpid()) > 1_000_000


def test_rss_reads_proc_statm_when_it_exists(monkeypatch, tmp_path):
    statm = tmp_path / "statm"
    statm.write_text("100 25 3 0 0 0 0")
    monkeypatch.setitem(sys.modules, "psutil", None)
    real = PS.Path
    monkeypatch.setattr(PS, "Path", lambda p: statm if str(p).startswith("/proc/") else real(p))
    monkeypatch.setattr(PS.os, "sysconf", lambda _name: 4096)
    assert PS._rss_bytes(99) == 25 * 4096


@pytest.mark.parametrize("outcome", [
    subprocess.CompletedProcess([], 0, stdout="2048\n"),               # ps answers: KiB
    subprocess.CompletedProcess([], 0, stdout=""),                     # ps knows no such pid
    subprocess.CompletedProcess([], 0, stdout="n/a"),                  # not a number
    OSError("no ps"),
    subprocess.TimeoutExpired("ps", 2),
])
def test_rss_via_ps_never_raises(monkeypatch, outcome):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(PS.Path, "exists", lambda self: False)

    def fake_run(*a, **k):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(PS.subprocess, "run", fake_run)
    got = PS._rss_bytes(5)
    assert got == (2048 * 1024 if isinstance(outcome, subprocess.CompletedProcess) and outcome.stdout.strip() == "2048"
                   else None)


# ----------------------------------------------------------------- watching ---

def test_watch_returns_when_the_child_exits_normally(monkeypatch):
    monkeypatch.setattr(PS, "_rss_bytes", lambda pid: 10)
    proc = FakeProc(lifetime=0.05)
    PS._watch(proc, 30.0, 2**30)
    assert not proc.killed


def test_watch_kills_a_child_past_the_wall_clock(monkeypatch):
    monkeypatch.setattr(PS, "_rss_bytes", lambda pid: 10)
    proc = FakeProc()
    with pytest.raises(PS.ParseAbortedError, match="exceeded the 0 s time limit"):
        PS._watch(proc, 0.0, 2**30)
    assert proc.killed and 5 in proc.joins                               # killed, then reaped with a 5 s join


def test_watch_kills_a_child_the_parent_sees_above_the_memory_cap(monkeypatch):
    monkeypatch.setattr(PS, "_rss_bytes", lambda pid: 300 * 2**20)
    proc = FakeProc()
    with pytest.raises(PS.ParseAbortedError, match="memory above 100 MB"):
        PS._watch(proc, 30.0, 100 * 2**20)
    assert proc.killed


def test_watch_does_not_kill_when_rss_is_unreadable(monkeypatch):
    monkeypatch.setattr(PS, "_rss_bytes", lambda pid: None)             # only the wall clock applies
    proc = FakeProc(lifetime=0.05)
    PS._watch(proc, 30.0, 1)
    assert not proc.killed


def test_watch_polls_rss_at_its_own_interval_not_every_loop(monkeypatch):
    reads = []
    monkeypatch.setattr(PS, "_rss_bytes", lambda pid: reads.append(pid) or 1)
    monkeypatch.setattr(PS, "RSS_POLL_S", 10.0)
    PS._watch(FakeProc(lifetime=0.1), 30.0, 2**30)
    assert reads == [4242]                                              # once at the start, then not again within 10 s


def test_watch_turns_the_childs_own_memory_exit_into_an_abort(monkeypatch):
    monkeypatch.setattr(PS, "_rss_bytes", lambda pid: 1)
    with pytest.raises(PS.ParseAbortedError, match="memory above 1 MB"):
        PS._watch(FakeProc(lifetime=0.02, exitcode=PS._MEMORY_EXIT), 30.0, 2**20)


# ---------------------------------------------------------------- answering ---

def _answer(tmp_path, payload: str | None, code: int | None = 0):
    out = tmp_path / "result.json"
    if payload is not None:
        out.write_text(payload, "utf-8")
    return PS._answer(out, code)


def test_answer_returns_the_ok_value(tmp_path):
    assert _answer(tmp_path, json.dumps({"ok": {"pages": []}})) == {"pages": []}


@pytest.mark.parametrize("payload", [None, "", "{not json", "\x00\xff"])
def test_answer_without_a_readable_result_is_an_abort_naming_the_exit_code(tmp_path, payload):
    with pytest.raises(PS.ParseAbortedError, match=r"died without an answer \(exit code -9\)"):
        _answer(tmp_path, payload, -9)


def test_answer_maps_each_error_kind_to_its_exception(tmp_path):
    with pytest.raises(ParseLimitError, match="glyph budget"):
        _answer(tmp_path, json.dumps({"error": {"limit": True, "message": "glyph budget"}}))
    with pytest.raises(FileNotFoundError, match="gone.pdf"):
        _answer(tmp_path, json.dumps({"error": {"missing": True, "message": "gone.pdf"}}))
    with pytest.raises(PS.ParseFailedError, match="^KeyError: 'k'$"):
        _answer(tmp_path, json.dumps({"error": {"type": "KeyError", "message": "'k'"}}))
    with pytest.raises(PS.ParseFailedError, match="^Error: $"):          # a result with neither ok nor an error
        _answer(tmp_path, json.dumps({}))
