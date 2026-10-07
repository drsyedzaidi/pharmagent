#!/usr/bin/env python3
"""Trace-based statement coverage for app/evidence (stdlib ``trace``; no pytest-cov needed).

Runs the evidence test files (or the pytest args given) in-process with a line tracer installed
on the main thread AND on every new thread (FastAPI runs sync endpoints and tools in worker
threads), and prints executed/total executable lines per ``app/evidence/*.py`` module, the totals
coming from the stdlib ``trace`` line finder. ``trace.Trace`` itself is not used: it traces the
main thread only and its ignore cache is keyed by bare module name, so ignoring a venv
``models.py`` silently hid ``app/evidence/models.py``. Exit status is 1 when the tests fail or any
module is below ``--min`` (default 80 %).

    cd backend && .venv/bin/python scripts/evidence_coverage.py [--min 80] [pytest args...]

A full traced run is slow; ``--data FILE`` merges the lines hit by this run into FILE (a JSON map of
file -> lines) and reports on the union, so the suite can run in chunks (each chunk's exit status only
reflects its own tests; the gate is the last chunk's report):

    scripts/evidence_coverage.py --data /tmp/cov.json --min 0 tests/test_evidence_a*.py
    scripts/evidence_coverage.py --data /tmp/cov.json tests/test_evidence_z*.py
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import trace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "app" / "evidence"
sys.path.insert(0, str(ROOT))


def _executable_lines(path: Path) -> set[int]:
    # stdlib helper used by trace.CoverageResults.write_results (private but stable since 2.x)
    return set(trace._find_executable_linenos(str(path)))  # noqa: SLF001


def _collect(pytest_args: list[str]) -> tuple[int, dict[Path, set[int]]]:
    """Run pytest with a tracer that records line events only in app/evidence/*.py frames."""
    import pytest

    targets = {str(p.resolve()) for p in EVIDENCE.rglob("*.py")} | {str(p) for p in EVIDENCE.rglob("*.py")}
    hit: dict[str, set[int]] = {}

    def local(frame, event, _arg):
        if event == "line":
            hit.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
        return local

    def global_(frame, event, _arg):
        if event == "call" and frame.f_code.co_filename in targets:
            hit.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
            return local
        return None

    threading.settrace(global_)
    sys.settrace(global_)
    try:
        rc = pytest.main(["-q", "-p", "no:cacheprovider", *pytest_args])
    finally:
        sys.settrace(None)
        threading.settrace(None)  # type: ignore[arg-type]
    merged: dict[Path, set[int]] = {}
    for filename, lines in hit.items():
        merged.setdefault(Path(filename).resolve(), set()).update(lines)
    return int(rc), merged


def _merge_data(hit: dict[Path, set[int]], data: Path | None) -> dict[Path, set[int]]:
    """Union of *hit* and the lines stored in *data*, which is rewritten with the union."""
    if data is None:
        return hit
    merged = {k: set(v) for k, v in hit.items()}
    if data.exists():
        for name, lines in json.loads(data.read_text("utf-8")).items():
            merged.setdefault(Path(name), set()).update(lines)
    data.write_text(json.dumps({str(k): sorted(v) for k, v in merged.items()}), "utf-8")
    return merged


def _missing(lines: set[int]) -> str:
    """Compact ``a-b, c`` ranges of the executable lines that no test reached."""
    out, run = [], []
    for n in sorted(x for x in lines if x is not None):   # the line finder lists one None per module
        if run and n != run[-1] + 1:
            out.append(run)
            run = []
        run.append(n)
    out.extend([run] if run else [])
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in out)


def run(pytest_args: list[str], minimum: float, data: Path | None = None) -> int:
    rc, hit = _collect(pytest_args)
    hit = _merge_data(hit, data)
    rows, below = [], []
    for mod in sorted(EVIDENCE.rglob("*.py")):                 # sub-packages (numgrammar/) are modules too
        total = _executable_lines(mod)
        done = hit.get(mod.resolve(), set()) & total
        pct = 100.0 * len(done) / len(total) if total else 100.0
        name = mod.relative_to(EVIDENCE).as_posix()
        rows.append((name, len(done), len(total), pct))
        if pct < minimum:
            below.append(name)
            print(f"{name} lines not executed: {_missing(total - done)}")
    width = max(len(r[0]) for r in rows)
    print(f"\n{'module':<{width}}  executed/total  statements")
    for name, done, total, pct in rows:
        print(f"{name:<{width}}  {done:>5}/{total:<5}     {pct:5.1f}%")
    all_done, all_total = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"{'ALL':<{width}}  {all_done:>5}/{all_total:<5}     {100.0 * all_done / all_total:5.1f}%")
    print("(trace-based executable-line count; not branch coverage)")
    if below:
        print(f"BELOW {minimum:.0f}%: {', '.join(below)}")
    return 1 if (rc or below) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min", type=float, default=80.0, help="minimum statement coverage per module")
    ap.add_argument("--data", type=Path, help="JSON file of hit lines merged across chunked runs")
    ap.add_argument("pytest_args", nargs="*", help="pytest arguments (default: tests/test_evidence_*.py)")
    ns = ap.parse_args(argv)
    args = ns.pytest_args or sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_evidence_*.py"))
    return run(args, ns.min, ns.data)


if __name__ == "__main__":
    sys.exit(main())
