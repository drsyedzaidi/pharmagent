"""scripts/evidence_coverage.py reports sub-packages of app/evidence (numgrammar/) as modules, by relative path."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evidence_coverage.py"
spec = importlib.util.spec_from_file_location("evidence_coverage_script_r6", SCRIPT)
cov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cov)


def test_the_report_lists_the_numgrammar_modules_by_relative_path_and_gates_them(monkeypatch, capsys):
    tokenizer = (cov.EVIDENCE / "numgrammar" / "tokenizer.py").resolve()
    monkeypatch.setattr(cov, "_collect", lambda _args: (0, {tokenizer: {1}}))
    status = cov.run([], 85.0)
    out = capsys.readouterr().out
    assert "numgrammar/tokenizer.py" in out and "numgrammar/alphabet.py" in out
    assert "BELOW 85%" in out and "numgrammar/tokenizer.py" in out.split("BELOW 85%")[1]
    assert status == 1


def test_every_numgrammar_module_is_inside_the_traced_set():
    names = {p.relative_to(cov.EVIDENCE).as_posix() for p in cov.EVIDENCE.rglob("*.py")}
    assert {"numgrammar/tokenizer.py", "numgrammar/scan.py", "numgrammar/alphabet.py", "verify.py"} <= names
