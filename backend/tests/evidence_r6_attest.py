"""The attestation of the r6 exhaustive tier: a committed record that both full runs passed on THIS code.

The exhaustive tests (every code point at every boundary of every expression, and every code point through the paths
around the tokenizer) take about 16 minutes on 12 workers and run only with ``R6_EXHAUSTIVE=1``, so nothing in the normal
gate would notice that the code changed after the last full run (the round-2 audit found logs for a commit that were not
at a stable path and were gated by nothing). Now each passing full run on a clean tree writes its summary into
``tests/evidence_r6_attestation.json``, together with a DIGEST of every file the runs execute (``app/evidence``, the
``evidence_r6_*`` support modules, the golden file and the two exhaustive test modules). The normal gate
(``test_evidence_r6_attestation.py``) recomputes the digest and fails when it differs: any change to ``app/evidence`` or to
the oracle needs both full runs again. The commit hash of the run is recorded for people; the digest is the proof (it
needs no git history, so a shallow checkout verifies it too).

    cd backend && R6_EXHAUSTIVE=1 python -m pytest -m exhaustive tests/test_evidence_r6_exhaustive.py tests/test_evidence_r6_scope.py
    git add tests/evidence_r6_attestation.json && git commit

The runs refuse a dirty tree (their log would be evidence of the tree, not of the commit) and write nothing under
``R6_ALLOW_DIRTY=1``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
ATTESTATION = Path(__file__).with_name("evidence_r6_attestation.json")
KINDS = ("tokens", "layer_b")
CODE_POINTS = 0x110000 - 2048                  # every Unicode scalar value (the 2 048 surrogates excluded)


def inputs() -> list[Path]:
    """Every file whose content decides what the two full runs prove."""
    tests = BACKEND / "tests"
    files = [*(BACKEND / "app" / "evidence").rglob("*.py"), *tests.glob("evidence_r6_*.py"),
             tests / "evidence_r6_golden.json", tests / "test_evidence_r6_exhaustive.py",
             tests / "test_evidence_r6_scope.py"]
    return sorted({p for p in files if p.is_file() and "__pycache__" not in p.parts})


def digest() -> str:
    h = hashlib.sha256()
    for path in inputs():
        h.update(path.relative_to(BACKEND).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def load(path: Path | None = None) -> dict[str, Any] | None:
    try:
        data = json.loads((path or ATTESTATION).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def record(kind: str, summary: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Store the summary of one passing full run. A record made for other code is dropped: the file only ever holds
    runs of the code whose digest it carries."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    target, now = path or ATTESTATION, digest()
    data = load(target) or {}
    if data.get("digest") != now:
        data = {}
    data = {**data, "digest": now, kind: summary}
    target.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", "utf-8")
    return data


def git_ancestor(head: str) -> bool | None:
    """True when *head* is HEAD or one of its ancestors, False when it is not, None when git cannot tell (no git, or a
    shallow checkout without that commit)."""
    def git(*args: str) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(["git", *args], cwd=BACKEND, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
    known = git("cat-file", "-e", f"{head}^{{commit}}")
    if known is None or known.returncode != 0:
        return None
    done = git("merge-base", "--is-ancestor", head, "HEAD")
    return None if done is None else done.returncode == 0


def problems(data: dict[str, Any] | None, *, pairs: int, sweeps: tuple[str, ...], now: str | None = None,
             ancestor: Any = git_ancestor) -> list[str]:
    """Why *data* is not proof that both full runs passed on the current code (empty when it is)."""
    if not data:
        return ["there is no attestation: run the two exhaustive tests on a clean commit (see evidence_r6_attest.py)"]
    out: list[str] = []
    if data.get("digest") != (now or digest()):
        out.append("app/evidence or an oracle file changed after the last full run: run both exhaustive tests again")
    tokens_run, layer_b = data.get("tokens") or {}, data.get("layer_b") or {}
    if not tokens_run:
        out.append("the token sweep (test_evidence_r6_exhaustive.py) has no passing full run")
    elif (tokens_run.get("errors"), tokens_run.get("pairs"), tokens_run.get("variants")) != (
            0, pairs, pairs * (CODE_POINTS)):
        out.append(f"the token sweep covered {tokens_run.get('pairs')} pairs / {tokens_run.get('variants')} variants "
                   f"with {tokens_run.get('errors')} errors; {pairs} pairs x {CODE_POINTS} code points and 0 errors needed")
    if not layer_b:
        out.append("the layer-B sweeps (test_evidence_r6_scope.py) have no passing full run")
    elif layer_b.get("errors") != 0 or layer_b.get("code_points") != {name: CODE_POINTS for name in sweeps}:
        out.append(f"the layer-B sweeps covered {layer_b.get('code_points')} with {layer_b.get('errors')} errors; "
                   f"{sorted(sweeps)} x {CODE_POINTS} and 0 errors needed")
    for kind, run in (("tokens", tokens_run), ("layer_b", layer_b)):
        if run and (run.get("dirty") or not run.get("head")):
            out.append(f"the {kind} run was made on a dirty or unknown tree: run it on a clean commit")
    heads = {run.get("head") for run in (tokens_run, layer_b) if run.get("head")}
    if len(heads) > 1:
        out.append(f"the two runs were made at different commits: {sorted(heads)}")
    for head in heads:
        if ancestor(head) is False:
            out.append(f"the run commit {head[:10]} is not an ancestor of HEAD: the attestation belongs to another branch")
    return out
