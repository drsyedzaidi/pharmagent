"""Run provenance: software versions, platform, and content hashes.

Captured into the audit chain at session creation and stamped into reports /
define.xml so any result can be traced to the exact software state that produced
it — a basic reproducibility / data-integrity (ALCOA+) requirement.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGES = ("numpy", "scipy", "pandas", "fastapi", "pydantic", "python-docx")


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "n/a"


@lru_cache(maxsize=1)
def _git_sha() -> str:
    """Short git SHA of the working tree, or 'n/a' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=2, check=False)
        sha = out.stdout.strip()
        return sha or "n/a"
    except Exception:
        return "n/a"


@lru_cache(maxsize=1)
def _collect(app_version: str) -> tuple[tuple[str, str], ...]:
    from app.conventions.identity import catalog_identity  # local: keep core import graph light

    prov = {
        "app_version": app_version,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in _PACKAGES:
        prov[pkg] = _pkg_version(pkg)
    prov.update(catalog_identity())        # conventions catalog version + canonical SHA-256
    return tuple(prov.items())


def collect_provenance(app_version: str = "0.1.0") -> dict[str, str]:
    """Software/platform fingerprint of this run, including the conventions catalog identity. Constant per process;
    a fresh dict each call, so a caller cannot alter the cached value."""
    return dict(_collect(app_version))


def verify_run_provenance(record: tuple[tuple[str, str], ...] | None, genesis) -> str:
    """Check a session's stored run provenance against its genesis audit entry.

    ``legacy``: no stored record (session created before run provenance was kept in state).
    ``verified``: the record hashes to the genesis entry's inputs_hash. ``mismatch``: anything else, including a
    record checked against an entry that is not the session's genesis entry."""
    from app.core.audit import hash_payload  # local: keep core import graph light

    if record is None:
        return "legacy"
    if getattr(genesis, "index", None) != 0 or getattr(genesis, "action", None) != "session_created":
        return "mismatch"
    return "verified" if hash_payload(dict(record)) == genesis.inputs_hash else "mismatch"


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes (for dataset integrity), or 'n/a' if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "n/a"
