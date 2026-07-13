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
def collect_provenance(app_version: str = "0.1.0") -> dict[str, str]:
    """Software/platform fingerprint of this run (cached; constant per process)."""
    prov = {
        "app_version": app_version,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in _PACKAGES:
        prov[pkg] = _pkg_version(pkg)
    return dict(prov)


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
