"""Confinement of stored file paths to the evidence data directory.

A source row of the ledger carries the absolute paths of its PDF and its parse cache. The ledger is internal,
but a path read from it is still input: before anything hashes, reads, parses or serves such a file it must
resolve (symlinks followed, ``..`` collapsed) to a place inside the evidence directory, so a row that points at
a temp file, another user's file, or a link inside the directory that leads out never gets that file opened.
``confine`` only inspects the path (a resolve reads links, never file contents).
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.evidence.models import EvidenceStateError

log = logging.getLogger(__name__)

__all__ = ["OutsideDataDirError", "confine"]


class OutsideDataDirError(EvidenceStateError, ValueError):
    """A stored file path leaves the evidence directory. A conflict of the ledger with the disk (HTTP 409,
    like a missing or changed file); also a ValueError, so callers that already refuse unreadable sources
    on ValueError refuse this too."""


def confine(path: str | Path, root: str | Path) -> Path:
    """The resolved *path* when it lies inside the resolved *root*, else OutsideDataDirError (and a log
    line). A path that does not exist yet resolves lexically and passes when it would lie inside."""
    try:
        resolved = Path(path).resolve()
        base = Path(root).resolve()
        inside = resolved.is_relative_to(base)
    except (OSError, ValueError, RuntimeError) as exc:        # NUL byte, symlink loop, unreadable link
        log.warning("evidence: stored path %r cannot be resolved (%s); refusing to read it", str(path)[:200], exc)
        raise OutsideDataDirError("source file lies outside the evidence directory") from exc
    if not inside:
        log.warning("evidence: stored path %r resolves to %r, outside the evidence directory %r; refusing to "
                    "read it", str(path)[:200], str(resolved)[:200], str(base))
        raise OutsideDataDirError("source file lies outside the evidence directory")
    return resolved
