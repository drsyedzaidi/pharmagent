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


RUN_PROVENANCE_SCHEMA = 1
QUARANTINED_STATUS = "mismatch"


class ProvenanceQuarantined(Exception):
    """The session's run provenance does not match its genesis audit entry. The session is read-only: tools, state
    writes and trusted exports are refused; audit inspection and raw state stay available."""


def genesis_outputs(session_id: str) -> dict:
    """Outputs of a new session's genesis audit entry: the session id plus a marker that a run-provenance record
    was written. The marker is what makes a deleted record detectable (it can no longer pass as legacy)."""
    return {"session_id": session_id, "run_provenance_schema": RUN_PROVENANCE_SCHEMA}


def verify_run_provenance(record: tuple[tuple[str, str], ...] | None, chain, session_id: str) -> str:
    """Check a session's stored run provenance against its audit chain.

    The whole chain must verify first (every entry recomputes and every link holds); field values alone prove nothing,
    because they can be edited without recomputing hashes. Then the genesis entry decides:
    ``verified``: it carries the provenance marker and the record (no repeated keys) hashes to its inputs_hash.
    ``legacy_unverified``: a chain-valid genesis written before records were kept (outputs = {"session_id"}) and no
    record. ``mismatch``: anything else.

    In hash-only mode a complete, consistent rebuild of the chain cannot be told from the original: that is why the
    assurance level is reported separately (hash_only vs keyed_seal)."""
    from app.core.audit import hash_payload  # local: keep core import graph light

    entries = getattr(chain, "entries", None)
    status = chain.verify_status() if entries else {}
    if not entries or not status.get("ok"):
        return "mismatch"
    genesis = entries[0]
    if genesis.index != 0 or genesis.action != "session_created":
        return "mismatch"
    if record is not None and len({k for k, _ in record}) != len(record):
        return "mismatch"                                        # a repeated key could hide a forged value
    if genesis.outputs_hash == hash_payload(genesis_outputs(session_id)):
        if record is None or status.get("degraded"):
            # New sessions always write v2 entries; a marker genesis in a degraded (v1: index/action not bound)
            # chain can only be a forgery.
            return "mismatch"
        return "verified" if hash_payload(dict(record)) == genesis.inputs_hash else "mismatch"
    if genesis.outputs_hash == hash_payload({"session_id": session_id}):
        return "legacy_unverified" if record is None else "mismatch"
    return "mismatch"


def _assurance_text(state) -> str:
    level = getattr(state, "run_provenance_assurance", None)
    if level == "keyed_seal":
        return "keyed_seal: audit chain authenticated by a keyed seal on every restore, sealed since creation"
    if level == "keyed_seal_attested_legacy":
        return ("keyed_seal_attested_legacy: sealed from an operator-attested migration baseline, not since creation; "
                "entries before the baseline are attested, not authenticated")
    if level == "keyed_seal_unconfirmed":
        return "keyed_seal_unconfirmed: keyed mode is on but this session's seal baseline is unknown"
    return ("hash_only: audit chain internally consistent, not cryptographically authenticated; a consistent "
            "rebuild or removal of trailing entries is not detectable in this mode")


def run_provenance_report_lines(state) -> list[str]:
    """Report text for the run provenance RECORDED at session creation. Never reads the current server's provenance.

    Raises ProvenanceQuarantined for a mismatched record, so no report or export is produced from it."""
    status = getattr(state, "run_provenance_status", None)
    if status == QUARANTINED_STATUS:
        raise ProvenanceQuarantined("run provenance does not match the audit record (run_provenance_status="
                                    "mismatch); the session is quarantined and no report may be generated")
    rec = dict(state.run_provenance) if getattr(state, "run_provenance", None) else None
    if status == "verified" and rec:
        g = rec.get
        return [
            "Run provenance (recorded at session creation and verified against the audit record): "
            f"PharmAgent v{g('app_version', 'n/a')} (git {g('git_sha', 'n/a')}) on Python {g('python', 'n/a')}, "
            f"{g('platform', 'n/a')}. Numerical libraries: numpy {g('numpy', 'n/a')}, scipy {g('scipy', 'n/a')}, "
            f"pandas {g('pandas', 'n/a')}.",
            f"Conventions catalog: version {g('conventions_catalog_version', 'n/a')}, canonical-JSON SHA-256 "
            f"{g('conventions_catalog_canonical_json_sha256', 'n/a')}.",
            "Run provenance status: verified (assurance: " + _assurance_text(state) + ").",
        ]
    if status == "legacy_unverified":
        return ["Run provenance status: legacy_unverified. This session was created before run provenance was "
                "recorded, so the software and the conventions catalog that produced these results are not recorded. "
                "They are not substituted from the current server."]
    return ["Run provenance status: not recorded. This state carries no verified run-provenance record, so the "
            "software and the conventions catalog that produced these results are not stated."]


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
