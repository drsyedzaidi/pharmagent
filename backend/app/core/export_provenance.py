"""Provenance for exported files, without altering standardized payloads.

CSV bytes and CDISC members (including define.xml) are never modified. Provenance travels instead as:
* a companion manifest (``pharmagent.export_provenance/1``) served next to every download;
* ``X-PharmAgent-*`` response headers on every HTTP download;
* a ``provenance.json`` member inside ZIP packages;
* a comment line in control streams (the manifest stays authoritative).

Everything is taken from the run provenance RECORDED at session creation, never from the current server. A
quarantined (mismatched) session is refused."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from typing import Any

from app.core.provenance import QUARANTINED_STATUS, ProvenanceQuarantined

SCHEMA = "pharmagent.export_provenance/1"
_DIGEST = "conventions_catalog_canonical_json_sha256"
_IDENTITY_KEYS = ("app_version", "git_sha", "python", "platform")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked(state) -> tuple[str, str, dict[str, str]]:
    status = getattr(state, "run_provenance_status", None) or "not_recorded"
    if status == QUARANTINED_STATUS:
        raise ProvenanceQuarantined("run provenance does not match the audit record (run_provenance_status="
                                    "mismatch); the session is quarantined and nothing may be exported")
    assurance = getattr(state, "run_provenance_assurance", None) or "hash_only"
    rec = dict(state.run_provenance) if status == "verified" and getattr(state, "run_provenance", None) else {}
    return status, assurance, rec


def manifest(state, *, session_id: str, kind: str, filename: str, payload: bytes,
             members: dict[str, str] | None = None) -> dict[str, Any]:
    """The export's provenance manifest. ``payload`` is the exact delivered bytes (checksummed)."""
    status, assurance, rec = _checked(state)
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "session_id": session_id,
        "export": {"kind": kind, "filename": filename,
                   "sha256": _sha256(payload) if payload is not None else None},
        "run_provenance_status": status,
        "run_provenance_assurance": assurance,
        # hash_only: chain internally consistent, not authenticated; keyed_seal: authenticated by a keyed seal
        "audit_integrity_mode": assurance,
        "conventions_catalog_version": rec.get("conventions_catalog_version"),
        _DIGEST: rec.get(_DIGEST),
        "run_identity": {k: rec.get(k) for k in _IDENTITY_KEYS} if rec else None,
        "source": "run provenance recorded at session creation (never the current server)",
    }
    if members is not None:
        out["members"] = members
    return out


def headers(state) -> dict[str, str]:
    status, assurance, rec = _checked(state)
    return {
        "X-PharmAgent-Run-Provenance-Status": status,
        "X-PharmAgent-Run-Provenance-Assurance": assurance,
        "X-PharmAgent-Conventions-Catalog-Version": rec.get("conventions_catalog_version") or "none",
        "X-PharmAgent-Conventions-Catalog-Digest": rec.get(_DIGEST) or "none",
    }


def control_comment(state, ext: str) -> str:
    """One comment line for a control stream (';' for NONMEM .ctl, '//' otherwise). The manifest is authoritative."""
    status, assurance, rec = _checked(state)
    mark = ";" if ext == "ctl" else "//"
    return (f"{mark} PharmAgent run provenance: {status} ({assurance}); conventions catalog "
            f"{rec.get('conventions_catalog_version') or 'none'}, "
            f"canonical-JSON sha256 {rec.get(_DIGEST) or 'none'}; "
            "the companion provenance manifest is authoritative\n")


def zip_with_manifest(state, zip_bytes: bytes, *, session_id: str, kind: str, filename: str) -> bytes:
    """Add ``provenance.json`` to a ZIP package, listing each original member's SHA-256.
    Members are copied unchanged."""
    src = zipfile.ZipFile(io.BytesIO(zip_bytes))
    members = {name: _sha256(src.read(name)) for name in src.namelist()}
    inner = manifest(state, session_id=session_id, kind=kind, filename=filename, payload=None, members=members)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            dst.writestr(info, src.read(info.filename))
        # Fixed timestamp: a wall-clock stamp would make the download and its companion manifest's
        # export.sha256 differ whenever the two requests straddle a 2 s DOS-time tick.
        pinfo = zipfile.ZipInfo("provenance.json", date_time=(1980, 1, 1, 0, 0, 0))
        pinfo.compress_type = zipfile.ZIP_DEFLATED
        pinfo.external_attr = 0o600 << 16  # ?rw-------, as writestr sets for a named member
        dst.writestr(pinfo, json.dumps(inner, indent=2, sort_keys=True))
    return buf.getvalue()
