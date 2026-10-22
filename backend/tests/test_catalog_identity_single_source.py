"""Commit 1 of the provenance corrections: one catalog loader/cache, duplicate JSON keys rejected, the digest named as
a canonical-JSON digest with documented reproduction, and session creation failing hard without a valid catalog."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.conventions import DEFAULT_CATALOG, load_catalog

KEY = "conventions_catalog_canonical_json_sha256"


def test_duplicate_json_keys_are_rejected(tmp_path):
    raw = Path(DEFAULT_CATALOG).read_text(encoding="utf-8")
    dup = tmp_path / "dup.json"
    dup.write_text(raw.replace('"catalog_version":', '"catalog_version": "0.0.1", "catalog_version":', 1))
    with pytest.raises(ValueError, match="duplicate"):
        load_catalog(dup)
    nested = tmp_path / "nested.json"
    nested.write_text(raw.replace('"title":', '"title": "x", "title":', 1))
    with pytest.raises(ValueError, match="duplicate"):
        load_catalog(nested)


def test_the_digest_is_named_as_a_canonical_json_digest_and_reproducible_as_documented():
    from app.conventions import identity
    from app.conventions.identity import catalog_identity

    ident = catalog_identity()
    assert set(ident) == {"conventions_catalog_version", KEY}
    raw = json.loads(Path(DEFAULT_CATALOG).read_text(encoding="utf-8"))
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert ident[KEY] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    doc = identity.__doc__
    assert "sort_keys=True" in doc and 'separators=(",", ":")' in doc and "ensure_ascii=False" in doc
    assert "not the SHA-256 of the file bytes" in doc


def test_applied_conventions_and_provenance_read_the_same_cached_catalog():
    from app.conventions.catalog_io import cached_catalog
    from app.conventions.identity import catalog_identity
    from app.conventions.runtime import applied_conventions

    rec = cached_catalog()
    assert rec is cached_catalog()                                   # one load per process
    assert catalog_identity()[KEY] == rec.canonical_json_sha256
    assert applied_conventions(["nca.linear_up_log_down"])["catalog_version"] == rec.catalog.catalog_version
    assert catalog_identity()["conventions_catalog_version"] == rec.catalog.catalog_version


def test_session_creation_fails_hard_without_a_valid_catalog(monkeypatch, tmp_path):
    from app.conventions import catalog_io
    from app.core import provenance
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore

    bad = tmp_path / "broken.json"
    bad.write_text("{not json")
    monkeypatch.setattr(catalog_io, "DEFAULT_CATALOG", bad)
    catalog_io.cached_catalog.cache_clear()
    provenance._collect.cache_clear()
    try:
        orch = Orchestrator(llm=MockLLM(), store=SessionStore(str(tmp_path / "s.db")))
        with pytest.raises(ValueError):
            orch.create_session()
        assert orch.sessions == {}                                    # nothing half-created
    finally:
        catalog_io.cached_catalog.cache_clear()
        provenance._collect.cache_clear()
