"""Run provenance records the conventions catalog identity (version + canonical SHA-256) at session creation.

Guarded area: provenance, orchestrator session creation and the audit genesis entry. The identity is written once,
when the run is created, bound into the hashed genesis audit entry, kept readable in state, and verifiable after
reload. Legacy sessions without it still load and are reported as legacy."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from app.conventions import DEFAULT_CATALOG, load_catalog
from app.core.audit import hash_payload
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState, PharmStateError, apply_writes
from app.core.store import SessionStore


def _canonical_sha256(raw: object) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


# ── catalog identity ───────────────────────────────────────────────────────────────────────────
def test_catalog_identity_is_the_version_and_a_canonical_sha256_of_the_file_content():
    from app.conventions.identity import catalog_identity

    ident = catalog_identity()
    raw = json.loads(Path(DEFAULT_CATALOG).read_text(encoding="utf-8"))
    assert ident == {"conventions_catalog_version": load_catalog().catalog_version,
                     "conventions_catalog_sha256": _canonical_sha256(raw)}
    assert len(ident["conventions_catalog_sha256"]) == 64


def test_the_digest_ignores_formatting_but_not_content(tmp_path):
    from app.conventions.identity import catalog_identity

    raw = json.loads(Path(DEFAULT_CATALOG).read_text(encoding="utf-8"))
    reformatted = tmp_path / "a.json"
    reformatted.write_text(json.dumps(raw, indent=7))
    changed = tmp_path / "b.json"
    changed.write_text(json.dumps({**raw, "catalog_version": raw["catalog_version"] + "9"}))
    base = catalog_identity()["conventions_catalog_sha256"]
    assert catalog_identity(reformatted)["conventions_catalog_sha256"] == base
    assert catalog_identity(changed)["conventions_catalog_sha256"] != base


def test_a_caller_cannot_corrupt_the_cached_identity_or_provenance():
    from app.conventions.identity import catalog_identity
    from app.core.provenance import collect_provenance

    a = catalog_identity()
    a["conventions_catalog_sha256"] = "x"
    assert catalog_identity()["conventions_catalog_sha256"] != "x"
    p = collect_provenance()
    p["git_sha"] = "tampered"
    assert collect_provenance()["git_sha"] != "tampered"


# ── new runs ───────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def orch(tmp_path):
    return Orchestrator(llm=MockLLM(), store=SessionStore(str(tmp_path / "s.db")))


def test_a_new_session_records_the_catalog_identity_at_creation(orch):
    from app.conventions.identity import catalog_identity
    from app.core.provenance import collect_provenance

    prov = orch.create_session().state.run_provenance
    assert prov is not None
    assert {k: prov[k] for k in catalog_identity()} == catalog_identity()
    assert prov == collect_provenance()                      # the same fingerprint the genesis entry hashes


def test_the_genesis_audit_entry_binds_the_recorded_provenance(orch):
    from app.core.provenance import verify_run_provenance

    sess = orch.create_session()
    genesis = sess.audit.entries[0]
    assert genesis.action == "session_created"
    assert hash_payload(sess.state.run_provenance) == genesis.inputs_hash
    assert verify_run_provenance(sess.state.run_provenance, genesis) == "verified"


def test_the_record_is_a_snapshot_taken_at_creation(orch, monkeypatch):
    sess = orch.create_session()
    before = dict(sess.state.run_provenance)
    import app.core.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "collect_provenance", lambda: {**before, "conventions_catalog_version": "9.9.9"})
    assert orch.get_session(sess.id).state.run_provenance == before
    assert orch.create_session().state.run_provenance["conventions_catalog_version"] == "9.9.9"


# ── persistence, legacy records, tampering ─────────────────────────────────────────────────────
def test_the_recorded_provenance_survives_serialisation_and_reload(tmp_path):
    from app.core.provenance import verify_run_provenance

    db = str(tmp_path / "s.db")
    first = Orchestrator(llm=MockLLM(), store=SessionStore(db))
    sid = first.create_session().id
    expected = first.get_session(sid).state.run_provenance
    reloaded = Orchestrator(llm=MockLLM(), store=SessionStore(db)).get_session(sid)
    assert reloaded.state.run_provenance == expected
    assert reloaded.audit.verify() and verify_run_provenance(reloaded.state.run_provenance,
                                                              reloaded.audit.entries[0]) == "verified"


def _rewrite_state(db: str, sid: str, mutate) -> None:
    con = sqlite3.connect(db)
    (state_json,) = con.execute("SELECT state_json FROM sessions WHERE id = ?", (sid,)).fetchone()
    state = json.loads(state_json)
    mutate(state)
    con.execute("UPDATE sessions SET state_json = ? WHERE id = ?", (json.dumps(state), sid))
    con.commit()
    con.close()


def test_a_legacy_session_without_the_field_loads_and_is_reported_as_legacy(tmp_path):
    from app.core.provenance import verify_run_provenance

    db = str(tmp_path / "s.db")
    sid = Orchestrator(llm=MockLLM(), store=SessionStore(db)).create_session().id
    _rewrite_state(db, sid, lambda s: s.pop("run_provenance", None))
    legacy = Orchestrator(llm=MockLLM(), store=SessionStore(db)).get_session(sid)
    assert legacy.state.run_provenance is None
    assert legacy.audit.verify()
    assert verify_run_provenance(legacy.state.run_provenance, legacy.audit.entries[0]) == "legacy"
    assert PharmState.model_validate({"session_id": "old"}).run_provenance is None


def test_a_tampered_stored_record_is_detected(tmp_path):
    from app.core.provenance import verify_run_provenance

    db = str(tmp_path / "s.db")
    sid = Orchestrator(llm=MockLLM(), store=SessionStore(db)).create_session().id
    _rewrite_state(db, sid, lambda s: s["run_provenance"].update(conventions_catalog_sha256="0" * 64))
    sess = Orchestrator(llm=MockLLM(), store=SessionStore(db)).get_session(sid)
    assert sess.audit.verify()                                 # the chain itself was not touched ...
    assert verify_run_provenance(sess.state.run_provenance, sess.audit.entries[0]) == "mismatch"   # ... but this was


def test_tampering_with_the_genesis_entry_breaks_the_audit_chain(orch):
    sess = orch.create_session()
    sess.audit.entries[0].inputs_hash = hash_payload({"forged": True})
    assert not sess.audit.verify()


def test_verification_rejects_a_record_checked_against_a_non_genesis_entry(orch):
    from app.core.provenance import verify_run_provenance

    sess = orch.create_session()
    fake = type(sess.audit.entries[0])(**{**sess.audit.entries[0].to_dict(), "action": "tool_called"})
    assert verify_run_provenance(sess.state.run_provenance, fake) == "mismatch"


@pytest.mark.parametrize("agent", sorted(AGENT_WRITE_FIELDS))
def test_no_agent_may_write_the_run_provenance(agent):
    with pytest.raises(PharmStateError):
        apply_writes(PharmState(), agent, {"run_provenance": {"conventions_catalog_version": "forged"}})


# ── concurrency ────────────────────────────────────────────────────────────────────────────────
def test_concurrent_sessions_record_the_same_immutable_catalog_identity(orch):
    from app.conventions.identity import catalog_identity

    made, errors = [], []

    def worker():
        try:
            made.append(orch.create_session())
        except Exception as exc:                                # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(made) == 8
    idents = {(s.state.run_provenance["conventions_catalog_version"], s.state.run_provenance["conventions_catalog_sha256"])
              for s in made}
    assert idents == {tuple(catalog_identity().values())}
    assert len({id(s.state.run_provenance) for s in made}) == 8   # each session owns its own copy
