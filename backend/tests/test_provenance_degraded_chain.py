"""Re-review LOW 4: v1 entries do not bind index/action. New sessions always write v2, so a provenance-marker genesis in a
degraded (v1) chain can only be a forgery: mismatch. A genuinely old v1 chain without the marker stays legacy."""
from __future__ import annotations

import json

import pytest

from app.core.audit import AuditChain, AuditEntry, hash_payload
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.provenance import verify_run_provenance
from app.core.store import SessionStore


@pytest.fixture
def orch(tmp_path):
    return Orchestrator(llm=MockLLM(), store=SessionStore(str(tmp_path / "s.db")))


def _as_v1(chain: AuditChain, outputs: dict | None = None) -> AuditChain:
    raw = json.loads(json.dumps(chain.to_list()))
    out = AuditChain()
    for i, d in enumerate(raw):
        e = AuditEntry(**{**d, "hash_version": 1})
        if i == 0 and outputs is not None:
            e.outputs_hash = hash_payload(outputs)
        if i:
            e.prev_hash = out.entries[-1].entry_hash
        e.entry_hash = e.compute_hash()
        out.entries.append(e)
    return out


def test_a_marker_genesis_in_a_degraded_v1_chain_is_a_mismatch(orch):
    sess = orch.create_session()
    v1 = _as_v1(sess.audit)
    assert v1.verify_status()["ok"] and v1.verify_status()["degraded"]
    assert verify_run_provenance(sess.state.run_provenance, v1, sess.id) == "mismatch"


def test_a_genuine_pre_marker_v1_chain_without_a_record_is_still_legacy(orch):
    sess = orch.create_session()
    v1 = _as_v1(sess.audit, outputs={"session_id": sess.id})
    assert verify_run_provenance(None, v1, sess.id) == "legacy_unverified"
