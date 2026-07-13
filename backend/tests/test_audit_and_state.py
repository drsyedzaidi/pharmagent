"""Audit hash-chain integrity and PharmState write-access enforcement."""
import pytest

from app.core.audit import AuditChain
from app.core.pharmstate import PharmState, PharmStateError, apply_writes


def test_audit_chain_intact():
    chain = AuditChain()
    for i in range(3):
        chain.append(agent="nca", tool="compute_nca", action="x",
                     inputs={"i": i}, outputs={"o": i}, timestamp=f"t{i}")
    assert chain.verify() is True
    assert len(chain.entries) == 3
    assert chain.entries[0].prev_hash == "0" * 64
    assert chain.entries[1].prev_hash == chain.entries[0].entry_hash


def test_audit_chain_detects_tampering():
    chain = AuditChain()
    chain.append(agent="a", tool="t", action="x", inputs={"v": 1},
                 outputs={"v": 1}, timestamp="t0")
    chain.append(agent="a", tool="t", action="x", inputs={"v": 2},
                 outputs={"v": 2}, timestamp="t1")
    # tamper with the first entry's recorded output hash
    chain.entries[0].outputs_hash = "deadbeef"
    assert chain.verify() is False


def test_pharmstate_allows_owned_field():
    st = PharmState(session_id="s1")
    st2 = apply_writes(st, "nca", {"nca_parameters": [{"subject": "S1"}]})
    assert st2.nca_parameters == [{"subject": "S1"}]
    assert st.nca_parameters is None  # original unchanged (immutable update)


def test_pharmstate_blocks_foreign_field():
    st = PharmState(session_id="s1")
    with pytest.raises(PharmStateError):
        apply_writes(st, "nca", {"qc_verdict": "PASS"})  # qc field, not nca's
