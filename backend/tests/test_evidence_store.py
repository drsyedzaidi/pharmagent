"""EvidenceStore: schema, hash-derived ids, idempotent inserts, decision paths, immutability triggers."""
from __future__ import annotations

import sqlite3

import pytest

from app.core.store import SessionStore
from app.evidence import store as evstore
from app.evidence.models import EvidenceStateError, Locator
from app.evidence.store import EvidenceStore

SHA_A = "a" * 64
SHA_B = "b" * 64


@pytest.fixture
def st() -> EvidenceStore:
    return evstore.configure(":memory:")


def _src(st: EvidenceStore, session: str = "s1", sha: str = SHA_A, **kw):
    base = dict(session_id=session, sha256=sha, filename="paper.pdf", pdf_path=f"/x/{sha}.pdf",
                parse_path=f"/x/{sha}.pages.json", pages=3, title="A title", metadata={"author": "Au"},
                text_chars=1200, parse_status="parsed", decimal_style=".", uploaded_by="tester")
    base.update(kw)
    return st.add_source(**base)


def _locator(page: int = 1, **kw) -> dict:
    return Locator(page=page, kind="text", snippet="Cmax was 3,412 ng/mL", **kw).to_json()


def _proposal(field: str = "Cmax", value: str = "3,412", group: str = "", **kw) -> dict:
    p = dict(field=field, group_label=group, value=value, value_num=3412.0, unit="ng/mL",
             locator=_locator(), extractor="table", verify_status="pass", verify_reason="", flags=[])
    p.update(kw)
    return p


# ---------------------------------------------------------------- schema ---

def test_schema_creates_idempotently_on_existing_sessions_db(tmp_path):
    db = tmp_path / "pharmagent.db"
    SessionStore(db)
    EvidenceStore(db)
    EvidenceStore(db)   # second open must not fail or duplicate objects
    names = {r[0] for r in sqlite3.connect(db).execute("SELECT name FROM sqlite_master")}
    assert {"sessions", "evidence_source", "evidence_config", "evidence_screen",
            "evidence_record", "evidence_dataset", "trg_evrec_frozen", "trg_evrec_nodelete"} <= names


def test_get_store_is_lazy_singleton_after_configure():
    a = evstore.configure(":memory:")
    assert evstore.get_store() is a
    b = evstore.configure(":memory:")
    assert evstore.get_store() is b and b is not a


# ---------------------------------------------------------------- sources ---

def test_add_source_dedupes_within_session_and_separates_sessions(st):
    s1 = _src(st)
    again = _src(st, title="different title ignored on dedupe")
    assert again.id == s1.id and again.title == "A title"
    assert s1.id.startswith("src_") and len(s1.id) == 16
    other = _src(st, session="s2")
    assert other.id != s1.id and other.sha256 == s1.sha256
    assert [s.id for s in st.list_sources("s1")] == [s1.id]
    assert st.get_source(s1.id).metadata == {"author": "Au"}
    assert st.get_source("src_nope") is None


def test_add_source_rejects_bad_parse_status(st):
    with pytest.raises(sqlite3.IntegrityError):
        _src(st, parse_status="weird")


# ----------------------------------------------------------------- config ---

def test_save_config_hash_stable_across_key_order_and_replaces_row(st):
    body1 = [{"id": "c1", "text": "human", "kind": "inclusion", "required": True, "keywords": ["human"]}]
    body2 = [{"keywords": ["human"], "required": True, "kind": "inclusion", "text": "human", "id": "c1"}]
    h1 = st.save_config("s1", "criteria", body1, "tester")
    h2 = st.save_config("s1", "criteria", body2, "tester")
    assert h1 == h2 and len(h1) == 64
    h3 = st.save_config("s1", "criteria", body1 + [{"id": "c2", "text": "rat", "kind": "exclusion",
                                                     "required": False, "keywords": ["rat"]}], "tester")
    assert h3 != h1
    cfg = st.get_config("s1", "criteria")
    assert cfg["body_hash"] == h3 and len(cfg["body"]) == 2 and cfg["saved_by"] == "tester"
    assert st.get_config("s1", "template") is None
    with pytest.raises(sqlite3.IntegrityError):
        st.save_config("s1", "bogus", {}, "tester")


# ---------------------------------------------------------------- screens ---

def _screen(source_id: str, chash: str = "h" * 64, verdict: str = "include") -> dict:
    return dict(source_id=source_id, criteria_hash=chash, verdict=verdict, score=0.8,
                reasoning="keywords matched", method="keyword",
                criteria=[{"criterion_id": "c1", "kind": "inclusion", "required": True, "met": True,
                           "evidence": [{"page": 1, "snippet": "healthy human volunteers", "span": None}],
                           "note": ""}])


def test_upsert_screens_idempotent_for_same_source_and_criteria(st):
    s = _src(st)
    first = st.upsert_screens("s1", [_screen(s.id)])
    second = st.upsert_screens("s1", [_screen(s.id, verdict="exclude")])
    assert [x.id for x in first] == [x.id for x in second]
    assert second[0].verdict == "include"        # existing row wins
    assert len(st.list_screens("s1")) == 1
    new = st.upsert_screens("s1", [_screen(s.id, chash="g" * 64)])
    assert new[0].id != first[0].id and len(st.list_screens("s1")) == 2
    assert first[0].id.startswith("scr_") and first[0].status == "proposed"


def test_decide_screen_once(st):
    s = _src(st)
    scr = st.upsert_screens("s1", [_screen(s.id)])[0]
    out = st.decide_screen(scr.id, "approved", "reviewer", "looks right")
    assert out.status == "approved" and out.decided_by == "reviewer" and out.decided_at.endswith("Z")
    with pytest.raises(EvidenceStateError, match="already decided"):
        st.decide_screen(scr.id, "rejected", "reviewer", "changed my mind")
    with pytest.raises(ValueError):
        st.decide_screen(scr.id, "proposed", "reviewer", "")
    assert st.list_screens("s1", status="approved")[0].id == scr.id


# ---------------------------------------------------------------- records ---

def test_insert_records_idempotent_on_record_hash(st):
    s = _src(st)
    recs = st.insert_records("s1", s, "t" * 64, [_proposal()])
    assert len(recs) == 1 and recs[0].id.startswith("rec_") and len(recs[0].id) == 20
    assert len(recs[0].record_hash) == 64 and recs[0].status == "proposed"
    again = st.insert_records("s1", s, "t" * 64, [_proposal(flags=[{"code": "x"}])])
    assert again == []                               # same claim identity -> skipped
    changed = st.insert_records("s1", s, "t" * 64, [_proposal(value="3,413")])
    assert len(changed) == 1 and changed[0].record_hash != recs[0].record_hash
    assert len(st.list_records("s1")) == 2


def test_record_hash_ignores_span_literal_status_and_flags(st):
    s = _src(st)
    a = st.insert_records("s1", s, "t" * 64, [_proposal()])[0]
    loc = _locator(span=[5, 10], literal="3,412")
    b = st.insert_records("s1", s, "t" * 64, [_proposal(locator=loc, verify_status="fail",
                                                        verify_reason="snippet_not_found")])
    assert b == [] and st.list_records("s1")[0].record_hash == a.record_hash


def test_decide_record_once_and_filters(st):
    s = _src(st)
    r = st.insert_records("s1", s, "t" * 64, [_proposal(), _proposal(field="AUC", value="24,560")])
    out = st.decide_record(r[0].id, "approved", "reviewer", "verified")
    assert out.status == "approved" and out.decision_reason == "verified"
    with pytest.raises(EvidenceStateError, match="already decided"):
        st.decide_record(r[0].id, "rejected", "reviewer", "")
    with pytest.raises(EvidenceStateError):
        st.decide_record("rec_missing", "approved", "reviewer", "")
    assert [x.id for x in st.list_records("s1", status="proposed")] == [r[1].id]
    assert [x.id for x in st.list_records("s1", source_id=s.id, field="AUC")] == [r[1].id]


def test_approved_record_is_immutable_at_the_sql_level(st):
    s = _src(st)
    r = st.insert_records("s1", s, "t" * 64, [_proposal()])[0]
    st.decide_record(r.id, "approved", "reviewer", "ok")
    conn = st._conn
    with pytest.raises(sqlite3.IntegrityError, match="approved record is immutable"):
        conn.execute("UPDATE evidence_record SET value='9' WHERE id=?", (r.id,))
    with pytest.raises(sqlite3.IntegrityError, match="approved record is immutable"):
        conn.execute("DELETE FROM evidence_record WHERE id=?", (r.id,))
    assert st.list_records("s1")[0].value == "3,412"


def test_reverify_touches_only_proposed_rows(st):
    s = _src(st)
    r = st.insert_records("s1", s, "t" * 64, [_proposal(), _proposal(field="AUC", value="24,560")])
    st.decide_record(r[0].id, "approved", "reviewer", "ok")
    loc = _locator(span=[0, 5], literal="3,412")
    assert st.reverify(r[0].id, "fail", "snippet_not_found", loc, [{"code": "x"}]) is False
    assert st.reverify(r[1].id, "fail", "snippet_not_found", loc, [{"code": "x"}]) is True
    rows = {x.id: x for x in st.list_records("s1")}
    assert rows[r[0].id].verify_status == "pass" and rows[r[0].id].flags == []
    assert rows[r[1].id].verify_status == "fail" and rows[r[1].id].locator["span"] == [0, 5]
    assert rows[r[1].id].flags == [{"code": "x"}]


def test_approved_record_hashes_sorted(st):
    s = _src(st)
    r = st.insert_records("s1", s, "t" * 64, [_proposal(value=v) for v in ("1", "2", "3")])
    for x in r[:2]:
        st.decide_record(x.id, "approved", "reviewer", "")
    st.decide_record(r[2].id, "rejected", "reviewer", "")
    hashes = st.approved_record_hashes("s1")
    assert hashes == sorted(hashes) and set(hashes) == {r[0].record_hash, r[1].record_hash}


# --------------------------------------------------------------- datasets ---

def test_add_and_list_datasets(st):
    ds = st.add_dataset("s1", "t" * 64, "/x/long.csv", "/x/wide.csv", "1" * 64, "2" * 64,
                        ["h2", "h1"], n_records=2, n_sources=1, exported_by="tester")
    assert ds.id == "ds_" + "1" * 12 and ds.record_hashes == ["h1", "h2"]
    assert [d.id for d in st.list_datasets("s1")] == [ds.id]
    assert st.add_dataset("s1", "t" * 64, "/x/long.csv", "/x/wide.csv", "1" * 64, "2" * 64,
                          ["h1"], n_records=1, n_sources=1, exported_by="tester").id == ds.id
    assert len(st.list_datasets("s1")) == 1


# ----------------------------------------------------------------- counts ---

def test_counts_shape(st):
    s = _src(st)
    st.upsert_screens("s1", [_screen(s.id)])
    r = st.insert_records("s1", s, "t" * 64, [_proposal(), _proposal(value="9")])
    st.decide_record(r[0].id, "approved", "reviewer", "")
    c = st.counts("s1")
    assert c == {"sources": 1, "screens": {"proposed": 1, "approved": 0, "rejected": 0},
                 "records": {"proposed": 1, "approved": 1, "rejected": 0}, "datasets": 0}
    assert st.counts("empty")["sources"] == 0
