"""Re-saving the template changes its hash, so extraction proposes every claim again as a new row. A new
proposal that repeats a claim the reviewer already decided under an earlier template (same source, field,
group, value and unit) carries a visible 'decided_earlier' flag naming that decision, so a rejected value
does not come back looking like a fresh one and a repeat of an approved one is not approved blindly."""
from __future__ import annotations

import pytest

from app.evidence import models as M
from tests.evidence_image_support import EV, build_papers, make_client, records, session_with
from tests.test_evidence_extract import TEMPLATE


@pytest.fixture(scope="module")
def papers(tmp_path_factory):
    return build_papers(tmp_path_factory)


@pytest.fixture
def client(tmp_path, monkeypatch):
    yield from make_client(tmp_path, monkeypatch)


def _decide(client, sid, rec_id, decision, reason):
    r = client.post(EV.format(sid=sid) + "/records/decide", json={"decisions": [
        {"record_id": rec_id, "decision": decision, "reason": reason}]})
    assert r.status_code == 200, r.text


def _resave_and_extract(client, sid):
    tmpl = {"fields": [*TEMPLATE["fields"], {"name": "extra_field", "type": "text"}]}
    assert client.post(EV.format(sid=sid) + "/template", json=tmpl).status_code == 200
    assert client.post(EV.format(sid=sid) + "/extract", json={}).status_code == 200


def _flag(rec, code=M.FLAG_DECIDED_EARLIER):
    return next((f for f in rec["flags"] if f["code"] == code), None)


def test_new_template_proposals_of_decided_claims_are_flagged(client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()})
    old = [r for r in records(client, sid) if r["status"] == "proposed"]
    rejected, approved = old[0], next(r for r in old[1:] if r["verify_status"] == "pass")
    _decide(client, sid, rejected["id"], "reject", "forged cell, checked against the PDF")
    _decide(client, sid, approved["id"], "approve", "matches the table")
    assert all(_flag(r) is None for r in old)

    _resave_and_extract(client, sid)
    fresh = [r for r in records(client, sid) if r["template_hash"] != rejected["template_hash"]
             and r["status"] == "proposed"]
    by = {(r["field"], r["value"], r["group_label"]): r for r in fresh}
    again_rej = by[(rejected["field"], rejected["value"], rejected["group_label"])]
    again_app = by[(approved["field"], approved["value"], approved["group_label"])]
    f_rej, f_app = _flag(again_rej), _flag(again_app)
    assert f_rej and f_rej["severity"] == "warn" and f_rej["origin"] == "script"
    assert "rejected" in f_rej["message"].lower() and "forged cell" in f_rej["message"]
    assert f_app and "approved" in f_app["message"].lower() and "matches the table" in f_app["message"]
    undecided = [r for r in fresh if r not in (again_rej, again_app)]
    assert undecided and all(_flag(r) is None for r in undecided)         # untouched claims stay calm


def test_flag_survives_reverification(client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()})
    first = next(r for r in records(client, sid) if r["status"] == "proposed" and r["verify_status"] == "pass")
    _decide(client, sid, first["id"], "reject", "no")
    _resave_and_extract(client, sid)
    assert client.post(EV.format(sid=sid) + "/records/verify", json={}).status_code == 200
    again = next(r for r in records(client, sid) if r["template_hash"] != first["template_hash"]
                 and (r["field"], r["value"], r["group_label"]) == (first["field"], first["value"], first["group_label"]))
    assert _flag(again) is not None


def test_same_template_repeat_is_not_flagged(client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()})
    first = next(r for r in records(client, sid) if r["status"] == "proposed")
    _decide(client, sid, first["id"], "reject", "no")
    assert client.post(EV.format(sid=sid) + "/extract", json={}).status_code == 200      # same template hash
    assert all(_flag(r) is None for r in records(client, sid))
