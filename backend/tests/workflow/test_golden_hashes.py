"""Pinned hash vectors, one per persisted record family.

A change to any of these means the hash recipe, a field, or a serialization rule changed: stored hashes would no
longer verify. Bump ``schema_version`` and migrate deliberately; never just re-pin. The Finding and ArtifactRef vectors
were also computed by hand from literal JSON (outside the model) when they were pinned.
"""
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.pharmstate import PharmState
from app.workflow.compat import project_pharmstate
from tests.workflow.factories import T0
from tests.workflow.test_hardening import FAMILY_IDS, families

GOLDEN = {
    "ArtifactRef": "508646c43ddfbabdd7d693f5d54021aef294db2296f7d536efa220b48376e227",
    "DatasetVersion": "e77a6af1b7fab13ba6a84cfc68795ea0b476c85a7ded17fceb0af514aa196061",
    "TransformationRecord": "f04f0abe435d084dd4eadf43073e525090a3e5928d6d4ff7feee851cdf1f11c0",
    "AssumptionSet": "930d9e935f2b7567a04079d6700e196fffa0a7eb2f72aca3d91804de03cdafc1",
    "ExecutionRecord": "01658d098131923fe51280b5ffb25cf305fc3a72b6bd5c4f7a2a12747ffbb0ae",
    "Finding": "7741cb758a8166f7e8684a686ee387bb35cdb1df0042d649b08c704c3b4ad143",
    "ReviewRecord": "79480a44e3d10c15501cb052ac7859188b2e7b8f41f0762300d9cfbaecf4eedf",
    "StageTransition": "9a48818ce1bec5c842e52bd76ec41c91d36bf9198d11437709723c1e11f2bd39",
    "WorkflowEvent": "0f056b842bb7aed788416ccfd0e9cf29d184fdd894fb4467200d2cc63b9af5d1",
    "LegacyProjection": "f4646242a16a6850dc1fd0f97a3ff6c2f57213c47152c0b148ed7cec2df15de7",
}


@pytest.mark.parametrize("name,rec", list(zip(FAMILY_IDS, families(), strict=True)), ids=FAMILY_IDS)
def test_every_record_family_hashes_to_its_pinned_vector(name, rec):
    assert type(rec).__name__ == name
    assert rec.content_hash == GOLDEN[name]
    assert type(rec).load(rec.canonical_bytes()).content_hash == GOLDEN[name]       # and survives a store/load cycle


def test_the_legacy_projection_hashes_to_its_pinned_vector():
    fixture = Path(__file__).parent / "fixtures" / "legacy_session_full.json"
    state = PharmState.model_validate(json.loads(fixture.read_text())["state"])
    projection = project_pharmstate(state, as_of=T0)
    assert type(projection).__name__ == "LegacyProjection"
    assert projection.content_hash == GOLDEN["LegacyProjection"]
    assert type(projection).load(projection.canonical_bytes()) == projection


def test_there_is_a_vector_for_every_record_family_in_the_package():
    from app.workflow.base import WorkflowContract

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    persisted = {c.__name__ for c in subclasses(WorkflowContract) if c.__module__.startswith("app.workflow.")}
    assert persisted == set(GOLDEN), f"unpinned or stale families: {persisted ^ set(GOLDEN)}"



# --- the risky parts: floats, big ints, non-ASCII, astral text, microseconds, offsets, populated maps ----------------
RICH_GOLDEN = {
    "WorkflowEvent": "b30b8de6b809e1d0c5e032c79d966649463a211b28f0f046a3ded115ceb660b7",
    "Finding": "d68840a95907fc4c752a6a57b5d64b670a93195bd8f1574e5ca45926ff729a49",
    "AssumptionSet": "91b2d21deee2685b6c52f44a52d0093bfa2e09cce8179de8a4f7e8406e2eafd2",
    "ExecutionRecord": "2ed90994f9cfe8091a03c29ad1c728404e3e1ca3771df168686a55cf7562ad24",
    "ReviewRecord": "69f40f5c5faa0aeaeafb1d3b0101b97ca445bcdfdc3f986c33c48016c0dbcb75",
}
# These five were computed by hand from literal JSON with only json + hashlib, outside the model, when pinned.


def _rich():
    from datetime import timedelta, timezone

    from app.workflow.contracts import AssumptionSet, Bounds, ExecutionRecord, Finding, ReviewRecord
    from app.workflow.events import WorkflowEvent
    from app.workflow.stages import ReviewGate, WorkflowStage
    tz = timezone(timedelta(hours=5, minutes=30))
    return {
        "WorkflowEvent": WorkflowEvent.create(
            id="event:e1", created_at=T0, producer="pharmagent.test", event_type="finding_raised",
            claimed_actor_kind="system", actor="qc", subject_id="finding:f1", subject_hash="a" * 64),
        "Finding": Finding.create(
            id="finding:rich", created_at=datetime(2026, 10, 26, 15, 0, 0, 123456, tzinfo=tz), producer="pharmagent.test",
            stage=WorkflowStage.DATA_CURATION, severity="info", code="unicode.check", source="automated",
            message="na" + chr(0xEF) + "ve " + chr(0xB5) + "g/L " + chr(0x1F9EA)),
        "AssumptionSet": AssumptionSet.create(
            id="assume:rich", created_at=T0, producer="pharmagent.test", route="oral",
            initial_estimates={"CL": 5.0, "V": 50.0}, bounds={"CL": Bounds(lower=0.1, upper=100.0)}),
        "ExecutionRecord": ExecutionRecord.create(
            id="exec:rich", created_at=T0, producer="pharmagent.test", stage=WorkflowStage.NCA, tool="compute_nca",
            engine="pharmagent", engine_version="0.1.0", environment_id="env:abc", seed=7, status="succeeded",
            config={"alpha": 0.1, "big": 1e22, "n": 2**70, "neg": -0.0, "nested": {"list": [1, 2.5, None, True, "x"]}},
            started_at=T0, finished_at=datetime(2026, 10, 26, 12, 5, tzinfo=UTC)),
        "ReviewRecord": ReviewRecord.create(
            id="review:rich", created_at=T0, producer="pharmagent.test", gate=ReviewGate.FINAL_MODEL, decision="rejected",
            actor_label="Dr. Jos" + chr(0xE9) + " " + chr(0xC1) + "lvarez", reason="insufficient evidence",
            comments=("see note",), rule_versions={"be.average_tost": "2.2.0"}),
    }


@pytest.mark.parametrize("name", list(RICH_GOLDEN))
def test_records_exercising_the_risky_encodings_hash_to_their_pinned_vectors(name):
    rec = _rich()[name]
    assert rec.content_hash == RICH_GOLDEN[name]
    assert type(rec).load(rec.canonical_bytes()) == rec


def test_the_canonical_bytes_use_python_float_ints_and_ascii_escapes_as_the_doc_says():
    raw = _rich()["ExecutionRecord"].canonical_bytes().decode()
    for fragment in ('"alpha":0.1', '"big":1e+22', '"neg":-0.0', '"n":1180591620717411303424', '"list":[1,2.5,null,true,"x"]'):
        assert fragment in raw
    finding_raw = _rich()["Finding"].canonical_bytes().decode()
    assert '"created_at":"2026-10-26T09:30:00.123456Z"' in finding_raw        # +05:30 normalised to UTC, microseconds kept
    assert "\\ud83e\\uddea" in finding_raw and finding_raw.isascii()                  # astral character as a surrogate pair
