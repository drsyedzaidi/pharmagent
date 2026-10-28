"""Pinned hash vectors, one per persisted record family.

A change to any of these means the hash recipe, a field, or a serialization rule changed: stored hashes would no
longer verify. Bump ``schema_version`` and migrate deliberately; never just re-pin. The Finding and ArtifactRef vectors
were also computed by hand from literal JSON (outside the model) when they were pinned.
"""
import json
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
    "WorkflowEvent": "bf8f527b02572b4a816bc36e115b6450b795c7a5afb1ac3fe8a70921ed5ad07a",
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
