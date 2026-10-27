"""Records are immutable all the way down; mutation attempts fail and caller-held inputs are never retained."""
import json
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from app.workflow.base import FrozenMap
from app.workflow.contracts import Bounds
from app.workflow.events import WorkflowEvent
from tests.workflow.factories import (
    T0,
    artifact,
    assumptions,
    dataset,
    execution,
    finding,
    review,
    transformation,
)


def event(detail=None):
    return WorkflowEvent(id="event:e1", created_at=T0, producer="p", event_type="finding_raised", actor_kind="system",
                         actor="qc", subject_id="finding:f1", subject_hash="a" * 64, detail=detail or {"a": [1, {"b": 2}]})


def everything():
    art = artifact(b"x")
    return [
        event(), transformation(parameters={"k": [1, {"z": [2]}]}), execution(config={"c": {"d": [1]}}),
        assumptions(initial_estimates={"CL": 5.0}, bounds={"CL": Bounds(lower=1.0, upper=9.0)},
                    model_assumptions=("one-compartment",), covariate_assumptions=("WT on CL",)),
        review(art, rule_versions={"be": "2.2.0"}, legacy_metadata={"old": [1, 2]}), dataset(), finding(), art,
    ]


def walk(obj, path="record"):
    """Every container reachable from a record must be immutable."""
    assert not isinstance(obj, (list, dict, set, bytearray)), f"mutable {type(obj).__name__} at {path}"
    if isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            walk(getattr(obj, name), f"{path}.{name}")
    elif isinstance(obj, FrozenMap):
        for k, v in obj.items():
            walk(v, f"{path}[{k!r}]")
    elif isinstance(obj, tuple):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]")


@pytest.mark.parametrize("rec", everything(), ids=lambda r: type(r).__name__)
def test_no_mutable_container_is_reachable_from_any_record(rec):
    walk(rec)


def test_mappings_refuse_item_assignment_deletion_and_methods_that_mutate():
    e = event()
    for op in (lambda: e.detail.__setitem__("x", 1), lambda: e.detail.__delitem__("a"), lambda: e.detail.update({"x": 1}),
               lambda: e.detail.pop("a"), lambda: e.detail.clear(), lambda: e.detail.setdefault("x", 1)):
        with pytest.raises((TypeError, AttributeError)):
            op()
    assert e.verify_integrity()


def test_nested_lists_and_mappings_are_immutable():
    e = event()
    assert isinstance(e.detail["a"], tuple) and isinstance(e.detail["a"][1], FrozenMap)
    with pytest.raises(AttributeError):
        e.detail["a"].append(2)
    with pytest.raises(TypeError):
        e.detail["a"][1]["b"] = 3
    with pytest.raises(TypeError):
        e.detail["a"][0] = 9
    assert e.verify_integrity()


@pytest.mark.parametrize("make", [lambda r: r.rule_versions, lambda r: r.legacy_metadata], ids=["rule_versions", "legacy_metadata"])
def test_review_mappings_are_immutable(make):
    r = review(artifact(b"x"), rule_versions={"be": "2.2.0"}, legacy_metadata={"old": 1})
    with pytest.raises(TypeError):
        make(r)["x"] = "y"
    assert r.verify_integrity()


def test_assumption_mappings_are_immutable():
    a = assumptions(initial_estimates={"CL": 5.0}, bounds={"CL": Bounds(lower=1.0, upper=9.0)})
    with pytest.raises(TypeError):
        a.initial_estimates["CL"] = 99.0
    with pytest.raises(TypeError):
        a.bounds["V"] = Bounds(lower=1.0)
    with pytest.raises(ValidationError):
        a.bounds["CL"].upper = 100.0
    assert a.verify_integrity()


def test_mutating_the_callers_input_after_construction_does_not_change_the_record():
    detail = {"a": [1, {"b": 2}]}
    e = event(detail)
    before_hash, before_bytes = e.content_hash, e.canonical_bytes()
    detail["a"].append(3)
    detail["a"][1]["b"] = 99
    detail["new"] = 1
    assert e.canonical_bytes() == before_bytes and e.verify_integrity() and e.content_hash == before_hash


def test_mutating_caller_held_inputs_for_every_mapping_field_is_isolated():
    params, cfg = {"k": [1]}, {"c": [1]}
    est, rules, legacy = {"CL": 5.0}, {"be": "1"}, {"o": [1]}
    t, x = transformation(parameters=params), execution(config=cfg)
    a = assumptions(initial_estimates=est)
    r = review(artifact(b"x"), rule_versions=rules, legacy_metadata=legacy)
    snapshots = [o.canonical_bytes() for o in (t, x, a, r)]
    params["k"].append(2); cfg["c"].append(2); est["CL"] = 0.0; rules["be"] = "evil"; legacy["o"].append(2)
    assert snapshots == [o.canonical_bytes() for o in (t, x, a, r)]
    assert all(o.verify_integrity() for o in (t, x, a, r))


def test_values_handed_out_by_dumps_are_copies_not_the_internal_state():
    e = event()
    dumped = e.model_dump()
    dumped["detail"]["a"].append("tampered")
    dumped["detail"]["x"] = 1
    assert e.verify_integrity() and "x" not in e.detail
    again = e.model_dump(mode="json")
    again["detail"]["a"][1]["b"] = 0
    assert e.detail["a"][1]["b"] == 2


def test_immutable_mappings_serialize_to_plain_json_and_round_trip():
    e = event()
    raw = e.canonical_bytes()
    assert json.loads(raw)["detail"] == {"a": [1, {"b": 2}]}
    again = WorkflowEvent.model_validate_json(raw)
    assert again == e and again.canonical_bytes() == raw and again.verify_integrity()
    assert isinstance(again.detail, FrozenMap) and isinstance(again.detail, Mapping)


def test_frozen_map_behaves_like_a_read_only_mapping():
    m = FrozenMap({"b": [1], "a": {"c": 2}})
    assert list(m) == ["a", "b"] and len(m) == 2 and "a" in m and m["b"] == (1,)
    assert m == {"a": {"c": 2}, "b": [1]} and m == FrozenMap({"a": {"c": 2}, "b": [1]})
    assert m.to_plain() == {"a": {"c": 2}, "b": [1]} and isinstance(m.to_plain()["b"], list)
    assert m != {"a": 1} and repr(m).startswith("FrozenMap(")
    with pytest.raises(TypeError):
        m["a"] = 1
    with pytest.raises(TypeError):
        FrozenMap({1: "non-string key"})
    plain = m.to_plain()
    plain["b"].append(2)
    assert m["b"] == (1,)


def test_key_order_does_not_change_the_hash():
    a = event({"x": 1, "y": 2})
    b = event({"y": 2, "x": 1})
    assert a.content_hash == b.content_hash


def test_frozen_map_attributes_cannot_be_rebound():
    m = FrozenMap({"a": 1})
    with pytest.raises(AttributeError):
        m._data = {}
    with pytest.raises(AttributeError):
        m.extra = 1


def test_frozen_map_survives_copy_deepcopy_and_pickle_as_an_equal_immutable_map():
    import copy
    import pickle
    m = FrozenMap({"a": [1, {"b": 2}]})
    for clone in (copy.copy(m), copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        assert clone == m and isinstance(clone, FrozenMap)
        with pytest.raises(TypeError):
            clone["x"] = 1


def test_a_record_deep_copy_stays_valid_and_independent():
    e = event()
    twin = e.model_copy(deep=True)
    assert twin == e and twin.verify_integrity() and twin.canonical_bytes() == e.canonical_bytes()


def test_freezing_is_idempotent_and_preserves_identity_for_frozen_maps():
    from app.workflow.base import freeze
    m = FrozenMap({"a": 1})
    assert freeze(m) is m and freeze(freeze({"a": [1]})) == freeze({"a": [1]})


# --- defaults are part of the contract: validators do not run on them, so they must already be immutable ----------
def _bare_event():
    return WorkflowEvent(id="event:d", created_at=T0, producer="p", event_type="finding_raised", actor_kind="system",
                         actor="qc", subject_id="finding:f1", subject_hash="a" * 64)


def _bare_review():
    from app.workflow.contracts import ReviewRecord
    from app.workflow.stages import ReviewGate
    return ReviewRecord(id="review:d", created_at=T0, producer="p", gate=ReviewGate.FINAL_MODEL, decision="rejected",
                        actor_label="Jane", reason="x")


def _bare_transformation():
    from app.workflow.contracts import DatasetVersionRef, TransformationRecord
    v1, v2 = dataset(), dataset(version=2, parents=("dsv:ds_1:1",))
    return TransformationRecord(id="t:d", created_at=T0, producer="p", parents=(v1.id,),
                                inputs=(DatasetVersionRef.to(v1),), outputs=(DatasetVersionRef.to(v2),), operation="x")


def _bare_execution():
    from app.workflow.contracts import ExecutionRecord
    from app.workflow.stages import WorkflowStage
    return ExecutionRecord(id="x:d", created_at=T0, producer="p", stage=WorkflowStage.NCA, tool="t", status="pending",
                           started_at=T0)


@pytest.mark.parametrize("build", [_bare_event, _bare_review, _bare_transformation, _bare_execution],
                         ids=lambda f: f.__name__.removeprefix("_bare_"))
def test_records_built_with_defaults_are_immutable_too(build):
    rec = build()
    walk(rec)
    mapping_fields = [n for n in type(rec).model_fields if isinstance(getattr(rec, n), FrozenMap)]
    assert mapping_fields, "this record has defaulted mapping fields"
    for name in mapping_fields:
        with pytest.raises(TypeError):
            getattr(rec, name)["leak"] = 1
    assert rec.verify_integrity()


def test_class_level_defaults_are_immutable_so_they_cannot_be_poisoned_for_later_instances():
    from app.workflow.contracts import ExecutionRecord, ReviewRecord, TransformationRecord
    for model, names in ((ReviewRecord, ("rule_versions", "legacy_metadata")), (TransformationRecord, ("parameters",)),
                         (ExecutionRecord, ("config",)), (WorkflowEvent, ("detail",))):
        for n in names:
            default = model.model_fields[n].get_default(call_default_factory=True)
            assert not isinstance(default, (dict, list, set)), f"{model.__name__}.{n} default is mutable"
            assert isinstance(default, FrozenMap)


def test_a_records_own_field_values_can_be_fed_back_into_a_constructor():
    e = event({"a": [1, {"b": 2}]})
    again = WorkflowEvent(**{**e.model_dump(), "detail": e.detail})
    assert again == e and again.verify_integrity()
    x = execution(config={"a": [1, {"b": 2}]})
    assert type(x).model_validate({**x.model_dump(), "config": x.config}) == x
    t = transformation(parameters={"k": (1, 2)})
    assert transformation(parameters=t.parameters).parameters == t.parameters
    a = assumptions(initial_estimates={"CL": 5.0}, bounds={"CL": Bounds(lower=1.0, upper=9.0)})
    assert assumptions(initial_estimates=a.initial_estimates, bounds=a.bounds) == a


def test_records_and_frozen_maps_are_hashable_and_hash_consistently_with_equality():
    m1, m2 = FrozenMap({"a": [1, {"b": 2}]}), FrozenMap({"a": [1, {"b": 2}]})
    assert m1 == m2 and hash(m1) == hash(m2)
    assert len({event(), event()}) == 1
    assert event({"x": 1}) != event({"x": 2})


def test_a_frozen_map_subclass_is_read_once_and_rebuilt_as_a_plain_stable_frozen_map():
    from app.workflow.base import freeze

    class Shifty(FrozenMap):
        calls = 0

        def __getitem__(self, key):                  # answers differently on every call
            Shifty.calls += 1
            return Shifty.calls

    out = freeze(Shifty({"a": 1}))
    assert type(out) is FrozenMap
    assert out["a"] == out["a"] == out.to_plain()["a"]          # a stable snapshot, not the subclass's accessors


def test_different_maps_hash_differently_and_equal_maps_identically():
    assert hash(FrozenMap({"a": 1})) != hash(FrozenMap({"a": 2}))
    assert hash(FrozenMap({"a": 1, "b": 2})) == hash(FrozenMap({"b": 2, "a": 1}))
    assert hash(event({"x": 1})) != hash(event({"x": 2}))


# --- depth and hostile text -----------------------------------------------------------------------------------------
def _deep(n):
    d = {"leaf": 1}
    for _ in range(n):
        d = {"k": [d]}
    return d


def test_over_deep_json_is_a_validation_error_not_a_recursion_error():
    from pydantic import ValidationError
    assert transformation(parameters=_deep(20)).verify_integrity()
    for n in (1000, 100_000):
        with pytest.raises(ValidationError, match="deep"):
            transformation(parameters=_deep(n))
    loop = []
    loop.append(loop)
    with pytest.raises(ValidationError, match="deep"):
        transformation(parameters={"a": loop})


def test_canonical_json_reports_recursion_as_a_contract_error():
    from app.workflow.errors import ContractError
    from app.workflow.hashing import canonical_json
    nested = []
    for _ in range(100_000):
        nested = [nested]
    with pytest.raises(ContractError):
        canonical_json(nested)


def test_frozen_maps_hold_only_json_like_values_so_they_hash_and_serialize_deterministically():
    for bad in ({1}, frozenset({1}), b"xy", bytearray(b"xy")):
        with pytest.raises(TypeError, match="sets or bytes"):
            FrozenMap({"a": bad})
        with pytest.raises(TypeError, match="sets or bytes"):
            FrozenMap({"a": [bad]})


def test_model_copy_refuses_unknown_fields_and_bad_values_with_clear_errors():
    from app.workflow.errors import ContractError
    r = review(artifact(b"x"))
    with pytest.raises(ContractError, match="unknown fields"):
        r.model_copy(update={"zzz": 1})
    with pytest.raises(ValidationError):
        r.model_copy(update={"bound_artifacts": [1], "content_hash": ""})
    with pytest.raises(ValidationError):
        r.model_copy(update={"decision": "maybe", "content_hash": ""})


def test_remaining_collections_are_count_capped():
    with pytest.raises(ValidationError):
        assumptions(model_assumptions=tuple(f"m{i}" for i in range(101)))
    with pytest.raises(ValidationError):
        assumptions(covariate_assumptions=tuple(f"c{i}" for i in range(101)))
    with pytest.raises(ValidationError):
        assumptions(initial_estimates={f"p{i}": 1.0 for i in range(1001)})
    with pytest.raises(ValidationError):
        review(artifact(b"x"), rule_versions={f"r{i}": "1" for i in range(1001)})


# --- datetimes: validated copies carry exact built-in datetimes ------------------------------------------------------
class LyingDatetime(datetime):
    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return False


def test_datetime_subclasses_are_normalised_to_plain_datetimes_on_construction():
    class Stamp(datetime):
        pass

    for field_owner in (finding(created_at=Stamp(2026, 10, 26, 12, tzinfo=UTC)),
                        execution(started_at=Stamp(2026, 10, 26, 12, tzinfo=UTC), finished_at=Stamp(2026, 10, 26, 13, tzinfo=UTC))):
        for name in ("created_at", "started_at", "finished_at"):
            if hasattr(field_owner, name) and getattr(field_owner, name) is not None:
                assert type(getattr(field_owner, name)) is datetime


def test_a_lying_datetime_cannot_dodge_the_ordering_rule_in_a_validated_copy():
    from app.workflow.errors import ContractError
    x = execution()
    object.__setattr__(x, "finished_at", LyingDatetime(2026, 10, 20, tzinfo=UTC))      # earlier than started_at
    object.__setattr__(x, "content_hash", x.compute_hash())
    with pytest.raises(ContractError):
        x.validated_copy()
    ok = execution().validated_copy()
    assert type(ok.started_at) is datetime and type(ok.finished_at) is datetime and type(ok.created_at) is datetime


# --- nesting cannot be built up one FrozenMap at a time ----------------------------------------------------------------
def test_nesting_depth_is_enforced_across_separately_built_frozen_maps():
    inner = FrozenMap({})
    with pytest.raises(ValueError, match="deep"):
        for _ in range(500):
            inner = FrozenMap({"k": inner})
    deep = FrozenMap({})
    for _ in range(60):
        deep = FrozenMap({"k": deep})
    with pytest.raises(ValueError, match="deep"):
        FrozenMap({"t": (deep, (deep,), [[[[[deep]]]]])})
    assert hash(deep) is not None and deep.to_plain()


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_verify_integrity_fails_closed_on_content_that_cannot_be_hashed():
    e = event()
    object.__setattr__(e, "detail", {"raw": b"bytes are not JSON", "s": {1, 2}})
    assert e.verify_integrity() is False


def test_verify_integrity_fails_closed_when_serialization_itself_raises():
    e = event()
    object.__setattr__(e, "detail", {"x": object()})
    with pytest.raises(Exception):
        e.compute_hash()                          # confirms this really is an exception path, not a mismatch
    assert e.verify_integrity() is False


def test_datetimes_at_the_edge_of_the_utc_range_are_a_validation_error_not_a_raw_overflow():
    from datetime import timedelta, timezone

    from pydantic import ValidationError
    for stamp in (datetime(1, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=5))),
                  datetime(9999, 12, 31, 23, 0, tzinfo=timezone(timedelta(hours=-5)))):
        with pytest.raises(ValidationError, match="UTC range"):
            finding(created_at=stamp)
        with pytest.raises(ValidationError, match="UTC range"):
            execution(started_at=stamp, finished_at=None, status="running")
    assert finding(created_at=datetime(1, 1, 1, 5, 0, tzinfo=timezone(timedelta(hours=5)))).created_at.year == 1
