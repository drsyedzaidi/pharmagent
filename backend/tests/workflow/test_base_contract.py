"""Shared behaviour of every versioned workflow contract."""
import hashlib
import json

import pytest
from pydantic import ValidationError

from app.workflow.hashing import canonical_json
from tests.workflow.factories import T0, artifact, dataset, finding


def test_contracts_are_immutable():
    f = finding()
    with pytest.raises(ValidationError):
        f.message = "changed"
    with pytest.raises(ValidationError):
        f.content_hash = "0" * 64


def test_unknown_fields_are_rejected():
    payload = json.loads(finding().canonical_bytes())
    payload["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        type(finding()).model_validate(payload)


def test_content_hash_is_computed_and_matches_an_independent_recomputation():
    f = finding()
    body = json.loads(f.canonical_bytes())
    body.pop("content_hash")
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    assert f.content_hash == expected and f.hash_algorithm == "sha256"


def test_content_hash_is_pinned_so_a_silent_format_change_fails():
    assert finding().content_hash == "516d6604c760cdf11bea272472d081d1ab91472fd525a192231c62da4ca291e3"


def test_equal_content_gives_equal_hash_and_any_change_gives_another():
    assert finding().content_hash == finding().content_hash
    assert finding(message="other").content_hash != finding().content_hash
    assert finding(severity="error").content_hash != finding().content_hash


def test_hash_ignores_timezone_spelling_of_the_same_instant():
    from datetime import timedelta, timezone
    east = T0.astimezone(timezone(timedelta(hours=5)))
    assert finding(created_at=east).content_hash == finding().content_hash
    assert finding(created_at=east).created_at.utcoffset() == timedelta(0)


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValidationError):
        finding(created_at=T0.replace(tzinfo=None))


def test_tampered_content_hash_is_rejected_on_load():
    data = json.loads(finding().canonical_bytes())
    data["message"] = "tampered"
    with pytest.raises(ValidationError, match="content_hash"):
        type(finding()).model_validate(data)


def test_round_trip_json_is_byte_stable():
    for obj in (finding(), artifact(), dataset()):
        raw = obj.canonical_bytes()
        again = type(obj).model_validate_json(raw)
        assert again == obj and again.canonical_bytes() == raw


def test_canonical_bytes_are_sorted_compact_ascii():
    raw = finding(message="naïve µ").canonical_bytes()
    assert raw.isascii()
    text = raw.decode()
    parsed = json.loads(text)
    assert list(parsed) == sorted(parsed)
    assert text == json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_canonical_json_rejects_non_finite_numbers():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})


def test_unsupported_schema_versions_are_refused_in_both_directions():
    data = json.loads(finding().canonical_bytes())
    for bad in (0, 2, 99):
        with pytest.raises(ValidationError, match="schema_version"):
            type(finding()).model_validate({**data, "schema_version": bad})


@pytest.mark.parametrize("bad", ["", " lead", "has space", "x" * 129, "a/b", "-start", "tab\t", "új"])
def test_invalid_identifiers_are_rejected(bad):
    with pytest.raises(ValidationError):
        finding(id=bad)


def test_producer_must_be_a_clean_nonempty_string():
    for bad in ("", "   ", "a\x00b"):
        with pytest.raises(ValidationError):
            finding(producer=bad)


def test_parents_are_unique_wellformed_and_not_self():
    with pytest.raises(ValidationError, match="unique"):
        finding(parents=("a", "a"))
    with pytest.raises(ValidationError, match="itself"):
        finding(parents=("finding:f1",))
    with pytest.raises(ValidationError):
        finding(parents=("bad id",))
    assert finding(parents=("dataset:ds_1:v1",)).parents == ("dataset:ds_1:v1",)


def test_migration_metadata_is_optional_and_hashed():
    from app.workflow.base import MigrationInfo
    m = MigrationInfo(from_schema_version=0, note="projected from legacy state")
    assert finding().migration is None
    assert finding(migration=m).content_hash != finding().content_hash
    with pytest.raises(ValidationError):
        MigrationInfo(from_schema_version=-1, note="x")
    with pytest.raises(ValidationError):
        MigrationInfo(from_schema_version=0, note="")


def test_unknown_marker_requires_a_reason_and_a_known_status():
    from app.workflow.base import Unknown
    assert Unknown(status="unknown", reason="not recorded").status == "unknown"
    with pytest.raises(ValidationError):
        Unknown(status="unknown", reason="")
    with pytest.raises(ValidationError):
        Unknown(status="maybe", reason="x")


# --- integrity of live objects (the seal is verified at load; verify_integrity covers objects already in memory) --
def _event():
    from app.workflow.events import WorkflowEvent
    return WorkflowEvent(id="event:e1", created_at=T0, producer="p", event_type="finding_raised", actor_kind="system",
                         actor="qc", subject_id="finding:f1", subject_hash="a" * 64, detail={"a": [1]})


def test_verify_integrity_passes_for_an_untouched_record():
    assert _event().verify_integrity() is True


def test_nested_containers_cannot_be_mutated_but_a_forced_rewrite_is_still_caught():
    e = _event()
    with pytest.raises(AttributeError):
        e.detail["a"].append(2)
    object.__setattr__(e, "actor", "evil")        # bypasses every guard; verify_integrity is the backstop
    assert e.verify_integrity() is False


def test_model_copy_cannot_smuggle_in_a_change_under_the_old_hash():
    e = _event()
    with pytest.raises(ValidationError, match="content_hash"):
        e.model_copy(update={"actor": "evil"})
    resealed = e.model_copy(update={"actor": "alice", "content_hash": ""})
    assert resealed.actor == "alice" and resealed.verify_integrity() and resealed.content_hash != e.content_hash
    assert e.model_copy().content_hash == e.content_hash


def test_model_construct_skips_validation_but_is_caught_by_verify_integrity():
    e = _event()
    forged = type(e).model_construct(**{**e.__dict__, "actor": "evil"})
    assert forged.verify_integrity() is False


def test_review_checks_its_own_integrity_before_it_can_count_as_an_approval():
    from tests.workflow.factories import review
    art = artifact(b"cleaned")
    r = review(art)
    forged = type(r).model_construct(**{**r.__dict__, "actor_label": "someone else"})
    cur = {art.id: art.digest}
    assert r.is_current_approval(cur) and forged.binding_status(cur) == "invalid" and not forged.is_current_approval(cur)


def test_a_hostile_current_mapping_cannot_vouch_for_every_artifact():
    from collections.abc import Mapping

    from tests.workflow.factories import review

    class Liar(Mapping):
        def __getitem__(self, key): return "f" * 64
        def __contains__(self, key): return True
        def __iter__(self): return iter(())
        def __len__(self): return 0

    r = review(artifact(b"cleaned"))
    assert r.binding_status(Liar()) == "missing"


@pytest.mark.parametrize("field,value", [("n_subjects", True), ("n_dose_records", True), ("n_observation_records", True)])
def test_booleans_are_not_accepted_as_counts(field, value):
    with pytest.raises(ValidationError):
        dataset(**{field: value})


def test_booleans_are_not_accepted_as_seed_lloq_or_schema_version():
    from app.workflow.contracts import BlqRepresentation
    from tests.workflow.factories import execution
    with pytest.raises(ValidationError):
        execution(seed=True)
    with pytest.raises(ValidationError):
        BlqRepresentation(basis="dv_nonpositive", lloq=True)
    data = json.loads(finding().canonical_bytes())
    with pytest.raises(ValidationError):
        type(finding()).model_validate({**data, "schema_version": True})


# --- text and size hardening ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["‮evil", "a b", "a​b", "​​", "x\u0085y", " lead", "trail ", " "])
def test_labels_reject_bidi_separator_zero_width_blank_looking_and_edge_whitespace(bad):
    with pytest.raises(ValidationError):
        finding(producer=bad)


def test_messages_reject_separators_and_invisible_text_but_keep_tabs_and_newlines():
    for bad in ("a b", "‮evil", "​​"):
        with pytest.raises(ValidationError):
            finding(message=bad)
    assert finding(message="a\tb\nc").message == "a\tb\nc"
    with pytest.raises(ValidationError, match="visible"):
        finding(message="\n\t\n")


def test_json_objects_are_size_capped():
    from app.workflow.events import WorkflowEvent
    base = dict(id="event:e1", created_at=T0, producer="p", event_type="finding_raised", actor_kind="system", actor="qc",
                subject_id="finding:f1", subject_hash="a" * 64)
    assert WorkflowEvent(**base, detail={"k": "x" * 60_000}).detail["k"]
    with pytest.raises(ValidationError, match="at most"):
        WorkflowEvent(**base, detail={"k": "x" * 70_000})


def test_collections_are_count_capped():
    from app.workflow.contracts import ColumnSchema
    from tests.workflow.factories import execution
    with pytest.raises(ValidationError):
        dataset(columns=tuple(ColumnSchema(name=f"c{i}") for i in range(2001)))
    with pytest.raises(ValidationError):
        execution(warnings=tuple(f"w{i}" for i in range(101)))
    with pytest.raises(ValidationError):
        execution(output_artifact_ids=tuple(f"a{i}" for i in range(1001)))
