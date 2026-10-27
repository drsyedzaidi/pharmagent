"""A claimed actor kind and a display label carry no authority; only (later) an authenticated principal will."""
import pytest
from pydantic import ValidationError

from tests.workflow.factories import artifact, review


def test_the_claimed_actor_kind_never_changes_binding_status_or_approval_validity():
    art = artifact(b"cleaned")
    current, changed = {art.id: art.digest}, {art.id: artifact(b"edited").digest}
    for kind in (None, "human", "agent", "system", "unknown"):
        r = review(art, claimed_actor_kind=kind)
        assert r.binding_status(current) == "current" and r.is_current_approval(current)
        assert r.binding_status(changed) == "stale" and not r.is_current_approval(changed)
        assert r.binding_status({}) == "missing"


def test_a_rejection_stays_a_rejection_whatever_kind_is_claimed():
    art = artifact(b"x")
    assert not review(art, decision="rejected", claimed_actor_kind="human").is_current_approval({art.id: art.digest})


def test_claimed_kind_is_optional_untrusted_and_validated():
    assert review().claimed_actor_kind is None
    with pytest.raises(ValidationError):
        review(claimed_actor_kind="superuser")
    assert "reviewer_kind" not in type(review()).model_fields
    assert "claimed_actor_kind" in type(review()).model_fields


def test_no_authenticated_principal_exists_until_phase_d():
    assert review().authenticated_principal_ref is None
    with pytest.raises(ValidationError):
        review(authenticated_principal_ref="user:alice")


def test_display_label_is_separate_and_required():
    assert review(actor_label="Dr. Alice Example").actor_label == "Dr. Alice Example"
    for bad in ("", "  ", "a\x00b"):
        with pytest.raises(ValidationError):
            review(actor_label=bad)


def test_legacy_reviewer_values_are_kept_as_labelled_metadata_not_authority():
    r = review(legacy_metadata={"legacy_reviewer": "jsmith", "legacy_role": "approver"})
    assert r.legacy_metadata["legacy_reviewer"] == "jsmith"
    art = artifact(b"x")
    claims_approval = review(art, decision="rejected", legacy_metadata={"approved": True, "legacy_role": "root"})
    assert not claims_approval.is_current_approval({art.id: art.digest})      # metadata cannot turn a rejection into one


@pytest.mark.parametrize("bad", ["Codex", "codex-agent", "Claude", "gpt-5", "GPT4", "ChatGPT", "mockllm", "assistant",
                                 "LLM reviewer", "Copilot", "OpenAI", "ＣＬＡＵＤＥ"])
def test_the_name_lint_still_refuses_labels_that_name_an_automated_agent(bad):
    with pytest.raises(ValidationError):
        review(actor_label=bad)




# --- the approval predicate re-validates the whole record, not just its (unkeyed) hash ------------------------------
def _rehashed(record, **changes):
    """A forged record: validators skipped via model_construct, then a matching hash computed for it."""
    forged = type(record).model_construct(**{**record.__dict__, **changes})
    object.__setattr__(forged, "content_hash", forged.compute_hash())
    return forged


def test_a_rehashed_forgery_with_no_bindings_is_not_a_current_approval():
    art = artifact(b"x")
    forged = _rehashed(review(art), bound_artifacts=())
    assert forged.verify_integrity()                               # the unkeyed hash alone is fooled
    assert forged.binding_status({}) == "invalid" and not forged.is_current_approval({})


def test_a_rehashed_forgery_with_a_binding_that_is_not_content_addressed_is_invalid():
    from app.workflow.contracts import ArtifactBinding
    art = artifact(b"x")
    bad = ArtifactBinding.model_construct(artifact_id="whatever", digest=art.digest)
    forged = _rehashed(review(art), bound_artifacts=(bad,))
    assert forged.binding_status({"whatever": art.digest}) == "invalid"


def test_the_advisory_name_lint_is_not_part_of_the_approval_predicate():
    """The label carries no authority, so re-validation checks structure and not the lint: tightening the lint later
    must not void approvals already recorded."""
    art = artifact()
    forged = _rehashed(review(art), actor_label="Codex")
    assert forged.binding_status({art.id: art.digest}) == "current"        # unchanged by what the label says
    import app.workflow.contracts as contracts
    kept = review(art, actor_label="Dr. Foo Bar")
    original = contracts.AUTOMATED_NAME_BRANDS
    try:
        contracts.AUTOMATED_NAME_BRANDS = (*original, "foo")               # a later release tightens the lint
        assert kept.binding_status({art.id: art.digest}) == "current" and kept.is_current_approval({art.id: art.digest})
        with pytest.raises(ValidationError):
            review(art, actor_label="Dr. Foo Bar")                         # but new records are held to it
    finally:
        contracts.AUTOMATED_NAME_BRANDS = original


# --- a hostile `current` mapping fails closed ----------------------------------------------------------------------
def test_current_must_be_a_mapping_of_plain_strings():
    from collections.abc import Mapping

    from app.workflow.errors import ContractError
    art = artifact(b"x")
    r = review(art)

    class Key(str):
        pass

    class Boom:
        def __str__(self):
            raise RuntimeError("boom")

    for bad in ({Key(art.id): art.digest}, {art.id: Key(art.digest)}, {art.id: Boom()},
                {1: "x"}):
        with pytest.raises(ContractError):
            r.binding_status(bad)

    class Endless(Mapping):
        def __getitem__(self, key):
            return "a" * 64

        def __iter__(self):
            while True:
                yield "k"

        def __len__(self):
            return 0

    for not_a_mapping in (None, [(art.id, art.digest)], "text", 5):
        with pytest.raises(ContractError, match="must be a mapping"):
            r.binding_status(not_a_mapping)

    with pytest.raises(ContractError, match="too many"):
        r.binding_status(Endless())
    assert r.binding_status({art.id: art.digest}) == "current"


def test_a_mapping_that_raises_while_being_read_fails_closed_with_the_contract_error():
    from collections.abc import Mapping

    from app.workflow.errors import ContractError

    class Exploding(Mapping):
        def __getitem__(self, key):
            raise RuntimeError("cannot read")

        def __iter__(self):
            return iter(["sha256:" + "a" * 64])

        def __len__(self):
            return 1

    with pytest.raises(ContractError, match="could not be read"):
        review(artifact(b"x")).binding_status(Exploding())


# --- comparisons must run on a validated copy, never on the live (possibly subclass-laden) object ---------------------
class LyingStr(str):
    """A str whose comparisons lie. Only reachable through model_construct / object.__setattr__ plus a rehash."""
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


def _resealed(record):
    object.__setattr__(record, "content_hash", record.compute_hash())
    return record


def test_a_digest_that_lies_about_equality_cannot_make_a_stale_binding_look_current():
    from app.workflow.contracts import ArtifactBinding
    art = artifact(b"cleaned")
    rec = review(art)
    liar = ArtifactBinding.model_construct(artifact_id=art.id, digest=LyingStr(art.digest))
    object.__setattr__(rec, "bound_artifacts", (liar,))
    _resealed(rec)
    wrong = {art.id: "0" * 64}
    assert rec.binding_status(wrong) == "stale" and not rec.is_current_approval(wrong)
    assert rec.is_current_approval({art.id: art.digest})                          # and it still works when genuinely current


def test_a_decision_that_lies_about_equality_cannot_turn_a_rejection_into_an_approval():
    art = artifact(b"cleaned")
    rec = review(art, decision="rejected")
    object.__setattr__(rec, "decision", LyingStr("rejected"))
    _resealed(rec)
    assert not rec.is_current_approval({art.id: art.digest})


def test_lineage_runs_on_validated_copies_so_a_lying_id_cannot_pass():
    """The planted parent hashes and compares equal to an existing version's id, so the LIVE object would find it in
    the index and report no missing parent; only the validated copy (plain str) exposes that it is a ghost."""
    from app.workflow.errors import ContractError
    from app.workflow.lineage import check_lineage
    from tests.workflow.factories import dataset

    class Ghost(str):
        def __eq__(self, other):
            return True

        def __hash__(self):
            return hash("dsv:real")

    real = dataset(id="dsv:real")
    child = dataset(id="dsv:child", parents=("dsv:real",))
    object.__setattr__(child, "parents", (Ghost("dsv:ghost"),))
    _resealed(child)
    with pytest.raises(ContractError, match="missing"):
        check_lineage([real, child])


def test_only_the_lint_objects_to_a_label_like_this_one():
    """Proves the lint itself (not another validator) is what rejects the labels parametrized above."""
    assert review(actor_label="Dr. Foo Bar").actor_label == "Dr. Foo Bar"
    with pytest.raises(ValidationError, match="automated agent"):
        review(actor_label="Dr. Claude Bernard-Bot")


def test_the_lint_does_not_reject_ordinary_names_that_merely_contain_a_brand_substring():
    for fine in ("Egypt Hassan", "Dr. Alice Example", "Syed S. Zaidi", "Robert Abbott"):
        assert review(actor_label=fine).actor_label == fine


def test_a_record_whose_stored_hash_was_blanked_is_not_valid():
    from app.workflow.errors import ContractError
    art = artifact(b"x")
    rec = review(art)
    object.__setattr__(rec, "content_hash", "")                # truncated or corrupted storage
    assert not rec.verify_integrity()
    assert rec.binding_status({art.id: art.digest}) == "invalid" and not rec.is_current_approval({art.id: art.digest})
    with pytest.raises(ContractError, match="integrity"):
        rec.validated_copy()


def test_the_structural_context_cannot_be_guessed_to_bypass_the_lint():
    from app.workflow.contracts import ReviewRecord
    data = review().model_dump()
    data.update(actor_label="Claude Bot", content_hash="")
    for ctx in ({"structural": True}, {"structural": 1}, {"STRUCTURAL": True}):
        with pytest.raises(ValidationError):
            ReviewRecord.model_validate(data, context=ctx)
    with pytest.raises(ValidationError):
        ReviewRecord.model_validate_json(ReviewRecord.model_validate({**data, "actor_label": "Dr. Foo"}).model_dump_json()
                                         .replace("Dr. Foo", "Claude Bot"), context={"structural": True})
