"""Dataset lineage is a DAG of opaque version ids; the display number is never identity."""
import pytest
from pydantic import ValidationError

from app.workflow.contracts import DatasetVersionRef
from app.workflow.errors import ContractError
from app.workflow.lineage import check_lineage
from tests.workflow.factories import T0, dataset, transformation

REF = DatasetVersionRef.to


def test_identity_is_the_opaque_id_and_the_display_number_is_optional_and_repeatable():
    a = dataset(version=1, id="dsv:alpha")
    b = dataset(version=1, id="dsv:beta")
    assert a.id != b.id and a.display_version == b.display_version == 1
    assert dataset(version=None, id="dsv:gamma").display_version is None
    for bad in (0, -1, True):
        with pytest.raises(ValidationError):
            dataset(version=bad, id="dsv:bad")


def test_a_root_has_no_parents_and_any_version_may_have_any_number_of_parents():
    root = dataset()
    child = dataset(version=7, id="dsv:ds_1:7", parents=(root.id,))        # a gap in numbering is fine
    assert child.parents == (root.id,)
    merged = dataset(version=1, id="dsv:merged", dataset_id="analysis_set", parents=("dsv:a", "dsv:b", "dsv:c"))
    assert len(merged.parents) == 3 and merged.dataset_id == "analysis_set"


def test_branched_lineage_two_cleaning_alternatives_from_one_source():
    src = dataset(id="dsv:src")
    a = dataset(id="dsv:clean_a", parents=("dsv:src",))
    b = dataset(id="dsv:clean_b", parents=("dsv:src",))
    check_lineage([src, a, b])


def test_multi_input_merge_of_dosing_and_observations_under_a_new_logical_identity():
    dose = dataset(id="dsv:dose", dataset_id="dosing")
    obs = dataset(id="dsv:obs", dataset_id="observations")
    merged = dataset(id="dsv:merged", dataset_id="analysis_set", parents=("dsv:dose", "dsv:obs"))
    t = transformation(inputs=(REF(dose), REF(obs)), outputs=(REF(merged),), operation="merge")
    check_lineage([dose, obs, merged], [t])
    assert [r.version_id for r in t.inputs] == ["dsv:dose", "dsv:obs"]


def test_one_transformation_may_produce_several_outputs():
    src = dataset(id="dsv:src")
    train = dataset(id="dsv:train", parents=("dsv:src",))
    test = dataset(id="dsv:test", parents=("dsv:src",))
    t = transformation(inputs=(REF(src),), outputs=(REF(train), REF(test)), operation="split")
    check_lineage([src, train, test], [t])


def test_a_missing_parent_is_refused():
    child = dataset(id="dsv:child", parents=("dsv:ghost",))
    with pytest.raises(ContractError, match="missing"):
        check_lineage([child])


def test_a_cycle_is_refused():
    a = dataset(id="dsv:a", parents=("dsv:b",))
    b = dataset(id="dsv:b", parents=("dsv:a",))
    with pytest.raises(ContractError, match="cycle"):
        check_lineage([a, b])
    c = dataset(id="dsv:c", parents=("dsv:d",))
    d = dataset(id="dsv:d", parents=("dsv:e",))
    e = dataset(id="dsv:e", parents=("dsv:c",))
    with pytest.raises(ContractError, match="cycle"):
        check_lineage([c, d, e])


def test_duplicate_version_ids_are_refused():
    with pytest.raises(ContractError, match="duplicate"):
        check_lineage([dataset(id="dsv:x"), dataset(id="dsv:x", source="other.csv")])


def test_a_transformation_must_reference_versions_that_exist_with_the_recorded_hash():
    src, out = dataset(id="dsv:src"), dataset(id="dsv:out", parents=("dsv:src",))
    t = transformation(inputs=(REF(src),), outputs=(REF(out),))
    with pytest.raises(ContractError, match="missing"):
        check_lineage([src], [t])                                     # output not supplied
    tampered = dataset(id="dsv:out", parents=("dsv:src",), source="changed.csv")
    with pytest.raises(ContractError, match="record_hash"):
        check_lineage([src, tampered], [t])                           # same id, different content


def test_outputs_must_descend_from_exactly_the_transformations_inputs():
    src, other = dataset(id="dsv:src"), dataset(id="dsv:other")
    out = dataset(id="dsv:out", parents=("dsv:other",))
    t = transformation(inputs=(REF(src),), outputs=(REF(out),))
    with pytest.raises(ContractError, match="parents"):
        check_lineage([src, other, out], [t])


def test_transformation_shape_is_validated():
    src = dataset(id="dsv:src")
    out = dataset(id="dsv:out", parents=("dsv:src",))
    with pytest.raises(ValidationError):
        transformation(inputs=(), outputs=(REF(out),), parents=())
    with pytest.raises(ValidationError):
        transformation(inputs=(REF(src),), outputs=())
    with pytest.raises(ValidationError, match="unique"):
        transformation(inputs=(REF(src), REF(src)), outputs=(REF(out),), parents=("dsv:src",))
    with pytest.raises(ValidationError, match="both"):
        transformation(inputs=(REF(src),), outputs=(REF(src),), parents=("dsv:src",))
    with pytest.raises(ValidationError, match="parents"):
        transformation(inputs=(REF(src),), outputs=(REF(out),), parents=("dsv:something_else",))


def test_a_ref_pins_the_exact_record():
    src = dataset(id="dsv:src")
    ref = REF(src)
    assert ref.version_id == "dsv:src" and ref.record_hash == src.content_hash
    with pytest.raises(ValidationError):
        DatasetVersionRef(version_id="bad id", record_hash=src.content_hash)
    with pytest.raises(ValidationError):
        DatasetVersionRef(version_id="dsv:src", record_hash="short")
    assert T0


def test_shared_ancestors_are_visited_once_and_are_not_mistaken_for_cycles():
    root = dataset(id="dsv:root")
    left = dataset(id="dsv:left", parents=("dsv:root",))
    right = dataset(id="dsv:right", parents=("dsv:root",))
    tip = dataset(id="dsv:tip", parents=("dsv:left", "dsv:right"))          # a diamond
    for order in ([tip, left, right, root], [root, left, right, tip], [left, tip, root, right]):
        check_lineage(order)


def test_duplicate_transformation_ids_are_refused():
    src, a, b = dataset(id="dsv:src"), dataset(id="dsv:a", parents=("dsv:src",)), dataset(id="dsv:b", parents=("dsv:src",))
    t1 = transformation(inputs=(REF(src),), outputs=(REF(a),), id="transform:same")
    t2 = transformation(inputs=(REF(src),), outputs=(REF(b),), id="transform:same")
    with pytest.raises(ContractError, match="duplicate transformation"):
        check_lineage([src, a, b], [t1, t2])


def test_an_output_cannot_be_claimed_by_two_transformations():
    src, y = dataset(id="dsv:src"), dataset(id="dsv:y", parents=("dsv:src",))
    t1 = transformation(inputs=(REF(src),), outputs=(REF(y),), id="transform:one")
    t2 = transformation(inputs=(REF(src),), outputs=(REF(y),), id="transform:two")
    with pytest.raises(ContractError, match="more than one transformation"):
        check_lineage([src, y], [t1, t2])


def test_lineage_checks_do_not_trust_stored_hashes_of_tampered_records():
    src, out = dataset(id="dsv:src"), dataset(id="dsv:out", parents=("dsv:src",))
    t = transformation(inputs=(REF(src),), outputs=(REF(out),))
    check_lineage([src, out], [t])
    object.__setattr__(src, "source", "EVIL.csv")          # rewrite after sealing; stored hash no longer matches content
    with pytest.raises(ContractError, match="integrity"):
        check_lineage([src, out], [t])
    src2 = dataset(id="dsv:src")
    object.__setattr__(t, "operation", "something else")
    with pytest.raises(ContractError, match="integrity"):
        check_lineage([src2, out], [t])


def test_check_lineage_accepts_only_exact_record_types():
    from app.workflow.contracts import DatasetVersion

    class Sneaky(DatasetVersion):
        def validated_copy(self):                      # a subclass that would vouch for itself
            return self

    good = dataset(id="dsv:x")
    sneaky = Sneaky.model_validate({**good.model_dump(), "content_hash": ""})      # re-sealed: new record type
    with pytest.raises(ContractError, match="exact"):
        check_lineage([sneaky])
    with pytest.raises(ContractError, match="exact"):
        check_lineage([good], [object()])
