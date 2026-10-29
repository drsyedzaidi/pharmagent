import hashlib

import pytest
from pydantic import ValidationError

from app.workflow.base import Unknown
from app.workflow.contracts import BlqRepresentation, ColumnSchema
from tests.workflow.factories import T0, artifact, dataset


def test_artifact_is_content_addressed():
    a = artifact(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    assert a.digest == digest and a.id == f"sha256:{digest}" and a.size_bytes == 5
    assert a.digest_algorithm == "sha256" and a.media_type == "text/csv" and a.producer == "pharmagent.test"
    assert artifact(b"hello").id == a.id and artifact(b"hellp").id != a.id


def test_artifact_id_must_match_its_digest():
    a = artifact(b"x")
    with pytest.raises(ValidationError, match="content address"):
        type(a).create(**{**a.model_dump(exclude={"content_hash"}), "id": "sha256:" + "0" * 64})


def test_artifact_lineage_records_parent_artifacts():
    raw = artifact(b"raw", kind="dataset")
    out = artifact(b"derived", parents=(raw.id,))
    assert out.parents == (raw.id,)


@pytest.mark.parametrize("bad", ["", "text", "text/", "/csv", "TEXT/CSV", "text/csv; x", "a b/c"])
def test_media_type_is_validated(bad):
    with pytest.raises(ValidationError):
        artifact(media_type=bad)


def test_artifact_kind_and_size_are_validated():
    with pytest.raises(ValidationError):
        artifact(kind="mystery")
    a = artifact(b"")
    assert a.size_bytes == 0
    with pytest.raises(ValidationError):
        type(a).create(**{**a.model_dump(exclude={"content_hash"}), "size_bytes": -1})


def test_digest_must_be_lowercase_hex_sha256():
    a = artifact(b"x")
    for bad in ("A" * 64, "g" * 64, "a" * 63):
        with pytest.raises(ValidationError):
            type(a).create(**{**a.model_dump(exclude={"content_hash"}), "digest": bad, "id": f"sha256:{bad}"})


@pytest.mark.parametrize("unit", ["ng/mL", "mg", "h", "1", "%", "\u00b5g/L", "10^9/L", "mL/min/1.73m^2"])
def test_valid_units_are_accepted(unit):
    assert ColumnSchema(name="X", unit=unit).unit == unit


@pytest.mark.parametrize("unit", ["", " ", " mg", "mg ", "m\x00g", "x" * 33, "mg\nL", "<script>"])
def test_invalid_units_are_rejected(unit):
    with pytest.raises(ValidationError):
        ColumnSchema(name="X", unit=unit)


def test_missing_unit_is_explicit_not_invented():
    col = ColumnSchema(name="X")
    assert isinstance(col.unit, Unknown) and col.unit.status == "unknown"


def test_column_names_must_be_unique_and_clean():
    c = ColumnSchema(name="ID")
    with pytest.raises(ValidationError, match="unique"):
        dataset(columns=(c, c))
    with pytest.raises(ValidationError):
        ColumnSchema(name="")
    with pytest.raises(ValidationError):
        ColumnSchema(name="a\x00b")


def test_subject_column_must_exist_when_columns_are_listed():
    with pytest.raises(ValidationError, match="subject_id_column"):
        dataset(subject_id_column="USUBJID")


def test_counts_cannot_be_negative_and_digest_must_be_hex():
    for kw in ({"n_subjects": -1}, {"n_dose_records": -1}, {"n_observation_records": -1}, {"data_digest": "xyz"}):
        with pytest.raises(ValidationError):
            dataset(**kw)


def test_blq_representation_is_explicit():
    b = BlqRepresentation(basis="cens_flag", censoring_column="CENS", lloq=0.5)
    assert dataset(blq=b).blq == b
    assert BlqRepresentation(basis="none").lloq.status == "unknown"
    with pytest.raises(ValidationError):
        BlqRepresentation(basis="cens_flag")          # a flag basis names its column
    with pytest.raises(ValidationError):
        BlqRepresentation(basis="none", censoring_column="CENS")
    with pytest.raises(ValidationError):
        BlqRepresentation(basis="dv_nonpositive", lloq=float("nan"))
    with pytest.raises(ValidationError):
        BlqRepresentation(basis="dv_nonpositive", lloq=-1.0)


def test_unrecorded_dataset_fields_default_to_explicit_unknown():
    d = dataset(source=Unknown(status="unavailable", reason="legacy"), data_digest=Unknown(status="unavailable", reason="legacy"))
    assert d.source.status == "unavailable" and d.data_digest.reason == "legacy"
    assert d.created_at == T0


