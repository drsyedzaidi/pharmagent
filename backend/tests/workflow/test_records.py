import pytest
from pydantic import ValidationError

from app.workflow.base import Unknown
from app.workflow.contracts import ArtifactBinding, Bounds
from app.workflow.stages import ReviewGate, WorkflowStage
from tests.workflow.factories import T0, T1, artifact, assumptions, execution, finding, review, transformation


# --- TransformationRecord -------------------------------------------------------------------------
def test_transformation_unknowns_are_explicit():
    t = transformation(actor=Unknown(status="unavailable", reason="legacy"), affected_rows=Unknown(status="unknown", reason="n/a"),
                       reversible=Unknown(status="unknown", reason="n/a"), reason=Unknown(status="unknown", reason="n/a"))
    assert t.actor.status == "unavailable"
    with pytest.raises(ValidationError):
        transformation(affected_rows=-1)
    with pytest.raises(ValidationError):
        transformation(operation="")


def test_transformation_parameters_must_be_json_and_finite():
    assert transformation(parameters={"a": [1, 2], "b": {"c": None}}).parameters["a"] == (1, 2)
    with pytest.raises(ValidationError):
        transformation(parameters={"a": float("nan")})
    with pytest.raises(ValidationError):
        transformation(parameters={"a": object()})


# --- AssumptionSet --------------------------------------------------------------------------------
def test_assumptions_default_to_unknown_not_invented_values():
    a = assumptions()
    for name in ("route", "dose_regimen", "analyte", "matrix", "blq_policy", "exclusion_policy", "missing_data_policy",
                 "model_assumptions", "parameterization", "initial_estimates", "bounds", "covariate_assumptions"):
        assert getattr(a, name).status == "unknown"


def test_assumptions_accept_recorded_values():
    a = assumptions(route="oral", blq_policy="M1: exclude later BLQ", initial_estimates={"CL": 5.0, "V": 50.0},
                    bounds={"CL": Bounds(lower=0.1, upper=100.0)}, covariate_assumptions=("WT on CL",))
    assert a.route == "oral" and a.bounds["CL"].upper == 100.0


def test_assumption_bounds_and_estimates_must_be_consistent():
    with pytest.raises(ValidationError):
        Bounds(lower=5.0, upper=1.0)
    with pytest.raises(ValidationError):
        Bounds(lower=float("nan"))
    with pytest.raises(ValidationError, match="outside"):
        assumptions(initial_estimates={"CL": 500.0}, bounds={"CL": Bounds(lower=0.1, upper=100.0)})
    with pytest.raises(ValidationError):
        assumptions(route="teleport")
    with pytest.raises(ValidationError):
        assumptions(initial_estimates={"CL": float("inf")})


# --- ExecutionRecord ------------------------------------------------------------------------------
def test_execution_records_engine_environment_seed_and_artifacts():
    e = execution(code_artifact_ids=("sha256:" + "1" * 64,), log_artifact_ids=("sha256:" + "2" * 64,),
                  output_artifact_ids=("sha256:" + "3" * 64,), warnings=("lambda_z unreliable",))
    assert e.engine == "pharmagent" and e.seed == 7 and e.stage is WorkflowStage.NCA and len(e.warnings) == 1


def test_execution_status_and_times_are_consistent():
    with pytest.raises(ValidationError, match="finished_at"):
        execution(status="succeeded", finished_at=None)
    with pytest.raises(ValidationError, match="finished_at"):
        execution(status="running", finished_at=T1)
    with pytest.raises(ValidationError, match="before"):
        execution(started_at=T1, finished_at=T0)
    assert execution(status="running", finished_at=None).status == "running"
    with pytest.raises(ValidationError):
        execution(status="exploded")


def test_execution_unknown_engine_details_stay_explicit_and_seed_can_be_absent():
    e = execution(engine=Unknown(status="unknown", reason="legacy"), engine_version=Unknown(status="unknown", reason="legacy"),
                  seed=Unknown(status="unavailable", reason="deterministic"))
    assert e.engine.status == "unknown" and e.seed.status == "unavailable"


def test_execution_artifact_ids_must_be_wellformed_and_unique():
    with pytest.raises(ValidationError):
        execution(output_artifact_ids=("bad id",))
    with pytest.raises(ValidationError, match="unique"):
        execution(output_artifact_ids=("a", "a"))


# --- Finding --------------------------------------------------------------------------------------
def test_finding_validation():
    assert finding().severity == "warning"
    for kw in ({"severity": "fatal"}, {"code": ""}, {"code": "has space"}, {"message": ""}, {"source": "robot"}):
        with pytest.raises(ValidationError):
            finding(**kw)


# --- ReviewRecord ---------------------------------------------------------------------------------
def test_review_binds_to_exact_artifact_hashes():
    art = artifact(b"cleaned")
    r = review(art)
    assert r.binding_status({art.id: art.digest}) == "current" and r.is_current_approval({art.id: art.digest})


def test_changing_the_artifact_makes_the_approval_stale():
    art = artifact(b"cleaned")
    r = review(art)
    changed = artifact(b"cleaned!")
    assert r.binding_status({art.id: changed.digest}) == "stale"
    assert not r.is_current_approval({art.id: changed.digest})


def test_missing_artifact_is_reported_not_treated_as_current():
    art = artifact(b"cleaned")
    r = review(art)
    assert r.binding_status({}) == "missing" and not r.is_current_approval({})
    assert r.binding_status({art.id: art.digest, "other": "f" * 64}) == "current"


def test_rejection_is_never_a_current_approval():
    art = artifact(b"cleaned")
    r = review(art, decision="rejected")
    assert r.binding_status({art.id: art.digest}) == "current" and not r.is_current_approval({art.id: art.digest})


def test_approval_requires_bound_artifacts_and_unique_ids():
    with pytest.raises(ValidationError, match="bind"):
        review(bound_artifacts=())
    art = artifact()
    b = {"artifact_id": art.id, "digest": art.digest}
    with pytest.raises(ValidationError, match="unique"):
        review(art, bound_artifacts=(b, b))
    with pytest.raises(ValidationError):
        ArtifactBinding(artifact_id=art.id, digest="short")


def test_review_carries_rule_versions_comments_and_gate():
    r = review(rule_versions={"be.average_tost": "2.2.0"}, comments=("looks fine",), gate=ReviewGate.FINAL_MODEL)
    assert r.rule_versions["be.average_tost"] == "2.2.0" and r.gate is ReviewGate.FINAL_MODEL and r.comments == ("looks fine",)
    with pytest.raises(ValidationError):
        review(rule_versions={"": "1"})
    with pytest.raises(ValidationError):
        review(reason="")


def test_each_gate_belongs_to_a_stage():
    from app.workflow.stages import GATE_STAGE
    assert set(GATE_STAGE) == set(ReviewGate) and len(ReviewGate) == 7
    assert GATE_STAGE[ReviewGate.FINAL_REPORT] is WorkflowStage.REPORTING


def test_message_fields_refuse_blank_and_control_text_but_allow_newlines():
    for bad in ("   ", "\x01bell", "a\x00b"):
        with pytest.raises(ValidationError):
            finding(message=bad)
    assert finding(message="line one\nline two").message == "line one\nline two"


def test_a_binding_cannot_name_one_artifact_and_carry_anothers_hash():
    art = artifact(b"reviewed")
    with pytest.raises(ValidationError, match="content address"):
        ArtifactBinding(artifact_id="whatever", digest=art.digest)
    assert ArtifactBinding(artifact_id=art.id, digest=art.digest).digest == art.digest
