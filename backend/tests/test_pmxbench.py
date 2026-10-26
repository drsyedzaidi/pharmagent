"""PMxbench scenario_00 adapter: fixture integrity, dataset preparation, submission
building and the CLI. Keyless; nothing here fits or simulates a model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from app.benchmarks import pmxbench as pmx
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.data_tools import NONMEM_NA_VALUES
from app.tools.pkmodel_tools import _build_subjects, _build_subjects_counted, _covariate_candidates

FIX = Path(__file__).parent / "fixtures" / "pmxbench" / "scenario_00"
MANIFEST = json.loads((FIX / "MANIFEST.json").read_text(encoding="utf-8"))
REPO_ROOT = Path(__file__).resolve().parents[2]
# Third-party installs and caches, not repository content.
_PRUNE = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(FIX / "data.csv", na_values=NONMEM_NA_VALUES)


@pytest.fixture(scope="module")
def prepared(raw):
    return pmx.prepare_dataset(raw)


def _template() -> dict:
    return yaml.safe_load((FIX / "submission.template.yaml").read_text(encoding="utf-8"))


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ── fixture integrity and the no-answer-key rule ─────────────────────────────

def test_fixture_files_match_the_pinned_upstream_blobs():
    for name, meta in MANIFEST["files"].items():
        data = (FIX / name).read_bytes()
        assert _git_blob_sha(data) == meta["git_blob_sha"], name
        if "sha256" in meta:
            assert hashlib.sha256(data).hexdigest() == meta["sha256"], name
            assert len(data) == meta["bytes"], name
    assert len(MANIFEST["source"]["commit"]) == 40


def test_fixture_folder_holds_only_the_public_payload_and_its_licence():
    assert {p.name for p in FIX.iterdir()} == {
        "data.csv", "sap.md", "submission.template.yaml", "README.md", "LICENSE", "MANIFEST.json"}
    assert (FIX / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_no_answer_key_file_anywhere_in_the_repository():
    found = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in _PRUNE]
        found += [os.path.join(root, f) for f in files
                  if f.lower().startswith("truth.") or f.lower() in {"generate.r", "score.py"}]
    assert found == []


# ── prepare_dataset on the real fixture ──────────────────────────────────────

def test_rate_is_derived_on_every_dosing_record(raw, prepared):
    out, report = prepared
    dosing = out["AMT"] > 0
    assert int(dosing.sum()) == 120 == report["n_dosing_records"]
    assert out.loc[dosing, "RATE"].tolist() == pytest.approx((out.loc[dosing, "AMT"] * 24.0).tolist())
    assert (out.loc[~dosing, "RATE"] == 0).all()
    assert out["RATE"].notna().all()


def test_cens_count_equals_blq_count_and_dv_is_the_lloq_on_censored_rows(raw, prepared):
    out, report = prepared
    assert int(out["CENS"].sum()) == int((raw["BLQ"] == 1).sum()) == 156 == report["n_censored"]
    assert (out["CENS"].to_numpy() == raw["BLQ"].to_numpy()).all()
    assert (out.loc[out["CENS"] == 1, "DV"] == pmx.SCENARIO_00_LLOQ).all()
    assert report["n_censored_predose"] + report["n_censored_postdose"] == 156


def test_no_zero_dv_is_left_uncensored(prepared):
    out, report = prepared
    obs = out["AMT"] == 0
    assert int((obs & (out["DV"] <= 0) & (out["CENS"] == 0)).sum()) == 0
    assert int((obs & out["DV"].isna()).sum()) == 0
    assert "no uncensored DV <= 0" in report["checks"]["validated"]


def test_roles_auto_detect_without_overrides(prepared):
    out, report = prepared
    assert tuple(out.columns) == pmx.ANALYSIS_COLUMNS
    assert detect_roles(list(out.columns)) == pmx.EXPECTED_ROLES
    assert set(pmx.COVARIATES).isdisjoint(detect_roles(list(out.columns)))
    assert {k: report["roles"][k] for k in pmx.EXPECTED_ROLES} == pmx.EXPECTED_ROLES


def test_rows_keep_source_order_so_rowid_maps_back(raw, prepared):
    out, report = prepared
    assert len(out) == len(raw) == len(report["rowid_by_row"])
    assert report["rowid_by_row"] == raw["ROWID"].astype(str).tolist()
    assert (out["ID"].to_numpy() == raw["ID"].to_numpy()).all()


def test_every_transformation_and_dropped_column_is_recorded(prepared):
    _, report = prepared
    assert [t["step"] for t in report["transformations"]] == [
        "derive_rate", "map_cens", "censored_dv_to_lloq", "drop_columns"]
    assert {d["column"] for d in report["dropped_columns"]} == {"DAY", "DOSE", "BLQ", "ROWID"}
    assert report["lloq"] == 0.1 and report["time_unit"] == "day"
    assert set(report["covariates"]["baseline_pearson_r"]) == {"WT~ALB", "WT~CRCL", "ALB~CRCL"}


def test_input_frame_is_not_modified(raw):
    before = raw.copy()
    pmx.prepare_dataset(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_pharmagent_subject_builder_reads_the_prepared_frame(prepared):
    out, _ = prepared
    roles = detect_roles(list(out.columns))
    subjects, multi, _ = _build_subjects(out, roles, with_blq=True)
    assert len(subjects) == 120 and not multi
    assert all(len(s["doses"]) == 1 and s["doses"][0]["rate"] == pytest.approx(s["doses"][0]["amt"] * 24.0)
               for s in subjects)
    assert {s["lloq"] for s in subjects if any(s["obs_blq"])} == {0.1}
    assert sum(sum(s["obs_blq"]) for s in subjects) == 36          # post-dose censored records
    assert all(set(s["cov"]) == set(pmx.COVARIATES) for s in subjects)
    # Automatic SCM candidates: only the pre-specified covariates (WT is the allometric column).
    cands, _ = _covariate_candidates(_build_subjects(out, roles)[0], ["CL", "V"])
    assert {c["covariate"] for c in cands} == {"ALB", "CRCL"}


def test_non_m3_consumers_exclude_censored_records_m1(prepared):
    """docs/pmxbench.md, G5 (resolved): without with_blq the builder applies M1, so the
    36 post-dose censored records are excluded and counted, never scored at the LLOQ."""
    out, report = prepared
    subjects, _, _, n_dropped = _build_subjects_counted(out, detect_roles(list(out.columns)))
    obs = [c for s in subjects for c in s["obs_c"]]
    assert len(obs) == 1440 - 36
    assert pmx.SCENARIO_00_LLOQ not in obs
    assert n_dropped == 36 == report["n_censored_postdose"]
    assert any("exclude them (M1)" in n for n in report["notes"])


def test_raw_file_would_break_m3_and_pollute_the_covariate_pool(raw):
    """Why the transformations exist: unprepared, the M3 limit resolves to 0 and the
    record key ROWID and DAY enter the covariate pool."""
    subjects, _, _ = _build_subjects(raw, detect_roles(list(raw.columns)), with_blq=True)
    assert {s["lloq"] for s in subjects if any(s["obs_blq"])} == {0.0}
    assert {"ROWID", "DAY"} <= set(subjects[0]["cov"])


# ── prepare_dataset fails closed ─────────────────────────────────────────────

def _tiny(**overrides) -> pd.DataFrame:
    base = {"ID": [1, 1, 1], "DAY": [0, 0, 1], "TIME": [0.0, 0.0417, 1.0], "DV": [None, 40.0, 0.0],
            "AMT": [100, 0, 0], "DOSE": [100] * 3, "WT": [70.0] * 3, "ALB": [4.0] * 3,
            "CRCL": [100.0] * 3, "BLQ": [0, 0, 1], "ROWID": [1, 2, 3]}
    return pd.DataFrame({**base, **overrides})


def test_tiny_valid_frame_prepares():
    out, report = pmx.prepare_dataset(_tiny())
    assert out["DV"].tolist()[1:] == [40.0, 0.1] and report["n_censored"] == 1


@pytest.mark.parametrize("overrides, message", [
    ({"BLQ": [0, 0, 0]}, "uncensored DV <= 0"),
    ({"DV": [None, 40.0, 0.05]}, "BLQ = 1 with DV != 0"),
    ({"BLQ": [1, 0, 1]}, "BLQ = 1 on a dosing record"),
    ({"BLQ": [0, 0, 2]}, "BLQ is not 0/1"),
    ({"DV": [None, None, 0.0]}, "observation without DV"),
    ({"ROWID": [1, 1, 3]}, "ROWID must be present and unique"),
    ({"AMT": [100, 50, 0], "DV": [None, None, 0.0]}, "exactly one dosing record at TIME 0"),
    ({"TIME": [1.0, 1.0417, 2.0]}, "exactly one dosing record at TIME 0"),
    ({"TIME": [0.0, 0.0, 0.0]}, "at least one post-dose observation"),
    ({"AMT": [-100, 0, 0]}, "negative AMT"),
    ({"ALB": [4.0, None, 4.0]}, "covariate ALB missing"),
    ({"DV": [".", "40", "abc"]}, "not numeric"),
])
def test_inconsistent_records_are_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        pmx.prepare_dataset(_tiny(**overrides))


def test_missing_columns_and_existing_role_columns_are_refused():
    with pytest.raises(ValueError, match="missing required columns"):
        pmx.prepare_dataset(_tiny().drop(columns=["CRCL"]))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        pmx.prepare_dataset(_tiny().assign(RATE=0.0))
    with pytest.raises(ValueError, match="positive"):
        pmx.prepare_dataset(_tiny(), lloq=0.0)


def test_columns_outside_the_specification_are_dropped_and_recorded():
    out, report = pmx.prepare_dataset(_tiny(EXTRA=[1, 2, 3]))
    assert "EXTRA" not in out.columns
    assert {"column": "EXTRA", "reason": "not in the scenario_00 data specification (sap.md Appendix A)"} \
        in report["dropped_columns"]


def test_nonmem_dot_strings_are_read_as_missing():
    out, _ = pmx.prepare_dataset(_tiny(DV=[".", "40", "0"]))
    assert pd.isna(out["DV"].iloc[0]) and out["DV"].iloc[2] == 0.1


# ── build_submission on synthetic PharmState ─────────────────────────────────

_META = {"dataset_id": "ds_s00", "n_records": 1680, "n_subjects": 120,
         "detected_roles": dict(pmx.EXPECTED_ROLES),
         "columns": [{"name": c} for c in pmx.ANALYSIS_COLUMNS]}
# Same key set as app.core.provenance.collect_provenance() (pinned by a test below).
_PROV = (("app_version", "0.1.0"), ("git_sha", "abc1234"), ("python", "3.13.1"),
         ("platform", "macOS-26"), ("numpy", "2.4.6"), ("scipy", "1.17.1"), ("pandas", "3.0.3"),
         ("fastapi", "0.136.3"), ("pydantic", "2.13.4"), ("python-docx", "1.2.0"),
         ("conventions_catalog_version", "1.4.1"),
         ("conventions_catalog_canonical_json_sha256", "0" * 64))


def _fit(model_key="iv_2cmt", *, theta=None, effects=(), error_model="proportional",
         sigma=None, converged=True) -> dict:
    """Mirrors the keys of app.compute.nlme._assemble that the adapter reads."""
    theta = theta or {"CL": 0.2, "VC": 3.1, "Q": 0.5, "VP": 2.4}
    return {"method": "FOCE-I", "model_key": model_key, "label": "2-cmt IV (linear)",
            "iiv_params": ["CL", "VC"], "error_model": error_model,
            "theta": theta, "sigma": sigma or {"prop": 0.15, "add": None},
            "covariate_effects": [{"param": p, "covariate": c, "kind": "power", "center": 4.3,
                                   "levels": None, "coefficient": b, "rse_pct": 10.0}
                                  for p, c, b in effects],
            "ofv": -100.0, "n_blq": 36, "converged": converged, "iterations": 40,
            "condition_number": 120.0, "cov_note": "ok"}


def _scm(selected=(), *, final_effects=None, model_key="iv_2cmt", error_model="proportional",
         note=None, converged=True) -> dict:
    """Mirrors app.compute.nlme.scm plus run_scm's selection_caveat/screened_out."""
    eff = final_effects if final_effects is not None else [(p, c, 0.8) for p, c in selected]
    final = _fit(model_key, theta={"CL": 0.21, "VC": 3.0, "Q": 0.5, "VP": 2.4}, effects=eff,
                 error_model=error_model, converged=converged)
    return {"status": "ok", "model_key": model_key, "label": "2-cmt IV (linear)", "base_ofv": -100.0,
            "final_ofv": -120.0, "forward_p": 0.05, "backward_p": 0.01, "n_candidates": 4,
            "selected": [{"param": p, "covariate": c, "kind": "power"} for p, c in selected],
            "steps": [], "screened_out": [], "final": final, **({"note": note} if note else {})}


def _state(**fields) -> PharmState:
    return PharmState(**{"dataset_metadata": _META, "run_provenance": _PROV,
                         "run_provenance_status": "verified", **fields})


def _nl(**kw) -> dict:
    return {"status": "ok", **_fit(**kw)}


def _keys(d: dict) -> dict:
    return {"top": set(d), "provenance": set(d["provenance"]), "answers": set(d["answers"])}


def _steps(sub: dict) -> str:
    return " | ".join(sub["provenance"]["analysis_steps"])


def test_submission_key_set_matches_the_template():
    sub = pmx.build_submission(_state(nlme_results=_nl(), scm_results=_scm()))
    assert _keys(sub) == _keys(_template())
    assert set(pmx.ANSWER_KEYS) == set(_template()["answers"])


def test_empty_state_answers_nothing():
    sub = pmx.build_submission(PharmState())
    assert sub["answers"] == dict.fromkeys(pmx.ANSWER_KEYS)
    assert sub["provenance"]["tool"] == "PharmAgent"
    assert sub["provenance"]["software"] is None and sub["provenance"]["model"] is None
    assert "outlier_records: not addressed" in _steps(sub)


def test_nlme_without_scm_reports_structure_and_leaves_covariates_null():
    sub = pmx.build_submission(_state(nlme_results=_nl()))
    a = sub["answers"]
    assert a["structural_ncmt"] == 2
    assert a["disposition_params"] == {"CL": 0.2, "V": 3.1, "Q": 0.5, "V2": 2.4}
    assert a["error_model"] == {"prop": 0.15}
    assert a["cov_effects"] is None and a["outlier_records"] is None
    assert "40 iterations, condition number 120.0" in _steps(sub)


@pytest.mark.parametrize("key, theta, expected", [
    ("iv_1cmt", {"CL": 0.2, "V": 5.0}, (1, {"CL": 0.2, "V": 5.0})),
    ("iv_3cmt", {"CL": 0.2, "VC": 3.0, "Q2": 0.5, "VP2": 2.0, "Q3": 0.1, "VP3": 1.0},
     (3, {"CL": 0.2, "V": 3.0, "Q": 0.5, "V2": 2.0, "Q3": 0.1, "V3": 1.0})),
])
def test_one_and_three_compartment_names(key, theta, expected):
    a = pmx.build_submission(_state(nlme_results=_nl(model_key=key, theta=theta)))["answers"]
    assert (a["structural_ncmt"], a["disposition_params"]) == expected


def test_scm_with_no_selection_gives_an_empty_map_from_the_nlme_fit():
    sub = pmx.build_submission(_state(nlme_results=_nl(), scm_results=_scm()))
    assert sub["answers"]["cov_effects"] == {}
    assert sub["answers"]["disposition_params"]["CL"] == 0.2      # base NLME fit is final
    assert "WT was never a candidate" in _steps(sub)


def test_scm_selected_covariates_use_the_scm_final_model_and_standard_names():
    sub = pmx.build_submission(_state(nlme_results=_nl(), scm_results=_scm([("CL", "ALB"), ("VC", "WT")])))
    a = sub["answers"]
    assert a["cov_effects"] == {"CL": {"ALB": 0.8}, "V": {"WT": 0.8}}
    assert a["disposition_params"]["CL"] == 0.21                    # SCM final, not base NLME
    assert "run_scm applies M3" in _steps(sub)


def test_an_scm_final_fit_without_censored_records_is_flagged_as_not_m3():
    scm = _scm([("CL", "ALB")])
    scm = {**scm, "final": {**scm["final"], "n_blq": 0}}     # e.g. an SCM run from before M3
    steps = _steps(pmx.build_submission(_state(nlme_results=_nl(), scm_results=scm)))
    assert "not fitted with M3" in steps and "run_scm applies M3" not in steps


@pytest.mark.parametrize("nl", [_nl(model_key="iv_3cmt", theta={"CL": 1, "VC": 1, "Q2": 1, "VP2": 1,
                                                                  "Q3": 1, "VP3": 1}),
                                _nl(error_model="combined", sigma={"prop": 0.1, "add": 0.05})])
@pytest.mark.parametrize("selected", [(), (("CL", "ALB"),)])
def test_scm_and_nlme_on_different_models_answer_nothing(nl, selected):
    sub = pmx.build_submission(_state(nlme_results=nl, scm_results=_scm(selected)))
    assert sub["answers"] == dict.fromkeys(pmx.ANSWER_KEYS)
    assert "describe different models" in _steps(sub)


def test_scm_selection_without_any_nlme_fit_uses_the_scm_final():
    a = pmx.build_submission(_state(scm_results=_scm([("CL", "ALB")])))["answers"]
    assert a["structural_ncmt"] == 2 and a["cov_effects"] == {"CL": {"ALB": 0.8}}


@pytest.mark.parametrize("scm", [_scm(note="no usable covariate candidates"),
                                 _scm(note="base model did not converge"),
                                 _scm(converged=False)])
def test_scm_that_did_not_search_leaves_cov_effects_null(scm):
    sub = pmx.build_submission(_state(nlme_results=_nl(), scm_results=scm))
    assert sub["answers"]["cov_effects"] is None
    assert sub["answers"]["structural_ncmt"] == 2
    assert "SCM did not complete a covariate search" in _steps(sub)


def test_selected_effect_missing_from_the_final_fit_leaves_cov_effects_null():
    broken = _scm([("CL", "ALB")], final_effects=[("CL", "CRCL", 0.3)])
    assert pmx.build_submission(_state(nlme_results=_nl(), scm_results=broken))["answers"]["cov_effects"] is None


def test_combined_error_reports_both_terms():
    a = pmx.build_submission(_state(nlme_results=_nl(error_model="combined",
                                                     sigma={"prop": 0.1, "add": 0.05})))["answers"]
    assert a["error_model"] == {"prop": 0.1, "add": 0.05}


@pytest.mark.parametrize("fields", [
    {"nlme_results": _nl(converged=False)},
    {"nlme_results": {**_nl(), "degenerate_params": ["Q"]}},
    {"nlme_results": _nl(model_key="oral_1cmt", theta={"CL": 1, "V": 2, "KA": 1})},
    {"nlme_results": {"status": "no_model"}},
    {"scm_results": _scm([("CL", "ALB")], converged=False)},
])
def test_unusable_final_model_leaves_every_answer_null(fields):
    sub = pmx.build_submission(_state(**fields))
    assert sub["answers"] == dict.fromkeys(pmx.ANSWER_KEYS)


@pytest.mark.parametrize("meta", [
    {**_META, "columns": [{"name": c} for c in ("ID", "DAY", "TIME", "DV", "AMT", "DOSE", "WT", "ALB",
                                                "CRCL", "BLQ", "ROWID")],
     "detected_roles": detect_roles(["ID", "DAY", "TIME", "DV", "AMT", "DOSE", "WT", "ALB", "CRCL",
                                     "BLQ", "ROWID"])},
    {**_META, "detected_roles": {**pmx.EXPECTED_ROLES, "RATE": None}},
    None,
])
def test_a_session_on_an_unprepared_dataset_answers_nothing(meta):
    sub = pmx.build_submission(_state(dataset_metadata=meta, nlme_results=_nl()))
    assert sub["answers"] == dict.fromkeys(pmx.ANSWER_KEYS)
    assert "not the prepare_dataset output" in _steps(sub)


def test_structural_comparison_is_described():
    pk = {"status": "ok", "mode": "compare", "ranking": [{"model_key": "iv_2cmt"}, {"model_key": "iv_1cmt"}]}
    assert "iv_2cmt < iv_1cmt" in _steps(pmx.build_submission(_state(pk_model_results=pk, nlme_results=_nl())))


def test_provenance_carries_verified_software_and_run_info():
    from app.core.provenance import collect_provenance
    # the llm_* fields (F-25) describe the acting model, not the software stack this fixture mirrors
    assert {k for k, _ in _PROV} == set(collect_provenance()) - {"llm_provider", "llm_model"}
    sub = pmx.build_submission(_state(nlme_results=_nl()),
                               run_info={"harness": "none", "model": "mock", "run_utc": "2026-10-24T00:00:00Z"})
    prov = sub["provenance"]
    assert prov["software"] == {"PharmAgent": "0.1.0 (git abc1234)", "python": "3.13.1",
                                "numpy": "2.4.6", "scipy": "1.17.1", "pandas": "3.0.3"}
    assert (prov["harness"], prov["model"], prov["harness_version"]) == ("none", "mock", None)
    with pytest.raises(ValueError, match="unknown run_info"):
        pmx.build_submission(PharmState(), run_info={"llm": "x"})


def test_unverified_provenance_is_not_reported_and_a_quarantined_session_is_refused():
    from app.core.provenance import ProvenanceQuarantined
    for status in ("legacy_unverified", None):
        assert pmx.build_submission(_state(run_provenance_status=status))["provenance"]["software"] is None
    with pytest.raises(ProvenanceQuarantined):
        pmx.build_submission(_state(run_provenance_status="mismatch", nlme_results=_nl()))


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_prepare_writes_a_loadable_csv_and_a_report(tmp_path, capsys):
    out = tmp_path / "s00.csv"
    assert pmx.main(["prepare", "--data", str(FIX / "data.csv"), "--out", str(out)]) == 0
    back = pd.read_csv(out, na_values=NONMEM_NA_VALUES)
    assert tuple(back.columns) == pmx.ANALYSIS_COLUMNS and len(back) == 1680
    assert int(back["CENS"].sum()) == 156 and back["DV"].isna().sum() == 120
    report = json.loads((tmp_path / "s00.report.json").read_text(encoding="utf-8"))
    assert report["source_sha256"] == MANIFEST["files"]["data.csv"]["sha256"]
    assert pmx.main(["prepare", "--data", str(FIX / "data.csv"), "--out", str(out)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_cli_submit_writes_yaml_matching_the_template(tmp_path):
    state_file, out = tmp_path / "state.json", tmp_path / "submission.yaml"
    state = _state(nlme_results={"status": "ok", **_fit()}, scm_results=_scm([("CL", "ALB")]))
    state_file.write_text(state.model_dump_json(), encoding="utf-8")
    assert pmx.main(["submit", "--state", str(state_file), "--out", str(out),
                     "--harness", "none", "--run-utc", "2026-10-24T00:00:00Z"]) == 0
    sub = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert _keys(sub) == _keys(_template())
    assert sub["answers"]["outlier_records"] is None
    assert sub["answers"]["cov_effects"] == {"CL": {"ALB": 0.8}}
    assert sub["provenance"]["run_utc"] == "2026-10-24T00:00:00Z"


def test_cli_submit_rejects_an_invalid_state(tmp_path, capsys):
    bad = tmp_path / "state.json"
    bad.write_text("{not json", encoding="utf-8")
    assert pmx.main(["submit", "--state", str(bad), "--out", str(tmp_path / "s.yaml")]) == 2
    assert "pmxbench submit" in capsys.readouterr().err
