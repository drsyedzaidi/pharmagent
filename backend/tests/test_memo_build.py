"""Briefing memo content: sections chosen from state, every number traced.

The memo is a template filled only from PharmState. These tests pin: which sections
appear, that missing ones are omitted with a one-line 'not run' note, that every
numeric token is in the value table, and that tags name the producing audit entry.
"""
import re

import pytest

from app.compute.memo import TAG_RE, AuditTrace, memo_bad_trace_tags, memo_untraced_numbers, strip_tags
from app.compute.memo_build import SECTION_TITLES, build_memo
from app.compute.memo_model import Table
from app.core.pharmstate import PharmState
from tests.memo_fixtures import IDX, full_state, full_trail


def _memo(state=None, trail=None, **kw):
    return build_memo(state if state is not None else full_state(),
                      trail if trail is not None else full_trail(), **kw)


def _by_key(memo):
    return {v.key: v for v in memo.values}


def test_empty_state_has_no_sections_and_lists_every_section_as_not_run():
    memo = build_memo(PharmState(), AuditTrace())
    assert memo.sections == ()
    assert [t for _, t in memo.omitted] == list(SECTION_TITLES.values())


def test_sections_are_chosen_from_what_exists_and_the_rest_noted_not_run():
    state = PharmState(qc_verdict="PASS", nca_summary=full_state().nca_summary)
    memo = _memo(state, AuditTrace())
    assert [s.key for s in memo.sections] == ["nca", "qc"]
    text = memo.text()
    for key, title in SECTION_TITLES.items():
        if key not in ("nca", "qc"):
            assert f"{title}: not run." in text          # exactly one line each
            assert text.count(f"{title}: not run.") == 1
    assert "Quality control: not run." not in text


def test_full_state_includes_all_sections_in_a_fixed_order():
    memo = _memo()
    assert [s.key for s in memo.sections] == list(SECTION_TITLES)
    assert memo.omitted == ()
    assert "not run" not in memo.text()


def test_every_number_in_the_memo_is_traced_and_every_tag_is_valid():
    memo = _memo()
    text = memo.text()
    assert memo_untraced_numbers(text, memo.values) == []
    assert memo_bad_trace_tags(text, full_trail().n_entries) == []
    assert len(memo.values) > 100                                 # a real memo, not a stub


def test_numeric_table_cells_all_carry_a_trace_tag():
    memo = _memo()
    checked = 0
    for block in memo.blocks():
        if not isinstance(block, Table):
            continue
        for row in block.rows:
            for cell in row:
                if re.search(r"[0-9]", strip_tags(cell)):
                    assert TAG_RE.search(cell), cell
                    checked += 1
    assert checked > 40


def test_an_invented_or_edited_number_is_caught():
    memo = _memo()
    assert memo_untraced_numbers(memo.text() + "\nAUCinf was 99.9 in the model.", memo.values) == ["99.9"]
    edited = memo.text().replace("2.71 [#", "2.72 [#", 1)
    assert "2.72" in memo_untraced_numbers(edited, memo.values)


def test_geometric_mean_cl_f_and_auc_are_headlined_from_state():
    memo = _memo()
    text = memo.text()
    gm = _by_key(memo)
    assert gm["nca_summary.CL_F.geomean"].text == "2.71"          # 2.7123 at two decimals
    assert gm["nca_summary.AUC_inf.geomean"].text == "118.8"      # 118.7651 at one decimal
    head = "\n".join(memo.headlines)
    assert "CL/F" in head and "2.71" in head and "118.8" in head
    assert "CL/F" in text and "AUCinf" in text


def test_trace_tags_point_at_the_audit_entry_that_produced_each_value():
    by = _by_key(_memo())
    expect = {
        "dataset_metadata.n_subjects": IDX["load"],
        "data_quality.blq_pct": IDX["profile"],
        "nca_summary.CL_F.geomean": IDX["nca"],
        "qc.n_checks": IDX["qc"],
        "pk_model_results.best.mean_aic": IDX["pk"],
        "nlme_results.ofv": IDX["nlme"],
        "nlme_results.theta.CL": IDX["nlme"],
        "scm_results.final_ofv": IDX["scm"],
        "forest_results.0.gmr": IDX["forest"],
        "vpc_results.gof.r2_log_ipred": IDX["vpc"],
        "diagnostics_results.cwres.mean": IDX["diag"],
    }
    for key, idx in expect.items():
        assert by[key].audit_index == idx, key


def test_each_indirect_comparison_is_matched_to_its_own_audit_entry_by_output_hash():
    by = _by_key(_memo())
    assert by["indirect_results.0.estimate"].audit_index == IDX["ind_hr"]
    assert by["indirect_results.1.estimate"].audit_index == IDX["ind_md"]
    # state edited after the fact: no audit entry has that output any more -> unknown (#?)
    state = full_state()
    comps = [dict(c) for c in state.indirect_results["comparisons"]]
    comps[0] = {**comps[0], "estimate": 0.9}
    edited = state.model_copy(update={"indirect_results": {"comparisons": comps}})
    assert _by_key(_memo(edited))["indirect_results.0.estimate"].audit_index is None


def test_without_an_audit_trail_tags_are_unknown_but_numbers_still_trace_to_state():
    memo = _memo(trail=AuditTrace())
    assert "[#?]" in memo.text() and not [t for t in TAG_RE.findall(memo.text()) if t != "[#?]"]
    assert all(v.audit_index is None for v in memo.values)
    assert memo_untraced_numbers(memo.text(), memo.values) == []


def test_qc_lists_only_the_failing_checks_verbatim():
    text = _memo().text()
    assert "AUC %extrap <= 20%" in text and "Lambda_z adj R^2 >= 0.80" in text
    assert "Sample size adequacy" not in text                     # PASS rows are summarised, not listed
    assert "CONDITIONAL PASS" in text and "high %extrap: [1]" in text


def test_structural_comparison_marks_the_best_model_and_converged_counts():
    text = _memo().text()
    assert "1-cmt oral transit abs." in text and "1-cmt oral (linear)" in text
    assert "12/12" in text and "11/12" in text
    by = _by_key(_memo())
    assert by["pk_model_results.best.mean_aic"].text == "-42.62"


def test_indirect_section_reports_estimate_ci_se_z_p_and_inputs():
    memo = _memo()
    text, by = memo.text(), _by_key(_memo())
    assert "Drug A vs Drug C via Placebo" in text
    assert by["indirect_results.0.estimate"].text == "0.625"
    assert by["indirect_results.0.se"].text == "0.361"
    assert by["indirect_results.0.z"].text == "-1.30"
    assert by["indirect_results.0.p_value"].text == "0.192"
    assert by["indirect_results.1.estimate"].text == "3.000"
    assert "transitivity" in text


def test_missing_values_render_as_dash_not_a_crash():
    state = full_state()
    nlme = {**state.nlme_results, "theta": {"CL": None, "V": float("nan")}, "ofv": None}
    memo = _memo(state.model_copy(update={"nlme_results": nlme}))
    assert memo_untraced_numbers(memo.text(), memo.values) == []
    keys = {v.key for v in memo.values}
    assert not keys & {"nlme_results.theta.CL", "nlme_results.theta.V", "nlme_results.ofv"}
    lines = memo.text().splitlines()
    assert "CL | - | 8.2 [#6] | 28.5 [#6] | 12.4 [#6]" in lines        # a dash, never 0.000
    assert "V | - | 11.4 [#6] | 19.3 [#6] | 18.7 [#6]" in lines


def test_build_is_deterministic_and_does_not_touch_state():
    state = full_state()
    before = state.model_dump()
    a, b = _memo(state), _memo(state)
    assert a.text() == b.text() and a.values == b.values
    assert state.model_dump() == before


def test_title_must_not_contain_digits_and_own_audit_index_is_tagged():
    with pytest.raises(ValueError, match="digits"):
        _memo(title="Study 101 briefing")
    with pytest.raises(ValueError, match="digits"):
        _memo(title=5)                                   # a non-string title is coerced, not a crash
    assert _memo(title=None).title == "Briefing memo"
    memo = _memo(own_index=18)                      # the entry after the fixture chain
    assert "Built by audit entry [#18]" in memo.text()
    assert memo_untraced_numbers(memo.text(), memo.values) == []
    assert memo.title == "Briefing memo"


def test_no_digits_in_template_wording_across_all_sections_and_not_run_notes():
    for state in (full_state(), PharmState(qc_verdict="PASS")):
        memo = _memo(state, AuditTrace())
        stripped = strip_tags(memo.text())
        numbers = [m for m in re.findall(r"\d+", stripped)]
        # every digit run must be inside a recorded value / literal
        assert memo_untraced_numbers(memo.text(), memo.values) == [], numbers[:5]


@pytest.mark.parametrize("title", ["Risk \u00bd", "\u2461 items", "\u2079\u2079 cases", "\uff19\uff19 cases"])
def test_title_rejects_every_numeral_not_just_ascii_digits(title):
    with pytest.raises(ValueError, match="digits"):
        _memo(title=title)


def test_study_info_is_shown_only_when_it_carries_no_numerals():
    state = full_state()                                   # compound/sponsor clean, study id has digits
    text = _memo(state).text()
    assert "Compound: Bupropion" in text and "Sponsor: Acme Pharma" in text
    assert "STUDY-101" not in text and "Study metadata containing digits is not shown." in text
    clean = state.model_copy(update={"study_info": state.study_info.model_copy(update={"study_id": "ALPHA"})})
    text = _memo(clean).text()
    assert "Study: ALPHA" in text and "not shown" not in text


def test_a_label_with_digits_in_state_refuses_the_memo():
    state = full_state()
    comp = {**state.indirect_results["comparisons"][0],
            "contrast": "Drug A cuts mortality 87 pct vs Drug C via Placebo",
            "treatments": {"A": "Drug A cuts mortality 87 pct", "B": "Placebo", "C": "Drug C"}}
    bad = state.model_copy(update={"indirect_results": {"comparisons": [comp]}})
    with pytest.raises(ValueError, match="digits"):
        _memo(bad)


def test_a_section_that_cannot_render_fails_closed_with_the_section_name():
    for field, value in (("dataset_metadata", {"n_subjects": 3, "dose_levels": 5}),
                         ("forest_results", {"status": "ok", "rows": ["a"]}),
                         ("nca_summary", {"descriptive": ["a"]})):
        state = PharmState(**{"qc_verdict": "PASS", field: value})
        with pytest.raises(ValueError, match="cannot render the"):
            _memo(state, AuditTrace())


def test_huge_integers_and_non_integral_fractions_render_as_a_dash_not_a_crash():
    from app.compute.memo_model import Recorder
    c = Recorder().at(1)
    assert c.count("k", 10 ** 400) == "-" and c.num("k", 10 ** 400) == "-"
    assert c.frac("k", 1.5, 3) == "-" and c.frac("k", 2, 3).startswith("2/3")
